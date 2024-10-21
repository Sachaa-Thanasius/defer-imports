"""Tests for defer_imports.

Notes
-----
A proxy's presence in a namespace is checked via stringifying the namespace and then substring matching with the
expected proxy repr, as that's the only way to inspect it without causing it to resolve.
"""

import ast
import contextlib
import importlib.util
import sys
import threading
import time
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, cast

import pytest

from defer_imports import (
    _BYTECODE_HEADER,
    _DEFER_PATH_HOOK,
    _DeferredFileLoader,
    _DeferredInstrumenter,
    install_import_hook,
)


# ============================================================================
# region -------- Helpers --------
# ============================================================================


def create_sample_module(
    path: Path,
    source: str,
    loader_type: type = _DeferredFileLoader,
    defer_module_level: bool = False,
):
    """Create a sample module based on the given attributes."""

    module_name = "sample"

    module_path = path / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")

    loader = loader_type(module_name, str(module_path))
    loader.defer_module_level = defer_module_level

    spec = importlib.util.spec_from_file_location(module_name, module_path, loader=loader)
    assert spec

    module = importlib.util.module_from_spec(spec)

    return spec, module, module_path


@contextlib.contextmanager
def temp_cache_module(name: str, module: types.ModuleType):
    """Add a module to sys.modules and then attempt to remove it on exit."""

    sys.modules[name] = module
    try:
        yield
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def better_key_repr(monkeypatch: pytest.MonkeyPatch):
    """Replace defer_imports._comptime._DeferredImportKey.__repr__ with a more verbose version for all tests."""

    def verbose_repr(self: object) -> str:
        return f"<key for {super(type(self), self).__repr__()} import>"

    monkeypatch.setattr("defer_imports._DeferredImportKey.__repr__", verbose_repr)


# endregion


# ============================================================================
# region -------- Unit tests --------
# ============================================================================


def test_path_hook_installation():
    """Test the API for putting/removing the defer_imports path hook from sys.path_hooks."""

    # It shouldn't be on there by default.
    assert _DEFER_PATH_HOOK not in sys.path_hooks
    before_length = len(sys.path_hooks)

    # It should be present after calling install.
    hook_ctx = install_import_hook()
    assert _DEFER_PATH_HOOK in sys.path_hooks
    assert len(sys.path_hooks) == before_length + 1

    # Calling uninstall should remove it.
    hook_ctx.uninstall()
    assert _DEFER_PATH_HOOK not in sys.path_hooks
    assert len(sys.path_hooks) == before_length

    # Calling uninstall if it's not present should do nothing to sys.path_hooks.
    hook_ctx.uninstall()
    assert _DEFER_PATH_HOOK not in sys.path_hooks
    assert len(sys.path_hooks) == before_length


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            """'''Module docstring here'''""",
            '''\
"""Module docstring here"""
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
del @_DeferredImportKey, @_DeferredImportProxy
''',
            id="inserts statements after module docstring",
        ),
        pytest.param(
            """from __future__ import annotations""",
            """\
from __future__ import annotations
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="Inserts statements after __future__ import",
        ),
        pytest.param(
            """\
from contextlib import nullcontext

import defer_imports

with defer_imports.until_use, nullcontext():
    import inspect
""",
            """\
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
from contextlib import nullcontext
import defer_imports
with defer_imports.until_use, nullcontext():
    import inspect
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="does nothing if used with another context manager",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use:
    import inspect
""",
            """\
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
import defer_imports
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import inspect
    if type(inspect) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('inspect')
        @local_ns[@_DeferredImportKey('inspect', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="regular import",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use:
    import importlib
    import importlib.abc
""",
            """\
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
import defer_imports
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import importlib
    if type(importlib) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('importlib')
        @local_ns[@_DeferredImportKey('importlib', @temp_proxy)] = @temp_proxy
    import importlib.abc
    if type(importlib) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('importlib')
        @local_ns[@_DeferredImportKey('importlib', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="mixed imports",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use:
    from . import a
""",
            """\
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
import defer_imports
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    from . import a
    if type(a) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('a')
        @local_ns[@_DeferredImportKey('a', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="relative import",
        ),
    ],
)
def test_instrumentation(before: str, after: str):
    """Test what code is generated by the instrumentation side of defer_imports."""

    filename = "<unknown>"
    orig_tree = ast.parse(before, filename, "exec")
    transformer = _DeferredInstrumenter(before, filename)
    new_tree = ast.fix_missing_locations(transformer.visit(orig_tree))

    assert f"{ast.unparse(new_tree)}\n" == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            """\
import inspect
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import inspect
    if type(inspect) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('inspect')
        @local_ns[@_DeferredImportKey('inspect', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="regular import",
        ),
        pytest.param(
            """\
import hello
import world
import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    import world
    if type(world) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('world')
        @local_ns[@_DeferredImportKey('world', @temp_proxy)] = @temp_proxy
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="multiple imports consecutively",
        ),
        pytest.param(
            """\
import hello
import world

print("hello")

import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    import world
    if type(world) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('world')
        @local_ns[@_DeferredImportKey('world', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
print('hello')
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="multiple imports separated by statement 1",
        ),
        pytest.param(
            """\
import hello
import world

def do_the_thing(a: int) -> int:
    return a

import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    import world
    if type(world) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('world')
        @local_ns[@_DeferredImportKey('world', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns

def do_the_thing(a: int) -> int:
    return a
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="multiple imports separated by statement 2",
        ),
        pytest.param(
            """\
import hello

def do_the_thing(a: int) -> int:
    import world
    return a
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns

def do_the_thing(a: int) -> int:
    import world
    return a
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="nothing done for imports within function",
        ),
        pytest.param(
            """\
import hello
from world import *
import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
from world import *
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="avoids doing anything with wildcard imports",
        ),
        pytest.param(
            """\
import foo
try:
    import hello
finally:
    pass
import bar
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
try:
    import hello
finally:
    pass
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import bar
    if type(bar) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('bar')
        @local_ns[@_DeferredImportKey('bar', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="avoids imports in try-finally",
        ),
        pytest.param(
            """\
import foo
with nullcontext():
    import hello
import bar
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
with nullcontext():
    import hello
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import bar
    if type(bar) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('bar')
        @local_ns[@_DeferredImportKey('bar', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="avoids imports in non-defer_imports.until_use with block",
        ),
        pytest.param(
            """\
import defer_imports
import foo
with defer_imports.until_use:
    import hello
import bar
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
import defer_imports
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import foo
    if type(foo) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('foo')
        @local_ns[@_DeferredImportKey('foo', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import hello
    if type(hello) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('hello')
        @local_ns[@_DeferredImportKey('hello', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
with defer_imports.until_use:
    @local_ns = locals()
    @temp_proxy = None
    import bar
    if type(bar) is @_DeferredImportProxy:
        @temp_proxy = @local_ns.pop('bar')
        @local_ns[@_DeferredImportKey('bar', @temp_proxy)] = @temp_proxy
    del @temp_proxy, @local_ns
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="still instruments imports in defer_imports.until_use with block",
        ),
        pytest.param(
            """\
try:
    import foo
except:
    pass
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
try:
    import foo
except:
    pass
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="escape hatch: try",
        ),
        pytest.param(
            """\
try:
    raise Exception
except:
    import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
try:
    raise Exception
except:
    import foo
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="escape hatch: except",
        ),
        pytest.param(
            """\
try:
    print('hi')
except:
    print('error')
else:
    import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
try:
    print('hi')
except:
    print('error')
else:
    import foo
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="escape hatch: else",
        ),
        pytest.param(
            """\
try:
    pass
finally:
    import foo
""",
            """\
import defer_imports
from defer_imports import _DeferredImportKey as @_DeferredImportKey, _DeferredImportProxy as @_DeferredImportProxy
try:
    pass
finally:
    import foo
del @_DeferredImportKey, @_DeferredImportProxy
""",
            id="escape hatch: finally",
        ),
    ],
)
def test_module_instrumentation(before: str, after: str):
    """Test what code is generated by the instrumentation side of defer_imports if applied at a module level."""

    filename = "<unknown>"
    orig_tree = ast.parse(before, filename, "exec")
    transformer = _DeferredInstrumenter(before, filename, module_level=True)
    new_tree = ast.fix_missing_locations(transformer.visit(orig_tree))

    assert f"{ast.unparse(new_tree)}\n" == after


# endregion


# ============================================================================
# region -------- Integration tests --------
# ============================================================================


def test_empty(tmp_path: Path):
    source = ""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)


def test_without_until_use_local(tmp_path: Path):
    source = "import contextlib"

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    assert module.contextlib is sys.modules["contextlib"]


def test_until_use_noop(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""

    spec, module, _ = create_sample_module(tmp_path, source, SourceFileLoader)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_partial_inspect_repr = "'inspect': <module 'inspect' from"
    assert expected_partial_inspect_repr in repr(vars(module))
    assert module.inspect is sys.modules["inspect"]

    def sample_func(a: int, c: float) -> float: ...

    assert str(module.inspect.signature(sample_func)) == "(a: int, c: float) -> float"


def test_regular_import(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_inspect_repr = "<key for 'inspect' import>: <proxy for 'import inspect'>"
    assert expected_inspect_repr in repr(vars(module))
    assert module.inspect
    assert expected_inspect_repr not in repr(vars(module))

    assert module.inspect is sys.modules["inspect"]

    def sample_func(a: int, c: float) -> float: ...

    assert str(module.inspect.signature(sample_func)) == "(a: int, c: float) -> float"


def test_regular_import_with_rename(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect as gin
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_gin_repr = "<key for 'gin' import>: <proxy for 'import inspect'>"

    assert expected_gin_repr in repr(vars(module))

    with pytest.raises(NameError):
        exec("inspect", vars(module))

    with pytest.raises(AttributeError):
        assert module.inspect

    assert expected_gin_repr in repr(vars(module))
    assert module.gin
    assert expected_gin_repr not in repr(vars(module))

    assert sys.modules["inspect"] is module.gin

    def sample_func(a: int, b: str) -> str: ...

    assert str(module.gin.signature(sample_func)) == "(a: int, b: str) -> str"


def test_regular_import_nested(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import importlib.abc
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_importlib_repr = "<key for 'importlib' import>: <proxy for 'import importlib.abc'>"
    assert expected_importlib_repr in repr(vars(module))

    assert module.importlib
    assert module.importlib.abc
    assert module.importlib.abc.MetaPathFinder

    assert expected_importlib_repr not in repr(vars(module))


def test_regular_import_nested_with_rename(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import collections.abc as xyz
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    # Make sure the right proxy is in the namespace.
    expected_xyz_repr = "<key for 'xyz' import>: <proxy for 'import collections.abc as ...'>"
    assert expected_xyz_repr in repr(vars(module))

    # Make sure the intermediate imports or proxies for them aren't in the namespace.
    with pytest.raises(NameError):
        exec("collections", vars(module))

    with pytest.raises(AttributeError):
        assert module.collections

    with pytest.raises(NameError):
        exec("collections.abc", vars(module))

    with pytest.raises(AttributeError):
        assert module.collections.abc

    # Make sure xyz resolves properly.
    assert expected_xyz_repr in repr(vars(module))
    assert module.xyz
    assert expected_xyz_repr not in repr(vars(module))
    assert module.xyz is sys.modules["collections"].abc

    # Make sure only the resolved xyz remains in the namespace.
    with pytest.raises(NameError):
        exec("collections", vars(module))

    with pytest.raises(AttributeError):
        assert module.collections

    with pytest.raises(NameError):
        exec("collections.abc", vars(module))

    with pytest.raises(AttributeError):
        assert module.collections.abc


def test_from_import(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    from inspect import isfunction, signature
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_isfunction_repr = "<key for 'isfunction' import>: <proxy for 'from inspect import isfunction'>"
    expected_signature_repr = "<key for 'signature' import>: <proxy for 'from inspect import signature'>"
    assert expected_isfunction_repr in repr(vars(module))
    assert expected_signature_repr in repr(vars(module))

    with pytest.raises(NameError):
        exec("inspect", vars(module))

    assert expected_isfunction_repr in repr(vars(module))
    assert module.isfunction
    assert expected_isfunction_repr not in repr(vars(module))
    assert module.isfunction is sys.modules["inspect"].isfunction

    assert expected_signature_repr in repr(vars(module))
    assert module.signature
    assert expected_signature_repr not in repr(vars(module))
    assert module.signature is sys.modules["inspect"].signature


def test_from_import_with_rename(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    from inspect import Signature as MySignature
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_my_signature_repr = "<key for 'MySignature' import>: <proxy for 'from inspect import Signature'>"
    assert expected_my_signature_repr in repr(vars(module))

    with pytest.raises(NameError):
        exec("inspect", vars(module))

    with pytest.raises(NameError):
        exec("Signature", vars(module))

    assert expected_my_signature_repr in repr(vars(module))
    assert str(module.MySignature) == "<class 'inspect.Signature'>"  # Resolves on use.
    assert expected_my_signature_repr not in repr(vars(module))
    assert module.MySignature is sys.modules["inspect"].Signature


def test_deferred_header_in_instrumented_pycache(tmp_path: Path):
    """Test that the defer_imports-specific bytecode header is being prepended to the bytecode cache files of
    defer_imports-instrumented modules.
    """

    source = """\
import defer_imports

with defer_imports.until_use:
    import asyncio
"""

    spec, module, path = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_cache = Path(importlib.util.cache_from_source(str(path)))
    assert expected_cache.is_file()

    with expected_cache.open("rb") as fp:
        header = fp.read(len(_BYTECODE_HEADER))
    assert header == _BYTECODE_HEADER


def test_error_if_non_import(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    print("Hello world")
"""

    spec, module, module_path = create_sample_module(tmp_path, source)
    assert spec.loader

    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(module_path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == 'print("Hello world")'


def test_error_if_import_in_class(tmp_path: Path):
    source = """\
import defer_imports

class Example:
    with defer_imports.until_use:
        from inspect import signature
"""

    # Boilerplate to dynamically create and load this module.
    spec, module, module_path = create_sample_module(tmp_path, source)
    assert spec.loader

    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(module_path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "    with defer_imports.until_use:\n        from inspect import signature"


def test_error_if_import_in_function(tmp_path: Path):
    source = """\
import defer_imports

def test():
    with defer_imports.until_use:
        import inspect

    return inspect.signature(test)
"""

    spec, module, module_path = create_sample_module(tmp_path, source)
    assert spec.loader

    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(module_path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "    with defer_imports.until_use:\n        import inspect"


def test_error_if_wildcard_import(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    from typing import *
"""

    spec, module, module_path = create_sample_module(tmp_path, source)
    assert spec.loader

    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(module_path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "from typing import *"


def test_top_level_and_submodules_1(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import importlib
    import importlib.abc
    import importlib.util
"""
    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    # Prevent the caching of these from interfering with the test.
    for mod in ("importlib", "importlib.abc", "importlib.util"):
        sys.modules.pop(mod, None)

    expected_importlib_repr = "<key for 'importlib' import>: <proxy for 'import importlib'>"
    expected_importlib_abc_repr = "<key for 'abc' import>: <proxy for 'import importlib.abc as ...'>"
    expected_importlib_util_repr = "<key for 'util' import>: <proxy for 'import importlib.util as ...'>"

    # Test that the importlib proxy is here and then resolves.
    assert expected_importlib_repr in repr(vars(module))
    assert module.importlib
    assert expected_importlib_repr not in repr(vars(module))

    # Test that the nested proxies carry over to the resolved importlib.
    module_importlib_vars = cast(dict[str, object], vars(module.importlib))

    assert expected_importlib_abc_repr in repr(module_importlib_vars)
    assert expected_importlib_util_repr in repr(module_importlib_vars)

    assert expected_importlib_abc_repr in repr(module_importlib_vars)
    assert module.importlib.abc
    assert expected_importlib_abc_repr not in repr(module_importlib_vars)

    assert expected_importlib_util_repr in repr(module_importlib_vars)
    assert module.importlib.util
    assert expected_importlib_util_repr not in repr(module_importlib_vars)


def test_top_level_and_submodules_2(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import asyncio
    import asyncio.base_events
    import asyncio.base_futures
    import asyncio.base_subprocess
    import asyncio.base_tasks
    import asyncio.constants
    import asyncio.coroutines
    import asyncio.events
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)


def test_mixed_from_same_module(tmp_path: Path):
    source = """\
import defer_imports

with defer_imports.until_use:
    import asyncio
    from asyncio import base_events
    from asyncio import base_futures
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    expected_asyncio_repr = "<key for 'asyncio' import>: <proxy for 'import asyncio'>"
    expected_asyncio_base_events_repr = "<key for 'base_events' import>: <proxy for 'from asyncio import base_events'>"
    expected_asyncio_base_futures_repr = (
        "<key for 'base_futures' import>: <proxy for 'from asyncio import base_futures'>"
    )

    # Make sure the right proxies are present.
    assert expected_asyncio_repr in repr(vars(module))
    assert expected_asyncio_base_events_repr in repr(vars(module))
    assert expected_asyncio_base_futures_repr in repr(vars(module))

    # Make sure resolving one proxy doesn't resolve or void the others.
    assert module.base_futures
    assert module.base_futures is sys.modules["asyncio.base_futures"]
    assert expected_asyncio_base_futures_repr not in repr(vars(module))
    assert expected_asyncio_base_events_repr in repr(vars(module))
    assert expected_asyncio_repr in repr(vars(module))

    assert module.base_events
    assert module.base_events is sys.modules["asyncio.base_events"]
    assert expected_asyncio_base_events_repr not in repr(vars(module))
    assert expected_asyncio_base_futures_repr not in repr(vars(module))
    assert expected_asyncio_repr in repr(vars(module))

    assert module.asyncio
    assert module.asyncio is sys.modules["asyncio"]
    assert expected_asyncio_base_events_repr not in repr(vars(module))
    assert expected_asyncio_base_futures_repr not in repr(vars(module))
    assert expected_asyncio_repr not in repr(vars(module))


def test_relative_imports(tmp_path: Path):
    """Test a synthetic package that uses relative imports within defer_imports.until_use blocks.

    The package has the following structure:
        .
        └───sample_pkg
            ├───__init__.py
            ├───a.py
            └───b.py
    """

    sample_pkg_path = tmp_path / "sample_pkg"
    sample_pkg_path.mkdir()
    sample_pkg_path.joinpath("__init__.py").write_text(
        """\
import defer_imports

with defer_imports.until_use:
    from . import a
    from .a import A
    from .b import B
""",
        encoding="utf-8",
    )
    sample_pkg_path.joinpath("a.py").write_text(
        """\
class A:
    def __init__(self, val: object):
        self.val = val
""",
        encoding="utf-8",
    )
    sample_pkg_path.joinpath("b.py").write_text(
        """\
class B:
    def __init__(self, val: object):
        self.val = val
""",
        encoding="utf-8",
    )

    package_name = "sample_pkg"
    package_init_path = str(sample_pkg_path / "__init__.py")

    loader = _DeferredFileLoader(package_name, package_init_path)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        loader=loader,
        submodule_search_locations=[],  # A signal that this is a package.
    )
    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)

    with temp_cache_module(package_name, module):
        spec.loader.exec_module(module)

        module_locals_repr = repr(vars(module))
        assert "<key for 'a' import>: <proxy for 'from sample_pkg import a'>" in module_locals_repr
        assert "<key for 'A' import>: <proxy for 'from sample_pkg.a import A'>" in module_locals_repr
        assert "<key for 'B' import>: <proxy for 'from sample_pkg.b import B'>" in module_locals_repr

        assert module.A
        assert repr(module.A("hello")).startswith("<sample_pkg.a.A object at")


def test_circular_imports(tmp_path: Path):
    """Test a synthetic package that does circular imports.

    The package has the following structure:
        .
        └───circular_pkg
            ├───__init__.py
            ├───main.py
            ├───x.py
            └───y.py
    """

    circular_pkg_path = tmp_path / "circular_pkg"
    circular_pkg_path.mkdir()
    circular_pkg_path.joinpath("__init__.py").write_text(
        """\
import defer_imports

with defer_imports.until_use:
    import circular_pkg.main
""",
        encoding="utf-8",
    )
    circular_pkg_path.joinpath("main.py").write_text(
        """\
from .x import X2
X2()
""",
        encoding="utf-8",
    )
    circular_pkg_path.joinpath("x.py").write_text(
        """\
def X1():
    return "X"

from .y import Y1

def X2():
    return Y1()
"""
    )

    circular_pkg_path.joinpath("y.py").write_text(
        """\
def Y1():
    return "Y"

from .x import X2

def Y2():
    return X2()
""",
        encoding="utf-8",
    )

    package_name = "circular_pkg"
    package_init_path = str(circular_pkg_path / "__init__.py")

    loader = _DeferredFileLoader(package_name, package_init_path)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        loader=loader,
        submodule_search_locations=[],  # A signal that this is a package.
    )
    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)

    with temp_cache_module(package_name, module):
        spec.loader.exec_module(module)

        assert module


def test_import_stdlib():
    """Test that defer_imports.until_use works when wrapping imports for most of the stdlib."""

    _missing: Any = object()

    # The path finder for the tests directory is already cached, so we need to temporarily reset that entry.
    _tests_path = str(Path(__file__).parent)
    _temp_cache = sys.path_importer_cache.pop(_tests_path, _missing)

    try:
        with install_import_hook(uninstall_after=True):
            import tests.sample_stdlib_imports

        # Sample-check the __future__ import.
        expected_future_import = "<key for '__future__' import>: <proxy for 'import __future__'>"
        assert expected_future_import in repr(vars(tests.sample_stdlib_imports))
    finally:
        # Revert changes to the path finder cache.
        if _temp_cache is not _missing:
            sys.path_importer_cache[_tests_path] = _temp_cache


@pytest.mark.skip(reason="Leaking patch problem is currently out of scope.")
def test_leaking_patch(tmp_path: Path):  # pragma: no cover
    """Test a synthetic package that demonstrates the "leaking patch" problem.

    Source: https://github.com/bswck/slothy/tree/bd0828a8dd9af63ca5c85340a70a14a76a6b714f/tests/leaking_patch

    The package has the following structure:
        .
        └───leaking_patch_pkg
            ├───__init__.py
            ├───a.py
            ├───b.py
            └───patcher.py
    """

    leaking_patch_pkg_path = tmp_path / "leaking_patch_pkg"
    leaking_patch_pkg_path.mkdir()
    leaking_patch_pkg_path.joinpath("__init__.py").touch()
    leaking_patch_pkg_path.joinpath("a.py").write_text(
        """\
import defer_imports

with defer_imports.until_use:
    from .b import B
""",
        encoding="utf-8",
    )
    leaking_patch_pkg_path.joinpath("b.py").write_text('B = "original thing"', encoding="utf-8")
    leaking_patch_pkg_path.joinpath("patching.py").write_text(
        """\
from unittest import mock

patcher = mock.patch("leaking_patch_pkg.b.B", "patched thing", create=True)
mock_B = patcher.start()
""",
        encoding="utf-8",
    )

    package_name = "leaking_patch_pkg"
    package_init_path = str(leaking_patch_pkg_path / "__init__.py")

    loader = _DeferredFileLoader(package_name, package_init_path)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        loader=loader,
        submodule_search_locations=[],  # A signal that this is a package.
    )
    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)

    with temp_cache_module(package_name, module):
        spec.loader.exec_module(module)
        exec(f"import {package_name}.patching; from {package_name}.b import B", vars(module))
        assert module.B == "original thing"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type statements are only valid in 3.12+")
def test_type_statement_312(tmp_path: Path):
    """Test that a proxy within a type statement doesn't resolve until accessed via .__value__.

    The package has the following structure:
        .
        └───type_stmt_pkg
            ├───__init__.py
            └───exp.py
    """

    type_stmt_pkg_path = tmp_path / "type_stmt_pkg"
    type_stmt_pkg_path.mkdir()
    type_stmt_pkg_path.joinpath("__init__.py").write_text(
        """\
import defer_imports

with defer_imports.until_use:
    from .exp import Expensive

type ManyExpensive = tuple[Expensive, ...]
"""
    )
    type_stmt_pkg_path.joinpath("exp.py").write_text("class Expensive: ...", encoding="utf-8")

    package_name = "type_stmt_pkg"
    package_init_path = str(type_stmt_pkg_path / "__init__.py")

    loader = _DeferredFileLoader(package_name, package_init_path)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        loader=loader,
        submodule_search_locations=[],  # A signal that this is a package.
    )
    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)

    with temp_cache_module(package_name, module):
        spec.loader.exec_module(module)
        expected_proxy_repr = "<key for 'Expensive' import>: <proxy for 'from type_stmt_pkg.exp import Expensive'>"

        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive) == "ManyExpensive"
        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive.__value__) == "tuple[type_stmt_pkg.exp.Expensive, ...]"
        assert expected_proxy_repr not in repr(vars(module))


# endregion


# ============================================================================
# region -------- Regression tests --------
# ============================================================================


@pytest.mark.flaky(reruns=3)
def test_thread_safety(tmp_path: Path):
    """Test that trying to access a lazily loaded import from multiple threads doesn't cause race conditions.

    Based on a test for importlib.util.LazyLoader in the CPython test suite.

    Notes
    -----
    This test is flaky, seemingly more so in CI than locally. Some information about the failure:

    -   It's the same every time: paraphrased, the "inspect" proxy doesn't have "signature" as an attribute. The proxy
        should be resolved before this happens, and is even guarded with a RLock and a boolean to prevent this.
    -   It seemingly only happens on pypy.
    -   The reproduction rate locally is ~1/100 when run across pypy3.9 and pypy3.10, 50 times each.
        -   Add 'pytest.mark.parametrize("q", range(50))' to the test for the repetitions.
        -   Run "hatch test -py pypy3.9,pypy3.10 -- tests/test_deferred.py::test_thread_safety".
    """

    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    class _Missing:
        """Singleton sentinel."""

    class CapturingThread(threading.Thread):
        """Thread subclass that captures a returned result or raised exception from the called target."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.result = _Missing
            self.exc = _Missing

        def run(self) -> None:  # pragma: no cover
            # This has minor modifications from threading.Thread.run() to catch the returned value or raised exception.
            try:
                self.result = self._target(*self._args, **self._kwargs)  # pyright: ignore
            except Exception as exc:  # noqa: BLE001
                self.exc = exc
            finally:
                del self._target, self._args, self._kwargs  # pyright: ignore

    def access_module_attr() -> object:
        time.sleep(0.2)
        return module.inspect.signature

    threads: list[CapturingThread] = []

    for i in range(20):
        thread = CapturingThread(name=f"Thread {i}", target=access_module_attr)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        assert thread.exc is _Missing
        assert callable(thread.result)  # pyright: ignore


# endregion
