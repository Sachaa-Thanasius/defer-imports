"""Tests for defer_imports.

Notes
-----
A proxy's presence in a namespace is checked via stringifying the namespace and then substring matching with the
expected proxy repr, as that's the only way to inspect it without causing it to resolve.
"""

import ast
import collections
import collections.abc
import importlib.util
import sys
import threading
import time
import unittest.mock
from importlib.abc import Loader
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Union, cast

import pytest

from defer_imports.ast_rewrite import (
    _ACTUAL_CTX_ASNAME,
    _ACTUAL_CTX_NAME,
    _BYTECODE_HEADER,
    _PATH_HOOK,
    _TEMP_ASNAMES,
    _DIFileLoader,
    _ImportsInstrumenter,
    import_hook,
)


# ============================================================================
# region -------- Helpers --------
# ============================================================================


NestedMapping = collections.abc.Mapping[str, Union["NestedMapping", str]]


SAMPLE_DOCSTRING = "Module docstring here"

MODULE_TEMPLATE = f"""\
from defer_imports.ast_rewrite import {_ACTUAL_CTX_NAME} as {_ACTUAL_CTX_ASNAME}
{{}}
del {_ACTUAL_CTX_ASNAME}
""".rstrip()

IMPORT_TEMPLATE = f"""\
with {_ACTUAL_CTX_ASNAME}:
    {{}}
    del {_TEMP_ASNAMES}
""".rstrip()


def module_template(*lines: str) -> str:
    return MODULE_TEMPLATE.format("\n".join(lines))


def import_template(*lines: str) -> str:
    return IMPORT_TEMPLATE.format("\n    ".join(lines))


asnames_template = f"{_TEMP_ASNAMES} = {{!r}}".format


def create_sample_module(path: Path, source: str, loader_type: type[Loader] = _DIFileLoader, *, exec_mod: bool = True):
    """Create a sample module based on the given attributes."""

    module_name = "sample"
    module_path = path / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")

    loader = loader_type(module_name, str(module_path))  # pyright: ignore [reportCallIssue]
    spec = importlib.util.spec_from_file_location(module_name, module_path, loader=loader)
    assert spec is not None

    module = importlib.util.module_from_spec(spec)

    if exec_mod:
        # NOTE: Use spec.loader instead of loader because of potential create_module() side-effects.
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return module


def create_dir_tree(path: Path, dir_contents: NestedMapping) -> None:
    """Create a tree of files based on a (nested) dict of file/directory names and (source) contents.

    Warning: Be careful when using escape sequences in file contents. Consider escaping them or using raw strings.
    """

    queue = collections.deque((path / filename, v) for filename, v in dir_contents.items())
    while queue:
        filepath, value = queue.popleft()
        if isinstance(value, dict):
            filepath.mkdir()
            queue.extend((filepath / filename, v) for filename, v in value.items())
        elif isinstance(value, str):
            filepath.write_text(value, encoding="utf-8")
        else:  # pragma: no cover
            msg = f"expected a dict or a string, got {value!r}"
            raise TypeError(msg)


# endregion


def test_path_hook_installation():
    """Test the API for putting/removing the defer_imports path hook on/from sys.path_hooks."""

    with unittest.mock.patch.object(sys, "path_hooks", list(sys.path_hooks)):
        # It shouldn't be on there by default.
        assert _PATH_HOOK not in sys.path_hooks
        before_length = len(sys.path_hooks)

        # It should not be present after just getting the hook context.
        hook_ctx = import_hook()
        assert _PATH_HOOK not in sys.path_hooks
        assert len(sys.path_hooks) == before_length

        # It should be present after calling install.
        hook_ctx.install()
        assert _PATH_HOOK in sys.path_hooks
        assert len(sys.path_hooks) == before_length + 1

        # Calling uninstall should remove it.
        hook_ctx.uninstall()
        assert _PATH_HOOK not in sys.path_hooks
        assert len(sys.path_hooks) == before_length

        # Calling uninstall if it's not present should do nothing to sys.path_hooks.
        hook_ctx.uninstall()
        assert _PATH_HOOK not in sys.path_hooks
        assert len(sys.path_hooks) == before_length


class TestASTRewrite:
    common_rewrite_cases = [
        pytest.param(
            *[""] * 2,
            id="empty module",
        ),
        pytest.param(
            *[f'"""{SAMPLE_DOCSTRING}"""'] * 2,
            id="docstring",
        ),
        pytest.param(
            *["from __future__ import annotations"] * 2,
            id="from __future__ import",
        ),
        pytest.param(
            *[f'"""{SAMPLE_DOCSTRING}"""\nfrom __future__ import annotations'] * 2,
            id="docstring and from __future__ import",
        ),
    ]

    @pytest.mark.parametrize(
        ("source", "expected_rewrite"),
        [
            *common_rewrite_cases,
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    import __future__
""",
                module_template("import defer_imports", import_template(asnames_template(None), "import __future__")),
                id="top-level __future__ import",
            ),
            pytest.param(
                """\
from contextlib import nullcontext

import defer_imports

with defer_imports.until_use, nullcontext():
    import inspect
""",
                """\
from contextlib import nullcontext
import defer_imports
with defer_imports.until_use, nullcontext():
    import inspect
""".rstrip(),
                id="does nothing if used with another context manager",
            ),
            pytest.param(
                f"""\
'''{SAMPLE_DOCSTRING}'''

import defer_imports

with defer_imports.until_use:
    import inspect
""",
                f'"""{SAMPLE_DOCSTRING}"""\n'
                + module_template("import defer_imports", import_template(asnames_template(None), "import inspect")),
                id="docstring then regular import",
            ),
            pytest.param(
                """\
from __future__ import annotations

import defer_imports

with defer_imports.until_use:
    import inspect
""",
                "from __future__ import annotations\n"
                + module_template("import defer_imports", import_template(asnames_template(None), "import inspect")),
                id="from __future__ then regular import",
            ),
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    import inspect
""",
                module_template("import defer_imports", import_template(asnames_template(None), "import inspect")),
                id="regular import",
            ),
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    import inspect as i
""",
                module_template("import defer_imports", import_template(asnames_template("i"), "import inspect as i")),
                id="regular import with rename 1",
            ),
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    import sys, os as so
""",
                module_template(
                    "import defer_imports",
                    import_template(asnames_template(None), "import sys", asnames_template("so"), "import os as so"),
                ),
                id="regular import with rename 2",
            ),
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    import importlib
    import importlib.abc
""",
                module_template(
                    "import defer_imports",
                    import_template(
                        asnames_template(None), "import importlib", asnames_template(None), "import importlib.abc"
                    ),
                ),
                id="mixed imports",
            ),
            pytest.param(
                """\
import defer_imports

with defer_imports.until_use:
    from . import a
""",
                module_template("import defer_imports", import_template(asnames_template((None,)), "from . import a")),
                id="relative import",
            ),
        ],
    )
    def test_regular_rewrite(self, source: str, expected_rewrite: str):
        """Test what code is generated by the instrumentation side of defer_imports."""

        transformer = _ImportsInstrumenter(source)
        new_tree = transformer.visit(ast.parse(source))
        actual_rewrite = ast.unparse(new_tree)

        assert actual_rewrite == expected_rewrite

    @pytest.mark.parametrize(
        ("source", "expected_rewrite"),
        [
            *common_rewrite_cases,
            pytest.param(
                "import inspect",
                module_template(import_template(asnames_template(None), "import inspect")),
                id="regular import",
            ),
            pytest.param(
                "\n".join(("import hello", "import world", "import foo")),
                module_template(
                    import_template(
                        asnames_template(None),
                        "import hello",
                        asnames_template(None),
                        "import world",
                        asnames_template(None),
                        "import foo",
                    )
                ),
                id="multiple imports consecutively",
            ),
            pytest.param(
                """\
import hello
import world

print("hello")

import foo
""",
                module_template(
                    import_template(asnames_template(None), "import hello", asnames_template(None), "import world"),
                    "print('hello')",
                    import_template(asnames_template(None), "import foo"),
                ),
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
                module_template(
                    import_template(asnames_template(None), "import hello", asnames_template(None), "import world"),
                    "def do_the_thing(a: int) -> int:",
                    "    return a",
                    import_template(asnames_template(None), "import foo"),
                ),
                id="multiple imports separated by statement 2",
            ),
            pytest.param(
                """\
import hello

def do_the_thing(a: int) -> int:
    import world
    return a
""",
                module_template(
                    import_template(asnames_template(None), "import hello"),
                    "def do_the_thing(a: int) -> int:",
                    "    import world",
                    "    return a",
                ),
                id="nothing done for imports within function",
            ),
            pytest.param(
                """\
import hello
from world import *
import foo
""",
                module_template(
                    import_template(asnames_template(None), "import hello"),
                    "from world import *",
                    import_template(asnames_template(None), "import foo"),
                ),
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
                module_template(
                    import_template(asnames_template(None), "import foo"),
                    "try:",
                    "    import hello",
                    "finally:",
                    "    pass",
                    import_template(asnames_template(None), "import bar"),
                ),
                id="avoids imports in try-finally",
            ),
            pytest.param(
                """\
import foo
with nullcontext():
    import hello
import bar
""",
                module_template(
                    import_template(asnames_template(None), "import foo"),
                    "with nullcontext():",
                    "    import hello",
                    import_template(asnames_template(None), "import bar"),
                ),
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
                module_template(
                    import_template(
                        asnames_template(None), "import defer_imports", asnames_template(None), "import foo"
                    ),
                    import_template(asnames_template(None), "import hello"),
                    import_template(asnames_template(None), "import bar"),
                ),
                id="still instruments imports in defer_imports.until_use with block",
            ),
            # NOTE: "\n".join() is less concise but more readable here than strings littered with "\n".
            pytest.param(
                *["\n".join(("try:", "    import foo", "except:", "    pass"))] * 2,
                id="escape hatch: try",
            ),
            pytest.param(
                *["\n".join(("try:", "    raise Exception", "except:", "    import foo"))] * 2,
                id="escape hatch: except",
            ),
            pytest.param(
                *["\n".join(("try:", "    print('hi')", "except:", "    print('error')", "else:", "    import foo"))]
                * 2,
                id="escape hatch: else",
            ),
            pytest.param(
                *["\n".join(("try:", "    pass", "finally:", "    import foo"))] * 2,
                id="escape hatch: finally",
            ),
        ],
    )
    def test_full_rewrite(self, source: str, expected_rewrite: str):
        """Test what code is generated by the instrumentation side of defer_imports if applied at a module level."""

        transformer = _ImportsInstrumenter(source, whole_module=True)
        new_tree = transformer.visit(ast.parse(source))
        actual_rewrite = ast.unparse(new_tree)

        # We can't and shouldn't depend on ast.unparse() matching our expected whitespace.
        actual_no_empty_lines = "\n".join(filter(str.strip, actual_rewrite.splitlines()))
        expected_no_empty_lines = "\n".join(filter(str.strip, expected_rewrite.splitlines()))

        assert actual_no_empty_lines == expected_no_empty_lines


@pytest.mark.usefixtures("preserve_sys_modules")
class TestImport:
    @pytest.fixture(autouse=True)
    def better_key_repr(self):
        """Replace _DIKey.__repr__ with a more verbose version for all tests."""

        def verbose_repr(self: object) -> str:
            return f"<key for {super(type(self), self).__repr__()} import>"

        with unittest.mock.patch("defer_imports.ast_rewrite._DIKey.__repr__", verbose_repr):
            yield

    def test_empty(self, tmp_path: Path):
        source = ""
        create_sample_module(tmp_path, source)

    def test_without_until_use_local(self, tmp_path: Path):
        source = "import contextlib"
        module = create_sample_module(tmp_path, source)

        assert module.contextlib is sys.modules["contextlib"]

    def test_until_use_noop(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""
        module = create_sample_module(tmp_path, source, SourceFileLoader)

        expected_partial_inspect_repr = "'inspect': <module 'inspect' from"
        assert expected_partial_inspect_repr in repr(vars(module))
        assert module.inspect is sys.modules["inspect"]

        def sample_func(a: int, c: float) -> float: ...

        assert str(module.inspect.signature(sample_func)) == "(a: int, c: float) -> float"

    def test_regular_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""
        module = create_sample_module(tmp_path, source)

        expected_inspect_repr = "<key for 'inspect' import>: <proxy for 'inspect' import>"
        assert expected_inspect_repr in repr(vars(module))
        assert module.inspect
        assert expected_inspect_repr not in repr(vars(module))

        assert module.inspect is sys.modules["inspect"]

        def sample_func(a: int, c: float) -> float: ...

        assert str(module.inspect.signature(sample_func)) == "(a: int, c: float) -> float"

    def test_regular_import_with_rename(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import inspect as gin
"""
        module = create_sample_module(tmp_path, source)

        expected_gin_repr = "<key for 'gin' import>: <proxy for 'inspect' import>"

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

    def test_regular_import_nested(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import importlib.abc
"""
        module = create_sample_module(tmp_path, source)

        expected_importlib_repr = "<key for 'importlib' import>: <proxy for 'importlib' import>"
        assert expected_importlib_repr in repr(vars(module))

        assert module.importlib
        assert module.importlib.abc
        assert module.importlib.abc.MetaPathFinder

        assert expected_importlib_repr not in repr(vars(module))

    def test_regular_import_nested_with_rename(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import collections.abc as xyz
"""
        module = create_sample_module(tmp_path, source)

        # Make sure the right proxy is in the namespace.
        expected_xyz_repr = "<key for 'xyz' import>: <proxy for 'collections.abc' import>"
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

    def test_from_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    from inspect import isfunction, signature
"""
        module = create_sample_module(tmp_path, source)

        expected_isfunction_repr = "<key for 'isfunction' import>: <proxy for 'inspect.isfunction' import>"
        expected_signature_repr = "<key for 'signature' import>: <proxy for 'inspect.signature' import>"
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

    def test_from_import_with_rename(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    from inspect import Signature as MySignature
"""
        module = create_sample_module(tmp_path, source)

        expected_my_signature_repr = "<key for 'MySignature' import>: <proxy for 'inspect.Signature' import>"
        assert expected_my_signature_repr in repr(vars(module))

        with pytest.raises(NameError):
            exec("inspect", vars(module))

        with pytest.raises(NameError):
            exec("Signature", vars(module))

        assert expected_my_signature_repr in repr(vars(module))
        assert str(module.MySignature) == "<class 'inspect.Signature'>"  # Resolves on use.
        assert expected_my_signature_repr not in repr(vars(module))
        assert module.MySignature is sys.modules["inspect"].Signature

    def test_deferred_header_in_instrumented_pycache(self, tmp_path: Path):
        """Test that the defer_imports-specific bytecode header is being prepended to the bytecode cache files of
        defer_imports-instrumented modules.
        """

        source = """\
import defer_imports

with defer_imports.until_use:
    import asyncio
"""
        module = create_sample_module(tmp_path, source)
        assert module.__spec__ is not None
        assert module.__spec__.origin is not None

        expected_cache = Path(importlib.util.cache_from_source(module.__spec__.origin))
        assert expected_cache.is_file()

        with expected_cache.open("rb") as fp:
            header = fp.read(len(_BYTECODE_HEADER))
        assert header == _BYTECODE_HEADER

    def test_error_if_non_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    print("Hello world")
"""
        module = create_sample_module(tmp_path, source, exec_mod=False)
        spec = module.__spec__
        assert spec is not None
        assert spec.loader is not None

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert module.__spec__ is not None
        assert exc_info.value.filename == str(module.__spec__.origin)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == 'print("Hello world")'

    def test_error_if_wildcard_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    from typing import *
"""
        module = create_sample_module(tmp_path, source, exec_mod=False)
        spec = module.__spec__
        assert spec is not None
        assert spec.loader is not None

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert module.__spec__ is not None
        assert exc_info.value.filename == str(module.__spec__.origin)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == "from typing import *"

    def test_top_level_and_submodules_1(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import importlib
    import importlib.abc
    import importlib.util
"""
        module = create_sample_module(tmp_path, source)

        # Prevent the caching of these from interfering with the test.
        for mod in ("importlib", "importlib.abc", "importlib.util"):
            sys.modules.pop(mod, None)

        expected_importlib_repr = "<key for 'importlib' import>: <proxy for 'importlib' import>"
        expected_importlib_abc_repr = "<key for 'abc' import>: <proxy for 'importlib.abc' import>"
        expected_importlib_util_repr = "<key for 'util' import>: <proxy for 'importlib.util' import>"

        # Test that the importlib proxy is here and then resolves.
        assert expected_importlib_repr in repr(vars(module))
        assert module.importlib
        assert expected_importlib_repr not in repr(vars(module))

        # Test that the nested proxies carry over to the resolved importlib.
        module_importlib_vars = cast("dict[str, object]", vars(module.importlib))

        assert expected_importlib_abc_repr in repr(module_importlib_vars)
        assert expected_importlib_util_repr in repr(module_importlib_vars)

        assert expected_importlib_abc_repr in repr(module_importlib_vars)
        assert module.importlib.abc.__spec__.name == "importlib.abc"
        assert expected_importlib_abc_repr not in repr(module_importlib_vars)

        assert expected_importlib_util_repr in repr(module_importlib_vars)
        assert module.importlib.util
        assert expected_importlib_util_repr not in repr(module_importlib_vars)

    def test_top_level_and_submodules_2(self, tmp_path: Path):
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
        create_sample_module(tmp_path, source)

    def test_mixed_from_same_module(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use:
    import asyncio
    from asyncio import base_events
    from asyncio import base_futures
"""
        module = create_sample_module(tmp_path, source)

        expected_asyncio_repr = "<key for 'asyncio' import>: <proxy for 'asyncio' import>"
        expected_asyncio_base_events_repr = "<key for 'base_events' import>: <proxy for 'asyncio.base_events' import>"
        expected_asyncio_base_futures_repr = (
            "<key for 'base_futures' import>: <proxy for 'asyncio.base_futures' import>"
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

    def test_relative_imports(self, tmp_path: Path):
        """Test a synthetic package that uses relative imports within defer_imports.until_use blocks."""

        package_name = "sample_pkg"
        dir_contents = {
            package_name: {
                "__init__.py": "\n".join(
                    (
                        "import defer_imports",
                        "",
                        "with defer_imports.until_use:",
                        "    from . import a",
                        "    from .a import A",
                        "    from .b import B",
                    )
                ),
                "a.py": "\n".join(
                    (
                        "class A:",
                        "    def __init__(self, val: object):",
                        "        self.val = val",
                    )
                ),
                "b.py": "\n".join(
                    (
                        "class B:",
                        "    def __init__(self, val: object):",
                        "        self.val = val",
                    )
                ),
            }
        }
        create_dir_tree(tmp_path, dir_contents)

        package_init_path = str(tmp_path / package_name / "__init__.py")
        loader = _DIFileLoader(package_name, package_init_path)
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_init_path,
            loader=loader,
            submodule_search_locations=[],  # A signal that this is a package.
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

        module_locals_repr = repr(vars(module))
        assert "<key for 'a' import>: <proxy for 'sample_pkg.a' import>" in module_locals_repr
        assert "<key for 'A' import>: <proxy for 'sample_pkg.a.A' import>" in module_locals_repr
        assert "<key for 'B' import>: <proxy for 'sample_pkg.b.B' import>" in module_locals_repr

        assert module.A
        assert repr(module.A("hello")).startswith("<sample_pkg.a.A object at")

    def test_circular_imports(self, tmp_path: Path):
        """Test a synthetic package that does circular imports."""

        package_name = "circular_pkg"
        dir_contents = {
            package_name: {
                "__init__.py": "\n".join(
                    (
                        "import defer_imports",
                        "",
                        "with defer_imports.until_use:",
                        "    import circular_pkg.main",
                    )
                ),
                "main.py": "\n".join(("from .x import X2", "X2()")),
                "x.py": "\n".join(
                    (
                        "def X1():",
                        '    return "X"',
                        "",
                        "from .y import Y1",
                        "",
                        "def X2():",
                        "    return Y1()",
                    )
                ),
                "y.py": "\n".join(
                    (
                        "def Y1():",
                        "    return 'Y'",
                        "",
                        "from .x import X2",
                        "",
                        "def Y2():",
                        "    return X2()",
                    )
                ),
            }
        }
        create_dir_tree(tmp_path, dir_contents)

        package_init_path = str(tmp_path / package_name / "__init__.py")
        loader = _DIFileLoader(package_name, package_init_path)
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_init_path,
            loader=loader,
            submodule_search_locations=[],  # A signal that this is a package.
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

        assert module

    def test_import_stdlib(self):
        """Test that defer_imports.until_use works when wrapping imports for most of the stdlib."""

        # The path finder for the tests directory is already cached, so we need to temporarily reset that entry.
        with unittest.mock.patch.dict(sys.path_importer_cache):
            sys.path_importer_cache.pop(str(Path(__file__).parent), None)

            with import_hook(uninstall_after=True):
                import tests.sample_stdlib_imports

            # Sample-check the __future__ import.
            expected_future_import = "<key for '__future__' import>: <proxy for '__future__' import>"
            assert expected_future_import in repr(vars(tests.sample_stdlib_imports))

    @pytest.mark.xfail(reason="Leaking patch problem is currently out of scope.")  # pragma: no cover
    def test_leaking_patch(self, tmp_path: Path):
        """Test a synthetic package that demonstrates the "leaking patch" problem.

        Source: https://github.com/bswck/slothy/tree/bd0828a8dd9af63ca5c85340a70a14a76a6b714f/tests/leaking_patch
        """

        package_name = "leaking_patch_pkg"
        dir_contents = {
            package_name: {
                "__init__.py": "",
                "a.py": "\n".join(
                    (
                        "import defer_imports",
                        "",
                        "with defer_imports.until_use:",
                        "    from .b import B",
                    )
                ),
                "b.py": 'B = "original thing"',
                "patching.py": "\n".join(
                    (
                        "from unittest import mock",
                        "",
                        'patcher = mock.patch("leaking_patch_pkg.b.B", "patched thing", create=True)',
                        "mock_B = patcher.start()",
                    )
                ),
            }
        }
        create_dir_tree(tmp_path, dir_contents)

        package_init_path = str(tmp_path / package_name / "__init__.py")
        loader = _DIFileLoader(package_name, package_init_path)
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_init_path,
            loader=loader,
            submodule_search_locations=[],  # A signal that this is a package.
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)

        sys.modules[package_name] = module
        spec.loader.exec_module(module)

        exec(f"import {package_name}.patching; from {package_name}.b import B", vars(module))
        assert module.B == "original thing"

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="type statements are only valid in 3.12+")
    def test_3_12_type_statement(self, tmp_path: Path):
        """Test that a proxy within a type statement doesn't resolve until accessed via .__value__."""

        package_name = "type_stmt_pkg"
        dir_contents = {
            package_name: {
                "__init__.py": "\n".join(
                    (
                        "import defer_imports",
                        "",
                        "with defer_imports.until_use:",
                        "    from .exp import Expensive",
                        "",
                        "type ManyExpensive = tuple[Expensive, ...]",
                    )
                ),
                "exp.py": "class Expensive: ...",
            }
        }
        create_dir_tree(tmp_path, dir_contents)

        package_init_path = str(tmp_path / package_name / "__init__.py")
        loader = _DIFileLoader(package_name, package_init_path)
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_init_path,
            loader=loader,
            submodule_search_locations=[],  # A signal that this is a package.
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

        expected_proxy_repr = "<key for 'Expensive' import>: <proxy for 'type_stmt_pkg.exp.Expensive' import>"

        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive) == "ManyExpensive"
        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive.__value__) == "tuple[type_stmt_pkg.exp.Expensive, ...]"
        assert expected_proxy_repr not in repr(vars(module))


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
        -   Add ``pytest.mark.parametrize("q", range(50))`` to the test for the repetitions.
        -   Run ``hatch test -py pypy3.9,pypy3.10 -- tests/test_deferred.py::test_thread_safety``.
    """

    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""
    module = create_sample_module(tmp_path, source)

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
                self.result = self._target(*self._args, **self._kwargs)  # pyright: ignore  # noqa: PGH003
            except Exception as exc:  # noqa: BLE001
                self.exc = exc
            finally:
                del self._target, self._args, self._kwargs  # pyright: ignore  # noqa: PGH003

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
        assert callable(thread.result)  # pyright: ignore  # noqa: PGH003
