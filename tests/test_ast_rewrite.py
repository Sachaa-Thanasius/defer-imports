import ast
import collections
import collections.abc
import importlib.util
import sys
import sysconfig
import threading
import typing
import unittest.mock
from importlib.abc import Loader
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

import defer_imports
from defer_imports._ast_rewrite import (
    _ACTUAL_CTX_ASNAME,
    _ACTUAL_CTX_NAME,
    _BYTECODE_HEADER,
    _PATH_HOOK,
    _TEMP_ASNAMES,
    _accumulate_dotted_parts,
    _DIFileLoader,
    _DIKey,
    _get_exact_key,
    _ImportsInstrumenter,
)


# ============================================================================
# region -------- Helpers --------
# ============================================================================


NestedMapping = collections.abc.Mapping[str, typing.Union["NestedMapping", str]]


SAMPLE_DOCSTRING = "Module docstring here"


def module_template(*lines: str) -> str:
    return "\n".join(
        (
            f"from defer_imports._ast_rewrite import {_ACTUAL_CTX_NAME} as {_ACTUAL_CTX_ASNAME}",
            "\n".join(lines),
            f"del {_ACTUAL_CTX_ASNAME}",
        )
    )


def import_template(*lines: str) -> str:
    return "\n".join(
        (
            f"with {_ACTUAL_CTX_ASNAME}():",
            "\n".join(f"    {line}" for line in lines),
            f"    del {_TEMP_ASNAMES}",
        )
    )


def asnames_template(arg: object = None) -> str:
    return f"{_TEMP_ASNAMES} = {arg!r}"


def normalize_ws(text: str) -> str:
    """Remove empty lines and normalize line endings to "\\n"."""

    return "\n".join(line for line in text.splitlines() if line.strip())


def create_dir_tree(path: Path, dir_contents: NestedMapping) -> None:
    """Create a tree of files based on a (nested) dict of file/directory names and (source) contents.

    Warning: Be careful when using escape sequences in file contents strings. Consider escaping them or using raw
    strings.
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
            msg = f"Expected a dict or a string, got {value!r}."
            raise TypeError(msg)


def create_sample_module(path: Path, source: str, loader_type: type[Loader] = _DIFileLoader, *, exec_mod: bool = True):
    """Create a sample module based on the given attributes."""

    module_name = "sample"
    module_path = path / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")

    loader = loader_type(module_name, str(module_path))  # pyright: ignore [reportCallIssue]
    spec = importlib.util.spec_from_file_location(module_name, module_path, loader=loader)
    assert spec is not None

    spec.loader_state = {"defer_whole_module": False}
    module = importlib.util.module_from_spec(spec)

    if exec_mod:
        # Use spec.loader instead of loader because of potential create_module() side-effects.
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return module


def create_sample_package(path: Path, package_name: str, dir_contents: NestedMapping):
    create_dir_tree(path, dir_contents)

    package_init_path = str(path / package_name / "__init__.py")
    loader = _DIFileLoader(package_name, package_init_path)
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init_path,
        loader=loader,
        submodule_search_locations=[],  # A signal that this is a package.
    )
    assert spec is not None

    spec.loader_state = {"defer_whole_module": False}
    module = importlib.util.module_from_spec(spec)
    # NOTE: Unlike in create_sample_module, we cache the module.
    sys.modules[package_name] = module

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


# endregion


@pytest.mark.parametrize(
    ("dotted_name", "start", "expected"),
    [
        ("a.b.c.d.e", 0, {"a", "a.b", "a.b.c", "a.b.c.d", "a.b.c.d.e"}),
        ("a.b.c.d.e", len("a.b."), {"a.b.c", "a.b.c.d", "a.b.c.d.e"}),
    ],
)
def test__accumulate_dotted_parts(dotted_name: str, start: int, expected: set[str]):
    assert _accumulate_dotted_parts(dotted_name, start) == expected


def test__get_exact_key():
    class MyStr(str):
        __slots__ = ()

    dct = {MyStr("hello"): 1, "world": 2, "people": 3}
    name = "hello"

    assert isinstance(_get_exact_key(name, dct), MyStr)


def test_path_hook_installation():
    """Test the API for putting/removing the defer_imports path hook on/from sys.path_hooks."""

    with unittest.mock.patch.object(sys, "path_hooks", list(sys.path_hooks)):
        # It shouldn't be on there by default.
        assert _PATH_HOOK not in sys.path_hooks
        before_length = len(sys.path_hooks)

        # It should not be present after just getting the hook context.
        hook_ctx = defer_imports.import_hook()
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


def test_path_hook_with_extension_module():
    """Test that extension modules still work. Ref: #18."""

    with defer_imports.import_hook():
        import regex  # noqa: F401


common_ast_rewrite_cases = [
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
        *common_ast_rewrite_cases,
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    import __future__
""",
            module_template("import defer_imports", import_template(asnames_template(), "import __future__")),
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

with defer_imports.until_use():
    import inspect
""",
            f'"""{SAMPLE_DOCSTRING}"""\n'
            + module_template("import defer_imports", import_template(asnames_template(), "import inspect")),
            id="docstring then regular import",
        ),
        pytest.param(
            """\
from __future__ import annotations

import defer_imports

with defer_imports.until_use():
    import inspect
""",
            "from __future__ import annotations\n"
            + module_template("import defer_imports", import_template(asnames_template(), "import inspect")),
            id="from __future__ then regular import",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    import inspect
""",
            module_template("import defer_imports", import_template(asnames_template(), "import inspect")),
            id="regular import",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    import inspect as i
""",
            module_template("import defer_imports", import_template(asnames_template("i"), "import inspect as i")),
            id="regular import with rename 1",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    import sys, os as so
""",
            module_template(
                "import defer_imports",
                import_template(asnames_template(), "import sys", asnames_template("so"), "import os as so"),
            ),
            id="regular import with rename 2",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    import importlib
    import importlib.abc
""",
            module_template(
                "import defer_imports",
                import_template(asnames_template(), "import importlib", asnames_template(), "import importlib.abc"),
            ),
            id="mixed imports",
        ),
        pytest.param(
            """\
import defer_imports

with defer_imports.until_use():
    from . import a
""",
            module_template("import defer_imports", import_template(asnames_template((None,)), "from . import a")),
            id="relative import",
        ),
    ],
)
def test_regular_ast_rewrite(source: str, expected_rewrite: str):
    """Test what code is generated by the instrumentation side of defer_imports."""

    transformer = _ImportsInstrumenter(source)
    new_tree = transformer.visit(ast.parse(source))
    actual_rewrite = ast.unparse(new_tree)

    assert normalize_ws(actual_rewrite) == normalize_ws(expected_rewrite)


@pytest.mark.parametrize(
    ("source", "expected_rewrite"),
    [
        *common_ast_rewrite_cases,
        pytest.param(
            "import inspect",
            module_template(import_template(asnames_template(), "import inspect")),
            id="regular import",
        ),
        pytest.param(
            "\n".join(("import hello", "import world", "import foo")),
            module_template(
                import_template(
                    asnames_template(),
                    "import hello",
                    asnames_template(),
                    "import world",
                    asnames_template(),
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
                import_template(asnames_template(), "import hello", asnames_template(), "import world"),
                "print('hello')",
                import_template(asnames_template(), "import foo"),
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
                import_template(asnames_template(), "import hello", asnames_template(), "import world"),
                "def do_the_thing(a: int) -> int:",
                "    return a",
                import_template(asnames_template(), "import foo"),
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
                import_template(asnames_template(), "import hello"),
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
                import_template(asnames_template(), "import hello"),
                "from world import *",
                import_template(asnames_template(), "import foo"),
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
                import_template(asnames_template(), "import foo"),
                "try:",
                "    import hello",
                "finally:",
                "    pass",
                import_template(asnames_template(), "import bar"),
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
                import_template(asnames_template(), "import foo"),
                "with nullcontext():",
                "    import hello",
                import_template(asnames_template(), "import bar"),
            ),
            id="avoids imports in non-defer_imports.until_use with block",
        ),
        pytest.param(
            """\
import defer_imports
import foo
with defer_imports.until_use():
    import hello
import bar
""",
            module_template(
                import_template(asnames_template(), "import defer_imports", asnames_template(), "import foo"),
                import_template(asnames_template(), "import hello"),
                import_template(asnames_template(), "import bar"),
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
            id="escape hatch: try-except",
        ),
        pytest.param(
            *["\n".join(("try:", "    print('hi')", "except:", "    print('error')", "else:", "    import foo"))] * 2,
            id="escape hatch: try-except-else",
        ),
        pytest.param(
            *["\n".join(("try:", "    pass", "finally:", "    import foo"))] * 2,
            id="escape hatch: try-finally",
        ),
    ],
)
def test_full_ast_rewrite(source: str, expected_rewrite: str):
    """Test what code is generated by the instrumentation side of defer_imports if applied at a module level."""

    transformer = _ImportsInstrumenter(source, whole_module=True)
    new_tree = transformer.visit(ast.parse(source))
    actual_rewrite = ast.unparse(new_tree)

    assert normalize_ws(actual_rewrite) == normalize_ws(expected_rewrite)


@pytest.mark.usefixtures("preserve_sys_modules")
class TestImport:
    # NOTE: A proxy's presence in a namespace is checked via stringifying the namespace and then substring matching
    # with the expected proxy repr, as that's one of the few ways to inspect it without causing it to resolve.

    @pytest.fixture(autouse=True)
    def better_key_repr(self):
        """Replace _DIKey.__repr__ with a more verbose version for all tests."""

        def verbose_repr(self: _DIKey) -> str:
            return f"<key for {super(type(self), self).__repr__()} import>"

        with unittest.mock.patch.object(_DIKey, "__repr__", verbose_repr):
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

with defer_imports.until_use():
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

    @pytest.mark.skipif(sys.dont_write_bytecode, reason="Bytecode caching is needed.")
    def test_deferred_header_in_instrumented_pycache(self, tmp_path: Path):
        """Test that the defer_imports-specific bytecode header is being prepended to the bytecode cache files of
        defer_imports-instrumented modules.
        """

        source = """\
import defer_imports

with defer_imports.until_use():
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

with defer_imports.until_use():
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
        assert exc_info.value.text == '    print("Hello world")\n'

    def test_error_if_wildcard_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use():
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
        assert exc_info.value.text == "    from typing import *\n"

    def test_error_if_multiline_non_import(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use():
    a = 1 + \
        2 + \
        3; b = 2
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
        assert exc_info.value.text == "    a = 1 +         2 +         3; b = 2\n"

    def test_name_clobbering(self, tmp_path: Path):
        source = """\
import defer_imports

importlib = None
with defer_imports.until_use():
    import importlib
"""
        module = create_sample_module(tmp_path, source)

        expected_importlib_repr = "<key for 'importlib' import>: <proxy for 'importlib' import>"
        assert expected_importlib_repr in repr(vars(module))

    def test_top_level_and_submodules_1(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use():
    import importlib
    import importlib.abc
    import importlib.util
"""
        module = create_sample_module(tmp_path, source)

        # Prevent the caching of these from interfering with the checking of each submodule's resolution.
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
        module_importlib_vars: dict[str, object] = module.importlib.__dict__

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

concurrent = None
with defer_imports.until_use():
    import concurrent.futures.process
"""
        module = create_sample_module(tmp_path, source)

        # Prevent the caching of these from interfering with the checking of each submodule's resolution.
        for mod in ("concurrent", "concurrent.futures", "concurrent.futures.process"):
            sys.modules.pop(mod, None)

        # Test that the parent module proxy is here and then resolves.
        expected_repr = "<key for 'concurrent' import>: <proxy for 'concurrent' import>"

        assert expected_repr in repr(vars(module))
        assert module.concurrent.__spec__.name == "concurrent"
        assert expected_repr not in repr(vars(module))

        # Test that the nested proxies carry over and resolve.
        expected_repr = "<key for 'futures' import>: <proxy for 'concurrent.futures' import>"

        assert expected_repr in repr(vars(module.concurrent))
        assert module.concurrent.futures.__spec__.name == "concurrent.futures"
        assert expected_repr not in repr(vars(module.concurrent))

        expected_repr = "<key for 'process' import>: <proxy for 'concurrent.futures.process' import>"

        assert expected_repr in repr(vars(module.concurrent.futures))
        assert module.concurrent.futures.process.__spec__.name == "concurrent.futures.process"
        assert expected_repr not in repr(vars(module.concurrent.futures))

    def test_top_level_and_submodules_3(self, tmp_path: Path):
        source = """\
import defer_imports

with defer_imports.until_use():
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

with defer_imports.until_use():
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
                        "with defer_imports.until_use():",
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
        module = create_sample_package(tmp_path, package_name, dir_contents)

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
                        "with defer_imports.until_use():",
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
        module = create_sample_package(tmp_path, package_name, dir_contents)

        assert module

    def test_import_stdlib(self):
        """Test that defer_imports.until_use works when wrapping imports for most of the stdlib."""

        # The path finder for the tests directory is already cached, so we need to temporarily reset that entry.
        with unittest.mock.patch.dict(sys.path_importer_cache):
            sys.path_importer_cache.pop(str(Path(__file__).parent), None)

            with defer_imports.import_hook(uninstall_after=True):
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
                        "with defer_imports.until_use():",
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
        module = create_sample_package(tmp_path, package_name, dir_contents)

        exec(f"import {package_name}.patching; from {package_name}.b import B", vars(module))
        assert module.B == "original thing"

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="type statements are only valid in 3.12+.")
    def test_3_12_type_statement(self, tmp_path: Path):
        """Test that a proxy within a type statement doesn't resolve until accessed via .__value__."""

        package_name = "type_stmt_pkg"
        dir_contents = {
            package_name: {
                "__init__.py": "\n".join(
                    (
                        "import defer_imports",
                        "",
                        "with defer_imports.until_use():",
                        "    from .exp import Expensive",
                        "",
                        "type ManyExpensive = tuple[Expensive, ...]",
                    )
                ),
                "exp.py": "class Expensive: ...",
            }
        }
        module = create_sample_package(tmp_path, package_name, dir_contents)

        expected_proxy_repr = "<key for 'Expensive' import>: <proxy for 'type_stmt_pkg.exp.Expensive' import>"

        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive) == "ManyExpensive"
        assert expected_proxy_repr in repr(vars(module))

        assert str(module.ManyExpensive.__value__) == "tuple[type_stmt_pkg.exp.Expensive, ...]"
        assert expected_proxy_repr not in repr(vars(module))

    @pytest.mark.xfail(
        sys.version_info >= (3, 13) and sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
        reason="Not safe without the GIL.",
    )
    def test_thread_safety(self, tmp_path: Path):
        """Test that trying to access a lazily loaded import from multiple threads doesn't cause race conditions.

        Based on a test for importlib.util.LazyLoader in the CPython test suite.
        """

        source = """\
import defer_imports

with defer_imports.until_use():
    import inspect
"""
        module = create_sample_module(tmp_path, source)

        num_threads = 20
        barrier = threading.Barrier(num_threads)

        results: list[object] = []

        def access_module_attr(barrier: threading.Barrier):
            barrier.wait()
            try:
                inspect_signature = module.inspect.signature
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
            else:
                results.append(inspect_signature)

        threads: list[threading.Thread] = []

        for _ in range(num_threads):
            thread = threading.Thread(target=access_module_attr, args=(barrier,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        from inspect import signature

        assert results == [signature] * num_threads
