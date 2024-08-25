"""Tests for defer_imports_until_use as it would be used.

Notes
-----
sys.modules.pop(module_name) before exec_module is prevalent to ensure that cached modules don't interfere with some
testing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from deferred._core import DeferredImportFileLoader


def create_sample_module(path: Path, source: str, loader_type: type):
    """Utility function for creating a sample module with the given path, source code, and loader."""

    tmp_file = path / "sample.py"
    tmp_file.write_text(source, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(module_name, path, loader=loader_type(module_name, str(path)))
    assert spec
    return spec, importlib.util.module_from_spec(spec)


class TestBasic:
    def test_regular_import(self, tmp_path: Path):
        source = """\
import sys

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import inspect

assert "<key for 'inspect' import>: <proxy for 'import inspect'>" in repr(vars())
assert inspect
assert "<key for 'inspect' import>: <proxy for 'import inspect'>" not in repr(vars())

assert inspect is sys.modules["inspect"]
assert str(inspect.signature(lambda a, c: c)) == "(a, c)"
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("inspect", None)  # Ensure it's fresh.
        spec.loader.exec_module(module)

    def test_regular_import_with_rename(self, tmp_path: Path):
        source = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import inspect as gin

assert "<key for 'gin' import>: <proxy for 'import inspect'>" in repr(vars())

with pytest.raises(NameError):
    inspect


assert "<key for 'gin' import>: <proxy for 'import inspect'>" in repr(vars())
assert gin
assert "<key for 'gin' import>: <proxy for 'import inspect'>" not in repr(vars())
assert sys.modules["inspect"] is gin
assert str(gin.signature(lambda a, b: b)) == "(a, b)"
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("inspect", None)
        spec.loader.exec_module(module)

    def test_regular_import_nested(self, tmp_path: Path):
        source = """\
import sys

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import collections.abc

assert "<key for 'collections' import>: <proxy for 'import collections.abc'>" in repr(vars())

assert collections
assert collections.abc
assert collections.abc.Sequence

assert "<key for 'collections' import>: <proxy for 'import collections.abc'>" not in repr(vars())
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("collections.abc", None)
        sys.modules.pop("collections", None)
        spec.loader.exec_module(module)

    def test_regular_import_nested_with_rename(self, tmp_path: Path):
        source = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import collections.abc as xyz

# Make sure the right proxy is in the namespace.
assert "<key for 'xyz' import>: <proxy for 'import collections.abc as ...'>" in repr(vars())

# Make sure the intermediate imports or proxies for them aren't in the namespace.
with pytest.raises(NameError):
    collections

with pytest.raises(NameError):
    collections.abc

# Make sure xyz resolves properly.
assert "<key for 'xyz' import>: <proxy for 'import collections.abc as ...'>" in repr(vars())
assert xyz
assert "<key for 'xyz' import>: <proxy for 'import collections.abc as ...'>" not in repr(vars())
assert xyz is sys.modules["collections"].abc

# Make sure only the resolved xyz remains in the namespace.
with pytest.raises(NameError):
    collections

with pytest.raises(NameError):
    collections.abc
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("collections.abc", None)
        sys.modules.pop("collections", None)
        spec.loader.exec_module(module)

    def test_from_import(self, tmp_path: Path):
        source = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    from inspect import isfunction, signature

assert "<key for 'isfunction' import>: <proxy for 'from inspect import isfunction'>" in repr(vars())
assert "<key for 'signature' import>: <proxy for 'from inspect import signature'>" in repr(vars())

with pytest.raises(NameError):
    inspect

assert "<key for 'isfunction' import>: <proxy for 'from inspect import isfunction'>" in repr(vars())
assert isfunction
assert "<key for 'isfunction' import>: <proxy for 'from inspect import isfunction'>" not in repr(vars())
assert isfunction is sys.modules["inspect"].isfunction

assert "<key for 'signature' import>: <proxy for 'from inspect import signature'>" in repr(vars())
assert signature
assert "<key for 'signature' import>: <proxy for 'from inspect import signature'>" not in repr(vars())
assert signature is sys.modules["inspect"].signature
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("inspect", None)
        spec.loader.exec_module(module)

    def test_from_import_with_rename(self, tmp_path: Path):
        source = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    from inspect import Signature as MySignature

assert "<key for 'MySignature' import>: <proxy for 'from inspect import Signature'>" in repr(vars())

with pytest.raises(NameError):
    inspect

with pytest.raises(NameError):
    Signature

assert "<key for 'MySignature' import>: <proxy for 'from inspect import Signature'>" in repr(vars())
assert str(MySignature) == "<class 'inspect.Signature'>"  # Resolves on use.
assert "<key for 'MySignature' import>: <proxy for 'from inspect import Signature'>" not in repr(vars())
assert MySignature is sys.modules["inspect"].Signature
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        sys.modules.pop("inspect", None)
        spec.loader.exec_module(module)


class TestSyntaxError:
    def test_error_if_non_import(self, tmp_path: Path):
        source = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    print("Hello world")
"""

        # Boilerplate to dynamically create and load this module.
        tmp_file = tmp_path / "sample.py"
        tmp_file.write_text(source, encoding="utf-8")

        module_name = "sample"
        path = tmp_file.resolve()

        spec = importlib.util.spec_from_file_location(
            module_name, path, loader=DeferredImportFileLoader(module_name, str(path))
        )

        assert spec
        assert spec.loader

        module = importlib.util.module_from_spec(spec)

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert exc_info.value.filename == str(path)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == 'print("Hello world")'

    def test_error_if_import_in_class(self, tmp_path: Path):
        source = """\
from deferred import defer_imports_until_use

class Example:
    with defer_imports_until_use:
        from inspect import signature
"""

        # Boilerplate to dynamically create and load this module.
        tmp_file = tmp_path / "sample.py"
        tmp_file.write_text(source, encoding="utf-8")

        module_name = "sample"
        path = tmp_file.resolve()

        spec = importlib.util.spec_from_file_location(
            module_name, path, loader=DeferredImportFileLoader(module_name, str(path))
        )

        assert spec
        assert spec.loader

        module = importlib.util.module_from_spec(spec)

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert exc_info.value.filename == str(path)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == "    with defer_imports_until_use:\n        from inspect import signature"

    def test_error_if_import_in_function(self, tmp_path: Path):
        source = """\
from deferred import defer_imports_until_use

def test():
    with defer_imports_until_use:
        import inspect

    return inspect.signature(test)
"""

        # Boilerplate to dynamically create and load this module.
        tmp_file = tmp_path / "sample.py"
        tmp_file.write_text(source, encoding="utf-8")

        module_name = "sample"
        path = tmp_file.resolve()

        spec = importlib.util.spec_from_file_location(
            module_name, path, loader=DeferredImportFileLoader(module_name, str(path))
        )

        assert spec
        assert spec.loader

        module = importlib.util.module_from_spec(spec)

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert exc_info.value.filename == str(path)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == "    with defer_imports_until_use:\n        import inspect"

    def test_error_if_wildcard_import(self, tmp_path: Path):
        source = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    from typing import *
"""

        # Boilerplate to dynamically create and load this module.
        tmp_file = tmp_path / "sample.py"
        tmp_file.write_text(source, encoding="utf-8")

        module_name = "sample"
        path = tmp_file.resolve()

        spec = importlib.util.spec_from_file_location(
            module_name, path, loader=DeferredImportFileLoader(module_name, str(path))
        )

        assert spec
        assert spec.loader

        module = importlib.util.module_from_spec(spec)

        with pytest.raises(SyntaxError) as exc_info:
            spec.loader.exec_module(module)

        assert exc_info.value.filename == str(path)
        assert exc_info.value.lineno == 4
        assert exc_info.value.offset == 5
        assert exc_info.value.text == "from typing import *"


class TestMixedImportTypes:
    def test_top_level_and_submodules(self, tmp_path: Path):
        source = """\
from pprint import pprint

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import importlib
    import importlib.abc
    import importlib.util
    import importlib.resources

print()
importlib
importlib.abc
importlib.util
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        for key in list(sys.modules):
            if key == "importlib" or key.startswith("importlib."):
                sys.modules.pop(key)
        spec.loader.exec_module(module)

    def test_mixed_from_same_module(self, tmp_path: Path):
        source = """\
import sys

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import asyncio
    from asyncio import base_events
    from asyncio import base_futures

# Make sure the right proxies are present.
assert "<key for 'asyncio' import>: <proxy for 'import asyncio'>" in repr(vars())
assert "<key for 'base_events' import>: <proxy for 'from asyncio import base_events'>" in repr(vars())
assert "<key for 'base_futures' import>: <proxy for 'from asyncio import base_futures'>" in repr(vars())

# Make sure resolving one proxy doesn't resolve or void the others.
assert base_futures
assert base_futures is sys.modules["asyncio.base_futures"]
assert "<key for 'base_futures' import>: <proxy for 'from asyncio import base_futures'>" not in repr(vars())
assert "<key for 'base_events' import>: <proxy for 'from asyncio import base_events'>" in repr(vars())
assert "<key for 'asyncio' import>: <proxy for 'import asyncio'>" in repr(vars())

assert base_events
assert base_events is sys.modules["asyncio.base_events"]
assert "<key for 'base_events' import>: <proxy for 'from asyncio import base_events'>" not in repr(vars())
assert "<key for 'base_futures' import>: <proxy for 'from asyncio import base_futures'>" not in repr(vars())
assert "<key for 'asyncio' import>: <proxy for 'import asyncio'>" in repr(vars())

assert asyncio
assert asyncio is sys.modules["asyncio"]
assert "<key for 'base_events' import>: <proxy for 'from asyncio import base_events'>" not in repr(vars())
assert "<key for 'base_futures' import>: <proxy for 'from asyncio import base_futures'>" not in repr(vars())
assert "<key for 'asyncio' import>: <proxy for 'import asyncio'>" not in repr(vars())
"""

        spec, module = create_sample_module(tmp_path, source, DeferredImportFileLoader)
        assert spec.loader
        for key in list(sys.modules):
            if key == "asyncio" or key.startswith("asyncio."):
                sys.modules.pop(key)
        spec.loader.exec_module(module)


class TestRecursiveImports:
    pass


class TestRelativeImports:
    pass


class TestFalseCircularImports:
    pass


class TestTrueCircularImports:
    pass


class TestThreadSafety:
    pass
