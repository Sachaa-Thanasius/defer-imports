import importlib.util
from pathlib import Path

from deferred._core import DeferredImportFixLoader


def test_regular_import(tmp_path: Path):
    sample_text = """\
import sys

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import inspect

assert "<key for 'inspect' import>: <proxy for 'inspect' import>" in repr(vars())
assert inspect
assert "<key for 'inspect' import>: <proxy for 'inspect' import>" not in repr(vars())

assert inspect is sys.modules["inspect"]
assert str(inspect.signature(lambda a, c: c)) == "(a, c)"
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_regular_import_with_rename(tmp_path: Path):
    sample_text = """\
import sys
from pprint import pprint

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import inspect as gin

assert "<key for 'gin' import>: <proxy for 'inspect' import>" in repr(vars())

with pytest.raises(NameError):
    inspect


assert "<key for 'gin' import>: <proxy for 'inspect' import>" in repr(vars())
assert gin
assert "<key for 'gin' import>: <proxy for 'inspect' import>" not in repr(vars())
assert sys.modules["inspect"] is gin
assert str(gin.signature(lambda a, b: b)) == "(a, b)"
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_regular_import_nested(tmp_path: Path):
    sample_text = """\
import sys

from deferred import defer_imports_until_use


with defer_imports_until_use:
    import collections.abc

assert "<key for 'collections' import>: <proxy for 'collections.abc' import>" in repr(vars())

assert collections
assert collections.abc
assert collections.abc.Sequence

assert "<key for 'collections' import>: <proxy for 'collections.abc' import>" not in repr(vars())
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_regular_import_nested_with_rename(tmp_path: Path):
    sample_text = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import collections.abc as xyz

# Make sure the right proxy is in the namespace.
assert "<key for 'xyz' import>: <proxy for 'collections.abc' import>" in repr(vars())

# Make sure the intermediate imports or proxies for them aren't in the namespace.
with pytest.raises(NameError):
    collections

with pytest.raises(NameError):
    collections.abc

# Make sure xyz resolves properly.
assert "<key for 'xyz' import>: <proxy for 'collections.abc' import>" in repr(vars())
assert xyz
assert "<key for 'xyz' import>: <proxy for 'collections.abc' import>" not in repr(vars())
assert xyz is sys.modules["collections"].abc

# Make sure only the resolved xyz remains in the namespace.
with pytest.raises(NameError):
    collections

with pytest.raises(NameError):
    collections.abc
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_from_import(tmp_path: Path):
    sample_text = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    from inspect import isfunction, signature

assert "<key for 'isfunction' import>: <proxy for 'inspect' import>" in repr(vars())
assert "<key for 'signature' import>: <proxy for 'inspect' import>" in repr(vars())

with pytest.raises(NameError):
    inspect

assert "<key for 'isfunction' import>: <proxy for 'inspect' import>" in repr(vars())
assert isfunction
assert "<key for 'isfunction' import>: <proxy for 'inspect' import>" not in repr(vars())
assert isfunction is sys.modules["inspect"].isfunction

assert "<key for 'signature' import>: <proxy for 'inspect' import>" in repr(vars())
assert signature
assert "<key for 'signature' import>: <proxy for 'inspect' import>" not in repr(vars())
assert signature is sys.modules["inspect"].signature
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_from_import_with_rename(tmp_path: Path):
    sample_text = """\
import sys

import pytest
from deferred import defer_imports_until_use


with defer_imports_until_use:
    from inspect import Signature as MySignature

assert "<key for 'MySignature' import>: <proxy for 'inspect' import>" in repr(vars())

with pytest.raises(NameError):
    inspect

with pytest.raises(NameError):
    Signature

assert "<key for 'MySignature' import>: <proxy for 'inspect' import>" in repr(vars())
assert str(MySignature) == "<class 'inspect.Signature'>"  # Resolves on use.
assert "<key for 'MySignature' import>: <proxy for 'inspect' import>" not in repr(vars())
assert MySignature is sys.modules["inspect"].Signature
"""

    # Boilerplate to dynamically create and load this module.
    tmp_file = tmp_path / "sample.py"
    tmp_file.write_text(sample_text, encoding="utf-8")

    module_name = "sample"
    path = tmp_file.resolve()

    spec = importlib.util.spec_from_file_location(
        module_name, path, loader=DeferredImportFixLoader(module_name, str(path))
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
