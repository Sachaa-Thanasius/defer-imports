import importlib.util
from pathlib import Path

import pytest
from deferred._core import DeferredImportFixLoader


def test_error_non_import(tmp_path: Path):
    sample_text = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    print("Hello world")
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
    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == 'print("Hello world")'


def test_error_import_in_class(tmp_path: Path):
    sample_text = """\
from deferred import defer_imports_until_use

class Example:
    with defer_imports_until_use:
        from inspect import signature
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
    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "    with defer_imports_until_use:\n        from inspect import signature"


def test_error_import_in_function(tmp_path: Path):
    sample_text = """\
from deferred import defer_imports_until_use

def test():
    with defer_imports_until_use:
        import inspect

    return inspect.signature(test)
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
    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "    with defer_imports_until_use:\n        import inspect"


def test_error_wildcard_import(tmp_path: Path):
    sample_text = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    from typing import *
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
    with pytest.raises(SyntaxError) as exc_info:
        spec.loader.exec_module(module)

    assert exc_info.value.filename == str(path)
    assert exc_info.value.lineno == 4
    assert exc_info.value.offset == 5
    assert exc_info.value.text == "from typing import *"
