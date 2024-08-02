import importlib.util
from pathlib import Path

from deferred._core import DeferredImportFixLoader


def test_error_import_in_function(tmp_path: Path):
    sample_text = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    import asyncio
    import asyncio.base_events
    import asyncio.base_futures
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
