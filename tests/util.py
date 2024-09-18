"""Test helpers."""

import importlib.util
from pathlib import Path

from defer_imports._comptime import DeferredFileLoader


def create_sample_module(
    path: Path,
    source: str,
    loader_type: type = DeferredFileLoader,
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
