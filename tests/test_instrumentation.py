"""Tests for deferred's compile-time transformations."""

import ast
import io
import tokenize

import pytest

from deferred._core import DeferredImportInstrumenter


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    import inspect
""",
            """\
from deferred._core import DeferredImportKey as @DeferredImportKey, DeferredImportProxy as @DeferredImportProxy
from deferred import defer_imports_until_use
with defer_imports_until_use:
    @global_ns = globals()
    @temp_proxy = None
    import inspect
    if type(inspect) is @DeferredImportProxy:
        @temp_proxy = @global_ns.pop('inspect')
        @global_ns[@DeferredImportKey('inspect', @temp_proxy)] = @temp_proxy
    del @temp_proxy
    del @global_ns
del @DeferredImportKey
del @DeferredImportProxy
""",
            id="regular import",
        ),
        pytest.param(
            """\
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import importlib
    import importlib.abc
""",
            """\
from deferred._core import DeferredImportKey as @DeferredImportKey, DeferredImportProxy as @DeferredImportProxy
from deferred import defer_imports_until_use
with defer_imports_until_use:
    @global_ns = globals()
    @temp_proxy = None
    import importlib
    if type(importlib) is @DeferredImportProxy:
        @temp_proxy = @global_ns.pop('importlib')
        @global_ns[@DeferredImportKey('importlib', @temp_proxy)] = @temp_proxy
    import importlib.abc
    if type(importlib) is @DeferredImportProxy:
        @temp_proxy = @global_ns.pop('importlib')
        @global_ns[@DeferredImportKey('importlib', @temp_proxy)] = @temp_proxy
    del @temp_proxy
    del @global_ns
del @DeferredImportKey
del @DeferredImportProxy
""",
            id="mixed import 1",
        ),
    ],
)
def test_instrumentation(before: str, after: str):
    before_bytes = before.encode()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(before_bytes).readline)

    transformer = DeferredImportInstrumenter("<unknown>", before_bytes, encoding)
    transformed_tree = ast.fix_missing_locations(transformer.visit(ast.parse(before)))

    assert f"{ast.unparse(transformed_tree)}\n" == after
