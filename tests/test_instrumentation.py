"""Tests for the compile-time hook part of deferred."""

import ast

from deferred._core import DeferredImportFixer


def test_regular_import():
    regular_case = """\
from deferred import defer_imports_until_use

with defer_imports_until_use:
    import inspect
"""

    expected = """\
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
"""

    tree = ast.parse(regular_case)
    modified_tree = ast.fix_missing_locations(DeferredImportFixer("<string>", regular_case).visit(tree))
    assert f"{ast.unparse(modified_tree)}\n" == expected


def test_mixed_import_1():
    regular_case = """\
from deferred import defer_imports_until_use


with defer_imports_until_use:
    import asyncio
    import asyncio.base_events
"""

    expected = """\
from deferred._core import DeferredImportKey as @DeferredImportKey, DeferredImportProxy as @DeferredImportProxy
from deferred import defer_imports_until_use
with defer_imports_until_use:
    @global_ns = globals()
    @temp_proxy = None
    import asyncio
    if type(asyncio) is @DeferredImportProxy:
        @temp_proxy = @global_ns.pop('asyncio')
        @global_ns[@DeferredImportKey('asyncio', @temp_proxy)] = @temp_proxy
    import asyncio.base_events
    if type(asyncio) is @DeferredImportProxy:
        @temp_proxy = @global_ns.pop('asyncio')
        @global_ns[@DeferredImportKey('asyncio', @temp_proxy)] = @temp_proxy
    del @temp_proxy
    del @global_ns
del @DeferredImportKey
"""

    tree = ast.parse(regular_case)
    modified_tree = ast.fix_missing_locations(DeferredImportFixer("<string>", regular_case).visit(tree))
    assert f"{ast.unparse(modified_tree)}\n" == expected
