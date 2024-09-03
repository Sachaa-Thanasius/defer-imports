# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A pure-Python implementation of PEP 690–esque lazy imports, but at a user's behest within a "defer_imports_until_use"
context manager.
"""

from ._console import DeferredInteractiveConsole
from ._core import __version__, install_defer_import_hook, uninstall_defer_import_hook, until_use


__all__ = (
    "__version__",
    "install_defer_import_hook",
    "uninstall_defer_import_hook",
    "until_use",
    "DeferredInteractiveConsole",
)
