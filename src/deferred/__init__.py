# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""An pure-Python implementation of PEP 690–esque lazy imports, but at a user's behest within a
"defer_imports_until_use" context manager.
"""

from ._core import defer_imports_until_use, install_defer_import_hook, uninstall_defer_import_hook


__all__ = ("defer_imports_until_use", "install_defer_import_hook", "uninstall_defer_import_hook")
