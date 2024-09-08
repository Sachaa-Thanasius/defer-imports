# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A library that implements PEP 690–esque lazy imports in pure Python, but at a user's behest within a context
manager.
"""

from ._core import __version__, install_defer_import_hook, uninstall_defer_import_hook, until_use


__all__ = ("__version__", "install_defer_import_hook", "uninstall_defer_import_hook", "until_use")
