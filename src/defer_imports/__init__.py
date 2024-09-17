# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A library that implements PEP 690–esque lazy imports in pure Python, but at a user's behest within a context
manager.
"""

from ._comptime import ImportHookContext, __version__, install_import_hook
from ._runtime import DeferredContext, until_use


__all__ = ("__version__", "install_import_hook", "ImportHookContext", "until_use", "DeferredContext")
