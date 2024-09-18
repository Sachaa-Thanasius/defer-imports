# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

from .console import DeferredInteractiveConsole


raise SystemExit(DeferredInteractiveConsole().interact())
