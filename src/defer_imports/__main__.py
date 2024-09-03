# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

if __name__ == "__main__":
    from .console import DeferredInteractiveConsole

    raise SystemExit(DeferredInteractiveConsole().interact())
