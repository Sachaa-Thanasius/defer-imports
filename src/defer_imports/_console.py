# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

from code import InteractiveConsole

from ._core import DeferredImportKey, DeferredImportProxy, DeferredInstrumenter


class DeferredInteractiveConsole(InteractiveConsole):
    """An emulator of the interactive Python interpreter, but with defer_import's compile-time hook baked in to ensure that
    defer_imports.until_use works as intended directly in the console.
    """

    def __init__(self) -> None:
        local_ns = {
            "__name__": "__console__",
            "__doc__": None,
            "@DeferredImportKey": DeferredImportKey,
            "@DeferredImportProxy": DeferredImportProxy,
        }
        super().__init__(local_ns)

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        try:
            code = self.compile(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1: Input is incorrect.
            self.showsyntaxerror(filename)
            return False

        if code is None:
            # Case 2: Input is incomplete.
            return True

        # Case 3: Input is complete.
        try:
            tree = DeferredInstrumenter(filename, source, "utf-8").instrument(symbol)
            code = compile(tree, filename, symbol)
        except SyntaxError:
            # Case 1, again.
            self.showsyntaxerror(filename)
            return False

        self.runcode(code)
        return False
