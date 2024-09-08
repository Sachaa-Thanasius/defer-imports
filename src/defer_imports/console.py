# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

import __future__

import ast
import code
import codeop

from . import _typing as _tp
from ._core import DeferredImportKey, DeferredImportProxy, DeferredInstrumenter


_features = [getattr(__future__, feat_name) for feat_name in __future__.all_feature_names]

__all__ = ("DeferredInteractiveConsole", "interact", "instrument_ipython")


class DeferredCompile(codeop.Compile):
    """A subclass of codeop.Compile that alters the compilation process with defer_imports's AST transformer."""

    def __call__(self, source: str, filename: str, symbol: str, **kwargs: object) -> _tp.CodeType:
        flags = self.flags
        if kwargs.get("incomplete_input", True) is False:
            flags &= ~codeop.PyCF_DONT_IMPLY_DEDENT  # pyright: ignore
            flags &= ~codeop.PyCF_ALLOW_INCOMPLETE_INPUT  # pyright: ignore
        assert isinstance(flags, int)

        og_ast_node = compile(source, filename, symbol, flags | ast.PyCF_ONLY_AST, True)
        transformer = DeferredInstrumenter(source, filename, "utf-8")
        new_ast_node = ast.fix_missing_locations(transformer.visit(og_ast_node))

        codeob = compile(new_ast_node, filename, symbol, flags, True)
        for feature in _features:
            if codeob.co_flags & feature.compiler_flag:
                self.flags |= feature.compiler_flag
        return codeob


class DeferredInteractiveConsole(code.InteractiveConsole):
    """An emulator of the interactive Python interpreter, but with defer_import's compile-time AST transformer baked in.

    This ensures that defer_imports.until_use works as intended when used directly in this console.
    """

    def __init__(self) -> None:
        local_ns = {
            "__name__": "__console__",
            "__doc__": None,
            "@DeferredImportKey": DeferredImportKey,
            "@DeferredImportProxy": DeferredImportProxy,
        }
        super().__init__(local_ns)
        self.compile = codeop.CommandCompiler()
        self.compile.compiler = DeferredCompile()


def interact() -> None:
    """Closely emulate the interactive Python console, but instrumented by defer_imports.

    This supports direct use of the defer_imports.until_use context manager.
    """

    DeferredInteractiveConsole().interact()


def instrument_ipython() -> None:
    """Add defer_import's compile-time AST transformer to a currently running IPython environment.

    This will ensure that defer_imports.until_use works as intended when used directly in a IPython console.
    """

    try:
        ipython_shell: _tp.Any = get_ipython()  # pyright: ignore
    except NameError:
        msg = "Not currently in an IPython/Jupyter environment."
        raise RuntimeError(msg) from None

    class DeferredIPythonInstrumenter(ast.NodeTransformer):
        def __init__(self):
            self.actual_transformer = DeferredInstrumenter("", "<unknown>", "utf-8")

        def visit(self, node: ast.AST) -> _tp.Any:
            self.actual_transformer.data = node
            self.actual_transformer.scope_depth = 0
            return ast.fix_missing_locations(self.actual_transformer.visit(node))

    ipython_shell.ast_transformers.append(DeferredIPythonInstrumenter())
