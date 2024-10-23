# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A library that implements PEP 690–esque lazy imports in pure Python."""

from __future__ import annotations

import builtins
import contextvars
import importlib.util
import itertools
import sys
import threading
import zipimport
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, PathFinder, SourceFileLoader


__version__ = "0.1.0"

__all__ = (
    "install_import_hook",
    "ImportHookContext",
    "until_use",
    "DeferredContext",
)

# Defining this constant locally only works with type checkers as long as they continue to special-case variables with
# this name.
TYPE_CHECKING = False


# ============================================================================
# region -------- Lazy import bootstrapping --------
#
# A "hack" to lazily import some modules in a different way to reduce import
# time.
# ============================================================================


def _lazy_import_module(name: str, package: typing.Optional[str] = None) -> types.ModuleType:
    """Lazily import a module. Has the same signature as ``importlib.import_module()``.

    This is for limited internal usage, especially since it intentionally does not work in all cases and has not been
    evaluated for thread safety.

    Notes
    -----
    This is based on importlib code as well as recipes found in the Python 3.12 importlib docs.

    Some types of ineligible imports:

    -   from imports (where the parent is also expected to be lazily imported)
    -   submodule imports (where the parent is also expected to be lazily imported)
    -   module imports where the modules replace themselves in sys.modules during execution
        -   often done for performance reasons, like replacing onself with a C-accelerated module
        -   e.g. collections.abc in CPython 3.13
    """

    # 1. Resolve the name.
    absolute_name = importlib.util.resolve_name(name, package)
    if absolute_name in sys.modules:
        return sys.modules[absolute_name]

    # 2. Find the module's parent if it exists.
    path = None
    if "." in absolute_name:
        parent_name, _, child_name = absolute_name.rpartition(".")
        # No point delaying the load of the parent when we need to access one of its attributes immediately.
        parent_module = importlib.import_module(parent_name)
        assert parent_module.__spec__ is not None
        path = parent_module.__spec__.submodule_search_locations

    # 3. Find the module spec.
    for finder in sys.meta_path:
        spec = finder.find_spec(absolute_name, path)
        if spec is not None:
            break
    else:
        msg = f"No module named {absolute_name!r}"
        raise ModuleNotFoundError(msg, name=absolute_name)

    # 4. Wrap the module loader with importlib.util.LazyLoader.
    if spec.loader is not None:
        spec.loader = loader = importlib.util.LazyLoader(spec.loader)
    else:
        msg = "missing loader"
        raise ImportError(msg, name=spec.name)

    # 5. Execute and return the module.
    module = importlib.util.module_from_spec(spec)
    sys.modules[absolute_name] = module
    loader.exec_module(module)

    if path is not None:
        setattr(parent_module, child_name, module)  # pyright: ignore [reportPossiblyUnboundVariable]

    return module


if TYPE_CHECKING:
    import ast
    import collections
    import importlib.abc as importlib_abc
    import io
    import os
    import tokenize
    import types
    import typing
    import warnings
else:
    # fmt: off
    ast             = _lazy_import_module("ast")
    collections     = _lazy_import_module("collections")
    importlib_abc   = _lazy_import_module("importlib.abc")
    io              = _lazy_import_module("io")
    os              = _lazy_import_module("os")
    tokenize        = _lazy_import_module("tokenize")
    types           = _lazy_import_module("types")
    typing          = _lazy_import_module("typing")
    warnings        = _lazy_import_module("warnings")
    # fmt: on


# endregion


# ============================================================================
# region -------- Shims for typing and annotation symbols --------
# ============================================================================


if TYPE_CHECKING:
    _final = typing.final
else:

    def _final(f: object) -> object:
        """Decorator to indicate final methods and final classes.

        Slightly modified version of typing.final to avoid importing from typing at runtime.
        """

        try:
            f.__final__ = True  # pyright: ignore # Runtime attribute assignment
        except (AttributeError, TypeError):  # pragma: no cover
            # Skip the attribute silently if it is not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass
        return f


if not TYPE_CHECKING and sys.version_info <= (3, 11):  # pragma: <=3.11 cover

    class _PlaceholderGenericAlias(type(list[int])):
        def __repr__(self) -> str:
            name = f'typing.{super().__repr__().rpartition(".")[2]}'
            return f"<import placeholder for {name}>"

    class _PlaceholderMeta(type):
        def __repr__(self) -> str:
            return f"<import placeholder for typing.{self.__name__}>"

    class _PlaceholderGenericMeta(_PlaceholderMeta):
        def __getitem__(self, item: object) -> _PlaceholderGenericAlias:
            return _PlaceholderGenericAlias(self, item)


if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    if TYPE_CHECKING:
        from typing import TypeAlias as _TypeAlias, TypeGuard as _TypeGuard
    else:
        _TypeAlias: typing.TypeAlias = "typing.TypeAlias"
        _TypeGuard: typing.TypeAlias = "typing.TypeGuard"
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias as _TypeAlias, TypeGuard as _TypeGuard
else:  # pragma: <3.10 cover
    _TypeAlias = _PlaceholderMeta("TypeAlias", (), {"__doc__": "Placeholder for typing.TypeAlias."})
    _TypeGuard = _PlaceholderGenericMeta("TypeGuard", (), {"__doc__": "Placeholder for typing.TypeGuard."})


if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    if TYPE_CHECKING:
        from typing import Self as _Self
    else:
        _Self: _TypeAlias = "typing.Self"
elif TYPE_CHECKING:
    from typing_extensions import Self as _Self
else:  # pragma: <3.11 cover
    _Self = _PlaceholderMeta("Self", (), {"__doc__": "Placeholder for typing.Self."})


if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    if TYPE_CHECKING:
        from collections.abc import Buffer as _ReadableBuffer
    else:
        _ReadableBuffer: _TypeAlias = "collections.abc.Buffer"
elif TYPE_CHECKING:
    from typing_extensions import Buffer as _ReadableBuffer
else:  # pragma: <3.12 cover
    _ReadableBuffer: _TypeAlias = "typing.Union[bytes, bytearray, memoryview]"


# endregion


# ============================================================================
# region -------- Vendored helpers --------
#
# Helper functions vendored from CPython in some way.
# ============================================================================


def _sliding_window(
    iterable: typing.Iterable[tokenize.TokenInfo],
    n: int,
) -> typing.Generator[tuple[tokenize.TokenInfo, ...], None, None]:
    """Collect tokens into overlapping fixed-length chunks or blocks.

    Notes
    -----
    Slightly modified version of the sliding_window recipe found in the Python 3.12 itertools docs.

    Examples
    --------
    >>> source = "def func(): ..."
    >>> tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    >>> [" ".join(item.string for item in window) for window in _sliding_window(tokens, 2)]
    ['def func', 'func (', '( )', ') :', ': ...', '... ', ' ']
    """

    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def _sanity_check(name: str, package: typing.Optional[str], level: int) -> None:
    """Verify arguments are "sane".

    Notes
    -----
    Slightly modified version of importlib._bootstrap._sanity_check to avoid depending on an implementation detail
    module at runtime.
    """

    if not isinstance(name, str):  # pyright: ignore [reportUnnecessaryIsInstance] # Account for user error.
        msg = f"module name must be str, not {type(name)}"
        raise TypeError(msg)
    if level < 0:
        msg = "level must be >= 0"
        raise ValueError(msg)
    if level > 0:
        if not isinstance(package, str):
            msg = "__package__ not set to a string"
            raise TypeError(msg)
        if not package:
            msg = "attempted relative import with no known parent package"
            raise ImportError(msg)
    if not name and level == 0:
        msg = "Empty module name"
        raise ValueError(msg)


def _calc___package__(globals: typing.MutableMapping[str, typing.Any]) -> typing.Optional[str]:
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None to represent that its proper value is unknown.

    Notes
    -----
    Slightly modified version of importlib._bootstrap._calc___package__ to avoid depending on an implementation detail
    module at runtime.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")

    if package is not None:
        if spec is not None and package != spec.parent:
            if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
                category = DeprecationWarning
            else:  # pragma: <3.12 cover
                category = ImportWarning

            msg = f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})"
            warnings.warn(msg, category, stacklevel=3)

        return package
    elif spec is not None:
        return spec.parent
    else:
        msg = "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__"
        warnings.warn(msg, ImportWarning, stacklevel=3)

        package = globals["__name__"]
        if "__path__" not in globals:
            package = package.rpartition(".")[0]  # pyright: ignore [reportOptionalMemberAccess]

        return package


# endregion


# ============================================================================
# region -------- Compile-time magic --------
#
# The AST transformer, import hook machinery, and import hook API.
# ============================================================================


_StrPath: _TypeAlias = "typing.Union[str, os.PathLike[str]]"
_ModulePath: _TypeAlias = "typing.Union[_StrPath, _ReadableBuffer]"
_SourceData: _TypeAlias = "typing.Union[_ReadableBuffer, str, ast.Module, ast.Expression, ast.Interactive]"


_BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Should be updated with every version release."""


_is_loaded_using_defer = False
"""Whether the defer_imports import loader is being used to load a module."""

_is_loaded_lock = threading.Lock()
"""A lock to guard reading from and writing to _is_loaded_using_defer."""


class _DeferredInstrumenter:
    """AST transformer that instruments imports within "with defer_imports.until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.

    Notes
    -----
    The transformer doesn't subclass ast.NodeTransformer but instead vendors its logic to avoid the upfront import cost.
    Additionally, it assumes the AST being instrumented is not empty and "with defer_imports.until_use" is used
    somewhere in it.
    """

    def __init__(
        self,
        data: typing.Union[_ReadableBuffer, str, ast.AST],
        filepath: _ModulePath = "<unknown>",
        encoding: str = "utf-8",
        *,
        module_level: bool = False,
    ) -> None:
        self.data = data
        self.filepath = filepath
        self.encoding = encoding
        self.module_level = module_level

        self.scope_depth = 0
        self.escape_hatch_depth = 0

    def visit(self, node: ast.AST) -> typing.Any:
        """Visit a node."""

        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Track Python scope changes. Used to determine if a use of defer_imports.until_use is global."""

        if self.module_level:
            return node
        else:
            self.scope_depth += 1
            try:
                return self.generic_visit(node)
            finally:
                self.scope_depth -= 1

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope

    def _visit_eager_import_block(self, node: ast.AST) -> ast.AST:
        """Track if the visitor is within a try-except block or a with statement."""

        if self.module_level:
            self.escape_hatch_depth += 1
            try:
                return self.generic_visit(node)
            finally:
                self.escape_hatch_depth -= 1
        else:
            return self.generic_visit(node)

    visit_Try = _visit_eager_import_block

    if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
        visit_TryStar = _visit_eager_import_block

    def _decode_source(self) -> str:
        """Get the source code corresponding to the given data."""

        if isinstance(self.data, ast.AST):
            # NOTE: An attempt is made here, but the node location information likely won't match up.
            return ast.unparse(ast.fix_missing_locations(self.data))
        elif isinstance(self.data, str):
            return self.data
        else:
            # Do the same thing as importlib.util.decode_source().
            newline_decoder = io.IncrementalNewlineDecoder(None, translate=True)
            # Expected buffer types (bytes, bytearray, memoryview) have a decode method.
            return newline_decoder.decode(self.data.decode(self.encoding))  # pyright: ignore

    def _get_node_context(self, node: ast.stmt):  # noqa: ANN202 # Version-dependent and too verbose.
        """Get the location context for a node.

        Notes
        -----
        The return value is meant to serve as the "details" argument for SyntaxError [1]_.

        References
        ----------
        .. [1] https://docs.python.org/3.14/library/exceptions.html#SyntaxError
        """

        text = ast.get_source_segment(self._decode_source(), node, padded=True)
        context = (str(self.filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = (node.end_col_offset + 1) if (node.end_col_offset is not None) else None
            context += (node.end_lineno, end_col_offset)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create an AST for changing the name of a variable in locals if the variable is a defer_imports proxy.

        The resulting node is almost equivalent to the following code::

            if type(name) is _DeferredImportProxy:
                @temp_proxy = @local_ns.pop("name")
                local_ns[@_DeferredImportKey("name", temp_proxy)] = @temp_proxy
        """

        if "." in name:
            name = name.partition(".")[0]

        return ast.If(
            test=ast.Compare(
                left=ast.Call(
                    func=ast.Name("type", ctx=ast.Load()),
                    args=[ast.Name(name, ctx=ast.Load())],
                    keywords=[],
                ),
                ops=[ast.Is()],
                comparators=[ast.Name("@_DeferredImportProxy", ctx=ast.Load())],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name("@temp_proxy", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name("@local_ns", ctx=ast.Load()), attr="pop", ctx=ast.Load()),
                        args=[ast.Constant(name)],
                        keywords=[],
                    ),
                ),
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name("@local_ns", ctx=ast.Load()),
                            slice=ast.Call(
                                func=ast.Name("@_DeferredImportKey", ctx=ast.Load()),
                                args=[ast.Constant(name), ast.Name("@temp_proxy", ctx=ast.Load())],
                                keywords=[],
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Name("@temp_proxy", ctx=ast.Load()),
                ),
            ],
            orelse=[],
        )

    @staticmethod
    def _initialize_local_ns() -> ast.Assign:
        """Create an AST that's equivalent to "@local_ns = locals()".

        The created @local_ns variable will be used as a temporary reference to the locals to avoid calling locals()
        repeatedly.
        """

        return ast.Assign(
            targets=[ast.Name("@local_ns", ctx=ast.Store())],
            value=ast.Call(func=ast.Name("locals", ctx=ast.Load()), args=[], keywords=[]),
        )

    @staticmethod
    def _initialize_temp_proxy() -> ast.Assign:
        """Create an AST that's equivalent to "@temp_proxy = None".

        The created @temp_proxy variable will be used as a temporary reference to the current proxy being "fixed".
        """

        return ast.Assign(targets=[ast.Name("@temp_proxy", ctx=ast.Store())], value=ast.Constant(None))

    def _substitute_import_keys(self, import_nodes: list[ast.stmt]) -> list[ast.stmt]:
        """Instrument the list of imports.

        Raises
        ------
        SyntaxError:
            If any of the given nodes are not an import or are a wildcard import.
        """

        new_import_nodes = list(import_nodes)

        for i in range(len(import_nodes) - 1, -1, -1):
            node = import_nodes[i]

            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports.until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_node_context(node))  # noqa: TRY004 # Syntax error displays better.

            for alias in node.names:
                if alias.name == "*":
                    msg = "import * not allowed in with defer_imports.until_use blocks"
                    raise SyntaxError(msg, self._get_node_context(node))

                new_import_nodes.insert(i + 1, self._create_import_name_replacement(alias.asname or alias.name))

        # Initialize helper variables.
        new_import_nodes[0:0] = (self._initialize_local_ns(), self._initialize_temp_proxy())

        # Delete helper variables after all is said and done to avoid namespace pollution.
        temp_names: list[ast.expr] = [ast.Name(name, ctx=ast.Del()) for name in ("@temp_proxy", "@local_ns")]
        new_import_nodes.append(ast.Delete(targets=temp_names))

        return new_import_nodes

    @staticmethod
    def is_until_use(node: ast.With) -> bool:
        """Only accept "with defer_imports.until_use"."""

        return len(node.items) == 1 and (
            isinstance(node.items[0].context_expr, ast.Attribute)
            and isinstance(node.items[0].context_expr.value, ast.Name)
            and node.items[0].context_expr.value.id == "defer_imports"
            and node.items[0].context_expr.attr == "until_use"
        )

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that "with defer_imports.until_use" blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If any of the following conditions are met, in order of priority:
                1. "defer_imports.until_use" is being used in a class or function scope.
                2. "defer_imports.until_use" block contains a statement that isn't an import.
                3. "defer_imports.until_use" block contains a wildcard import.
        """

        if not self.is_until_use(node):
            return self._visit_eager_import_block(node)

        if self.scope_depth > 0:
            msg = "with defer_imports.until_use only allowed at module level"
            raise SyntaxError(msg, self._get_node_context(node))

        node.body = self._substitute_import_keys(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports necessary to make defer_imports.until_use work properly.

        The imports are placed after the module docstring and after __future__ imports.
        """

        expect_docstring = True
        position = 0

        for sub in node.body:
            if (
                expect_docstring
                and isinstance(sub, ast.Expr)
                and isinstance(sub.value, ast.Constant)
                and isinstance(sub.value.value, str)
            ):
                expect_docstring = False
            elif isinstance(sub, ast.ImportFrom) and sub.module == "__future__" and sub.level == 0:
                pass
            else:
                break

            position += 1

        # Add necessary defer_imports imports.
        if self.module_level:
            node.body.insert(position, ast.Import(names=[ast.alias(name="defer_imports")]))
            position += 1

        defer_class_names = ("_DeferredImportKey", "_DeferredImportProxy")

        defer_aliases = [ast.alias(name=name, asname=f"@{name}") for name in defer_class_names]
        key_and_proxy_import = ast.ImportFrom(module="defer_imports", names=defer_aliases, level=0)
        node.body.insert(position, key_and_proxy_import)

        # Clean up the namespace.
        key_and_proxy_names: list[ast.expr] = [ast.Name(f"@{name}", ctx=ast.Del()) for name in defer_class_names]
        node.body.append(ast.Delete(targets=key_and_proxy_names))

        return self.generic_visit(node)

    @staticmethod
    def _is_non_wildcard_import(obj: object) -> _TypeGuard[typing.Union[ast.Import, ast.ImportFrom]]:
        """Check if a given object is an import AST without wildcards."""

        return isinstance(obj, (ast.Import, ast.ImportFrom)) and obj.names[0].name != "*"

    @staticmethod
    def _is_defer_imports_import(node: typing.Union[ast.Import, ast.ImportFrom]) -> bool:
        """Check if the given import node imports from defer_imports."""

        if isinstance(node, ast.Import):
            return any(alias.name.partition(".")[0] == "defer_imports" for alias in node.names)
        else:
            return node.module is not None and node.module.partition(".")[0] == "defer_imports"

    def _wrap_import_stmts(self, nodes: list[typing.Any], start: int) -> ast.With:
        """Wrap consecutive import nodes within a list of statements using a "defer_imports.until_use" block and
        instrument them.

        The first node must be an import node.
        """

        def _is_non_wildcard_index(index: int) -> bool:
            return self._is_non_wildcard_import(nodes[index])

        import_range = tuple(itertools.takewhile(_is_non_wildcard_index, range(start, len(nodes))))
        import_slice = slice(import_range[0], import_range[-1] + 1)
        import_nodes = nodes[import_slice]

        instrumented_nodes = self._substitute_import_keys(import_nodes)
        wrapper_node = ast.With(
            [ast.withitem(ast.Attribute(ast.Name("defer_imports", ast.Load()), "until_use", ast.Load()))],
            body=instrumented_nodes,
        )

        nodes[import_slice] = [wrapper_node]
        return wrapper_node

    def _is_import_to_instrument(self, value: ast.AST) -> bool:
        return (
            # Only when module-level instrumentation is enabled.
            self.module_level
            # Only at global scope.
            and self.scope_depth == 0
            # Only for import nodes without wildcards.
            and self._is_non_wildcard_import(value)
            # Only outside of escape hatch blocks.
            and (self.escape_hatch_depth == 0 and not self._is_defer_imports_import(value))
        )

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Called if no explicit visitor function exists for a node.

        In addition to regular functionality, conditionally intercept global sequences of import statements to wrap them
        in "with defer_imports.until_use" blocks.
        """

        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values: list[typing.Any] = []
                for i, value in enumerate(old_value):  # pyright: ignore [reportUnknownArgumentType, reportUnknownVariableType]
                    if self._is_import_to_instrument(value):  # pyright: ignore [reportUnknownArgumentType]
                        value = self._wrap_import_stmts(old_value, i)  # noqa: PLW2901 # pyright: ignore [reportUnknownArgumentType]
                    elif isinstance(value, ast.AST):
                        value = self.visit(value)  # noqa: PLW2901
                        if value is None:
                            continue
                        if not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


def _check_source_for_defer_usage(data: typing.Union[_ReadableBuffer, str]) -> tuple[str, bool]:
    """Get the encoding of the given code and also check if it uses "with defer_imports.until_use"."""

    _TOK_NAME, _TOK_OP = tokenize.NAME, tokenize.OP

    if isinstance(data, str):
        token_stream = tokenize.generate_tokens(io.StringIO(data).readline)
        encoding = "utf-8"
    else:
        token_stream = tokenize.tokenize(io.BytesIO(data).readline)
        encoding = next(token_stream).string

    uses_defer = any(
        (tok1.type == _TOK_NAME and tok1.string == "with")
        and (tok2.type == _TOK_NAME and tok2.string == "defer_imports")
        and (tok3.type == _TOK_OP and tok3.string == ".")
        and (tok4.type == _TOK_NAME and tok4.string == "until_use")
        for tok1, tok2, tok3, tok4 in _sliding_window(token_stream, 4)
    )

    return encoding, uses_defer


def _check_ast_for_defer_usage(data: ast.AST) -> tuple[str, bool]:
    """Check if the given AST uses "with defer_imports.until_use". Also assume "utf-8" is the the encoding."""

    encoding = "utf-8"
    uses_defer = any(isinstance(node, ast.With) and _DeferredInstrumenter.is_until_use(node) for node in ast.walk(data))
    return encoding, uses_defer


class _DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments .py files which use "with defer_imports.until_use: ..."."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.defer_module_level: bool = False

    def get_data(self, path: str) -> bytes:
        """Return the data from path as raw bytes.

        Notes
        -----
        If the path points to a bytecode file, check for a defer_imports-specific header. If the header is invalid,
        raise OSError to invalidate the bytecode; importlib._boostrap_external.SourceLoader.get_code expects this [1]_.

        Another option is to monkeypatch importlib.util.cache_from_source, as beartype [2]_ and typeguard do, but that
        seems unnecessary.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        .. [2] https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/_clawimpload.py#L177-L312
        """

        data = super().get_data(path)

        if not path.endswith(tuple(BYTECODE_SUFFIXES)):
            return data

        if not data.startswith(b"defer_imports"):
            msg = '"defer_imports" header missing from bytecode'
            raise OSError(msg)

        if not data.startswith(_BYTECODE_HEADER):
            msg = '"defer_imports" header is outdated'
            raise OSError(msg)

        return data[len(_BYTECODE_HEADER) :]

    def set_data(self, path: str, data: _ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        Notes
        -----
        If the file is a bytecode one, prepend a defer_imports-specific header to it. That way, instrumented bytecode
        can be identified and invalidated later if necessary [1]_.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = _BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)

    # NOTE: Signature of SourceFileLoader.source_to_code at runtime isn't consistent with signature in typeshed.
    def source_to_code(self, data: _SourceData, path: _ModulePath, *, _optimize: int = -1) -> types.CodeType:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Compile "data" into a code object, but not before potentially instrumenting it.

        Parameters
        ----------
        data: _SourceData
            Anything that compile() can handle.
        path: _ModulePath
            Where the data was retrieved from (when applicable).

        Returns
        -------
        types.CodeType
            The compiled code object.
        """

        if not data:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        if isinstance(data, ast.AST):
            encoding, uses_defer = _check_ast_for_defer_usage(data)
        else:
            encoding, uses_defer = _check_source_for_defer_usage(data)

        if not uses_defer:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        if isinstance(data, ast.AST):
            orig_tree = data
        else:
            orig_tree = ast.parse(data, path, "exec")

        transformer = _DeferredInstrumenter(data, path, encoding, module_level=self.defer_module_level)
        new_tree = ast.fix_missing_locations(transformer.visit(orig_tree))

        return super().source_to_code(new_tree, path, _optimize=_optimize)  # pyright: ignore # See note above.

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the module, but only after getting state from module.__spec__.loader_state if present."""

        global _is_loaded_using_defer  # noqa: PLW0603 # Reading of/writing to global is guarded with threading lock.

        if (spec := module.__spec__) and spec.loader_state is not None:
            self.defer_module_level = spec.loader_state["defer_module_level"]

        # Signal to defer_imports.until_use that it's not a no-op during this module's execution.
        with _is_loaded_lock:
            _temp = _is_loaded_using_defer
            _is_loaded_using_defer = True

        try:
            return super().exec_module(module)
        finally:
            with _is_loaded_lock:
                _is_loaded_using_defer = _temp


class _DeferredFileFinder(FileFinder):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    def find_spec(self, fullname: str, target: typing.Optional[types.ModuleType] = None) -> typing.Optional[ModuleSpec]:
        """Try to find a spec for "fullname" on sys.path or "path", with some deferral state attached.

        Notes
        -----
        This utilizes ModuleSpec.loader_state to pass the deferral configuration to the loader. loader_state is
        under-documented [1]_, but it is meant to be used for this kind of thing [2]_.

        References
        ----------
        .. [1] https://github.com/python/cpython/issues/89527
        .. [2] https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader_state
        """

        spec = super().find_spec(fullname, target)

        if spec is not None and isinstance(spec.loader, _DeferredFileLoader):
            # NOTE: We're locking in defer_imports configuration for this module between finding it and loading it.
            #       However, it's possible to delay getting the configuration until module execution. Not sure what's
            #       best.
            config = _current_defer_config.get(None)

            if config is None:
                defer_module_level = False
            else:
                defer_module_level = config.apply_all or bool(
                    config.module_names
                    and (
                        fullname in config.module_names
                        or (config.recursive and any(mod.startswith(f"{fullname}.") for mod in config.module_names))
                    )
                )

                if config.loader_class is not None:
                    # We assume the class has the same initialization signature as the stdlib loader classes.
                    spec.loader = config.loader_class(fullname, spec.loader.path)  # pyright: ignore [reportCallIssue]

            spec.loader_state = {"defer_module_level": defer_module_level}

        return spec


_DEFER_PATH_HOOK = _DeferredFileFinder.path_hook((_DeferredFileLoader, SOURCE_SUFFIXES))
"""Singleton import path hook that enables defer_imports's instrumentation."""


_current_defer_config: contextvars.ContextVar[_DeferConfig] = contextvars.ContextVar("_current_defer_config")
"""The current configuration for defer_imports's instrumentation."""


class _DeferConfig:
    """Configuration container whose contents are used to determine how a module should be instrumented."""

    def __init__(
        self,
        apply_all: bool,
        module_names: typing.Sequence[str],
        recursive: bool,
        loader_class: typing.Optional[type[importlib_abc.Loader]],
    ) -> None:
        self.apply_all = apply_all
        self.module_names = module_names
        self.recursive = recursive
        self.loader_class = loader_class

    def __repr__(self) -> str:
        attrs = ("apply_all", "module_names", "recursive", "loader_class")
        return f'{type(self).__name__}({", ".join(f"{attr}={getattr(self, attr)!r}" for attr in attrs)})'


@_final
class ImportHookContext:
    """The context manager returned by install_import_hook(). Can reset defer_imports's configuration to its previous
    state and uninstall defer_import's import path hook.
    """

    def __init__(self, _config_ctx_tok: contextvars.Token[_DeferConfig], _uninstall_after: bool) -> None:
        self._tok = _config_ctx_tok
        self._uninstall_after = _uninstall_after

    def __enter__(self) -> _Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.reset()
        if self._uninstall_after:
            self.uninstall()

    def reset(self) -> None:
        """Attempt to reset the import hook configuration.

        If already reset, does nothing.
        """

        try:
            tok = self._tok
        except AttributeError:
            pass
        else:
            _current_defer_config.reset(tok)
            del self._tok

    def uninstall(self) -> None:
        """Attempt to remove the path hook from sys.path_hooks and invalidate path entry caches.

        If already removed, does nothing.
        """

        try:
            sys.path_hooks.remove(_DEFER_PATH_HOOK)
        except ValueError:
            pass
        else:
            PathFinder.invalidate_caches()


def install_import_hook(
    *,
    uninstall_after: bool = False,
    apply_all: bool = False,
    module_names: typing.Sequence[str] = (),
    recursive: bool = False,
    loader_class: typing.Optional[type[importlib_abc.Loader]] = None,
) -> ImportHookContext:
    r"""Install defer_imports's import hook if it isn't already installed, and optionally configure it. Must be called
    before using defer_imports.until_use.

    The configuration knobs are for instrumenting any global import statements, not only ones wrapped by
    defer_imports.until_use.

    This should be run before the code it is meant to affect is executed. One place to put do that is __init__.py of a
    package or app.

    Parameters
    ----------
    uninstall_after: bool, default=False
        Whether to uninstall the import hook upon exit if this function is used as a context manager.
    apply_all: bool, default=False
        Whether to apply module-level import deferral, i.e. instrumentation of all imports, to all modules henceforth.
        Has higher priority than module_names. More suitable for use in applications.
    module_names: Sequence[str], optional
        A set of modules to apply module-level import deferral to. Has lower priority than apply_all. More suitable for
        use in libraries.
    recursive: bool, default=False
        Whether module-level import deferral should apply recursively the submodules of the given module_names. Has the
        same proirity as module_names. If no module names are given, this has no effect.
    loader_class: type[importlib_abc.Loader] | None, optional
        An import loader class for defer_imports to use instead of the default machinery. If supplied, it is assumed to
        have an initialization signature matching ``(fullname: str, path: str) -> None``.

    Returns
    -------
    ImportHookContext
        A object that can be used to reset the import hook's configuration to its previous state or uninstall it, either
        automatically by using it as a context manager or manually using its rest() and uninstall methods.
    """

    if _DEFER_PATH_HOOK not in sys.path_hooks:
        try:
            # zipimporter doesn't provide find_spec until 3.10, so it technically doesn't meet the protocol.
            hook_insert_index = sys.path_hooks.index(zipimport.zipimporter) + 1  # pyright: ignore [reportArgumentType]
        except ValueError:
            hook_insert_index = 0

        sys.path_hooks.insert(hook_insert_index, _DEFER_PATH_HOOK)

    config = _DeferConfig(apply_all, module_names, recursive, loader_class)
    config_ctx_tok = _current_defer_config.set(config)
    return ImportHookContext(config_ctx_tok, uninstall_after)


# endregion


# ============================================================================
# region -------- Runtime magic --------
#
# The proxies, __import__ replacement, and until_use API.
# ============================================================================


_original_import = contextvars.ContextVar("_original_import", default=builtins.__import__)
"""What builtins.__import__ last pointed to."""

_is_deferred = contextvars.ContextVar("_is_deferred", default=False)
"""Whether imports in import statements should be deferred."""


class _DeferredImportProxy:
    """Proxy for a deferred __import__ call."""

    def __init__(
        self,
        name: str,
        global_ns: typing.MutableMapping[str, object],
        local_ns: typing.MutableMapping[str, object],
        fromlist: typing.Sequence[str],
        level: int = 0,
    ) -> None:
        self.defer_proxy_name = name
        self.defer_proxy_global_ns = global_ns
        self.defer_proxy_local_ns = local_ns
        self.defer_proxy_fromlist = fromlist
        self.defer_proxy_level = level

        # Only used in cases of non-from-import submodule aliasing a la "import a.b as c".
        self.defer_proxy_sub: str | None = None

    @property
    def defer_proxy_import_args(self):  # noqa: ANN202 # Too verbose.
        """A tuple of args that can be passed into __import__."""

        return (
            self.defer_proxy_name,
            self.defer_proxy_global_ns,
            self.defer_proxy_local_ns,
            self.defer_proxy_fromlist,
            self.defer_proxy_level,
        )

    def __repr__(self) -> str:
        if self.defer_proxy_fromlist:
            imp_stmt = f"from {self.defer_proxy_name} import {', '.join(self.defer_proxy_fromlist)}"
        elif self.defer_proxy_sub:
            imp_stmt = f"import {self.defer_proxy_name} as ..."
        else:
            imp_stmt = f"import {self.defer_proxy_name}"

        return f"<proxy for {imp_stmt!r}>"

    def __getattr__(self, name: str, /) -> _Self:
        if name in self.defer_proxy_fromlist:
            from_proxy = type(self)(*self.defer_proxy_import_args)
            from_proxy.defer_proxy_fromlist = (name,)
            return from_proxy

        elif ("." in self.defer_proxy_name) and (name == self.defer_proxy_name.rpartition(".")[2]):
            submodule_proxy = type(self)(*self.defer_proxy_import_args)
            submodule_proxy.defer_proxy_sub = name
            return submodule_proxy

        else:
            msg = f"proxy for module {self.defer_proxy_name!r} has no attribute {name!r}"
            raise AttributeError(msg)


class _DeferredImportKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_proxy", "is_resolving", "lock")

    def __new__(cls, key: str, proxy: _DeferredImportProxy, /) -> _Self:
        return super().__new__(cls, key)

    def __init__(self, key: str, proxy: _DeferredImportProxy, /) -> None:
        self.defer_key_proxy = proxy
        self.is_resolving = False
        self.lock = threading.RLock()

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if not super().__eq__(value):
            return False

        # Only the first thread to grab the lock should resolve the deferred import.
        with self.lock:
            # Reentrant calls from the same thread shouldn't re-trigger the resolution.
            # This can be caused by self-referential imports, e.g. within __init__.py files.
            if not self.is_resolving:
                self.is_resolving = True

                if not _is_deferred.get():
                    self._resolve()

        return True

    def __hash__(self) -> int:
        return super().__hash__()

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.defer_key_proxy

        # 1. Perform the original __import__ and pray.
        module: types.ModuleType = _original_import.get()(*proxy.defer_proxy_import_args)

        # 2. Transfer nested proxies over to the resolved module.
        module_vars = vars(module)
        for attr_key, attr_val in vars(proxy).items():
            if isinstance(attr_val, _DeferredImportProxy) and not hasattr(module, attr_key):
                # NOTE: This doesn't use setattr() because pypy normalizes the attr key type to str.
                module_vars[_DeferredImportKey(attr_key, attr_val)] = attr_val

                # Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = module_vars

        # 3. Replace the proxy with the resolved module or module attribute in the relevant namespace.
        # 3.1. Get the regular string key and the relevant namespace.
        key = str(self)
        namespace = proxy.defer_proxy_local_ns

        # 3.2. Replace the deferred version of the key to avoid it sticking around.
        # This will trigger __eq__ again, so we use is_deferred to prevent recursion.
        _is_def_tok = _is_deferred.set(True)
        try:
            namespace[key] = namespace.pop(key)
        finally:
            _is_deferred.reset(_is_def_tok)

        # 3.3. Resolve any requested attribute access.
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def _deferred___import__(
    name: str,
    globals: typing.MutableMapping[str, object],
    locals: typing.MutableMapping[str, object],
    fromlist: typing.Optional[typing.Sequence[str]] = None,
    level: int = 0,
) -> typing.Any:
    """An limited replacement for __import__ that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()

    package = _calc___package__(globals) if (level != 0) else None
    _sanity_check(name, package, level)

    # This technically repeats work since it recalculates level internally, but it's better for maintenance than keeping
    # a copy of importlib._bootstrap._resolve_name() around.
    if level > 0:
        name = importlib.util.resolve_name(f'{"." * level}{name}', package)
        level = 0

    # Handle submodule imports if relevant top-level imports already occurred in the call site's module.
    if not fromlist and ("." in name):
        name_parts = name.split(".")
        try:
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # NOTE: We assume that if base_parent is a ModuleType or _DeferredImportProxy, then it shouldn't be getting
            #       clobbered. Not sure if this is right, but it feels like the safest move.
            if isinstance(base_parent, (type(sys), _DeferredImportProxy)):
                # Nest submodule proxies as needed.
                for limit, attr_name in enumerate(name_parts[1:], start=2):
                    if attr_name not in vars(parent):
                        nested_proxy = _DeferredImportProxy(".".join(name_parts[:limit]), globals, locals, (), level)
                        nested_proxy.defer_proxy_sub = attr_name
                        setattr(parent, attr_name, nested_proxy)
                        parent = nested_proxy
                    else:
                        parent = getattr(parent, attr_name)

                return base_parent

    return _DeferredImportProxy(name, globals, locals, fromlist, level)


@_final
class DeferredContext:
    """A context manager within which imports occur lazily. Not reentrant. Use via defer_imports.until_use.

    This will only work correctly if install_import_hook is called first elsewhere.

    Raises
    ------
    SyntaxError
        If defer_imports.until_use is used improperly, e.g.:
            1. It is being used in a class or function scope.
            2. It contains a statement that isn't an import.
            3. It contains a wildcard import.

    Notes
    -----
    As part of its implementation, this temporarily replaces builtins.__import__.
    """

    __slots__ = ("_is_active", "_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        with _is_loaded_lock:
            self._is_active = _is_loaded_using_defer

        if self._is_active:
            self._defer_ctx_token = _is_deferred.set(True)
            self._import_ctx_token = _original_import.set(builtins.__import__)
            builtins.__import__ = _deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        if self._is_active:
            _original_import.reset(self._import_ctx_token)
            _is_deferred.reset(self._defer_ctx_token)
            builtins.__import__ = _original_import.get()


until_use: typing.Final[DeferredContext] = DeferredContext()


# endregion
