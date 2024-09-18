# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""The implementation for defer_imports's compile-time magic."""

from __future__ import annotations

import ast
import io
import os
import sys
import tokenize
import zipimport
from collections import deque
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, PathFinder, SourceFileLoader
from itertools import islice, takewhile

from . import _typing as _tp


__version__ = "0.0.2"


# ============================================================================
# region -------- Helper functions --------
# ============================================================================


def sliding_window(iterable: _tp.Iterable[_tp.T], n: int) -> _tp.Generator[tuple[_tp.T, ...]]:
    """Collect data into overlapping fixed-length chunks or blocks.

    Notes
    -----
    Slightly modified version of a recipe in the Python 3.12 itertools docs.

    Examples
    --------
    >>> ["".join(window) for window in sliding_window('ABCDEFG', 4)]
    ['ABCD', 'BCDE', 'CDEF', 'DEFG']
    """

    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


# endregion


# ============================================================================
# region -------- Main implementation --------
# ============================================================================


StrPath: _tp.TypeAlias = "_tp.Union[str, os.PathLike[str]]"
SourceData: _tp.TypeAlias = "_tp.Union[_tp.ReadableBuffer, str, ast.Module, ast.Expression, ast.Interactive]"


TOK_NAME, TOK_OP = tokenize.NAME, tokenize.OP

BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Should be updated with every version release."""


class DeferredInstrumenter(ast.NodeTransformer):
    """AST transformer that instruments imports within "with defer_imports.until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.

    Notes
    -----
    This assumes the module is not empty and "with defer_imports.until_use" is used somewhere in it.
    """

    def __init__(
        self,
        data: _tp.Union[_tp.ReadableBuffer, str, ast.AST],
        filepath: _tp.Union[StrPath, _tp.ReadableBuffer],
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

    # region ---- Scope tracking ----

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Track Python scope changes. Used to determine if defer_imports.until_use usage is global."""

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

        self.escape_hatch_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.escape_hatch_depth -= 1

    visit_Try = _visit_eager_import_block

    if sys.version_info >= (3, 11):
        visit_TryStar = _visit_eager_import_block

    # endregion

    # region ---- Basic instrumentation ----

    def _decode_source(self) -> str:
        """Get the source code corresponding to the given data."""

        if isinstance(self.data, ast.AST):
            # NOTE: An attempt is made here, but the node location information likely won't match up.
            return ast.unparse(ast.fix_missing_locations(self.data))
        elif isinstance(self.data, str):
            return self.data
        else:
            # Based on importlib.util.decode_source.
            newline_decoder = io.IncrementalNewlineDecoder(None, translate=True)
            return newline_decoder.decode(self.data.decode(self.encoding))  # pyright: ignore

    def _get_node_context(self, node: ast.stmt):  # noqa: ANN202 # Version-dependent and too verbose.
        """Get the location context for a node. That context will be used as an argument to SyntaxError."""

        text = ast.get_source_segment(self._decode_source(), node, padded=True)
        context = (str(self.filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = (node.end_col_offset + 1) if (node.end_col_offset is not None) else None
            context += (node.end_lineno, end_col_offset)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create an AST for changing the name of a variable in locals if the variable is a defer_imports proxy.

        The resulting node if unparsed is almost equivalent to the following::

            if type(name) is @DeferredImportProxy:
                @temp_proxy = @local_ns.pop("name")
                local_ns[@DeferredImportKey("name", temp_proxy)] = @temp_proxy
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
                comparators=[ast.Name("@DeferredImportProxy", ctx=ast.Load())],
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
                                func=ast.Name("@DeferredImportKey", ctx=ast.Load()),
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
    def check_With_for_defer_usage(node: ast.With) -> bool:
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

        if not self.check_With_for_defer_usage(node):
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
            top_level_import = ast.Import(names=[ast.alias(name="defer_imports")])
            node.body.insert(position, top_level_import)
            position += 1

        defer_class_names = ("DeferredImportKey", "DeferredImportProxy")

        defer_aliases = [ast.alias(name=name, asname=f"@{name}") for name in defer_class_names]
        key_and_proxy_import = ast.ImportFrom(module="defer_imports._runtime", names=defer_aliases, level=0)
        node.body.insert(position, key_and_proxy_import)

        # Clean up the namespace.
        key_and_proxy_names: list[ast.expr] = [ast.Name(f"@{name}", ctx=ast.Del()) for name in defer_class_names]
        node.body.append(ast.Delete(targets=key_and_proxy_names))

        return self.generic_visit(node)

    # endregion

    # region ---- Global imports instrumentation ----

    @staticmethod
    def _is_non_wildcard_import(obj: object) -> _tp.TypeGuard[_tp.Union[ast.Import, ast.ImportFrom]]:
        """Check if a given object is an import AST without wildcards."""

        return isinstance(obj, (ast.Import, ast.ImportFrom)) and obj.names[0].name != "*"

    @staticmethod
    def _is_defer_imports_import(node: _tp.Union[ast.Import, ast.ImportFrom]) -> bool:
        """Check if the given import node imports from defer_imports."""

        if isinstance(node, ast.Import):
            return any(alias.name.partition(".")[0] == "defer_imports" for alias in node.names)
        else:
            return node.module is not None and node.module.partition(".")[0] == "defer_imports"

    def _wrap_import_stmts(self, nodes: list[_tp.Any], start: int) -> ast.With:
        """Wrap a list of consecutive import nodes from a list of statements using a "defer_imports.until_use" block and
        instrument them.

        The first node must be guaranteed to be an import node.
        """

        import_range = tuple(takewhile(lambda i: self._is_non_wildcard_import(nodes[i]), range(start, len(nodes))))
        import_slice = slice(import_range[0], import_range[-1] + 1)
        import_nodes = nodes[import_slice]

        instrumented_nodes = self._substitute_import_keys(import_nodes)
        wrapper_node = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Attribute(
                        value=ast.Name("defer_imports", ctx=ast.Load()),
                        attr="until_use",
                        ctx=ast.Load(),
                    )
                )
            ],
            body=instrumented_nodes,
        )

        nodes[import_slice] = [wrapper_node]
        return wrapper_node

    def _is_import_to_instrument(self, value: ast.AST) -> bool:
        return (
            # Only when global instrumentation is enabled.
            self.module_level
            # Only at global scope.
            and self.scope_depth == 0
            # Only with import nodes that we are prepared to handle.
            and self._is_non_wildcard_import(value)
            # Only outside of escape hatch blocks.
            and (self.escape_hatch_depth == 0 and not self._is_defer_imports_import(value))
        )

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Called if no explicit visitor function exists for a node.

        Extended Summary
        ----------------
        Almost a copy of ast.NodeVisitor.generic_visit, but we intercept global sequences of import statements to wrap
        them in a "with defer_imports.until_use" block and instrument them.
        """

        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values: list[_tp.Any] = []
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

    # endregion


def check_source_for_defer_usage(data: _tp.Union[_tp.ReadableBuffer, str]) -> tuple[str, bool]:
    """Get the encoding of the given code and also check if it uses "with defer_imports.until_use"."""

    if isinstance(data, str):
        token_stream = tokenize.generate_tokens(io.StringIO(data).readline)
        encoding = "utf-8"
    else:
        token_stream = tokenize.tokenize(io.BytesIO(data).readline)
        encoding = next(token_stream).string

    uses_defer = any(
        (tok1.type == TOK_NAME and tok1.string == "with")
        and (tok2.type == TOK_NAME and tok2.string == "defer_imports")
        and (tok3.type == TOK_OP and tok3.string == ".")
        and (tok4.type == TOK_NAME and tok4.string == "until_use")
        for tok1, tok2, tok3, tok4 in sliding_window(token_stream, 4)
    )

    return encoding, uses_defer


def check_ast_for_defer_usage(data: ast.AST) -> tuple[str, bool]:
    """Check if the given AST uses "with defer_imports.until_use". Also assume "utf-8" is the the encoding."""

    encoding = "utf-8"
    uses_defer = any(
        isinstance(node, ast.With) and DeferredInstrumenter.check_With_for_defer_usage(node) for node in ast.walk(data)
    )
    return encoding, uses_defer


class DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments .py files which use "with defer_imports.until_use: ..."."""

    defer_module_level: bool

    def create_module(self, spec: ModuleSpec) -> _tp.Optional[_tp.ModuleType]:
        # This method is always run before source_to_code in regular circumstances.
        self.defer_module_level = spec.loader_state["defer_module_level"] if (spec.loader_state is not None) else False
        return super().create_module(spec)

    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: SourceData,
        path: _tp.Union[StrPath, _tp.ReadableBuffer],
        *,
        _optimize: int = -1,
    ) -> _tp.CodeType:
        # NOTE: Signature of SourceFileLoader.source_to_code at runtime isn't consistent with the version in typeshed.

        if not data:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        # Check if the given data uses "with defer_imports.until_use".
        if isinstance(data, ast.AST):
            encoding, uses_defer = check_ast_for_defer_usage(data)
        else:
            encoding, uses_defer = check_source_for_defer_usage(data)

        if not uses_defer:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        # Instrument the AST of the given data.
        if isinstance(data, ast.AST):
            orig_tree = data
        else:
            orig_tree = ast.parse(data, path, "exec")

        transformer = DeferredInstrumenter(data, path, encoding, module_level=self.defer_module_level)
        new_tree = ast.fix_missing_locations(transformer.visit(orig_tree))

        return super().source_to_code(new_tree, path, _optimize=_optimize)  # pyright: ignore # See note above.

    def get_data(self, path: str) -> bytes:
        """Return the data from path as raw bytes.

        Notes
        -----
        If the path points to a bytecode file, check for a defer_imports-specific header. If the header is invalid, raise
        OSError to invalidate the bytecode; importlib._boostrap_external.SourceLoader.get_code expects this [1]_.

        Another option is to monkeypatch importlib.util.cache_from_source, as beartype [2]_ and typeguard [3]_ do, but that seems
        less safe.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        .. [2] https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/_clawimpload.py#L177-L312
        .. [3] https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/clawimpcache.py#L22-L26
        """

        data = super().get_data(path)

        if not path.endswith(tuple(BYTECODE_SUFFIXES)):
            return data

        if not data.startswith(b"defer_imports"):
            msg = '"defer_imports" header missing from bytecode'
            raise OSError(msg)

        if not data.startswith(BYTECODE_HEADER):
            msg = '"defer_imports" header is outdated'
            raise OSError(msg)

        return data[len(BYTECODE_HEADER) :]

    def set_data(self, path: str, data: _tp.ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        Notes
        -----
        If the file is a bytecode one, prepend a defer_imports-specific header to it. That way, instrumented bytecode can be
        identified and invalidated later if necessary [1]_.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)


LOADER_DETAILS = (DeferredFileLoader, SOURCE_SUFFIXES)


class DeferredFileFinder(FileFinder):
    def __init__(
        self,
        path: str,
        *loader_details: tuple[type[_tp.Loader], list[str]],
        defer_globally: bool = False,
        deferred_modules: _tp.Sequence[str] = (),
        recursive: bool = False,
    ) -> None:
        super().__init__(path, *loader_details)
        self.defer_globally = defer_globally
        self.deferred_modules = deferred_modules
        self.defer_recursive = recursive

    def find_spec(self, fullname: str, target: _tp.Optional[_tp.ModuleType] = None) -> _tp.Optional[ModuleSpec]:
        """Try to find a spec for "fullname" on sys.path or "path", with some modifications based on deferral state."""

        spec = super().find_spec(fullname, target)
        if spec is not None and isinstance(spec.loader, DeferredFileLoader):
            defer_module_level = self.defer_globally or (
                bool(self.deferred_modules)
                and (
                    fullname in self.deferred_modules
                    or (self.defer_recursive and any(mod.startswith(f"{fullname}.") for mod in self.deferred_modules))
                )
            )

            # It's under-documented, but spec.loader_state is meant for this kind of thing.
            # Ref: https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader_state
            # Ref: https://github.com/python/cpython/issues/89527
            spec.loader_state = {"defer_module_level": defer_module_level}
        return spec

    @classmethod
    def path_hook(
        cls,
        *loader_details: tuple[type[_tp.Loader], list[str]],
        defer_all: bool = False,
        deferred_modules: _tp.Sequence[str] = (),
        recursive: bool = False,
    ) -> _tp.Callable[[str], _tp.Self]:
        def path_hook_for_DeferredFileFinder(path: str) -> _tp.Self:
            """Path hook for DeferredFileFinder."""

            if not os.path.isdir(path):  # noqa: PTH112
                msg = "only directories are supported"
                raise ImportError(msg, path=path)

            return cls(
                path,
                *loader_details,
                defer_globally=defer_all,
                deferred_modules=deferred_modules,
                recursive=recursive,
            )

        return path_hook_for_DeferredFileFinder


def _invalidate_path_entry_caches() -> None:
    """Invalidate import-related path entry caches in some way."""

    # NOTE: PathFinder.invalidate_caches is expensive because it imports importlib.metadata, but we have to just bear
    #       that for now, unfortunately. Price of being a good citizen, I suppose.
    PathFinder.invalidate_caches()


# endregion


# ============================================================================
# region -------- Public API --------
# ============================================================================


@_tp.final
class ImportHookContext:
    """The context manager returned by install_import_hook.

    Parameters
    ----------
    path_hook: Callable[[str], PathEntryFinderProtocol]
        A path hook to uninstall. Can be uninstalled manually with the uninstall method or automatically upon
        exiting the context manager.

    Attributes
    ----------
    path_hook: Callable[[str], PathEntryFinderProtocol]
        A path hook to uninstall. Can be uninstalled manually with the uninstall method or automatically upon
        exiting the context manager.
    """

    def __init__(self, path_hook: _tp.Callable[[str], _tp.PathEntryFinderProtocol]) -> None:
        self.path_hook = path_hook

    def __enter__(self) -> _tp.Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        """Attempt to remove the path hook from sys.path_hooks. If successful, also invalidate path entry caches."""

        try:
            sys.path_hooks.remove(self.path_hook)
        except ValueError:
            pass
        else:
            _invalidate_path_entry_caches()


def install_import_hook(
    *,
    apply_all: bool = False,
    module_names: _tp.Sequence[str] = (),
    recursive: bool = False,
) -> ImportHookContext:
    """Insert a custom defer_imports path hook in sys.path_hooks. Must be called before using defer_imports.until_use.

    Also provides optional configuration for instrumenting ALL import statements, not only ones wrapped by the
    defer_imports.until_use context manager.

    This should be run before the rest of your code. One place to put it is in __init__.py of your package.

    Parameters
    ----------
    is_global: bool, default=False
        Whether to apply module-level import deferral, i.e. instrumentation of all imports, to all modules henceforth.
        Mutually exclusive with and has higher priority than module_names.
    module_names: Sequence[str], optional
        A set of modules to apply module-level import deferral to. Mutually exclusive with and has lower priority than
        is_global.
    recursive: bool, default=False
        Whether module-level import deferral should apply recursively the submodules of the given module_names. If no
        module names are given, this has no effect.

    Returns
    -------
    ImportHookContext
        A object that can be used to uninstall the import hook, either manually by calling its uninstall method or
        automatically by using it as a context manager.
    """

    path_hook = DeferredFileFinder.path_hook(
        LOADER_DETAILS,
        defer_all=apply_all,
        deferred_modules=module_names,
        recursive=recursive,
    )

    try:
        hook_insert_index = sys.path_hooks.index(zipimport.zipimporter) + 1  # pyright: ignore [reportArgumentType]
    except ValueError:
        hook_insert_index = 0

    _invalidate_path_entry_caches()
    sys.path_hooks.insert(hook_insert_index, path_hook)

    return ImportHookContext(path_hook)


# endregion
