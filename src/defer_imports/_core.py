# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""The implementation details for defer_imports's magic."""

from __future__ import annotations

import ast
import builtins
import contextvars
import io
import sys
import tokenize
import warnings
from collections import deque
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, PathFinder, SourceFileLoader
from itertools import islice, takewhile
from threading import RLock

from . import _typing as _tp


__version__ = "0.0.2"


# ============================================================================
# region -------- Vendored helpers --------
#
# The helper functions should reflect the behavior of the corresponding functions in all supported CPython versions.
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


def sanity_check(name: str, package: _tp.Optional[str], level: int) -> None:
    """Verify arguments are "sane".

    Notes
    -----
    Slightly modified version of importlib._bootstrap._sanity_check to avoid depending on an an implementation detail
    module at runtime.
    """

    if not isinstance(name, str):  # pyright: ignore [reportUnnecessaryIsInstance]
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


def calc___package__(globals: _tp.MutableMapping[str, _tp.Any]) -> _tp.Optional[str]:
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    Notes
    -----
    Slightly modified version of importlib._bootstrap._calc___package__ to avoid depending on an implementation detail
    module at runtime.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")

    if package is not None:
        if spec is not None and package != spec.parent:
            category = DeprecationWarning if sys.version_info >= (3, 12) else ImportWarning
            warnings.warn(
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",
                category,
                stacklevel=3,
            )
        return package

    if spec is not None:
        return spec.parent

    warnings.warn(
        "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__",
        ImportWarning,
        stacklevel=3,
    )
    package = globals["__name__"]
    if "__path__" not in globals:
        package = package.rpartition(".")[0]  # pyright: ignore [reportOptionalMemberAccess]

    return package


def resolve_name(name: str, package: str, level: int) -> str:
    """Resolve a relative module name to an absolute one.

    Notes
    -----
    Slightly modified version of importlib._bootstrap._resolve_name to avoid depending on an implementation detail
    module at runtime.
    """

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


# endregion


# ============================================================================
# region -------- Compile-time hook --------
# ============================================================================


StrPath: _tp.TypeAlias = "_tp.Union[str, _tp.PathLike[str]]"
SourceData: _tp.TypeAlias = "_tp.Union[_tp.ReadableBuffer, str, ast.Module, ast.Expression, ast.Interactive]"


should_instrument_globally = contextvars.ContextVar("should_instrument_globally", default=False)
"""Whether the instrumentation should apply globally."""


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
        encoding: str,
    ) -> None:
        self.data = data
        self.filepath = filepath
        self.encoding = encoding

        self.scope_depth = 0
        self.escape_hatch_depth = 0

    # region ---- Scope tracking ----

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Track Python scope changes. Used to determine if defer_imports.until_use usage is global."""

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
        """Track if the visitor is within a try-except block or a non-defer with statement."""

        self.escape_hatch_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.escape_hatch_depth -= 1

    visit_Try = _visit_eager_import_block

    if sys.version_info >= (3, 11):
        visit_TryStar = _visit_eager_import_block

    # endregion

    # region ---- Until_use-wrapped imports instrumentation ----

    def _decode_source(self) -> str:
        """Get the source code corresponding to the given data."""

        if isinstance(self.data, ast.AST):
            # NOTE: An attempt is made here, but the node location information likely won't match up.
            return ast.unparse(self.data)
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
            value=ast.Call(ast.Name("locals", ctx=ast.Load()), args=[], keywords=[]),
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

        new_import_nodes: list[ast.stmt] = list(import_nodes)

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
            self._visit_eager_import_block(node)

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
                isinstance(sub, ast.Expr)
                and isinstance(sub.value, ast.Constant)
                and isinstance(sub.value.value, str)
                and expect_docstring
            ):
                expect_docstring = False
            elif isinstance(sub, ast.ImportFrom) and sub.module == "__future__" and sub.level == 0:
                pass
            else:
                break

            position += 1

        # Import defer classes.
        if should_instrument_globally.get():
            top_level_import = ast.Import(names=[ast.alias(name="defer_imports")])
            node.body.insert(position, top_level_import)
            position += 1

        defer_class_names = ("DeferredImportKey", "DeferredImportProxy")

        defer_aliases = [ast.alias(name=name, asname=f"@{name}") for name in defer_class_names]
        key_and_proxy_import = ast.ImportFrom(module="defer_imports._core", names=defer_aliases, level=0)
        node.body.insert(position, key_and_proxy_import)

        # Clean up the namespace.
        key_and_proxy_names: list[ast.expr] = [ast.Name(f"@{name}", ctx=ast.Del()) for name in defer_class_names]
        node.body.append(ast.Delete(targets=key_and_proxy_names))

        return self.generic_visit(node)

    # endregion

    # region ---- Global imports instrumentation ----

    @staticmethod
    def _identify_regular_import(obj: object) -> _tp.TypeGuard[_tp.Union[ast.Import, ast.ImportFrom]]:
        """Check if a given object is an import AST without wildcards."""

        return isinstance(obj, (ast.Import, ast.ImportFrom)) and obj.names[0].name != "*"

    @staticmethod
    def _is_defer_imports_import(node: _tp.Union[ast.Import, ast.ImportFrom]) -> bool:
        """Check if the given import node imports from defer_imports."""

        if isinstance(node, ast.Import):
            return any(alias.name.partition(".")[0] == "defer_imports" for alias in node.names)
        else:
            return node.module is not None and node.module.partition(".")[0] == "defer_imports"

    def _wrap_import_stmts(self, nodes: list[ast.stmt], start: int) -> ast.With:
        """Wrap a list of consecutive import nodes from a list of statements in a "defer_imports.until_use" block and
        instrument them.

        The first node must be guaranteed to be an import node.
        """

        import_range = tuple(takewhile(lambda i: self._identify_regular_import(nodes[i]), range(start, len(nodes))))
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

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Called if no explicit visitor function exists for a node.

        Summary
        -------
        Almost a copy of ast.NodeVisitor.generic_vist, but intercepts global sequences of import statements to wrap
        them in a "with defer_imports.until_use" block and instrument them.
        """

        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values: list[_tp.Any] = []
                for i, value in enumerate(old_value):  # pyright: ignore
                    if (
                        # Only do this when the user has enabled global instrumentation.
                        should_instrument_globally.get()
                        # Only instrument import nodes that we are prepared to handle.
                        and self._identify_regular_import(value)  # pyright: ignore [reportUnknownArgumentType]
                        # Only instrument imports in global scopes.
                        and self.scope_depth == 0
                        # Only instrument imports outside of defer "with" blocks.
                        and self.escape_hatch_depth == 0
                        and not self._is_defer_imports_import(value)
                    ):
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

    tok_NAME, tok_OP = tokenize.NAME, tokenize.OP

    if isinstance(data, str):
        token_stream = tokenize.generate_tokens(io.StringIO(data).readline)
        encoding = "utf-8"
    else:
        token_stream = tokenize.tokenize(io.BytesIO(data).readline)
        encoding = next(token_stream).string

    uses_defer = any(
        (tok1.type == tok_NAME and tok1.string == "with")
        and (tok2.type == tok_NAME and tok2.string == "defer_imports")
        and (tok3.type == tok_OP and tok3.string == ".")
        and (tok4.type == tok_NAME and tok4.string == "until_use")
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

    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: SourceData,
        path: _tp.Union[StrPath, _tp.ReadableBuffer],
        *,
        _optimize: int = -1,
    ) -> _tp.CodeType:
        # NOTE: InspectLoader is the virtual superclass of SourceFileLoader thanks to ABC registration, so typeshed
        #       reflects that. However, there's some mismatch in source_to_code signatures. Can it be fixed with a PR?

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
            # TODO: This isn't safe when a syntax error occurs since we modify the tree but raise before fixing node
            #       locations. This also makes it difficult to point at the actual source code location. Find a way to
            #       deepcopy the tree first.
            orig_ast = data
        else:
            orig_ast = ast.parse(data, path, "exec")

        transformer = DeferredInstrumenter(data, path, encoding)
        new_ast = ast.fix_missing_locations(transformer.visit(orig_ast))

        return super().source_to_code(new_ast, path, _optimize=_optimize)  # pyright: ignore # See note above.

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


DEFERRED_PATH_HOOK = FileFinder.path_hook((DeferredFileLoader, SOURCE_SUFFIXES))


# endregion


# ============================================================================
# region -------- Runtime hook --------
# ============================================================================


original_import = contextvars.ContextVar("original_import", default=builtins.__import__)
"""What builtins.__import__ currently points to."""

is_deferred = contextvars.ContextVar("is_deferred", default=False)
"""Whether imports should be deferred."""


class DeferredImportProxy:
    """Proxy for a deferred __import__ call."""

    def __init__(
        self,
        name: str,
        global_ns: _tp.MutableMapping[str, object],
        local_ns: _tp.MutableMapping[str, object],
        fromlist: _tp.Sequence[str],
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

    def __getattr__(self, name: str, /) -> _tp.Self:
        if name in self.defer_proxy_fromlist:
            from_proxy = type(self)(*self.defer_proxy_import_args)
            from_proxy.defer_proxy_fromlist = (name,)
            return from_proxy

        elif name == self.defer_proxy_name.rpartition(".")[2]:
            submodule_proxy = type(self)(*self.defer_proxy_import_args)
            submodule_proxy.defer_proxy_sub = name
            return submodule_proxy

        else:
            msg = f"proxy for module {self.defer_proxy_name!r} has no attribute {name!r}"
            raise AttributeError(msg)


class DeferredImportKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_str", "defer_key_proxy", "is_resolving", "lock")

    def __new__(cls, key: str, proxy: DeferredImportProxy, /) -> _tp.Self:
        return super().__new__(cls, key)

    def __init__(self, key: str, proxy: DeferredImportProxy, /) -> None:
        self.defer_key_str = str(key)
        self.defer_key_proxy = proxy

        self.is_resolving = False
        self.lock = RLock()

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if self.defer_key_str != value:
            return False

        # Only the first thread to grab the lock should resolve the deferred import.
        with self.lock:
            # Reentrant calls from the same thread shouldn't re-trigger the resolution.
            # This can be caused by self-referential imports, e.g. within __init__.py files.
            if not self.is_resolving:
                self.is_resolving = True

                if not is_deferred.get():
                    self._resolve()

        return True

    def __hash__(self) -> int:
        return hash(self.defer_key_str)

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.defer_key_proxy

        # Perform the original __import__ and pray.
        module: _tp.ModuleType = original_import.get()(*proxy.defer_proxy_import_args)

        # Transfer nested proxies over to the resolved module.
        module_vars = vars(module)
        for attr_key, attr_val in vars(proxy).items():
            if isinstance(attr_val, DeferredImportProxy) and not hasattr(module, attr_key):
                # This could have used setattr() if pypy didn't normalize the attr key type to str, so we resort to
                # direct placement in the module's __dict__ to avoid that.
                module_vars[DeferredImportKey(attr_key, attr_val)] = attr_val

                # Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = module_vars

        # Replace the proxy with the resolved module or module attribute in the relevant namespace.

        # 1. Get the regular string key and the relevant namespace.
        key = self.defer_key_str
        namespace = proxy.defer_proxy_local_ns

        # 2. Replace the deferred version of the key to avoid it sticking around.
        #    This will trigger __eq__ again, so use is_deferred to prevent recursive resolution.
        _is_def_tok = is_deferred.set(True)
        try:
            namespace[key] = namespace.pop(key)
        finally:
            is_deferred.reset(_is_def_tok)

        # 3. Resolve any requested attribute access.
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def deferred___import__(  # noqa: ANN202
    name: str,
    globals: _tp.MutableMapping[str, object],
    locals: _tp.MutableMapping[str, object],
    fromlist: _tp.Optional[_tp.Sequence[str]] = None,
    level: int = 0,
):
    """An limited replacement for __import__ that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()

    package = calc___package__(locals)
    sanity_check(name, package, level)

    # Resolve the names of relative imports.
    if level > 0:
        name = resolve_name(name, package, level)  # pyright: ignore [reportArgumentType]
        level = 0

    # Handle submodule imports if relevant top-level imports already occurred in the call site's module.
    if not fromlist and ("." in name):
        name_parts = name.split(".")
        try:
            # TODO: Consider adding a condition that base_parent must be a ModuleType or a DeferredImportProxy, to
            #       avoid attaching proxies to a random thing that would've normally been clobbered by the import?
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # Nest submodule proxies as needed.
            for limit, attr_name in enumerate(name_parts[1:], start=2):
                if attr_name not in vars(parent):
                    nested_proxy = DeferredImportProxy(".".join(name_parts[:limit]), globals, locals, (), level)
                    nested_proxy.defer_proxy_sub = attr_name
                    setattr(parent, attr_name, nested_proxy)
                    parent = nested_proxy
                else:
                    parent = getattr(parent, attr_name)

            return base_parent

    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# ============================================================================
# region -------- Public API --------
# ============================================================================


def install_import_hook(*, is_global: bool = False) -> ImportHookContext:
    """Insert defer_imports's path hook right before the default FileFinder one in sys.path_hooks.

    This should be run before the rest of your code. One place to put it is in __init__.py of your package.

    Returns
    -------
    ImportHookContext
        A object that can be used to uninstall the import hook, either manually by calling its uninstall method or
        automatically by using it as a context manager.
    """

    global_apply_tok = should_instrument_globally.set(is_global)

    if DEFERRED_PATH_HOOK not in sys.path_hooks:
        # NOTE: PathFinder.invalidate_caches() is expensive because it imports importlib.metadata, but we have to just bear
        #       that for now, unfortunately. Price of being a good citizen, I suppose.
        for i, hook in enumerate(sys.path_hooks):
            if hook.__qualname__.startswith("FileFinder.path_hook"):
                sys.path_hooks.insert(i, DEFERRED_PATH_HOOK)
                PathFinder.invalidate_caches()
                break

    return ImportHookContext(global_apply_tok)


class ImportHookContext:
    def __init__(self, tok: contextvars.Token[bool]) -> None:
        self._tok = tok

    def __enter__(self) -> _tp.Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        """Remove defer_imports's path hook if it's in sys.path_hooks."""

        # Ensure the token is only used once.
        if self._tok is not None:
            should_instrument_globally.reset(self._tok)
            self._tok = None

        try:
            sys.path_hooks.remove(DEFERRED_PATH_HOOK)
        except ValueError:
            pass
        else:
            # NOTE: Use the same invalidation mechanism as install_import_hook() does.
            PathFinder.invalidate_caches()


@_tp.final
class DeferredContext:
    """The type for defer_imports.until_use."""

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = is_deferred.set(True)
        self._import_ctx_token = original_import.set(builtins.__import__)
        builtins.__import__ = deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        original_import.reset(self._import_ctx_token)
        is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = original_import.get()


until_use: _tp.Final[DeferredContext] = DeferredContext()
"""A context manager within which imports occur lazily. Not reentrant.

This will not work correctly if install_defer_import_hook() was not called first elsewhere.

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


# endregion
