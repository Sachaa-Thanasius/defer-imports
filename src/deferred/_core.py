from __future__ import annotations

import ast
import builtins
import collections
import contextvars
import io
import itertools
import sys
import tokenize
import warnings
from importlib.machinery import (
    BYTECODE_SUFFIXES,
    SOURCE_SUFFIXES,
    FileFinder,
    ModuleSpec,
    PathFinder,
    SourceFileLoader,
)

from . import _typing as _tp


# region -------- Compile-time hook


BYTECODE_HEADER = b"deferred0.0.1"
"""Custom header for deferred-instrumented bytecode files. Should be updated with every version release."""


class DeferredInstrumenter(ast.NodeTransformer):
    """AST transformer that "instruments" imports within "with defer_imports_until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.
    """

    def __init__(
        self,
        filepath: _tp.Union[_tp.StrPath, _tp.ReadableBuffer],
        data: _tp.ReadableBuffer,
        encoding: str,
    ) -> None:
        self.filepath = filepath
        self.data = data
        self.encoding = encoding
        self.scope_depth = 0

    def transform(self) -> _tp.Any:
        """Transform the tree created from the given data and filepath."""

        return self.visit(ast.parse(self.data, self.filepath, "exec"))

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Track Python scope changes. Used to determine if defer_imports_until_use usage is valid."""

        self.scope_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.scope_depth -= 1

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope

    def _decode_source(self) -> str:
        newline_decoder = io.IncrementalNewlineDecoder(None, translate=True)
        return newline_decoder.decode(self.data.decode(self.encoding))  # pyright: ignore

    def _get_node_context(self, node: ast.stmt):  # noqa: ANN202 # Return annotation is verbose and version-dependent.
        """Get the location context for a node. That context will be used as an argument to SyntaxError."""

        text = ast.get_source_segment(self._decode_source(), node, padded=True)
        context = (str(self.filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            context += (node.end_lineno, node.end_col_offset + 1)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create an AST for changing the name of a variable in locals if the variable is a deferred proxy."""

        if "." in name:
            name = name.partition(".")[0]

        # NOTE: Creating the AST directly is also an option, but this felt more maintainable.
        if_tree = ast.parse(
            f"if type({name}) is DeferredImportProxy:\n"
            f"    temp_proxy = local_ns.pop('{name}')\n"
            f"    local_ns[DeferredImportKey('{name}', temp_proxy)] = temp_proxy"
        )
        if_node = if_tree.body[0]
        assert isinstance(if_node, ast.If)

        # Adjust some of the names to be inaccessible by normal users.
        for node in ast.walk(if_node):
            if isinstance(node, ast.Name) and node.id in {
                "temp_proxy",
                "local_ns",
                "DeferredImportProxy",
                "DeferredImportKey",
            }:
                node.id = f"@{node.id}"
        return if_node

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
            sub_node = import_nodes[i]

            if not isinstance(sub_node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports_until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_node_context(sub_node))  # noqa: TRY004

            for alias in sub_node.names:
                if alias.name == "*":
                    msg = "import * not allowed in with defer_imports_until_use blocks"
                    raise SyntaxError(msg, self._get_node_context(sub_node))

                new_import_nodes.insert(i + 1, self._create_import_name_replacement(alias.asname or alias.name))

        # Initialize helper variables.
        new_import_nodes[0:0] = (self._initialize_local_ns(), self._initialize_temp_proxy())

        # Delete helper variables after all is said and done to avoid namespace pollution.
        temp_names: list[ast.expr] = [ast.Name(name, ctx=ast.Del()) for name in ("@temp_proxy", "@local_ns")]
        new_import_nodes.append(ast.Delete(targets=temp_names))

        return new_import_nodes

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that "with defer_imports_until_use" blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If any of the following conditions are met, in order of priority:
                1. "defer_imports_until_use" is being used in a class or function scope.
                2. "defer_imports_until_use" block contains a statement that isn't an import.
                3. "defer_imports_until_use" block contains a wildcard import.
        """

        if not (
            len(node.items) == 1
            and (
                (
                    # Allow "with defer_imports_until_use".
                    isinstance(node.items[0].context_expr, ast.Name)
                    and node.items[0].context_expr.id == "defer_imports_until_use"
                )
                or (
                    # Allow "with deferred.defer_imports_until_use".
                    isinstance(node.items[0].context_expr, ast.Attribute)
                    and isinstance(node.items[0].context_expr.value, ast.Name)
                    and node.items[0].context_expr.value.id == "deferred"
                    and node.items[0].context_expr.attr == "defer_imports_until_use"
                )
            )
        ):
            return self.generic_visit(node)

        if self.scope_depth != 0:
            msg = "with defer_imports_until_use only allowed at module level"
            raise SyntaxError(msg, self._get_node_context(node))

        node.body = self._substitute_import_keys(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports necessary to make defer_imports_until_use work properly. The import is placed after the
        module docstring and after __future__ imports.

        Notes
        -----
        This assumes the module is not empty and "with defer_imports_until_use" is used somewhere in it.
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

        # Import key and proxy classes.
        key_class = DeferredImportKey.__name__
        proxy_class = DeferredImportProxy.__name__

        defer_aliases = [ast.alias(name=name, asname=f"@{name}") for name in (key_class, proxy_class)]
        key_and_proxy_import = ast.ImportFrom(module="deferred._core", names=defer_aliases, level=0)
        node.body.insert(position, key_and_proxy_import)

        # Clean up the namespace.
        key_and_proxy_names: list[ast.expr] = [ast.Name(f"@{name}", ctx=ast.Del()) for name in (key_class, proxy_class)]
        node.body.append(ast.Delete(targets=key_and_proxy_names))

        return self.generic_visit(node)


def _match_token(token: tokenize.TokenInfo, **kwargs: object) -> bool:
    """Check if a given token's attributes match the given kwargs."""

    return all(getattr(token, name) == val for name, val in kwargs.items())


def sliding_window(iterable: _tp.Iterable[_tp.T], n: int) -> _tp.Iterable[tuple[_tp.T, ...]]:
    """Collect data into overlapping fixed-length chunks or blocks.

    Copied from 3.12 itertools docs.

    Examples
    --------
    >>> ["".join(window) for window in sliding_window('ABCDEFG', 4)]
    ['ABCD', 'BCDE', 'CDEF', 'DEFG']
    """

    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


class DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments .py files which use "with defer_imports_until_use: ..."."""

    @classmethod
    def _check_source_for_defer_usage(cls, data: _tp.ReadableBuffer) -> tuple[str, bool]:
        """Get the encoding of the code and also check if it uses "with defer_imports_until_use"."""

        tok_NAME, tok_OP = tokenize.NAME, tokenize.OP

        token_stream = tokenize.tokenize(io.BytesIO(data).readline)
        encoding = next(token_stream).string
        uses_defer = any(
            _match_token(tok1, type=tok_NAME, string="with")
            and (
                (
                    _match_token(tok2, type=tok_NAME, string="defer_imports_until_use")
                    and _match_token(tok3, type=tok_OP, string=":")
                )
                or (
                    _match_token(tok2, type=tok_NAME, string="deferred")
                    and _match_token(tok3, type=tok_OP, string=".")
                    and _match_token(tok4, type=tok_NAME, string="defer_imports_until_use")
                )
            )
            for tok1, tok2, tok3, tok4 in sliding_window(token_stream, 4)
        )
        return encoding, uses_defer

    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: _tp.ReadableBuffer,
        path: _tp.Union[_tp.ReadableBuffer, _tp.StrPath],
        *,
        _optimize: int = -1,
    ) -> _tp.CodeType:
        # NOTE: InspectLoader is the virtual superclass of SourceFileLoader thanks to ABC registration, so typeshed
        #       reflects that. However, there's a slight mismatch in source_to_code signatures. Make a PR?

        # Defer to regular machinery if the module is empty or doesn't use defer_imports_until_use.
        if not data:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        encoding, uses_defer = self._check_source_for_defer_usage(data)
        if not uses_defer:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        instrumenter = DeferredInstrumenter(path, data, encoding)
        instrumented_tree = ast.fix_missing_locations(instrumenter.transform())
        return compile(instrumented_tree, path, "exec", dont_inherit=True, optimize=_optimize)

    def get_data(self, path: str) -> bytes:
        """Return the data from path as raw bytes.

        Notes
        -----
        If the path points to a bytecode file, check for a deferred-specific header. If the header is invalid, raising
        OSError will invalidate the bytecode.

        Source: https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        data = super().get_data(path)

        if not path.endswith(tuple(BYTECODE_SUFFIXES)):
            return data

        if not data.startswith(b"deferred"):
            msg = '"deferred" header missing from bytecode'
            raise OSError(msg)

        if not data.startswith(BYTECODE_HEADER):
            msg = '"deferred" header is outdated'
            raise OSError(msg)

        return data[len(BYTECODE_HEADER) :]

    def set_data(self, path: str, data: _tp.ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        Notes
        -----
        If the file is a bytecode one, prepend a deferred-specific header to it. That way, instrumented bytecode can be
        identified and invalidated it if necessary. Another option is to monkeypatch importlib.util.cache_from_source,
        as beartype and typeguard do, but that seems less safe.

        Source: https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)


DEFERRED_PATH_HOOK = FileFinder.path_hook((DeferredFileLoader, SOURCE_SUFFIXES))


# endregion


# region -------- Runtime hook


original_import = contextvars.ContextVar("original_import", default=builtins.__import__)
"""What builtins.__import__ currently points to."""

is_deferred = contextvars.ContextVar("is_deferred", default=False)
"""Whether imports should be deferred."""


class DeferredImportProxy:
    """Proxy for a deferred __import__ call."""

    def __init__(
        self,
        name: str,
        global_ns: dict[str, object],
        local_ns: dict[str, object],
        fromlist: _tp.Optional[tuple[str, ...]],
        level: int = 0,
    ) -> None:
        self.defer_proxy_name = name
        self.defer_proxy_global_ns = global_ns
        self.defer_proxy_local_ns = local_ns
        self.defer_proxy_fromlist: tuple[str, ...] = fromlist if (fromlist is not None) else ()
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

    def __getattr__(self, attr: str, /):
        sub_proxy = type(self)(*self.defer_proxy_import_args)

        if attr in self.defer_proxy_fromlist:
            sub_proxy.defer_proxy_fromlist = (attr,)
        else:
            sub_proxy.defer_proxy_sub = attr

        return sub_proxy


class DeferredImportKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_str", "defer_key_proxy", "is_recursing")

    def __new__(cls, key: str, proxy: DeferredImportProxy, /):
        return super().__new__(cls, key)

    def __init__(self, key: str, proxy: DeferredImportProxy, /):
        self.defer_key_str = str(key)
        self.defer_key_proxy = proxy
        self.is_recursing = False

    def __repr__(self) -> str:
        return f"<key for {self.defer_key_str!r} import>"

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if self.defer_key_str != value:
            return False

        if not self.is_recursing:
            self.is_recursing = True

            if not is_deferred.get():
                self._resolve()

        return True

    def __hash__(self) -> int:
        return hash(self.defer_key_str)

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.defer_key_proxy

        # Perform the original __import__ and pray.
        module = original_import.get()(*proxy.defer_proxy_import_args)

        # Transfer nested proxies over to the resolved module.
        for attr_key, attr_val in vars(proxy).items():
            if isinstance(attr_val, DeferredImportProxy) and not hasattr(module, attr_key):
                setattr(module, DeferredImportKey(attr_key, attr_val), attr_val)
                # Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = vars(module)

        # Replace the proxy with the resolved module or module attribute in the relevant namespace.

        # First, get the regular string key and the relevant namespace.
        key = self.defer_key_str
        namespace = proxy.defer_proxy_local_ns

        # Second, remove the deferred key to avoid it sticking around.
        # NOTE: This is necessary to prevent recursive resolution for proxies, since __eq__ will be triggered again.
        _is_def_tok = is_deferred.set(True)
        try:
            # TODO: Figure out why this works and del namespace[key] doesn't.
            namespace[key] = namespace.pop(key)
        finally:
            is_deferred.reset(_is_def_tok)

        # Finally, resolve any requested attribute access.
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def calc___package__(globals: dict[str, _tp.Any]) -> _tp.Optional[str]:
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    Slightly modified version of importlib._bootstrap._calc___package__.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")
    if package is not None:
        if spec is not None and package != spec.parent:
            # TODO: Keep the warnings up to date with CPython.
            category = DeprecationWarning if sys.version_info >= (3, 12) else ImportWarning
            warnings.warn(
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",
                category,
                stacklevel=3,
            )
        return package
    elif spec is not None:
        return spec.parent
    else:
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

    Slightly modified version of importlib._bootstrap._resolve_name.
    """

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


def deferred___import__(  # noqa: ANN202
    name: str,
    globals: dict[str, object],
    locals: dict[str, object],
    fromlist: _tp.Optional[tuple[str, ...]] = None,
    level: int = 0,
    /,
):
    """An limited replacement for __import__ that supports deferred imports by returning proxies."""

    # Resolve the names of relative imports.
    if level > 0:
        package = calc___package__(locals)
        name = resolve_name(name, package, level)  # pyright: ignore [reportArgumentType]
        level = 0

    # Handle submodule imports if relevant top-level imports already occurred in the call site's module.
    if not fromlist and ("." in name):
        name_parts = name.split(".")
        try:
            # TODO: Consider adding a condition that base_parent must be a ModuleType or a DeferredImportProxy, to
            #       avoid attaching proxies to a random thing that would've normally been clobbered by the import
            #       first?
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # Nest submodule proxies as needed.
            # TODO: Modifying a member of the passed-in locals isn't ideal. Still better than modifying the locals
            #       mapping directly, and avoiding *that* is a major reason for the hybrid instrumentation approach.
            #       Still, is there a better way to do this or maybe a better place for it?
            for bound, attr_name in enumerate(name_parts[1:], start=2):
                if attr_name not in vars(parent):
                    nested_proxy = DeferredImportProxy(".".join(name_parts[:bound]), globals, locals, (), level)
                    nested_proxy.defer_proxy_sub = attr_name
                    setattr(parent, attr_name, nested_proxy)
                    parent = nested_proxy
                else:
                    parent = getattr(parent, attr_name)

            return base_parent

    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# region -------- Public API


def install_defer_import_hook() -> None:
    """Insert deferred's path hook right before the default FileFinder one in sys.path_hooks.

    This can be called in a few places, e.g. __init__.py of a package, a .pth file in site-packages, etc.
    """

    if DEFERRED_PATH_HOOK in sys.path_hooks:
        return

    # TODO: Consider all options for finder cache invalidation. Some form of it is necessary because we went the
    #       sys.path_hooks route.
    #       1)  sys.path_importer_cache.clear() - Not enough. Everything breaks.
    #       2)  importlib.invalidate_caches() - Works, but might be overkill since it hits every meta path finder.
    #           Calls 3 among other things.
    #       3)  PathFinder.invalidate_caches() - Works, but it's heavy due to importing importlib.metadata.
    #       ?)  inlining - Copy the implementation of 3 sans importlib.metdata part? Might be incorrect.
    #
    # Goal: Avoid further increasing startup time on first runs, i.e. before any bytecode caching.
    for i, hook in enumerate(sys.path_hooks):
        if hook.__qualname__.startswith("FileFinder.path_hook"):
            sys.path_hooks.insert(i, DEFERRED_PATH_HOOK)
            PathFinder.invalidate_caches()
            return


def uninstall_defer_import_hook() -> None:
    """Remove deferred's path hook if it's in sys.path_hooks."""

    try:
        sys.path_hooks.remove(DEFERRED_PATH_HOOK)
    except ValueError:
        pass
    else:
        # TODO: See comment in install_defer_import_hook. Sync here.
        PathFinder.invalidate_caches()


@_tp.final
class DeferredContext:
    """The type for defer_imports_until_use."""

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = is_deferred.set(True)
        self._import_ctx_token = original_import.set(builtins.__import__)
        builtins.__import__ = deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        original_import.reset(self._import_ctx_token)
        is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = original_import.get()


defer_imports_until_use: _tp.Final[DeferredContext] = DeferredContext()
"""A context manager within which imports occur lazily.

Raises
------
SyntaxError
    If defer_imports_until_use is used improperly, e.g.:
        1. It is being used in a class or function scope.
        2. Its block contains a statement that isn't an import.
        3. Its block contains a wildcard import.

Notes
-----
As part of its implementation, this temporarily replaces builtins.__import__.
"""


# endregion
