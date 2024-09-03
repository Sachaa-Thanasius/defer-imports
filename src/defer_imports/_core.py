# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""The implementation details for defer_imports's magic."""

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
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, PathFinder, SourceFileLoader

from . import _typing as _tp


__version__ = "0.0.1"


# region -------- Compile-time hook


SourceData: _tp.TypeAlias = "_tp.Union[_tp.ReadableBuffer, str, ast.Module, ast.Expression, ast.Interactive]"

BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Should be updated with every version release."""


class DeferredInstrumenter(ast.NodeTransformer):
    """AST transformer that instruments imports within "with defer_imports.until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.
    """

    def __init__(self, filepath: _tp.Union[_tp.StrPath, _tp.ReadableBuffer], data: SourceData, encoding: str) -> None:
        self.filepath = filepath
        self.data = data
        self.encoding = encoding
        self.scope_depth = 0

    def instrument(self, mode: str = "exec") -> _tp.Any:
        """Transform the tree created from the given data and filepath."""

        if isinstance(self.data, ast.AST):  # noqa: SIM108 # Readability
            to_visit = self.data
        else:
            to_visit = ast.parse(self.data, self.filepath, mode)

        return ast.fix_missing_locations(self.visit(to_visit))

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Track Python scope changes. Used to determine if defer_imports.until_use usage is valid."""

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
        """Get the source code corresponding to the given data."""

        if isinstance(self.data, ast.AST):
            # NOTE: An attempt is made here, but the node location information likely won't match up.
            return ast.unparse(self.data)
        elif isinstance(self.data, str):  # noqa: RET505 # Readability
            return self.data
        else:
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
                raise SyntaxError(msg, self._get_node_context(node))  # noqa: TRY004

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
        return len(node.items) == 1 and (
            # Allow "with defer_imports.until_use".
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
            return self.generic_visit(node)

        if self.scope_depth != 0:
            msg = "with defer_imports.until_use only allowed at module level"
            raise SyntaxError(msg, self._get_node_context(node))

        node.body = self._substitute_import_keys(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports necessary to make defer_imports.until_use work properly. The import is placed after the
        module docstring and after __future__ imports.

        Notes
        -----
        This assumes the module is not empty and "with defer_imports.until_use" is used somewhere in it.
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

        defer_class_names = ("DeferredImportKey", "DeferredImportProxy")

        # Import key and proxy classes.
        defer_aliases = [ast.alias(name=name, asname=f"@{name}") for name in defer_class_names]
        key_and_proxy_import = ast.ImportFrom(module="defer_imports._core", names=defer_aliases, level=0)
        node.body.insert(position, key_and_proxy_import)

        # Clean up the namespace.
        key_and_proxy_names: list[ast.expr] = [ast.Name(f"@{name}", ctx=ast.Del()) for name in defer_class_names]
        node.body.append(ast.Delete(targets=key_and_proxy_names))

        return self.generic_visit(node)


def match_token(token: tokenize.TokenInfo, **kwargs: object) -> bool:
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
        match_token(tok1, type=tok_NAME, string="with")
        and match_token(tok2, type=tok_NAME, string="defer_imports")
        and match_token(tok3, type=tok_OP, string=".")
        and match_token(tok4, type=tok_NAME, string="until_use")
        for tok1, tok2, tok3, tok4 in sliding_window(token_stream, 4)
    )

    return encoding, uses_defer


def check_ast_for_defer_usage(data: ast.AST) -> tuple[str, bool]:
    """Check if the given AST uses "with defer_imports.until_use". Also assume "utf-8" is the the encoding."""

    uses_defer = any(
        isinstance(node, ast.With) and DeferredInstrumenter.check_With_for_defer_usage(node) for node in ast.walk(data)
    )
    encoding = "utf-8"
    return encoding, uses_defer


class DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments .py files which use "with defer_imports.until_use: ..."."""

    @staticmethod
    def check_for_defer_usage(data: SourceData) -> tuple[str, bool]:
        return check_ast_for_defer_usage(data) if isinstance(data, ast.AST) else check_source_for_defer_usage(data)

    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: SourceData,
        path: _tp.Union[_tp.StrPath, _tp.ReadableBuffer],
        *,
        _optimize: int = -1,
    ) -> _tp.CodeType:
        # NOTE: InspectLoader is the virtual superclass of SourceFileLoader thanks to ABC registration, so typeshed
        #       reflects that. However, there's some mismatch in source_to_code signatures. Can it be fixed with a PR?

        if not data:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        encoding, uses_defer = self.check_for_defer_usage(data)

        if not uses_defer:
            return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # See note above.

        tree = DeferredInstrumenter(path, data, encoding).instrument()
        return super().source_to_code(tree, path, _optimize=_optimize)  # pyright: ignore # See note above.

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

    def __getattr__(self, name: str, /):
        sub_proxy = type(self)(*self.defer_proxy_import_args)

        if name in self.defer_proxy_fromlist:
            sub_proxy.defer_proxy_fromlist = (name,)
        elif name == self.defer_proxy_name.rpartition(".")[2]:
            sub_proxy.defer_proxy_sub = name
        else:
            msg = f"module {self.defer_proxy_name!r} has no attribute {name!r}"
            raise AttributeError(msg)

        return sub_proxy


class DeferredImportKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_str", "defer_key_proxy", "is_resolving", "lock")

    def __new__(cls, key: str, proxy: DeferredImportProxy, /):
        return super().__new__(cls, key)

    def __init__(self, key: str, proxy: DeferredImportProxy, /) -> None:
        self.defer_key_str = str(key)
        self.defer_key_proxy = proxy

        self.is_resolving = False
        self.lock = original_import.get()("threading").RLock()

    def __repr__(self) -> str:
        return f"<key for {self.defer_key_str!r} import>"

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

        # 1. Let the regular string key and the relevant namespace.
        key = self.defer_key_str
        namespace = proxy.defer_proxy_local_ns

        # 2. Replace the deferred version of the key to avoid it sticking around.
        #    This is_deferred usage is necessary to prevent recursive resolution, since __eq__ will be triggered again.
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


def calc___package__(globals: _tp.MutableMapping[str, _tp.Any]) -> _tp.Optional[str]:
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    Slightly modified version of importlib._bootstrap._calc___package__.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")

    # TODO: Keep the warnings in sync with supported CPython versions.
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
    globals: _tp.MutableMapping[str, object],
    locals: _tp.MutableMapping[str, object],
    fromlist: _tp.Optional[_tp.Sequence[str]] = None,
    level: int = 0,
):
    """An limited replacement for __import__ that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()

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
            #       avoid attaching proxies to a random thing that would've normally been clobbered by the import?
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # Nest submodule proxies as needed.
            # TODO: Is there a better way to do this or maybe a better place for it? Modifying a member of the
            #       passed-in locals isn't ideal.
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


# region -------- Public API


def install_defer_import_hook() -> None:
    """Insert defer_imports's path hook right before the default FileFinder one in sys.path_hooks.

    This can be called in a few places, e.g. __init__.py of a package, a .pth file in site packages, etc.
    """

    if DEFERRED_PATH_HOOK in sys.path_hooks:
        return

    # NOTE: PathFinder.invalidate_caches() is expensive because it imports importlib.metadata, but we have to just bear
    #       that for now, unfortunately. Price of being a good citizen, I suppose.
    for i, hook in enumerate(sys.path_hooks):
        if hook.__qualname__.startswith("FileFinder.path_hook"):
            sys.path_hooks.insert(i, DEFERRED_PATH_HOOK)
            PathFinder.invalidate_caches()
            return


def uninstall_defer_import_hook() -> None:
    """Remove defer_imports's path hook if it's in sys.path_hooks."""

    try:
        sys.path_hooks.remove(DEFERRED_PATH_HOOK)
    except ValueError:
        pass
    else:
        # NOTE: Whatever invalidation mechanism install_defer_import_hook() uses should be used here as well.
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
