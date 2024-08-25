from __future__ import annotations

import ast
import builtins
import contextvars
import importlib.machinery
import io
import sys
import tokenize

from ._utils import Final, HasLocationAttributes, ReadableBuffer, StrPath, calc_package, final, pairwise, resolve_name


# region -------- Compile-time hook


BYTECODE_HEADER = b"deferred0.0.1"
"""Custom header for deferred-instrumented bytecode files. Should be updated with every version release."""


class DeferredImportInstrumenter(ast.NodeTransformer):
    """AST transformer that "instruments" imports within "with defer_imports_until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.
    """

    def __init__(self, filename: str, data: ReadableBuffer, encoding: str) -> None:
        self.filename = filename
        self.data = data
        self.encoding = encoding
        self.scope_depth = 0

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef) -> ast.AST:
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

    def _get_node_context(self, node: HasLocationAttributes):
        """Get the location context for a node. That context will be used as an argument to SyntaxError."""

        assert isinstance(node, ast.AST)

        text = ast.get_source_segment(self._decode_source(), node, padded=True)
        context = (self.filename, node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):
            context += (node.end_lineno, node.end_col_offset + 1)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create an AST equivalent to the following statements (slightly simplified):

        if type(name) is DeferredImportProxy:
            temp_proxy = global_ns.pop('name')
            global_ns[DeferredImportKey('name', temp_proxy)] = temp_proxy
        """

        if "." in name:
            name = name.rpartition(".")[0]

        mini_tree = ast.parse(
            f"if type({name}) is DeferredImportProxy:\n"
            f"    temp_proxy = global_ns.pop('{name}')\n"
            f"    global_ns[DeferredImportKey('{name}', temp_proxy)] = temp_proxy"
        )
        if_node = mini_tree.body[0]
        assert isinstance(if_node, ast.If)

        # Adjust some of the names to be inaccessible via regular name resolution.
        for node in ast.walk(if_node):
            if isinstance(node, ast.Name) and node.id in {
                "temp_proxy",
                "global_ns",
                "DeferredImportProxy",
                "DeferredImportKey",
            }:
                node.id = f"@{node.id}"
        return if_node

    @staticmethod
    def _create_global_ns_var() -> ast.Assign:
        """Create an AST that's equivalent to "@global_ns = globals()".

        The created @global_ns variable will be used as a temporary reference to the result of globals() to avoid
        calling globals() repeatedly.
        """

        return ast.Assign(
            targets=[ast.Name("@global_ns", ctx=ast.Store())],
            value=ast.Call(ast.Name("globals", ctx=ast.Load()), args=[], keywords=[]),
        )

    @staticmethod
    def _create_temp_proxy_var() -> ast.Assign:
        """Create an AST that's equivalent to "@temp_proxy = None".

        The created @temp_proxy variable will be used as a temporary reference to the current proxy being "fixed".
        """

        return ast.Assign(targets=[ast.Name("@temp_proxy", ctx=ast.Store())], value=ast.Constant(None))

    def _substitute_import_keys(self, import_nodes: list[ast.stmt]) -> list[ast.stmt]:
        """Instrument the list of imports."""

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

        new_import_nodes[0:0] = (self._create_global_ns_var(), self._create_temp_proxy_var())

        # Delete temporary variables after all is said and done to avoid namespace pollution.
        temp_proxy_deletion = ast.Delete(targets=[ast.Name("@temp_proxy", ctx=ast.Del())])
        global_ns_deletion = ast.Delete(targets=[ast.Name("@global_ns", ctx=ast.Del())])
        new_import_nodes.extend((temp_proxy_deletion, global_ns_deletion))

        return new_import_nodes

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that "with defer_imports_until_use" blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If any of the following conditions are met, in order of priority:
                1. "defer_imports_until_use" is being used in a non-module scope.
                2. "defer_imports_until_use" block contains a statement that isn't an import.
                3. "defer_imports_until_use" block contains a wildcard import.
        """

        if not (
            len(node.items) == 1
            and isinstance(node.items[0].context_expr, ast.Name)
            and node.items[0].context_expr.id == "defer_imports_until_use"
        ):
            return self.generic_visit(node)

        if self.scope_depth != 0:
            msg = "with defer_imports_until_use only allowed at module level"
            raise SyntaxError(msg, self._get_node_context(node))

        node.body = self._substitute_import_keys(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports necessary to make defer_imports_until_use work properly. The import is placed after the
        module docstring and any "from __future__" imports.

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

        defer_aliases = [
            ast.alias(name="DeferredImportKey", asname="@DeferredImportKey"),
            ast.alias(name="DeferredImportProxy", asname="@DeferredImportProxy"),
        ]
        key_import = ast.ImportFrom(module="deferred._core", names=defer_aliases, level=0)
        node.body.insert(position, key_import)

        # Clean up the namespace.
        deferred_key_deletion = ast.Delete(targets=[ast.Name("@DeferredImportKey", ctx=ast.Del())])
        deferred_proxy_deletion = ast.Delete(targets=[ast.Name("@DeferredImportProxy", ctx=ast.Del())])
        node.body.extend((deferred_key_deletion, deferred_proxy_deletion))

        return self.generic_visit(node)


class DeferredImportFileLoader(importlib.machinery.SourceFileLoader):
    """A file loader that instruments modules that are using "with defer_imports_until_use: ..."."""

    @staticmethod
    def _check_for_defer_usage(data: ReadableBuffer) -> tuple[str, bool]:
        """Get the encoding of the code and also check if it uses "with defer_imports_until_use"."""

        token_stream = tokenize.tokenize(io.BytesIO(data).readline)
        encoding = next(token_stream).string
        uses_defer = any(
            first_tok.type == tokenize.NAME
            and first_tok.string == "with"
            and second_tok.type == tokenize.NAME
            and second_tok.string == "defer_imports_until_use"
            for first_tok, second_tok in pairwise(token_stream)
        )
        return encoding, uses_defer

    # InspectLoader is the virtual superclass of SourceFileLoader thanks to ABC registration, so typeshed reflects
    # that. However, it has a slightly different source_to_code signature.
    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: ReadableBuffer,
        path: ReadableBuffer | StrPath,
        *,
        _optimize: int = -1,
    ):
        encoding, uses_defer = self._check_for_defer_usage(data)
        if not uses_defer:
            return compile(data, path, "exec", dont_inherit=True, optimize=_optimize)

        transformer = DeferredImportInstrumenter(str(path), data, encoding)
        instrumented_tree = ast.fix_missing_locations(transformer.visit(ast.parse(data, path, "exec")))
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

        if not path.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            return data

        if not data.startswith(b"deferred"):
            msg = '"deferred" header missing from bytecode'
            raise OSError(msg)

        if not data.startswith(BYTECODE_HEADER):
            msg = '"deferred" header is outdated'
            raise OSError(msg)

        return data[len(BYTECODE_HEADER) :]

    def set_data(self, path: str, data: ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        Notes
        -----
        If the file is a bytecode one, prepend a deferred-specific header to it. That way, instrumented bytecode can be
        identified and invalidated it if necessary. Another option is to monkeypatch importlib.util.cache_from_source,
        as beartype and typeguard do, but that seems less safe.

        Source: https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            data = BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)


class DeferredImportFileFinder(importlib.machinery.FileFinder):
    """deferred's custom file-based finder.

    Currently a stub but may be utilized in the future.
    """


# endregion


# region -------- Runtime hook


_MISSING = object()

original_import = contextvars.ContextVar("current_import", default=builtins.__import__)
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
        fromlist: tuple[str, ...] = (),
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
    def defer_proxy_import_args(self):
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

    def __getattr__(self, attr: str):
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

    __slots__ = ("defer_key_str", "defer_key_proxy")

    defer_key_str: str
    defer_key_proxy: DeferredImportProxy

    def __new__(cls, key: str, proxy: DeferredImportProxy, /):
        self = super().__new__(cls, key)
        self.defer_key_str = str(key)
        self.defer_key_proxy = proxy
        return self

    def __repr__(self, /) -> str:
        return f"<key for {self.defer_key_str!r} import>"

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if self.defer_key_str != value:
            return False

        if not is_deferred.get():
            self._resolve()

        return True

    def __hash__(self, /) -> int:
        return hash(self.defer_key_str)

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.defer_key_proxy

        # Perform the original __import__ and pray.
        module = original_import.get()(*proxy.defer_proxy_import_args)

        # Transfer nested proxies over to the resolved item.
        for attr_name, attr_val in vars(proxy).items():
            if isinstance(attr_name, DeferredImportKey) and not hasattr(module, attr_name.defer_key_str):
                assert isinstance(attr_val, DeferredImportProxy)
                # NOTE: Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = vars(module)
                setattr(module, attr_name, attr_val)

        # ---- Replace the proxy on the relevant namespace with the resolved module or module attribute.

        # First, pick the relevant namespace.
        namespace = proxy.defer_proxy_local_ns

        # Second, remove the deferred key to avoid it sticking around.
        # NOTE: This is necessary to prevent recursive resolution for nested proxies.
        _is_def_tok = is_deferred.set(True)
        try:
            del namespace[self]
        finally:
            is_deferred.reset(_is_def_tok)

        # Finally, resolve any requested attribute access and replace using a regular string key.
        key = self.defer_key_str

        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub is not None:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def _deferred_import(
    name: str,
    globals: dict[str, object],
    locals: dict[str, object],
    fromlist: tuple[str, ...] | None = None,
    level: int = 0,
    /,
):
    """An implementation of __import__ supporting defer imports.

    The arguments are less flexible to indicate this should shouldn't ever be called as __import__.
    """

    # Resolve the names of relative imports.
    if level > 0:
        package = calc_package(globals)
        name = resolve_name(name, package, level)

    # FIXME: This check for cached modules is incomplete.
    if (module := sys.modules.get(name, _MISSING)) is not _MISSING:
        return module

    # Handle submodule imports if relevant top-level imports already happened.
    if not fromlist and ("." in name):
        base_name, *rest_name = name.split(".")
        try:
            parent = globals[base_name]
        except KeyError:
            pass
        else:
            # Nest the submodule proxies as needed.
            # XXX: Mucking with a member of the passed-in globals within __import__ isn't ideal. Is there a better way?
            #      Maybe via the instrumentation instead?
            for i, sub_name in enumerate(rest_name, start=1):
                if sub_name not in vars(parent):
                    nested_name = ".".join((base_name, *rest_name[:i]))
                    nested_proxy = DeferredImportProxy(nested_name, globals, locals, ("<<<",), level)
                    setattr(parent, DeferredImportKey(sub_name, nested_proxy), nested_proxy)
                    parent = nested_proxy

            return globals[base_name]

    fromlist = fromlist if (fromlist is not None) else ()
    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# region -------- Public API


class DeferredImportError(ImportError):
    """Exception raised when an import goes awry while using the deferred import machinery."""


def install_defer_import_hook() -> None:
    """Insert deferred's path hook right before the default FileFinder one.

    This can be called in a few places, e.g. __init__.py of a package, a .pth file in site-packages, etc.
    """

    for i, hook in enumerate(sys.path_hooks):
        if "FileFinder.path_hook" in hook.__qualname__:
            loader_details = (DeferredImportFileLoader, importlib.machinery.SOURCE_SUFFIXES)
            sys.path_hooks.insert(i, DeferredImportFileFinder.path_hook(loader_details))
            sys.path_importer_cache.clear()
            return


def uninstall_defer_import_hook() -> None:
    """Remove deferred's path hook if it's in sys.path_hooks."""

    for i, hook in enumerate(sys.path_hooks):
        if "DeferredImportFileFinder.path_hook" in hook.__qualname__:
            del sys.path_hooks[i]
            sys.path_importer_cache.clear()
            return


@final
class DeferContext:
    """The type for defer_imports_until_use. No public interface beyond that; this is only meant to be used via
    defer_imports_until_use.
    """

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = is_deferred.set(True)
        self._import_ctx_token = original_import.set(builtins.__import__)
        builtins.__import__ = _deferred_import

    def __exit__(self, *exc_info: object) -> None:
        original_import.reset(self._import_ctx_token)
        is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = original_import.get()


defer_imports_until_use: Final[DeferContext] = DeferContext()
"""A context manager within which imports occur lazily. Reentrant.

Raises
------
SyntaxError
    If defer_imports_until_use is used improperly, e.g. within a function or class.

Notes
-----
This temporarily replaces builtins.__import__ as part of its implementation.
"""


# endregion
