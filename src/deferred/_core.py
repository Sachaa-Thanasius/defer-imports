from __future__ import annotations

import ast
import builtins
import contextvars
import importlib.machinery
import io
import sys
import tokenize

from ._utils import Final, HasLocationAttributes, ReadableBuffer, StrPath, final, pairwise


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

        if type(name) is DeferredProxy:
            temp_proxy = global_ns.pop(name)
            global_ns(DeferredImportKey(name, temp_proxy)) = temp_proxy
        """

        if "." in name:
            name = name.rpartition(".")[0]
        return ast.If(
            test=ast.Compare(
                left=ast.Call(
                    func=ast.Name("type", ctx=ast.Load()), args=[ast.Name(name, ctx=ast.Load())], keywords=[]
                ),
                ops=[ast.Is()],
                comparators=[ast.Name("@DeferredImportProxy", ctx=ast.Load())],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name("@temp_proxy", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name("@global_ns", ctx=ast.Load()), attr="pop", ctx=ast.Load()),
                        args=[ast.Constant(name)],
                        keywords=[],
                    ),
                ),
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name("@global_ns", ctx=ast.Load()),
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
        new_import_nodes.append(temp_proxy_deletion)
        global_ns_deletion = ast.Delete(targets=[ast.Name("@global_ns", ctx=ast.Del())])
        new_import_nodes.append(global_ns_deletion)

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

        deferred_key_deletion = ast.Delete(targets=[ast.Name("@DeferredImportKey", ctx=ast.Del())])
        node.body.append(deferred_key_deletion)
        deferred_proxy_deletion = ast.Delete(targets=[ast.Name("@DeferredImportProxy", ctx=ast.Del())])
        node.body.append(deferred_proxy_deletion)

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

        If the path points to a bytecode file, check for a deferred-specific header. If the header is invalid, raising
        OSError will invalidate the bytecode.

        Source: https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        data = super().get_data(path)

        if not path.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            return data

        if not data.startswith(b"deferred"):
            msg = "No deferred header"
            raise OSError(msg)

        if not data.startswith(BYTECODE_HEADER):
            msg = "deferred header mismatch"
            raise OSError(msg)

        return data[len(BYTECODE_HEADER) :]

    def set_data(self, path: str, data: ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        If the file is a bytecode one, prepend a deferred-specific header to it. That way, we can identify our
        instrumented bytecode and invalidate it if necessary. An alternative way to go about this is to monkeypatch
        importlib._bootstrap_external.cache_from_source, as beartype and typeguard do, but that seems more annoying.

        Source: https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            data = BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)


# endregion


# region -------- Runtime hook


_MISSING = object()

original_import = contextvars.ContextVar("current_import", default=builtins.__import__)
"""A contextvar for tracking what builtins.__import__ currently is."""

is_deferred = contextvars.ContextVar("is_deferred", default=False)
"""A contextvar for determining if executing code is currently within defer_imports_until_use."""


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
            stmt = f"from {self.defer_proxy_name} import {', '.join(self.defer_proxy_fromlist)}"
        elif self.defer_proxy_sub:
            stmt = f"import {self.defer_proxy_name} as ..."
        else:
            stmt = f"import {self.defer_proxy_name}"

        return f"<proxy for {stmt!r}>"

    def __getattr__(self, attr: str):
        sub_proxy = type(self)(*self.defer_proxy_import_args)
        if attr in self.defer_proxy_fromlist:
            sub_proxy.defer_proxy_fromlist = (attr,)
        else:
            sub_proxy.defer_proxy_sub = attr
        return sub_proxy


class DeferredImportKey:
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_key", "defer_key_proxy")

    def __init__(self, key: str, proxy: DeferredImportProxy) -> None:
        self.defer_key_key = key
        self.defer_key_proxy = proxy

    def __repr__(self) -> str:
        return f"<key for {self.defer_key_key!r} import>"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if self.defer_key_key != value:
            return False

        if not is_deferred.get():
            self._resolve()

        return True

    def __hash__(self) -> int:
        return hash(self.defer_key_key)

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        key = self.defer_key_key
        proxy = self.defer_key_proxy
        namespace = proxy.defer_proxy_local_ns
        module = original_import.get()(*proxy.defer_proxy_import_args)

        # Replace the proxy on the namespace with the resolved module or module attribute in the namespace.
        del namespace[self]

        # if any((type(attr) is DeferredImportKey) for attr in vars(proxy)):
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub is not None:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def _deferred_import(
    name: str,
    globals: dict[str, object] = None,
    locals: dict[str, object] | None = None,
    fromlist: tuple[str, ...] | None = None,
    level: int = 0,
):
    """The implementation of __import__ supporting defer imports."""

    # A few sanity checks in case someone uses __import__ within defer_imports_until_use.
    if locals is None:
        msg = "deferred import machinery only allowed at module level"
        raise DeferredImportError(msg)

    if fromlist == ("*",):
        msg = "import * not allowed within deferred import machinery"
        raise DeferredImportError(msg)

    # Return cached modules for absolute imports. Relative imports are harder.
    if level == 0 and (module := sys.modules.get(name, _MISSING)) is not _MISSING:
        return module

    fromlist = fromlist if (fromlist is not None) else ()

    base_name, *rest_name = name.split(".")
    if not fromlist and len(rest_name) and (base_name in globals):
        proxy = DeferredImportProxy(name, globals, locals, ("__name__",), level)
        setattr(globals[name], DeferredImportKey(rest_name[0], proxy), proxy)

    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# region -------- Public API


class DeferredImportError(ImportError):
    """Exception raised when an import goes awry while using the deferred import machinery."""


def install_defer_import_hook() -> None:
    """Insert deferred's path hook right before the default FileFinder one.

    This can be called in a few places, e.g. __init__.py of your package, a .pth file in site-packages, etc.
    """

    for i, hook in enumerate(sys.path_hooks):
        if "FileFinder.path_hook" in hook.__qualname__:
            supported_file_loader = (DeferredImportFileLoader, importlib.machinery.SOURCE_SUFFIXES)
            new_hook = importlib.machinery.FileFinder.path_hook(supported_file_loader)
            new_hook.is_deferred_import_hook = True  # pyright: ignore # Runtime attribute assignment.
            sys.path_hooks.insert(i, new_hook)
            sys.path_importer_cache.clear()
            return


def uninstall_defer_import_hook() -> None:
    """Remove deferred's path hook if it's in sys.path_hooks."""

    for i, hook in enumerate(sys.path_hooks):
        if "FileFinder.path_hook" in hook.__qualname__ and getattr(hook, "is_deferred_import_hook", False):
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
