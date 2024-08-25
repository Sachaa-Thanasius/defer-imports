from __future__ import annotations

import ast
import builtins
import contextvars
import importlib.machinery
import io
import sys
import tokenize


# region -------- Expensive-import-avoidance hacks


TYPING = False

if TYPING:
    import os
    from collections.abc import Generator, Iterable, Sequence
    from typing import Final, Protocol, TypeVar, Union, final

    if sys.version_info >= (3, 12):
        from typing import Buffer as ReadableBuffer
    else:
        from typing_extensions import Buffer as ReadableBuffer

    if sys.version_info >= (3, 10):
        from itertools import pairwise
    else:

        def pairwise(iterable: Iterable[_T]) -> zip[tuple[_T, _T]]: ...

    _T = TypeVar("_T")

    StrPath = Union[str, os.PathLike[str]]

    class HasLocationAttributes(Protocol):
        lineno: int
        col_offset: int
        end_lineno: int | None
        end_col_offset: int | None

    def reverse_enumerate(sequence: Sequence[_T], start: int = -1) -> Generator[tuple[int, _T]]:
        """Yield index-value pairs from the given sequence, but in reverse. This generator defaults to starting at the end
        of the sequence unless the start index is given.
        """
        ...  # noqa: PIE790
else:
    from itertools import tee

    StrPath = str
    ReadableBuffer = "bytes | bytearray | memoryview"

    def final(f):
        """Placeholder for typing.final. Copied from typing with minimal changes."""

        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            # Skip the attribute silently if it is not writable.
            # AttributeError: if the object has __slots__ or a read-only property
            # TypeError: if it's a builtin class
            pass
        return f

    class Final:
        """Placeholder for typing.Final."""

    class HasLocationAttributes:
        """Placeholder for protocol representing an ast node's location attributes."""

    def pairwise(iterable) -> zip[tuple[object, object]]:
        """Pairwise recipe copied from itertools."""
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG

        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def reverse_enumerate(sequence: list, start: int = -1):
        """Yield index-value pairs from the given sequence, but in reverse. This generator defaults to starting at the end
        of the sequence unless the start index is given.
        """

        if start == -1:
            start = len(sequence) - 1

        for i in range(start, -1, -1):
            yield i, sequence[i]


def decode_source(source_bytes: ReadableBuffer) -> str:
    """Slightly modified copy of importlib.util.decode_source."""

    source_bytes_readline = io.BytesIO(source_bytes).readline
    encoding = tokenize.detect_encoding(source_bytes_readline)
    newline_decoder = io.IncrementalNewlineDecoder(None, True)
    return newline_decoder.decode(source_bytes.decode(encoding[0]))  # pyright: ignore


# endregion


# region -------- Compile-time hook


class DeferredImportFixer(ast.NodeTransformer):
    """AST transformer that "instruments" imports within "with defer_imports_until_use: ..." blocks so that their
    results are assigned to custom keys in the global namespace.
    """

    def __init__(self, filename: str, source: str) -> None:
        self.filename = filename
        self.source = source
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

    def _get_node_context(self, node: HasLocationAttributes):
        """Get the location context for a node. That context will be used as an argument to SyntaxError."""

        assert isinstance(node, ast.AST)

        text = ast.get_source_segment(self.source, node, padded=True)
        context = (self.filename, node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):
            context += (node.end_lineno, node.end_col_offset + 1)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create a tuple of AST equivalent to the following statements (slightly simplified):

            temp_proxy = global_ns.pop(name)
            global_ns(DeferredImportKey(name, temp_proxy)) = temp_proxy

        Notes
        -----
        temp_proxy is a temporary reference to the current proxy being "fixed". It is deleted at the end of the "with"
        block to avoid polluting the namespace.
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
    def _create_global_ns_assignment() -> ast.Assign:
        """Create an AST that's equivalent to "@global_ns = globals()".

        Notes
        -----
        @global_ns will serve as a temporary reference to the module globals to avoid calling globals() repeatedly.
        It is deleted at the end of the "with" block to avoid polluting the namespace.
        """

        return ast.Assign(
            targets=[ast.Name("@global_ns", ctx=ast.Store())],
            value=ast.Call(ast.Name("globals", ctx=ast.Load()), args=[], keywords=[]),
        )

    def _substitute_import_keys(self, import_nodes: list[ast.stmt]) -> list[ast.stmt]:
        """Instrument the list of imports."""

        new_import_nodes: list[ast.stmt] = list(import_nodes)

        for i, sub_node in reverse_enumerate(import_nodes):
            if not isinstance(sub_node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports_until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_node_context(sub_node))  # noqa: TRY004

            for alias in sub_node.names:
                if alias.name == "*":
                    msg = "import * not allowed in with defer_imports_until_use blocks"
                    raise SyntaxError(msg, self._get_node_context(sub_node))

                new_import_nodes.insert(i + 1, self._create_import_name_replacement(alias.asname or alias.name))

        # A temporary reference to globals().
        new_import_nodes.insert(0, self._create_global_ns_assignment())
        # A temporary variable for proxies to reside in.
        new_import_nodes.insert(
            1, ast.Assign(targets=[ast.Name("@temp_proxy", ctx=ast.Store())], value=ast.Constant(None))
        )

        # Delete temporary variables after all is said and done.
        temp_proxy_deletion = ast.Delete(targets=[ast.Name("@temp_proxy", ctx=ast.Del())])
        new_import_nodes.append(temp_proxy_deletion)
        global_ns_deletion = ast.Delete(targets=[ast.Name("@global_ns", ctx=ast.Del())])
        new_import_nodes.append(global_ns_deletion)

        return new_import_nodes

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that "with defer_imports_until_use" blocks are being used correctly and if so, hook all imports
        within.

        Raises
        ------
        SyntaxError:
            If any of the following conditions are met, in order of priority:

                1. "defer_imports_until_use" is being used in a non-module scope.
                2. "defer_imports_until_use" contains a statement that isn't an import.
                3. "defer_imports_until_use" contains a wildcard import.
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

        return self.generic_visit(node)


class DeferredImportFixLoader(importlib.machinery.SourceFileLoader):
    """A file loader that instruments modules that are using "with defer_imports_until_use: ..."."""

    @staticmethod
    def _check_for_defer_usage(data: ReadableBuffer) -> bool:
        """Check that "with defer_imports_until_use" is actually used in the code."""

        return any(
            first_tok.type == tokenize.NAME
            and first_tok.string == "with"
            and second_tok.type == tokenize.NAME
            and second_tok.string == "defer_imports_until_use"
            for first_tok, second_tok in pairwise(tokenize.tokenize(io.BytesIO(data).readline))
        )

    # InspectLoader is the virtual superclass of SourceFileLoader thanks to ABC registration, so typeshed reflects
    # that. However, in reality, it has a slightly different source_to_code signature.
    def source_to_code(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        data: ReadableBuffer,
        path: ReadableBuffer | StrPath,
        *,
        _optimize: int = -1,
    ):
        if not self._check_for_defer_usage(data):
            return compile(data, path, "exec", dont_inherit=True, optimize=_optimize)

        # Get the source to make better error messages.
        source = decode_source(data)
        orig_tree = ast.parse(data, path, "exec")

        # NOTE: Locations could be fixed where needed to avoid full tree traversal, but that ruins the error messages.
        new_tree = ast.fix_missing_locations(DeferredImportFixer(str(path), source).visit(orig_tree))
        return compile(new_tree, path, "exec", dont_inherit=True, optimize=_optimize)


# endregion


# region -------- Runtime hook


_MISSING = object()

original_import = contextvars.ContextVar("original_import", default=builtins.__import__)
"""A contextvar for tracking what builtins.__import__ currently is."""


class DeferredImportProxy:
    """Proxy for a deferred __import__ call."""

    __slots__ = (
        "defer_proxy_name",
        "defer_proxy_global_ns",
        "defer_proxy_local_ns",
        "defer_proxy_fromlist",
        "defer_proxy_level",
        "defer_proxy_requested_attrs",
    )

    def __init__(self, name: str, global_ns, local_ns, fromlist: tuple[str, ...] = (), level: int = 0) -> None:
        self.defer_proxy_name = name
        self.defer_proxy_global_ns: dict[str, object] = global_ns
        self.defer_proxy_local_ns: dict[str, object] | None = local_ns
        self.defer_proxy_fromlist = fromlist
        self.defer_proxy_level = level

        self.defer_proxy_requested_attrs: list[str] = []

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

    @property
    def defer_proxy_namespace(self):
        """The "closest" existing namespace between the passed-in locals and globals, with a preference for locals."""

        return self.defer_proxy_local_ns if (self.defer_proxy_local_ns is not None) else self.defer_proxy_global_ns

    def __repr__(self) -> str:
        return f"<proxy for {self.defer_proxy_name!r} import>"

    def __getattr__(self, attr: str):
        sub_proxy = type(self)(*self.defer_proxy_import_args)

        if attr in self.defer_proxy_fromlist:
            sub_proxy.defer_proxy_fromlist = (attr,)
        else:
            sub_proxy.defer_proxy_requested_attrs = [*self.defer_proxy_requested_attrs, attr]

        return sub_proxy


class DeferredImportKey:
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("key", "proxy")

    def __init__(self, key: str, proxy: DeferredImportProxy) -> None:
        self.key = key
        self.proxy = proxy

    def __repr__(self) -> str:
        return f"<key for {self.key!r} import>"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if self.key != value:
            return False

        self._resolve()
        return True

    def __hash__(self) -> int:
        return hash(self.key)

    def _resolve(self):
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.proxy
        namespace = proxy.defer_proxy_namespace
        module = original_import.get()(*proxy.defer_proxy_import_args)

        # Replace the proxy on the namespace with the resolved module or module attribute in the namespace.
        del namespace[self]

        if proxy.defer_proxy_fromlist:
            namespace[self.key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_requested_attrs:
            namespace[self.key] = getattr(module, proxy.defer_proxy_requested_attrs[0])
        else:
            namespace[self.key] = module


def _deferred_import(name: str, globals=None, locals=None, fromlist=None, level: int = 0):
    """The implementation of __import__ supporting defer imports."""

    if level == 0 and (module := sys.modules.get(name, _MISSING)) is not _MISSING:
        return module

    globals = globals if (globals is not None) else {}
    fromlist = fromlist if (fromlist is not None) else ()
    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# region -------- Public API


def install_defer_import_hook() -> None:
    """Insert deferred's path hook right before the default FileFinder one.

    This can be called in a few places, e.g. __init__.py of your package, a .pth file in site-packages, etc.
    """

    for i, hook in enumerate(sys.path_hooks):
        if "FileFinder.path_hook" in hook.__qualname__:
            supported_file_loader = (DeferredImportFixLoader, importlib.machinery.SOURCE_SUFFIXES)
            new_hook = importlib.machinery.FileFinder.path_hook(supported_file_loader)
            new_hook._deferred_import_hook = True  # pyright: ignore # Runtime attribute assignment.
            sys.path_hooks.insert(i, new_hook)
            sys.path_importer_cache.clear()
            return


def uninstall_defer_import_hook() -> None:
    """Remove deferred's path hook if it's in sys.path_hooks."""

    for i, hook in enumerate(sys.path_hooks):
        if "FileFinder.path_hook" in hook.__qualname__ and getattr(hook, "_deferred_import_hook", False):
            del sys.path_hooks[i]
            sys.path_importer_cache.clear()
            return


@final
class DeferContext:
    """The type for defer_imports_until_use. No public interface beyond that; this is only meant to be used via
    defer_imports_until_use.
    """

    __slots__ = ("_token",)

    def __enter__(self):
        self._token = original_import.set(builtins.__import__)
        builtins.__import__ = _deferred_import

    def __exit__(self, *exc_info: object) -> None:
        original_import.reset(self._token)
        builtins.__import__ = original_import.get()


defer_imports_until_use: Final = DeferContext()
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
