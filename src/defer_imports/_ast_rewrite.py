# Some of the code and comments below is adapted from
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.

from __future__ import annotations

import builtins
import contextvars
import io
import os  # noqa: F401 # Used in type alias.
import sys
import threading
import types
from importlib.machinery import BYTECODE_SUFFIXES, ModuleSpec, PathFinder, SourceFileLoader

from . import __version__, _lazy_load


# PYUPDATE: py3.14 - importlib.abc might be cheap enough to eagerly import.
with _lazy_load.until_module_use:
    import ast
    import importlib.abc  # noqa: F401 # Used in type alias.
    import tokenize
    import typing as t
    import warnings


__all__ = ("install_import_hook",)


# ============================================================================
# region -------- Compatibility shims --------
# ============================================================================


TYPE_CHECKING = False


if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeGuard
elif sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    TypeAlias: t.TypeAlias = "t.TypeAlias"
    TypeGuard: t.TypeAlias = "t.TypeGuard"
else:  # pragma: <3.10 cover

    class TypeAlias:
        """Placeholder for typing.TypeAlias."""

    class TypeGuard:
        """Placeholder for typing.TypeGuard."""

        __class_getitem__ = classmethod(types.GenericAlias)


if TYPE_CHECKING:
    from typing_extensions import Self
elif sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    Self: t.TypeAlias = "t.Self"
else:  # pragma: <3.11 cover

    class Self:
        """Placeholder for typing.Self."""


if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    from collections.abc import Buffer as ReadableBuffer
elif TYPE_CHECKING:
    from typing_extensions import Buffer as ReadableBuffer
else:  # pragma: <3.12 cover
    ReadableBuffer: TypeAlias = "t.Union[bytes, bytearray, memoryview]"


# This allows _increment_location to be typed better.
if TYPE_CHECKING:
    _AST = t.TypeVar("_AST", bound=ast.AST)
else:
    _AST: TypeAlias = "ast.AST"


# endregion


# ============================================================================
# region -------- Vendored helpers --------
#
# These are adapted from importlib._bootstrap and importlib.util to avoid
# depending on private APIs and allow changes.
#
# PYUPDATE: Ensure these are consistent with upstream, aside from our
# customizations.
# ============================================================================


# Adapted from importlib._bootstrap.
def _resolve_name(name: str, package: str, level: int) -> str:  # pragma: no cover
    """Resolve a relative module name to an absolute one."""

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


# Adapted from importlib._bootstrap.
def _sanity_check(name: str, package: t.Optional[str], level: int) -> None:  # pragma: no cover
    """Verify arguments are "sane"."""

    if not isinstance(name, str):  # pyright: ignore [reportUnnecessaryIsInstance] # Account for user error.
        msg = f"module name must be str, not {name.__class__}"
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


# Adapted from importlib._bootstrap.
def _calc___package__(globals: t.MutableMapping[str, t.Any]) -> t.Optional[str]:  # pragma: no cover
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None to represent that its proper value is unknown.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")

    if package is not None:
        if spec is not None and package != spec.parent:
            if sys.version_info >= (3, 12):
                category = DeprecationWarning
            else:
                category = ImportWarning

            warnings.warn(f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})", category, stacklevel=3)

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


# Adapted from importlib.util.
# Changes:
# - Don't import tokenize inline, since that could cause issues while our module spec finder is on sys.meta_path.
# - Rearrange slightly to allow cleaner usage of type-checker directives.
#
#     - An aside about the function's typing:
#
#       decode_source is typed in typeshed as taking Buffer, but memoryview fits that while lacking a decode method.
#       A custom DecodableBuffer protocol would technically be the best fit here, but that causes other issues:
#
#         1. Creating a protocol subclass would require either a) using typing at import time, specifically
#            typing.Protocol, or b) putting that protocol in a new module and lazily importing it. Both are currently
#            annoying.
#         2. The type issues would rise to usage sites, e.g. in DeferredInstrumenter where a ReadableBuffer is passed
#            in. The problem is viral, as is the "solution" of replacing ReadableBuffer with DecodableBuffer everywhere.
#
#       Therefore, we will keep the original annotation and allow this to raise if a Buffer without a decode method is
#       passed in.
def _decode_source(source_bytes: ReadableBuffer) -> str:  # pragma: no cover
    """Decode bytes representing source code and return the string.

    Universal newline support is used in the decoding.
    """

    newline_decoder = io.IncrementalNewlineDecoder(None, translate=True)
    encoding = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)[0]
    source: str = source_bytes.decode(encoding)  # pyright: ignore [reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    return newline_decoder.decode(source)  # pyright: ignore [reportUnknownArgumentType]


# endregion


# ============================================================================
# region -------- Compile-time magic --------
#
# The AST transformer, import hook machinery, and import hook API.
# ============================================================================


_ModulePath: TypeAlias = "t.Union[str, os.PathLike[str], ReadableBuffer]"
_SourceData: TypeAlias = "t.Union[ReadableBuffer, str]"
_LoaderInit: TypeAlias = "t.Callable[[str, str], importlib.abc.Loader]"


_BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Differs for every version."""

_INTERNALS_NAMES: t.Final = ("_DeferredImportKey", "_DeferredImportProxy", "_actual_until_use")
_INTERNALS_ASNAMES: t.Final[tuple[str, str, str]] = tuple(f"@{name}" for name in _INTERNALS_NAMES)  # pyright: ignore [reportAssignmentType]
_KEY_CLS_NAME, _PROXY_CLS_NAME, _ACTUAL_CTX_NAME = _INTERNALS_ASNAMES
_TEMP_PROXY_NAME: t.Final = "@temp_proxy"
_LOCAL_NS_NAME: t.Final = "@local_ns"


def _increment_location(node: _AST, line_n: int = 1, col_n: int = 0) -> _AST:
    """Increment the location values of each node in a tree, starting at `node`.

    This is useful to "move code" to a different location in a file.

    Parameters
    ----------
    node: _AST
        The tree to alter the location of.
    line_n: int, default=1
        How much to increment the line number and end line number of each node. Defaults to 1.
    col_n: int, default=0
        How much to increment the column offset and end column offset of each node. Defaults to 0.

    Returns
    -------
    _AST
        The tree starting from the original given node.

    Notes
    -----
    The implementation is partially adapted from `ast.increment_lineno`.
    """

    for child in ast.walk(node):
        # TypeIgnore is a special case where lineno is not an attribute
        # but rather a field of the node itself.
        if isinstance(child, ast.TypeIgnore):
            child.lineno = getattr(child, "lineno", 0) + line_n
            continue

        if "lineno" in child._attributes:
            child.lineno = getattr(child, "lineno", 0) + line_n  # pyright: ignore [reportAttributeAccessIssue]

        if "end_lineno" in child._attributes and (end_lineno := getattr(child, "end_lineno", 0)) is not None:
            child.end_lineno = end_lineno + line_n  # pyright: ignore [reportAttributeAccessIssue]

        if "col_offset" in child._attributes:
            child.col_offset = getattr(child, "col_offset", 0) + col_n  # pyright: ignore [reportAttributeAccessIssue]

        if (
            "end_col_offset" in child._attributes
            and (end_col_offset := getattr(child, "end_col_offset", 0)) is not None
        ):
            child.end_col_offset = end_col_offset + col_n  # pyright: ignore [reportAttributeAccessIssue]

    return node


def _is_until_use_node(node: ast.With) -> bool:
    """Check if the node matches ``with defer_imports.until_use``."""

    return len(node.items) == 1 and (
        isinstance(node.items[0].context_expr, ast.Attribute)
        and isinstance(node.items[0].context_expr.value, ast.Name)
        and node.items[0].context_expr.value.id == "defer_imports"
        and node.items[0].context_expr.attr == "until_use"
    )


# TODO: Make the variable names that this uses more "hygienic".
# Maybe add a prefix similar to pytest with "@py_", e.g. "@di_" instead of just "@"?
# TODO: Finish making this accurately create/update node location information.
class _DeferredInstrumenter:
    """AST transformer that instruments imports within ``with defer_imports.until_use: ...`` blocks.

    The results of those imports will be assigned to custom keys in the local namespace.

    Notes
    -----
    The transformer doesn't subclass `ast.NodeTransformer` but instead vendors its logic to avoid the upfront import
    cost of `ast`.
    """

    # PYUPDATE: Ensure visit and generic_visit are consistent with upstream, aside from our customizations.

    def __init__(self, data: _SourceData, filepath: _ModulePath = "<unknown>", *, module_level: bool = False) -> None:
        self.data: _SourceData = data
        self.filepath: _ModulePath = filepath
        self.module_level: bool = module_level

        self.escape_hatch_depth: int = 0
        self.did_any_instrumentation: bool = False

    def visit(self, node: ast.AST) -> t.Any:
        """Visit a node."""

        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        """Avoid visiting non-global scopes."""

        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = visit_ClassDef = _visit_scope

    def _visit_eager_import_block(self, node: ast.AST) -> ast.AST:
        """Track if the visitor is within a ``try-except`` block or a ``with`` statement."""

        self.escape_hatch_depth += 1
        node = self.generic_visit(node)
        self.escape_hatch_depth -= 1
        return node

    visit_Try = _visit_eager_import_block

    if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
        visit_TryStar = _visit_eager_import_block

    def _get_node_location(self, node: ast.stmt):  # noqa: ANN202 # Version-dependent and too verbose.
        """Get the location context for a node.

        Notes
        -----
        The return value is meant to serve as the details argument for SyntaxError [1]_.

        References
        ----------
        .. [1] https://docs.python.org/3.14/library/exceptions.html#SyntaxError
        """

        source = self.data if isinstance(self.data, str) else _decode_source(self.data)
        text = ast.get_source_segment(source, node, padded=True)
        context = (str(self.filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = (node.end_col_offset + 1) if (node.end_col_offset is not None) else None
            context += (node.end_lineno, end_col_offset)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str) -> ast.If:
        """Create an AST for changing the name of a variable in locals if the variable is a defer_imports proxy.

        The resulting node is roughly equivalent to the following::

            if {name}.__class__ is _DeferredImportProxy:
                temp_proxy = local_ns[_DeferredImportKey("{name}", temp_proxy)] = local_ns.pop("{name}")
        """

        if "." in name:
            name = name.partition(".")[0]

        return ast.If(
            test=ast.Compare(
                left=ast.Attribute(value=ast.Name(name, ctx=ast.Load()), attr="__class__", ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Name(_PROXY_CLS_NAME, ctx=ast.Load())],
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(_TEMP_PROXY_NAME, ctx=ast.Store()),
                        ast.Subscript(
                            value=ast.Name(_LOCAL_NS_NAME, ctx=ast.Load()),
                            slice=ast.Call(
                                func=ast.Name(_KEY_CLS_NAME, ctx=ast.Load()),
                                args=[ast.Constant(name), ast.Name(_TEMP_PROXY_NAME, ctx=ast.Load())],
                                keywords=[],
                            ),
                            ctx=ast.Store(),
                        ),
                    ],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(_LOCAL_NS_NAME, ctx=ast.Load()), attr="pop", ctx=ast.Load()),
                        args=[ast.Constant(name)],
                        keywords=[],
                    ),
                )
            ],
            orelse=[],
        )

    def _validate_until_use_body(self, nodes: list[ast.stmt]) -> TypeGuard[list[t.Union[ast.Import, ast.ImportFrom]]]:
        """Validate that the statements within a `defer_imports.until_use` block are instrumentable."""

        for node in nodes:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports.until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_node_location(node))  # noqa: TRY004 # Syntax error displays better.

            if any(alias.name == "*" for alias in node.names):
                msg = "import * not allowed in with defer_imports.until_use blocks"
                raise SyntaxError(msg, self._get_node_location(node))

        return True

    def _substitute_import_keys(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> list[ast.stmt]:
        """Instrument a *non-empty* list of imports.

        Raises
        ------
        SyntaxError
            If any of the given nodes are not imports or are wildcard imports.
        """

        self.did_any_instrumentation = True

        new_nodes: list[ast.stmt] = []

        # Start with some helper variables.
        # A reference to locals() to avoid calling locals() repeatedly.
        new_nodes.append(
            ast.Assign(
                targets=[ast.Name(_LOCAL_NS_NAME, ctx=ast.Store())],
                value=ast.Call(func=ast.Name("locals", ctx=ast.Load()), args=[], keywords=[]),
            )
        )
        # A reference to the current proxy being "fixed".
        new_nodes.append(ast.Assign([ast.Name(_TEMP_PROXY_NAME, ctx=ast.Store())], ast.Constant(None)))

        # Add the imports + namespace adjustments.
        for node in import_nodes:
            new_nodes.append(node)
            new_nodes.extend(self._create_import_name_replacement(alias.asname or alias.name) for alias in node.names)

        # Clean up the helper variables.
        new_nodes.append(ast.Delete([ast.Name(name, ctx=ast.Del()) for name in (_TEMP_PROXY_NAME, _LOCAL_NS_NAME)]))

        return new_nodes

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that ``with defer_imports.until_use: ...`` blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If any of the following conditions are met, in order of priority:
                1. "defer_imports.until_use" is being used in a class or function scope.
                2. "defer_imports.until_use" block contains a statement that isn't an import.
                3. "defer_imports.until_use" block contains a wildcard import.
        """

        if not _is_until_use_node(node):
            return self._visit_eager_import_block(node)

        # Replace the dummy context manager with the one that will actually replace __import__.
        new_ctx_expr = ast.copy_location(ast.Name(_ACTUAL_CTX_NAME, ctx=ast.Load()), node.items[0].context_expr)
        node.items[0].context_expr = new_ctx_expr

        # TODO: Assert can be optimized out. Find a different way of making validating + typeguard work.
        assert self._validate_until_use_body(node.body)
        node.body = self._substitute_import_keys(node.body)
        return ast.fix_missing_locations(node)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports and cleanup necessary to make `defer_imports.until_use` work properly.

        If `defer_imports.until_use` isn't actually used, do nothing.

        Imports will be inserted after the module docstring and after `__future__` imports.
        """

        node = self.generic_visit(node)  # pyright: ignore [reportAssignmentType] # We know it'll return a module.

        if not self.did_any_instrumentation:
            return node

        # First, get past the module docstring and __future__ imports.
        expect_docstring = True

        position = 0
        for position, sub in enumerate(node.body):  # noqa: B007
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

        # Then, add necessary defer_imports import.
        lineno = position + 1
        cumm_col_offset = len(f"from {__spec__.name} import ")
        internals_aliases: list[ast.alias] = []

        for name, asname in zip(_INTERNALS_NAMES, _INTERNALS_ASNAMES):
            new_alias = ast.alias(name, asname, lineno=lineno, col_offset=cumm_col_offset, end_lineno=lineno)
            cumm_col_offset += len(name) + 4 + len(asname)  # Account for " as " with 4.
            new_alias.end_col_offset = cumm_col_offset
            cumm_col_offset += 2  # Account for ", ".
            internals_aliases.append(new_alias)

        cumm_col_offset -= 2  # Discard nonexistent final ", ".

        internals_import = ast.ImportFrom(
            __spec__.name,
            internals_aliases,
            0,
            lineno=lineno,
            col_offset=0,
            end_lineno=lineno,
            end_col_offset=cumm_col_offset,
        )
        node.body.insert(position, internals_import)

        for child in node.body[position + 1 :]:
            ast.increment_lineno(child)

        # Finally, clean up the namespace.
        lineno = node.body[-1].lineno
        cumm_col_offset = 4  # Account for "del ".
        del_targets: list[ast.expr] = []

        for asname in _INTERNALS_ASNAMES:
            del_target = ast.Name(asname, ctx=ast.Del(), lineno=lineno, col_offset=cumm_col_offset, end_lineno=lineno)
            cumm_col_offset += len(asname)
            del_target.end_col_offset = cumm_col_offset
            cumm_col_offset += 2  # Account for ", ".
            del_targets.append(del_target)

        cumm_col_offset -= 2  # Discard nonexistent final ", ".

        node.body.append(
            ast.Delete(del_targets, lineno=lineno, col_offset=0, end_lineno=lineno, end_col_offset=cumm_col_offset)
        )

        return node

    @staticmethod
    def _is_defer_imports_import(node: t.Union[ast.Import, ast.ImportFrom]) -> bool:
        """Check if the given import node imports from `defer_imports`."""

        # TODO: Is this actually needed?

        if isinstance(node, ast.Import):
            return any(alias.name.partition(".")[0] == "defer_imports" for alias in node.names)
        else:
            return node.module is not None and node.module.partition(".")[0] == "defer_imports"

    def _is_import_to_instrument(self, value: object) -> TypeGuard[t.Union[ast.Import, ast.ImportFrom]]:
        return (
            # Only for import nodes without wildcards.
            isinstance(value, (ast.Import, ast.ImportFrom))
            and value.names[0].name != "*"
            # Only outside of escape hatch blocks.
            and (self.escape_hatch_depth == 0 and not self._is_defer_imports_import(value))
        )

    def _wrap_import_stmts(self, nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> ast.With:
        """Wrap consecutive import nodes within a list of statements using a `defer_imports.until_use` block and
        instrument them.

        The first node must be an import node.
        """

        lineno = nodes[0].lineno
        end_col_offset = 5 + len(_ACTUAL_CTX_NAME)  # Account for "with ".

        instrumented_nodes = [_increment_location(node, 0, 4) for node in self._substitute_import_keys(nodes)]

        return ast.With(
            [
                ast.withitem(
                    ast.Name(
                        _ACTUAL_CTX_NAME,
                        ast.Load(),
                        lineno=lineno,
                        col_offset=5,  # Account for "with ".
                        end_lineno=lineno,
                        end_col_offset=end_col_offset,
                    )
                )
            ],
            instrumented_nodes,
            lineno=lineno,
            col_offset=0,
            end_lineno=lineno,
            end_col_offset=end_col_offset + 1,  # Account for ":".
        )

    def generic_visit(self, node: ast.AST) -> ast.AST:  # noqa: PLR0912
        """Called if no explicit visitor function exists for a node.

        This differs from the regular generic_visit by conditionally intercepting global sequences of import statements
        to wrap them in ``with defer_imports.until_use`` blocks.
        """

        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values: list[t.Any] = []
                start_idx = 0

                for value in old_value:  # pyright: ignore [reportUnknownVariableType]
                    if isinstance(value, ast.AST):
                        if self.module_level:
                            if self._is_import_to_instrument(value):
                                start_idx += 1
                            elif start_idx > 0:
                                new_values[-start_idx:] = [self._wrap_import_stmts(new_values[-start_idx:])]
                                start_idx = 0

                        value = self.visit(value)  # noqa: PLW2901

                        if value is None:
                            continue
                        if not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue

                    new_values.append(value)

                if self.module_level and (start_idx > 0):
                    new_values[-start_idx:] = [self._wrap_import_stmts(new_values[-start_idx:])]

                old_value[:] = new_values

            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)

        return node


class _DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments ``.py`` files which use ``with defer_imports.until_use: ...``."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.defer_module_level: bool = False

    def get_data(self, path: str) -> bytes:
        """Return the data from `path` as raw bytes.

        If `path` points to a bytecode file, validate that it has a `defer_imports`-specific header.

        Raises
        ------
        OSError
            If the path points to a bytecode file with an invalid `defer_imports`-specific header.

        Notes
        -----
        `importlib._boostrap_external.SourceLoader.get_code` expects this method to potentially raise `OSError` [1]_.

        Another option is to monkeypatch `importlib.util.cache_from_source`, as beartype [2]_ and typeguard do, but that
        seems excessive for this use case.

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

    def set_data(self, path: str, data: ReadableBuffer, *, _mode: int = 0o666) -> None:
        """Write bytes data to a file.

        Notes
        -----
        If the file is a bytecode one, prepend a `defer_imports`-specific header to it. That way, instrumented bytecode
        can be identified and invalidated later if necessary [1]_.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = _BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)

    # NOTE: We're purposefully not supporting the case where data is an AST object.
    # NOTE: The signatures of SourceFileLoader.source_to_code at runtime and in typeshed aren't consistent regarding
    # the _optimize parameter.
    def source_to_code(self, data: _SourceData, path: _ModulePath, *, _optimize: int = -1) -> types.CodeType:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Compile `data` into a code object, but not before potentially instrumenting it.

        Parameters
        ----------
        data: _SourceData
            Anything that `compile()` can handle, i.e. string or bytes.
        path: _ModulePath
            Where the data was retrieved from (when applicable).

        Returns
        -------
        types.CodeType
            The compiled code object.
        """

        if data:
            orig_tree = ast.parse(data, path, "exec")
            if self.defer_module_level or any(
                (isinstance(node, ast.With) and _is_until_use_node(node)) for node in ast.walk(orig_tree)
            ):
                instrumenter = _DeferredInstrumenter(data, path, module_level=self.defer_module_level)
                new_tree = instrumenter.visit(orig_tree)
                return super().source_to_code(new_tree, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

        return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the module (while setting and maintaining relevant state)."""

        if (spec := module.__spec__) is not None and spec.loader_state is not None:
            self.defer_module_level = spec.loader_state["defer_module_level"]

        return super().exec_module(module)


class _DeferredPathFinder(PathFinder):
    _original_pathfinder: t.ClassVar[type[PathFinder]]

    @staticmethod
    def _get_loader_class(config: _DeferConfig) -> _LoaderInit:
        if config.loader_class is None:
            return _DeferredFileLoader
        else:
            return config.loader_class

    @staticmethod
    def _get_instrumentation_level(fullname: str, config: _DeferConfig) -> bool:
        """Determine whether only imports in until_use blocks should be instrumented or all imports should be.

        Returns `True` for the former, `False` for the latter.
        """

        # NOTE: This could be written as one boolean expression, but currently, splitting it out makes the hierarchy
        # of configuration options a bit clearer.

        if config.apply_all:
            return True

        if not config.module_names:
            return False

        if fullname in config.module_names:
            return True

        return config.recursive and any(fullname.startswith(f"{mod}.") for mod in config.module_names)

    @classmethod
    def find_spec(
        cls,
        fullname: str,
        path: t.Optional[t.Sequence[str]] = None,
        target: t.Optional[types.ModuleType] = None,
    ) -> t.Optional[ModuleSpec]:
        """Try to find a spec for `fullname` on sys.path or `path`.

        If a spec is found, its loader is overriden and some deferral-related state is attached.

        Notes
        -----
        This utilizes `ModuleSpec.loader_state` to pass the deferral configuration to the loader. loader_state is
        under-documented [1]_, but it is meant to be used for this kind of thing [2]_.

        References
        ----------
        .. [1] https://github.com/python/cpython/issues/89527
        .. [2] https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader_state
        """

        spec = super().find_spec(fullname, path, target)

        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            # NOTE: We're locking in defer_imports configuration for this module between finding it and loading it.
            # However, it's possible to delay getting the configuration until module execution. Not sure what's best.
            config = _current_defer_config.get(None)

            if config is not None:
                defer_module_level = cls._get_instrumentation_level(fullname, config)
                loader_class = cls._get_loader_class(config)
            else:
                defer_module_level = False
                loader_class = _DeferredFileLoader

            spec.loader = loader_class(spec.loader.name, spec.loader.path)
            spec.loader_state = {"defer_module_level": defer_module_level}

        return spec


_current_defer_config: contextvars.ContextVar[_DeferConfig] = contextvars.ContextVar("_current_defer_config")
"""The current configuration for defer_imports's instrumentation."""


class _DeferConfig:
    """Configuration container whose contents are used to determine how a module should be instrumented."""

    def __init__(
        self,
        apply_all: bool,
        module_names: t.Sequence[str],
        recursive: bool,
        loader_class: t.Optional[_LoaderInit],
    ) -> None:
        self.apply_all = apply_all
        self.module_names = module_names
        self.recursive = recursive
        self.loader_class = loader_class

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"apply_all={self.apply_all!r}, module_names={self.module_names!r}, "
            f"recursive={self.recursive!r}, loader_class={self.loader_class!r}"
            ")"
        )


class _ImportHookContext:
    """The context manager returned by install_import_hook(). Can reset defer_imports's configuration to its previous
    state and uninstall defer_import's import meta path finder.
    """

    def __init__(self, _config_ctx_tok: contextvars.Token[_DeferConfig], _uninstall_after: bool) -> None:
        self._tok = _config_ctx_tok
        self._uninstall_after = _uninstall_after

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_dont_care: object) -> None:
        self.reset()
        if self._uninstall_after:
            self.uninstall()

    def reset(self) -> None:
        """Attempt to reset the import hook configuration. If already reset, does nothing."""

        try:
            tok = self._tok
        except AttributeError:
            pass
        else:
            _current_defer_config.reset(tok)
            del self._tok

    def uninstall(self) -> None:
        """Attempt to replace the custom meta path finder in sys.meta_path with the original.

        If the custom finder is already gone, does nothing.
        """

        try:
            finder_index = sys.meta_path.index(_DeferredPathFinder)
        except ValueError:
            pass
        else:
            sys.meta_path[finder_index] = _DeferredPathFinder._original_pathfinder


def install_import_hook(
    *,
    uninstall_after: bool = False,
    apply_all: bool = False,
    module_names: t.Sequence[str] = (),
    recursive: bool = False,
    loader_class: t.Optional[_LoaderInit] = None,
) -> _ImportHookContext:
    """Install defer_imports's import hook if it isn't already installed, and optionally configure it. Must be called
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
        Has higher priority than `module_names`. More suitable for use in applications.
    module_names: t.Sequence[str], optional
        A set of modules to apply module-level import deferral to. Has lower priority than apply_all. More suitable for
        use in libraries.
    recursive: bool, default=False
        Whether module-level import deferral should apply recursively the submodules of the given module_names. Has the
        same priority as `module_names`. If no module names are given, this has no effect.
    loader_class: _LoaderInit, optional
        An import loader class for `defer_imports` to use instead of the default machinery. If supplied, it is assumed to
        have an initialization signature matching ``(fullname: str, path: str) -> None``.

    Returns
    -------
    ImportHookContext
        A object that can be used to reset the import hook's configuration to its previous state or uninstall it, either
        automatically by using it as a context manager or manually using its rest() and uninstall methods.
    """

    if isinstance(module_names, str):
        msg = "module_names should be a sequence of strings, not a string."
        raise TypeError(msg)

    if _DeferredPathFinder not in sys.meta_path:
        try:
            path_finder_index = sys.meta_path.index(PathFinder)
        except ValueError:
            sys.meta_path.insert(0, _DeferredPathFinder)
        else:
            # We know the finder class at this index is PathFinder.
            _DeferredPathFinder._original_pathfinder = sys.meta_path[path_finder_index]  # pyright: ignore [reportAttributeAccessIssue]
            sys.meta_path[path_finder_index] = _DeferredPathFinder

    config = _DeferConfig(apply_all, module_names, recursive, loader_class)
    config_ctx_tok = _current_defer_config.set(config)
    return _ImportHookContext(config_ctx_tok, uninstall_after)


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
        global_ns: t.MutableMapping[str, t.Any],
        local_ns: t.MutableMapping[str, t.Any],
        fromlist: t.Sequence[str],
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

    def __getattr__(self, name: str, /) -> Self:
        if name in self.defer_proxy_fromlist:
            from_proxy = self.__class__(*self.defer_proxy_import_args)
            from_proxy.defer_proxy_fromlist = (name,)
            return from_proxy

        elif ("." in self.defer_proxy_name) and (name == self.defer_proxy_name.rpartition(".")[2]):
            submodule_proxy = self.__class__(*self.defer_proxy_import_args)
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

    defer_key_proxy: _DeferredImportProxy
    is_resolving: bool
    lock: threading.RLock

    def __new__(cls, key: str, proxy: _DeferredImportProxy, /) -> Self:
        self = super().__new__(cls, key)
        self.defer_key_proxy = proxy
        self.is_resolving = False
        self.lock = threading.RLock()
        return self

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
                # NOTE: This doesn't use setattr() because pypy normalizes the attr key type to `str`.
                module_vars[_DeferredImportKey(attr_key, attr_val)] = attr_val

                # Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = module_vars

        # 3. Replace the proxy with the resolved module or module attribute in the relevant namespace.
        # 3.1. Get the regular string key and the relevant namespace.
        key = str(self)
        namespace = proxy.defer_proxy_local_ns

        # 3.2. Replace the deferred version of the key to avoid it sticking around.
        # This will trigger __eq__ again, so we temporarily set is_deferred to prevent recursion.
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
    globals: t.MutableMapping[str, t.Any],
    locals: t.MutableMapping[str, t.Any],
    fromlist: t.Optional[t.Sequence[str]] = None,
    level: int = 0,
) -> t.Any:
    """An limited replacement for `__import__` that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()

    package = _calc___package__(globals) if (level != 0) else None
    _sanity_check(name, package, level)

    if level > 0:
        # _sanity_check ensures this.
        assert package is not None
        name = _resolve_name(name, package, level)
        level = 0

    # TODO: Return modules cached in sys.modules here?

    # Handle submodule imports if relevant top-level imports already occurred in the call site's module.
    if not fromlist and ("." in name):
        name_parts = name.split(".")
        try:
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # NOTE: We assume that if base_parent is a module or a proxy, then it shouldn't be getting
            #       clobbered. Not sure if this is right, but it feels like the safest move.
            if isinstance(base_parent, (types.ModuleType, _DeferredImportProxy)):
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


class _DeferredContext:
    """A context manager within which imports occur lazily. Not reentrant. Use via `defer_imports.until_use`.

    If defer_imports isn't set up properly, e.g. `install_import_hook` is not called first elsewhere, this should be a
    no-op equivalent to `contextlib.nullcontext`.

    Raises
    ------
    SyntaxError
        If `defer_imports.until_use` is used improperly, e.g.:
            1. It is being used in a class or function scope.
            2. It contains a statement that isn't an import.
            3. It contains a wildcard import.

    Notes
    -----
    As part of its implementation, this temporarily replaces builtins.__import__.
    """

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = _is_deferred.set(True)
        self._import_ctx_token = _original_import.set(builtins.__import__)
        builtins.__import__ = _deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        _original_import.reset(self._import_ctx_token)
        _is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = _original_import.get()


_actual_until_use: t.Final[_DeferredContext] = _DeferredContext()


# endregion
