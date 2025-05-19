# Some of the code and comments below is adapted from
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap.py
# and https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap_external.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.

from __future__ import annotations

import ast
import builtins
import collections
import contextvars
import io
import sys
import types
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, SourceFileLoader

from . import __version__, lazy_load as _lazy_load


with _lazy_load.until_module_use:
    import os
    import threading
    import tokenize
    import typing as t
    import warnings


__all__ = ("import_hook",)


# ============================================================================
# region -------- Compatibility shims --------
# ============================================================================


TYPE_CHECKING = False


if TYPE_CHECKING:
    from typing_extensions import TypeAlias
elif sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    TypeAlias: t.TypeAlias = "t.TypeAlias"
else:  # pragma: <3.10 cover

    class TypeAlias:
        """Placeholder for typing.TypeAlias."""


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


if TYPE_CHECKING:
    _final = t.final
else:

    def final(f: object) -> object:  # pragma: no cover (tested in stdlib)
        """Decorator to indicate final methods and final classes."""
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            # Skip the attribute silently if it is not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass
        return f

    _final = final
    del final


if sys.version_info >= (3, 10):
    _SyntaxErrorContext: TypeAlias = (
        "tuple[t.Optional[str], t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int], t.Optional[int]]"
    )
else:
    _SyntaxErrorContext: TypeAlias = "tuple[t.Optional[str], t.Optional[int], t.Optional[int], t.Optional[str]]"


# compile()'s internals, and thus wrappers of it (e.g. ast.parse()), dropped support in 3.12 for non-bytes buffers as
# the filename argument (see https://github.com/python/cpython/issues/98393).
if sys.version_info >= (3, 12):
    _ModulePath: TypeAlias = "t.Union[str, os.PathLike[str], bytes, os.PathLike[bytes]]"
else:
    _ModulePath: TypeAlias = "t.Union[str, os.PathLike[str], ReadableBuffer, os.PathLike[bytes]]"


# endregion


_SourceData: TypeAlias = "t.Union[ReadableBuffer, str]"


# ============================================================================
# region -------- Vendored helpers --------
#
# These are adapted from standard library modules to avoid depending on
# private APIs and/or allow changes.
#
# PYUPDATE: Ensure these are consistent with upstream, aside from our
# customizations.
# ============================================================================


# Adapted from importlib._bootstrap.
def _resolve_name(name: str, package: str, level: int) -> str:  # pragma: no cover (tested in stdlib)
    """Resolve a relative module name to an absolute one."""

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


# Adapted from importlib._bootstrap.
def _calc___package__(globals: t.MutableMapping[str, t.Any]) -> t.Optional[str]:  # pragma: no cover (tested in stdlib)
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.
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


# Adapted from importlib._bootstrap_external.
# Changes:
# - Don't import tokenize inline.
# - Rearrange slightly to allow cleaner usage of type-checker directives.
#
# NOTE: The parameter type should be narrower (here and in typeshed), but for now, we just let this raise if
# source_bytes.decode() doesn't exist. That shouldn't ever happen with our specific source retrieval pipeline.
def _decode_source(source_bytes: ReadableBuffer) -> str:  # pragma: no cover (tested in stdlib)
    """Decode bytes representing source code and return the string.

    Universal newline support is used in the decoding.
    """

    newline_decoder = io.IncrementalNewlineDecoder(None, translate=True)
    encoding = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)[0]
    source: str = source_bytes.decode(encoding)  # pyright: ignore [reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    return newline_decoder.decode(source)  # pyright: ignore [reportUnknownArgumentType]


# Adapted from ast.
# Changes:
# - Inline the helper functions.
# - Use io.StringIO with newline=None instead of regex to parse lines ending with newlines; it's a bit faster and
#   uses a module that we were importing anyway.
#
# NOTE: Technically, the type of `node` would be more accurately represented with something like
# ast.AST & LocationAttrsProtocol, but Python doesn't have intersections yet and, for us, import-time typing usage bad.
def get_source_segment(source: str, node: t.Union[ast.expr, ast.stmt], *, padded: bool = False) -> t.Optional[str]:
    """Get the source code segment of `source` that generated `node`.

    Parameters
    ----------
    source: str
        The source code.
    node: ast.expr | ast.stmt
        An AST created from `source`, with location information.
    padded: bool, default=False
        Whether to pad, with spaces, the first line of a multi-line statement to match its original position.

    Returns
    -------
    str | None
        The found source segment, or None if some location information (`lineno`, `end_lineno`, `col_offset`, or
        `end_col_offset`) is missing.
    """

    try:
        if (node.end_lineno is None) or (node.end_col_offset is None):
            return None

        col_offset = node.col_offset
        end_col_offset = node.end_col_offset

        # Convert from 1-indexed to 0-indexed.
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
    except AttributeError:
        return None

    # Split a string into lines while ignoring form feed and other chars.
    # This mimics how the Python parser splits source code.
    with io.StringIO(source, newline=None) as source_buffer:
        # NOTE: We could use itertools.islice() here to avoid materializing a potentially much larger list of lines.
        lines = list(source_buffer)[lineno : end_lineno + 1]

    if lineno == end_lineno:
        return lines[0].encode()[col_offset:end_col_offset].decode()

    if padded:
        # Replace all chars, except '\f\t', with spaces before the node's start on its starting line.
        padding = "".join((c if (c in "\f\t") else " ") for c in lines[lineno].encode()[:col_offset].decode())
    else:
        padding = ""

    lines[lineno] = padding + lines[lineno].encode()[col_offset:].decode()
    lines[end_lineno] = lines[end_lineno].encode()[:end_col_offset].decode()
    return "".join(lines)


# endregion


# ============================================================================
# region -------- AST transformation --------
# ============================================================================


# NOTE: We make our generated variables more hygienic by prefixing their names with "_@di_".
# There are a few reasons for this specific prefix:
#
# 1. The "@" isn't a valid character for identifiers, so there won't be conflicts with "regular" user code.
#    pytest does something similar.
# 2. The "di" namespaces the symbols somewhat in case third parties augment the code further with similar codegen.
#    pytest does something similiar.
# 3. The starting "_" will prevent the symbols from being picked up by code that programmatically accesses a namespace
#    during module execution but avoids symbols starting with an underscore.
#     - An example of a common pattern in the standard library that meets this criteria:
#       __all__ = [name for name in globals() if name[:1] != "_"]  # noqa: ERA001
#    TODO: Do we actually want to do this one? Surely it's bound to backfire in converse use cases.
_HYGIENE_PREFIX = "_@di_"

_ACTUAL_CTX_NAME = "_actual_until_use"
_ACTUAL_CTX_ASNAME = f"{_HYGIENE_PREFIX}_actual_until_use"
_TEMP_ASNAMES = f"{_HYGIENE_PREFIX}_temp_asnames"

#: The names of location attributes of AST nodes.
_AST_LOC_ATTRS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def _is_until_use_node(node: ast.With) -> bool:
    """Check if the node matches ``with defer_imports.until_use: ...``."""

    return len(node.items) == 1 and (
        isinstance(node.items[0].context_expr, ast.Attribute)
        and isinstance(node.items[0].context_expr.value, ast.Name)
        and node.items[0].context_expr.value.id == "defer_imports"
        and node.items[0].context_expr.attr == "until_use"
    )


class _ImportsInstrumenter(ast.NodeTransformer):
    """AST transformer that instruments imports within ``with defer_imports.until_use: ...`` blocks.

    The results of those imports will be assigned to custom keys in the local namespace.
    """

    # NOTE: ast.fix_missing_locations and ast.copy_location make location bookkeeping easier but slow compilation by
    # ~30%, so we avoid them.

    # PYUPDATE: Ensure generic_visit is consistent with upstream, aside from our customizations.
    # PYUPDATE: py3.14 - Take advantage of better defaults for node parameters, e.g. ast.Load() for ctx,
    # late-initialized empty lists for parameters that take lists, etc.

    def __init__(self, source: _SourceData, filepath: _ModulePath = "<unknown>", *, whole_module: bool = False) -> None:
        self.source: _SourceData = source
        self.filepath: _ModulePath = filepath
        self.rewrite_whole_module: bool = whole_module

        self.escape_hatch_depth: int = 0
        self.did_any_instrumentation: bool = False

    def _substitute_import_keys(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> list[ast.stmt]:
        """Instrument a *non-empty* list of imports."""

        self.did_any_instrumentation = True
        loc = {attr: getattr(import_nodes[0], attr) for attr in _AST_LOC_ATTRS}

        new_nodes: list[ast.stmt] = []

        # Create a temporary variable to hold the "asnames" for each import, i.e. what variable(s) the result(s) will be
        # saved into.
        for node in import_nodes:
            loc["lineno"], loc["end_lineno"] = node.lineno, node.end_lineno
            if isinstance(node, ast.Import):
                for alias in node.names:
                    asnames_name = ast.Name(_TEMP_ASNAMES, ctx=ast.Store(), **loc)
                    temp_asnames = ast.Assign([asnames_name], value=ast.Constant(alias.asname, **loc), **loc)
                    new_nodes.append(temp_asnames)
                    new_nodes.append(ast.Import(names=[alias], **loc))
            else:
                asnames_name = ast.Name(_TEMP_ASNAMES, ctx=ast.Store(), **loc)
                asnames_vals: list[ast.expr] = [ast.Constant(alias.asname, **loc) for alias in node.names]
                temp_asnames = ast.Assign([asnames_name], value=ast.Tuple(asnames_vals, ctx=ast.Load(), **loc), **loc)
                new_nodes.append(temp_asnames)
                new_nodes.append(node)

        # Clean up the temporary helper.
        del_stmt = ast.Delete(targets=[ast.Name(_TEMP_ASNAMES, ctx=ast.Del(), **loc)], **loc)
        new_nodes.append(del_stmt)

        return new_nodes

    def _wrap_import_stmts(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> ast.With:
        """Wrap a list of import nodes with a `defer_imports.until_use` block and instrument them."""

        loc = {attr: getattr(import_nodes[0], attr) for attr in _AST_LOC_ATTRS}
        with_items = [ast.withitem(context_expr=ast.Name(_ACTUAL_CTX_ASNAME, ctx=ast.Load(), **loc))]
        return ast.With(items=with_items, body=self._substitute_import_keys(import_nodes), **loc)

    def _is_import_to_wrap(self, node: ast.AST) -> bool:
        """Determine whether a node is a global import that should be instrumented."""

        return (
            self.escape_hatch_depth == 0
            # Must be an import node.
            and (
                isinstance(node, ast.Import)
                or
                # No wildcard or future "from" imports.
                (isinstance(node, ast.ImportFrom) and node.names[0].name != "*" and node.module != "__future__")
            )
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
                        if self.rewrite_whole_module:
                            if self._is_import_to_wrap(value):
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

                if self.rewrite_whole_module and start_idx > 0:
                    new_values[-start_idx:] = [self._wrap_import_stmts(new_values[-start_idx:])]

                old_value[:] = new_values

            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)

        return node

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

    def _get_error_context(self, node: ast.stmt) -> _SyntaxErrorContext:
        """Get a node's location context in a form compatible with `SyntaxError`'s constructor [1].

        References
        ----------
        .. [1] https://docs.python.org/3.14/library/exceptions.html#SyntaxError
        """

        source = self.source if isinstance(self.source, str) else _decode_source(self.source)
        text = get_source_segment(source, node, padded=True)
        filepath = self.filepath if isinstance(self.filepath, (str, bytes, os.PathLike)) else bytes(self.filepath)
        context = (os.fsdecode(filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = (node.end_col_offset + 1) if (node.end_col_offset is not None) else None
            context += (node.end_lineno, end_col_offset)
        return context

    def _validate_until_use_body(self, nodes: list[ast.stmt]) -> list[t.Union[ast.Import, ast.ImportFrom]]:
        """Validate that the statements within a `defer_imports.until_use` block are instrumentable.

        Raises
        ------
        SyntaxError
            If any of the given nodes are not imports or are wildcard imports.
        """

        for node in nodes:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports.until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_error_context(node))  # noqa: TRY004 # Syntax error displays better.

            if node.names[0].name == "*":
                msg = "import * not allowed in with defer_imports.until_use blocks"
                raise SyntaxError(msg, self._get_error_context(node))

        # Warning: Don't mutate the list from outside the function to invalidate our type guard.
        return nodes  # pyright: ignore [reportReturnType]

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that ``with defer_imports.until_use: ...`` blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If a `defer_imports.until_use` block contains a wildcard import or non-import statement.
        """

        if not _is_until_use_node(node):
            return self._visit_eager_import_block(node)

        # Replace the dummy context manager with the one that will actually replace __import__.
        old_ctx_expr = node.items[0].context_expr
        loc = {attr: getattr(old_ctx_expr, attr) for attr in _AST_LOC_ATTRS}
        node.items[0].context_expr = ast.Name(_ACTUAL_CTX_ASNAME, ctx=ast.Load(), **loc)

        # Actually instrument the import nodes.
        node.body = self._substitute_import_keys(self._validate_until_use_body(node.body))

        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports and cleanup necessary to make `defer_imports.until_use` work properly.

        If the module is empty or `defer_imports.until_use` isn't actually used, do nothing.
        """

        if not node.body:
            return node

        node = self.generic_visit(node)  # pyright: ignore [reportAssignmentType] # We know it'll return a module.

        if not self.did_any_instrumentation:
            return node

        # First, get past the module docstring and __future__ imports. We don't want to break those.
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

        loc = {attr: getattr(node.body[position], attr) for attr in _AST_LOC_ATTRS}

        # Then, add necessary defer_imports import.
        ctx_alias = ast.alias(_ACTUAL_CTX_NAME, _ACTUAL_CTX_ASNAME, **loc)
        import_stmt = ast.ImportFrom(module=__spec__.name, names=[ctx_alias], level=0, **loc)
        node.body.insert(position, import_stmt)

        # Finally, clean up the namespace via deletion.
        loc["lineno"], loc["end_lineno"] = node.body[-1].lineno, node.body[-1].end_lineno
        del_stmt = ast.Delete(targets=[ast.Name(_ACTUAL_CTX_ASNAME, ctx=ast.Del(), **loc)], **loc)
        node.body.append(del_stmt)

        return node


# endregion


# ============================================================================
# region -------- importlib import hooks --------
#
# The module loader, module finder, and import hook API to attach those to the
# import system.
# ============================================================================


_BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Differs for every version."""


def _walk_globals(node: ast.AST) -> t.Generator[ast.AST]:
    """Recursively yield descendent nodes of a tree starting at `node`, including `node` itself.

    Nodes that introduce a new scope, such as class and function definitions, are skipped entirely.
    """

    scope_node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    todo = collections.deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, scope_node_types))
        yield node


# PYUPDATE: py3.14 - Consider inheriting from importlib.abc.SourceLoader instead since it's (probably) cheap again.
class _DeferredFileLoader(SourceFileLoader):
    """A file loader that instruments ``.py`` files which use ``with defer_imports.until_use: ...``."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.defer_whole_module: bool = False

    def get_data(self, path: str) -> bytes:
        """Return the data from `path` as raw bytes.

        If `path` points to a bytecode file, validate that it has a `defer_imports`-specific header.

        Raises
        ------
        OSError
            If the path points to a bytecode file with an invalid `defer_imports`-specific header.
            `importlib.machinery.SourceLoader.get_code()` expects this error from this function.
        """

        # NOTE: Another option is to monkeypatch `importlib.util.cache_from_source`, as beartype and typeguard do,
        # but that seems excessive for this use case.
        # Ref: https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/_clawimpload.py#L177-L312
        # NOTE: See facebookincubator/cinderx's strict loader for a much more sophisticated code rewriter that also deals with this.

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

        If the file is a bytecode one, prepend a `defer_imports`-specific header to it. That way, instrumented bytecode
        can be identified and invalidated later if necessary [1]_.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = _BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)

    # NOTE: We're purposefully not supporting data being an AST object, as that's not the use case for this method.
    # NOTE: The signatures of SourceFileLoader.source_to_code at runtime and in typeshed aren't currently consistent.
    # Ref: https://github.com/python/typeshed/pull/13880
    # Ref: https://github.com/python/typeshed/issues/13881
    def source_to_code(self, data: _SourceData, path: _ModulePath, *, _optimize: int = -1) -> types.CodeType:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Compile the source `data` into a code object, possibly instrumenting it along the way.

        Parameters
        ----------
        data: _SourceData
            A string or buffer type that `compile()` supports.
        """

        if data:
            orig_tree = ast.parse(data, path, "exec")

            if self.defer_whole_module or any(
                (isinstance(node, ast.With) and _is_until_use_node(node)) for node in _walk_globals(orig_tree)
            ):
                instrumenter = _ImportsInstrumenter(data, path, whole_module=self.defer_whole_module)
                new_tree = instrumenter.visit(orig_tree)
                return super().source_to_code(new_tree, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003
            else:
                return super().source_to_code(orig_tree, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

        return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

    def create_module(self, spec: ModuleSpec) -> t.Optional[types.ModuleType]:
        """Use default semantics for module creation."""

        # This state is needed for self.source_to_code(), which is called by self.exec_module().
        if spec.loader_state is not None:
            self.defer_whole_module = spec.loader_state["defer_whole_module"]

        return super().create_module(spec)


class _DeferredFileFinder(FileFinder):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path!r})"

    @staticmethod
    def _is_full_module_rewrite(fullname: str, config: _DeferConfig) -> bool:
        """Determine whether all imports should be instrumented instead of just `until_use`-encapsulated imports."""

        # This could be written as one boolean expression, but currently, splitting it out makes the hierarchy
        # of configuration options a bit clearer.

        if config.apply_all:
            return True

        if not config.module_names:
            return False

        if fullname in config.module_names:
            return True

        return config.recursive and any(fullname.startswith(f"{mod}.") for mod in config.module_names)

    def find_spec(self, fullname: str, target: t.Optional[types.ModuleType] = None) -> t.Optional[ModuleSpec]:
        """Try to find a spec for the specified module.

        If a spec is found, its loader may be overriden and some state may be passed on via `ModuleSpec.loader_state`
        [1]_ [2]_.

        Returns
        -------
        t.Optional[ModuleSpec]
            The matching spec, or `None` if not found.

        References
        ----------
        .. [1] https://github.com/python/cpython/issues/89527
        .. [2] https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader_state
        """

        spec = super().find_spec(fullname, target)

        # Be precise so that we don't replace user-defined loader classes.
        if (spec is not None) and (spec.loader is not None) and (spec.loader.__class__ is SourceFileLoader):
            config = _current_defer_config.get(None)
            defer_whole_module = self._is_full_module_rewrite(fullname, config) if (config is not None) else False

            # pyright doesn't respect the "obj.__class__ is ..." type guard pattern.
            spec.loader = _DeferredFileLoader(spec.loader.name, spec.loader.path)  # pyright: ignore [reportUnknownMemberType, reportAttributeAccessIssue]
            spec.loader_state = {"defer_whole_module": defer_whole_module}

        return spec


_LOADER_DETAILS = (_DeferredFileLoader, SOURCE_SUFFIXES)
_PATH_HOOK = _DeferredFileFinder.path_hook(_LOADER_DETAILS)


_current_defer_config: contextvars.ContextVar[_DeferConfig] = contextvars.ContextVar("_current_defer_config")
"""The current configuration for defer_imports's instrumentation."""


class _DeferConfig:
    """Configuration for determining whether a module should be fully instrumented."""

    __slots__ = ("apply_all", "module_names", "recursive")

    def __init__(self, apply_all: bool, module_names: t.Sequence[str], recursive: bool) -> None:
        self.apply_all = apply_all
        self.module_names = module_names
        self.recursive = recursive

    def __repr__(self, /) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"apply_all={self.apply_all!r}, module_names={self.module_names!r}, recursive={self.recursive!r}, "
            + ")"
        )


@_final
class ImportHookContext:
    """An installer and configurer for defer_imports's import hook. Must be called before using
    `defer_imports.until_use` to make it work, otherwise it will be a no-op.

    The configuration knobs are for determining which modules should be "globally" instrumented, i.e. having all their
    global imports rewritten. Imports within `defer_imports.until_use` blocks are always instrumented.

    This should be run before the code it is meant to affect is executed. One place to put do that is ``__init__.py``
    of a package or application.

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
    """

    __slots__ = ("_config", "_uninstall_after", "_tok")

    def __init_subclass__(cls, *args: object, **kwargs: object) -> t.NoReturn:
        msg = f"Type {cls.__name__!r} is not an acceptable base type."
        raise TypeError(msg)

    def __init__(
        self,
        *,
        uninstall_after: bool = False,
        apply_all: bool = False,
        module_names: t.Sequence[str] = (),
        recursive: bool = False,
    ) -> None:
        if isinstance(module_names, str):
            msg = "module_names should be a sequence of strings, not a string."
            raise TypeError(msg)

        self._config = _DeferConfig(apply_all, module_names, recursive)
        self._uninstall_after: bool = uninstall_after
        self._tok: contextvars.Token[_DeferConfig] | None = None

    def __enter__(self, /) -> Self:
        self.install()
        return self

    def __exit__(self, *_dont_care: object) -> None:
        self.reset()
        if self._uninstall_after:
            self.uninstall()

    def install(self, /) -> None:
        if _PATH_HOOK not in sys.path_hooks:
            for i, hook in enumerate(sys.path_hooks):
                if hook.__name__ == "path_hook_for_FileFinder":
                    file_finder_index = i
                    break
            else:
                msg = "No file-based path hook found to be superseded."
                raise RuntimeError(msg)

            sys.path_hooks.insert(file_finder_index, _PATH_HOOK)

            # HACK: We do some monkeypatching here so that the cached finders for sys.path entries use the right finder
            # class. This should be safe; _DeferredFinder is a subclass of FileFinder and has the same instance state.
            #
            # Alternatives:
            # - Create and insert a new PathFinder subclass into sys.meta_path, or patch the existing one.
            #   That would be a bigger monkeypatch in some ways, but it's the route that typeguard takes
            #   (and one we took previously).
            # - Delete the sys.path_importer_cache entries instead of monkeypatching them.
            #   This is recommended by the docs and is technically more correct, but it can cause a big slowdown on
            #   startup.
            for finder in sys.path_importer_cache.values():
                if (finder is not None) and (finder.__class__ is FileFinder):
                    finder.__class__ = _DeferredFileFinder

        self._tok = _current_defer_config.set(self._config)

    def reset(self, /) -> None:
        """Attempt to reset the import hook configuration. If already reset, does nothing."""

        if self._tok is not None:
            _current_defer_config.reset(self._tok)
            self._tok = None

    def uninstall(self, /) -> None:
        """Attempt to remove the custom path hook in `sys.path_hooks`."""

        try:
            sys.path_hooks.remove(_PATH_HOOK)
        except ValueError:
            pass

        # Undo any monkeypatching done by self.install(), and remove the presence of _DeferredFileFinder entirely.
        # We don't also have to invalidate the finder cache, I think; FileFinder's cache is just for potential module
        # locations, which the monkeypatch doesn't affect.
        for finder in sys.path_importer_cache.values():
            if (finder is not None) and (finder.__class__ is _DeferredFileFinder):
                finder.__class__ = FileFinder


import_hook = ImportHookContext


# endregion


# ============================================================================
# region -------- Runtime magic --------
#
# The proxies, __import__ replacement, and actual until_use context manager.
# ============================================================================


_ImportArgs: TypeAlias = "tuple[str, t.MutableMapping[str, t.Any], t.MutableMapping[str, t.Any], t.Optional[str]]"


_original_import = contextvars.ContextVar("_original_import", default=builtins.__import__)
"""What builtins.__import__ last pointed to."""

_is_deferred = contextvars.ContextVar[bool]("_is_deferred", default=False)
"""Whether imports in import statements should be deferred."""


def _handle_import_key(import_name: str, nmsp: t.MutableMapping[str, t.Any], start_idx: int = 0, /) -> None:
    # Precondition: full_name is a dotted name.

    while True:
        end_idx = import_name.find(".", start_idx)

        at_end_of_name = end_idx == -1
        if not at_end_of_name:
            submod_name = import_name[start_idx:end_idx]
        else:
            submod_name = import_name[start_idx:]

        # NOTE: Normal "==" semantics via base_tree.__contains__ causes a performance issue because _DIKey's very slow
        # __eq__ always take priority over str's.
        existing_key = next(filter(submod_name.__eq__, nmsp), None)

        if existing_key is not None:
            if existing_key.__class__ is _DIKey:
                if existing_key._DIKey__submod_names:  # pyright: ignore [reportAttributeAccessIssue, reportUnknownMemberType]
                    existing_key._DIKey__submod_names.add(import_name)  # pyright: ignore  # noqa: PGH003 # Too verbose.
                else:
                    existing_key._DIKey__submod_names = {import_name}  # pyright: ignore  # noqa: PGH003 # Too verbose.
                break

            if not at_end_of_name:
                nmsp = nmsp[submod_name].__dict__

        elif not at_end_of_name:
            full_submod_name = import_name[:end_idx]

            sub_names: set[str] = set()
            while (end_idx := import_name.find(".", start_idx)) != -1:
                sub_names.add(import_name[:end_idx])
                start_idx = end_idx + 1
            sub_names.add(import_name)

            # Replace the namespaces as well to make sure the proxy is replaced in the right place.
            # NOTE: We can't use setattr() on the module here because PyPy would normalize the attr key type to
            # `str`. Instead, use the module dict directly.
            nmsp[_DIKey(submod_name, (full_submod_name, nmsp, nmsp, None), sub_names)] = _DIProxy(full_submod_name)
            break

        else:
            nmsp[_DIKey(submod_name, (import_name, nmsp, nmsp, None))] = _DIProxy(import_name)
            break

        start_idx = end_idx + 1


class _DIProxy:
    # NOTE: Mangle instance attribute name(s) to help avoid further exposure if an instance leaks and a user tries to
    # use it as a regular module.

    __slots__ = ("__import_name",)

    def __init__(self, import_name: str, /) -> None:
        self.__import_name = import_name

    def __repr__(self, /) -> str:
        return f"<proxy for {self.__import_name!r} import>"

    def __getattr__(self, name: str, /) -> Self:
        return self.__class__(f"{self.__import_name}.{name}")


class _DIKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    # NOTE: Mangle instance attribute name(s) to help avoid further exposure if an instance leaks and a user tries to
    # use it as a regular string.

    __slots__ = ("__import_args", "__submod_names", "__is_resolving", "__lock")

    __import_args: _ImportArgs
    __submod_names: t.Optional[set[str]]
    __is_resolving: bool
    __lock: threading.RLock

    def __new__(cls, asname: str, import_args: _ImportArgs, submod_names: t.Optional[set[str]] = None, /) -> Self:
        self = super().__new__(cls, asname)
        self.__import_args = import_args
        self.__submod_names = submod_names
        self.__is_resolving = False
        self.__lock = threading.RLock()
        return self

    def __eq__(self, value: object, /) -> bool:
        # Micro-optimization: Use str.__eq__() instead of super().__eq__().

        if _is_deferred.get():
            return str.__eq__(self, value)

        is_eq = str.__eq__(self, value)
        if is_eq is not True:
            return is_eq

        # Only the first thread to grab the lock should resolve the deferred import.
        with self.__lock:
            # Reentrant calls from the same thread shouldn't re-trigger the resolution.
            # This can be caused by self-referential imports, e.g. within __init__.py files.
            if not self.__is_resolving:
                self.__is_resolving = True
                self.__resolve_import()

        return True

    __hash__ = str.__hash__

    def __resolve_import(self, /) -> None:
        """Resolve the import and replace the deferred key and placeholder in the relevant namespace with the result."""

        raw_asname = str(self)
        imp_name, imp_globals, imp_locals, from_item = self.__import_args

        # 1. Perform the original __import__ and pray.
        from_list = (from_item,) if (from_item is not None) else ()
        module: types.ModuleType = _original_import.get()(imp_name, imp_globals, imp_locals, from_list, 0)

        # 2. Create nested keys and proxies as needed in the resolved module.
        if self.__submod_names:
            starting_point = len(imp_name) + 1
            # Avoid triggering our __eq__ again.
            _is_deferred_tok = _is_deferred.set(True)
            try:
                for submod_name in self.__submod_names:
                    _handle_import_key(submod_name, module.__dict__, starting_point)
            finally:
                _is_deferred.reset(_is_deferred_tok)

            self.__submod_names = None

        # 3. Replace the deferred version of the key in the relevant namespace to avoid it sticking around.
        # Avoid triggering our __eq__ again (would be a recursive trigger too).
        _is_deferred_tok = _is_deferred.set(True)
        try:
            imp_locals[raw_asname] = imp_locals.pop(raw_asname)
        finally:
            _is_deferred.reset(_is_deferred_tok)

        # 4. Resolve any requested attribute access and replace the proxy with the result in the relevant namespace.
        if from_item is not None:
            imp_locals[raw_asname] = getattr(module, from_item)
        elif ("." in imp_name) and ("." not in raw_asname):
            attr = module
            for attr_name in imp_name.rpartition(".")[2].split("."):
                attr = getattr(attr, attr_name)
            imp_locals[raw_asname] = attr
        else:
            imp_locals[raw_asname] = module


def _deferred___import__(  # noqa: PLR0912
    name: str,
    globals: t.MutableMapping[str, t.Any],
    locals: t.MutableMapping[str, t.Any],
    fromlist: t.Optional[t.Sequence[str]] = None,
    level: int = 0,
) -> t.Any:
    """An limited replacement for `__import__` that supports deferred imports by returning proxies."""

    # Preconditions:
    # 1. Only invoked by syntactic import statements; cannot be called manually via __import__().
    #     - Allows a stricter signature and less input validation because we have a more limited range of inputs and
    #       can depend on the builtin parser to handle some validation, e.g. making sure level >= 0.
    # 2. Only called within the context of "with _actual_until_use: ...".
    #     - _is_deferred is set to True and thus _DIKey instances won't trigger resolution.

    # Do minimal input validation on top of the parser's work.
    if level > 0:
        # These checks are adapted and inlined from importlib.__import__() and importlib._bootstrap._sanity_check()
        # since we don't need all of them.
        package = _calc___package__(globals)

        if not isinstance(package, str):  # pragma: no cover (tested in stdlib)
            msg = "__package__ not set to a string"
            raise TypeError(msg)

        if not package:  # pragma: no cover (tested in stdlib)
            msg = "attempted relative import with no known parent package"
            raise ImportError(msg)

        name = _resolve_name(name, package, level)

    # Invariant: locals[_TEMP_ASNAMES] must exist.
    #
    # The AST transformer guarantees that it exists as `tuple[str | None, ...]` when fromlist is populated, or as
    # `str | None` otherwise.
    # Since we can't dependently annotate it that way, and annotating it as a union would require isinstance checks to
    # satisfy the type checker, leaving it as Any "satisfies" the type checker with less runtime cost.
    asname: t.Any = locals[_TEMP_ASNAMES]

    if not fromlist:
        if asname:
            if "." in name:
                # Case 1: import a.b as c
                # NOTE: Pretending it's a "from" import is cheaper than the alternative.
                parent_name, _, submod_name = name.rpartition(".")
                locals[_DIKey(asname, (parent_name, globals, locals, submod_name))] = None
                result = _DIProxy(parent_name)
            else:
                # Case 2: import a as c
                locals[_DIKey(asname, (name, globals, locals, None))] = None
                result = _DIProxy(name)

        elif "." in name:
            # Case 3: import a.b
            parent_name = name.partition(".")[0]
            try:
                preexisting = locals[parent_name]
            except KeyError:
                locals[_DIKey(parent_name, (parent_name, globals, locals, None), {name})] = None
                result = _DIProxy(parent_name)
            else:
                # Heuristic: If locals[parent_name] is not a module or proxy, pretend it didn't exist and clobber it.
                if not isinstance(preexisting, (types.ModuleType, _DIProxy)):
                    locals[_DIKey(parent_name, (parent_name, globals, locals, None), {name})] = None
                    result = _DIProxy(parent_name)
                else:
                    # Case 3.a: import importlib[.abc], -> importlib.util <-
                    _handle_import_key(name, locals)
                    result = preexisting

        else:
            # Case 4: import a
            locals[_DIKey(name, (name, globals, locals, None))] = None
            result = _DIProxy(name)

    else:
        # Case 5: from ... import ... [as ...]
        from_asname: str | None
        for from_name, from_asname in zip(fromlist, asname):
            locals[_DIKey(from_asname or from_name, (name, globals, locals, from_name))] = None

        result = _DIProxy(name)

    return result


class _DeferredContext:
    """A context manager within which imports occur lazily. Not reentrant. Use via `defer_imports.until_use`.

    If defer_imports isn't set up properly, e.g. `import_hook.install()` is not called first elsewhere, this should be a
    no-op equivalent to `contextlib.nullcontext`.

    Raises
    ------
    SyntaxError
        If `defer_imports.until_use` is used improperly, e.g. it contains a wildcard import or a non-import statement.

    Notes
    -----
    As part of its implementation, this temporarily replaces `builtins.__import__`.
    """

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self, /) -> None:
        self._defer_ctx_token = _is_deferred.set(True)
        self._import_ctx_token = _original_import.set(builtins.__import__)
        builtins.__import__ = _deferred___import__

    def __exit__(self, *_dont_care: object) -> None:
        _original_import.reset(self._import_ctx_token)
        _is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = _original_import.get()


#: The context manager that replaces until_use after instrumentation.
_actual_until_use = _DeferredContext()


# endregion
