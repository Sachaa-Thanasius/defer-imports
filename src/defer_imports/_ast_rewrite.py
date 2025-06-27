from __future__ import annotations

import ast
import builtins
import contextvars
import sys
import threading
import types
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, SourceFileLoader

from . import __version__ as _version, lazy_load as _lazy_load


with _lazy_load.until_module_use():
    import collections
    import io
    import itertools
    import os
    import tokenize
    import typing as t
    import warnings


# ============================================================================
# region -------- Compatibility shims --------
#
# Version-dependent or hidden-from-type-checking imports and definitions.
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


# collections always imports collections.abc in 3.13+ (which typeshed isn't aware of yet).
# Ref: https://github.com/python/cpython/pull/125415
# However, collections.abc.Buffer was added in 3.12. We settle on this for a middle ground.
if TYPE_CHECKING:
    from typing_extensions import Buffer as ReadableBuffer
elif sys.version_info >= (3, 13):  # pragma: >=3.13 cover
    ReadableBuffer: t.TypeAlias = "collections.abc.Buffer"
else:  # pragma: <3.13 cover
    ReadableBuffer: TypeAlias = "t.Union[bytes, bytearray, memoryview]"


if TYPE_CHECKING:
    final = t.final
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


if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    _SyntaxContext: TypeAlias = (
        "tuple[t.Optional[str], t.Optional[int], t.Optional[int], t.Optional[str], t.Optional[int], t.Optional[int]]"
    )
else:  # pragma: <3.10 cover
    _SyntaxContext: TypeAlias = "tuple[t.Optional[str], t.Optional[int], t.Optional[int], t.Optional[str]]"


# compile()'s internals, and thus wrappers of it (e.g. ast.parse()), dropped support in 3.12 for non-bytes buffers as
# the filename argument.
# Ref: https://github.com/python/cpython/issues/98393
if sys.version_info >= (3, 12):  # pragma: >=3.12 cover
    _ModulePath: TypeAlias = "t.Union[str, os.PathLike[str], bytes]"
else:  # pragma: <3.12 cover
    _ModulePath: TypeAlias = "t.Union[str, os.PathLike[str], ReadableBuffer]"


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
#
# License info
# ------------
# The original sources are
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap.py
# and https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap_external.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.
#
# If any changes are made to the adapted constructs, a short summary of those
# changes accompanies their definitions.
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
# Changes:
# - Account for warnings being different across versions.
def _calc___package__(globals: t.Mapping[str, t.Any]) -> t.Optional[str]:  # pragma: no cover (tested in stdlib)
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


# endregion


# ============================================================================
# region -------- AST transformation --------
# ============================================================================


# NOTE: Technically, something like ast.AST & LocationAttrsProtocol would be more accurate, but:
# 1.  Python doesn't have intersections yet, and
# 2.  Using a local protocol without eagerly importing typing or having another module isn't doable until 3.12.
_ASTWithLocation: TypeAlias = "t.Union[ast.expr, ast.stmt]"


# NOTE: Make our generated variables more hygienic by prefixing their names with "_@di_". A few reasons for this choice:
#
# 1.  The "@" isn't a valid character for Python identifiers, so there won't be conflicts with "regular" user code.
#     pytest does something similar.
# 2.  The "di" namespaces the symbols somewhat. pytest does something similiar.
# 3.  The starting "_" will prevent the symbols from being picked up by code that programmatically accesses a namespace
#     during module execution but avoids symbols starting with an underscore.
#     - An example of a common pattern in the standard library that does this:
#       __all__ = [name for name in globals() if name[:1] != "_"]  # noqa: ERA001
_HYGIENE_PREFIX = "_@di_"

_ACTUAL_CTX_NAME = "_DIContext"
_ACTUAL_CTX_ASNAME = f"{_HYGIENE_PREFIX}{_ACTUAL_CTX_NAME}"
_TEMP_ASNAMES = f"{_HYGIENE_PREFIX}_temp_asnames"

#: The names of location attributes of AST nodes.
_AST_LOC_ATTRS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def _is_until_use_node(node: ast.AST, /) -> bool:
    """Check if the node matches ``with defer_imports.until_use(): ...``."""

    if not (isinstance(node, ast.With) and len(node.items) == 1):
        return False

    context_expr = node.items[0].context_expr
    if not isinstance(context_expr, ast.Call):
        return False

    func = context_expr.func
    if not (isinstance(func, ast.Attribute) and func.attr == "until_use"):
        return False

    expr_value = func.value
    return isinstance(expr_value, ast.Name) and expr_value.id == "defer_imports"


def _get_joined_source_lines(source: str, node: _ASTWithLocation) -> t.Optional[str]:
    """Get the source code lines of `source` that generated `node`, or None if `node` lacks location information."""

    try:
        (lineno, end_lineno) = (node.lineno, node.end_lineno)
    except AttributeError:
        return None

    if end_lineno is None:
        return None

    # Convert from 1-indexed to 0-indexed.
    lineno -= 1
    end_lineno -= 1

    # Split a string into lines while ignoring form feed and other chars.
    # This mimics how the Python parser splits source code.
    with io.StringIO(source, newline=None) as source_buffer:
        return "".join(itertools.islice(source_buffer, lineno, end_lineno + 1))


class _ImportsInstrumenter(ast.NodeTransformer):
    """AST transformer that instruments imports within ``with defer_imports.until_use(): ...`` blocks.

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

    def _add_asname_trackers(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> list[ast.stmt]:
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

    def _wrap_imports_list(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> ast.With:
        """Wrap a list of import nodes with a `defer_imports.until_use` block and instrument them."""

        loc = {attr: getattr(import_nodes[0], attr) for attr in _AST_LOC_ATTRS}
        call = ast.Call(func=ast.Name(_ACTUAL_CTX_ASNAME, ctx=ast.Load(), **loc), args=[], keywords=[], **loc)
        return ast.With(items=[ast.withitem(context_expr=call)], body=self._add_asname_trackers(import_nodes), **loc)

    def _is_instrumentable_import(self, node: ast.AST) -> bool:
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
                imports_counter = 0

                for value in old_value:  # pyright: ignore [reportUnknownVariableType]
                    if isinstance(value, ast.AST):
                        # DI addition: This if block.
                        if self.rewrite_whole_module:
                            if self._is_instrumentable_import(value):
                                imports_counter += 1
                            elif imports_counter > 0:
                                new_values[-imports_counter:] = [self._wrap_imports_list(new_values[-imports_counter:])]
                                imports_counter = 0

                        value = self.visit(value)  # noqa: PLW2901

                        if value is None:
                            continue
                        if not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue

                    new_values.append(value)

                # DI addition: This if block.
                if self.rewrite_whole_module and imports_counter > 0:
                    new_values[-imports_counter:] = [self._wrap_imports_list(new_values[-imports_counter:])]

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

    def _get_syntax_context(self, node: _ASTWithLocation) -> _SyntaxContext:
        """Get a node's location context in a form compatible with `SyntaxError`'s constructor [1].

        References
        ----------
        .. [1] https://docs.python.org/3.14/library/exceptions.html#SyntaxError
        """

        filepath = self.filepath if isinstance(self.filepath, (str, bytes, os.PathLike)) else bytes(self.filepath)
        source = self.source if isinstance(self.source, str) else _decode_source(self.source)
        text = _get_joined_source_lines(source, node)
        context = (os.fsdecode(filepath), node.lineno, node.col_offset + 1, text)

        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = node.end_col_offset
            context += (node.end_lineno, (end_col_offset + 1) if (end_col_offset is not None) else None)

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
                raise SyntaxError(msg, self._get_syntax_context(node))  # noqa: TRY004 # Syntax error displays better.

            if node.names[0].name == "*":
                msg = "import * not allowed in with defer_imports.until_use blocks"
                raise SyntaxError(msg, self._get_syntax_context(node))

        # Warning: Don't mutate the list from outside the function to invalidate our type guard.
        return nodes  # pyright: ignore [reportReturnType]

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that ``with defer_imports.until_use(): ...`` blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If a `defer_imports.until_use` block contains a wildcard import or non-import statement.
        """

        if not _is_until_use_node(node):
            return self._visit_eager_import_block(node)

        # Replace the dummy context manager with the one that will actually replace __import__.
        loc = {attr: getattr(node.items[0].context_expr, attr) for attr in _AST_LOC_ATTRS}
        node.items[0].context_expr = ast.Call(ast.Name(_ACTUAL_CTX_ASNAME, ctx=ast.Load(), **loc), [], [], **loc)

        # Actually instrument the import nodes.
        node.body = self._add_asname_trackers(self._validate_until_use_body(node.body))

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
        for position, sub in enumerate(node.body):  # noqa: B007 # position is used after the loop.
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


#: Custom header for defer_imports-instrumented bytecode files. Differs for every version.
_BYTECODE_HEADER = f"defer_imports{_version}".encode()


#: The current configuration for defer_imports's instrumentation.
_current_config = contextvars.ContextVar[tuple[str, ...]]("_current_config", default=())


def _walk_globals(node: ast.AST) -> t.Generator[ast.AST, None, None]:
    """Recursively yield descendent nodes of a tree starting at `node`, including `node` itself.

    *Child* nodes that introduce a new scope, such as class and function definitions, are skipped entirely.
    """

    SCOPE_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    unvisited = collections.deque([node])
    while unvisited:
        node = unvisited.popleft()
        unvisited.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, SCOPE_NODE_TYPES))
        yield node


# PYUPDATE: py3.14 - Consider inheriting from importlib.abc.SourceLoader instead since it's (probably) cheap again.
class _DIFileLoader(SourceFileLoader):
    """A file loader that instruments ``.py`` files which use ``with defer_imports.until_use(): ...``."""

    def get_data(self, path: str) -> bytes:
        """Return the data from `path` as raw bytes.

        If `path` points to a bytecode file, validate that it has a `defer_imports`-specific header.

        Raises
        ------
        OSError
            If the path points to a bytecode file with an invalid `defer_imports`-specific header.
            `importlib.machinery.SourceLoader.get_code()` expects this error from this function.
        """

        # NOTE: There are other options:
        # 1.  Monkeypatch `importlib.util.cache_from_source`, as beartype and typeguard do.
        #     Ref: https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/_clawimpload.py#L177-L312
        # 2.  Do whatever facebookincubator/cinderx's strict loader does, which involves overriding get_code().

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

        If the file is a bytecode one, add a `defer_imports`-specific header to it. That way, instrumented bytecode
        can be identified and invalidated later if necessary [1]_.

        References
        ----------
        .. [1] https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/
        """

        if path.endswith(tuple(BYTECODE_SUFFIXES)):
            data = _BYTECODE_HEADER + data

        return super().set_data(path, data, _mode=_mode)

    # NOTE: In 3.12+, the path parameter should only accept bytes, not any buffer.
    # Ref: https://github.com/python/typeshed/issues/13881
    # NOTE: We're purposefully not supporting data being an AST object, as that's not the use case for this method.
    def source_to_code(self, data: _SourceData, path: _ModulePath, *, _optimize: int = -1) -> types.CodeType:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Compile the source `data` into a code object, possibly instrumenting it along the way.

        Parameters
        ----------
        data: _SourceData
            A string or buffer type that `compile()` supports.
        path: _ModulePath
            The file from which `data` was read.
        """

        if not data:
            return compile(data, path, "exec", dont_inherit=True, optimize=_optimize)

        orig_tree: ast.Module = compile(data, path, "exec", ast.PyCF_ONLY_AST, dont_inherit=True, optimize=_optimize)

        if not (self.defer_whole_module or any(map(_is_until_use_node, _walk_globals(orig_tree)))):
            return compile(orig_tree, path, "exec", dont_inherit=True, optimize=_optimize)

        instrumenter = _ImportsInstrumenter(data, path, whole_module=self.defer_whole_module)
        new_tree = instrumenter.visit(orig_tree)
        return compile(new_tree, path, "exec", dont_inherit=True, optimize=_optimize)

    def create_module(self, spec: ModuleSpec) -> t.Optional[types.ModuleType]:
        """Use default semantics for module creation. Also, get some state from the spec."""

        self.defer_whole_module: bool = spec.loader_state["defer_whole_module"]
        return super().create_module(spec)


class _DIFileFinder(FileFinder):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path!r})"

    @staticmethod
    def _is_full_module_rewrite(config: t.Optional[tuple[str, ...]], fullname: str) -> bool:
        """Determine whether all global imports should be instrumented *in addition to* `until_use`-wrapped imports."""

        return bool(config) and (
            (config[0] == "*")
            or (fullname in config)
            or any((mod.endswith(".*") and fullname.startswith(mod[:-1])) for mod in config)
        )

    def find_spec(self, fullname: str, target: t.Optional[types.ModuleType] = None) -> t.Optional[ModuleSpec]:
        """Try to find a spec for the specified module.

        If found, attach some loader-specific state and potentially replace the loader.
        """

        spec = super().find_spec(fullname, target)

        if (spec is not None) and ((loader := spec.loader) is not None):
            # Lock in the config between finding and loading.
            config = _current_config.get()
            spec.loader_state = {"defer_whole_module": self._is_full_module_rewrite(config, fullname)}

            # HACK: Support for monkeypatch in import_hook.install().
            #
            # Those patched finders won't return a spec with a _DIFileLoader loader due to preset instance state.
            # Account for those with a very specific check so that we don't override user-defined loaders.
            if loader.__class__ is SourceFileLoader:
                loader.__class__ = _DIFileLoader

        return spec


_PATH_HOOK = _DIFileFinder.path_hook((_DIFileLoader, SOURCE_SUFFIXES))


@final
class ImportHookContext:
    """An installer and configurer for defer_imports's import hook.

    Before this is called, `defer_imports.until_use` will be a no-op.

    Install *before* importing modules this is meant to affect. One place to do that is ``__init__.py`` of a package
    or application.

    Installation ensures that imports within `defer_imports.until_use` blocks are always instrumented within any modules
    imported *afterwards*. The configuration knobs are only for determining which modules should be globally
    instrumented, having *all* their global imports rewritten (excluding those in escape hatches).

    Parameters
    ----------
    module_names: t.Sequence[str], default=()
        A set of modules to apply global-scope import statement instrumentation to.

        - If passed ``["*"]``, global-scope import statement instrumentation will occur in all modules imported
        henceforth. (Better suited for applications than libraries.)
        - If one of the module names ends with ".*", global-scope import statement instrumentation will occur in that
        module and its submodules recursively. (Better suited for libraries than applications.)
    uninstall_after: bool, default=False
        Whether to uninstall the import hook upon exit if this function is used as a context manager.
    """

    __slots__ = ("_config", "_uninstall_after", "_config_token")

    def __init_subclass__(cls, *args: object, **kwargs: object) -> t.NoReturn:
        msg = f"Type {cls.__name__!r} is not an acceptable base type."
        raise TypeError(msg)

    def __init__(self, /, module_names: t.Sequence[str] = (), *, uninstall_after: bool = False) -> None:
        if isinstance(module_names, str):
            msg = "module_names should be a sequence of strings, not a string."
            raise TypeError(msg)

        self._config: tuple[str, ...] = tuple(module_names)
        self._uninstall_after: bool = uninstall_after
        self._config_token: contextvars.Token[tuple[str, ...]] | None = None

    def __enter__(self, /) -> Self:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.reset()
        if self._uninstall_after:
            self.uninstall()

    def install(self, /) -> None:
        """Install the custom import hook that allows `defer_imports.until_use` to work."""

        if _PATH_HOOK not in sys.path_hooks:
            for i, hook in enumerate(sys.path_hooks):
                # NOTE: This is a pretty fragile search criteria, but we don't have any good alternatives.
                if hook.__name__ == "path_hook_for_FileFinder":
                    sys.path_hooks.insert(i, _PATH_HOOK)
                    break
            else:
                msg = "No file-based path hook found to be superseded."
                raise RuntimeError(msg)

            # HACK: Monkeypatch cached finders for sys.path entries to have the right finder class.
            # Safe as long as the instances have the same class layout and state. The latter isn't true, but it's
            # accomodated for in _DiFileFinder.find_spec().
            #
            # Alternatives:
            # - Create and insert a new PathFinder subclass into sys.meta_path, or monkeypatch the existing one.
            #   This is what typeguard does, but it's more extreme in some ways.
            # - Clear sys.path_importer_cache instead of monkeypatching its values.
            #   This is recommended by the docs and is more "correct", but it can cause a big slowdown on startup.
            for finder in sys.path_importer_cache.values():
                if (finder is not None) and (finder.__class__ is FileFinder):
                    finder.__class__ = _DIFileFinder

        self._config_token = _current_config.set(self._config)

    def reset(self, /) -> None:
        """Attempt to reset the import hook configuration. If already reset, does nothing."""

        if self._config_token is not None:
            _current_config.reset(self._config_token)
            self._config_token = None

    def uninstall(self, /) -> None:
        """Attempt to uninstall the custom import hook. If already uninstalled, does nothing."""

        try:
            sys.path_hooks.remove(_PATH_HOOK)
        except ValueError:
            pass

        # Undo the monkeypatching done in self.install().
        # We don't have to invalidate the unaffected finder caches, which only contain potential module locations.
        for finder in sys.path_importer_cache.values():
            if (finder is not None) and (finder.__class__ is _DIFileFinder):
                finder.__class__ = FileFinder


import_hook = ImportHookContext


# endregion


# ============================================================================
# region -------- Runtime magic --------
#
# The proxies, __import__ replacement, and actual until_use context manager.
# ============================================================================


_ImportArgs: TypeAlias = "tuple[str, dict[str, t.Any], dict[str, t.Any], t.Optional[str]]"


#: Whether imports in import statements should be deferred.
_is_deferred: bool = False

#: A lock to guard _is_deferred.
_is_deferred_lock = threading.RLock()


class _TempDeferred:
    """A re-entrant context manager for temporarily setting `_is_deferred`.

    Used to prevent deferred imports from resolving while within the `until_use` block. It also helps avoid deadlocks
    with certain cases of self-referential deferred imports.
    """

    __slots__ = ("previous",)

    def __init__(self, /) -> None:
        self.previous: bool = False

    def __enter__(self, /) -> Self:
        global _is_deferred  # noqa: PLW0603

        _is_deferred_lock.acquire()
        self.previous = _is_deferred
        _is_deferred = True

        return self

    def __exit__(self, *exc_info: object) -> None:
        global _is_deferred  # noqa: PLW0603

        _is_deferred = self.previous
        _is_deferred_lock.release()


def _accumulate_dotted_parts(dotted_name: str, start: int, /) -> set[str]:
    """Return an accumulation of dot-separated components from a dotted string."""

    sub_names: set[str] = set()
    while (end := dotted_name.find(".", start)) != -1:
        sub_names.add(dotted_name[:end])
        start = end + 1
    sub_names.add(dotted_name)
    return sub_names


# NOTE: Ways to get the exact key from a dict without triggering the expensive _DIKey.__eq__ continuously.
# It can take a significant amount of time. Normal "==" semantics are triggered by most presence checks, e.g.
# nmsp.__contains__, and that causes a performance issue because _DIKey's very slow __eq__ always take priority over
# str's. Thus, we avoid those routes.
# PYUPDATE: py3.14 - Check that this fast path still works.
if sys.implementation.name == "cpython" and (3, 9) <= sys.version_info < (3, 14):  # pragma: cpython cover

    def _get_exact_key(name: str, dct: dict[str, t.Any]) -> t.Optional[str]:
        keys = {name}.intersection(dct)
        return keys.pop() if keys else None

else:  # pragma: cpython no cover

    def _get_exact_key(name: str, dct: dict[str, t.Any]) -> t.Optional[str]:
        return next(filter(name.__eq__, dct), None)


def _handle_import_key(import_name: str, nmsp: dict[str, t.Any], start_idx: t.Optional[int] = None, /) -> None:
    """Ensure that a dotted import name (e.g., "a.b.c") is represented as a chain of deferred proxies
    in the target namespace.
    """

    # NOTE: We can't use setattr() on modules directly because PyPy normalizes the attr key type to `str` in setattr().
    # Instead, we assign into the module dict.

    if start_idx is None:
        start_idx = import_name.find(".") + 1

    end_idx = import_name.find(".", start_idx)
    at_final_part = end_idx == -1

    submod_name = import_name[start_idx : (end_idx if (not at_final_part) else None)]
    existing_key = _get_exact_key(submod_name, nmsp)

    if existing_key is not None:
        if isinstance(existing_key, _DIKey):
            existing_key._di_add_submodule_name(import_name)
        else:
            # Recurse with a nested namespace.
            _handle_import_key(import_name, nmsp[submod_name].__dict__, end_idx + 1)
    elif not at_final_part:
        full_submod_name = import_name[:end_idx]
        sub_names = _accumulate_dotted_parts(import_name, end_idx + 1)
        nmsp[_DIKey(submod_name, (full_submod_name, nmsp, nmsp, None), sub_names)] = _DIProxy(full_submod_name)
    else:
        nmsp[_DIKey(submod_name, (import_name, nmsp, nmsp, None))] = _DIProxy(import_name)


class _DIProxy:
    __slots__ = ("__import_name",)

    def __init__(self, import_name: str, /) -> None:
        self.__import_name: str = import_name

    def __repr__(self, /) -> str:
        return f"<proxy for {self.__import_name!r} import>"

    def __getattr__(self, name: str, /) -> Self:
        return self.__class__(f"{self.__import_name}.{name}")


class _DIKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("__import_args", "__is_resolved", "__lock", "__submod_names")

    __import_args: _ImportArgs
    __is_resolved: bool
    __lock: threading.Lock
    __submod_names: t.Optional[set[str]]

    def __new__(cls, obj: object, import_args: _ImportArgs, submod_names: t.Optional[set[str]] = None, /) -> Self:
        self = super().__new__(cls, obj)

        self.__import_args = import_args
        self.__is_resolved = False
        self.__lock = threading.Lock()
        self.__submod_names = submod_names

        return self

    def __eq__(self, value: object, /) -> bool:
        if _is_deferred:
            return super().__eq__(value)

        is_eq = super().__eq__(value)
        if is_eq is not True:
            return is_eq

        # Only the first thread to grab the lock should resolve the deferred import.
        with self.__lock:
            if self.__is_resolved:
                return True

            self.__resolve_import()
            self.__is_resolved = True
            return True

    __hash__ = str.__hash__

    def _di_add_submodule_name(self, submod_name: str, /) -> None:
        if self.__submod_names:
            self.__submod_names.add(submod_name)
        else:
            self.__submod_names = {submod_name}

    def __resolve_import(self, /) -> None:
        """Resolve the import and replace the deferred key and placeholder in the relevant namespace with the result."""

        # We must temporarily prevent _DIKeys from resolving while we do things that may re-trigger their __eq__.
        temp_deferred = _TempDeferred()

        raw_asname = str(self)
        imp_name, imp_globals, imp_locals, from_name = self.__import_args
        from_list = (from_name,) if (from_name is not None) else ()

        with temp_deferred:
            # 1. Perform the builtin __import__ and pray.
            # __eq__ trigger: Internal import machinery when it tries to assign a submodule to a parent module.
            module: types.ModuleType = builtins.__import__(imp_name, imp_globals, imp_locals, from_list, 0)

            # 2. Replace the deferred key in the relevant namespace to avoid it sticking around.
            # __eq__ trigger: dict.pop()
            imp_locals[raw_asname] = imp_locals.pop(raw_asname)

        # 3. Resolve any requested attribute access, then replace the proxy with the result in the relevant namespace.
        if from_name is not None:
            imp_locals[raw_asname] = getattr(module, from_name)
        elif ("." in imp_name) and ("." not in raw_asname):
            # Equvalent to functools.reduce(getattr, imp_name.split(".")[1:], module), but without an import.
            attr = module
            for attr_name in imp_name.split(".")[1:]:
                attr = getattr(attr, attr_name)
            imp_locals[raw_asname] = attr
        else:
            imp_locals[raw_asname] = module

        # 4. Create nested keys and proxies as needed in the resolved module.
        if self.__submod_names:
            with temp_deferred:
                # __eq__ trigger: _handle_import_key()
                for submod_name in self.__submod_names:
                    _handle_import_key(submod_name, module.__dict__)

                self.__submod_names = None


def _deferred___import__(
    name: str,
    globals: dict[str, t.Any],
    locals: dict[str, t.Any],
    fromlist: t.Optional[t.Sequence[str]] = None,
    level: int = 0,
) -> t.Union[types.ModuleType, _DIProxy]:
    """A limited replacement for `__import__` that supports deferred imports by returning proxies.

    Should only be invoked by ``import`` statements.

    Refer to `__import__` for more information on the expected arguments.
    """

    # Thanks to our AST transformer, asname should be a tuple[str | None, ...] when fromlist is populated, or a
    # str | None otherwise.
    # Since we can't dependently annotate it that way, and annotating it as a union would require isinstance checks to
    # satisfy the type checker later on, keeping it as Any "satisfies" the type checker with less runtime cost.
    try:
        asname: t.Any = locals[_TEMP_ASNAMES]
    except KeyError:
        msg = "attempted deferred import in a module not instrumented by defer_imports"
        raise ImportError(msg, name=name) from None

    # Depend on the parser for some input validation for import statements. It should ensure that fromlist is all
    # strings, level >= 0, that kind of thing. The remaining validation and transformation below is adapted from
    # importlib.__import__() and importlib._bootstrap._sanity_check().
    if level > 0:  # pragma: no cover (tested in stdlib)
        package = _calc___package__(globals)

        if not isinstance(package, str):
            msg = "__package__ not set to a string"
            raise TypeError(msg)

        if not package:
            msg = "attempted relative import with no known parent package"
            raise ImportError(msg)

        name = _resolve_name(name, package, level)

    # Handle the various types of import statements.
    if fromlist:
        # Case 1: from ... import ... [as ...]
        from_asname: str | None
        for from_name, from_asname in zip(fromlist, asname):
            visible_name = from_asname or from_name
            locals[_DIKey(visible_name, (name, globals, locals, from_name))] = locals.pop(visible_name, None)
        result = _DIProxy(name)

    elif "." not in name:
        # Case 2 & 3: import a [as c]
        visible_name = asname or name
        locals[_DIKey(visible_name, (name, globals, locals, None))] = locals.pop(visible_name, None)
        result = _DIProxy(name)

    elif asname:
        # Case 4: import a.b.c as d
        # It's less work to treat this as a "from" import, e.g. from a.b import c as d.
        parent_name, _, submod_as_from_name = name.rpartition(".")
        locals[_DIKey(asname, (parent_name, globals, locals, submod_as_from_name))] = locals.pop(asname, None)
        result = _DIProxy(parent_name)

    else:
        # Case 5: import a.b
        root_name = name.partition(".")[0]

        # _get_exact_key() can trigger _DIKey.__eq__, which can trigger resolution.
        if not _is_deferred:
            msg = "attempted deferred import outside the context of a ``with defer_imports.until_use(): ...`` block"
            raise ImportError(msg, name=name)

        existing_key = _get_exact_key(root_name, locals)

        if (existing_key is not None) and isinstance(existing_key, _DIKey):
            # Case 5.1: The name of the root module was imported via defer_imports in the same namespace.
            existing_key._di_add_submodule_name(name)
            result = locals[root_name]
        else:
            # Case 5.2: The name of the root module isn't present or wasn't created/placed by us.
            sub_names = _accumulate_dotted_parts(name, len(root_name) + 1)
            locals[_DIKey(root_name, (root_name, globals, locals, None), sub_names)] = locals.pop(root_name, None)
            result = _DIProxy(root_name)

    return result


class _DIContext:  # pyright: ignore [reportUnusedClass]
    """The context manager that replaces `defer_imports.until_use` after instrumentation."""

    __slots__ = ("_temp_deferred", "_original___import__")

    def __enter__(self, /) -> None:
        self._temp_deferred = _TempDeferred().__enter__()
        self._original___import__ = builtins.__import__
        builtins.__import__ = _deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        builtins.__import__ = self._original___import__
        self._temp_deferred.__exit__(*exc_info)


class NullContext:
    """A context manager within which imports occur lazily. Not re-entrant. Use via `defer_imports.until_use`.

    If defer_imports isn't set up properly, i.e. `import_hook().install()` is not called first elsewhere, this should be
    a no-op equivalent to `contextlib.nullcontext`.

    Raises
    ------
    SyntaxError
        If `defer_imports.until_use` is used improperly, e.g. it contains a wildcard import or a non-import statement.

    Notes
    -----
    This temporarily replaces `builtins.__import__`.
    """

    def __enter__(self, /) -> None:
        pass

    def __exit__(self, *exc_info: object) -> None:
        pass


until_use = NullContext


# endregion
