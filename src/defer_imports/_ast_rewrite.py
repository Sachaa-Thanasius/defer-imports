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

import builtins
import contextvars
import io
import os
import sys
import types
from importlib.machinery import BYTECODE_SUFFIXES, SOURCE_SUFFIXES, FileFinder, ModuleSpec, SourceFileLoader

from . import __version__, _lazy_load


# PYUPDATE: py3.14 - importlib.abc might be cheap enough to eagerly import.
with _lazy_load.until_module_use:
    import ast
    import importlib.abc  # noqa: F401 # Used in a type alias.
    import threading
    import tokenize
    import typing as t
    import warnings


# ============================================================================
# region -------- Compatibility shims --------
#
# Definitions for symbols whose availability depends on the Python version.
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


__all__ = ("install_import_hook",)


_ASTLocation: TypeAlias = "dict[t.Literal['lineno', 'col_offset', 'end_lineno', 'end_col_offset'], int]"
_SourceData: TypeAlias = "t.Union[ReadableBuffer, str]"
_LoaderInit: TypeAlias = "t.Callable[[str, str], importlib.abc.Loader]"


# ============================================================================
# region -------- Vendored helpers --------
#
# These are adapted from importlib._bootstrap and
# importlib._bootstrap_external to avoid depending on private APIs and allow
# changes.
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
def _sanity_check(name: str, package: t.Optional[str], level: int) -> None:  # pragma: no cover (tested in stdlib)
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


# endregion


# ============================================================================
# region -------- AST transformer --------
# ============================================================================


_BYTECODE_HEADER = f"defer_imports{__version__}".encode()
"""Custom header for defer_imports-instrumented bytecode files. Differs for every version."""


# NOTE: We make our generated variables more hygienic by prefixing their names with "_@di_".
# This has a few benefits:
#
# 1. The variables can't be accidentally accessed by regular user code (that doesn't programmatically access namespaces
#    via locals/globals/vars), since "@" isn't a valid symbol in identifiers. pytest does something similar.
# 2. The variables are somewhat namespaced with "di" in case other generated code is added by third parties.
#    pytest does something similiar.
# 3. The variables start with an underscore so that code that programmatically accesses the global namespace
#    during module execution, but avoids symbols starting with an underscore, won't pick it up.
#     - An example of a common pattern in the standard library that meets this criteria:
#       __all__ = [name for name in globals() if name[:1] != "_"]  # noqa: ERA001
#    TODO: Do we actually want to do this one? Surely it's bound to backfire in converse use cases.
_HYGIENE_PREFIX = "_@di_"

_INTERNALS_NAMES = ("_DeferredImportKey", "_DeferredImportProxy", "_actual_until_use")
_INTERNALS_ASNAMES = tuple(f"{_HYGIENE_PREFIX}{name}" for name in _INTERNALS_NAMES)
_KEY_CLS_NAME, _PROXY_CLS_NAME, _ACTUAL_CTX_NAME = _INTERNALS_ASNAMES
_TEMP_PROXY_NAME = f"{_HYGIENE_PREFIX}temp_proxy"
_LOCAL_NS_NAME = f"{_HYGIENE_PREFIX}local_ns"


def _is_until_use_node(node: ast.With) -> bool:
    """Check if the node matches ``with defer_imports.until_use: ...``."""

    return len(node.items) == 1 and (
        isinstance(node.items[0].context_expr, ast.Attribute)
        and isinstance(node.items[0].context_expr.value, ast.Name)
        and node.items[0].context_expr.value.id == "defer_imports"
        and node.items[0].context_expr.attr == "until_use"
    )


class _DeferredInstrumenter:
    """AST transformer that instruments imports within ``with defer_imports.until_use: ...`` blocks.

    The results of those imports will be assigned to custom keys in the local namespace.

    Notes
    -----
    This doesn't subclass `ast.NodeTransformer` but instead vendors its logic to avoid the upfront import cost of `ast`.
    """

    # PYUPDATE: Ensure visit and generic_visit are consistent with upstream, aside from our customizations.
    # PYUPDATE: py3.14 - Take advantage of better defaults for node parameters, e.g. ast.Load() for ctx,
    # late-initialized empty lists for parameters that take lists, etc.

    def __init__(self, source: _SourceData, filepath: _ModulePath = "<unknown>", *, module_level: bool = False) -> None:
        self.source: _SourceData = source
        self.filepath: _ModulePath = filepath
        self.rewrite_whole_module: bool = module_level

        self.escape_hatch_depth: int = 0
        self.did_any_instrumentation: bool = False

    def visit(self, node: ast.AST) -> t.Any:
        """Visit a node."""

        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def _wrap_import_stmts(self, nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> ast.With:
        """Wrap a list of import nodes with a `defer_imports.until_use` block and instrument them."""

        lineno = nodes[0].lineno
        loc: _ASTLocation = {"lineno": lineno, "col_offset": 0, "end_lineno": lineno, "end_col_offset": 0}
        with_items = [ast.withitem(ast.Name(_ACTUAL_CTX_NAME, ctx=ast.Load(), **loc))]
        return ast.With(items=with_items, body=self._substitute_import_keys(nodes), **loc)

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
                            if (
                                # Only for import nodes without wildcards.
                                isinstance(value, (ast.Import, ast.ImportFrom))
                                and value.names[0].name != "*"
                                # Only outside of escape hatch blocks.
                                and (self.escape_hatch_depth == 0)
                            ):
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

                if self.rewrite_whole_module and (start_idx > 0):
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
        text = ast.get_source_segment(source, node, padded=True)
        filepath = self.filepath if isinstance(self.filepath, (str, bytes, os.PathLike)) else bytes(self.filepath)
        context = (os.fsdecode(filepath), node.lineno, node.col_offset + 1, text)
        if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
            end_col_offset = (node.end_col_offset + 1) if (node.end_col_offset is not None) else None
            context += (node.end_lineno, end_col_offset)
        return context

    @staticmethod
    def _create_import_name_replacement(name: str, loc: _ASTLocation) -> ast.If:
        """Create an AST for changing the name of a variable in locals if the variable is a defer_imports proxy.

        The resulting node is roughly equivalent to the following code if unparsed::

            if {name}.__class__ is _DeferredImportProxy:
                temp_proxy = local_ns[_DeferredImportKey("{name}", temp_proxy)] = local_ns.pop("{name}")
        """

        if "." in name:
            name = name.partition(".")[0]

        return ast.If(
            test=ast.Compare(
                left=ast.Attribute(ast.Name(name, ctx=ast.Load(), **loc), attr="__class__", ctx=ast.Load(), **loc),
                ops=[ast.Is()],
                comparators=[ast.Name(_PROXY_CLS_NAME, ctx=ast.Load(), **loc)],
                **loc,
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(_TEMP_PROXY_NAME, ctx=ast.Store(), **loc),
                        ast.Subscript(
                            value=ast.Name(_LOCAL_NS_NAME, ctx=ast.Load(), **loc),
                            slice=ast.Call(
                                func=ast.Name(_KEY_CLS_NAME, ctx=ast.Load(), **loc),
                                args=[ast.Constant(name, **loc), ast.Name(_TEMP_PROXY_NAME, ctx=ast.Load(), **loc)],
                                keywords=[],
                                **loc,
                            ),
                            ctx=ast.Store(),
                            **loc,
                        ),
                    ],
                    value=ast.Call(
                        func=ast.Attribute(ast.Name(_LOCAL_NS_NAME, ctx=ast.Load(), **loc), "pop", ast.Load(), **loc),
                        args=[ast.Constant(name, **loc)],
                        keywords=[],
                        **loc,
                    ),
                    **loc,
                )
            ],
            orelse=[],
            **loc,
        )

    def _validate_until_use_body(self, nodes: list[ast.stmt]) -> list[t.Union[ast.Import, ast.ImportFrom]]:
        """Validate that the statements within a `defer_imports.until_use` block are instrumentable."""

        for node in nodes:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                msg = "with defer_imports.until_use blocks must only contain import statements"
                raise SyntaxError(msg, self._get_error_context(node))  # noqa: TRY004 # Syntax error displays better.

            if any(alias.name == "*" for alias in node.names):
                msg = "import * not allowed in with defer_imports.until_use blocks"
                raise SyntaxError(msg, self._get_error_context(node))

        # We won't mutate the list from outside the function to invalidate our type guard, scout's honor.
        return nodes  # pyright: ignore [reportReturnType]

    def _substitute_import_keys(self, import_nodes: list[t.Union[ast.Import, ast.ImportFrom]]) -> list[ast.stmt]:
        """Instrument a *non-empty* list of imports.

        Raises
        ------
        SyntaxError
            If any of the given nodes are not imports or are wildcard imports.
        """

        self.did_any_instrumentation = True

        # Create a node location set that can be re-used, passed around, and modified.
        # However, it must not be modified outside of this function.
        lineno = import_nodes[0].lineno
        loc: _ASTLocation = {"lineno": lineno, "col_offset": 0, "end_lineno": lineno, "end_col_offset": 0}

        new_nodes: list[ast.stmt] = []

        # Start with some helper variables.
        # A reference to locals() to avoid calling locals() repeatedly.
        locals_call = ast.Call(func=ast.Name("locals", ctx=ast.Load(), **loc), args=[], keywords=[], **loc)
        local_ns = ast.Assign(targets=[ast.Name(_LOCAL_NS_NAME, ctx=ast.Store(), **loc)], value=locals_call, **loc)
        new_nodes.append(local_ns)

        # A reference to the current proxy being "fixed".
        temp_proxy_name: list[ast.expr] = [ast.Name(_TEMP_PROXY_NAME, ctx=ast.Store(), **loc)]
        temp_proxy = ast.Assign(targets=temp_proxy_name, value=ast.Constant(None, **loc), **loc)
        new_nodes.append(temp_proxy)

        # Add the imports + namespace adjustments.
        for node in import_nodes:
            new_nodes.append(node)
            loc["lineno"] = loc["end_lineno"] = node.lineno
            new_nodes.extend(
                self._create_import_name_replacement(alias.asname or alias.name, loc) for alias in node.names
            )

        # Clean up the helper variables via deletion.
        loc["lineno"] = loc["end_lineno"] = import_nodes[-1].lineno
        new_nodes.append(
            ast.Delete([ast.Name(name, ctx=ast.Del(), **loc) for name in (_TEMP_PROXY_NAME, _LOCAL_NS_NAME)], **loc)
        )

        return new_nodes

    def visit_With(self, node: ast.With) -> ast.AST:
        """Check that ``with defer_imports.until_use: ...`` blocks are valid and if so, hook all imports within.

        Raises
        ------
        SyntaxError:
            If a `defer_imports.until_use` block contains a statement that isn't an import or contains a wildcard
            import.
        """

        if not _is_until_use_node(node):
            return self._visit_eager_import_block(node)

        # Replace the dummy context manager with the one that will actually replace __import__.
        new_ctx_expr = ast.copy_location(ast.Name(_ACTUAL_CTX_NAME, ctx=ast.Load()), node.items[0].context_expr)
        node.items[0].context_expr = new_ctx_expr

        node.body = self._substitute_import_keys(self._validate_until_use_body(node.body))
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Insert imports and cleanup necessary to make `defer_imports.until_use` work properly.

        If `defer_imports.until_use` isn't actually used, do nothing.

        Imports will be inserted after the module docstring and after `__future__` imports.
        """

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

        # Then, add necessary defer_imports import.
        lineno = position + 1
        loc: _ASTLocation = {"lineno": lineno, "col_offset": 0, "end_lineno": lineno, "end_col_offset": 0}
        aliases = [ast.alias(name, asname, **loc) for name, asname in zip(_INTERNALS_NAMES, _INTERNALS_ASNAMES)]
        node.body.insert(position, ast.ImportFrom(module=__spec__.name, names=aliases, level=0, **loc))

        # Finally, clean up the namespace via deletion.
        loc["lineno"] = loc["end_lineno"] = node.body[-1].lineno
        names: list[ast.expr] = [ast.Name(asname, ctx=ast.Del(), **loc) for asname in _INTERNALS_ASNAMES]
        node.body.append(ast.Delete(targets=names, **loc))

        return node


# endregion


# ============================================================================
# region -------- importlib import hooks --------
#
# The module loader, module finder, and import hook API to attach those to the
# import system.
# ============================================================================


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

    # NOTE: We're purposefully not supporting data being an AST object.
    # NOTE: The signatures of SourceFileLoader.source_to_code at runtime and in typeshed aren't currently consistent.
    # ref: https://github.com/python/typeshed/issues/13881
    # ref: https://github.com/python/typeshed/pull/13880
    def source_to_code(self, data: _SourceData, path: _ModulePath, *, _optimize: int = -1) -> types.CodeType:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Compile `data` into a code object, but not before potentially instrumenting it.

        Parameters
        ----------
        data: _SourceData
            A string or buffer type that `compile()` supports.
        """

        if data:
            orig_tree = ast.parse(data, path, "exec")
            if self.defer_whole_module or any(
                (isinstance(node, ast.With) and _is_until_use_node(node)) for node in ast.walk(orig_tree)
            ):
                instrumenter = _DeferredInstrumenter(data, path, module_level=self.defer_whole_module)
                new_tree = instrumenter.visit(orig_tree)
                return super().source_to_code(new_tree, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

        return super().source_to_code(data, path, _optimize=_optimize)  # pyright: ignore # noqa: PGH003

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the module."""

        # This state is needed for self.source_to_code().
        if (spec := module.__spec__) is not None and spec.loader_state is not None:
            self.defer_whole_module = spec.loader_state["defer_whole_module"]

        return super().exec_module(module)


class _DeferredFileFinder(FileFinder):
    def __repr__(self):
        return f"{self.__class__.__name__}({self.path!r})"

    @staticmethod
    def _get_loader_class(config: _DeferConfig) -> _LoaderInit:
        if config.loader_class is None:
            return _DeferredFileLoader
        else:
            return config.loader_class

    @staticmethod
    def _is_full_module_rewrite(fullname: str, config: _DeferConfig) -> bool:
        """Determine whether all imports should be instrumented instead of just `until_use`-encapsulated imports."""

        # NOTE: This could be written as one boolean expression, but currently, splitting it out makes the hierarchy
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

        If a spec is found, its loader may be overriden and some state may be passed on via
        `ModuleSpec.loader_state` [1]_ [2]_.

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

        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            # NOTE: We're locking in defer_imports configuration for this module between finding it and loading it.
            # However, it's possible to delay getting the configuration until module execution. Not sure what's best.
            config = _current_defer_config.get(None)

            if config is not None:
                defer_whole_module = self._is_full_module_rewrite(fullname, config)
                loader_class = self._get_loader_class(config)
            else:
                defer_whole_module = False
                loader_class = _DeferredFileLoader

            spec.loader = loader_class(spec.loader.name, spec.loader.path)
            spec.loader_state = {"defer_whole_module": defer_whole_module}

        return spec


_LOADER_DETAILS = (_DeferredFileLoader, SOURCE_SUFFIXES)
_PATH_HOOK = _DeferredFileFinder.path_hook(_LOADER_DETAILS)


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
    state and uninstall defer_import's import path hook.
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

        if (tok := getattr(self, "_tok", None)) is not None:
            _current_defer_config.reset(tok)
            del self._tok

    def uninstall(self) -> None:
        """Attempt to remove the custom path hook in `sys.path_hook` and undo monkeypatching of cached finders."""

        try:
            sys.path_hooks.remove(_PATH_HOOK)
        except ValueError:
            pass

        for finder in sys.path_importer_cache.values():
            if (finder is not None) and (finder.__class__ is _DeferredFileFinder):
                finder.__class__ = FileFinder


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
        have an initialization signature matching ``(fullname: str, path: str) -> Loader``.

    Returns
    -------
    ImportHookContext
        A object that can be used to reset the import hook's configuration to its previous state or uninstall it, either
        automatically by using it as a context manager or manually using its rest() and uninstall methods.
    """

    if isinstance(module_names, str):
        msg = "module_names should be a sequence of strings, not a string."
        raise TypeError(msg)

    if _PATH_HOOK not in sys.path_hooks:
        file_finder_index = next(
            (i for i, hook in enumerate(sys.path_hooks) if hook.__name__ == "path_hook_for_FileFinder"),
            len(sys.path_hooks),
        )
        sys.path_hooks.insert(file_finder_index, _PATH_HOOK)

        # HACK: We do some monkeypatching here so that the cached finders for sys.path entries use the right finder
        # class. This should be safe; _DeferredFinder is a subclass of FileFinder and has the same instance state.
        #
        # Alternatives:
        # - Create and insert a new PathFinder subclass into sys.meta_path, or patch the existing one.
        #   That would be a bigger monkeypatch, but it's the route that typeguard takes.
        # - Delete the sys.path_importer_cache entries instead of monkeypatching them.
        #   This is the docs's recommendation and is technically more correct, but it causes a big slowdown on startup.
        #
        # TODO: Determine if this is *still* overkill. It would be more useful in the .pth file case.
        for finder in sys.path_importer_cache.values():
            if (finder is not None) and (finder.__class__ is FileFinder):
                finder.__class__ = _DeferredFileFinder

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
        if (is_eq := super().__eq__(value)) is not True:
            return is_eq

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
        _is_deferred_tok = _is_deferred.set(True)
        try:
            namespace[key] = namespace.pop(key)
        finally:
            _is_deferred.reset(_is_deferred_tok)

        # 3.3. Resolve any requested attribute access.
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def _deferred___import__(
    name: str,
    globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    locals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    fromlist: t.Optional[t.Sequence[str]] = (),
    level: int = 0,
) -> t.Any:
    """An limited replacement for `__import__` that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()
    globals = globals if (globals is not None) else {}  # noqa: A001
    locals = locals if (locals is not None) else {}  # noqa: A001

    package = _calc___package__(globals) if (level != 0) else None
    _sanity_check(name, package, level)

    if level > 0:
        # _sanity_check ensures this.
        assert package is not None
        name = _resolve_name(name, package, level)
        level = 0

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
            1. It contains a statement that isn't an import.
            2. It contains a wildcard import.

    Notes
    -----
    As part of its implementation, this temporarily replaces `builtins.__import__`.
    """

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = _is_deferred.set(True)
        self._import_ctx_token = _original_import.set(builtins.__import__)
        builtins.__import__ = _deferred___import__

    def __exit__(self, *_dont_care: object) -> None:
        _original_import.reset(self._import_ctx_token)
        _is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = _original_import.get()


_actual_until_use: t.Final[_DeferredContext] = _DeferredContext()


# endregion
