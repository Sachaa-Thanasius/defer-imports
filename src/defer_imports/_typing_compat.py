# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A shim for typing- and annotation-related symbols, implemented with several other lazy import mechanisms."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec


__all__ = (
    # -- __getattr__-based types or values
    "Callable",
    "Generator",
    "Iterable",
    "MutableMapping",
    "Sequence",
    "Loader",
    "ReadableBuffer",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "T",
    "PathEntryFinderProtocol",
    # -- pure definition
    "final",
    # -- LazyLoader-based helpers
    "lazy_import_module",
    "lazy_loader_context",
)

TYPE_CHECKING = False


# ============================================================================
# region -------- importlib.util.LazyLoader-based laziness --------
# ============================================================================


def lazy_import_module(name: str, package: typing.Optional[str] = None) -> types.ModuleType:
    """Lazily import a name using ``importlib.util.LazyLoader``.

    Notes
    -----
    Slightly modified version of a recipe found in the Python 3.12 importlib docs.
    """

    absolute_name = importlib.util.resolve_name(name, package)
    try:
        return sys.modules[absolute_name]
    except KeyError:
        pass

    spec = importlib.util.find_spec(absolute_name)
    if spec is None:
        msg = f"No module named {name!r}"
        raise ModuleNotFoundError(msg)

    if spec.loader is None:
        msg = "missing loader"
        raise ImportError(msg, name=spec.name)

    spec.loader = loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


if TYPE_CHECKING:
    import types
    import typing
else:
    types = lazy_import_module("types")
    typing = lazy_import_module("typing")


class _LazyFinder:
    """An meta-path finder that wraps a found spec's loader with ``importlib.util.LazyLoader``."""

    def find_spec(
        self,
        fullname: str,
        path: typing.Sequence[str] | None,
        target: types.ModuleType | None = None,
        /,
    ) -> ModuleSpec:
        for finder in sys.meta_path:
            if finder != self:
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        else:
            msg = f"No module named {fullname!r}"
            raise ModuleNotFoundError(msg, name=fullname)

        if spec.loader is None:
            msg = "missing loader"
            raise ImportError(msg, name=spec.name)

        spec.loader = importlib.util.LazyLoader(spec.loader)
        return spec


_LAZY_FINDER = _LazyFinder()


class _LazyLoaderContext:
    """A context manager that temporarily modifies sys.meta_path to make enclosed import statements lazy.

    This operates using ``importlib.util.LazyLoader``. Consequentially, it only works on top-level non-submodule import
    statements.

    Examples
    --------
    These import statements should act lazy::

        with lazy_loader_context:
            import asyncio

    These should act eager::

        with lazy_loader_context:
            import asyncio.base_events
            from typing import Final
            from . import hello
    """

    def __enter__(self) -> None:
        if _LAZY_FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _LAZY_FINDER)

    def __exit__(self, *exc_info: object):
        try:
            sys.meta_path.remove(_LAZY_FINDER)
        except ValueError:
            pass


lazy_loader_context = _LazyLoaderContext()


# endregion


# ============================================================================
# region -------- module-level-__getattr__-based laziness --------
# ============================================================================


def __getattr__(name: str) -> typing.Any:  # noqa: PLR0911, PLR0912
    # region ---- Pure imports

    if name in {"Callable", "Generator", "Iterable", "MutableMapping", "Sequence"}:
        global Callable, Generator, Iterable, MutableMapping, Sequence

        from collections.abc import Callable, Generator, Iterable, MutableMapping, Sequence

        return globals()[name]

    if name == "Loader":
        global Loader

        from importlib.abc import Loader

        return globals()[name]

    # endregion

    # region ---- Imports with fallbacks

    if name == "ReadableBuffer":
        global ReadableBuffer

        if sys.version_info >= (3, 12):
            from collections.abc import Buffer as ReadableBuffer
        elif TYPE_CHECKING:
            from typing_extensions import Buffer as ReadableBuffer
        else:
            from typing import Union

            ReadableBuffer = Union[bytes, bytearray, memoryview]

        return globals()[name]

    if name == "Self":
        global Self

        if sys.version_info >= (3, 11):
            from typing import Self
        elif TYPE_CHECKING:
            from typing_extensions import Self
        else:

            class Self:
                """Placeholder for typing.Self."""

        return globals()[name]

    if name in {"TypeAlias", "TypeGuard"}:
        global TypeAlias, TypeGuard

        if sys.version_info >= (3, 10):
            from typing import TypeAlias, TypeGuard
        elif TYPE_CHECKING:
            from typing_extensions import TypeAlias, TypeGuard
        else:

            class TypeAlias:
                """Placeholder for typing.TypeAlias."""

            class TypeGuard:
                """Placeholder for typing.TypeGuard."""

        return globals()[name]

    # endregion

    # region ---- Composed types/values with imports involved

    if name == "T":
        global T

        T = typing.TypeVar("T")

        return globals()[name]

    if name == "PathEntryFinderProtocol":
        global PathEntryFinderProtocol

        class PathEntryFinderProtocol(typing.Protocol):
            # Copied from _typeshed.importlib.
            def find_spec(self, fullname: str, target: types.ModuleType | None = ..., /) -> ModuleSpec | None: ...

        return globals()[name]

    # endregion

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_initial_global_names = tuple(globals())


def __dir__() -> list[str]:
    return list(set(_initial_global_names + __all__))


# endregion


if TYPE_CHECKING:
    from typing import final
else:

    def final(f: object) -> object:
        """Decorator to indicate final methods and final classes.

        Slightly modified version of typing.final to avoid importing from typing at runtime.
        """

        try:
            f.__final__ = True  # pyright: ignore # Runtime attribute assignment
        except (AttributeError, TypeError):  # pragma: no cover
            # Skip the attribute silently if it is not writable.
            # AttributeError: if the object has __slots__ or a read-only property
            # TypeError: if it's a builtin class
            pass
        return f
