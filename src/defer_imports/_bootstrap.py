# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A shim for typing- and annotation-related symbols, implemented with several other lazy import mechanisms."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec


__all__ = (
    "lazy_import_module",
    "lazy_loader_context",
    "T",
    "PathEntryFinderProtocol",
)

TYPE_CHECKING = False


def lazy_import_module(name: str, package: typing.Optional[str] = None) -> types.ModuleType:
    """Lazily import a module via ``importlib.util.LazyLoader``.

    The "package" argument is required when performing a relative import.

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

    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
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

    This operates using ``importlib.util.LazyLoader``. Consequentially, it only works on non-from non-submodule import
    statements.
    """

    def __enter__(self) -> None:
        if _LAZY_FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _LAZY_FINDER)

    def __exit__(self, *exc_info: object) -> None:
        try:
            sys.meta_path.remove(_LAZY_FINDER)
        except ValueError:
            pass


lazy_loader_context = _LazyLoaderContext()


def __getattr__(name: str) -> typing.Any:
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

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_initial_global_names = tuple(globals())


def __dir__() -> list[str]:
    return list(set(_initial_global_names + __all__))
