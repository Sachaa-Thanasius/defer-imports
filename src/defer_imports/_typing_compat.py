# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""A __getattr__-based lazy import shim for typing- and annotation-related symbols."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec


__all__ = (
    # -- from collections.abc
    "Callable",
    "Generator",
    "Iterable",
    "MutableMapping",
    "Sequence",
    # -- from typing
    "Any",
    "Final",
    "Optional",
    "Union",
    # -- from types
    "CodeType",
    "ModuleType",
    # -- from importlib.abc
    "Loader",
    # -- from os
    "PathLike",
    # -- imported with fallbacks
    "ReadableBuffer",
    "Self",
    "TypeAlias",
    "TypeGuard",
    # -- needs import for definition
    "T",
    "AcceptsInput",
    "PathEntryFinderProtocol",
    # -- pure definition
    "final",
)

TYPE_CHECKING = False

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


def __getattr__(name: str) -> object:  # noqa: PLR0911, PLR0912, PLR0915
    # ---- Pure imports
    if name in {"Callable", "Generator", "Iterable", "MutableMapping", "Sequence"}:
        global Callable, Generator, Iterable, MutableMapping, Sequence

        from collections.abc import Callable, Generator, Iterable, MutableMapping, Sequence

        return globals()[name]

    if name in {"Any", "Final", "Optional", "Union"}:
        global Any, Final, Optional, Union

        from typing import Any, Final, Optional, Union

        return globals()[name]

    if name in {"CodeType", "ModuleType"}:
        global CodeType, ModuleType

        from types import CodeType, ModuleType

        return globals()[name]

    if name == "Loader":
        global Loader

        from importlib.abc import Loader

        return globals()[name]

    if name == "PathLike":
        global PathLike

        from os import PathLike

        return globals()[name]

    # ---- Imports with fallbacks
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

    # ---- Composed types/values with imports involved
    if name == "T":
        global T

        from typing import TypeVar

        T = TypeVar("T")
        return globals()[name]

    if name == "AcceptsInput":
        global AcceptsInput

        from typing import Protocol

        class AcceptsInput(Protocol):
            def __call__(self, prompt: str = "") -> str: ...

        return globals()[name]

    if name == "PathEntryFinderProtocol":
        global PathEntryFinderProtocol

        from typing import Protocol

        class PathEntryFinderProtocol(Protocol):
            # Copied from _typeshed.importlib.
            def find_spec(self, fullname: str, target: ModuleType | None = ..., /) -> ModuleSpec | None: ...

        return globals()[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_initial_global_names = tuple(globals())


def __dir__() -> list[str]:
    return list(set(_initial_global_names + __all__))
