# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

# pyright: reportUnsupportedDunderAll=none

"""A __getattr__-based lazy import shim for typing- and annotation-related symbols."""

from __future__ import annotations

import sys


__all__ = (
    "Any",
    "CodeType",
    "Final",
    "Generator",
    "Iterable",
    "ModuleType",
    "MutableMapping",
    "Optional",
    "ReadableBuffer",
    "Self",
    "Sequence",
    "StrPath",
    "T",
    "TypeAlias",
    "Union",
    "final",
)


def final(f: object) -> object:
    """Decorator to indicate final methods and final classes.

    Slightly modified version of typing.final.
    """

    try:
        f.__final__ = True  # pyright: ignore # Runtime attribute assignment
    except (AttributeError, TypeError):  # pragma: no cover
        # Skip the attribute silently if it is not writable.
        # AttributeError: if the object has __slots__ or a read-only property
        # TypeError: if it's a builtin class
        pass
    return f


def __getattr__(name: str) -> object:  # noqa: PLR0911, PLR0912
    # Let's cache the return values in the global namespace to avoid subsequent calls to __getattr__ if possible.

    # ---- Pure imports
    if name in {"Generator", "Iterable", "MutableMapping", "Sequence"}:
        import collections.abc

        globals()[name] = res = getattr(collections.abc, name)
        return res

    if name in {"Any", "Final", "Optional", "Union"}:
        import typing

        globals()[name] = res = getattr(typing, name)
        return res

    if name in {"CodeType", "ModuleType"}:
        import types

        globals()[name] = res = getattr(types, name)
        return res

    # ---- Imports with fallbacks
    if name == "ReadableBuffer":
        if sys.version_info >= (3, 12):
            from collections.abc import Buffer as ReadableBuffer
        else:
            from typing import Union

            ReadableBuffer = Union[bytes, bytearray, memoryview]

        globals()[name] = ReadableBuffer
        return ReadableBuffer

    if name == "Self":
        if sys.version_info >= (3, 11):
            from typing import Self
        else:

            class Self:
                """Placeholder for typing.Self."""

        globals()[name] = Self
        return Self

    if name == "TypeAlias":
        if sys.version_info >= (3, 10):
            from typing import TypeAlias
        else:

            class TypeAlias:
                """Placeholder for typing.TypeAlias."""

        globals()[name] = TypeAlias
        return TypeAlias

    # ---- Composed types/values with imports involved
    if name == "StrPath":
        import os
        from typing import Union

        globals()[name] = StrPath = Union[str, os.PathLike[str]]
        return StrPath

    if name == "T":
        from typing import TypeVar

        globals()[name] = T = TypeVar("T")  # pyright: ignore [reportGeneralTypeIssues]
        return T

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_original_global_names = list(globals())


def __dir__() -> list[str]:
    # This will hopefully make potential debugging easier.
    return [*_original_global_names, *__all__]
