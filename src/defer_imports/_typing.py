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
    # Let's cache the return values in the global namespace to avoid repeat calls.

    # ---- Pure imports
    if name in {"Generator", "Iterable", "MutableMapping", "Sequence"}:
        global Generator, Iterable, MutableMapping, Sequence

        from collections.abc import Generator, Iterable, MutableMapping, Sequence

        return globals()[name]

    if name in {"Any", "Final", "Optional", "Union"}:
        global Any, Final, Optional, Union

        from typing import Any, Final, Optional, Union

        return globals()[name]

    if name in {"CodeType", "ModuleType"}:
        global CodeType, ModuleType

        from types import CodeType, ModuleType

        return globals()[name]

    # ---- Imports with fallbacks
    if name == "ReadableBuffer":
        global ReadableBuffer

        if sys.version_info >= (3, 12):
            from collections.abc import Buffer as ReadableBuffer
        else:
            from typing import Union

            ReadableBuffer = Union[bytes, bytearray, memoryview]

        return globals()[name]

    if name == "Self":
        global Self

        if sys.version_info >= (3, 11):
            from typing import Self
        else:

            class Self:
                """Placeholder for typing.Self."""

        return globals()[name]

    if name == "TypeAlias":
        global TypeAlias

        if sys.version_info >= (3, 10):
            from typing import TypeAlias
        else:

            class TypeAlias:
                """Placeholder for typing.TypeAlias."""

        return globals()[name]

    # ---- Composed types/values with imports involved
    if name == "StrPath":
        global StrPath

        import os
        from typing import Union

        StrPath = Union[str, os.PathLike[str]]
        return globals()[name]

    if name == "T":
        global T

        from typing import TypeVar

        T = TypeVar("T")
        return globals()[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_original_global_names = list(globals())


def __dir__() -> list[str]:
    # This will hopefully make potential debugging easier.
    return [*_original_global_names, *__all__]
