# pyright: reportUnsupportedDunderAll=none
# ruff: noqa: F822
"""Typing-related constructs that are annoying to import for one reason or another."""

import sys


__all__ = (
    "T",
    "TYPING",
    "Any",
    "CodeType",
    "Final",
    "Generator",
    "Iterable",
    "Optional",
    "ReadableBuffer",
    "StrPath",
    "final",
)

TYPING = False
"""Constant that is True at type-checking time but False at runtime. Similar to typing.TYPE_CHECKING."""


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


def __getattr__(name: str) -> object:  # noqa: PLR0911
    if name == "T":
        from typing import TypeVar

        return TypeVar("T")

    if name == "Any":
        from typing import Any

        return Any

    if name == "CodeType":
        from types import CodeType

        return CodeType

    if name == "Optional":
        from typing import Optional

        return Optional

    if name == "Final":
        from typing import Final

        return Final
    if name == "Generator":
        from collections.abc import Generator

        return Generator

    if name == "Iterable":
        from collections.abc import Iterable

        return Iterable

    if name == "ReadableBuffer":
        if sys.version_info >= (3, 12):
            from collections.abc import Buffer as ReadableBuffer
        else:
            from typing import Union

            ReadableBuffer = Union[bytes, bytearray, memoryview]

        return ReadableBuffer

    if name == "StrPath":
        import os
        from typing import Union

        return Union[str, os.PathLike[str]]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
