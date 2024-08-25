# pyright: reportUnsupportedDunderAll=none
# ruff: noqa: F822
"""A __getattr__-based lazy import shim for typing- and annotation-related symbols."""

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
    # Let's cache the return values in the global namespace to avoid subsequent calls to __getattr__ if possible.

    if name == "T":
        from typing import TypeVar

        globals()["T"] = T = TypeVar("T")  # pyright: ignore [reportGeneralTypeIssues]
        return T

    if name == "Any":
        from typing import Any

        globals()["Any"] = Any
        return Any

    if name == "CodeType":
        from types import CodeType

        globals()["CodeType"] = CodeType
        return CodeType

    if name == "Optional":
        from typing import Optional

        globals()["Optional"] = Optional
        return Optional

    if name == "Final":
        from typing import Final

        globals()["Final"] = Final
        return Final
    if name == "Generator":
        from collections.abc import Generator

        globals()["Generator"] = Generator
        return Generator

    if name == "Iterable":
        from collections.abc import Iterable

        globals()["Iterable"] = Iterable
        return Iterable

    if name == "ReadableBuffer":
        if sys.version_info >= (3, 12):
            from collections.abc import Buffer as ReadableBuffer
        else:
            from typing import Union

            ReadableBuffer = Union[bytes, bytearray, memoryview]

        globals()["ReadableBuffer"] = ReadableBuffer
        return ReadableBuffer

    if name == "StrPath":
        import os
        from typing import Union

        globals()["StrPath"] = StrPath = Union[str, os.PathLike[str]]
        return StrPath

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
