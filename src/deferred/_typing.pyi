import os

from typing_extensions import TypeAlias

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
    "Union",
    "final",
)

from collections.abc import Generator, Iterable
from types import CodeType
from typing import TYPE_CHECKING as TYPING, Any, Final, Optional, TypeVar, Union, final

from typing_extensions import Buffer as ReadableBuffer

T = TypeVar("T")  # noqa: PYI001

StrPath: TypeAlias = str | os.PathLike[str]
