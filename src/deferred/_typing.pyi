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
    "ModuleType",
    "MutableMapping",
    "Optional",
    "ReadableBuffer",
    "Sequence",
    "StrPath",
    "Union",
    "final",
)

from collections.abc import Generator, Iterable, MutableMapping, Sequence
from types import CodeType, ModuleType
from typing import TYPE_CHECKING as TYPING, Any, Final, Optional, TypeVar, Union, final

from typing_extensions import Buffer as ReadableBuffer

T = TypeVar("T")  # noqa: PYI001

StrPath: TypeAlias = str | os.PathLike[str]
