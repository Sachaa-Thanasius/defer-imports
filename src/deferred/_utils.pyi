import os
import sys
from typing import TYPE_CHECKING as TYPING, Final, Protocol, final

from typing_extensions import Buffer as ReadableBuffer, TypeAlias

if sys.version_info >= (3, 10):
    from itertools import pairwise
elif TYPING:
    from collections.abc import Iterable
    from typing import TypeVar

    _T = TypeVar("_T")

    def pairwise(iterable: Iterable[_T]) -> zip[tuple[_T, _T]]: ...

StrPath: TypeAlias = str | os.PathLike[str]

class HasLocationAttributes(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None

__all__ = ("TYPING", "StrPath", "ReadableBuffer", "final", "Final", "HasLocationAttributes", "pairwise")
