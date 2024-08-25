import os
import sys

__all__ = (
    "TYPING",
    "StrPath",
    "ReadableBuffer",
    "final",
    "Final",
    "CodeType",
    "pairwise",
    "calc_package",
    "resolve_name",
)

from types import CodeType
from typing import TYPE_CHECKING as TYPING, Final, final

from typing_extensions import Buffer as ReadableBuffer, TypeAlias

StrPath: TypeAlias = str | os.PathLike[str]

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from collections.abc import Iterable
    from typing import TypeVar

    _T = TypeVar("_T")

    def pairwise(iterable: Iterable[_T]) -> zip[tuple[_T, _T]]: ...

def calc_package(globals: dict[str, object]) -> str | None: ...  # noqa: A002
def resolve_name(name: str, package: str, level: int) -> str: ...
