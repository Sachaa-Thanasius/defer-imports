import os
from collections.abc import Generator, Iterable

__all__ = (
    "TYPING",
    "StrPath",
    "ReadableBuffer",
    "final",
    "Final",
    "CodeType",
    "sliding_window",
    "calc_package",
    "resolve_name",
)

from types import CodeType
from typing import TYPE_CHECKING as TYPING, Final, TypeVar, final

from typing_extensions import Buffer as ReadableBuffer, TypeAlias

_T = TypeVar("_T")

StrPath: TypeAlias = str | os.PathLike[str]

def sliding_window(iterable: Iterable[_T], n: int) -> Generator[tuple[_T, ...]]: ...
def calc_package(globals: dict[str, object]) -> str | None: ...  # noqa: A002
def resolve_name(name: str, package: str, level: int) -> str: ...
