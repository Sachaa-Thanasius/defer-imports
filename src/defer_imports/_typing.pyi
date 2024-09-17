# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

__all__ = (
    # collections.abc
    "Callable",
    "Generator",
    "Iterable",
    "MutableMapping",
    "Sequence",
    # typing
    "Any",
    "Final",
    "Optional",
    "Union",
    # types
    "CodeType",
    "ModuleType",
    # os
    "PathLike",
    # importlib.abc
    "Loader",
    # import with fallbacks
    "ReadableBuffer",
    "Self",
    "TypeAlias",
    "TypeGuard",
    # import and then defined
    "T",
    "PathEntryFinderProtocol",
    # actually defined
    "final",
)

from collections.abc import Callable, Generator, Iterable, MutableMapping, Sequence
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from os import PathLike
from types import CodeType, ModuleType
from typing import Any, Final, Optional, Protocol, TypeVar, Union, final

from typing_extensions import Buffer as ReadableBuffer, Self, TypeAlias, TypeGuard

T = TypeVar("T")  # noqa: PYI001

# Copied from _typeshed.importlib.
class PathEntryFinderProtocol(Protocol):
    def find_spec(self, fullname: str, target: ModuleType | None = ..., /) -> ModuleSpec | None: ...
