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
    # import with fallbacks
    "ReadableBuffer",
    "Self",
    "TypeAlias",
    "TypeGuard",
    # import and then defined
    "T",
    # actually defined
    "final",
)

from collections.abc import Callable, Generator, Iterable, MutableMapping, Sequence
from os import PathLike
from types import CodeType, ModuleType
from typing import Any, Final, Optional, TypeVar, Union, final

from typing_extensions import Buffer as ReadableBuffer, Self, TypeAlias, TypeGuard

T = TypeVar("T")  # noqa: PYI001
