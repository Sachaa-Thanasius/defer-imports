# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

import os

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

from collections.abc import Generator, Iterable, MutableMapping, Sequence
from types import CodeType, ModuleType
from typing import Any, Final, Optional, TypeVar, Union, final

from typing_extensions import Buffer as ReadableBuffer, Self, TypeAlias

T = TypeVar("T")  # noqa: PYI001

StrPath: TypeAlias = str | os.PathLike[str]
