"""Sketch of a custom proxy-aware module dict.

An actual version would have to be written in C for various reasons, e.g. to guard critical sections.

Ref: https://discuss.python.org/t/pep-690-lazy-imports-again/19661/30
Ref: https://peps.python.org/pep-0690/#implementation
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, ValuesView
from typing import Any, final, overload

from typing_extensions import Self


def _resolve_proxy(proxy: Proxy) -> Any:
    # Placeholder.
    return proxy


_MISSING: Any = object()


@final
class Proxy:
    # Placeholder.
    ...


@final
class ProxyAwareValuesView(ValuesView[Any]):
    __slots__ = ()

    _mapping: ProxyAwareDict

    def __iter__(self, /) -> Iterator[Any]:
        for key, value in dict.items(self._mapping):  # pyright: ignore [reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(value, Proxy):
                self._mapping[key] = value = _resolve_proxy(value)  # noqa: PLW2901
            yield value


@final
class ProxyAwareItemsView(ItemsView[str, Any]):
    __slots__ = ()

    _mapping: ProxyAwareDict

    def __iter__(self, /) -> Iterator[tuple[str, Any]]:
        for key, value in dict.items(self._mapping):  # pyright: ignore [reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(value, Proxy):
                self._mapping[key] = value = _resolve_proxy(value)  # noqa: PLW2901
            yield (key, value)


@final
class ProxyAwareDict(dict[str, Any]):
    __slots__ = ()

    def __getitem__(self, key: str, /) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, Proxy):
            self[key] = value = _resolve_proxy(value)
        return value

    def __or__(self, value: object, /) -> Self:
        if not isinstance(value, dict):
            return NotImplemented
        return self.__class__(super().__or__(value))  # pyright: ignore [reportUnknownArgumentType]

    def __ror__(self, value: object, /) -> Self:
        if not isinstance(value, dict):
            return NotImplemented
        return self.__class__(super().__ror__(value))  # pyright: ignore [reportUnknownArgumentType]

    def get(self, key: str, default: object = None, /) -> Any:
        value = super().get(key, default)
        if isinstance(value, Proxy):
            value = _resolve_proxy(value)
        return value

    @overload
    def pop(self, key: str, /) -> Any: ...
    @overload
    def pop(self, key: str, default: Any, /) -> Any: ...
    def pop(self, key: str, default: Any = _MISSING, /) -> Any:  # pyright: ignore [reportIncompatibleMethodOverride]
        value = super().pop(key) if (default is _MISSING) else super().pop(key, default)
        if isinstance(value, Proxy):
            value = _resolve_proxy(value)
        return value

    def popitem(self, /) -> tuple[str, Any]:
        key, value = super().popitem()
        if isinstance(value, Proxy):
            value = _resolve_proxy(value)
        return (key, value)

    def copy(self, /) -> Self:
        return self.__class__(super().copy())

    def values(self, /) -> ProxyAwareValuesView:  # pyright: ignore [reportIncompatibleMethodOverride]
        return ProxyAwareValuesView(self)

    def items(self, /) -> ProxyAwareItemsView:  # pyright: ignore [reportIncompatibleMethodOverride]
        return ProxyAwareItemsView(self)
