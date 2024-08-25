import sys
import warnings


TYPING = False
"""Constant that is True at type-checking time but False at runtime. Similar to typing.TYPE_CHECKING."""

StrPath = str
ReadableBuffer = "bytes | bytearray | memoryview"


def final(f: object) -> object:
    """Placeholder for typing.final. Copied from typing with minimal changes."""

    try:
        f.__final__ = True  # pyright: ignore # Runtime attribute assignment
    except (AttributeError, TypeError):
        # Skip the attribute silently if it is not writable.
        # AttributeError: if the object has __slots__ or a read-only property
        # TypeError: if it's a builtin class
        pass
    return f


class Final:
    """Placeholder for typing.Final."""


CodeType = type(final.__code__)


if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from itertools import tee

    def pairwise(iterable) -> zip[tuple[object, object]]:
        """Pairwise recipe copied from itertools.

        Examples
        --------
        >>> list(pairwise("ABCDEFG"))
        [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')]
        """

        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def calc_package(globals: dict[str, object]):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    Slightly modified version of importlib._bootstrap._calc___package__.
    """

    package = globals.get("__package__")
    spec = globals.get("__spec__")
    if package is not None:
        if spec is not None and package != spec.parent:
            warnings.warn(
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",
                ImportWarning,
                stacklevel=3,
            )
        return package
    elif spec is not None:
        return spec.parent
    else:
        warnings.warn(
            "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__",
            ImportWarning,
            stacklevel=3,
        )
        package = globals["__name__"]
        if "__path__" not in globals:
            package = package.rpartition(".")[0]
    return package


def resolve_name(name: str, package: str, level: int) -> str:
    """Resolve a relative module name to an absolute one.

    Slightly modified version of importlib._bootstrap._resolve_name.
    """

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


class HasLocationAttributes:
    """Placeholder for protocol representing an ast node's location attributes."""


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
    "HasLocationAttributes",
)
