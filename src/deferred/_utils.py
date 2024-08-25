import sys


if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from itertools import tee

    def pairwise(iterable) -> zip[tuple[object, object]]:  # pyright: ignore
        """Pairwise recipe copied from itertools.

        Examples
        --------
        >>> list(pairwise("ABCDEFG"))
        [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')]
        """

        a, b = tee(iterable)  # pyright: ignore
        next(b, None)  # pyright: ignore
        return zip(a, b)  # pyright: ignore


TYPING = False

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


class HasLocationAttributes:
    """Placeholder for protocol representing an ast node's location attributes."""


__all__ = ("TYPING", "StrPath", "ReadableBuffer", "final", "Final", "HasLocationAttributes", "pairwise")
