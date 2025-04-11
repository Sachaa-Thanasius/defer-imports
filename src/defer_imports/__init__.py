"""A library that implements PEP 690–esque lazy imports in pure Python."""

__version__ = "1.0.0.dev0"

__all__ = ("until_use",)


class _DummyContext:
    """A placeholder context manager that does nothing on its own. Must be used as ``with defer_imports.until_use: ...``.

    This serves as a marker that will be replaced by `defer_imports` machinery, if that is active, at import time.
    """

    def __enter__(self, /) -> None:
        pass

    def __exit__(self, *_dont_care: object) -> None:
        pass


until_use = _DummyContext()
