"""A library that implements PEP 690–esque lazy imports in pure Python."""

__version__ = "1.0.0.dev0"

__all__ = ("until_use",)


class _DummyContext:
    """A placeholder context manager that does nothing on its own.

    Should not be manually constructed: use through `defer_imports.until_use`.
    """

    def __enter__(self, /) -> None:
        pass

    def __exit__(self, *_dont_care: object) -> None:
        pass


until_use = _DummyContext()
