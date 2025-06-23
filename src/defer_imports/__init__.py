"""A library that implements PEP 690-esque lazy imports in pure Python."""

__version__ = "1.0.0.dev0"

__all__ = ("import_hook", "until_use")


from ._ast_rewrite import import_hook, until_use
