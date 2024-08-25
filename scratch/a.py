from deferred import defer_imports_until_use


with defer_imports_until_use:
    from .b import B

print(B)
