import sys

import defer_imports.ast_rewrite


def profile_defer_imports_global() -> None:
    for _ in range(100):
        with defer_imports.ast_rewrite.import_hook(uninstall_after=True, apply_all=True):
            import bench.sample_defer_global  # pyright: ignore [reportUnusedImport]

        del sys.modules["bench.sample_defer_global"]


if __name__ == "__main__":
    profile_defer_imports_global()
