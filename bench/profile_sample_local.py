import sys

import defer_imports


def profile_defer_imports_local() -> None:
    for _ in range(100):
        with defer_imports.import_hook(uninstall_after=True):
            import bench.sample_defer_local  # pyright: ignore [reportUnusedImport]

        del sys.modules["bench.sample_defer_local"]


if __name__ == "__main__":
    profile_defer_imports_local()
