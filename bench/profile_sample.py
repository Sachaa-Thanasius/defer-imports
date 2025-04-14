import defer_imports._ast_rewrite


def profile_defer_imports_global() -> None:
    with defer_imports._ast_rewrite.install_import_hook(uninstall_after=True, apply_all=True):
        import bench.sample_defer_global  # pyright: ignore [reportUnusedImport]


if __name__ == "__main__":
    profile_defer_imports_global()
