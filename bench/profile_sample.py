import defer_imports._ast_rewrite


def bench_defer_imports_global() -> None:
    with defer_imports._ast_rewrite.install_import_hook(uninstall_after=True, apply_all=True):
        import bench.sample_defer_global


if __name__ == "__main__":
    bench_defer_imports_global()
