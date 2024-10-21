# pyright: reportUnusedImport=none
"""Simple benchark script for comparing the import time of the Python standard library when using regular imports,
defer_imports-influence imports, and slothy-influenced imports.

The sample scripts being imported are generated with bench/generate_samples.py.
"""

import platform
import shutil
import sys
import time
from pathlib import Path

import defer_imports


class CatchTime:
    """A context manager that measures the time taken to execute its body."""

    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object):
        self.elapsed = time.perf_counter() - self.elapsed


def bench_regular() -> float:
    with CatchTime() as ct:
        import bench.sample_regular
    return ct.elapsed


def bench_slothy() -> float:
    with CatchTime() as ct:
        import bench.sample_slothy
    return ct.elapsed


def bench_defer_imports_local() -> float:
    with CatchTime() as ct:
        hook_ctx = defer_imports.install_import_hook()
        import bench.sample_defer_local
    hook_ctx.uninstall()
    return ct.elapsed


def bench_defer_imports_global() -> float:
    with CatchTime() as ct:
        hook_ctx = defer_imports.install_import_hook(apply_all=True)
        import bench.sample_defer_global
    hook_ctx.uninstall()
    return ct.elapsed


def remove_pycaches() -> None:
    """Remove all cached Python bytecode files from the current working directory."""

    for dir_ in Path().rglob("__pycache__"):
        shutil.rmtree(dir_)

    for file in Path().rglob("*.py[co]"):
        file.unlink()


def pretty_print_results(results: dict[str, float], minimum: float) -> None:
    """Format and print results as an reST-style list table."""

    impl_header = "Implementation"
    impl_len = len(impl_header)
    impl_divider = "=" * impl_len

    version_header = "Version"
    version_len = len(version_header)
    version_divider = "=" * version_len

    benchmark_len = 22
    benchmark_header = "Benchmark".ljust(benchmark_len)
    benchmark_divider = "=" * benchmark_len

    time_len = 19
    time_header = "Time".ljust(time_len)
    time_divider = "=" * time_len

    divider = "  ".join((impl_divider, version_divider, benchmark_divider, time_divider))
    header = "  ".join((impl_header, version_header, benchmark_header, time_header))

    impl = platform.python_implementation().ljust(impl_len)
    version = f"{sys.version_info.major}.{sys.version_info.minor}".ljust(version_len)

    if sys.dont_write_bytecode:
        print("Run once with __pycache__ folders removed and bytecode caching disallowed")
    else:
        print("Run once with bytecode caching allowed")

    print()
    print(divider)
    print(header)
    print(divider)

    for bench_type, result in results.items():
        fmt_bench_type = bench_type.ljust(benchmark_len)
        fmt_result = f"{result:.5f}s ({result / minimum:.2f}x)".ljust(time_len)

        print(impl, version, fmt_bench_type, fmt_result, sep="  ")

    print(divider)


BENCH_FUNCS = {
    "regular": bench_regular,
    "slothy": bench_slothy,
    "defer_imports (local)": bench_defer_imports_local,
    "defer_imports (global)": bench_defer_imports_global,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exec-order",
        action="extend",
        nargs=4,
        choices=BENCH_FUNCS.keys(),
        type=str,
        help="The order in which the influenced (or not influenced) imports are run",
    )
    args = parser.parse_args()

    if sys.dont_write_bytecode:
        remove_pycaches()

    exec_order: list[str] = args.exec_order or list(BENCH_FUNCS)

    results = {type_: BENCH_FUNCS[type_]() for type_ in exec_order}
    minimum = min(results.values())

    pretty_print_results(results, minimum)


if __name__ == "__main__":
    raise SystemExit(main())
