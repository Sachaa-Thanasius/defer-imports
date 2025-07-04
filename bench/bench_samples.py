# pyright: reportUnusedImport=none

"""Simple benchark script for comparing the import time of the Python standard library when using regular imports and
defer_imports-influence imports.

The sample scripts being imported were generated with bench/generate_samples.py.
"""

import argparse
import platform
import shutil
import sys
import time
import unittest.mock
from pathlib import Path

import defer_imports


class TimeCatcher:
    """A context manager that measures the time taken to execute its body."""

    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object):
        self.elapsed = time.perf_counter() - self.elapsed


def bench_regular() -> float:
    with TimeCatcher() as tc:
        import bench.sample_regular
    return tc.elapsed


def bench_defer_imports_local() -> float:
    with TimeCatcher() as tc:  # noqa: SIM117
        with defer_imports.import_hook(uninstall_after=True):
            import bench.sample_defer_local
    return tc.elapsed


def bench_defer_imports_global() -> float:
    with TimeCatcher() as tc:  # noqa: SIM117
        with defer_imports.import_hook(["*"], uninstall_after=True):
            import bench.sample_defer_global
    return tc.elapsed


BENCH_FUNCS = {
    "regular": bench_regular,
    "defer_imports (local)": bench_defer_imports_local,
    "defer_imports (global)": bench_defer_imports_global,
}


def remove_pycaches() -> None:
    """Remove all cached Python bytecode files from the current directory."""

    curr_dir = Path()

    for cache_dir in curr_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir)

    for cache_file in curr_dir.rglob("*.py[co]"):
        cache_file.unlink()


def pprint_results(results: dict[str, float], minimum: float) -> None:
    """Format and print results as an rST-style list table."""

    sep = " " * 2

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

    divider = sep.join((impl_divider, version_divider, benchmark_divider, time_divider))
    header = sep.join((impl_header, version_header, benchmark_header, time_header))

    impl = platform.python_implementation().ljust(impl_len)
    version = f"{sys.version_info.major}.{sys.version_info.minor}".ljust(version_len)

    print(divider)
    print(header)
    print(divider)

    for bench_type, result in results.items():
        fmt_bench_type = bench_type.ljust(benchmark_len)
        fmt_result = f"{result:.5f}s ({result / minimum:.2f}x)".ljust(time_len)

        print(impl, version, fmt_bench_type, fmt_result, sep=sep)

    print(divider)


def main() -> None:
    parser = argparse.ArgumentParser()

    default_exec_order = list(BENCH_FUNCS)
    parser.add_argument(
        "--exec-order",
        action="extend",
        nargs=len(default_exec_order),
        choices=default_exec_order,
        type=str,
        help="The order in which the influenced (or not influenced) imports are run",
    )
    parser.add_argument(
        "--remove-pycaches",
        action="store_true",
        help="Whether to remove __pycache__ directories and other bytecode cache files.",
    )
    args = parser.parse_args()

    # See how long it actually takes to compile.
    if args.remove_pycaches:
        remove_pycaches()

    exec_order: list[str] = args.exec_order or default_exec_order

    results: dict[str, float] = {}
    for type_ in exec_order:
        with unittest.mock.patch.dict(sys.modules):
            results[type_] = BENCH_FUNCS[type_]()

    minimum = min(results.values())

    if sys.dont_write_bytecode:
        print("Run once with __pycache__ folders removed and bytecode caching disallowed")
    else:
        print("Run once with bytecode caching allowed")

    print()

    pprint_results(results, minimum)


if __name__ == "__main__":
    raise SystemExit(main())
