# pyright: reportUnusedImport=none
"""Simple benchark script for comparing the import time of the Python standard library when using regular imports,
defer_imports-influence imports, and slothy-influenced imports.

The sample scripts being imported are generated with benchmark/generate_samples.py.
"""

import platform
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


def remove_pycaches() -> None:
    """Remove all cached Python bytecode files from the current working directory."""

    for file in Path().rglob("*.py[co]"):
        file.unlink()

    for dir_ in Path().rglob("__pycache__"):
        # Sometimes, files with atypical names are still in these.
        for file in dir_.iterdir():
            if file.is_file():
                file.unlink()
        dir_.rmdir()


def bench_regular() -> float:
    with CatchTime() as ct:
        import benchmark.sample_regular
    return ct.elapsed


def bench_defer_imports() -> float:
    defer_imports.install_defer_import_hook()

    with CatchTime() as ct:
        import benchmark.sample_defer_imports

    defer_imports.uninstall_defer_import_hook()
    return ct.elapsed


def bench_slothy() -> float:
    with CatchTime() as ct:
        import benchmark.sample_slothy
    return ct.elapsed


BENCH_FUNCS = {
    "regular": bench_regular,
    "slothy": bench_slothy,
    "defer_imports": bench_defer_imports,
}


def main() -> None:
    import argparse

    # Get arguments from user.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exec-order",
        action="extend",
        nargs=3,
        choices=BENCH_FUNCS.keys(),
        type=str,
        help="The order in which the the influences (or not influenced) imports are run",
    )
    args = parser.parse_args()

    # Do any remaining setup.
    if sys.dont_write_bytecode:
        remove_pycaches()

    exec_order = args.exec_order or list(BENCH_FUNCS)

    # Perform benchmarking.
    results = {type_: BENCH_FUNCS[type_]() for type_ in exec_order}
    minimum = min(results.values())

    # Format and print results as an reST-style list table.
    impl_header = "Implementation"
    impl_len = len(impl_header)
    impl_divider = "=" * impl_len

    version_header = "Version"
    version_len = len(version_header)
    version_divider = "=" * version_len

    benchmark_len = 10
    benchmark_header = "Benchmark".ljust(benchmark_len)
    benchmark_divider = "=" * benchmark_len

    time_len = 19
    time_header = "Time".ljust(time_len)
    time_divider = "=" * time_len

    divider = "  ".join((impl_divider, version_divider, benchmark_divider, time_divider))

    impl = platform.python_implementation().ljust(impl_len)
    version = f"{sys.version_info.major}.{sys.version_info.minor}".ljust(version_len)

    if sys.dont_write_bytecode:
        print("Run once with __pycache__ folders removed and bytecode caching disallowed")
    else:
        print("Run once with bytecode caching allowed")

    print()
    print(divider)
    print(impl_header, version_header, benchmark_header, time_header, sep="  ")
    print(divider)

    for bench_type, result in results.items():
        fmt_bench_type = bench_type.ljust(benchmark_len)
        fmt_result = f"{result:.5f}s ({result / minimum:.2f}x)".ljust(time_len)

        print(impl, version, fmt_bench_type, fmt_result, sep="  ")

    print(divider)


if __name__ == "__main__":
    raise SystemExit(main())
