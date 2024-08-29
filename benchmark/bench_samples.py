# pyright: reportUnusedImport=none
"""Simple benchark script for comparing the import time of the Python standard library when using regular imports,
deferred-influence imports, and slothy-influenced imports.
"""

import sys
import time
from pathlib import Path

import deferred


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


def bench_deferred() -> float:
    deferred.install_defer_import_hook()

    with CatchTime() as ct:
        import benchmark.sample_deferred

    deferred.uninstall_defer_import_hook()
    return ct.elapsed


def bench_slothy() -> float:
    with CatchTime() as ct:
        import benchmark.sample_slothy
    return ct.elapsed


BENCH_FUNCS = {
    "regular": bench_regular,
    "slothy": bench_slothy,
    "deferred": bench_deferred,
}


def main() -> None:
    import argparse

    # Get arguments from user.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove-pycache",
        action="store_true",
        help="Whether to remove pycache files in cwd beforehand and ensure bytecode isn't written",
    )
    parser.add_argument(
        "--exec-order",
        action="extend",
        nargs=3,
        choices=BENCH_FUNCS.keys(),
        type=str,
        help="The order in which the the influences (or not influenced) imports are run",
    )
    args = parser.parse_args()

    # Perform benchmarking.
    if args.remove_pycache:
        remove_pycaches()
        sys.dont_write_bytecode = True

    exec_order = args.exec_order or list(BENCH_FUNCS)

    # TODO: Investigate how to make multiple iterations work.
    results = {type_: BENCH_FUNCS[type_]() for type_ in exec_order}
    minimum = min(results.values())

    # Format and print outcomes.
    header = "Time to import most of the stdlib"
    subheader = f"(with caching = {not args.remove_pycache})"
    separator = "-" * max(len(header), len(subheader))

    print("\n")
    print(header)
    print(separator)
    print(subheader)
    print(separator)

    for type_, result in results.items():
        print(f"{f'{type_}:':15} {result:.7f}s  ({result / minimum:.2f}x)")


if __name__ == "__main__":
    raise SystemExit(main())
