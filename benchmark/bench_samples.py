# pyright: reportUnusedImport=none
# ruff: noqa: T201
"""Simple benchark script for comparing the import time of the Python standard library when using regular imports,
deferred-influence imports, and slothy-influenced imports.
"""

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-pycache", action="store_true", help="Recursively remove pycache files in cwd")
    parser.add_argument("--exec-order", action="extend", nargs=3, type=str)
    args = parser.parse_args()

    if args.remove_pycache:
        remove_pycaches()

    if args.exec_order:
        results = {type_: BENCH_FUNCS[type_]() for type_ in args.exec_order}
    else:
        results = {type_: func() for type_, func in BENCH_FUNCS.items()}

    header = "Time to import stdlib"

    print()
    print(header)
    print("-" * len(header))
    for type_, result in results.items():
        print(f"{f'{type_}:':15} {result}")


if __name__ == "__main__":
    raise SystemExit(main())
