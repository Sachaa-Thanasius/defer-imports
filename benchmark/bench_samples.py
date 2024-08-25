# ruff: noqa: T201
import shutil
import time
from pathlib import Path

import deferred


def bench_regular() -> float:
    start_time = time.perf_counter()
    import benchmark.sample_regular  # type: ignore

    end_time = time.perf_counter()
    return end_time - start_time


def bench_deferred() -> float:
    deferred.install_defer_import_hook()
    start_time = time.perf_counter()
    import benchmark.sample_deferred  # type: ignore

    end_time = time.perf_counter()
    deferred.uninstall_defer_import_hook()
    return end_time - start_time


def bench_slothy() -> float:
    start_time = time.perf_counter()
    import benchmark.sample_slothy  # type: ignore

    end_time = time.perf_counter()
    return end_time - start_time


BENCH_FUNCS = {
    "regular": bench_regular,
    "slothy": bench_slothy,
    "deferred": bench_deferred,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-pycache", action="store_true")
    parser.add_argument("--exec-order", action="extend", nargs=3, type=str)
    args = parser.parse_args()

    if args.remove_pycache:
        for relevant_dir in ("benchmark", "src", "tests"):
            for file in Path(relevant_dir).glob("**/__pycache__"):
                shutil.rmtree(file)

    if args.exec_order:
        results = {}
        for type_ in args.exec_order:
            results[type_] = BENCH_FUNCS[type_]()
    else:
        results = {type_: func() for type_, func in BENCH_FUNCS.items()}

    print("Benching all")
    print("-" * 30)
    for type_, result in results.items():
        print(f"{type_}: {result}")


if __name__ == "__main__":
    raise SystemExit(main())
