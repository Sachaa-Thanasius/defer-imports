"""Regressions tests for flaky/past behavioral issues."""

from pathlib import Path
from typing import Any

import pytest

from tests.util import create_sample_module


class _Missing:
    """Private sentinel."""


@pytest.mark.flaky(reruns=3)
def test_thread_safety(tmp_path: Path):
    """Test that trying to access a lazily loaded import from multiple threads doesn't cause race conditions.

    Based on a test for importlib.util.LazyLoader in the CPython test suite.

    Notes
    -----
    This test is flaky, seemingly more so in CI than locally. Some information about the failure:

    -   It's the same every time: paraphrased, the "inspect" proxy doesn't have "signature" as an attribute. The proxy
        should be resolved before this happens, and is even guarded with a RLock and a boolean to prevent this.
    -   It seemingly only happens on pypy.
    -   The reproduction rate locally is ~1/100 when run across pypy3.9 and pypy3.10, 50 times each.
        -   Add 'pytest.mark.parametrize("q", range(50))' to the test for the repetitions.
        -   Run "hatch test -py pypy3.9,pypy3.10 -- tests/test_deferred.py::test_thread_safety".
    """

    source = """\
import defer_imports

with defer_imports.until_use:
    import inspect
"""

    spec, module, _ = create_sample_module(tmp_path, source)
    assert spec.loader
    spec.loader.exec_module(module)

    import threading
    import time

    class CapturingThread(threading.Thread):
        """Thread subclass that captures a returned result or raised exception from the called target."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.result = _Missing
            self.exc = _Missing

        def run(self) -> None:  # pragma: no cover
            # This has minor modifications from threading.Thread.run() to catch the returned value or raised exception.
            try:
                self.result = self._target(*self._args, **self._kwargs)  # pyright: ignore
            except Exception as exc:  # noqa: BLE001
                self.exc = exc
            finally:
                del self._target, self._args, self._kwargs  # pyright: ignore

    def access_module_attr() -> object:
        time.sleep(0.2)
        return module.inspect.signature

    threads: list[CapturingThread] = []

    for i in range(20):
        thread = CapturingThread(name=f"Thread {i}", target=access_module_attr)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
        assert thread.exc is _Missing
        assert callable(thread.result)  # pyright: ignore
