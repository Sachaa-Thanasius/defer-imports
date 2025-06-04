# Most of the tests below are adapted from
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/test/test_importlib/test_lazy.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import threading
import time
import types
import unittest.mock
from collections.abc import Sequence
from importlib.machinery import ModuleSpec

import pytest

from defer_imports.lazy_load import _LazyFinder, _LazyLoader, _LazyModuleType, until_module_use


# PYUPDATE: Keep in sync with test.support.threading_helper._can_start_thread().
def _can_start_thread() -> bool:
    """Detect whether Python can start new threads.

    Some WebAssembly platforms do not provide a working pthread
    implementation. Thread support is stubbed and any attempt
    to create a new thread fails.

    - wasm32-wasi does not have threading.
    - wasm32-emscripten can be compiled with or without pthread
    support (-s USE_PTHREADS / __EMSCRIPTEN_PTHREADS__).
    """

    if sys.platform == "emscripten":  # pragma: no cover
        return sys._emscripten_info.pthreads
    elif sys.platform == "wasi":  # pragma: no cover
        return False
    else:
        # assume all other platforms have working thread support.
        return True


#: Skip tests that require threading to work.
requires_working_threading = pytest.mark.skipif(not _can_start_thread(), reason="requires threading support")


class CollectInit:
    def __init__(self, *args: object, **kwargs: object):
        self.args = args
        self.kwargs = kwargs

    def exec_module(self, module: types.ModuleType): ...


class TestLazyLoaderFactory:
    def test_init(self):
        factory = _LazyLoader.factory(CollectInit)
        # E.g. what importlib.machinery.FileFinder instantiates loaders with
        # plus keyword arguments.
        lazy_loader = factory("module name", "module path", kw="kw")
        loader = lazy_loader.loader
        assert loader.args == ("module name", "module path")
        assert loader.kwargs == {"kw": "kw"}

    def test_validation(self):
        # No exec_module(), no lazy loading.
        with pytest.raises(TypeError):
            _LazyLoader.factory(object)


class TestingImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    module_name = "lazy_loader_test"
    mutated_name = "changed"
    loaded = None
    load_count = 0
    source_code = f"attr = 42; __name__ = {mutated_name!r}"

    def find_spec(
        self,
        name: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        if name != self.module_name:  # pragma: no cover
            return None
        return importlib.util.spec_from_loader(name, _LazyLoader(self))

    def exec_module(self, module: types.ModuleType) -> None:
        time.sleep(0.01)  # Simulate a slow load.
        exec(self.source_code, module.__dict__)
        self.loaded = module
        self.load_count += 1


@pytest.mark.usefixtures("preserve_sys_modules")
class TestLazyLoader:
    def new_module(self, source_code: str | None = None, loader: TestingImporter | None = None) -> types.ModuleType:
        if loader is None:
            loader = TestingImporter()
        if source_code is not None:
            loader.source_code = source_code

        spec = importlib.util.spec_from_loader(TestingImporter.module_name, _LazyLoader(loader))
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Module is now lazy.
        assert loader.loaded is None
        return module

    @pytest.fixture
    def module(self) -> types.ModuleType:
        return self.new_module()

    def test_init(self):
        with pytest.raises(TypeError):
            # Classes that don't define exec_module() trigger TypeError.
            _LazyLoader(object)

    def test_e2e(self):
        # End-to-end test to verify the load is in fact lazy.
        importer = TestingImporter()
        assert importer.loaded is None
        with unittest.mock.patch.object(sys, "meta_path", [importer]):
            module = importlib.import_module(importer.module_name)
        assert importer.loaded is None
        # Trigger load.
        assert module.__loader__ == importer
        assert importer.loaded is not None
        assert module == importer.loaded

    def test_attr_unchanged(self, module: types.ModuleType):
        # An attribute only mutated as a side-effect of import should not be
        # changed needlessly.
        assert TestingImporter.mutated_name == module.__name__

    def test_new_attr(self, module: types.ModuleType):
        # A new attribute should persist.
        module.new_attr = 42
        assert module.new_attr == 42

    def test_mutated_preexisting_attr(self, module: types.ModuleType):
        # Changing an attribute that already existed on the module --
        # e.g. __name__ -- should persist.
        module.__name__ = "bogus"
        assert module.__name__ == "bogus"

    def test_mutated_attr(self, module: types.ModuleType):
        # Changing an attribute that comes into existence after an import
        # should persist.
        module.attr = 6
        assert module.attr == 6

    def test_delete_eventual_attr(self, module: types.ModuleType):
        # Deleting an attribute should stay deleted.
        del module.attr
        assert not hasattr(module, "attr")

    def test_delete_preexisting_attr(self, module: types.ModuleType):
        del module.__name__
        assert not hasattr(module, "__name__")

    def test_module_substitution_error(self):
        fresh_module = types.ModuleType(TestingImporter.module_name)
        sys.modules[TestingImporter.module_name] = fresh_module
        module = self.new_module()
        with pytest.raises(ValueError, match="substituted"):
            _ = module.__name__

    def test_module_already_in_sys(self, module: types.ModuleType):
        sys.modules[TestingImporter.module_name] = module
        # Force the load; just care that no exception is raised.
        _ = module.__name__

    @requires_working_threading
    def test_module_load_race(self):
        loader = TestingImporter()
        module = self.new_module(loader=loader)
        assert loader.load_count == 0

        class RaisingThread(threading.Thread):
            exc = None

            def run(self):
                try:
                    super().run()
                except Exception as exc:  # pragma: no cover # noqa: BLE001
                    self.exc = exc

        def access_module():
            return module.attr

        threads: list[RaisingThread] = []
        for _ in range(2):
            thread = RaisingThread(target=access_module)
            threads.append(thread)
            thread.start()

        # Races could cause errors
        for thread in threads:
            thread.join()
            assert thread.exc is None

        # Or multiple load attempts
        assert loader.load_count == 1

    def test_lazy_self_referential_modules(self, monkeypatch: pytest.MonkeyPatch):
        # Directory modules with submodules that reference the parent can attempt to access
        # the parent module during a load. Verify that this common pattern works with lazy loading.
        # json is a good example in the stdlib.

        for name in list(sys.modules):
            if name.startswith("json"):
                monkeypatch.delitem(sys.modules, name, raising=False)

        # Standard lazy loading, unwrapped
        spec = importlib.util.find_spec("json")
        assert spec is not None
        assert spec.loader is not None

        loader = _LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["json"] = module
        loader.exec_module(module)

        # Trigger load with attribute lookup, ensure expected behavior
        test_load = module.loads("{}")
        assert test_load == {}

    def test_lazy_module_type_override(self):
        # Verify that lazy loading works with a module that modifies
        # its __class__ to be a custom type.

        # Example module from PEP 726
        module = self.new_module(
            source_code="""\
import sys
from types import ModuleType

CONSTANT = 3.14

class ImmutableModule(ModuleType):
    def __setattr__(self, name, value):
        raise AttributeError('Read-only attribute!')

    def __delattr__(self, name):
        raise AttributeError('Read-only attribute!')

sys.modules[__name__].__class__ = ImmutableModule
"""
        )
        sys.modules[TestingImporter.module_name] = module
        assert isinstance(module, _LazyModuleType)
        assert module.CONSTANT == 3.14
        with pytest.raises(AttributeError):
            module.CONSTANT = 2.71
        with pytest.raises(AttributeError):
            del module.CONSTANT

    def test_special_case___spec__(self, module: types.ModuleType):
        # Verify that getting/modifying module.__spec__ doesn't trigger the load.
        assert object.__getattribute__(module, "__class__") is _LazyModuleType
        _ = module.__spec__
        assert object.__getattribute__(module, "__class__") is _LazyModuleType
        module.__spec__.name = "blahblahblah"
        assert object.__getattribute__(module, "__class__") is _LazyModuleType

    @requires_working_threading
    def test_module_find_race(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delitem(sys.modules, "inspect", raising=False)

        class RaisingThread(threading.Thread):
            exc = None

            def run(self):
                try:
                    super().run()
                except Exception as exc:  # pragma: no cover # noqa: BLE001
                    self.exc = exc

        def find_spec():
            with until_module_use:
                return importlib.util.find_spec("inspect")

        threads: list[RaisingThread] = []
        for _ in range(10):
            thread = RaisingThread(target=find_spec)
            threads.append(thread)
            thread.start()

        # Races could cause errors
        for thread in threads:
            thread.join()
            assert thread.exc is None


class TestLazyFinder:
    @pytest.mark.parametrize("mod_name", ["sys", "zipimport"])
    def test_doesnt_wrap_non_source_file_loaders(self, mod_name: str):
        spec = _LazyFinder.find_spec(mod_name)
        assert spec is not None
        assert not isinstance(spec.loader, _LazyLoader)

    def test_wraps_source_file_loader(self):
        spec = _LazyFinder.find_spec("inspect")
        assert spec is not None
        assert isinstance(spec.loader, _LazyLoader)

    def test_warning_if_missing_from_meta_path(self, monkeypatch: pytest.MonkeyPatch):
        with unittest.mock.patch.object(sys, "meta_path", list(sys.meta_path)):
            monkeypatch.delitem(sys.modules, "inspect", raising=False)

            with pytest.warns(ImportWarning) as record:  # noqa: SIM117
                with until_module_use:
                    import inspect  # noqa: F401

                    sys.meta_path.remove(_LazyFinder)

            assert len(record) == 1
            assert record[0].message.args[0] == "_LazyFinder unexpectedly missing from sys.meta_path"

    def test_e2e(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delitem(sys.modules, "inspect", raising=False)

        # Lazily imported.
        with until_module_use:
            import inspect
        assert object.__getattribute__(inspect, "__class__") is _LazyModuleType

        # When lazily imported again, still unloaded.
        with until_module_use:
            import inspect
        assert object.__getattribute__(inspect, "__class__") is _LazyModuleType

        # When regularly imported but untouched, still unloaded.
        import inspect

        assert object.__getattribute__(inspect, "__class__") is _LazyModuleType

        # Only on accessing a variable is it loaded.
        _ = inspect.signature

        assert object.__getattribute__(inspect, "__class__") is types.ModuleType
