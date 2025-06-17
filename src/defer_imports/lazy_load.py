from __future__ import annotations

import sys as _sys
import threading as _threading
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec, SourceFileLoader as _SourceFileLoader


__all__ = ("until_module_use",)


# ============================================================================
# region -------- Compatibility shims --------
# ============================================================================


TYPE_CHECKING = False


# importlib.abc.Loader changed location in 3.10 to become cheaper to import,
# but importlib.abc became cheap again in 3.14.
if TYPE_CHECKING or _sys.version_info >= (3, 14):  # pragma: >=3.14 cover
    from importlib.abc import Loader as _Loader
else:
    try:  # pragma: >=3.10 cover
        from importlib._abc import Loader as _Loader
    except ImportError:  # pragma: <3.10 cover
        from importlib.abc import Loader as _Loader


if TYPE_CHECKING:
    from typing_extensions import Self as _Self
elif _sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    _Self: _t.TypeAlias = "_t.Self"
else:  # pragma: <3.11 cover

    class Self:
        """Placeholder for typing.Self."""

    _Self = Self
    del Self


if TYPE_CHECKING:
    from _typeshed.importlib import MetaPathFinderProtocol as _MetaPathFinderProtocol
else:

    class MetaPathFinderProtocol:
        """Placeholder for _typeshed.importlib.MetaPathFinderProtocol."""

    _MetaPathFinderProtocol = MetaPathFinderProtocol
    del MetaPathFinderProtocol


if TYPE_CHECKING:
    import typing as _typing

    _final = _typing.final
else:

    def final(f: object) -> object:  # pragma: no cover (tested in stdlib)
        """Decorator to indicate final methods and final classes."""
        try:
            f.__final__ = True
        except (AttributeError, TypeError):
            # Skip the attribute silently if it is not writable.
            # AttributeError happens if the object has __slots__ or a
            # read-only property, TypeError if it's a builtin class.
            pass
        return f

    _final = final
    del final


# endregion


# ============================================================================
# region -------- Module loader --------
#
# Much of this is adapted from standard library modules to avoid depending on
# private APIs and/or allow changes.
#
# PYUPDATE: Ensure these are consistent with upstream, aside from our
# customizations.
#
# License info
# ------------
# The original sources are
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/util.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.
#
# If any changes are made to the adapted constructs, a short summary of those
# changes accompanies their definitions.
# ============================================================================


# Adapted from importlib.util.
# Changes:
# - Special-case __spec__ in the lazy module type to avoid loading being unnecessarily triggered by internal importlib
#   machinery.
# - Adjust method signatures slightly to be more in line with ModuleType's.
# - Do some slight personalization.
class _LazyModuleType(_types.ModuleType):
    """A subclass of the module type which triggers loading upon attribute access."""

    def __getattribute__(self, name: str, /) -> _t.Any:
        """Trigger the load of the module and return the attribute."""

        __spec__: _ModuleSpec = object.__getattribute__(self, "__spec__")

        # HACK: Prevent internal import machinery from triggering the load early, but with a tradeoff.
        #
        # The importlib machinery unnecessarily causes a load when it checks a lazy module in sys.modules to see if it
        # is initialized (the relevant code is in importlib._bootstrap._find_and_load()). Since the machinery determines
        # that via an attribute on module.__spec__, return the spec without loading.
        #
        # However, this does make our lazy module a leaky abstraction: a user can get __spec__ from a lazy module and
        # modify it without causing a load. However, it's the best we can do at the moment.
        #
        # Extra
        # -----
        # I would further restrict this to only work when importlib internals request __spec__, but I don't know how.
        # I attempted to do so via stack frame examination:
        #     - sys._getframemodulename
        #     - sys._getframe
        #     - traceback.tb_frame.f_back...
        # Unfortunately, none of the above could even see a frame where __spec__ is requested by
        # importlib._bootstrap._find_and_load(); the import statement somehow directly requests it?
        # I'm assuming bytecode shenanigans are involved.
        #
        # The only other alternative I can think of is locally rewriting and monkeypatching internal importlib machinery
        # to account for _LazyModuleType's behavior, but that's a terrible idea for too many reasons.
        if name == "__spec__":
            return __spec__

        loader_state = __spec__.loader_state
        with loader_state["lock"]:
            # Only the first thread to get the lock should trigger the load
            # and reset the module's class. The rest can now getattr().
            if object.__getattribute__(self, "__class__") is _LazyModuleType:
                __class__ = loader_state["__class__"]

                # Reentrant calls from the same thread must be allowed to proceed without
                # triggering the load again.
                # exec_module() and self-referential imports are the primary ways this can
                # happen, but in any case we must return something to avoid deadlock.
                if loader_state["is_loading"]:
                    return __class__.__getattribute__(self, name)
                loader_state["is_loading"] = True

                __dict__: dict[str, _t.Any] = __class__.__getattribute__(self, "__dict__")

                # All module metadata must be gathered from __spec__ in order to avoid
                # using mutated values.
                # Get the original name to make sure no object substitution occurred
                # in sys.modules.
                original_name = __spec__.name

                # Figure out exactly what attributes were mutated between the creation
                # of the module and now.
                attrs_then: dict[str, _t.Any] = loader_state["__dict__"]
                attrs_now = __dict__
                attrs_updated = {
                    key: value
                    for key, value in attrs_now.items()
                    # Code that set an attribute may have kept a reference to the
                    # assigned object, making identity more important than equality.
                    if (key not in attrs_then) or (attrs_now[key] is not attrs_then[key])
                }

                assert __spec__.loader is not None, "This spec must have an actual loader."
                __spec__.loader.exec_module(self)

                # If exec_module() was used directly there is no guarantee the module
                # object was put into sys.modules.
                original_mod = _sys.modules.get(original_name, None)
                if (original_mod is not None) and (self is not original_mod):
                    msg = f"module object for {original_name!r} substituted in sys.modules during a lazy load"
                    raise ValueError(msg)

                # Update after loading since that's what would happen in an eager
                # loading situation.
                __dict__ |= attrs_updated

                # Finally, stop triggering this method, if the module did not
                # already update its own __class__.
                if isinstance(self, _LazyModuleType):  # pyright: ignore [reportUnnecessaryIsInstance]
                    object.__setattr__(self, "__class__", __class__)

        return getattr(self, name)

    def __delattr__(self, name: str, /) -> None:
        """Trigger the load and then perform the deletion."""

        # To trigger the load and raise an exception if the attribute
        # doesn't exist.
        self.__getattribute__(name)
        delattr(self, name)


# Adapted from importlib.util.
# Changes:
# - Move threading import within exec_module to the top level to avoid circular import issues.
#     a. This may cause issues when this module is used in emscripten or wasi.
#     b. This may cause issues when this module is used with gevent.
# - Do some slight personalization.
class LazyLoader(_Loader):
    """A loader that creates a module which defers loading until attribute access."""

    # PYUPDATE: py3.12 - Use an accurate protocol instead of Loader in the annotations of these duck-typed methods.

    @staticmethod
    def __check_eager_loader(loader: object) -> None:
        if not hasattr(loader, "exec_module"):
            msg = "loader must define exec_module()"
            raise TypeError(msg)

    @classmethod
    def factory(cls, loader: type[_Loader]) -> _t.Callable[..., _Self]:
        """Construct a callable which returns the eager loader made lazy."""

        cls.__check_eager_loader(loader)
        return lambda *args, **kwargs: cls(loader(*args, **kwargs))

    def __init__(self, loader: _Loader) -> None:
        self.__check_eager_loader(loader)
        self.loader = loader

    def create_module(self, spec: _ModuleSpec) -> _t.Optional[_types.ModuleType]:
        return self.loader.create_module(spec)

    def exec_module(self, module: _types.ModuleType) -> None:
        """Make the module load lazily."""

        assert module.__spec__ is not None, "The module should have been initialized with a spec."

        module.__spec__.loader = self.loader
        module.__loader__ = self.loader

        # Don't need to worry about deep-copying as trying to set an attribute
        # on an object would have triggered the load,
        # e.g. ``module.__spec__.loader = None`` would trigger a load from
        # trying to access module.__spec__.
        loader_state = {
            "__dict__": module.__dict__.copy(),
            "__class__": module.__class__,
            "lock": _threading.RLock(),
            "is_loading": False,
        }
        module.__spec__.loader_state = loader_state
        module.__class__ = _LazyModuleType


# endregion


# ============================================================================
# region -------- Module finder --------
# ============================================================================


class _LazyFinder:
    """A finder proxy that uses `_LazyLoader` to wrap loaders of source module specs."""

    def __init__(self, finder: _MetaPathFinderProtocol) -> None:
        if not hasattr(finder, "find_spec"):
            msg = "finder must define find_spec()"
            raise TypeError(msg)

        object.__setattr__(self, "_finder", finder)

    def find_spec(
        self,
        name: str,
        path: _t.Optional[_t.Sequence[str]] = None,
        target: _t.Optional[_types.ModuleType] = None,
    ) -> _t.Optional[_ModuleSpec]:
        spec = self._finder.find_spec(name, path, target)

        # Only be lazy for source modules to avoid issues with extension modules having uninitialized state,
        # especially since loading can't currently be triggered by the C APIs that interact with that state,
        # e.g. PyModule_GetState.
        # Ref: https://github.com/python/cpython/issues/85963
        if (spec is not None) and ((loader := spec.loader) is not None) and isinstance(loader, _SourceFileLoader):
            spec.loader = LazyLoader(loader)

        return spec

    def __getattribute__(self, name: str, /) -> _t.Any:
        if name in {"_finder", "find_spec"}:
            return object.__getattribute__(self, name)

        original_finder = object.__getattribute__(self, "_finder")
        return getattr(original_finder, name)

    def __setattr__(self, name: str, value: _t.Any, /) -> None:
        return setattr(self._finder, name, value)

    def __delattr__(self, name: str, /) -> None:
        return delattr(self._finder, name)


# endregion


#: A lock for preventing our code from data-racing itself when modifying sys.meta_path.
_meta_path_lock = _threading.Lock()


@_final
class LazyFinderContext:
    """A context manager within which some imports of modules will occur "lazily". Not re-entrant.

    Caveats:
    - The modules being imported must be written in pure Python. Anything else will be imported eagerly.
    - ``from`` imports may be evaluated eagerly.
    - In a nested import such as ``import a.b.c``, only ``c`` will be lazily imported. ``a`` and ``a.b`` will be eagerly
    imported. This may change in the future.
    - Modules that perform their own import hacks might not cooperate with this. For instance, at one point, `collections`
    put `collections.abc` in `sys.modules` in an unusual way at import time, so attempting to lazy-load `collections.abc`
    would just break.
    """

    def __init_subclass__(cls, *args: object, **kwargs: object) -> _t.NoReturn:
        msg = f"Type {cls.__name__!r} is not an acceptable base type."
        raise TypeError(msg)

    def __enter__(self, /) -> None:
        with _meta_path_lock:
            for i, finder in enumerate(_sys.meta_path):
                if not isinstance(finder, _LazyFinder):
                    _sys.meta_path[i] = _LazyFinder(finder)

    def __exit__(self, *exc_info: object) -> None:
        with _meta_path_lock:
            for i, finder in enumerate(_sys.meta_path):
                if isinstance(finder, _LazyFinder):
                    _sys.meta_path[i] = finder._finder


until_module_use = LazyFinderContext


# Ensure our type annotations are valid if evaluated at runtime.
with until_module_use():
    import typing as _t
