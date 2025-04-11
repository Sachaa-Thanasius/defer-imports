# A substantial portion of the code and comments below is adapted from
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/util.py
# and https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/Lib/importlib/_bootstrap.py
# with the original copyright being:
# Copyright (c) 2001 Python Software Foundation; All Rights Reserved
#
# The license in its original form may be found at
# https://github.com/python/cpython/blob/49234c065cf2b1ea32c5a3976d834b1d07b9b831/LICENSE
# and in this repository at ``LICENSE_cpython``.

from __future__ import annotations

import _imp
import sys
import threading
import types
import warnings
from importlib.machinery import ModuleSpec, SourceFileLoader


__all__ = ("until_module_use",)


# ============================================================================
# region -------- Compatibility shims --------
# ============================================================================


TYPE_CHECKING = False


# importlib.abc.Loader changed location in 3.10 to become cheaper to import,
# but importlib.abc became cheap again in 3.14.
if TYPE_CHECKING or sys.version_info >= (3, 14):  # pragma: >=3.14 cover
    from importlib.abc import Loader
else:
    try:  # pragma: >=3.10 cover
        from importlib._abc import Loader
    except ImportError:  # pragma: <3.10 cover
        from importlib.abc import Loader


if TYPE_CHECKING:
    from typing_extensions import Self
elif sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    Self: t.TypeAlias = "t.Self"
else:  # pragma: <3.11 cover

    class Self:
        """Placeholder for typing.Self."""


# endregion


# ============================================================================
# region -------- Module loader --------
#
# These are adapted from importlib.util to avoid depending on private APIs and
# allow changes.
#
# PYUPDATE: Ensure these are consistent with upstream, aside from our
# customizations.
# ============================================================================


# Adapted from importlib.util.
# Changes include:
# - Special-case __spec__ in the lazy module type to avoid loading being unnecessarily triggered by internal importlib
#   machinery.
# - Adjust method signatures slightly to be more in line with object's.
# - Do some slight personalization.
class _LazyModuleType(types.ModuleType):
    """A subclass of the module type which triggers loading upon attribute access."""

    def __getattribute__(self, name: str, /) -> t.Any:
        """Trigger the load of the module and return the attribute."""

        __spec__: ModuleSpec = object.__getattribute__(self, "__spec__")

        # NOTE: We want to avoid the importlib machinery unnecessarily causing a load
        # when it checks a lazy module in sys.modules to see if it is initialized
        # (The relevant code is in importlib._bootstrap._find_and_load()). Since the machinery determines that
        # via an attribute on module.__spec__, return the spec without loading.
        #
        # This does mean a user can get __spec__ from a lazy module and modify it without causing a load.
        # Beware: the consequences are unknown.
        #
        # Extra notes
        # -----------
        # I would further restrict this to only work when importlib internals request __spec__, but I don't know how.
        # I attempted the following:
        #
        # 1. Stack frame examination via:
        #     - sys._getframemodulename
        #     - sys._getframe
        #     - traceback.tb_frame.f_back...
        #    Unfortunately, none of the above could even see a frame where __spec__ is requested by
        #    importlib._bootstrap._find_and_load(); the import statement somehow directly requests it?
        #    My guess is that bytecode shenanigans are involved.
        # 2. Is there even another way?
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

                __dict__: dict[str, t.Any] = __class__.__getattribute__(self, "__dict__")

                # All module metadata must be gathered from __spec__ in order to avoid
                # using mutated values.
                # Get the original name to make sure no object substitution occurred
                # in sys.modules.
                original_name = __spec__.name

                # Figure out exactly what attributes were mutated between the creation
                # of the module and now.
                attrs_then: dict[str, t.Any] = loader_state["__dict__"]
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
                original_mod = sys.modules.get(original_name, None)
                if (original_mod is not None) and (self is not original_mod):
                    msg = f"module object for {original_name!r} substituted in sys.modules during a lazy load"
                    raise ValueError(msg)

                # Update after loading since that's what would happen in an eager
                # loading situation.
                __dict__ |= attrs_updated

                # Finally, stop triggering this method, if the module did not
                # already update its own __class__.
                if isinstance(self, _LazyModuleType):
                    object.__setattr__(self, "__class__", __class__)

        return getattr(self, name)

    def __delattr__(self, name: str, /) -> None:
        """Trigger the load and then perform the deletion."""

        # To trigger the load and raise an exception if the attribute
        # doesn't exist.
        self.__getattribute__(name)
        delattr(self, name)


# Adapted from importlib.util.
# Changes include:
# - Move threading import within exec_module to the top level to avoid circular import issues.
#     a. This may cause issues when this module is used in emscripten or wasi.
#        TODO: Test this.
#     b. This may cause issues when this module is used with gevent.
#        TODO: Test this.
# - Do some slight personalization.
class _LazyLoader(Loader):
    """A loader that creates a module which defers loading until attribute access."""

    @staticmethod
    def __check_eager_loader(loader: t.Union[Loader, type[Loader]]) -> None:
        if not hasattr(loader, "exec_module"):
            msg = "loader must define exec_module()"
            raise TypeError(msg)

    @classmethod
    def factory(cls, loader: type[Loader]) -> t.Callable[..., Self]:
        """Construct a callable which returns the eager loader made lazy."""

        cls.__check_eager_loader(loader)
        return lambda *args, **kwargs: cls(loader(*args, **kwargs))

    def __init__(self, loader: Loader) -> None:
        self.__check_eager_loader(loader)
        self.loader = loader

    def create_module(self, spec: ModuleSpec) -> t.Optional[types.ModuleType]:
        return self.loader.create_module(spec)

    def exec_module(self, module: types.ModuleType) -> None:
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
            "lock": threading.RLock(),
            "is_loading": False,
        }
        module.__spec__.loader_state = loader_state
        module.__class__ = _LazyModuleType


# endregion


# ============================================================================
# region -------- Module finder --------
#
# Some of this code, specifically _ImportLockContext and
# _find_spec_without_lazyfinder(), was adapted from importlib._bootstrap to
# avoid depending on private APIs and allow changes.
#
# PYUPDATE: Ensure the adapted code is consistent with upstream, aside from
# our customizations.
# ============================================================================


# Adapted from importlib._bootstrap.
class _ImportLockContext:
    """Context manager for the import lock."""

    def __enter__(self, /) -> None:
        """Acquire the import lock."""

        _imp.acquire_lock()

    def __exit__(self, *_dont_care: object) -> None:
        """Release the import lock regardless of any raised exceptions."""

        _imp.release_lock()


# Adapted from importlib._bootstrap._find_spec.
# Changes:
# - Backport bugfixes (gh-130094).
# - Skip _LazyFinder when trying finders on sys.meta_path.
def _find_spec_without_lazyfinder(  # noqa: PLR0912
    name: str,
    path: t.Optional[t.Sequence[str]],
    target: t.Optional[types.ModuleType] = None,
) -> t.Optional[ModuleSpec]:  # pragma: no cover
    """Find a module's spec.

    Ignore the presence of `_LazyFinder` on `sys.meta_path`.
    """

    meta_path = sys.meta_path
    if meta_path is None:  # pyright: ignore [reportUnnecessaryComparison]
        # PyImport_Cleanup() is running or has been called.
        msg = "sys.meta_path is None, Python is likely shutting down"
        raise ImportError(msg)

    # gh-130094: Copy sys.meta_path so that we have a consistent view of the
    # list while iterating over it.
    meta_path = list(meta_path)
    if not meta_path:
        warnings.warn("sys.meta_path is empty", ImportWarning)  # noqa: B028

    # We check sys.modules here for the reload case.  While a passed-in
    # target will usually indicate a reload there is no guarantee, whereas
    # sys.modules provides one.
    is_reload = name in sys.modules
    for finder in meta_path:
        # NOTE: Just skip _LazyFinder.
        if finder is _LazyFinder:
            continue

        with _ImportLockContext():
            try:
                find_spec = finder.find_spec
            except AttributeError:
                continue
            else:
                spec = find_spec(name, path, target)

        if spec is not None:
            # The parent import may have already imported this module.
            if not is_reload and name in sys.modules:
                module = sys.modules[name]
                try:
                    __spec__ = module.__spec__
                except AttributeError:
                    # We use the found spec since that is the one that
                    # we would have used if the parent module hadn't
                    # beaten us to the punch.
                    return spec
                else:
                    if __spec__ is None:
                        return spec
                    else:
                        return __spec__
            else:
                return spec

    return None


class _LazyFinder:
    """A finder that wraps loaders for source modules with `_LazyLoader`."""

    @classmethod
    def find_spec(
        cls,
        name: str,
        path: t.Optional[t.Sequence[str]] = None,
        target: t.Optional[types.ModuleType] = None,
    ) -> t.Optional[ModuleSpec]:
        spec = _find_spec_without_lazyfinder(name, path, target)

        # Skip being lazy for non-source modules to avoid issues with extension modules having
        # uninitialized state, especially since loading can't currently be triggered by PyModule_GetState.
        # Ref: https://github.com/python/cpython/issues/85963
        if (spec is not None) and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _LazyLoader(spec.loader)

        return spec


#: A lock for preventing our code from data-racing itself when modifying sys.meta_path.
_meta_path_lock = threading.RLock()


class _LazyFinderContext:
    """The type of `defer_imports.until_module_use`. Should not be manually constructed."""

    def __enter__(self, /) -> None:
        with _meta_path_lock:
            sys.meta_path.insert(0, _LazyFinder)

    def __exit__(self, *_dont_care: object) -> None:
        try:
            with _meta_path_lock:
                sys.meta_path.remove(_LazyFinder)
        except ValueError:
            warnings.warn("_LazyFinder unexpectedly missing from sys.meta_path", ImportWarning, stacklevel=2)


until_module_use: t.Final[_LazyFinderContext] = _LazyFinderContext()
"""A context manager within which some imports will occur "lazily".

The modules being imported must be written in pure Python. Anything else will be imported eagerly.

``from`` imports may be evaluated eagerly.

In a nested import such as ``import a.b.c``, only ``c`` will be lazily imported.
``a`` and ``a.b`` will be eagerly imported. This may change in the future.

Modules with import side effects might not cooperate with this.
For instance, `collections` puts `collections.abc` in `sys.modules` in an unusual way at import time,
meaning lazy-loading `collections.abc` will just break.
"""


# Ensure our type annotations are valid if evaluated at runtime.
with until_module_use:
    import typing as t


# endregion
