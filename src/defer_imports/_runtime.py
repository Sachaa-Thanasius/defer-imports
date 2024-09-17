# SPDX-FileCopyrightText: 2024-present Sachaa-Thanasius
#
# SPDX-License-Identifier: MIT

"""The implementation for defer_imports's runtime magic."""

from __future__ import annotations

import builtins
import contextvars
import sys
import warnings
from importlib.machinery import ModuleSpec
from threading import RLock

from . import _typing as _tp


# ============================================================================
# region -------- Helper functions --------
# ============================================================================


def sanity_check(name: str, package: _tp.Optional[str], level: int) -> None:
    """Verify arguments are "sane".

    Notes
    -----
    Slightly modified version of importlib._bootstrap._sanity_check to avoid depending on an an implementation detail
    module at runtime.
    """

    if not isinstance(name, str):  # pyright: ignore [reportUnnecessaryIsInstance]
        msg = f"module name must be str, not {type(name)}"
        raise TypeError(msg)
    if level < 0:
        msg = "level must be >= 0"
        raise ValueError(msg)
    if level > 0:
        if not isinstance(package, str):
            msg = "__package__ not set to a string"
            raise TypeError(msg)
        if not package:
            msg = "attempted relative import with no known parent package"
            raise ImportError(msg)
    if not name and level == 0:
        msg = "Empty module name"
        raise ValueError(msg)


def calc___package__(globals: _tp.MutableMapping[str, _tp.Any]) -> _tp.Optional[str]:
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    Notes
    -----
    Slightly modified version of importlib._bootstrap._calc___package__ to avoid depending on an implementation detail
    module at runtime.
    """

    package: str | None = globals.get("__package__")
    spec: ModuleSpec | None = globals.get("__spec__")

    if package is not None:
        if spec is not None and package != spec.parent:
            category = DeprecationWarning if sys.version_info >= (3, 12) else ImportWarning
            warnings.warn(
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",
                category,
                stacklevel=3,
            )
        return package

    if spec is not None:
        return spec.parent

    warnings.warn(
        "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__",
        ImportWarning,
        stacklevel=3,
    )
    package = globals["__name__"]
    if "__path__" not in globals:
        package = package.rpartition(".")[0]  # pyright: ignore [reportOptionalMemberAccess]

    return package


def resolve_name(name: str, package: str, level: int) -> str:
    """Resolve a relative module name to an absolute one.

    Notes
    -----
    Slightly modified version of importlib._bootstrap._resolve_name to avoid depending on an implementation detail
    module at runtime.
    """

    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        msg = "attempted relative import beyond top-level package"
        raise ImportError(msg)
    base = bits[0]
    return f"{base}.{name}" if name else base


# endregion


# ============================================================================
# region -------- Main implementation --------
# ============================================================================


original_import = contextvars.ContextVar("original_import", default=builtins.__import__)
"""What builtins.__import__ currently points to."""

is_deferred = contextvars.ContextVar("is_deferred", default=False)
"""Whether imports should be deferred."""


class DeferredImportProxy:
    """Proxy for a deferred __import__ call."""

    def __init__(
        self,
        name: str,
        global_ns: _tp.MutableMapping[str, object],
        local_ns: _tp.MutableMapping[str, object],
        fromlist: _tp.Sequence[str],
        level: int = 0,
    ) -> None:
        self.defer_proxy_name = name
        self.defer_proxy_global_ns = global_ns
        self.defer_proxy_local_ns = local_ns
        self.defer_proxy_fromlist = fromlist
        self.defer_proxy_level = level

        # Only used in cases of non-from-import submodule aliasing a la "import a.b as c".
        self.defer_proxy_sub: str | None = None

    @property
    def defer_proxy_import_args(self):  # noqa: ANN202 # Too verbose.
        """A tuple of args that can be passed into __import__."""

        return (
            self.defer_proxy_name,
            self.defer_proxy_global_ns,
            self.defer_proxy_local_ns,
            self.defer_proxy_fromlist,
            self.defer_proxy_level,
        )

    def __repr__(self) -> str:
        if self.defer_proxy_fromlist:
            imp_stmt = f"from {self.defer_proxy_name} import {', '.join(self.defer_proxy_fromlist)}"
        elif self.defer_proxy_sub:
            imp_stmt = f"import {self.defer_proxy_name} as ..."
        else:
            imp_stmt = f"import {self.defer_proxy_name}"

        return f"<proxy for {imp_stmt!r}>"

    def __getattr__(self, name: str, /) -> _tp.Self:
        if name in self.defer_proxy_fromlist:
            from_proxy = type(self)(*self.defer_proxy_import_args)
            from_proxy.defer_proxy_fromlist = (name,)
            return from_proxy

        elif ("." in self.defer_proxy_name) and (name == self.defer_proxy_name.rpartition(".")[2]):
            submodule_proxy = type(self)(*self.defer_proxy_import_args)
            submodule_proxy.defer_proxy_sub = name
            return submodule_proxy

        else:
            msg = f"proxy for module {self.defer_proxy_name!r} has no attribute {name!r}"
            raise AttributeError(msg)


class DeferredImportKey(str):
    """Mapping key for an import proxy.

    When referenced, the key will replace itself in the namespace with the resolved import or the right name from it.
    """

    __slots__ = ("defer_key_proxy", "is_resolving", "lock")

    def __new__(cls, key: str, proxy: DeferredImportProxy, /) -> _tp.Self:
        return super().__new__(cls, key)

    def __init__(self, key: str, proxy: DeferredImportProxy, /) -> None:
        self.defer_key_proxy = proxy
        self.is_resolving = False
        self.lock = RLock()

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, str):
            return NotImplemented
        if not super().__eq__(value):
            return False

        # Only the first thread to grab the lock should resolve the deferred import.
        with self.lock:
            # Reentrant calls from the same thread shouldn't re-trigger the resolution.
            # This can be caused by self-referential imports, e.g. within __init__.py files.
            if not self.is_resolving:
                self.is_resolving = True

                if not is_deferred.get():
                    self._resolve()

        return True

    def __hash__(self) -> int:
        return super().__hash__()

    def _resolve(self) -> None:
        """Perform an actual import for the given proxy and bind the result to the relevant namespace."""

        proxy = self.defer_key_proxy

        # 1. Perform the original __import__ and pray.
        module: _tp.ModuleType = original_import.get()(*proxy.defer_proxy_import_args)

        # 2. Transfer nested proxies over to the resolved module.
        module_vars = vars(module)
        for attr_key, attr_val in vars(proxy).items():
            if isinstance(attr_val, DeferredImportProxy) and not hasattr(module, attr_key):
                # This could have used setattr() if pypy didn't normalize the attr key type to str, so we resort to
                # direct placement in the module's __dict__ to avoid that.
                module_vars[DeferredImportKey(attr_key, attr_val)] = attr_val

                # Change the namespaces as well to make sure nested proxies are replaced in the right place.
                attr_val.defer_proxy_global_ns = attr_val.defer_proxy_local_ns = module_vars

        # 3. Replace the proxy with the resolved module or module attribute in the relevant namespace.
        # 3.1. Get the regular string key and the relevant namespace.
        key = str(self)
        namespace = proxy.defer_proxy_local_ns

        # 3.2. Replace the deferred version of the key to avoid it sticking around.
        # This will trigger __eq__ again, so we use is_deferred to prevent recursion.
        _is_def_tok = is_deferred.set(True)
        try:
            namespace[key] = namespace.pop(key)
        finally:
            is_deferred.reset(_is_def_tok)

        # 3.3. Resolve any requested attribute access.
        if proxy.defer_proxy_fromlist:
            namespace[key] = getattr(module, proxy.defer_proxy_fromlist[0])
        elif proxy.defer_proxy_sub:
            namespace[key] = getattr(module, proxy.defer_proxy_sub)
        else:
            namespace[key] = module


def deferred___import__(  # noqa: ANN202
    name: str,
    globals: _tp.MutableMapping[str, object],
    locals: _tp.MutableMapping[str, object],
    fromlist: _tp.Optional[_tp.Sequence[str]] = None,
    level: int = 0,
):
    """An limited replacement for __import__ that supports deferred imports by returning proxies."""

    fromlist = fromlist or ()

    package = calc___package__(locals)
    sanity_check(name, package, level)

    # Resolve the names of relative imports.
    if level > 0:
        name = resolve_name(name, package, level)  # pyright: ignore [reportArgumentType]
        level = 0

    # Handle submodule imports if relevant top-level imports already occurred in the call site's module.
    if not fromlist and ("." in name):
        name_parts = name.split(".")
        try:
            # TODO: Consider adding a condition that base_parent must be a ModuleType or a DeferredImportProxy, to
            #       avoid attaching proxies to a random thing that would've normally been clobbered by the import?
            base_parent = parent = locals[name_parts[0]]
        except KeyError:
            pass
        else:
            # Nest submodule proxies as needed.
            for limit, attr_name in enumerate(name_parts[1:], start=2):
                if attr_name not in vars(parent):
                    nested_proxy = DeferredImportProxy(".".join(name_parts[:limit]), globals, locals, (), level)
                    nested_proxy.defer_proxy_sub = attr_name
                    setattr(parent, attr_name, nested_proxy)
                    parent = nested_proxy
                else:
                    parent = getattr(parent, attr_name)

            return base_parent

    return DeferredImportProxy(name, globals, locals, fromlist, level)


# endregion


# ============================================================================
# region -------- Public API --------
# ============================================================================


@_tp.final
class DeferredContext:
    """The type for defer_imports.until_use."""

    __slots__ = ("_import_ctx_token", "_defer_ctx_token")

    def __enter__(self) -> None:
        self._defer_ctx_token = is_deferred.set(True)
        self._import_ctx_token = original_import.set(builtins.__import__)
        builtins.__import__ = deferred___import__

    def __exit__(self, *exc_info: object) -> None:
        original_import.reset(self._import_ctx_token)
        is_deferred.reset(self._defer_ctx_token)
        builtins.__import__ = original_import.get()


until_use: _tp.Final[DeferredContext] = DeferredContext()
"""A context manager within which imports occur lazily. Not reentrant.

This will not work correctly if install_import_hook() was not called first elsewhere.

Raises
------
SyntaxError
    If defer_imports.until_use is used improperly, e.g.:
        1. It is being used in a class or function scope.
        2. It contains a statement that isn't an import.
        3. It contains a wildcard import.

Notes
-----
As part of its implementation, this temporarily replaces builtins.__import__.
"""


# endregion
