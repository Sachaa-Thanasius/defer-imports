from __future__ import annotations

import importlib
import pkgutil
import sys
import types

import pytest

import defer_imports


if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    from inspect import get_annotations
else:  # pragma: no cover (tested in stdlib)
    import functools
    from collections.abc import Callable, Mapping
    from typing import no_type_check

    @no_type_check
    def get_annotations(  # noqa: PLR0912, PLR0915
        obj: Callable[..., object] | type | types.ModuleType,
        *,
        globals: Mapping[str, object] | None = None,  # noqa: A002
        locals: Mapping[str, object] | None = None,  # noqa: A002
        eval_str: bool = False,
    ) -> dict[str, object]:
        """Adapted version of inspect.get_annotations(). See its documentation for details."""

        ann: dict[str, object] | None
        obj_dict: Mapping[str, object] | None
        obj_globals: dict[str, object] | None
        unwrap: Callable[..., object] | type | None

        if isinstance(obj, type):
            # class
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict and hasattr(obj_dict, "get"):
                ann = obj_dict.get("__annotations__", None)
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = None
            else:
                ann = None

            if (
                (module_name := getattr(obj, "__module__", None)) is not None
                and (module := sys.modules.get(module_name, None)) is not None
            ):  # fmt: skip
                obj_globals = getattr(module, "__dict__", None)
            else:
                obj_globals = None

            obj_locals = dict(vars(obj))
            unwrap = obj
        elif isinstance(obj, types.ModuleType):
            # module
            ann = getattr(obj, "__annotations__", None)
            obj_globals = obj.__dict__
            obj_locals = None
            unwrap = None
        elif callable(obj):
            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__globals__", None)
            obj_locals = None
            unwrap = obj
        else:
            msg = f"{obj!r} is not a module, class, or callable."
            raise TypeError(msg)

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            msg = f"{obj!r}.__annotations__ is neither a dict nor None"
            raise ValueError(msg)  # noqa: TRY004

        if not ann:
            return {}

        if not eval_str:
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals  # noqa: A001
        if locals is None:
            locals = obj_locals or {}  # noqa: A001

        return {
            key: (value if (not isinstance(value, str)) else eval(value, globals, locals))  # noqa: S307
            for key, value in ann.items()
        }


def can_have_annotations(obj: object) -> bool:
    return isinstance(obj, (type, types.ModuleType)) or callable(obj)


@pytest.mark.parametrize(
    "module",
    [
        importlib.import_module(mod_info.name)
        for mod_info in pkgutil.iter_modules(
            path=defer_imports.__spec__.submodule_search_locations,
            prefix=defer_imports.__spec__.name + ".",
        )
    ],
)
def test_library_annotations_are_valid(module: types.ModuleType):
    get_annotations(module, eval_str=True)

    for val in filter(can_have_annotations, module.__dict__.values()):
        get_annotations(val, eval_str=True)
