=============
defer-imports
=============

.. image:: https://img.shields.io/github/license/Sachaa-Thanasius/defer-imports.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. image:: https://img.shields.io/pypi/v/defer-imports.svg
    :target: https://pypi.org/project/defer-imports
    :alt: PyPI version info

.. image:: https://img.shields.io/pypi/pyversions/defer-imports.svg
    :target: https://pypi.org/project/defer-imports
    :alt: PyPI supported Python versions


A library that implements `PEP 690`_–esque lazy imports in pure Python.

**Note: This is still in development.**


.. contents::
    :local:
    :depth: 2


Installation
============

**defer-imports requires Python 3.9+.**

This can be installed via pip::

    python -m pip install defer-imports


Usage
=====

See the docstrings and comments in the codebase for more details.

Setup
-----

To do its work, ``defer-imports`` must hook into the Python import system. Include the following call somewhere such that it will be executed before your code:

.. code-block:: python

    import defer_imports

    # For all usage, import statements *within the module the hook is installed from* 
    # are not affected. In this case, that would be this module.

    defer_imports.install_import_hook()

    import your_code


The function call's result can be used as a context manager, which makes sense when passing in configuration arguments. That way, unrelated code or usage of this library isn't polluted:

.. code-block:: python

    import defer_imports

    with defer_imports.install_import_hook(module_names=(__name__,)) as hook_ctx:
        import your_code


Making this call without arguments allows user code with imports contained within the ``defer_imports.until_use`` context manager to be deferred until referenced. However, its several configuration parameters allow toggling global instrumentation (affecting all import statements) and adjusting the granularity of that global instrumentation.

**WARNING: Avoid using the hook as anything other than a context manager when passing in module-specific configuration; otherwise, the explicit (or default) configuration will persist and may cause other packages using defer_imports to behave differently than expected.**

.. code-block:: python

    import defer_imports

    # Ex 1. Henceforth, instrument all import statements in other pure-Python modules
    # so that they are deferred. Off by default. If on, it has priority over any other
    # configuration passed in alongside it.
    #
    # Better suited for applications.
    defer_imports.install_import_hook(apply_all=True)

    # Ex 2. Henceforth, instrument all import statements *only* in modules whose names
    # are in the given sequence of strings.
    #
    # Better suited for libraries.
    with defer_imports.install_import_hook(module_names=(__name__,)):
        ...

    # Ex 3. Henceforth, instrument all import statements *only* in modules whose names
    # are in the given sequence *or* whose names indicate they are submodules of any
    # of the sequence members.
    #
    # In this case, the discord, discord.types, and discord.abc.other modules would all
    # be affected.
    #
    # Better suited for libraries.
    with defer_imports.install_import_hook(module_names=("discord",), recursive=True):
        ...


Example
-------

Assuming the path hook was registered normally (i.e. without providing any configuration), you can use the ``defer_imports.until_use`` context manager to decide which imports should be deferred. For instance:

.. code-block:: python

    import defer_imports

    with defer_imports.until_use:
        import inspect
        from typing import Final

    # inspect and Final won't be imported until referenced.

**WARNING: If the context manager is not used as defer_imports.until_use, it will not be instrumented properly. until_use by itself, aliases of it, and the like are currently not supported.**

If the path hook *was* registered with configuration, then within the affected modules, most module-level import statements will be instrumented. There are two supported exceptions: import statements within ``try-except-else-finally`` blocks and within non- ``defer_imports.until_use`` ``with`` blocks. Such imports are still performed eagerly. These "escape hatches" mostly match those described in PEP 690. 


Use Cases
---------

-   Anything that could benefit from overall decreased startup/import time if the symbols resulting from imports aren't used *at* import time.

    -   If one wants module-level, expensive imports that are rarely needed in common code paths.

        -   A good fit for this is a CLI tool and its subcommands.

    -   If imports are necessary to get symbols that are only used within annotations.

        -   Such imports can be unnecessarily expensive or cause import chains depending on how one's code is organized.
        -   The current workaround for this is to perform the problematic imports within ``if typing.TYPE_CHECKING: ...`` blocks and then stringify the fake-imported, nonexistent symbols to prevent NameErrors at runtime; however, the resulting annotations will raise errors if ever introspected. Using ``with defer_imports.until_use: ...`` instead would ensure that the symbols will be imported and saved in the local namespace, but only upon introspection, making the imports non-circular and almost free in most circumstances.


Features
========

-   Supports multiple Python runtimes/implementations.
-   Supports all syntactically valid Python import statements.
-   Cooperates with type-checkers like pyright and mypy.
-   Has an API for automatically instrumenting all valid import statements, not just those used within the provided context manager.

    -   Has escape hatches for eager importing: ``try-except-else-finally`` and ``with`` blocks.


Caveats
=======

-   Intentionally doesn't support deferred importing within class or function scope.
-   Eagerly loads wildcard imports.
-   May clash with other import hooks.

    -   Examples of popular packages using clashing import hooks: |typeguard|_, |beartype|_, |jaxtyping|_, |torchtyping|_, |pyximport|_
    -   It's possible to work around this by reaching into ``defer-imports``'s internals, combining its instrumentation machinery with that of another library's, then creating a custom import hook using that machinery, but such a scenario is currently not well-supported beyond ``defer_imports.install_import_hook()`` accepting a ``loader_class`` argument.

-   Can't automatically resolve deferred imports in a namespace when that namespace is being iterated over, leaving a hole in its abstraction.

    -   When using dictionary iteration methods on a dictionary or namespace that contains a deferred import key/proxy pair, the members of that pair will be visible, mutable, and will not resolve automatically. PEP 690 specifically addresses this by modifying the builtin ``dict``, allowing each instance to know if it contains proxies and then resolve them automatically during iteration (see the second half of its `"Implementation" section <https://peps.python.org/pep-0690/#implementation>`_ for more details). Note that qualifying ``dict`` iteration methods include ``dict.items()``, ``dict.values()``, etc., as well as the builtin functions ``locals()``, ``globals()``, ``vars()``, and ``dir()``.

        As of right now, nothing can be done about this using pure Python without massively slowing down ``dict``. Accordingly, users should try to avoid interacting with deferred import keys/proxies if encountered while iterating over module dictionaries; the result of doing so is not guaranteed.


Why?
====

Lazy imports alleviate several of Python's current pain points. Because of that, `PEP 690`_ was put forth to integrate lazy imports into CPython; see that proposal and the surrounding discussions for more information about the history, implementations, benefits, and costs of lazy imports.

Though that proposal was rejected, there are well-established third-party libraries that provide lazy import mechanisms, albeit with more constraints. Most do not have APIs as integrated or ergonomic as PEP 690's, but that makes sense; most predate the PEP and were not created with that goal in mind.

Existing libraries that do intentionally inject or emulate PEP 690's semantics and API don't fill my needs for one reason or another. For example, |slothy|_ (currently) limits itself to specific Python implementations by relying on the existence of call stack frames. I wanted to create something similar that relies on public implementation-agnostic APIs as much as possible.


How?
====

The core of this package is quite simple: when import statments are executed, the resulting values are special proxies representing the delayed import, which are then saved in the local namespace with special keys instead of normal string keys. When a user requests the normal string key corresponding to the import, the relevant import is executed and both the special key and the proxy replace themselves with the correct string key and import result. Everything stems from this.

The ``defer_imports.until_use`` context manager is what causes the proxies to be returned by the import statements: it temporarily replaces ``builtins.__import__`` with a version that will give back proxies that store the arguments needed to execute the *actual* import at a later time.

Those proxies don't use those stored ``__import__`` arguments themselves, though; the aforementioned special keys are what use the proxy's stored arguments to trigger the late import. These keys are aware of the namespace, the *dictionary*, they live in, are aware of the proxy they are the key for, and have overriden their ``__eq__`` and ``__hash__`` methods so that they know when they've been queried. In a sense, they're like descriptors, but instead of "owning the dot", they're "owning the brackets". Once such a key has been matched (i.e. someone uses the name of the import), it can use its corresponding proxy's stored arguments to execute the late import and *replace itself and the proxy* in the local namespace. That way, as soon as the name of the deferred import is referenced, all a user sees in the local namespace is a normal string key and the result of the resolved import.

The missing intermediate step is making sure these special proxies are stored with these special keys in the namespace. After all, Python name binding semantics only allow regular strings to be used as variable names/namespace keys; how can this be bypassed? ``defer-imports``'s answer is a little compile-time instrumentation. When a user calls ``defer_imports.install_import_hook()`` to set up the library machinery (see "Setup" above), what they are doing is installing an import hook that will modify the code of any given Python file that uses the ``defer_imports.until_use`` context manager. Using AST transformation, it adds a few lines of code around imports within that context manager to reassign the returned proxies to special keys in the local namespace (via ``locals()``).

With this methodology, we can avoid using implementation-specific hacks like frame manipulation to modify the locals. We can even avoid changing the contract of ``builtins.__import__``, which specifically says it does not modify the global or local namespaces that are passed into it. We may modify and replace members of it, but at no point do we change its size while within ``__import__`` by removing or adding anything.


Benchmarks
==========

There are currently a few ways of measuring activation and/or import time:

-   A local benchmark script for timing the import of a significant portion of the standard library.

    -   Invokable with ``python -m bench.bench_samples`` or ``hatch run bench:bench``.
    -   To prevent bytecode caching from impacting the benchmark, run with |python -B|_, which will set ``sys.dont_write_bytecode`` to ``True`` and cause the benchmark script to purge all existing ``__pycache__`` folders in the project directory.
    -   PyPy is excluded from the benchmark since it takes time to ramp up.
    -   An sample run across versions using ``hatch``:

        (Run once with ``__pycache__`` folders removed and ``sys.dont_write_bytecode=True``):

        ==============  =======  =============  ===================
        Implementation  Version  Benchmark      Time
        ==============  =======  =============  ===================
        CPython         3.9      regular        0.48585s (409.31x)
        CPython         3.9      slothy         0.00269s (2.27x)
        CPython         3.9      defer-imports  0.00119s (1.00x)
        \-\-            \-\-     \-\-           \-\-
        CPython         3.10     regular        0.41860s (313.20x)
        CPython         3.10     slothy         0.00458s (3.43x)   
        CPython         3.10     defer-imports  0.00134s (1.00x)
        \-\-            \-\-     \-\-           \-\-
        CPython         3.11     regular        0.60501s (279.51x)
        CPython         3.11     slothy         0.00570s (2.63x)
        CPython         3.11     defer-imports  0.00216s (1.00x)
        \-\-            \-\-     \-\-           \-\-
        CPython         3.12     regular        0.53233s (374.40x)
        CPython         3.12     slothy         0.00552s (3.88x)
        CPython         3.12     defer-imports  0.00142s (1.00x)   
        \-\-            \-\-     \-\-           \-\-
        CPython         3.13     regular        0.53704s (212.19x)
        CPython         3.13     slothy         0.00319s (1.26x)
        CPython         3.13     defer-imports  0.00253s (1.00x)
        ==============  =======  =============  ===================

-   Commands for only measuring import time of the library, using built-in Python timing tools like |timeit|_ and |python -X importtime|_.

    -   Examples::

            python -m timeit -n 1 -r 1 -- "import defer_imports"
            hatch run bench:import-time defer_imports
            python -X importtime -c "import defer_imports"
            hatch run bench:simple-import-time defer_imports

    -   Substitute ``defer_imports`` in the above commands with other modules, e.g. ``slothy``, to compare.
    -   The results can vary greatly between runs. If possible, only compare the resulting time(s) when collected from the same process.


Acknowledgements
================

The design of this library was inspired by the following:

-   |demandimport|_
-   |apipkg|_
-   |metamodule|_
-   |modutil|_
-   `SPEC 1 <https://scientific-python.org/specs/spec-0001/>`_ / |lazy-loader|_
-   `PEP 690`_ and its authors
-   `Jelle Zijlstra's pure-Python proof of concept <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_
-   |slothy|_
-   |ideas|_
-   `Sinbad <https://github.com/mikeshardmind>`_'s feedback

Without them, this would not exist.


..
    Common/formatted hyperlinks


.. _PEP 690: https://peps.python.org/pep-0690/

.. |timeit| replace:: ``timeit``
.. _timeit: https://docs.python.org/3/library/timeit.html

.. |python -B| replace:: ``python -B``
.. _python -B: https://docs.python.org/3/using/cmdline.html#cmdoption-B

.. |python -X importtime| replace:: ``python -X importtime``
.. _python -X importtime: https://docs.python.org/3/using/cmdline.html#cmdoption-X

.. |typeguard| replace:: ``typeguard``
.. _typeguard: https://github.com/agronholm/typeguard

.. |beartype| replace:: ``beartype``
.. _beartype: https://github.com/beartype/beartype

.. |jaxtyping| replace:: ``jaxtyping``
.. _jaxtyping: https://github.com/patrick-kidger/jaxtyping

.. |torchtyping| replace:: ``torchtyping``
.. _torchtyping: https://github.com/patrick-kidger/torchtyping

.. |pyximport| replace:: ``pyximport``
.. _pyximport: https://github.com/cython/cython/tree/master/pyximport

.. |demandimport| replace:: ``demandimport``
.. _demandimport: https://github.com/bwesterb/py-demandimport

.. |apipkg| replace:: ``apipkg``
.. _apipkg: https://github.com/pytest-dev/apipkg

.. |metamodule| replace:: ``metamodule``
.. _metamodule: https://github.com/njsmith/metamodule

.. |modutil| replace:: ``modutil``
.. _modutil: https://github.com/brettcannon/modutil

.. |lazy-loader| replace:: ``lazy-loader``
.. _lazy-loader: https://github.com/scientific-python/lazy-loader

.. |slothy| replace:: ``slothy``
.. _slothy: https://github.com/bswck/slothy

.. |ideas| replace:: ``ideas``
.. _ideas: https://github.com/aroberge/ideas
