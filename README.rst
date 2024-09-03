=============
defer-imports
=============

.. image:: https://img.shields.io/github/license/Sachaa-Thanasius/defer-imports.svg
    :alt: License: MIT
    :target: https://opensource.org/licenses/MIT


A library that implements `PEP 690 <https://peps.python.org/pep-0690/>`_–esque lazy imports in pure Python, but at a user's behest within a context manager.


Installation
============

**Requires Python 3.9+**

This can be installed via pip::

    python -m pip install git@https://github.com/Sachaa-Thanasius/defer-imports

It can also easily be vendored, as it has zero dependencies and is less than 1,000 lines of code.


Usage
=====

Setup
-----

``defer-imports`` hooks into the Python import system with a path hook. That path hook needs to be registered before code using the import-delaying context manager, ``defer_imports.until_use``, is parsed. To do that, include the following somewhere such that it will be executed before your code:

.. code:: python

    import defer_imports

    defer_imports.install_defer_import_hook()


Example
-------

Assuming the path hook has been registered, you can use the ``defer_imports.until_use`` context manager to decide which imports should be deferred. For instance:

.. code:: python

    import defer_imports

    with defer_imports.until_use:
        import inspect
        from typing import Final

    # inspect and Final won't be imported until referenced.


Use Cases
---------

-   If imports are necessary to get symbols that are only used within annotations, but such imports would cause import chains.

    -   The current workaround for this is to perform the problematic imports within ``if typing.TYPE_CHECKING: ...`` blocks and then stringify the fake-imported, nonexistent symbols to prevent NameErrors at runtime; however, the resulting annotations raise errors on introspection. Using ``with defer_imports.until_use: ...`` instead would ensure that the symbols will be imported and saved in the local namespace, but only upon introspection, making the imports non-circular and almost free in most circumstances.

-   If expensive imports are only necessary for certain code paths that won't always be taken, e.g. in subcommands in CLI tools.


Features
========

-   Supports multiple Python runtimes/implementations.
-   Supports all syntactically valid Python import statements.
-   Doesn't break type-checkers like pyright and mypy.


Caveats
=======

-   Doesn't support deferred importing within class or function scope.
-   Doesn't support wildcard imports.
-   Doesn't have an API for giving users a choice to automatically defer all imports on a module, library, or application scale.
-   Has a relatively hefty one-time setup cost.


Why?
====

Lazy imports, in theory, alleviate several pain points that Python has currently. I'm not alone in thinking that: `PEP 690 <https://peps.python.org/pep-0690/>`_ was put forth to integrate lazy imports into CPython for that reason and explains the benefits much better than I can. While that proposal was rejected, there are other options in the form of third-party libraries that implement lazy importing, albeit with some constraints. Most do not have an API that is as general and ergonomic as what PEP 690 laid out, but they didn't aim to fill those shoes in the first place. Some examples:

-   `demandimport <https://github.com/bwesterb/py-demandimport>`_
-   `apipkg <https://github.com/pytest-dev/apipkg>`_
-   `modutil <https://github.com/brettcannon/modutil>`_
-   `metamodule <https://github.com/njsmith/metamodule/>`_
-   `SPEC 1 <https://scientific-python.org/specs/spec-0001/>`_ and its implementation, `lazy-loader <https://github.com/scientific-python/lazy-loader>`_
-   And countless more

Then along came `slothy <https://github.com/bswck/slothy>`_, a library that seems to do it better, having been constructed with feedback from multiple CPython core developers as well as one of the minds behind PEP 690. It was the main inspiration for this project. However, the library (currently) limits itself to specific Python implementations by relying on the existence of frames that represent the call stack. For many use cases, that's perfectly fine; PEP 690's implementation was for CPython specifically, and to my knowledge, some of the most popular Python runtimes outside of CPython provide call stack access in some form. Still, I thought that there might be a way to do something similar while avoiding such implementation-specific APIs. After feedback and discussion, that thought crystalized into this library.


How?
====

The core of this package is quite simple: when import statments are executed, the resulting values are special proxies representing the delayed import, which are then saved in the local namespace with special keys instead of normal string keys. When a user requests the normal string key corresponding to the import, the relevant import is executed and both the special key and the proxy replace themselves with the correct string key and import result. Everything stems from this.

The ``defer_imports.until_use`` context manager is what causes the proxies to be returned by the import statements: it temporarily replaces ``builtins.__import__`` with a version that will give back proxies that store the arguments needed to execute the *actual* import at a later time.

Those proxies don't use those stored ``__import__`` arguments themselves, though; the aforementioned special keys are what use the proxy's stored arguments to trigger the late import. These keys are aware of the namespace, the *dictionary*, they live in, are aware of the proxy they are the key for, and have overriden their ``__eq__`` and ``__hash__`` methods so that they know when they've been queried. In a sense, they're like descriptors, but instead of "owning the dot", they're "owning the brackets". Once such a key has been matched (i.e. someone uses the name of the import), it can use its corresponding proxy's stored arguments to execute the late import and *replace itself and the proxy* in the local namespace. That way, as soon as the name of the deferred import is referenced, all a user sees in the local namespace is a normal string key and the result of the resolved import.

The missing intermediate step is making sure these special proxies are stored with these special keys in the namespace. After all, Python name binding semantics only allow regular strings to be used as variable names/namespace keys; how can this be bypassed? ``defer-imports``'s answer is a little compile-time instrumentation. When a user calls ``defer_imports.install_deferred_import_hook()`` to set up the library machinery (see "Setup" above), what they are actually doing is installing an import hook that will modify the code of any given Python file that uses the ``defer_imports.until_use`` context manager. Using AST transformation, it adds a few lines of code around imports within that context manager to reassign the returned proxies to special keys in the local namespace (via ``locals()``).

With this methodology, we can avoid using implementation-specific hacks like frame manipulation to modify the locals. We can even avoid changing the contract of ``builtins.__import__``, which specifically says it does not modify the global or local namespaces that are passed into it. We may modify and replace members of it, but at no point do we change its size while within ``__import__`` by removing or adding anything.


Benchmarks
==========

A bit rough, but there are currently two ways of measuring activation and/or import time:

-   ``python -m benchmark.bench_samples`` (run with ``--help`` to see more information)

    -   To prevent bytecode caching from impacting the benchmark, run with `python -B <https://docs.python.org/3/using/cmdline.html#cmdoption-B>`_, which will set ``sys.dont_write_bytecode`` to ``True`` and cause the benchmark script to purge all existing ``__pycache__`` folders in the project directory.
    -   PyPy is excluded from the benchmark since it takes time to ramp up. 
    -   The cost of registering ``defer-imports``'s import hook is ignored since that is a one-time startup cost that will hopefully be reduced in time.
    -   An sample run across versions using ``hatch run benchmark:bench``:

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

-   ``python -m timeit -n 1 -r 1 -- "import defer_imports"``

    -   Substitute ``defer_imports`` with other modules, e.g. ``slothy``, to compare.
    -   This has great variance, so only value the resulting time relative to another import's time in the same process if possible.


Acknowledgements
================

-   All the packages mentioned in "Why?" above, for providing inspiration.
-   `PEP 690 <https://peps.python.org/pep-0690/>`_ and its authors, for pushing lazy imports to the point of almost being accepted as a core part of CPython's import system.
-   Jelle Zijlstra, for so easily creating and sharing a `sample implementation <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_ that ``slothy`` and ``defer-imports`` are based on.
-   `slothy <https://github.com/bswck/slothy>`_, for being a major reference and inspiration for this project.
-   Sinbad, for all his feedback.
