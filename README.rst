========
deferred
========

.. image:: https://github.com/Sachaa-Thanasius/deferred/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/Sachaa-Thanasius/deferred/actions/workflows/ci.yml
   :alt: CI Status

.. image:: https://img.shields.io/github/license/Sachaa-Thanasius/deferred.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

An pure-Python implementation of `PEP 690 <https://peps.python.org/pep-0690/>`_–esque lazy imports, but at a user's behest within a ``defer_imports_until_use`` context manager.


Installation
============

**Requires Python 3.9+**

This can be installed via pip:

.. code:: sh

    python -m pip install git@https://github.com/Sachaa-Thanasius/deferred

Additionally, ``deferred`` can easily be vendored; it has zero dependencies and is fairly small (less than 1,000 lines of code).


Example
=======

.. code:: python

    from deferred import defer_imports_until_use

    with defer_imports_until_use:
        import inspect
        from typing import TypeVar

    # inspect and TypeVar won't be imported until referenced. For imports that are only used for annotations,
    # this import cost can be avoided entirely by making sure all annotations are strings.


Setup
=====
``deferred`` hooks into the Python import system with a path hook. That path hook needs to be registered before code using ``defer_imports_until_use`` is executed. To do that, include the following somewhere such that it will be executed before your code:

.. code:: python

    import deferred

    deferred.install_defer_import_hook()


Documentation
=============

See this README as well as docstrings and comments in the code.


Features
--------

-   Python implementation–agnostic, in theory

    -   The main dependency is on ``locals()`` at module scope to maintain its current API: specifically, that its return value will be a read-through, *write-through*, dict-like view of the module locals.

-   Supports all valid Python import statements.


Caveats
-------

-   Doesn't support lazy importing in class or function scope
-   Doesn't support wildcard imports
-   (WIP) Has an initial setup cost that could be smaller. 


Benchmarks
==========

There are two ways of measuring activation and/or import time currently:

-   ``python -m benchmark.bench_samples`` (run with ``--help`` to see more information)

    -   To prevent bytecode caching from impacting the benchmark, run with `python -B <https://docs.python.org/3/using/cmdline.html#cmdoption-B>`_, which will set ``sys.dont_write_bytecode`` to ``True``.
    -   PyPy is excluded from the benchmark since it takes time to ramp up. 
    -   This benchmark excludes the cost of registering ``deferred``'s import hook since that has a one-time startup cost that will hopefully be reduced in time. 
    -   Results after one run: (Run once with ``__pycache__`` folders removed and ``sys.dont_write_bytecode=True``):

        ==============  =======  ==========  ===================
        Implementation  Version  Benchmark   Time
        ==============  =======  ==========  ===================
        CPython         3.9      regular     0.48585s (409.31x)
        CPython         3.9      slothy      0.00269s (2.27x)
        CPython         3.9      deferred    0.00119s (1.00x)
        \-\-            \-\-     \-\-        \-\-
        CPython         3.10     regular     0.41860s (313.20x)
        CPython         3.10     slothy      0.00458s (3.43x)   
        CPython         3.10     deferred    0.00134s (1.00x)
        \-\-            \-\-     \-\-        \-\-
        CPython         3.11     regular     0.60501s (279.51x)
        CPython         3.11     slothy      0.00570s (2.63x)
        CPython         3.11     deferred    0.00216s (1.00x)
        \-\-            \-\-     \-\-        \-\-
        CPython         3.12     regular     0.53233s (374.40x)
        CPython         3.12     slothy      0.00552s (3.88x)
        CPython         3.12     deferred    0.00142s (1.00x)   
        \-\-            \-\-     \-\-        \-\-
        CPython         3.13     regular     0.53704s (212.19x)
        CPython         3.13     slothy      0.00319s (1.26x)
        CPython         3.13     deferred    0.00253s (1.00x)
        ==============  =======  ==========  ===================

-   ``python -m timeit -n 1 -r 1 -- "import deferred"``

    -   Substitute ``deferred`` with other modules, e.g. ``slothy``, to compare.
    -   This has great variance, so only value the resulting time relative to another import's time in the same process if possible.


Why?
====

Lazy imports, in theory, alleviate several pain points that Python has currently. I'm not alone in thinking that; `PEP 690 <https://peps.python.org/pep-0690/>`_ was put forth to integrate lazy imports into CPython for that reason and explains the benefits much better than I can. While that was rejected, there are other options in the form of third-party libraries that implement lazy importing, albeit with some constraints. Most do not have an API that is as general and ergonomic as what PEP 690 laid out, but they didn't aim to fill those shoes in the first place. Some examples:

-   `demandimport <https://github.com/bwesterb/py-demandimport>`_
-   `apipkg <https://github.com/pytest-dev/apipkg>`_
-   `modutil <https://github.com/brettcannon/modutil>`_
-   `SPEC 1 <https://scientific-python.org/specs/spec-0001/>`_
-   And countless more.

Then along came `slothy <https://github.com/bswck/slothy>`_, a library that seems to do it better, having been constructed with feedback from multiple CPython core developers as well as one of the minds behind PEP 690. It was the main inspiration for this project. However, the library (currently) also ties itself to specific Python implementations by depending on the existence of frames that represent the call stack. That's perfectly fine; PEP 690's implementation was for CPython specifically, and to my knowledge, the most popular Python runtimes provide call stack access in some form. Still, I thought that there might be a way to do something similar while remaining implementation-independent, avoiding as many internal APIs as possible. After feedback and discussion, that thought crystalized into this library.


How?
====
The core of this package is quite simple: when import statments are executed, the resulting values are special proxies representing the delayed import, which are then saved in the local namespace with special keys instead of normal string keys. When a user requests the normal string key corresponding to the import, the relevant import is executed and both the special key and the proxy replace themselves with the correct string key and import result. Everything stems from this.

The ``defer_imports_until_used`` context manager is what causes the proxies to be returned by the import statements: it temporarily replaces ``builtins.__import__`` with a version that will give back proxies that store the arguments needed to execute the *actual* import at a later time.

Those proxies don't use those stored ``__import__`` arguments themselves, though; the aforementioned special keys are what use the proxy's stored arguments to trigger the late import. These keys are aware of the namespace, the *dictionary*, they live in, are aware of the proxy they are the key for, and have overriden their ``__eq__`` and ``__hash__`` methods so that they know when they've been queried. In a sense, they're almost like descriptors, but instead of "owning the dot", they're "owning the brackets". Once they've been matched (i.e. someone uses the name of the import), they can use the proxy's stored ``__import__`` arguments to execute the late import and *replace themselves* in the local namespace. That way, as soon as the name of the deferred import is referenced, all a user sees in the local namespace is a normal string key and the result of the resolved import.

The final step is actually assigning these special proxies to the special keys. After all, Python name binding semantics only allow regular strings to be used as variable names/namespace keys; how can this be bypassed? Well, this is where a little bit of instrumentation comes into play. When a user calls ``deferred.install_deferred_import_hook()`` to set up the ``deferred`` machinery (see :ref:`Setup`), what they are actually doing is installing an import hook that will modify the code of any given Python file that users the ``defer_imports_until_use`` context manager. It adds a few lines of code such that the return values of imports within the context manager are reassigned to special keys in the local namespace, accessed and modified via ``locals()``. With this method, we can avoid using frame hacks to modify the locals and even avoid changing the contract of ``builtins.__import__``, which specifically says it does not modify the global or local namespaces that are passed into it.


Acknowledgements
================

-   All the packages mentioned in "Why" above.
-   `PEP 690 <https://peps.python.org/pep-0690/>`_ and its authors, for pushing lazy imports to the point of almost being accepted as a core part of CPython's import system.
-   Jelle Zijlstra, for so easily creating and sharing a `sample implementation <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_ that ``slothy`` and ``deferred`` are based on.
-   `slothy <https://github.com/bswck/slothy>`_, for inspiring this project.
-   Sinbad, for the initial feedback.
