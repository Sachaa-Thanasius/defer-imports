========
deferred
========

|CI Status| |License|

.. |CI Status| image:: https://github.com/Sachaa-Thanasius/deferred/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/Sachaa-Thanasius/deferred/actions/workflows/ci.yml
    :alt: CI Status

.. |License| image:: https://img.shields.io/github/license/Sachaa-Thanasius/deferred.svg
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


Features and Caveats
--------------------

-   Python implementation–agnostic, in theory

    -   The main dependency is on ``locals()`` at module scope to maintain its current API: specifically, that its return value will be a read-through, *write-through*, dict-like view of the module locals.

-   Not that slow, especially with support from bytecode caching
-   Doesn't support lazy importing in class or function scope
-   Doesn't support wildcard imports


Benchmarks
==========

There are two ways of measuring activation and/or import time currently:

-   ``python -m benchmark.bench_samples`` (run with ``--help`` to see more information)
-   ``python -X importtime -c "import deferred"`` (substitute ``deferred`` with other modules, e.g. ``slothy``, to compare)


Why?
====

Lazy imports, in theory, alleviate several pain points that Python has currently. I'm not alone in thinking that; `PEP 690 <https://peps.python.org/pep-0690/>`_ was put forth to integrate lazy imports into CPython for that reason and explains the benefits much better than I can. While that was rejected, there are other options in the form of third-party libraries that implement lazy importing with some constraints. Most do not have an API that is as general and ergonomic as what PEP 690 laid out, but they didn't aim to fill those shoes in the first place. Some examples:

-   `demandimport <https://github.com/bwesterb/py-demandimport>`_
-   `apipkg <https://github.com/pytest-dev/apipkg>`_
-   `modutil <https://github.com/brettcannon/modutil>`_
-   `SPEC 1 <https://scientific-python.org/specs/spec-0001/>`_
-   And countless more.

Then along came `slothy <https://github.com/bswck/slothy>`_, a library that seems to do it better, having been constructed with feedback from multiple CPython core developers as well as one of the minds behind PEP 690. It was the main inspiration for this project. However, the library also (currently) ties itself to specific Python implementations by depending on the existence of frames that represent the call stack. That's perfectly fine; PEP 690's implementation was for CPython specifically, and to my knowledge, the most popular Python runtimes provide call stack access in some form. Still, I thought that there might be a way to do something similar while remaining implementation-independent, avoiding as many internal APIs as possible. After feedback and discussion, that thought crystalized into this library.


Acknowledgements
================

-   `PEP 690 <https://peps.python.org/pep-0690/>`_ and its authors, for pushing lazy imports to the point of almost being accepted as a core part of CPython's import system.
-   Jelle Zijlstra, for so easily creating and sharing a `sample implementation <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_ of the core concept that ``slothy`` and now ``deferred`` use.
-   `slothy <https://github.com/bswck/slothy>`_, for making something great with that concept.
-   All the packages mentioned in "Why" above, for filling people's needs and laying the groundwork for what's come.
-   Sinbad, for the feedback.
