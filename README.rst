========
deferred
========

|License| |Pyright| |Ruff| |pre-commit|

.. |License| image:: https://img.shields.io/github/license/Sachaa-Thanasius/deferred.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. |Pyright| image:: https://img.shields.io/badge/pyright-checked-informational.svg
    :target: https://github.com/microsoft/pyright/
    :alt: Type-checker: Pyright

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Linter and Formatter: Ruff

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

An pure-Python implementation of PEP 690–esque lazy imports, but at a user's behest within a ``defer_imports_until_use`` context manager.


Installation
============

**Requires Python 3.9+**

This can be installed via pip:

.. code:: sh

    python -m pip install git@https://github.com/Sachaa-Thanasius/deferred


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

Lazy imports, in theory, alleviate several pain points that Python has currently. I'm not alone in thinking that; `PEP 690 <https://peps.python.org/pep-0690/>`_ was put forth to integrate lazy imports into the language for that reason and explains the benefits much better than I can. While that was rejected, there are other options in the form of third-party libraries that implement lazy importing with some constraints. Most do not have an API that is as general and ergonomic as what PEP 690 laid out, but they didn't aim to fill those shoes in the first place (e.g. `demandimport <https://github.com/bwesterb/py-demandimport>`_, `apipkg <https://github.com/pytest-dev/apipkg>`_, `modutil <https://github.com/brettcannon/modutil>`_, `SPEC 1 <https://scientific-python.org/specs/spec-0001/>`_, and more).

Then along came `slothy <https://github.com/bswck/slothy>`_, a library that does it better, having been constructed with feedback from multiple CPython core developers as well as one of the minds behind PEP 690. Its core concept is powerful, and it's the main inspiration for this project. However, the library also ties itself to specific Python runtimees by depending on the existence of frames. While that's fine — PEP 690 was for CPython, after all — I thought, after discussion with and feedback from others, that there was a way that could be less implementation dependent, more "pure", and thus might be more maintainable in the long run. Thus, ``deferred``.


Acknowledgements
================

-   `PEP 690 <https://peps.python.org/pep-0690/>`_, for pushing this feature to the point of almost being accepted as a fundamental part of CPython
-   Jelle Zijlstra, for so easily creating the core concept that ``slothy`` and now ``deferred`` rely on and sharing it in a `gist <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_.
-   `slothy <https://github.com/bswck/slothy>`_, for making something great with that concept.
-   All the packages mentioned in "Why" above, for filling people's needs and laying the groundwork for what's come.
-   Sinbad, for the feedback and for pushing me towards a hybrid approach.
