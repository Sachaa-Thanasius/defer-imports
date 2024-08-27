========
deferred
========

|License| |Pyright| |Ruff| |pre-commit|

.. |License| image:: https://img.shields.io/github/license/Sachaa-Thanasius/deferred.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. |Pyright| image:: https://img.shields.io/badge/pyright-checked-informational.svg
    :target: https://github.com/microsoft/pyright/
    :alt: Pyright

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

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

The methodology is somewhat rough: at the moment, time to import is being measured with both the ``benchmark/bench_samples.py`` script (run with ``python -m benchmark.bench_samples``) and python's importtime command-line function (e.g. run with ``python -X importtime -c "import deferred"``).


Why?
====

I wasn't satisfied with the state of lazy imports in Python and wanted to put my own spin on it while avoiding CPython implementation details as much as possible.


Acknowledgements
================

-   Thanks to PEP 690 for pushing this feature and two pure-Python pieces of code for serving as starting points and references.

    -   `PEP 690 <https://peps.python.org/pep-0690/>`_
    -   `Jelle's lazy gist <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_
    -   `slothy <https://github.com/bswck/slothy>`_ (based on the previous gist)

-   Thanks to Sinbad for the feedback and for unintentionally pushing me towards this approach.
