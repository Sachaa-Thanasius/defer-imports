========
deferred
========

An pure-Python implementation of PEP 690–esque lazy imports, but at a user's behest within the ``defer_imports_until_use`` context manager.


Installation
============

**Requires Python 3.9+**

This can be installed via pip:

.. code:: sh

    python -m pip install git@https://github.com/Sachaa-Thanasius/deferred


Documentation
=============

See this README as well as docstrings and comments in the code.


Features and Caveats
--------------------

- Python implementation–agnostic, in theory

    - The only dependency is on globals() at module scope to maintain its current API: specifically, that its return value will be a read-through, *write-through*, dict-like view of the module globals

- Not that slow, especially with support from bytecode caching
- Doesn't support lazy importing in class or function scope
- Doesn't support wildcard imports


Benchmarks
==========

These are quite rough. At the moment, time to import is being measured with the following command, with variations for different module names (e.g. bench_slothy for ``slothy``):

.. code:: sh

    python -X importtime -c "import benchmark.bench_defer"


TODO
====

- [x] Investigate if this package would benefit from a custom optimization suffix for bytecode. UPDATE: Added in a different way without monkeypatching, thanks to `this blog post <https://gregoryszorc.com/blog/2017/03/13/from-__past__-import-bytes_literals/>`_.

    - Signs point to yes, but I'm not a fan of the monkeypatching seemingly involved, nor of having to import ``importlib.util``.
    - See beartype and its justification `for <https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/_clawimpload.py#L177-L312>`_ `this <https://github.com/beartype/beartype/blob/e9eeb4e282f438e770520b99deadbe219a1c62dc/beartype/claw/_importlib/clawimpcache.py#L22-L26>`_.

- [x] Fix subpackage imports being broken if done within ``defer_imports_until_use`` like this:

    .. code:: python

        from deferred import defer_imports_until_use

        with defer_imports_until_use:
            import importlib
            import importlib.abc
            import importlib.util

    - I could not explain to you why it works, but it does.

- [ ] Add tests for the following:

    - [ ] Relative imports
    - [ ] Combinations of different import types
    - [ ] True circular imports
    - [ ] Thread safety (see importlib.util.LazyLoader for reference?)
    - [ ] Other python implementations/platforms

- [ ] Make this able to import the entire standard library, including all the subpackage imports uncommented (see benchmark/all_stdlib_modules.py)
- [ ] Make this be able to run on normal code. It currently breaks pip, readline, and who knows what else in the standard library, possibly because of the subpackage imports issue.


Acknowledgements
================

- This wouldn't have been possible without using PEP 690 and two other pieces of code as starting points.

    - `PEP 690 <https://peps.python.org/pep-0690/>`_
    - `Jelle's lazy gist <https://gist.github.com/JelleZijlstra/23c01ceb35d1bc8f335128f59a32db4c>`_
    - `slothy <https://github.com/bswck/slothy>`_ (also based on the previous gist)

- Thanks to Sinbad for the feedback and for unintentionally pushing me towards this approach.
