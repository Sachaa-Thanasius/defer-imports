-   Test how nested layers of configuration for ast_rewrite actually interact. A few starting questions/notes:
    -   Does the uninstall option actually do what we want?
    -   Will the uninstall clobber everything every time it runs? Do we want that?
    -   What are the myriad of use cases this tries to solve, and how many of them can it actually solve?
        -   Inside a library:

            A configuration of (module_names=library_name, recursive=True) set in inside __init__.py should be enough, as
            long as the library's __init__.py itself contains no imports to defer. Maybe a mechanism could be added to
            determine the library name without spelling it out ("." passed in and then use frame hacking?), but it should
            be scoped appropriately as is.

            -   Example: astpretty

                Can this be made to work for libraries that *only* have __init__.py, or are otherwise a single-file module
                distribution? I don't think so; the module cannot easily change its loader mid-execution in a way that
                affects or restarts the execution, right?

                This might work if defer_imports is installed, activated via .pth file, and targets *everything*. That might
                be the only way, though.

            -   Example: packaging

                If a library doesn't reset the configuration, e.g. they just call import_hook().install() because they don't
                want to eagerly import submodules, the config hierarchy will break. It'll just persist until something else
                overrides it. Maybe import_hook should only be used as a context manager? But how is a library with an empty
                __init__.py but many submodules supposed to take advantage of this then?

                There's something here. Maybe pre-seeding sys.path_importer_cache for the package directory would work? But
                how does that interact with namespace packages then?

        -   Inside an app:

            A configuration of (apply_all=True) should be enough. You kinda want it to be indiscriminate (within reason) to
            get maximum gain.

-   Redo the configuration passing so more of the parameters are unified if possible, along the lines of what typeguard does.
    -   "." for the current module.
    -   ".*" for the current module and submodules recursively.
    -   "name" for the {name} module.
    -   "name.*" for the {name} module and submodules recursively.
    -   None or "" for all modules?