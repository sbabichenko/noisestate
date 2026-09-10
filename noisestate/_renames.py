"""The 0.7 renames, with every old spelling still working and naming what replaced it.

0.6.9 removed the 0.5 spellings outright, which is right for a spelling nobody should still be
using.  These are different: they are renames of names that were correct, just inconsistent, so
they warn rather than raise.  The old name keeps doing the right thing until 0.8 and the message
carries the new one.

The conventions the renames settle, so a new name can be placed without looking it up:

    require_*   raises when the answer is no          require_ok, require_converged
    *_ok        returns a bool, never raises          resolution_ok
    has_/drives_ a predicate on stored data           has_means, drives_means
    with_*      a changed copy of a model             with_params, with_finite
    to_*        another representation                to_dict
    verb        computes something new                solve, refine, describe, validate
    noun        stored data                           costs, kernels, status
"""
import warnings

SINCE = "0.7"
REMOVED_IN = "0.8"


def warn(old: str, new: str) -> None:
    warnings.warn(f"{old} was renamed {new} in {SINCE} and is removed in {REMOVED_IN}; use {new}",
                  DeprecationWarning, stacklevel=3)


def _qualified(old: str, new: str) -> str:
    """`new` under the same owner as `old`, so the message reads "Model.owner() ... use Model.owner_of"
    rather than trailing a bare attribute name the reader has to place."""
    owner = old.split(".", 1)[0]
    return f"{owner}.{new}" if "." in old else new


def renamed_method(old: str, new: str):
    """The old name as a method that warns once per call site and delegates to the new one."""
    def call(self, *args, **kwargs):
        warn(old, _qualified(old, new))
        return getattr(self, new)(*args, **kwargs)
    call.__name__ = old.rsplit(".", 1)[-1].rstrip("()")
    call.__doc__ = f"Deprecated in {SINCE}, removed in {REMOVED_IN}: use ``{new}``."
    return call


def renamed_property(old: str, new: str):
    """The old name as a read-only property that warns and reads the new one."""
    def get(self):
        warn(old, _qualified(old, new))
        return getattr(self, new)
    return property(get, doc=f"Deprecated in {SINCE}, removed in {REMOVED_IN}: use ``{new}``.")


def renamed_module_attr(module_globals: dict, renames: dict, fallback=None):
    """A PEP 562 ``__getattr__`` serving the old top-level names from `renames` {old: new}.

    `fallback` is the module's previous ``__getattr__``, and anything not renamed here goes to it.
    Chaining rather than replacing matters: the package already had one, which explains the names
    0.6 removed instead of raising a bare AttributeError, and assigning over it silently threw
    those explanations away.
    """
    def __getattr__(name):
        if name in renames:
            warn(f"noisestate.{name}", f"noisestate.{renames[name]}")
            return module_globals[renames[name]]
        if fallback is not None:
            return fallback(name)
        raise AttributeError(f"module 'noisestate' has no attribute {name!r}")
    return __getattr__
