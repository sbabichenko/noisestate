"""The three engines by name, and the construction solve() and sweep() share.

    from noisestate import engines
    S = engines.stationary(model, settings={"anderson_m": 10}); res = S.solve(tol=1e-12)
    S, numerics = engines.build(model, Numerics(nodes=24), past=old)

`stationary` is the age-grid engine (StationarySolver), `spectral` the piecewise-spectral triangle
(SpectralFiniteSolver: finite horizons and transitions), `cells` the first-order uniform-cell cross-check
(FiniteSolver).  `build(model, numerics, ...)` resolves the numerics against the model (the engine from the
horizon kind unless given, the model's own grid fields unless overridden), lays them on the model so the
engine reads them, and constructs the engine with the options that belong to it."""
from __future__ import annotations

from typing import Optional, Tuple

from .numerics import Numerics
from .spec import Model
from .stationary import StationarySolver as stationary
from .finite_spectral import SpectralFiniteSolver as spectral
from .finite import FiniteSolver as cells

ENGINE_CLASSES = {"stationary": stationary, "spectral": spectral, "cells": cells}

__all__ = ["stationary", "spectral", "cells", "ENGINE_CLASSES", "solver", "build", "default_start"]


def default_start(S, start_policy=None) -> str:
    """The start of a solve when none is given: "stationary" on an engine with a stationary continuation (a
    transition continued after T: its maps are where a settled transition ends), else "zero"."""
    if start_policy is not None:
        return start_policy
    return "stationary" if getattr(getattr(S, "c", None), "cont", None) is not None else "zero"


def build(model: Model, numerics=None, *, verbose: bool = False, naive_observers: Optional[dict] = None,
          past=None, continuation=None) -> Tuple[object, Numerics]:
    """The engine for `model` under `numerics` (a Numerics, a dict of its fields or None), constructed; and
    the resolved Numerics it runs with.  naive_observers belongs to the stationary engine, past and
    continuation to the spectral one; giving either to another engine is a TypeError."""
    given = Numerics.of(numerics)
    num = model.numerics.merged(given).resolved(model.horizon.kind)
    if num != model.numerics.resolved(model.horizon.kind):
        model = model.with_numerics(num)                     # the engines read the grid off the model's horizon
    if num.engine not in ENGINE_CLASSES:
        raise ValueError(f"unknown engine {num.engine!r}; one of {sorted(ENGINE_CLASSES)}")
    kw = {"verbose": verbose}
    if num.settings.changed():
        kw["settings"] = num.settings
    if num.engine == "stationary":
        if past is not None or continuation is not None:
            raise TypeError("past= and continuation= belong to a finite horizon or a transition (the spectral engine), not the stationary one")
        if naive_observers is not None:
            kw["naive_observers"] = naive_observers
    else:
        if naive_observers is not None:
            raise TypeError("naive_observers= belongs to the stationary engine")
        if num.engine == "spectral":
            if past is not None:
                kw["past"] = past
            if continuation is not None:
                kw["continuation"] = continuation
        elif past is not None or continuation is not None:
            raise TypeError("the cell engine has no past or continuation; use numerics.engine 'spectral'")
    return ENGINE_CLASSES[num.engine](model, **kw), num


def solver(model: Model, numerics=None, **kw):
    """The engine `model`'s numerics select, constructed and returned.

    THE PUBLIC FACTORY.  build() below it is the internal two-value form -- it returns the engine
    AND the resolved Numerics, which the solve path needs and a caller constructing an engine by
    hand does not.  The specification named build() for both; they are different functions, and
    this is the one to reach for.  `settings` is accepted as an alias of numerics.settings.
    """
    if "settings" in kw:
        numerics = Numerics.of(numerics).merged(Numerics(settings=kw.pop("settings")))
    return build(model, numerics, **kw)[0]

