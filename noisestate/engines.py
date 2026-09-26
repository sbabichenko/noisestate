"""The two engines by name, and the construction solve() and sweep() share.

    from noisestate import engines
    S = engines.stationary(model, settings={"anderson_m": 10}); res = S.solve(tol=1e-12)
    S = engines.solver(model, Numerics(nodes=24), past=old)

`stationary` is the age-grid engine (StationarySolver), `spectral` the piecewise-spectral triangle
(SpectralFiniteSolver: finite horizons and transitions).  The first-order uniform-cell cross-check lives
outside the package, in extras/cells.py.

THE PUBLIC SURFACE IS THE TWO ENGINES AND `solver`.  `solver(model, numerics, ...)` resolves the
numerics against the model (the engine from the horizon kind unless given, the model's own grid
fields unless overridden), lays them on the model so the engine reads them, and returns the engine.
`_build` beneath it also returns the RESOLVED NUMERICS, which the solve path records and nobody
constructing an engine by hand wants; it is internal machinery and named accordingly."""
from __future__ import annotations

from typing import Tuple

from .numerics import Numerics
from .spec import Model
from .stationary import StationarySolver as stationary
from .finite_spectral import SpectralFiniteSolver as spectral

ENGINE_CLASSES = {"stationary": stationary, "spectral": spectral}

__all__ = ["stationary", "spectral", "ENGINE_CLASSES", "solver", "default_start"]


def default_start(S, start_policy=None) -> str:
    """The start of a solve when none is given: "stationary" on an engine with a stationary continuation (a
    transition continued after T: its maps are where a settled transition ends), else "zero"."""
    if start_policy is not None:
        return start_policy
    return "stationary" if getattr(getattr(S, "c", None), "cont", None) is not None else "zero"


def _build(model: Model, numerics=None, *, verbose: bool = False, past=None, continuation=None) -> Tuple[object, Numerics]:
    """The engine for `model` under `numerics` (a Numerics, a dict of its fields or None), constructed; and
    the resolved Numerics it runs with.  past and continuation belong to the spectral engine; giving them to
    another engine is a TypeError."""
    given = Numerics.of(numerics)
    num = model.numerics.merged(given).resolved(model.horizon.kind)
    if num != model.numerics.resolved(model.horizon.kind):
        model = model.with_numerics(num)                     # the engines read the grid off model.numerics
    if num.engine not in ENGINE_CLASSES:
        raise ValueError(f"unknown engine {num.engine!r}; one of {sorted(ENGINE_CLASSES)}")
    kw = {"verbose": verbose}
    if num.settings.changed():
        kw["settings"] = num.settings
    if num.engine == "stationary":
        if past is not None or continuation is not None:
            raise TypeError("past= and continuation= belong to a finite horizon or a transition (the spectral engine), not the stationary one")
    else:
        if num.engine == "spectral":
            if past is not None:
                kw["past"] = past
            if continuation is not None:
                kw["continuation"] = continuation
    return ENGINE_CLASSES[num.engine](model, **kw), num


def solver(model: Model, numerics=None, **kw):
    """The engine `model`'s numerics select, constructed and returned.

    THE PUBLIC FACTORY, and the only one.  _build() beneath it returns the engine AND the resolved
    Numerics -- the solve path records those, and a caller constructing an engine by hand does not
    want them -- so it is internal.  The specification named one function where there are two;
    this is the public half.  `settings` is accepted as an alias of numerics.settings.
    """
    if "settings" in kw:
        numerics = Numerics.of(numerics).merged(Numerics(settings=kw.pop("settings")))
    return _build(model, numerics, **kw)[0]

