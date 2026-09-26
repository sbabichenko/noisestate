"""Parameter sweeps with warm starts, and JSON-ready results.

    from noisestate import sweep
    rows = sweep("examples/ch4_kyle_back.yaml", "eps", [0.2, 0.1, 0.05, 0.02])
    rows[0].result.summary(); rows[0].result.to_dict(); rows[0].seconds

Each point starts from the previous point's equilibrium (raw maps for the stationary engine,
action kernels for the spectral finite engine), which is what makes a slider in an interactive
front end cheap: after the first point, a step costs a few best responses.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import yaml

from .spec import Model
from .past import Past
from .results import TriangleResult
from . import engines
from .engines import default_start


#  The two horizon lengths a sweep can move.  They are separate parameters because they are
#  separate quantities: L is how far back a kernel is carried, T is when the game ends.
HORIZON_LENGTHS = ("horizon.window", "horizon.T")


def _load_dict(model: Union[str, dict, Model]) -> dict:
    if isinstance(model, str):
        # through load(): a file's name default and its relative horizon.past.model, resolved from its own directory
        from . import load
        return load(model).to_dict()
    if isinstance(model, Model):
        return model.to_dict()
    if not isinstance(model, dict):
        raise TypeError(f"expected a Model, a dict or a path, not {type(model).__name__}")
    return copy.deepcopy(model)



def warm_start(prev) -> Optional[dict]:
    """The object the next solve should start from, for a previous result of the same engine."""
    if prev is None:
        return None
    if isinstance(prev, TriangleResult):
        c = prev.compiled
        return {a.name: np.stack([prev.world[c.block(u)] for u in a.controls]) for a in prev.model.agents}
    return prev.maps


@dataclass(frozen=True)
class SweepPoint:
    """One point of a sweep: the model at one parameter value, and what solving it there took.

    A CONTINUATION point, not a comparison scenario -- it carries the predictor and branch fields
    (`change`, `jump`) that only make sense along a path.  It shares CONVENTIONS with
    ScenarioResult, not identity: both are frozen dataclasses, both carry result / seconds /
    evaluations / converged under those names, and both expose to_dict().  Making them one type
    would merge two different objects; giving them different field names for the same thing would
    make a reader check which is which.
    """
    param: str
    value: float
    result: object
    seconds: float
    evaluations: int
    converged: bool
    change: Optional[float] = None      # strategy change from the previous point on the same grid
    jump: bool = False                  # that change far above the sweep's typical: a branch jump?

    def to_dict(self) -> dict:
        return {"param": self.param, "value": self.value, "seconds": self.seconds,
                "evaluations": self.evaluations, "converged": self.converged,
                "change": self.change, "jump": self.jump, "result": self.result.to_dict()}


def sweep(model: Union[str, dict, Model], param: str, values: Iterable[float], numerics=None,
          solver_kw: Optional[dict] = None, solve_kw: Optional[dict] = None, verbose: bool = False) -> "List[SweepPoint]":
    """Solve the model at each value of `param` (a key of `params`, or "horizon.window" / "horizon.T": the
    stationary model, the horizon T of a finite one or of a transition, which then warm-starts each point from
    the previous maps read on the new grid, the stationary maps beyond it), warm-starting each point from
    the linear extrapolation of the last two equilibria in the parameter (a secant predictor;
    markedly more robust at hard points such as a small trading cost).  numerics (a Numerics or a dict of
    its fields) is laid over the model's own at every point; solver_kw goes to each point's engine (verbose,
    past, continuation), solve_kw to each point's solve() (tol, max_evaluations, deadline,
    progress, diagnostics, start); a point stopped at a bound is a row with converged False, and the sweep goes on.
    Returns [{"param", "value", "result", "seconds", "evaluations", "converged", "change", "jump"}] in the
    given order; "change" is the relative change of the strategy from the previous point on the same grid
    (the raw maps; the action kernels on the finite spectral engine)
    and "jump" flags a change more than five times the sweep's median (a possible branch jump).  A transition
    model's past is solved once and shared by every point (solver_kw={"past": ...} gives it); its
    continuation, the new model's stationary equilibrium, is solved at each point."""
    base = _load_dict(model)
    if param not in HORIZON_LENGTHS and param not in (base.get("params") or {}):
        raise ValueError(f"{param!r} is not a parameter of the model (params: {sorted((base.get('params') or {}))}; "
                         "'horizon.window' sweeps the lag window L, 'horizon.T' the terminal time)")
    solver_kw = dict(solver_kw or {})
    if (base.get("horizon") or {}).get("kind") == "transition" and "past" not in solver_kw:
        solver_kw["past"] = Past.from_block(Model.from_dict(base).horizon.past)      # the past solved once for every point
    built: List[dict] = []
    prev = prev2 = None
    for v in values:
        d = copy.deepcopy(base)
        if param in HORIZON_LENGTHS:
            d.setdefault("horizon", {})[param.split(".", 1)[1]] = float(v)
        else:
            d.setdefault("params", {})[param] = float(v)
        m = Model.from_dict(d)
        S = engines.solver(m, numerics, **solver_kw)
        t0 = time.time()
        start_from = None
        if prev is not None and not S.same_grid(prev.compiled) and hasattr(S, "warm_maps_from"):
            start_from = S.warm_maps_from(prev)                    # another grid of the same model (a sweep over T)
        if prev is not None and S.same_grid(prev.compiled):
            w1 = warm_start(prev)
            if prev2 is not None and S.same_grid(prev2.compiled):
                w0 = warm_start(prev2); e1, e0 = built[-1]["value"], built[-2]["value"]
                if abs(e1 - e0) > 0:
                    start_from = {k: w1[k] + (w1[k] - w0[k]) * (float(v) - e1) / (e1 - e0) for k in w1}
            if start_from is None:
                start_from = w1
        kw = dict(solve_kw or {})
        if start_from is None:
            kw["start_policy"] = default_start(S, kw.get("start_policy"))
        res = S.solve(start_from=start_from, **kw)
        change = None
        if prev is not None and S.same_grid(prev.compiled):
            a1, a0 = warm_start(res), warm_start(prev)
            num = max(np.abs(a1[k] - a0[k]).max() for k in a1); den = max(max(np.abs(a0[k]).max() for k in a0), 1e-12)
            change = float(num / den)
        built.append(dict(param=param, value=float(v), result=res, seconds=time.time() - t0,
                          evaluations=int(res.evaluations), converged=bool(res.converged), change=change))
        if verbose:
            print(f"{param} = {v:g}: {'ok' if res.converged else 'NOT converged'} in {res.evaluations} evaluations, {time.time()-t0:.1f}s", flush=True)
        prev2, prev = prev, res
    # continuity: a point whose change from its predecessor is far above the sweep's typical change is a
    # candidate branch jump (the change per point, not per unit of parameter: a geometric sweep moves
    # the same amount per point).  It is computed BEFORE the points are built: a SweepPoint is frozen,
    # so a field cannot be patched in afterwards, which is what the mutable rows used to do.
    changes = [b["change"] for b in built if b["change"] is not None]
    med = float(np.median(changes)) if changes else 0.0
    rows = [SweepPoint(**b, jump=bool(b["change"] is not None and med > 0 and b["change"] > 5 * med))
            for b in built]
    if verbose:
        for r in rows:
            if r.jump:
                print(f"  {param} = {r.value:g}: change {r.change:.2g} against a typical {med:.2g}: possible branch jump", flush=True)
    return rows
