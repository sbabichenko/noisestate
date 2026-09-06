"""Parameter sweeps with warm starts, and JSON-ready results.

    from noisestate import sweep
    rows = sweep("examples/ch4_kyle_back.yaml", "eps", [0.2, 0.1, 0.05, 0.02])
    rows[0]["result"].summary(); rows[0]["result"].to_dict(); rows[0]["seconds"]

Each point starts from the previous point's equilibrium (raw maps for the stationary engine,
action kernels for the spectral finite engine), which is what makes a slider in an interactive
front end cheap: after the first point, a step costs a few best responses.
"""
from __future__ import annotations

import copy
import time
from typing import Iterable, List, Optional, Union

import numpy as np
import yaml

from .spec import Model, ModelBuilder
from .numerics import Numerics
from .past import Past
from .results import TriangleResult
from .engines import build, default_start


def _load_dict(model: Union[str, dict, Model]) -> dict:
    if isinstance(model, str):
        with open(model) as fh:
            return yaml.safe_load(fh)
    if isinstance(model, ModelBuilder):
        return model.to_dict()
    if isinstance(model, Model):
        return model.to_dict()
    if not isinstance(model, dict):
        raise TypeError(f"expected a Model, a ModelBuilder, a dict or a path, not {type(model).__name__}")
    return copy.deepcopy(model)


def make_solver(model: Model, numerics=None, **kw):
    """The engine the model's numerics select (noisestate.engines.build), constructed with `kw` (verbose,
    naive_observers, past, continuation; `settings` is accepted as an alias of numerics.settings)."""
    if "settings" in kw:
        numerics = Numerics.of(numerics).merged(Numerics(settings=kw.pop("settings")))
    return build(model, numerics, **kw)[0]


def warm_start(prev) -> Optional[dict]:
    """The object the next solve should start from, for a previous result of the same engine."""
    if prev is None:
        return None
    if isinstance(prev, TriangleResult):
        c = prev.compiled
        return {a.name: np.stack([prev.world[c.block(u)] for u in a.controls]) for a in prev.model.agents}
    return prev.maps


def sweep(model: Union[str, dict, Model, ModelBuilder], param: str, values: Iterable[float], numerics=None,
          solver_kw: Optional[dict] = None, solve_kw: Optional[dict] = None, verbose: bool = False) -> List[dict]:
    """Solve the model at each value of `param` (a key of `params`, or "horizon.window": the window L of a
    stationary model, the horizon T of a finite one or of a transition, which then warm-starts each point from
    the previous maps read on the new grid, the stationary maps beyond it), warm-starting each point from
    the linear extrapolation of the last two equilibria in the parameter (a secant predictor;
    markedly more robust at hard points such as a small trading cost).  numerics (a Numerics or a dict of
    its fields) is laid over the model's own at every point; solver_kw goes to each point's engine (verbose,
    naive_observers, past, continuation), solve_kw to each point's solve() (tol, max_evaluations, deadline,
    progress, diagnostics, start); a point stopped at a bound is a row with converged False, and the sweep goes on.
    Returns [{"param", "value", "result", "seconds", "evaluations", "converged", "change", "jump"}] in the
    given order; "change" is the relative change of the strategy from the previous point on the same grid
    (the raw maps; the action kernels on the finite spectral engine)
    and "jump" flags a change more than five times the sweep's median (a possible branch jump).  A transition
    model's past is solved once and shared by every point (solver_kw={"past": ...} gives it); its
    continuation, the new model's stationary equilibrium, is solved at each point."""
    base = _load_dict(model)
    if param != "horizon.window" and param not in (base.get("params") or {}):
        raise ValueError(f"{param!r} is not a parameter of the model (params: {sorted((base.get('params') or {}))}; "
                         "'horizon.window' sweeps the window or horizon)")
    solver_kw = dict(solver_kw or {})
    if (base.get("horizon") or {}).get("kind") == "transition" and "past" not in solver_kw:
        solver_kw["past"] = Past.from_block(Model.from_dict(base).horizon.past)      # the past solved once for every point
    rows: List[dict] = []
    prev = prev2 = None
    for v in values:
        d = copy.deepcopy(base)
        if param == "horizon.window":
            d.setdefault("horizon", {})["window"] = float(v)
        else:
            d.setdefault("params", {})[param] = float(v)
        m = Model.from_dict(d)
        S = make_solver(m, numerics, **solver_kw)
        t0 = time.time()
        init = None
        if prev is not None and not S.same_grid(prev.compiled) and hasattr(S, "warm_maps_from"):
            init = S.warm_maps_from(prev)                    # another grid of the same model (a sweep over T)
        if prev is not None and S.same_grid(prev.compiled):
            w1 = warm_start(prev)
            if prev2 is not None and S.same_grid(prev2.compiled):
                w0 = warm_start(prev2); e1, e0 = rows[-1]["value"], rows[-2]["value"]
                if abs(e1 - e0) > 0:
                    init = {k: w1[k] + (w1[k] - w0[k]) * (float(v) - e1) / (e1 - e0) for k in w1}
            if init is None:
                init = w1
        kw = dict(solve_kw or {})
        if init is None:
            kw["start"] = default_start(S, kw.get("start"))
        res = S.solve(init=init, **kw)
        change = None
        if prev is not None and S.same_grid(prev.compiled):
            a1, a0 = warm_start(res), warm_start(prev)
            num = max(np.abs(a1[k] - a0[k]).max() for k in a1); den = max(max(np.abs(a0[k]).max() for k in a0), 1e-12)
            change = float(num / den)
        rows.append({"param": param, "value": float(v), "result": res, "seconds": time.time() - t0, "evaluations": int(res.evaluations),
                     "converged": bool(res.converged), "change": change})
        if verbose:
            print(f"{param} = {v:g}: {'ok' if res.converged else 'NOT converged'} in {res.evaluations} evaluations, {time.time()-t0:.1f}s", flush=True)
        prev2, prev = prev, res
    # continuity: a point whose change from its predecessor is far above the sweep's typical change is a
    # candidate branch jump (the change per point, not per unit of parameter: a geometric sweep moves
    # the same amount per point)
    changes = [r["change"] for r in rows if r["change"] is not None]
    med = float(np.median(changes)) if changes else 0.0
    for r in rows:
        r["jump"] = bool(r["change"] is not None and med > 0 and r["change"] > 5 * med)
        if verbose and r["jump"]:
            print(f"  {param} = {r['value']:g}: change {r['change']:.2g} against a typical {med:.2g}: possible branch jump", flush=True)
    return rows
