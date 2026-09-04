"""Parameter sweeps with warm starts, and JSON-ready results.

    from noisestate.sweep import sweep, result_to_dict
    rows = sweep("examples/ch4_kyle_back.yaml", "eps", [0.2, 0.1, 0.05, 0.02])
    rows[0]["result"].summary(); rows[0]["seconds"]

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

from .spec import Model
from .results import TriangleResult
from .stationary import StationarySolver
from .finite import FiniteSolver
from .finite_spectral import SpectralFiniteSolver


def _load_dict(model: Union[str, dict, Model]) -> dict:
    if isinstance(model, str):
        with open(model) as fh:
            return yaml.safe_load(fh)
    if isinstance(model, Model):
        return model.to_dict()
    return copy.deepcopy(model)


def make_solver(model: Model, **kw):
    kind = model.horizon.kind
    if kind == "stationary":
        return StationarySolver(model, **kw)
    if kind == "finite_cells":
        return FiniteSolver(model, **kw)
    return SpectralFiniteSolver(model, **kw)


def warm_start(prev) -> Optional[dict]:
    """The object the next solve should start from, for a previous result of the same engine."""
    if prev is None:
        return None
    if isinstance(prev, TriangleResult):
        c = prev.compiled
        return {a.name: np.stack([prev.Z[c.block(u)] for u in a.controls]) for a in prev.model.agents}
    return prev.maps


def sweep(model: Union[str, dict, Model], param: str, values: Iterable[float], solver_kw: Optional[dict] = None,
          solve_kw: Optional[dict] = None, verbose: bool = False, predictor: str = "secant") -> List[dict]:
    """Solve the model at each value of `param` (a key of `params`), warm-starting each point from
    the previous one.  predictor="secant" starts from the linear extrapolation of the last two
    equilibria in the parameter (a tangent predictor; markedly more robust at hard points such as a
    small trading cost); "previous" starts from the last equilibrium.
    Returns [{"value", "result", "seconds", "evaluations", "converged", "change", "jump"}] in the given
    order; "change" is the relative change of the action kernels from the previous point and "jump"
    flags a change per unit parameter step more than five times the sweep's median (a possible
    branch jump)."""
    base = _load_dict(model)
    if param not in (base.get("params") or {}):
        raise ValueError(f"{param!r} is not a parameter of the model (params: {sorted((base.get('params') or {}))})")
    rows: List[dict] = []
    prev = prev2 = None
    for v in values:
        d = copy.deepcopy(base)
        d.setdefault("params", {})[param] = float(v)
        m = Model.from_dict(d)
        S = make_solver(m, **(solver_kw or {}))
        t0 = time.time()
        init = None
        if prev is not None and prev.compiled.N == S.c.N:
            w1 = warm_start(prev)
            if predictor == "secant" and prev2 is not None and prev2.compiled.N == S.c.N:
                w0 = warm_start(prev2); e1, e0 = rows[-1]["value"], rows[-2]["value"]
                if abs(e1 - e0) > 0:
                    init = {k: w1[k] + (w1[k] - w0[k]) * (float(v) - e1) / (e1 - e0) for k in w1}
            if init is None:
                init = w1
        res = S.solve(init=init, **(solve_kw or {}))
        change = None
        if prev is not None and prev.compiled.N == S.c.N:
            a1, a0 = warm_start(res), warm_start(prev)
            num = max(np.abs(a1[k] - a0[k]).max() for k in a1); den = max(max(np.abs(a0[k]).max() for k in a0), 1e-12)
            change = float(num / den)
        rows.append({"value": float(v), "result": res, "seconds": time.time() - t0, "evaluations": int(res.iterations),
                     "converged": bool(res.converged), "change": change})
        if verbose:
            print(f"{param} = {v:g}: {'ok' if res.converged else 'NOT converged'} in {res.iterations} evaluations, {time.time()-t0:.1f}s", flush=True)
        prev2, prev = prev, res
    # continuity: a point whose change (per unit of parameter step) is far above the sweep's typical
    # change is a candidate branch jump
    rates = [r["change"] / abs(r["value"] - rows[i - 1]["value"]) for i, r in enumerate(rows) if i and r["change"] is not None and r["value"] != rows[i - 1]["value"]]
    med = float(np.median(rates)) if rates else 0.0
    for i, r in enumerate(rows):
        rate = (r["change"] / abs(r["value"] - rows[i - 1]["value"])) if (i and r["change"] is not None and r["value"] != rows[i - 1]["value"]) else None
        r["jump"] = bool(rate is not None and med > 0 and rate > 5 * med)
        if verbose and r["jump"]:
            print(f"  {param} = {r['value']:g}: change per unit step {rate:.2g} against a typical {med:.2g}: possible branch jump", flush=True)
    return rows


def result_to_dict(res) -> dict:
    """JSON-serialisable view of a result (same as res.to_dict())."""
    return res.to_dict()
