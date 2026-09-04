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
from .stationary import StationarySolver, Result
from .finite import FiniteSolver
from .finite_spectral import SpectralFiniteSolver, SpectralResult


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
    if isinstance(prev, SpectralResult):
        c = prev.compiled
        return {a.name: np.stack([prev.Z[c.block(u)] for u in a.controls]) for a in prev.model.agents}
    return prev.maps


def sweep(model: Union[str, dict, Model], param: str, values: Iterable[float], solver_kw: Optional[dict] = None,
          solve_kw: Optional[dict] = None, verbose: bool = False, predictor: str = "secant") -> List[dict]:
    """Solve the model at each value of `param` (a key of `params`), warm-starting each point from
    the previous one.  predictor="secant" starts from the linear extrapolation of the last two
    equilibria in the parameter (a tangent predictor; markedly more robust at hard points such as a
    small trading cost); "previous" starts from the last equilibrium.
    Returns [{"value", "result", "seconds", "evaluations", "converged"}] in the given order."""
    base = _load_dict(model)
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
        rows.append({"value": float(v), "result": res, "seconds": time.time() - t0, "evaluations": int(res.iterations),
                     "converged": bool(res.converged)})
        if verbose:
            print(f"{param} = {v:g}: {'ok' if res.converged else 'NOT converged'} in {res.iterations} evaluations, {time.time()-t0:.1f}s", flush=True)
        prev2, prev = prev, res
    return rows


def result_to_dict(res) -> dict:
    """JSON-serialisable view of a result: grid, kernels per quantity and channel, raw maps, costs,
    and (stationary) the first-order-condition decomposition."""
    c = res.compiled
    if isinstance(res, SpectralResult):
        grid = {"kind": "finite_triangle", "breakpoints": [float(b) for b in c.g.bp], "nodes_per_side": c.g.nt,
                "t": c.g.t.tolist(), "age": c.g.a.tolist(), "s": c.g.s.tolist()}
    elif isinstance(res, Result):
        grid = {"kind": "stationary", "breakpoints": [float(b) for b in c.grid.breakpoints], "nodes_per_panel": c.grid.n,
                "ages": c.grid.nodes.tolist()}
    else:
        grid = {"kind": "finite_cells", "cells": int(c.N), "h": float(c.h)}
    out = {"model": res.model.name, "converged": bool(res.converged), "residual": float(res.residual),
           "evaluations": int(res.iterations), "seconds": float(res.seconds), "grid": grid,
           "discount": float(c.rho), "channels": list(c.channels), "kernels": {}, "maps": {}, "foc": {},
           "costs": {k: float(v) for k, v in res.costs.items()}}
    if grid["kind"] == "finite_cells":
        for name in c.prim:
            out["kernels"][name] = {ch: res.kernel(name, ch).tolist() for ch in c.channels}
        return out
    for name in c.prim:
        K = res.kernel(name)
        out["kernels"][name] = {ch: K[:, k].tolist() for k, ch in enumerate(c.channels)}
    for a in res.model.agents:
        g = res.maps[a.name]
        out["maps"][a.name] = {u: {r.name: g[ui, ri].tolist() for ri, r in enumerate(a.signals)} for ui, u in enumerate(a.controls)}
        if getattr(res, "foc", None) and a.name in res.foc:
            out["foc"][a.name] = {u: {part: {ch: arr[:, k].tolist() for k, ch in enumerate(c.channels)}
                                      for part, arr in dec.items()} for u, dec in res.foc[a.name].items()}
    return out
