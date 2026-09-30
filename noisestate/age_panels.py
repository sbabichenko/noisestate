"""Age panels for a stationary model, chosen automatically: placed from the model's lags and window before the first
solve, refined where the kernels' Chebyshev tails say the panels do not resolve them, within a budget.

The stationary engine holds every kernel in shock age on panels [b_0, b_1], ..., [b_{P-1}, L], numerics.nodes Chebyshev-
Lobatto nodes each (grid.py).  A model whose numerics give none of nodes, unit_range and breakpoints (the unit may be given:
it names the lags' common unit) gets its panels here; any of the three given is the expert's grid, solved as given.

1. A priori (prior).  Without lags or delays: one panel [0, L] at 16 nodes, the engine's default.  With lags: unit panels
   (width the unit u, by default the smallest lag) through max(the largest lag, UNIT_SPAN u), then panels growing by GROWTH
   (1.5), cut at L - lag for every lag and at L - j u for j up to EDGE_UNITS (the kernels are cut off at L, so a read at a
   lag, or at a sum of lags, breaks at L less it), at AUTO_NODES (12) nodes, every cut snapped to the unit lattice (the multiples of u from 0 and back from L, where the kernels break:
   a lag's echoes and the window's edge read at the lags; a delayed row's panels, closed under the delay, stay whole
   units).  The exact-shift path (stationary.Compiled.exact) makes such panels as good as the kernels' smoothness allows:
   a lag is an argument of the quadratures, not a resampling, so nothing needs the unit panels beyond the lags themselves
   (Chapter 5's market: unit panels through 2 tau, growth 1.5 on the lattice, 12 nodes, N = 132, kernels 2.6e-8 from a
   576-node uniform reference and the cost 3.2e-10 in one round, where the former default, 224 nodes resampled, was at
   8.4e-8 and the shipped 168 nodes at 1.9e-6 and 2.6e-8; growth 2 left the tail's panels at 2.7e-6).  A model whose lags the exact path does not carry (instant observations, level rows,
   monitoring, risk aversion: stationary.Compiled._exact_mode) keeps the engine's former defaults, unit panels through
   eight units then doubling, at 16 nodes: resampled shifts need the unit panels.

2. A posteriori (tails).  Every kernel of the result -- each agent's maps on each row, and the world's kernel of every
   primary on every shock -- is expanded in Chebyshev polynomials on every panel, and the panel's indicator is the largest
   of |c_{n-2}| + |c_{n-1}| over the kernels, relative to the kernel's peak over all ages and shocks.  It tracks the error
   against a uniform reference within a few times on Chapter 5's grids (it overstates it where the coefficients decay
   fast).  A result whose largest indicator is at most settings.auto_grid_tol (1e-7) is accepted.  Otherwise the panels
   above it are cut in two at the lattice point nearest their middle (a panel of one unit is not cut: the lags' panels,
   and a transition reads a past's panels on the unit lattice), or where that cannot help -- only unit panels above it, a model on the resampling path (bisected panels
   would not be aligned with the lags) or with a delayed row (its panels are closed under the delay, which splits a cut
   off the unit lattice below the unit) -- every panel gets two more nodes; the model is solved again from the last maps
   interpolated onto the new grid, at most MAX_ROUNDS times.

3. The budget is time_panels': no automatic grid passes settings.auto_panels_max unknowns (nU nR N of the largest agent)
   or an estimated peak memory of settings.auto_panels_memory MB (time_panels' estimate over the answering agents, a
   tie's representative once, plus the exact-shift path's dense tensors, about 12 N^3 doubles).  Where the next round would pass it, the best result so
   far is returned with res.panels["resolved"] False and res.panels["suggested"] the breakpoints of that next round, which
   res.sharpen() solves.

res.panels = {"route": "a priori" | "refined", "unit", "nodes", "breakpoints", "tails" (per panel, the accepted grid's),
"history" [(panels, N, largest tail, representation error or None (the checks run on the grid kept only), the first
agent's cost, evaluations)], "evaluations" (per round), "resolved", "suggested"}; res.numerics.breakpoints and nodes are the grid used (a solve with them reproduces the result).
suggest(model) gives the a priori grid without solving.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

UNIT_SPAN = 2           # unit panels through at least this many units (and every lag)
GROWTH = 1.5            # the panels beyond grow by this factor (2 left Chapter 5's tail at 2.7e-6 where 1.5 is at 2.3e-7)
AUTO_NODES = 12         # nodes per panel of the a priori grid of a model with lags (10 left Chapter 5's first panel at 7.5e-7)
PLAIN_NODES = 16        # ... of one without (one panel on [0, L], the engine's default)
MAX_ROUNDS = 3          # refinement rounds at most
EDGE_UNITS = 2          # cuts at L - j u for j up to this: the window's edge read at the lags and at their sums (Chapter 5's
                        # orders kinked at L - 2 tau inside the last panel, 2e-7 of their peak where its tail read 7e-8)
RESAMPLED_UNIT_RANGE = 8    # a model on the resampling path: unit panels through this many units (the former default)


def eligible(model, numerics) -> bool:
    """Whether a solve of `model` under `numerics` (a Numerics, a dict or None) chooses its own age panels: a stationary
    horizon on the stationary engine with none of nodes, unit_range, breakpoints given (by the model's numerics or these)."""
    from .numerics import Numerics
    if model.horizon.kind != "stationary":
        return False
    num = model.numerics.merged(Numerics.of(numerics))
    if (num.engine or "stationary") != "stationary":
        return False
    return num.nodes is None and num.unit_range is None and num.breakpoints is None


def _exact_capable(model) -> bool:
    """Whether the model's lags run on the exact-shift path (stationary.Compiled._exact_mode's feature test)."""
    if any(a.instant or a.risk_aversion or a.integrals for a in model.agents):
        return False
    if any(r.level for a in model.agents for r in a.signals):
        return False
    return not any(len(model.privy(a.name)) > 1 for a in model.agents)


def prior(model, unit: Optional[float] = None) -> Tuple[List[float], int]:
    """The a priori grid (breakpoints, nodes) of section 1 of the module docstring."""
    from .grid import AgeGrid
    L = float(model.horizon.extent)
    lags = [float(l) for l in model.all_lags()]
    if not lags:
        return [0.0, L], PLAIN_NODES
    u = float(unit) if unit else min(lags)
    if not _exact_capable(model):
        return [float(b) for b in AgeGrid.breakpoints_from_delays(L, lags, u, min(L, RESAMPLED_UNIT_RANGE * u))], PLAIN_NODES
    k = int(np.ceil(max(max(lags), UNIT_SPAN * u) / u - 1e-9))
    bp = [float(b) for b in AgeGrid.breakpoints_from_delays(L, lags, u, min(L, k * u), growth=GROWTH)]
    # every cut on the unit lattice (the multiples of u from 0 and back from L): the kernels break there -- a lag's echoes
    # k u, and the window's edge read at the lags, L - k u (on a window of 6 the kink at L - 2 tau inside a panel held a
    # Chapter 5 market at 3e-5 whatever its nodes) -- and a delayed row's panels, closed under the delay, stay whole units
    lat = _lattice(u, L)
    bp = {_snap(b, lat) for b in bp}
    bp |= {round(L - j * u, 12) for j in range(1, EDGE_UNITS + 1) if L - j * u > k * u + 1e-9}
    return sorted(bp), AUTO_NODES


def _lattice(u: float, L: float) -> np.ndarray:
    """The unit lattice of the window: the multiples of u from 0 and back from L, within [0, L]."""
    k = int(np.floor(L / u + 1e-9))
    return np.unique(np.round(np.concatenate([np.arange(k + 1) * u, L - np.arange(k + 1) * u, [L]]), 12))


def _snap(b: float, lat: np.ndarray) -> float:
    return float(lat[np.argmin(np.abs(lat - b))])


def suggest(model, numerics=None) -> Optional[dict]:
    """The a priori grid a solve of `model` would start from, {"breakpoints", "nodes"}, without solving; None when the model's
    numerics give the grid (eligible is False)."""
    if not eligible(model, numerics):
        return None
    from .numerics import Numerics
    bp, n = prior(model, model.numerics.merged(Numerics.of(numerics)).unit)
    return {"breakpoints": bp, "nodes": n}


_ANALYSIS: Dict[int, np.ndarray] = {}


def _analysis(n: int) -> np.ndarray:
    """(n, n): the Chebyshev coefficients of the interpolant through values at n Lobatto nodes (ascending)."""
    A = _ANALYSIS.get(n)
    if A is None:
        x = -np.cos(np.pi * np.arange(n) / (n - 1))
        A = _ANALYSIS[n] = np.linalg.inv(np.polynomial.chebyshev.chebvander(x, n - 1))
    return A


def tails(res) -> np.ndarray:
    """Per panel of the result's grid, the largest Chebyshev tail |c_{n-2}| + |c_{n-1}| of its kernels (every agent's map on
    every row, the world's kernel of every primary on every shock) relative to the kernel's peak over all ages and shocks."""
    c = res.compiled; g = c.grid; N, n, P = g.N, g.n, g.P
    A = _analysis(n)[-2:]                                            # the two highest coefficients
    out = np.zeros(P)

    def add(K):                                                      # K (N, m): kernels sharing one peak
        peak = float(np.abs(K).max())
        if not peak > 0.0:
            return
        C = np.abs(np.einsum("kj,pjm->pkm", A, K.reshape(P, n, -1))).sum(axis=1)     # (P, m)
        np.maximum(out, C.max(axis=1) / peak, out=out)
    W = np.asarray(res.world)
    for p in range(len(c.prim)):
        add(W[p * N:(p + 1) * N])
    for a in res.model.agents:
        if c.rep[a.name] != a.name:
            continue
        g_ = np.asarray(res.maps[a.name])                            # (nU, nR, N)
        add(g_.reshape(-1, N).T)
    return out


def _budget_ok(model, bp, nodes, settings, exact: bool = False) -> bool:
    """Whether the grid fits time_panels' budget: the unknowns nU nR N of the largest agent, and the estimated peak memory,
    time_panels' per answering agent (a tie answers once, by its representative) plus, on the exact-shift path, the
    shifted operators' dense tensors (about a dozen of N^3 doubles: 560 MB at N = 180)."""
    from .time_panels import MEMORY_BASE, MEMORY_PER
    N = (len(bp) - 1) * nodes
    tied = {n for g in model.ties for n in g[1:]}
    sizes = [len(a.controls) * len(a.signals) * N for a in model.agents if a.name not in tied]
    if max(sizes, default=0) > settings.auto_panels_max:
        return False
    tensors = 12 * 8e-6 * N ** 3 if exact else 0.0
    return MEMORY_BASE + MEMORY_PER * sum(s * s for s in sizes) + tensors <= settings.auto_panels_memory


def _next_grid(bp, nodes, tl, tol, unit, exact, L) -> Tuple[List[float], int]:
    """The next round's grid: the panels whose tail is above tol cut in two on the unit lattice point nearest their middle
    (a panel of one unit is not cut), or on the resampling path, or where no panel above tol can be cut, two more nodes
    everywhere."""
    if not exact:
        return list(bp), nodes + 2
    lat = _lattice(unit, L) if unit else None
    out = [bp[0]]
    for p, (lo, hi) in enumerate(zip(bp[:-1], bp[1:])):
        if tl[p] > tol:
            mid = 0.5 * (lo + hi)
            if lat is not None:
                inside = lat[(lat > lo + 1e-9) & (lat < hi - 1e-9)]
                mid = float(inside[np.argmin(np.abs(inside - mid))]) if inside.size else None
            if mid is not None:
                out.append(round(mid, 12))
        out.append(hi)
    if len(out) == len(bp):                                          # nothing left to cut: more nodes
        return list(bp), nodes + 2
    return out, nodes


def _history_row(res, tl) -> tuple:
    rep = max(res.representation_error.values()) if res.representation_error else None
    first = res.model.agents[0].name
    return (res.compiled.grid.P, int(res.compiled.N), float(tl.max()), None if rep is None else float(rep), float(res.costs.get(first, np.nan)),
            int(res.evaluations))


def solve(model, numerics, start_from, start_policy, run: dict, diagnostics: bool, build, verbose: bool = False):
    """The solve of an eligible model (the module docstring): the a priori grid, then refinement rounds within the budget."""
    from .numerics import Numerics
    from .engines import default_start
    given = Numerics.of(numerics)
    merged = model.numerics.merged(given)
    # bisecting keeps the lags' panels exact on the exact-shift path; a delayed row's panels are closed under the delay
    # (a cut off the unit lattice would split the panels below the unit): those grids get more nodes instead
    exact = _exact_capable(model) and not any(r.delay for a in model.agents for r in a.signals)
    unit = merged.unit if merged.unit else (min(model.all_lags()) if model.all_lags() else None)
    bp, nodes = prior(model, merged.unit)
    info = {"route": "a priori", "unit": unit, "history": [], "resolved": None, "suggested": None}
    res = None
    prev = None
    for rnd in range(MAX_ROUNDS + 1):
        num = given.merged(Numerics(breakpoints=bp, nodes=nodes))
        S, num_r = build(model, num)
        tol = S.settings.auto_grid_tol
        kw = dict(run)
        if prev is None:
            kw.update(start_from=start_from, start_policy=default_start(S, start_policy) if start_from is None else None)
            if kw["start_policy"] is None:
                del kw["start_policy"]
        else:
            kw["start_from"] = S.interpolate_maps(prev)
        if verbose:
            print(f"  -- age panels, round {rnd}: {len(bp) - 1} panels x {nodes} nodes (N = {(len(bp) - 1) * nodes})", flush=True)
        res = S.solve(diagnostics=False, **kw)                       # the checks once, on the grid kept (below)
        tl = tails(res)
        info["history"].append(_history_row(res, tl))
        ok = bool(tl.max() <= tol)
        if ok or not res.converged:
            break
        nbp, nn = _next_grid(bp, nodes, tl, tol, unit, exact, float(model.horizon.extent))
        if not _budget_ok(model, nbp, nn, S.settings, bool(getattr(S.c, 'exact', False))):
            info["suggested"] = nbp if nn == nodes else None
            info["suggested_nodes"] = nn
            break
        if rnd == MAX_ROUNDS:
            info["suggested"] = nbp
            info["suggested_nodes"] = nn
            break
        prev, bp, nodes = res, nbp, nn
        info["route"] = "refined"
    if diagnostics:                                                  # the checks of the grid kept: its engine's own pass
        import time
        t0 = time.time()
        res.solve_kw.pop("diagnostics", None)
        S._diagnostics(res)
        res.seconds += time.time() - t0
        info["history"][-1] = _history_row(res, tl)
    info.update(nodes=nodes, breakpoints=[float(b) for b in bp], tails=[float(x) for x in tl], resolved=bool(tl.max() <= tol),
                evaluations=[h[5] for h in info["history"]])
    if not info["resolved"] and res.converged:
        warnings.warn(f"{model.name}: the automatic age panels stopped at N = {(len(bp) - 1) * nodes} with a Chebyshev tail of "
                      f"{tl.max():.1e} (above settings.auto_grid_tol {tol:g}); res.sharpen() re-solves on the suggested grid "
                      "(res.panels['suggested'])", stacklevel=3)
    res.panels = info
    return res
