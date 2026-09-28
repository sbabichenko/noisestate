"""Panels for a finite horizon that one panel does not resolve: cut from both ends, graded by the model's time scales.

A finite horizon without lags is one panel in time and in age: every kernel is a polynomial of numerics.nodes degree
in each over [0, T].  That resolves a game whose time scales are of the order of T, and nothing much shorter: a
kernel decays in age at the closed loop's rates, the filter settles at the start and the value function turns at the
end, each within a few of the model's time constants tau, and a polynomial on [0, T] cannot follow a layer of width
tau << T.  The one-agent regulator of the tests (rates of about 3) has its cost 0.13% off at T = 10 on 12 nodes, 190%
off at T = 30, and a singular best response at T = 200; raising the nodes does not help (24 and 40 fail as 12 does).

A model without lags is autonomous, so its kernels move fast in three places only: at small ages (the decay of a
shock's effect), at the start (the filter's transient) and at the end (the terminal turn of the continuation), and
the triangle's time and age panels share one breakpoint sequence.  The panels are therefore cut at both ends and grow
geometrically towards the middle (graded_breakpoints): the first width w, then 2w, 4w, ... from 0 and from T.  A
panel away from the ends is wide where the kernels have long decayed (a piece far below the diagonal) or vary at
the slow rates only.  On the regulator at T = 30 and 12 nodes, 7 graded panels (w = 1) give the cost to 5e-10 and a
representation error of 5.6e-7, where 10 uniform panels give 3e-6 and 8.4e-5.

The width is found, not guessed: when a model eligible for it (numerics.breakpoints not given, no lags or delays, no
window of a past, no continuation; the default grid is then the one panel) fails the resolution check on that panel by
more than more nodes fix (its representation error above GRADE_ABOVE, 5e-4; with diagnostics off asked only of kernels
whose Chebyshev tail is above TAIL_ABOVE) or its best response is singular there, a cheap trial asks whether grading cuts
the error at all (_grading_helps: one best response per agent at a few nodes on the one panel and on seven graded panels,
tenfold or not); only then does solve() re-solve on graded panels with a first width w = 2^floor(log2(T/4)), halved
(each grid warm-started from the last) until the representation error passes settings.resolution_tol, the unknowns
would pass settings.auto_panels_max, or the error stops falling.  A model the one panel resolves, or whose failure
grading does not cut, is solved exactly as before, bit for bit: the default grid is always tried first.  res.numerics
then holds the breakpoints used (a solve with them reproduces the result) and res.message says how they were found;
when no grading passes the check within the budget, the finest one tried is returned with a warning naming the
breakpoints to try, and a model singular on every grid is refused with advice to rescale time.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np


GRADE_ABOVE = 5e-4      # a one-panel representation error above this is re-solved on graded panels (1e-3 at T = 5 on the
                        # regulator, where the cost is 2e-5 off; the modest failures more nodes fix, 1e-5 .. 1e-4, stay as they were)
TAIL_ABOVE = 1e-3       # solve(diagnostics=False) asks the representation error (one best response per agent) only of kernels whose
                        # Chebyshev tail (kernel_tail) is above this: 2.6e-3 at T = 5, 5.9e-4 at T = 3, 2.3e-5 on Chapter 1's game
TRIAL_NODES = 4         # ... at this many nodes (against the one panel at 4 and at 8)
TRIAL_SPLIT = 128.0     # the trial's graded grid: ends of T / 128 to T / 256 (thirteen panels)
GRADE_GAIN = 0.1        # ... when graded panels cut it tenfold in a trial at a few nodes (_grading_helps; else the
                        # one panel's result is returned as before: an error that grading leaves, a random walk the trader sees
                        # at 0.8 on Kyle-Back at T = 1, is not a time scale)


def graded_breakpoints(T: float, w: float, growth: float = 2.0) -> List[float]:
    """Breakpoints on [0, T] with panels of width w, growth w, growth^2 w, ... from 0 and, mirrored, from T, meeting in
    one middle panel (merged into its neighbours' when it would be shorter than half the widest of them)."""
    T, w = float(T), float(w)
    if not 0 < w < T / 2:
        return [0.0, T]
    left, step = [0.0], w
    while left[-1] + step < T / 2 - 1e-12 * T:
        left.append(left[-1] + step)
        step *= growth
    # the middle panel [left[-1], T - left[-1]]: too short next to its neighbours, drop the innermost cut on each side
    if len(left) > 2 and T - 2 * left[-1] < 0.5 * (left[-1] - left[-2]):
        left.pop()
    pts = sorted({round(b, 12) for b in left} | {round(T - b, 12) for b in left})
    return [float(b) for b in pts]


def eligible(S) -> bool:
    """Whether the spectral engine S may re-cut its panels: one panel from the default (numerics.breakpoints not
    given, no lags or delays), no window of a past (initial shocks are fine) and no continuation."""
    c = getattr(S, "c", None)
    g = getattr(c, "g", None)
    if g is None or not hasattr(g, "P"):
        return False
    m = S.model
    return (m.numerics.breakpoints is None and not m.all_lags() and g.L is None and getattr(c, "cont", None) is None
            and g.P == 1 and S.settings.auto_panels_max > 0)


def _singular(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "singular" in str(exc) and type(exc).__name__ == "ValueError"


def solve_graded(S0, num, first, first_error, solve_kw: dict, diagnostics: bool, build, verbose: bool = False):
    """The graded re-solve of solve() (see the module docstring).  S0: the engine of the one panel, first its result
    (None when its best response was singular, first_error the exception); num the resolved Numerics it ran with;
    build(model, numerics) -> (engine, numerics).  Returns the result to give back (first, unchanged, when it passes
    or when the grading is not needed), or raises first_error when no graded grid solves either."""
    from .numerics import Numerics
    model = S0.model
    T = float(model.horizon.extent)
    tol = S0.settings.resolution_tol
    if first is not None:
        if not first.converged:
            return first                               # a bound or a stall: not a question of resolution
        if not first.representation_error and kernel_tail(first) <= TAIL_ABOVE:
            return first                               # without the diagnostics: resolved kernels need no best response to say so
        if first.representation_error and max(first.representation_error.values()) <= max(tol, GRADE_ABOVE):
            return first                               # resolved, or short of it by what more nodes fix: as before
        if not _grading_helps(S0, first, num, T, build):
            return first                               # an error grading does not cut: not a matter of time scales, as before
        rep = max(first.representation_error.values()) if first.representation_error else S0.representation_probe(first)
        if rep <= max(tol, GRADE_ABOVE):
            return first
        history = [(1, [0.0, T], rep)]
    else:
        if not _regular_when_graded(S0, num, T, build):
            raise first_error                          # singular for another reason (graded panels do not change it)
        history = [(1, [0.0, T], None)]
    prev, best, best_rep, best_S = first, None, np.inf, None
    budget = S0.settings.auto_panels_max
    w = 2.0 ** np.floor(np.log2(T / 4.0))             # a power of two: the cuts are then short binary fractions
    total_evals = first.evaluations if first is not None else 0
    last_error = first_error
    while True:
        bp = graded_breakpoints(T, w)
        S, num2 = build(model, Numerics.of(num.to_dict()).merged(Numerics(breakpoints=bp)))
        tol = S.settings.resolution_tol
        budget = S.settings.auto_panels_max
        size = max(len(a.controls) * len(a.signals) for a in model.agents) * S.c.N
        if size > budget:
            break
        start = S.interpolate_maps(prev) if prev is not None else None
        if verbose:
            print(f"  -- graded panels from both ends, first width {w:g}: {len(bp) - 1} panels, {S.c.N} nodes", flush=True)
        try:
            res = S.solve(start_from=start, diagnostics=False, **solve_kw)
        except ValueError as exc:
            if not _singular(exc):
                raise
            last_error = exc
            history.append((len(bp) - 1, bp, None))
            prev = None
            w /= 2.0
            continue
        total_evals += res.evaluations
        if not res.converged:                          # a graded solve that stalls is kept only when the one panel had nothing
            if first is None and best is None:
                best, best_S = res, S
            history.append((len(bp) - 1, bp, None))
            break
        rep = S.representation_probe(res)
        history.append((len(bp) - 1, bp, rep))
        if rep < best_rep:
            best, best_rep, best_S = res, rep, S
        if rep <= tol:
            break
        falls = [h[2] for h in history if h[2] is not None]
        if len(falls) >= 3 and falls[-1] > 0.5 * falls[-2] and falls[-2] > 0.5 * falls[-3]:
            break                                      # not falling with the grading: not a matter of time scales
        prev = res
        w /= 2.0
    if best is None:                                   # every grid tried was singular (or past the budget at once)
        if first is not None:
            _warn_unresolved(model, first, history, budget, T)
            return first
        raise ValueError(f"{last_error}; panels graded from both ends by the time scales were singular as well, down to "
                         f"{history[-1][0]} panels (numerics.breakpoints {_fmt(history[-1][1])}) within settings.auto_panels_max "
                         f"({budget} unknowns): the model's fastest rate may be many orders of magnitude above 1 / T; rescale "
                         "time (or the noise) so that its rates are of order one") from last_error
    res = best
    if diagnostics:
        res.solve_kw.pop("diagnostics", None)
        best_S._diagnostics(res)
    res.evaluations = total_evals
    tried = "; ".join(f"{p} panel{'s' if p > 1 else ''}: " + ("singular" if r is None else f"{r:.1e}") for p, _, r in history)
    res.message += (f"; panels graded from both ends by the time scales (numerics.breakpoints {_fmt(res.numerics.breakpoints)}), "
                    f"the one panel of [0, {T:g}] under-resolved or singular (representation error by grid: {tried})")
    if not best_rep <= tol:
        _warn_unresolved(model, res, history, budget, T)
    return res


def _grading_helps(S0, first, num, T: float, build) -> bool:
    """The trial that keeps a failure grading cannot fix, or that more nodes fix better, from paying for a graded fixed
    point: one best response per agent to the one panel's maps, read on the one panel at TRIAL_NODES (4) nodes, on thirteen
    graded panels (ends of T / 128 to T / 256) at 4, and on the one panel at 8; grading is taken when it
    cuts the representation error by GRADE_GAIN and below what the doubled nodes give.  Kyle-Back with a random-walk value
    the trader sees stays near 0.4 graded at T = 1; Chapter 1's game at 4 nodes is resolved by 8 (a node count, not a time
    scale); the regulator at T = 10 falls from 0.1 to 1e-4 graded, to 2e-2 at 12 nodes."""
    from types import SimpleNamespace
    from .numerics import Numerics
    n = TRIAL_NODES
    reps = []
    for nodes, bp in ((n, None), (n, graded_breakpoints(T, 2.0 ** np.floor(np.log2(T / TRIAL_SPLIT)))), (2 * n, None)):
        S, _ = build(S0.model, Numerics.of(num.to_dict()).merged(Numerics(nodes=nodes) if bp is None else Numerics(nodes=nodes, breakpoints=bp)))
        try:
            reps.append(S.representation_probe(SimpleNamespace(maps=S.interpolate_maps(first))))
        except ValueError as exc:
            if not _singular(exc):
                raise
            reps.append(np.inf)
    return reps[1] < GRADE_GAIN * reps[0] and reps[1] < reps[2]


def _regular_when_graded(S0, num, T: float, build) -> bool:
    """Whether a one panel's singular best response is a matter of time scales: the best responses to the zero start on the
    trial's graded grid (_grading_helps') are regular.  A threshold that refuses every system (settings.foc_rcond 1) is
    singular there too, and the solve raises at once instead of grading down to the budget."""
    from .numerics import Numerics
    S, _ = build(S0.model, Numerics.of(num.to_dict()).merged(
        Numerics(nodes=TRIAL_NODES, breakpoints=graded_breakpoints(T, 2.0 ** np.floor(np.log2(T / TRIAL_SPLIT))))))
    try:
        zero = S.zero_maps()
        for a in S.model.agents:
            if S.c.rep[a.name] == a.name:
                S.best_response(a, zero, project=False)
    except ValueError as exc:
        if not _singular(exc):
            raise
        return False
    return True


def _fmt(bp) -> str:
    return "[" + ", ".join(f"{b:g}" for b in bp) + "]"


def _warn_unresolved(model, res, history, budget, T):
    last = [h for h in history if h[0] > 1]
    bp = last[-1][1] if last else graded_breakpoints(T, T / 4)
    finer = graded_breakpoints(T, (bp[1] - bp[0]) / 4 if len(bp) > 2 else T / 16)
    warnings.warn(f"{model.name}: the finite horizon T = {T:g} is long against the model's time scales and no graded panel grid "
                  f"within settings.auto_panels_max ({budget} unknowns) passes the resolution check; the finest tried was "
                  f"numerics.breakpoints {_fmt(bp)}.  Try finer panels at the ends (e.g. {_fmt(finer)}) with fewer nodes, "
                  "raise settings.auto_panels_max, or rescale time so that the fast rates are of order one", stacklevel=3)


def kernel_tail(res) -> float:
    """How far the equilibrium's kernels are from resolved on their pieces: on every piece of the triangle, the largest of the
    Chebyshev coefficients of the two highest degrees in time or in age (the interpolant's own error estimate), relative to
    the kernel's peak, the largest over the primaries and the world's columns.  A kernel resolved to eight digits has it
    near 1e-8 (1.4e-5 on the one-agent regulator at T = 1, 12 nodes); one that varies on a time scale far below the
    panel's width has it near its own size (9e-3 at T = 10, 0.15 at T = 30)."""
    from numpy.polynomial import chebyshev as C
    g = res.compiled.g
    nP = len(res.compiled.prim)
    Z = np.asarray(res.world)[:, :g.N if res.world.ndim == 1 else res.world.shape[1]]
    Z = Z.reshape(nP, g.N, -1)
    peak = np.abs(Z).max(axis=(1, 2))
    inv = {}
    worst = 0.0
    for pc in g.pieces:
        key = (pc.nt, pc.na)
        if key not in inv:
            x = lambda n: np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
            inv[key] = (np.linalg.inv(C.chebvander(x(pc.nt), pc.nt - 1)), np.linalg.inv(C.chebvander(x(pc.na), pc.na - 1)))
        Vt, Va = inv[key]
        V = Z[:, pc.offset:pc.offset + pc.n].reshape(nP, pc.nt, pc.na, -1)
        c = np.einsum("ij,pjkc,lk->pilc", Vt, V, Va)
        top = np.maximum(np.abs(c[:, -2:]).max(axis=(1, 2, 3)), np.abs(c[:, :, -2:]).max(axis=(1, 2, 3)))
        worst = max(worst, float(np.max(top / np.maximum(peak, 1e-300))))
    return worst
