"""transition(old, new, T) / transition(old, new, settle=tol): the equilibrium path of a regime change, from the
old stationary regime to the new, on an explicit horizon T or on the horizon a march in T finds for a settle tolerance.

    from noisestate import transition
    res = transition("examples/ch3_two_player.yaml", new_model, T=6.0, numerics={"nodes": 12})
    res = transition("examples/ch3_two_player.yaml", new_model, settle=1e-4, numerics={"nodes": 12})
    res.past, res.stationary, res.settled, res.extra["window"], res.march, res.summary()

The old regime is a converged StationaryResult or a stationary model (a Model, ModelBuilder, dict or
path, solved on the fly); the new regime is a model of any kind, whose horizon is rewritten to a
transition on [0, T] continued by its own stationary equilibrium (solved on the past's window at
numerics.continuation_nodes, default the transition's nodes).  The solve starts from the new stationary maps
(start="stationary"), the one default of every transition with a continuation (this helper, solve(model,
past=, continuation=) and a file of kind transition alike): it is where a settled transition ends and, on a
moderate change, a few best responses from the answer.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from . import engines
from .spec import Model, ModelBuilder
from .numerics import Numerics
from .past import Past
from .expr import SweepPoint


def _model_of(obj) -> Model:
    from . import load
    if isinstance(obj, str):
        return load(obj)
    if isinstance(obj, dict):
        return Model.from_dict(obj)
    if isinstance(obj, ModelBuilder):
        return obj.build()
    if isinstance(obj, Model):
        return obj
    raise TypeError(f"the new model must be a Model, a ModelBuilder, a dict or a path, not {type(obj).__name__}")


def _past_block(old, past: Past) -> dict:
    """The horizon.past block recording where the past came from: the old model file, its dict (a result's
    model inlined), or the initial shocks."""
    if isinstance(old, str):
        return {"model": old}
    if isinstance(old, (list, tuple)):
        return {"initial": [sh if isinstance(sh, dict) else sh.to_dict() for sh in old]}
    if isinstance(old, (Model, ModelBuilder)):
        return {"model": old.to_dict()}
    if isinstance(old, dict):
        return {"model": dict(old)}
    src = getattr(old, "source", old)                       # a Past made from a result, or the result itself
    if hasattr(src, "model") and hasattr(src, "world"):
        return {"model": src.model.to_dict()}
    return {"initial": [sh.to_dict() for sh in past.initial]}


def _transition_model(old, new, T: float, num: Numerics, continuation):
    """(model, past, continuation): the new model rewritten to horizon kind transition on [0, T] with the past's
    block and the numerics laid over (the shared construction of transition() and transition_gap())."""
    past = Past.of(old)
    m = _model_of(new)
    d = m.to_dict()
    hz = d.setdefault("horizon", {}); nm = d.setdefault("numerics", {})
    for k in ("breakpoints", "unit_range", "engine"):
        nm.pop(k, None)
    nm["nodes"] = 12 if num.nodes is None else num.nodes
    if num.continuation_nodes is not None:
        nm["continuation_nodes"] = num.continuation_nodes
    for k in ("unit", "tol", "damping", "max_newton", "variable"):
        if getattr(num, k) is not None:
            nm[k] = getattr(num, k)
    if num.settings.changed():
        nm["settings"] = num.settings.changed()
    block = _past_block(old, past)
    if not past.window > 0 and continuation == "stationary":
        continuation = "end"
    hz.update(kind="transition", window=float(T), past=block,
              continuation=continuation if isinstance(continuation, str) else "stationary")
    for k in ("stationary", "settle"):
        hz.pop(k, None)
    return Model.from_dict(d), past, continuation


def gap_passes(S, maps: Dict[str, np.ndarray], ranges: Dict[str, tuple]) -> Dict[str, Dict[str, float]]:
    """The best-response pass of the settle monitor: given the maps (every agent's rule on the strip, the
    buffer's frozen at the stationary ones), each agent's best response is computed once and its distance from
    the stationary rule is measured on the identified nodes with time in each range [lo, hi] of `ranges`
    (label -> (lo, hi); the buffer excluded): the largest difference over controls, rows and nodes, relative
    to the stationary rule's own peak, as {label: {agent: gap}}.  At a converged transition the best response
    is the solved rule to tolerance, so this is the maps' own distance from the stationary ones on the range;
    from the stationary maps it is the one-shot deviation."""
    c = S.c; g = c.g
    eps = 1e-9 * max(1.0, c.Tg)
    out = {label: {} for label in ranges}
    for a in S.model.agents:
        gm, _ = S.best_response(a, maps)
        fr = c.frozen[a.name]
        keep = S._identified(a).reshape(len(a.signals), -1)[:, :c.N] & ~c.buffer
        dev = np.abs(gm[:, :, :c.N] - fr) * keep
        for label, (lo, hi) in ranges.items():
            sel = (g.t >= lo - eps) & (g.t <= hi + eps)
            out[label][a.name] = float((dev * sel).max() / max(1e-300, np.abs(fr).max()))
    return out


def gap_pass(S, maps: Dict[str, np.ndarray], lo: float, hi: float) -> Dict[str, float]:
    """gap_passes on the one range [lo, hi]: {agent: gap}."""
    return gap_passes(S, maps, {"range": (lo, hi)})["range"]


FLOOR_FACTOR = 2.0        # a gap within this factor of the grid's floor is the floor: the march stops there


def _unit_of(model: Model, L: float) -> float:
    """The march's unit: numerics.unit when given, else the smallest lag or delay, else the past's window L (a
    model without lags has no unit of its own; the strip is then cut at the multiples of T below L)."""
    if model.horizon.unit:
        return float(model.horizon.unit)
    lags = [float(d) for d in model.all_lags() if d and d > 0]
    return min(lags) if lags else float(L)


def transition_gap(old, new, numerics=None, continuation="stationary") -> Dict[str, float]:
    """The T = 0 pass of the settle march: with no transition solved at all, every agent keeps the new model's
    stationary rules from date zero with the inheritance (the old regime's shocks in the state and in what
    each agent has observed) attached, and one best response per agent is computed from that; returned is,
    per agent, the relative distance of the best-response rule from the stationary rule (max over the
    identified nodes, relative to the stationary rule's peak).  Under a tolerance the stationary equilibrium is
    the transition and nothing needs solving; otherwise the size says how far the inheritance moves the
    rules.  The strip needs at least one time panel, so the pass runs on the smallest strip the engine builds,
    [0, u] with u the unit (numerics.unit, else the smallest lag, else the past's window L: _unit_of), the
    strategies frozen at the stationary rules on [0, u] and the buffer [u, u + L] alike (a horizon below the
    window keeps the old shocks alive on the buffer, where the strip carries them as the band); the
    transition() march starts from this point.  Caveat: with the old model equal to the new one the gap is not
    zero but the grid's one-shot floor (Chapter 3 at 12 nodes: 3.8e-6 and 1.3e-5, the same at T = 1, 3 and 6;
    the two-firm market 2e-4 at 5 nodes), so a tolerance below that floor is never met (the march measures
    that floor first: settle_floor).  `old`, `new`, `numerics` and `continuation` are transition()'s (a
    stationary continuation is required: the pass needs the rules the strategies are held at)."""
    num = Numerics.of(numerics)
    past = Past.of(old)
    if not past.window > 0 or continuation == "end":
        raise ValueError("transition_gap needs a past with a window and a stationary continuation: the pass measures the "
                         "best response from the new model's stationary rules with the old regime's shocks attached")
    model, past, continuation = _transition_model(old, new, past.window, num, continuation)
    u = _unit_of(model, past.window)
    if abs(u - past.window) > 1e-12:
        model = model.with_horizon(window=float(u))
    S = engines.build(model, None, past=past, continuation=continuation)[0]
    return gap_pass(S, S.stationary_start(), 0.0, S.c.T)


def settle_floor(model: Model, cont, numerics=None) -> Dict[str, float]:
    """The grid's floor of the settle monitor at these numerics, per agent: the one-shot deviation of the
    continuation's own stationary rules on the strip of `model` (a transition model at some T) with the
    continuation as its own past, the same-model gap (transition_gap of the new model as its own regime).  A
    settle tolerance below it is never met; a gap within FLOOR_FACTOR of it is the floor.  One strip build and
    one best response per agent."""
    S = engines.build(model, numerics, past=Past.of(cont), continuation=cont)[0]
    return gap_pass(S, S.stationary_start(), 0.0, S.c.T)


def _stationary_result(S, gap: Dict[str, float], kw: dict, diagnostics: bool):
    """The transition result when nothing needs solving (the T = 0 pass under settle, or the tolerance below the
    floor): the continuation's stationary maps on the strip built (stationary_start), the world their closed
    loop, no evaluation; the result type is the engine's (a TransitionResult), march_window 0 (res.extra["window"])."""
    maps = S.stationary_start()
    Z = S.c.closed_loop(maps)
    res = S.RESULT(model=S.model, compiled=S.c, maps=maps, world=Z, converged=True, residual=float(max(gap.values())),
                   evaluations=0, seconds=0.0, message="the stationary equilibrium: the T = 0 pass is under settle (no solve)",
                   solver_class=type(S), solver_kw=S.solver_kw, settings=S.settings,
                   solve_kw={k: kw[k] for k in ("tol", "damping", "max_newton", "variable") if k in kw})
    if not diagnostics:
        res.solve_kw["diagnostics"] = False
    S._finish(res)
    res.march_window = 0.0
    return res


def march(make_model: Callable[[float], Model], past: Past, continuation, settle: float, numerics=None, step: Optional[float] = None,
          max_window: Optional[int] = None, verbose: bool = False, **solve_kw):
    """The march in T (docs/design/transition_settle_march.md): make_model(T) is the transition model on [0, T].  The
    first point is the T = 0 pass (transition_gap: the best response from the stationary rules on the first strip
    built, nothing solved); under `settle` the stationary equilibrium is the transition and the result is the
    continuation's maps on that strip with no solve (_stationary_result: res.march_stop "settled at T = 0",
    window 0).  Then the floor (settle_floor: the same-model gap on the first strip, one best response per agent;
    res.march_floor, res.extra["settle_floor"]): a tolerance below it is never met, so the march stops at once with
    march_stop "floor" and the stationary result (the `settle floor` row flags SETTLE BELOW THE GRID'S FLOOR, advice
    raise numerics.nodes).  Otherwise the first solve is at T = L and T grows by one window per step (each T snapped
    up to a multiple of the unit; `step=` sets the step, unit steps below the first window are its business: a first
    window cannot certify anything, its monitor window [0, T] holding the initial transient), the continuation
    solved once.  Each step is local: the solve is warm-started from the previous maps read on the new grid with
    the stationary rules on the new stretch (warm_maps_from), and the strategies before T_prev - L are frozen at
    those values (SpectralFiniteSolver.freeze_before: the end effect leaks back at the closed-loop rate only, so
    they are converged), the unknowns being the new stretch plus the last window of the old horizon; the fixed
    point runs on that reduced vector, and the diagnostics (res.foc, the second-order check, the representation
    error) run once, on the final strip, with nothing frozen.  After each solve the monitor: gap_pass on the
    converged maps over the last window [T - L, T], the range of the `settled` diagnostic (on Chapter 3 the explicit
    T = 9 solve settles at 2.7e-6, so the march stops at the smallest T whose explicit solve settles under the
    tolerance).  Stops when every agent's gap is under `settle` ("settled"), at the floor (a gap within FLOOR_FACTOR
    of it: "floor", the `settle floor` row), or when T would pass max_window * L (default 8 windows: "max_window");
    in the last two cases res.settled keeps the last solve's diagnostic and its flag when above settled_tol.
    After the last local step one polishing pass runs with every unknown free, warm-started from that step's fixed
    point (one to three evaluations), so the returned maps are the whole strip's fixed point to the solve tolerance
    (the local step alone leaves the previous handover frozen into the early part: 2e-8 on Chapter 3 at 12 nodes);
    its count is the last row's "polish".  res.march is the list of rows {"T", "gap", "evaluations", "seconds",
    "monitor", "unknowns"} (T = 0 first, evaluations 0; unknowns the map unknowns solved for, summed over agents;
    seconds includes the polish), res.extra["window"] the T found.
    solve_kw goes to every solve (tol, max_evaluations, deadline, progress, diagnostics)."""
    if not settle > 0:
        raise ValueError("settle must be a positive tolerance (the relative distance of the best-response rules from the stationary ones)")
    if not past.window > 0 or continuation == "end":
        raise ValueError("transition(settle=) needs a past with a window and a stationary continuation: the march measures the "
                         "best-response rules against the new model's stationary rules")
    L = float(past.window)
    if step is not None and not step > 0:
        raise ValueError("step must be positive (a multiple of the unit; default one window)")
    max_window = 8 if max_window is None else int(max_window)
    if max_window < 1:
        raise ValueError("max_window must be at least 1 (windows of the past's L)")
    Tmax = max_window * L; eps = 1e-9 * max(1.0, Tmax)
    u = _unit_of(make_model(L), L)

    def snap(x: float) -> float:
        return float(np.ceil(x / u - 1e-9) * u)

    def after(T: float) -> float:
        """The next horizon: T + step when given, else T + L, snapped up to a multiple of the unit."""
        return snap(T + (float(step) if step is not None else L))

    def unknowns(S) -> int:
        return int(sum(S._identified(a).sum() * len(a.controls) for a in S.model.agents))
    rows = []; prev = None; T = snap(u) if step is not None else snap(L); stop = None; floor = None
    while True:
        S, num = engines.build(make_model(T), numerics, verbose=verbose, past=past, continuation=continuation)
        continuation = S.c.cont                                     # solved once, shared by every step
        kw = {**num.solve_kw(), **solve_kw}
        diagnostics = kw.pop("diagnostics", True)                   # run once, on the final strip
        t0 = time.time()
        if prev is None:
            gap0 = gap_pass(S, S.stationary_start(), 0.0, T)
            rows.append(SweepPoint({"T": 0.0, "gap": gap0, "evaluations": 0, "seconds": time.time() - t0, "monitor": f"[0, {T:g}] from the stationary rules",
                                    "unknowns": 0}))
            if verbose:
                print(f"settle march: T = 0 (the stationary rules): gap {max(gap0.values()):.2e}, {time.time() - t0:.1f}s", flush=True)
            if max(gap0.values()) <= settle:
                stop = "settled at T = 0"
                res = _stationary_result(S, gap0, kw, diagnostics); break
            t1 = time.time()
            floor = settle_floor(make_model(T), continuation, numerics)
            if verbose:
                print(f"settle march: the grid's floor {max(floor.values()):.2e} (the same-model gap on [0, {T:g}]), {time.time() - t1:.1f}s", flush=True)
            if settle < max(floor.values()):
                stop = "floor"
                res = _stationary_result(S, gap0, kw, diagnostics); break
            t0 = time.time()
            res = S.solve(start="stationary", diagnostics=False, **kw)
        else:
            init = S.warm_maps_from(prev)
            act = S.warm_actions_from(prev, init)
            t_lo = max(0.0, prev.compiled.T - L)
            bp = S.c.g.bp[S.c.g.bp <= t_lo + eps]
            S.freeze_before(float(bp[-1]) if len(bp) else 0.0, init, actions=act)
            res = S.solve(init=act, diagnostics=False, **kw)
        gap = gap_pass(S, res.maps, T - L, T)
        rows.append(SweepPoint({"T": float(T), "gap": gap, "evaluations": int(res.evaluations), "seconds": time.time() - t0,
                                "monitor": "[T - L, T]", "unknowns": unknowns(S)}))
        if verbose:
            print(f"settle march: T = {T:g}: gap {max(gap.values()):.2e} on [T - L, T], {res.evaluations} evaluations, "
                  f"{rows[-1]['unknowns']} unknowns, {time.time() - t0:.1f}s", flush=True)
        if max(gap.values()) <= settle:
            stop = "settled"; break
        if max(gap.values()) <= FLOOR_FACTOR * max(floor.values()):
            stop = "floor"; break
        if after(T) > Tmax + eps:
            stop = "max_window"; break
        prev = res; T = after(T)
    if res.evaluations and S.c.P_lo > 0:
        # the polish: the local step left the previous step's end effect frozen into the early part (2e-8 on Chapter 3
        # at 12 nodes); one pass with every unknown free, warm-started from the local step's fixed point, makes the
        # maps the whole strip's fixed point to the solve tolerance again (one to three evaluations)
        t0 = time.time()
        S.freeze_before(0.0)
        init = res.actions if res.actions is not None else res.maps
        polished = S.solve(init=init, diagnostics=False, **kw)
        polished.actions = polished.actions if polished.actions is not None else res.actions
        rows[-1]["polish"] = int(polished.evaluations); rows[-1]["seconds"] += time.time() - t0
        if verbose:
            print(f"settle march: polish on the whole strip at T = {T:g}: {polished.evaluations} evaluations, {time.time() - t0:.1f}s", flush=True)
        res = polished
    if res.evaluations and diagnostics:                              # the deferred diagnostics, on the whole strip
        S.freeze_before(0.0)
        res.solve_kw.pop("diagnostics", None)
        S._diagnostics(res)
    res.march = rows; res.march_stop = stop
    res.march_settle = float(settle); res.march_floor = floor
    # the excess cost's tail from the march's own gap sequence: the ratio of the last two gaps, a window apart (at
    # the floor the gaps no longer measure the transient: the loss path's own decay, already in the result, stays)
    if stop != "floor" and len(rows) >= 3 and abs((rows[-1]["T"] - rows[-2]["T"]) - L) <= eps:
        factor = {a: rows[-1]["gap"][a] / rows[-2]["gap"][a] for a in rows[-1]["gap"] if rows[-2]["gap"][a] > 0}
        S.excess_tail(res, factor=factor, source="march gaps")
    return res


def march_model(model: Model, numerics=None, past=None, continuation=None, **solve_kw):
    """The march for a model of kind transition with horizon.settle (the file form of solve()): the past from
    its block (or the keyword), the model at each T by with_horizon(window=T)."""
    hz = model.horizon
    if past is None:
        past = Past.from_block(hz.past)
    past = Past.of(past)
    if continuation is None:
        continuation = hz.continuation or "stationary"
    return march(lambda T: model.with_horizon(window=float(T), settle=None), past, continuation, hz.settle, numerics, **solve_kw)


def transition(old, new, T: Optional[float] = None, numerics=None, continuation="stationary",
               settle: Optional[float] = None, step: Optional[float] = None, max_window: Optional[int] = None, **solve_kw):
    """The transition from the stationary regime `old` (a converged StationaryResult, a Past, or a stationary
    Model / ModelBuilder / dict / path solved on the fly; a list of initial shocks is accepted, with the game
    then ending at T) to the regime `new` (a Model / ModelBuilder / dict / path; its horizon becomes
    `transition` with window T, the discount kept), under `numerics` (a Numerics or a dict of its fields laid
    over the new model's: nodes per side, default 12; continuation_nodes for the stationary continuation,
    default the same; the unit kept, the breakpoints and unit_range of the old kind dropped), continued by the
    new model's stationary equilibrium (`continuation="stationary"`, solved on the past's window, or a
    converged StationaryResult of the new model; "end" ends the game at T).  Keyword arguments go to solve()
    (tol, max_evaluations, verbose, ...); the start is the new stationary maps unless `start` is given or the
    game ends at T.  Returns the Result with res.past and res.stationary attached.

    Exactly one of `T` and `settle` is given.  With `settle`, the horizon is an output: the march in T of
    march() (start at the T = 0 pass of transition_gap on the smallest strip, one unit; the first solve at T = L
    and one window per step, or `step` when given, each solve
    warm-started from the previous, stop when the best-response rules on the last window [T - L, T] are within
    `settle` of the stationary rules or at `max_window` windows, default 8); the result carries
    res.extra["window"] (the T found), res.march (the rows (T, gap, evaluations, seconds, monitor)),
    res.march_stop ("settled", "settled at T = 0" or "max_window": the settled flag then stays) and
    res.settled as before."""
    if (T is None) == (settle is None):
        raise ValueError("transition() takes exactly one of T (the horizon) and settle (the tolerance the horizon is found for)")
    if T is None and (continuation == "end" or continuation is None):
        raise ValueError("transition(settle=) needs a stationary continuation: the march measures the rules against it")
    num = Numerics.of(numerics)
    if T is not None:
        if step is not None or max_window is not None:
            raise ValueError("step and max_window belong to the march: transition(old, new, settle=...)")
        model, past, continuation = _transition_model(old, new, T, num, continuation)
        from . import solve
        return solve(model, past=past, continuation=continuation, **solve_kw)      # start: solve()'s default, "stationary" with a continuation
    past = Past.of(old)
    if not past.window > 0:
        raise ValueError("transition(settle=) needs a past with a window (a stationary regime): the march starts from one unit")
    return march(lambda T: _transition_model(old, new, T, num, continuation)[0], past, continuation, settle, None,
                 step=step, max_window=max_window, **solve_kw)
