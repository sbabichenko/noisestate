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
import warnings
from dataclasses import replace
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


def _transition_model(old, new, T: float, num: Numerics, continuation, stationary=None):
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
    if stationary and stationary.get("window") is not None:
        hz["stationary"] = {"window": stationary["window"]}
    return Model.from_dict(d), past, continuation


def _numerics_of(numerics, nodes, stationary) -> Numerics:
    num = Numerics.of(numerics)
    if nodes is not None:
        warnings.warn("transition(nodes=) is deprecated, pass Numerics(nodes=) (the alias goes in 0.6)", DeprecationWarning, stacklevel=3)
        num = replace(num, nodes=int(nodes))
    if stationary is not None:
        warnings.warn("transition(stationary=) is deprecated, pass Numerics(continuation_nodes=) and horizon.stationary.window "
                      "(the alias goes in 0.6)", DeprecationWarning, stacklevel=3)
    if stationary and stationary.get("nodes") is not None:
        num = replace(num, continuation_nodes=int(stationary["nodes"]))
    return num


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


def transition_gap(old, new, numerics=None, continuation="stationary") -> Dict[str, float]:
    """The T = 0 pass of the settle march: with no transition solved at all, every agent keeps the new model's
    stationary rules from date zero with the inheritance (the old regime's shocks in the state and in what
    each agent has observed) attached, and one best response per agent is computed from that; returned is,
    per agent, the relative distance of the best-response rule from the stationary rule (max over the
    identified nodes, relative to the stationary rule's peak).  Under a tolerance the stationary equilibrium is
    the transition and nothing needs solving; otherwise the size says how far the inheritance moves the
    rules.  The engine cannot build a strip shorter than the past's window (the old shocks must be forgotten
    by T), so the pass runs on the smallest window [0, L], the strategies frozen at the stationary rules on
    [0, L] and the buffer [L, 2L] alike: the transition() march starts from this point.  Caveat: with the
    old model equal to the new one the gap is not zero but the grid's closed-loop floor (the identity tests
    see ~1e-9 at 16 nodes on Chapter 3, 2e-4 at 5 nodes on the two-firm market), so a tolerance below that
    floor is never met.  `old`, `new`, `numerics` and `continuation` are transition()'s (a stationary
    continuation is required: the pass needs the rules the strategies are held at)."""
    num = Numerics.of(numerics)
    past = Past.of(old)
    if not past.window > 0 or continuation == "end":
        raise ValueError("transition_gap needs a past with a window and a stationary continuation: the pass measures the "
                         "best response from the new model's stationary rules with the old regime's shocks attached")
    model, past, continuation = _transition_model(old, new, past.window, num, continuation)
    S = engines.build(model, None, past=past, continuation=continuation)[0]
    return gap_pass(S, S.stationary_start(), 0.0, S.c.T)


def _monitor_range(T: float, L: float):
    """(lo, hi, label) the time range the march's monitor measures the gap on at horizon T: the window before the
    last one, [T - 2L, T - L], once T >= 2L (the gap on the last window is the handover and is never zero);
    before that what exists below T - L, and the whole strip at T = L."""
    eps = 1e-9 * max(1.0, T)
    if T >= 2 * L - eps:
        return T - 2 * L, T - L, "[T - 2L, T - L]"
    if T > L + eps:
        return 0.0, T - L, "[0, T - L]"
    return 0.0, T, "[0, T]"


def march(make_model: Callable[[float], Model], past: Past, continuation, settle: float, numerics=None, step: Optional[float] = None,
          max_window: Optional[int] = None, verbose: bool = False, **solve_kw):
    """The march in T (docs/design/transition_settle_march.md): make_model(T) is the transition model on [0, T].  The
    first point is the T = 0 pass (transition_gap: the best response from the stationary rules on the smallest
    strip [0, L], nothing solved); under `settle` the smallest-window solve, one or two evaluations from the
    stationary start, is the transition.  Otherwise T grows from L by `step` (default one window L; a
    multiple of the unit), each solve warm-started from the previous maps read on the new grid with the
    stationary rules on the new stretch (warm_maps_from, the sweep's warm start), the continuation solved once,
    and after each solve the monitor: gap_pass on the converged maps over the window before the last one
    (_monitor_range: [T - 2L, T - L] once T >= 2L, the strip below T - L before that, the whole strip at T = L;
    the row's "monitor" says which).  Stops when every agent's gap is under `settle` or when T would pass
    max_window * L (default 8 windows): then res.settled keeps the last solve's diagnostic, its flag when
    above settled_tol, and res.march_stop says "max_window".  res.march is the list of rows {"T", "gap",
    "gap_last", "evaluations", "seconds", "monitor"} (T = 0 first, evaluations 0; gap_last the same pass on
    the last window [T - L, T], the handover, what `settled` measures), res.extra["window"] the T found.
    The monitor certifies the solve one window back: the march's T is one window past the smallest T whose
    explicit solve settles under the tolerance on its last window.
    solve_kw goes to every solve (tol, max_evaluations, deadline, progress, diagnostics)."""
    if not settle > 0:
        raise ValueError("settle must be a positive tolerance (the relative distance of the best-response rules from the stationary ones)")
    if not past.window > 0 or continuation == "end":
        raise ValueError("transition(settle=) needs a past with a window and a stationary continuation: the march measures the "
                         "best-response rules against the new model's stationary rules")
    L = float(past.window)
    step = L if step is None else float(step)
    if not step > 0:
        raise ValueError("step must be positive (a multiple of the unit; default one window)")
    max_window = 8 if max_window is None else int(max_window)
    if max_window < 1:
        raise ValueError("max_window must be at least 1 (windows of the past's L)")
    Tmax = max_window * L; eps = 1e-9 * max(1.0, Tmax)
    rows = []; prev = None; T = L; stop = None
    while True:
        S, num = engines.build(make_model(T), numerics, verbose=verbose, past=past, continuation=continuation)
        continuation = S.c.cont                                     # solved once, shared by every step
        kw = {**num.solve_kw(), **solve_kw}
        t0 = time.time()
        if prev is None:
            gap0 = gap_pass(S, S.stationary_start(), 0.0, T)
            rows.append(SweepPoint({"T": 0.0, "gap": gap0, "evaluations": 0, "seconds": time.time() - t0, "monitor": "[0, L] from the stationary rules"}))
            if verbose:
                print(f"settle march: T = 0 (the stationary rules): gap {max(gap0.values()):.2e}, {time.time() - t0:.1f}s", flush=True)
            if max(gap0.values()) <= settle:
                stop = "settled at T = 0"
            t0 = time.time()
            res = S.solve(start="stationary", **kw)
        else:
            res = S.solve(init=S.warm_maps_from(prev), **kw)
        lo, hi, label = _monitor_range(T, L)
        gaps = gap_passes(S, res.maps, {"monitor": (lo, hi), "last": (T - L, T)})
        gap = gaps["monitor"]
        rows.append(SweepPoint({"T": float(T), "gap": gap, "gap_last": gaps["last"], "evaluations": int(res.evaluations),
                                "seconds": time.time() - t0, "monitor": label}))
        if verbose:
            print(f"settle march: T = {T:g}: gap {max(gap.values()):.2e} on {label}, {res.evaluations} evaluations, "
                  f"{time.time() - t0:.1f}s", flush=True)
        if stop is not None:
            break
        if max(gap.values()) <= settle:
            stop = "settled"; break
        if T + step > Tmax + eps:
            stop = "max_window"; break
        prev = res; T = T + step
    res.march = rows; res.march_stop = stop
    res.march_settle = float(settle)
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


def transition(old, new, T: Optional[float] = None, numerics=None, continuation="stationary", nodes=None, stationary=None,
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
    game ends at T.  `nodes=` and `stationary={"nodes": m}` are accepted as aliases of the numerics fields
    until 0.6, each with a DeprecationWarning.  Returns the Result with res.past and res.stationary attached.

    Exactly one of `T` and `settle` is given.  With `settle`, the horizon is an output: the march in T of
    march() (start at the T = 0 pass of transition_gap, grow T by `step`, default one window, each solve
    warm-started from the previous, stop when the best-response rules on the window before the last are within
    `settle` of the stationary rules or at `max_window` windows, default 8); the result carries
    res.extra["window"] (the T found), res.march (the rows (T, gap, evaluations, seconds, monitor)),
    res.march_stop ("settled", "settled at T = 0" or "max_window": the settled flag then stays) and
    res.settled as before."""
    if (T is None) == (settle is None):
        raise ValueError("transition() takes exactly one of T (the horizon) and settle (the tolerance the horizon is found for)")
    if T is None and (continuation == "end" or continuation is None):
        raise ValueError("transition(settle=) needs a stationary continuation: the march measures the rules against it")
    num = _numerics_of(numerics, nodes, stationary)
    if T is not None:
        if step is not None or max_window is not None:
            raise ValueError("step and max_window belong to the march: transition(old, new, settle=...)")
        model, past, continuation = _transition_model(old, new, T, num, continuation, stationary)
        from . import solve
        return solve(model, past=past, continuation=continuation, **solve_kw)      # start: solve()'s default, "stationary" with a continuation
    past = Past.of(old)
    if not past.window > 0:
        raise ValueError("transition(settle=) needs a past with a window (a stationary regime): the march starts from its window L")
    return march(lambda T: _transition_model(old, new, T, num, continuation, stationary)[0], past, continuation, settle, None,
                 step=step, max_window=max_window, **solve_kw)
