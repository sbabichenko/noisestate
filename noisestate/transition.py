"""transition(old, new, T): the equilibrium path of a regime change, from the old stationary regime to the new.

    from noisestate import transition
    res = transition("examples/ch3_two_player.yaml", new_model, T=6.0, numerics={"nodes": 12})
    res.past, res.stationary, res.settled, res.summary()

The old regime is a converged StationaryResult or a stationary model (a Model, ModelBuilder, dict or
path, solved on the fly); the new regime is a model of any kind, whose horizon is rewritten to a
transition on [0, T] continued by its own stationary equilibrium (solved on the past's window at
numerics.continuation_nodes, default the transition's nodes).  The solve starts from the new stationary maps
(start="stationary"), the one default of every transition with a continuation (this helper, solve(model,
past=, continuation=) and a file of kind transition alike): it is where a settled transition ends and, on a
moderate change, a few best responses from the answer.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Dict

import numpy as np

from . import engines
from .spec import Model, ModelBuilder
from .numerics import Numerics
from .past import Past


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


def gap_pass(S, maps: Dict[str, np.ndarray], lo: float, hi: float) -> Dict[str, float]:
    """The best-response pass of the settle monitor: given the maps (every agent's rule on the strip, the
    buffer's frozen at the stationary ones), each agent's best response is computed once and its distance from
    the stationary rule is measured on the identified nodes with time in [lo, hi] (the buffer excluded): the
    largest difference over controls, rows and nodes, relative to the stationary rule's own peak.  At a
    converged transition the best response is the solved rule to tolerance, so this is the maps' own distance
    from the stationary ones on that window; from the stationary maps it is the one-shot deviation."""
    c = S.c; g = c.g
    eps = 1e-9 * max(1.0, c.Tg)
    sel = ~c.buffer & (g.t >= lo - eps) & (g.t <= hi + eps)
    out = {}
    for a in S.model.agents:
        gm, _ = S.best_response(a, maps)
        fr = c.frozen[a.name]
        keep = S._identified(a).reshape(len(a.signals), -1)[:, :c.N] & sel
        dev = np.abs(gm[:, :, :c.N] - fr) * keep
        out[a.name] = float(dev.max() / max(1e-300, np.abs(fr).max()))
    return out


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


def transition(old, new, T: float, numerics=None, continuation="stationary", nodes=None, stationary=None, **solve_kw):
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
    until 0.6, each with a DeprecationWarning.  Returns the Result with res.past and res.stationary attached."""
    num = _numerics_of(numerics, nodes, stationary)
    model, past, continuation = _transition_model(old, new, T, num, continuation, stationary)
    from . import solve
    return solve(model, past=past, continuation=continuation, **solve_kw)      # start: solve()'s default, "stationary" with a continuation
