"""transition(old, new, T): the equilibrium path of a regime change, from the old stationary regime to the new.

    from noisestate import transition
    res = transition("examples/ch3_two_player.yaml", new_model, T=6.0, nodes=12)
    res.past, res.stationary, res.settled, res.summary()

The old regime is a converged StationaryResult or a stationary model (a Model, ModelBuilder, dict or
path, solved on the fly); the new regime is a model of any kind, whose horizon is rewritten to a
transition on [0, T] continued by its own stationary equilibrium (solved at `nodes` per panel on the
past's window).  The solve starts from the new stationary maps (start="stationary"), which is where a
settled transition ends and, on a moderate change, a few best responses from the answer; solve()
itself keeps start="zero".
"""
from __future__ import annotations

from .spec import Model, ModelBuilder
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
    if hasattr(src, "model") and hasattr(src, "Z"):
        return {"model": src.model.to_dict()}
    return {"initial": [sh.to_dict() for sh in past.initial]}


def transition(old, new, T: float, nodes: int = 12, continuation="stationary", stationary=None, **solve_kw):
    """The transition from the stationary regime `old` (a converged StationaryResult, a Past, or a stationary
    Model / ModelBuilder / dict / path solved on the fly; a list of initial shocks is accepted, with the game
    then ending at T) to the regime `new` (a Model / ModelBuilder / dict / path; its horizon becomes
    `transition` with window T and `nodes` per side, the discount and unit kept), continued by the new
    model's stationary equilibrium (`continuation="stationary"`, solved at `nodes` per panel on the past's
    window, or a converged StationaryResult of the new model; `stationary={"nodes": m}` sizes the solve;
    "end" ends the game at T).  Keyword arguments go to solve() (tol, max_evaluations, verbose, settings,
    ...); the start is the new stationary maps unless `start` is given or the game ends at T.  Returns the
    TransitionResult with res.past and res.stationary attached."""
    past = Past.of(old)
    m = _model_of(new)
    d = m.to_dict()
    hz = d.setdefault("horizon", {})
    for k in ("breakpoints", "unit_range"):
        hz.pop(k, None)
    block = _past_block(old, past)
    if not past.window > 0 and continuation == "stationary":
        continuation = "end"
    hz.update(kind="transition", window=float(T), nodes=int(nodes), past=block,
              continuation=continuation if isinstance(continuation, str) else "stationary")
    if stationary:
        hz["stationary"] = dict(stationary)
    model = Model.from_dict(d)
    from . import solve
    kw = dict(solve_kw)
    if "start" not in kw and continuation != "end":
        kw["start"] = "stationary"
    return solve(model, past=past, continuation=continuation, **kw)
