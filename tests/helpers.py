"""The builders the transition and means tests share (importable as `helpers` from every test).

Nothing here solves differently from the tests it was lifted from: an example is loaded by name, a model
is solved stationary at a number of nodes and handed to the spectral finite engine as its own past and
continuation, the delayed Chapter 1 game gets its stationary equivalent at a window, the two-firm Chapter 5
market is built with its ties and linear terms dropped, one best response from the frozen stationary maps
is measured against them node by node, and a solve is reduced to a record (costs, evaluations, hashes of Z)
that extras/compare_baseline.py writes and tests/test_baseline.py compares.  `slow` is the NOISESTATE_SLOW
gate (tests/SLOW.md lists the gated tests and what each pins); it also carries the `slow` marker, so
`-m slow` selects them.
"""
import hashlib
import os

import numpy as np
import pytest

import noisestate as ns

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples") + os.sep
REFS = os.path.join(HERE, "refs")
IVP = dict(method="DOP853", rtol=1e-12, atol=1e-14)

SLOW_ON = bool(os.environ.get("NOISESTATE_SLOW"))


def slow(reason="slow; set NOISESTATE_SLOW=1"):
    """Decorator: the test runs under NOISESTATE_SLOW=1 only, and is marked `slow`."""
    skip = pytest.mark.skipif(not SLOW_ON, reason=reason)
    return lambda f: pytest.mark.slow(skip(f))


def slow_param(*values, reason="slow; set NOISESTATE_SLOW=1"):
    """One parametrize case gated the same way: pytest.param(*values, marks=[slow, skipif])."""
    return pytest.param(*values, marks=[pytest.mark.slow, pytest.mark.skipif(not SLOW_ON, reason=reason)])


def example_path(name):
    """examples/<name>.yaml."""
    return EX + name + ".yaml"


def example(name):
    """The shipped example `name` (without .yaml) as a Model."""
    return ns.load(example_path(name))


def example_dict(name):
    """The shipped example `name` as its raw dictionary (to edit before Model.from_dict)."""
    return ns.read_yaml(example_path(name))


def stationary(m, nodes, window=None):
    """`m` solved stationary at `nodes` per panel and checked: the past (and continuation) of a same-model
    transition.  With `window` the horizon is set to kind stationary at that window (the finite examples)."""
    hz = dict(nodes=nodes) if window is None else dict(kind="stationary", window=window, nodes=nodes)
    return ns.solve(m.with_horizon(**hz)).check()


def same_model_solver(m, stat, T, nodes, continuation=True, settings=None, **hz):
    """The spectral finite engine on the strip [0, T] at `nodes` with `stat` (m's stationary equilibrium) as the
    past and, unless continuation=False, as the continuation; `hz` goes to with_horizon (unit, unit_range)."""
    return ns.SpectralFiniteSolver(m.with_horizon(kind="finite", window=T, nodes=nodes, **hz), settings=settings,
                                   past=stat, continuation=stat if continuation else None)


def same_model_setup(m, T, nodes, stat_nodes=None, window=None, continuation=True, **hz):
    """(stat, solver): `m` as its own past and continuation on [0, T] (stationary at stat_nodes, default nodes)."""
    stat = stationary(m, nodes if stat_nodes is None else stat_nodes, window)
    return stat, same_model_solver(m, stat, T, nodes, continuation=continuation, **hz)


def delayed_stationary(nodes, window=1.0):
    """examples/ch1_delayed_finite.yaml solved as a stationary game at `window` (its stationary equivalent)."""
    return stationary(example("ch1_delayed_finite"), nodes, window)


def two_firm_market(**over):
    """examples/make_ch5_cycle_market.build(N=2, ...) with the ties and the linear loss terms dropped (kappa
    with them): every first-order condition reads controls across the band, no cyclic symmetry, no means."""
    from make_ch5_cycle_market import build
    d = build(N=2, **over).to_dict()
    d["ties"] = []; d["params"].pop("kappa", None)
    for a in d["agents"].values():
        a["loss"] = [term for term in a["loss"] if len(term) == 3]
    return ns.Model.from_dict(d)


def strip_maps(stat, solver):
    """The stationary maps of `stat` carried onto the solver's strip (the map at age a, on both shock families)."""
    g = solver.c.g; gs = stat.compiled.grid
    return {a.name: np.einsum("fn,urn->urf", gs.interp(g.a), stat.maps[a.name]) for a in solver.model.agents}


def stationary_on_strip(stat, g, name):
    """The stationary kernel at the strip's nodes, a node on its piece's top age edge taking the left limit (the
    kernels jump at a delayed row's delay and at the window edge)."""
    gs = stat.compiled.grid; K = stat.kernel(name); top = np.abs(g.a - g.a1) < 1e-9
    out = gs.interp(g.a, side=+1) @ K
    out[top] = gs.interp(g.a[top], side=-1) @ K
    return out


def one_shot_deviation(solver):
    """Each agent's best response to the solver's frozen stationary maps: the relative deviation from those maps
    on the identified nodes (a vector over the strip's nodes), by agent."""
    c = solver.c; out = {}
    for a in solver.model.agents:
        gm, _ = solver.best_response(a, c.frozen)
        keep = solver._identified(a).reshape(len(a.signals), -1)[:, :c.N]
        out[a.name] = (np.abs(gm - c.frozen[a.name]) * keep / np.abs(c.frozen[a.name]).max()).max(axis=(0, 1))
    return out


def one_shot_from_the_stationary_maps(m, stat, T, nodes, **kw):
    """Each agent's best response to the frozen stationary maps on the strip [0, T] with `stat` as past and
    continuation: the relative deviation from those maps on the identified nodes, by agent, and the solver."""
    solver = same_model_solver(m, stat, T, nodes, **kw)
    return one_shot_deviation(solver), solver


# ------------------------------------------------ the discounted one-agent model with a prior
A1, H1, R1, T1 = -0.3, 1.5, 0.5, 3.0
P0 = 0.8


def prior_model(rho=0.5, nodes=12):
    """One agent on dX = (A1 X + D) dt + dw0 with the signal H1 X dt + dw1, loss X^2 + R1 D^2, finite at T1."""
    return {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": A1, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": H1}, "noise": {"w1": 1.0}}},
                             "loss": [[1.0, "X", "X"], [R1, "D", "D"]]}},
            "horizon": {"kind": "finite", "window": T1, "nodes": nodes, "discount": rho}}


# ------------------------------------------------ the record of a solve
def sha_bits(Z):
    """SHA-256 of Z's raw float64 bytes, row major."""
    return hashlib.sha256(np.ascontiguousarray(Z, dtype=np.float64).tobytes()).hexdigest()


def solve_record(res):
    """A solved result reduced to what a re-baseline compares: the costs (full repr through JSON), their parts,
    the evaluation count, the residual, `settled` for a transition, the means where they are scalars, Z's
    shape and the SHA of its raw bytes (information; Z itself goes to the record's .npz)."""
    means = {k: float(v) for k, v in res.means.items() if np.ndim(v) == 0}
    rec = {"engine": type(res).__name__, "converged": bool(res.converged), "evaluations": int(res.evaluations),
           "residual": float(res.residual), "costs": {k: float(v) for k, v in res.costs.items()},
           "cost_parts": {k: {p: float(x) for p, x in v.items()} for k, v in res.cost_parts.items()},
           "means": means, "Z_shape": list(np.shape(res.world)), "Z_bits": sha_bits(res.world)}
    if getattr(res, "settled", None) is not None:
        rec["settled"] = float(res.settled)
    return rec
