"""Regression tests for what the randomized campaign (extras/fuzz, tests/test_fuzz.py) found, each on its shrunk repro."""
import os
import sys

import numpy as np
import pytest

import noisestate as ns

HERE = os.path.dirname(os.path.abspath(__file__))
FUZZ = os.path.join(HERE, "..", "extras", "fuzz")
if FUZZ not in sys.path:
    sys.path.insert(0, FUZZ)

import lqg  # noqa: E402


def _correlated(c, breakpoints=None, nodes=12):
    """One agent; its row's noise also loads the state's shock (w0), with weight c."""
    noise = {"w1": 1.0, "w0": c} if c else {"w1": 1.0}
    d = {"shocks": ["w0", "w1"], "states": {"X": {"drift": {"X": -0.5, "D": 1.0}, "noise": {"w0": 1.0}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.5}, "noise": noise}},
                          "loss": [[1.0, "X", "X"], [0.3, "D", "D"]]}},
         "horizon": {"kind": "finite", "T": 1.0, "discount": 0.0}, "numerics": {"nodes": nodes}}
    if breakpoints:
        d["numerics"]["breakpoints"] = breakpoints
    return d


@pytest.mark.parametrize("breakpoints", [None, [0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0]])
@pytest.mark.parametrize("c", [0.3, 0.8])
def test_a_row_correlated_with_a_state_is_exact_on_the_finite_engine(c, breakpoints):
    """A row whose noise loads the state's shock is answered at once (the Kalman gain at t = 0 is Sig Gam' (Gam Gam')^-1),
    so the map at age 0 is not zero on a triangle's degenerate corner row, where the projection on the rows has no
    quadrature weight.  The corners were left to the ridge (zero): the reported equilibrium, built from the raw maps,
    was 3.5e-4 off the closed form at 12 nodes (1.1e-4 at 20: algebraic), while the best response's own world was
    exact, and the representation error sat at 0.5 to 0.9 whatever the nodes (UNDER-RESOLVED: raise nodes, which did
    not help).  The corners now carry the point condition, as with a past: exact to 1e-11."""
    d = _correlated(c, breakpoints)
    res = ns.solve(ns.Model.from_dict(d)).require_ok()
    assert res.costs["a"] == pytest.approx(lqg.solve(d)["a"], rel=1e-10)
    assert res.representation_error["a"] < 1e-8


def test_the_finite_chapter_6_market_converges_spectrally():
    """The same corner on Chapter 6's finite market: the flow row loads wZ, which also drives the inventory Q.  Before,
    the market maker's cost crept 0.1285511, 0.1285441, 0.1285388, 0.1285368 at 8, 10, 14, 18 nodes with a
    representation error of 0.15 at every one; now 0.12853419 at 10 nodes (2.3e-7 from 14), 0.1285342204 at 14 and 18, the error 1.5e-6 at 10."""
    import generate as G
    res = {n: ns.solve(ns.Model.from_dict(G.ch6_market(0.1, 0.2, 0.5, 0.0, 1.0, True, "finite", n))) for n in (10, 14)}
    assert res[10].costs["mm"] == pytest.approx(res[14].costs["mm"], rel=1e-6)
    assert res[14].costs["mm"] == pytest.approx(0.1285342204, rel=1e-9)
    assert max(res[10].representation_error.values()) < 1e-5
