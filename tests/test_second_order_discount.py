"""The second-order check at a positive discount is the discounted objective's own form: a deviation's responses weighted
by e^{-rho tau / 2} at their age (StationarySolver._half_discounted; docs/guards.md).  The unweighted (average-cost) form,
used before, is the flow loss of a deviation made at every date, the past's included; with a loss that is not positive
semidefinite its sign is not the discounted objective's."""
import os
import sys

import pytest

import noisestate as ns
from helpers import HERE

sys.path.insert(0, os.path.join(HERE, "..", "extras", "fuzz"))
import lqg  # noqa: E402


def impact(lam, window=24.0, nodes=48):
    """One agent against its own transient price impact: dX = (-X + lam D) dt + dW, a view dy = X dt + 0.5 dv, loss
    X D + D^2, rho = 1.  Convex over every feasible deviation iff 1 + lam / (1 + rho / 2) >= 0, lam >= -1.5: the symbol
    of the e^{-rho t / 2}-rescaled form, and the discounted Riccati equation's condition for a stabilizing solution;
    the unweighted form's threshold is 1 + lam >= 0."""
    return ns.Model.from_dict({
        "name": "impact", "shocks": ["w", "v"], "states": {"X": {"drift": {"X": -1.0, "D": lam}, "noise": {"w": 1.0}}},
        "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"v": 0.5}}},
                         "loss": [[1.0, "X", "D"], [1.0, "D", "D"]]}},
        "horizon": {"kind": "stationary", "window": window, "discount": 1.0}, "numerics": {"nodes": nodes}})


def test_a_convex_discounted_problem_is_not_called_a_saddle():
    """lam = -1.1: the package's equilibrium is the discounted Riccati optimum (-0.1561916; the window of 24 is 2e-6
    short), and the check passes; the unweighted form read -0.26 of the largest curvature here."""
    m = impact(-1.1); res = ns.solve(m).require_converged()
    assert abs(res.costs["a"] - lqg.solve(m.to_dict())["a"]) < 1e-5
    so = res.second_order["a"]
    assert so["ok"] and so["min"] > 1e-3


def test_a_problem_past_the_discounted_threshold_is_flagged():
    """lam = -1.6: no stabilizing Riccati solution (the agent pumps its own impact without bound); the check says so."""
    m = impact(-1.6, window=12.0, nodes=24)
    with pytest.raises(Exception, match="not stable|Riccati"):
        lqg.solve(m.to_dict())
    assert not ns.solve(m).second_order["a"]["ok"]


def test_the_transparent_market_trader_is_at_a_minimum():
    """Chapter 6's transparent market at gamma = 0.1 on its window of 8: the unweighted form read -0.062 (a saddle); the
    discounted objective's form is +0.0059, and the equilibrium is a critical point of that objective, not of the
    unweighted one (their slopes along the unweighted form's lowest direction: 2e-4 against 5.8e-2, nsissues LOG)."""
    from test_monitoring import market
    res = ns.solve(market(0.1, nodes=24)).require_converged()
    for a, so in res.second_order.items():
        assert so["ok"] and so["min"] > 0, (a, so)
    assert res.second_order["trader"]["min"] == pytest.approx(0.00594, abs=2e-4)
