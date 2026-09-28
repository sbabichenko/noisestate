"""Regression tests for the 2026-09-28 limits pass: a finite horizon long against the model's time scales (graded panels,
time_panels.py), the Chapter 5 market's representation floor (the window edge less a lag as a cut), and the stationary
engine's risk-averse agents at and past the breakdown (and their entropic cost's lattice at long windows)."""
import math
import warnings

import numpy as np
import pytest

import noisestate as ns
from noisestate import Numerics
from noisestate.grid import AgeGrid
from noisestate.time_panels import graded_breakpoints

from helpers import slow


def lqg(T, nodes=12, **params):
    """One agent regulating a state it sees through noise, dX = D dt + sigma dW0, dy = sqrt(p) X dt + dW1, loss X^2 + r D^2:
    rates of about 3 (the filter sigma sqrt p, the regulator 1 / sqrt r)."""
    p = {"p": 3.0, "r": 0.1, "sigma": 1.0, **params}
    return ns.Model.from_dict({"name": "lqg", "params": p, "shocks": ["w0", "w1"], "states": {"X": "D dt + sigma dw0"},
                               "agents": {"a": {"controls": "D", "observes": {"y": "sqrt(p) X dt + dw1"}, "loss": "X^2 + r D^2"}},
                               "horizon": {"T": T}, "numerics": {"nodes": nodes}})


def lqg_cost(T, p=3.0, r=0.1, s=1.0):
    """The separation theorem's cost: int_0^T Sigma + P p Sigma^2 dt (Kalman from a known start, Riccati to T)."""
    from scipy.integrate import quad
    sig = lambda t: s / math.sqrt(p) * math.tanh(s * math.sqrt(p) * t)
    ric = lambda t: math.sqrt(r) * math.tanh((T - t) / math.sqrt(r))
    pts = [x for x in (1 / (s * math.sqrt(p)), T - math.sqrt(r)) if 0 < x < T]
    return quad(lambda t: sig(t) + ric(t) * p * sig(t) ** 2, 0, T, epsabs=0, epsrel=1e-13, points=pts or None, limit=200)[0]


# ------------------------------------------------------------------------------------------------ long horizons
def test_graded_breakpoints_cut_both_ends():
    assert graded_breakpoints(30.0, 1.0) == [0.0, 1.0, 3.0, 7.0, 23.0, 27.0, 29.0, 30.0]
    assert graded_breakpoints(10.0, 2.0) == [0.0, 2.0, 8.0, 10.0]          # the middle [2, 8] next to 2 and 2 wide
    assert graded_breakpoints(1.0, 0.5) == [0.0, 1.0]                       # a first width of half the horizon: one panel


def test_a_horizon_one_panel_resolves_is_solved_as_before():
    """T = 1 at 12 nodes passes the resolution check on the one panel: no grading, no breakpoints recorded, and the same
    numbers as with the grading switched off."""
    res = ns.solve(lqg(1.0))
    off = ns.solve(lqg(1.0), {"settings": {"auto_panels_max": 0}})
    assert res.numerics.breakpoints is None and "graded" not in res.message
    assert res.costs == off.costs and np.array_equal(res.maps["a"], off.maps["a"])
    assert abs(res.costs["a"] - lqg_cost(1.0)) < 1e-9


def test_a_long_horizon_is_graded_to_the_closed_form():
    """T = 10 on one panel of 12 nodes: cost 1.3e-3 off the closed form, representation error 1.8e-2 (the resolution row
    failed and nothing else said so).  The graded re-solve lands on [0, 1, 3, 7, 9, 10]-type panels (w = 2, then 1) and
    the closed form to 1e-8; the breakpoints are recorded, and solving with them gives the same result."""
    one = ns.solve(lqg(10.0), {"settings": {"auto_panels_max": 0}})
    assert abs(one.costs["a"] / lqg_cost(10.0) - 1) > 1e-3 and max(one.representation_error.values()) > 1e-2
    res = ns.solve(lqg(10.0))
    assert res.converged and res.numerics.breakpoints is not None and "graded" in res.message
    assert abs(res.costs["a"] / lqg_cost(10.0) - 1) < 1e-8
    assert max(res.representation_error.values()) <= 1e-6 and res.diagnostics.statuses["resolution"] is ns.Status.PASSED
    again = ns.solve(lqg(10.0), {"breakpoints": res.numerics.breakpoints})
    assert again.costs == res.costs


def test_explicit_breakpoints_are_never_regraded():
    res = ns.solve(lqg(10.0), {"breakpoints": [0.0, 5.0, 10.0]}, diagnostics=False)
    assert res.numerics.breakpoints == [0.0, 5.0, 10.0] and "graded" not in res.message


def test_time_scales_far_below_the_horizon_ask_for_a_rescaling():
    """The noise scaled by 1e8 (a filter rate of 1.7e8 on T = 1): singular on the one panel and on every graded grid within
    the budget; the error says the grading was tried and what to do."""
    with pytest.raises(ValueError, match="rescale"):
        ns.solve(lqg(1.0, sigma=1e8), diagnostics=False)


@slow()
def test_a_thirty_time_constant_horizon_is_graded():
    """T = 30 (190% off on one panel): graded panels [0, 1, 3, 7, 23, 27, 29, 30] give the closed form to 1e-9 with the
    representation error 5.6e-7, where 10 uniform panels gave 2.9e-6 and 8.4e-5."""
    res = ns.solve(lqg(30.0))
    assert res.numerics.breakpoints == [0.0, 1.0, 3.0, 7.0, 23.0, 27.0, 29.0, 30.0]
    assert abs(res.costs["a"] / lqg_cost(30.0) - 1) < 1e-9 and max(res.representation_error.values()) <= 1e-6
    assert res.second_order["a"]["ok"]


@slow()
def test_a_hundred_time_units_are_graded_at_eight_nodes():
    """T = 100 at 8 nodes: 2950 times the closed form on the one panel (smooth kernels of the wrong equilibrium, a Chebyshev
    tail of 0.16), graded to 15 panels and the cost to 1e-7 (55 s)."""
    res = ns.solve(lqg(100.0, nodes=8), diagnostics=False)
    assert len(res.numerics.breakpoints) > 10 and abs(res.costs["a"] / lqg_cost(100.0) - 1) < 1e-7


def test_a_failure_more_nodes_fix_is_left_to_the_nodes():
    """Chapter 1's game at 4 nodes fails the resolution check; the trial finds 8 nodes resolve it better than grading, so
    the one panel's result stands, as before (and a threshold that refuses every system raises at once)."""
    m = ns.load(ns.example("ch1_two_player_finite")).with_numerics(nodes=4)
    res = ns.solve(m)
    assert res.numerics.breakpoints is None and "graded" not in res.message
    with pytest.raises(ValueError, match="singular"):
        ns.solve(m, {"settings": {"foc_rcond": 1.0}})


# ------------------------------------------------------------------------------------------------ Chapter 5's floor
def test_the_window_edge_less_a_lag_is_a_cut_of_the_geometric_panels():
    """The kernels jump at L - d (a lagged read past the window is cut off): 2.3e-6 and 7.7e-6 of the peak on the Chapter 5
    market's prices and orders at L - tau = 23.5.  Inside the last geometric panel [15, 24] that jump was an error no node
    count removed (the representation error's floor, 1.1e-6 .. 1.9e-6 at 12 .. 16 nodes)."""
    assert AgeGrid.breakpoints_from_delays(24.0, [0.5], 0.5, 8.0)[-6:] == [8.0, 9.0, 11.0, 15.0, 23.5, 24.0]
    assert AgeGrid.breakpoints_from_delays(8.0, [0.5], 0.5, 8.0) == list(np.arange(0.0, 8.0 + 1e-12, 0.5))   # unit panels to L: as before
    m = ns.load(ns.example("ch5_cycle_market"))
    assert 23.5 in [float(b) for b in ns.engines.stationary(m).c.grid.breakpoints]


@slow()
def test_the_chapter_5_market_is_resolved_and_the_error_falls_with_nodes():
    """The shipped example (8 nodes) passes the resolution check: 3.6e-7, 7.0e-8 at 12 nodes (1.3e-5 and 1.1e-6 before)."""
    m = ns.load(ns.example("ch5_cycle_market"))
    r8, r12 = ns.solve(m), ns.solve(m.with_numerics(nodes=12))
    e8, e12 = max(r8.representation_error.values()), max(r12.representation_error.values())
    assert e8 < 1e-6 and e12 < e8 / 3
    assert abs(r8.costs["firm0"] - r12.costs["firm0"]) < 3e-7


# ------------------------------------------------------------------------------------------------ stationary CARA
def lq_signal(theta, window=8.0, nodes=16):
    """tests/test_cara_ext.py's one-agent model under consistent planning, with the signal sqrt(3) X dt + dW1."""
    return ns.Model.from_dict({
        "name": "lqs", "params": {"sigma": 1.0, "r": 0.1, "p": 3.0}, "shocks": ["w0", "w1"], "states": {"X": "D1 dt + sigma dw0"},
        "agents": {"p1": {"controls": "D1", "observes": "sqrt(p) X dt + dw1", "loss": "X^2 + r D1^2", "risk_aversion": theta}},
        "horizon": {"window": window, "discount": 0.5}, "numerics": {"nodes": nodes}})


def test_the_entropic_lattice_steps_are_capped():
    """The date-0 entropic cost's lattices were L / 40, 80, 160: the step, and the lattice error, grew with the window (the gap
    between the two finest levels 6e-4 / 2.5e-3 / 9.3e-3 at L = 4 / 8 / 16 on the model below at theta 0.5, and the
    extrapolated excess 1.1% off at 16).  Capped at 0.2 (lattices through L), and unchanged up to L = 8."""
    from noisestate.stationary import StationarySolver
    assert StationarySolver.entropic_steps(8.0) == [8.0 / 40, 8.0 / 80, 8.0 / 160]
    assert StationarySolver.entropic_steps(4.0) == [4.0 / 40, 4.0 / 80, 4.0 / 160]
    assert np.allclose(StationarySolver.entropic_steps(16.0), [0.2, 0.1, 0.05])
    assert np.allclose(StationarySolver.entropic_steps(10.0), [0.125, 0.0625, 0.03125])


@slow()
def test_stationary_risk_aversion_near_the_breakdown_is_reached_in_steps():
    """theta 1: the path 0, 0.5, 1 jumped onto a spurious fixed point past the breakdown (theta mu_max 8.95, reported not
    converged); with the finite engine's steps (0.783, 0.942, 1) it reaches the equilibrium, theta mu_max 0.886 (25 s)."""
    res = ns.solve(lq_signal(1.0), diagnostics=False)
    assert res.converged and 0.85 < res.risk["p1"]["theta_mu_max"] < 0.92
    assert "theta scaled by 0, 0.5, 0.783, 0.942, 1" in res.message


@slow()
def test_stationary_risk_aversion_past_the_breakdown_raises():
    """theta 2: no equilibrium past theta about 1.2 (the steps' fixed points there lie past the breakdown or their
    correction's solve fails); RiskBreakdown saying where the path stopped (1.17), as on the finite engine, not a result
    that is merely not converged (130 s: the halved steps)."""
    with pytest.raises(ns.RiskBreakdown) as exc:
        ns.solve(lq_signal(2.0), diagnostics=False)
    assert exc.value.reached is not None and 1.0 <= exc.value.reached < 1.25


@slow()
def test_the_entropic_lattice_does_not_coarsen_with_the_window():
    """At L = 16 the lattice gap is the L = 8 one (2.5e-3, was 9.3e-3)."""
    r8 = ns.solve(lq_signal(0.5, 8.0), diagnostics=False).risk["p1"]
    r16 = ns.solve(lq_signal(0.5, 16.0), diagnostics=False).risk["p1"]
    assert r16["entropic_lattice_gap"] < 1.1 * r8["entropic_lattice_gap"]
