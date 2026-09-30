"""The publication policy's `window cost` check: the costs' truncation by the stationary window, measured.

The `window` check bounds the kernels' tail (2% of a kernel's peak over the last tenth of the window), which is a bound
on the strategies' truncation, not on the costs': the randomized-testing campaign found one-agent problems passing it
with costs 5e-4 from their closed form, the Kyle-Back example passes it at L = 8 with the trader's cost 6.5e-5 from its
long-window value, and the README's Chapter 1 script at L = 3 with its costs 16% off.  `window cost` re-solves on a
window 1.5 times as long (the grid extended by panels like the last one, started from the equilibrium) and requires the
costs to move by less than window_cost_tol (1e-6 of the largest cost, refine()'s level).  It is measured on request:
res.check_window(), which require_ok() runs when its policy needs it; the exploratory policy and the minimum an
equilibrium is verified by do not."""
import pytest

import noisestate as ns
from noisestate.diagnostics import MINIMUM, Policy, Status

# extras/fuzz lqg1_stationary seed 253: one agent, two controls, two states; the closed form (extras/fuzz/lqg.py, the
# discounted Riccati and Kalman-Bucy filter) is 0.5264090; on this window the package's cost is 0.526152
LQG = {"name": "lqg253", "shocks": ["wX0", "wX1", "wa00"],
       "states": {"X0": {"drift": {"X0": -0.3965, "D0b": -0.5674}, "noise": {"wX0": 0.541}},
                  "X1": {"drift": {"X0": 0.02936, "X1": -0.834, "D0a": 0.6847, "D0b": -0.1573}, "noise": {"wX1": 0.627}}},
       "agents": {"a0": {"controls": ["D0a", "D0b"], "signals": {"y0": {"drift": {"X1": 1.795}, "noise": {"wa00": 1.007}}},
                         "loss": [[1.453, "D0a", "D0a"], [0.388, "D0a", "D0b"], [-1.687, "D0a", "X1"], [-0.354, "D0a", "X0"],
                                  [1.212, "D0b", "D0b"], [-0.6061, "D0b", "X1"], [-0.5417, "D0b", "X0"], [1.66, "X1", "X1"],
                                  [0.3162, "X1", "X0"], [0.3866, "X0", "X0"]]}},
       "horizon": {"kind": "stationary", "window": 8.0, "discount": 0.4932}, "numerics": {"nodes": 18}}
CLOSED_FORM = 0.5264090


def test_the_kernel_tail_passes_a_cost_the_longer_window_moves():
    res = ns.solve(ns.Model.from_dict(LQG)).require_converged()
    assert res.window_tail < 0.02 and res.diagnostics.statuses["window"] is Status.PASSED        # 1.5%: the kernels pass
    assert abs(res.costs["a0"] - CLOSED_FORM) > 2e-4                                            # 0.526152: 4.9e-4 off
    st = res.diagnostics.statuses
    assert st["window cost"] is Status.SKIPPED                      # not measured by the solve
    assert res.diagnostics.assess(Policy.EXPLORATORY).accepted and res.window_check is None     # nothing run for it
    with pytest.raises(ns.DiagnosticsError, match="WINDOW TOO SHORT FOR THE COSTS"):
        res.require_ok()
    wc = res.window_check
    assert wc is not None and wc.converged and not wc.ok and wc.window == 12.0
    # the measured change is the cost's own error to within the longer window's (3%: 4.8e-4 against 4.9e-4)
    assert wc.cost_change == pytest.approx(abs(res.costs["a0"] - CLOSED_FORM) / abs(res.costs["a0"]), rel=0.1)
    assert res.diagnostics.statuses["window cost"] is Status.FAILED
    assert "window cost" not in MINIMUM and "window cost" in Policy.PUBLICATION.required


def test_a_window_long_enough_for_the_costs_passes():
    d = dict(LQG, horizon={"kind": "stationary", "window": 20.0, "discount": 0.4932}, numerics={"nodes": 30})
    res = ns.solve(ns.Model.from_dict(d))
    assert res.require_ok() is res and res.window_check.cost_change < 1e-6
    assert res.costs["a0"] == pytest.approx(CLOSED_FORM, abs=1e-6)
    assert res.diagnostics.statuses["window cost"] is Status.PASSED
    assert res.to_dict()["window_check"]["ok"] is True


def test_turning_the_diagnostics_off_measures_nothing():
    res = ns.solve(ns.Model.from_dict(LQG), diagnostics=False)
    with pytest.raises(ns.DiagnosticsError):
        res.require_ok()
    assert res.window_check is None and res.diagnostics.statuses["window cost"] is Status.SKIPPED


def test_a_grid_given_by_breakpoints_is_extended_with_the_window():
    """A model whose numerics give breakpoints (as every model solved on automatic age panels carries them) was refused
    by check_window(): the window was lengthened before the grid, and breakpoints ending at the old window failed
    validation in between."""
    d = dict(LQG, numerics={"nodes": 12, "breakpoints": [0.0, 1.0, 2.0, 4.0, 8.0]})
    res = ns.solve(ns.Model.from_dict(d)).require_converged()
    wc = res.check_window()
    assert wc.window == 12.0 and wc.converged
    assert list(wc.longer.model.numerics.breakpoints)[:5] == [0.0, 1.0, 2.0, 4.0, 8.0]
    assert wc.longer.model.numerics.breakpoints[-1] == 12.0
