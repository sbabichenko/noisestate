import os, numpy as np, pytest
import noisestate as ns
from noisestate.finite_spectral import SpectralCompiled, SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__))

def _finite(name, **hz):
    """The example with these overrides.  `window=` names the HORIZON'S LENGTH here, which on these
    finite examples is the terminal time T -- the helper puts it under the key the kind keeps."""
    d = ns.read_yaml(os.path.join(HERE, "..", "examples", name))
    d["numerics"] = {**d.get("numerics", {}), **{k: hz.pop(k) for k in list(hz) if k in ("nodes", "unit", "unit_range", "breakpoints")}}
    if "window" in hz and d["horizon"].get("kind", "stationary") != "stationary":
        hz["T"] = hz.pop("window")
    d["horizon"].update(hz)
    return d

def test_delayed_finite_spectral_converges_and_matches_cells():
    """Control lag 0.25 in the state and a delayed observation for player 2 (Chapter 2 style).
    Reference: the first-order cell scheme, Richardson-extrapolated from 60 and 120 cells
    (cost 0.4839003 / 0.4767628, computed 2026-09-03)."""
    d = ns.read_yaml(os.path.join(HERE, "..", "examples", "ch1_delayed_finite.yaml"))
    d.setdefault("numerics", {})["nodes"] = 6
    S = SpectralFiniteSolver(ns.Model.from_dict(d)); res = S.solve()
    assert res.converged and res.residual < 1e-7
    assert len(S.c.g.pieces) == 10                         # 4 time panels of width 0.25
    assert abs(res.costs["player1"] - 0.4839003) < 2e-4 and abs(res.costs["player2"] - 0.4767628) < 2e-4
    # player 2 cannot respond to shocks younger than its observation delay
    t = np.array([0.6, 0.8]); assert np.abs(res.evaluate("D2", "w2", t, t - 0.1)).max() < 1e-10
    assert np.abs(res.evaluate("D2", "w2", t, t - 0.4)).max() > 0.05


def test_loss_lag_that_does_not_divide_the_window_solves():
    """A lagged loss atom (D1@0.3) on a window of 1.0: the panels are closed under the lag (T - k 0.3 as well
    as k 0.3: 7 panels, 28 pieces) where the engine used to die on a bare assertion in map_shift.  Reference:
    the cell scheme Richardson-extrapolated from 40 and 80 cells (0.3947289 / 0.3973767, computed 2026-09-05);
    the spectral costs move by 1e-5 between 4 and 8 nodes per side (player 1: 0.3945422 / 0.3945322)."""
    d = _finite("ch1_two_player_finite.yaml", nodes=4)
    d["agents"]["player1"]["loss"].append([0.1, "D1@0.3", "X"])
    with pytest.warns(UserWarning, match="closed under the lag"):
        res = ns.solve(ns.Model.from_dict(d)).require_converged()
    assert len(res.compiled.g.pieces) == 28 and res.residual < 1e-7
    assert abs(res.costs["player1"] - 0.3947289) < 1e-3 and abs(res.costs["player2"] - 0.3973767) < 1e-3


def test_delay_at_a_trailing_breakpoint_solves():
    """A row delay of 0.9 on a window of 1.0 lost its breakpoint (the trailing panel [0.9, 1] was merged into
    the one before) and was then rejected as 'not a breakpoint'; the panels are now {0, 0.1, 0.9, 1}."""
    d = _finite("ch1_two_player_finite.yaml", nodes=4); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.9
    with pytest.warns(UserWarning, match="closed under the lag"):
        res = ns.solve(ns.Model.from_dict(d)).require_converged()
    assert [round(float(b), 6) for b in res.compiled.g.bp] == [0.0, 0.1, 0.9, 1.0] and len(res.compiled.g.pieces) == 6
    t = np.array([0.92, 0.97])              # player 2 reads nothing younger than 0.9 and nothing before t = 0.9
    assert np.abs(res.evaluate("D2", "w2", t, t - 0.05)).max() == 0.0
    assert np.abs(res.evaluate("D2", "w2", np.array([0.5, 0.8]), np.array([0.3, 0.7]))).max() == 0.0
    assert np.abs(res.evaluate("D2", "w2", t, t - 0.95)).max() > 1e-3


def test_window_not_a_multiple_of_the_lag_solves_with_a_warning():
    """The delayed example at window 1.1 with its delayed row removed (a drift lag only) died on the assertion;
    it solves on the closure (9 panels, 45 pieces) and warns with the counts.  Reference: cells 44 and 88,
    Richardson 0.5502747 (2026-09-05); the game is symmetric."""
    d = _finite("ch1_delayed_finite.yaml", nodes=3, window=1.1); del d["agents"]["player2"]["signals"]["y2"]["delay"]
    with pytest.warns(UserWarning, match="9 panels, 45 pieces"):
        res = ns.solve(ns.Model.from_dict(d)).require_converged()
    assert len(res.compiled.g.pieces) == 45
    assert abs(res.costs["player1"] - 0.5502747) < 1e-3 and abs(res.costs["player1"] - res.costs["player2"]) < 1e-9


def test_lag_off_the_panel_unit_is_rejected_with_the_unit_to_set():
    d = _finite("ch1_delayed_finite.yaml"); d["agents"]["player1"]["loss"].append([0.1, "D1@0.3", "X"])
    with pytest.raises(ValueError, match=r"common divisor of the lags \[0.25, 0.3\] \(0.05 works\)"):
        ns.solve(ns.Model.from_dict(d))
    d = _finite("ch1_delayed_finite.yaml", breakpoints=[0, 0.25, 0.3, 0.5, 0.75, 1.0])   # closed under 0.25, not under 0.3
    with pytest.warns(UserWarning, match="closed under the lag"):
        c = SpectralCompiled(ns.Model.from_dict(d))
    with pytest.raises(ValueError, match="not closed under the lag 0.3"):
        c.map_shift(0.3)
