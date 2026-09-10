"""Regression tests for the issues found in the 2026-09-04 cold review."""
import os, numpy as np, pytest
import noisestate as ns
from noisestate.sweep import sweep
from noisestate.spec import safe_eval
from noisestate.stationary import StationarySolver
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_sweep_over_a_model_object_actually_sweeps():
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    rows = sweep(m, "eps", [0.2, 0.1])
    assert abs(rows[0]["result"].costs["trader1"] - rows[1]["result"].costs["trader1"]) > 1e-3
    with pytest.raises(ValueError, match="not a parameter"):
        sweep(m, "epsilon", [0.2])


def test_ties_require_structural_identity():
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["params"]["r2"] = 5.0; d["ties"] = [["player1", "player2"]]
    with pytest.raises(ValueError, match="not structurally identical"):
        ns.Model.from_dict(d)
    d["params"]["r2"] = d["params"]["r1"]; d["params"]["p2"] = d["params"]["p1"]
    ns.Model.from_dict(d)                                   # symmetric: accepted


def test_parameters_may_be_expressions_in_earlier_parameters():
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["params"]["p2"] = "2*p1"
    m = ns.Model.from_dict(d)
    assert m.params["p2"] == 2 * m.params["p1"]


def test_expression_evaluator_is_sandboxed():
    assert safe_eval("sqrt(p1) + 2*x**2 - min(1, 2)", {"p1": 4.0, "x": 3.0}) == 2 + 18 - 1
    for bad in ("().__class__.__base__.__subclasses__()", "__import__('os')", "p1.real", "[1,2][0]", "lambda: 1", "q"):
        with pytest.raises(ValueError):
            safe_eval(bad, {"p1": 1.0})


def test_failed_solve_is_reported_and_check_raises():
    from noisestate.accel import ConvergenceError
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    res = StationarySolver(m).solve(tol=1e-15, max_newton=0)      # unattainable tolerance
    assert not res.converged and "residual" in res.message and "NOT converged" in res.summary()
    with pytest.raises(ConvergenceError):
        res.require_converged()


def test_defective_state_matrix_is_handled():
    """A double integrator (defective A) must propagate exactly: x2' = x1, x1' = 0 -> x2(a) = a."""
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d.setdefault("numerics", {})["nodes"] = 8
    d["channels"].append("w3")
    d["states"]["X2"] = {"drift": {"X": 1.0}, "noise": {"w3": 1e-6}}
    d["agents"]["player1"]["loss"].append([0.0, "X2", "X2"])
    from noisestate.finite_spectral import SpectralCompiled
    c = SpectralCompiled(ns.Model.from_dict(d))
    E = c.expA(np.array([0.0, 0.5, 2.0]))
    assert np.allclose(E[:, 1, 0], [0.0, 0.5, 2.0]) and np.allclose(E[:, 0, 0], 1.0)


def test_misaligned_delay_is_rejected_by_the_spectral_engine():
    d = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml")); d.setdefault("numerics", {})["breakpoints"] = [0, 0.3, 1.0]
    with pytest.raises(ValueError, match="not a breakpoint"):
        ns.solve(ns.Model.from_dict(d))
