"""Vectors in the Python form: State("X", n), Control("D", n), shocks(n), matrices acting by @, quadratic forms
x @ Q @ x.  A vector model is built from its components, so it is checked against a closed form that knows nothing
of the expansion: the single-agent LQG with a matrix state, whose average cost is tr(P) + tr(S G G') with S the
control Riccati solution, P the filter's and G = P H' (loss x'x + r u'u, unit noise intensities)."""
import numpy as np, pytest
from scipy.linalg import solve_continuous_are as care
import noisestate as ns
from noisestate import dt

A = np.array([[-0.5, 0.2], [0.0, -0.3]]); H = np.array([[1.0, 0.0], [0.5, 1.0]]); R = 0.3


def game(**hz):
    dW = ns.shocks(2); dV = ns.shocks("V0 V1")
    X = ns.State("X", 2); D = ns.Control("D", 2)
    X.d = (A @ X + D) * dt + dW
    me = ns.Agent("me", controls=D, observes=H @ X * dt + dV, loss=X @ X + R * (D @ D))
    return ns.Game(X, me, **hz), X, D, me


def test_a_vector_model_is_its_components():
    g, X, D, me = game(window=4.0, nodes=8)
    eq = g.to_equations()
    assert eq["states"] == {"X0": "(-0.5 X0 + 0.2 X1 + D0) dt + dW0", "X1": "(-0.3 X1 + D1) dt + dW1"}
    assert eq["agents"]["me"]["controls"] == ["D0", "D1"]
    assert eq["agents"]["me"]["observes"] == {"y0": "X0 dt + dV0", "y1": "(0.5 X0 + X1) dt + dV1"}
    assert eq["agents"]["me"]["loss"] == "X0^2 + X1^2 + 0.3 D0^2 + 0.3 D1^2"


def test_the_matrix_lqg_closed_form():
    """Window 10 at 16 nodes: the average cost to 5e-6 relative (measured 9.1e-7; 1.8e-5 at window 6, the
    truncation)."""
    S = care(A, np.eye(2), np.eye(2), R * np.eye(2)); P = care(A.T, H.T, np.eye(2), np.eye(2)); G = P @ H.T
    J = np.trace(P) + np.trace(S @ G @ G.T)
    res = game(window=10.0, nodes=16)[0].solve().require_converged()
    assert abs(res.costs["me"] - J) / J < 5e-6


def test_vector_responses_and_estimates_on_both_engines():
    for hz in ({"window": 6.0, "nodes": 10}, {"T": 2.0, "nodes": 10}):
        g, X, D, me = game(**hz)
        res = g.solve().require_converged()
        t = np.array([0.5, 1.0])
        v = res.response(X, to="W0").over(t)
        assert v.shape == (2, 2)
        assert np.allclose(v[:, 0], res.response(X[0], to="W0").over(t)) and np.allclose(v[:, 1], res.response("X1", to="W0").over(t))
        e = res.response(X, to="W0", seen_by=me).over(t)          # an agent with two controls (the estimate was broken)
        assert e.shape == (2, 2) and np.all(np.isfinite(e))


def test_vector_algebra_and_its_errors():
    X = ns.State("X", 3); y = ns.Control("u", 2)
    assert len(X) == 3 and X[1].name == "X1" and len(X[1:]) == 2
    with pytest.raises(ValueError, match="do not combine"):
        X + y
    with pytest.raises(TypeError, match="dot product"):
        X * X
    with pytest.raises(ValueError, match="columns"):
        np.eye(2) @ X
    with pytest.raises(ValueError, match="vector of 3"):
        X.d = y * dt
    Y = ns.define("Y", np.ones((2, 3)) @ X)
    assert [d.name for d in Y] == ["Y0", "Y1"]


def test_a_saved_vector_model_loads_as_the_same_model(tmp_path):
    g = game(window=4.0, nodes=8)[0]
    path = str(tmp_path / "v.yaml"); g.save(path)
    assert ns.load(path).to_dict(numeric=True) == g.to_dict(numeric=True)
