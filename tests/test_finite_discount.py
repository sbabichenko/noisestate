"""Both finite engines against the closed form of a discounted one-agent finite-horizon problem.

Exact solution, independent of noisestate:

  dX = (a X + D) dt + dw0,  dY = h X dt + dw1,  loss X^2 + r D^2,  cost E int_0^T e^{-rho t} loss dt,  X(0) = 0.
Certainty equivalence: D_t = -K(t) xhat_t with K = S / r from the discounted backward Riccati equation
S' = rho S - 2 a S + S^2 / r - 1, S(T) = 0, and xhat the Kalman estimate, d xhat = (a xhat + D) dt + G (dY - h xhat dt),
G = P h, P' = 2 a P - h^2 P^2 + 1, P(0) = 0.  With Sigma = E xhat^2, Sigma' = 2 (a - K) Sigma + G^2, Sigma(0) = 0,
the cost is int_0^T e^{-rho t} [(1 + r K^2) Sigma + P] dt (E X^2 = Sigma + P).  The kernels are the impulse responses
of the (X, xhat) closed loop: a unit shock of w0 at s sets X = 1, one of w1 sets xhat = G(s), and then
X' = a X - K xhat, xhat' = G h X + (a - K - G h) xhat up to t, with D(t, s) = -K(t) xhat(t).  Everything is
integrated by DOP853 at rtol 1e-12.  A sign flip in the engines' discount (the continuation weights or the cost
Gram) would move the cost by a factor of order e^{rho T} = 4.5 and the kernels by 0.1; rho = 0 runs as the control."""
import numpy as np, pytest
from scipy.integrate import solve_ivp
import noisestate as ns

a, h, r, T = -0.3, 1.5, 0.5, 3.0
IVP = dict(method="DOP853", rtol=1e-12, atol=1e-14)


def exact(rho):
    """The discounted cost and a kernel(name, channel, t, s) function."""
    Sb = solve_ivp(lambda t, y: [rho * y[0] - 2 * a * y[0] + y[0] ** 2 / r - 1], (T, 0.0), [0.0], dense_output=True, **IVP)
    Pf = solve_ivp(lambda t, y: [2 * a * y[0] - h * h * y[0] ** 2 + 1], (0.0, T), [0.0], dense_output=True, **IVP)
    assert Sb.success and Pf.success
    K = lambda t: float(Sb.sol(t)[0]) / r; P = lambda t: float(Pf.sol(t)[0]); G = lambda t: P(t) * h
    f = solve_ivp(lambda t, y: [2 * (a - K(t)) * y[0] + G(t) ** 2, np.exp(-rho * t) * ((1 + r * K(t) ** 2) * y[0] + P(t))],
                  (0.0, T), [0.0, 0.0], **IVP)
    assert f.success

    def kernel(name, channel, t, s):
        """Response of X or D at the times t to unit shocks of the channel at the times s (s < t)."""
        out = np.zeros(len(t))
        for si in np.unique(s):
            y0 = [1.0, 0.0] if channel == "w0" else [0.0, G(si)]
            sol = solve_ivp(lambda u, y: [a * y[0] - K(u) * y[1], G(u) * h * y[0] + (a - K(u) - G(u) * h) * y[1]],
                            (si, T), y0, dense_output=True, **IVP)
            assert sol.success
            for i in np.flatnonzero(s == si):
                X, xhat = sol.sol(t[i]); out[i] = X if name == "X" else -K(t[i]) * xhat
        return out

    return float(f.y[1, -1]), kernel


def model(rho, kind, nodes):
    return {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}}},
                             "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
            "horizon": {"kind": kind, "window": T, "nodes": nodes, "discount": rho}}


TS = np.array([0.5, 1.0, 1.0, 2.0, 2.0, 2.8, 2.8]); SS = TS - np.array([0.1, 0.1, 0.5, 0.5, 1.5, 0.1, 2.0])


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_spectral_finite_engine_matches_the_discounted_closed_form(rho):
    """12 nodes per side: the cost to 1e-8 (measured -5e-9 at rho = 0 and -1e-8 at 0.5; 3e-5 at 8 nodes, 2e-4 at 6)
    and the kernels to 6e-5 (D on w1, the one that reads K(t) and G(s) directly; 1e-5 and below on the others)."""
    J, kernel = exact(rho)
    res = ns.solve(model(rho, "finite", 12)).check()
    assert abs(res.costs["a"] - J) < 1e-7
    for name, ch in (("D", "w1"), ("D", "w0"), ("X", "w0"), ("X", "w1")):
        assert np.abs(res.evaluate(name, ch, TS, SS) - kernel(name, ch, TS, SS)).max() < 2e-4, (name, ch)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_cell_engine_is_first_order_and_its_richardson_pair_matches_the_closed_form(rho):
    """48 and 96 cells: the cost error halves (measured 4.5e-2 / 2.3e-2 at rho = 0, 2.4e-2 / 1.2e-2 at 0.5, ratio 1.98)
    and the Richardson pair is within 4e-4 (measured 4.0e-4 and 2.6e-4; the pair (24, 48) is within 1.8e-3)."""
    J, _ = exact(rho)
    e48 = ns.solve(model(rho, "finite_cells", 48)).check().costs["a"] - J
    e96 = ns.solve(model(rho, "finite_cells", 96)).check().costs["a"] - J
    assert 1.9 < e48 / e96 < 2.1
    assert abs(2 * e96 - e48) < 1e-3
