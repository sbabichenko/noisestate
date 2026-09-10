"""The stationary engine against the exact solution of a one-agent problem with a delayed observation.

Exact solution, independent of noisestate:

  dX = (a X + D) dt + dw0,  dY = h X dt + dw1 observed with delay delta,  loss X^2 + r D^2 (average cost).
Certainty equivalence with a classical (nested) information pattern: D_t = -K xhat_t with xhat_t = E[X_t | Y_s, s <= t - delta],
K = S / r from the control Riccati equation 2 a S - S^2 / r + 1 = 0, and xhat_t = e^{a delta} m_{t-delta} + int_0^delta e^{a s} D_{t-s} ds
where m is the undelayed stationary Kalman estimate, dm = (a - P h^2) m dt + D dt + P h dY, 2 a P - h^2 P^2 + 1 = 0.
The kernels in shock age are the impulse responses of the closed loop, a delay-differential system integrated by the
method of steps (DOP853, rtol 1e-12), with z(t) = int_0^delta e^{a s} D(t-s) ds as a state.  The exact cost is the
integral of the flow loss over the kernels, carried as extra states."""
import numpy as np
from scipy.integrate import solve_ivp
from helpers import example_path, example_dict, slow

a, h, r, delta = -0.3, 1.5, 0.5, 0.5
S = (2 * a + np.sqrt(4 * a * a + 4 / r)) / (2 / r)
assert abs(2 * a * S - S * S / r + 1) < 1e-12
K = S / r
P = (a + np.sqrt(a * a + h * h)) / (h * h)
assert abs(2 * a * P - h * h * P * P + 1) < 1e-12
Ph = P * h
E = np.exp(a * delta)


def impulse(channel, A_max=40.0):
    """Kernels (X, D) in age for a unit impulse in the channel at age 0: dense solutions per step [k delta, (k+1) delta]."""
    # state: X, m, z, JX = int X^2, JD = int D^2
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) if channel == 0 else np.array([0.0, Ph, 0.0, 0.0, 0.0])
    steps = []                                             # dense solution of step k on [k delta, (k+1) delta]

    def state(s, k):                                       # the state at time s taken from step k (zero before the impulse)
        if k < 0:
            return np.zeros(5)
        return steps[k](min(max(s, k * delta), (k + 1) * delta))

    def D_at(s, k):
        if k < 0:
            return 0.0
        return -K * (E * state(s - delta, k - 1)[1] + state(s, k)[2])

    def rhs_for(k):
        def rhs(t, y):
            X, m, z, JX, JD = y
            m_d = state(t - delta, k - 1)[1]; D_d = D_at(t - delta, k - 1)
            D = -K * (E * m_d + z)
            return [a * X + D, (a - Ph * h) * m + D + Ph * h * X, D - E * D_d + a * z, X * X, D * D]
        return rhs

    y = y0.copy(); k = 0
    while k * delta < A_max - 1e-12:
        sol = solve_ivp(rhs_for(k), (k * delta, (k + 1) * delta), y, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        assert sol.success
        steps.append(sol.sol); y = sol.y[:, -1]; k += 1
    JX, JD = y[3], y[4]

    def kernel(ages, sides):
        """X and D at the ages; at a step boundary the side picks the limit (-1: from the step below)."""
        Xv = np.zeros(len(ages)); Dv = np.zeros(len(ages))
        for i, (t, sd) in enumerate(zip(ages, sides)):
            k = int(np.floor(t / delta + 1e-9))
            if abs(t - k * delta) < 1e-9 and sd < 0 and k > 0:
                k -= 1
            Xv[i] = state(t, k)[0]; Dv[i] = D_at(t, k)
        return Xv, Dv

    return kernel, JX, JD




def test_stationary_engine_matches_the_exact_delayed_solution():
    import noisestate as ns
    k0, JX0, JD0 = impulse(0); k1, JX1, JD1 = impulse(1)
    exact_cost = (JX0 + JX1) + r * (JD0 + JD1)
    model = {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
             "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}, "delay": delta}},
                              "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
             "horizon": {"kind": "stationary", "window": 10.0}, "numerics": {"nodes": 16}}
    res = ns.solve(model).require_converged()
    assert abs(res.costs["a"] - exact_cost) < 1e-8
    assert res.representation_error["a"] < 1e-10 and res.second_order["a"]["ok"]
    g = res.compiled.grid; sides = np.where((np.arange(g.N) % g.n) == g.n - 1, -1, 1)     # read the exact kernel from the node's side
    X0, D0 = k0(res.ages, sides); X1, D1 = k1(res.ages, sides)
    for name, ch, exact in (("X", "w0", X0), ("X", "w1", X1), ("D", "w0", D0), ("D", "w1", D1)):
        assert np.abs(res.kernel(name, ch) - exact).max() < 2e-5, (name, ch)
    assert np.abs(res.kernel("D", "w1")[res.ages < delta - 1e-12]).max() == 0.0             # exactly causal


def test_finite_engine_on_the_delayed_problem_converges_spectrally():
    """The finite engine at T = 3 on the same one-agent delayed model: the representation error falls
    spectrally with the nodes per side and the cost settles (there is no closed form for the finite
    horizon's discounted integral, so convergence is the check)."""
    import noisestate as ns
    model = {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
             "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}, "delay": delta}},
                              "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
             "horizon": {"kind": "finite", "window": 3.0}, "numerics": {"nodes": 5}}
    r5 = ns.solve(model).require_converged(); model["numerics"]["nodes"] = 6; r6 = ns.solve(model).require_converged()
    assert r5.representation_error["a"] < 1e-5 and r6.representation_error["a"] < 1e-6
    assert abs(r5.costs["a"] - r6.costs["a"]) < 2e-6 and r6.second_order["a"]["ok"]
    assert np.abs(r6.kernel("D", "w1")[r6.grid.a < delta - 1e-12]).max() == 0.0


def test_delayed_ch1_example_both_players_representable():
    import noisestate as ns
    r = ns.solve(example_path("ch1_delayed_finite")).require_converged()
    assert all(v < 1e-8 for v in r.representation_error.values()), r.representation_error


@slow("slow (9 s; the undelayed multi-panel grid is tests/test_unit_range_finite.py's at 5 nodes); set NOISESTATE_SLOW=1")
def test_lagged_undelayed_finite_model_is_resolved_at_six_nodes():
    """The delayed Chapter 1 example with its delay removed keeps its control lags: a multi-panel
    finite grid without delayed rows.  The top-edge read defect moved these too; pinned against a
    finer solve."""
    import noisestate as ns
    d = example_dict("ch1_delayed_finite")
    for a in d["agents"].values():
        for row in a["signals"].values():
            row["delay"] = 0.0
    d.setdefault("numerics", {})["nodes"] = 6; r6 = ns.solve(d).require_converged(); d.setdefault("numerics", {})["nodes"] = 10; r10 = ns.solve(d).require_converged()
    assert max(r6.representation_error.values()) < 1e-6
    assert abs(r6.costs["player1"] - r10.costs["player1"]) < 1e-6
