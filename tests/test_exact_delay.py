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
    model = {"shocks": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
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
    model = {"shocks": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
             "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}, "delay": delta}},
                              "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
             "horizon": {"kind": "finite", "T": 3.0}, "numerics": {"nodes": 5}}
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


#  --------------------------------------------------------- the delay under a discount
#  The closed form above is at average cost.  What the discount adds is two factors in the
#  dissertation's first variation: e^{-rho tau} on the agent's own delayed read of its own control,
#  and e^{-rho Delta} on the continuation that reaches it through ANOTHER agent's delayed row.  The
#  first is written out (engine.py, "delayed read of the control itself"); the second is not, and
#  these tests are why it does not need to be.

def _cross_delayed(rho, window=3.0, nodes=16, only_delayed=False):
    """p2 watches p1's state through a row observed with a delay of 0.5."""
    import noisestate as ns
    p2_sees = {} if only_delayed else {"y2": "X dt + dv2"}
    p2_sees["flow"] = {"d": "X dt + 0.5 dv3", "delay": 0.5}
    return ns.Model.from_dict({
        "name": "cross", "params": {"r": 0.5},
        "shocks": ["w0", "v1", "v3"] if only_delayed else ["w0", "v1", "v2", "v3"],
        "states": {"X": "(D1 + D2) dt + dw0"},
        "agents": {"p1": {"controls": "D1", "observes": {"y1": "X dt + dv1"}, "loss": "X^2 + r D1^2"},
                   "p2": {"controls": "D2", "observes": p2_sees, "loss": "X^2 + r D2^2"}},
        "horizon": {"window": window, "discount": rho}, "numerics": {"nodes": nodes}})


def test_a_delayed_observer_cannot_react_before_the_delay():
    """Why the continuation through another agent needs no explicit exp(-rho Delta).

    The dissertation's first variation carries e^{-rho Delta_k} on the term that reaches player i
    through player k's DELAYED row.  The engine writes no such factor: its continuation operator is
    the rho-discounted correlation of the impulse response, and the delay is in that response's
    SUPPORT -- so the integral weights the delayed reaction by e^{-rho Delta} and later of its own
    accord.  That argument is only sound if the support really does start at the delay.

    With the delayed row as the agent's ONLY information the reaction is exactly zero before it.
    (With an undelayed row as well it is not, and correctly so: that part arrives through the state
    at once.  An earlier version of this check missed the distinction.)
    """
    import noisestate as ns
    m = _cross_delayed(0.5, only_delayed=True)
    res = ns.solve(m).require_converged()
    c = ns.engines.stationary(m).c
    R = c.closed_loop(res.maps, excluded="p1", impulse_controls=["D1"])[:, c.nW:]
    i = list(c.prim).index("D2")
    reaction = R[i * c.N:(i + 1) * c.N, 0]
    ages = c.grid.nodes
    assert np.abs(reaction[ages < 0.5 - 1e-9]).max() == 0.0        # nothing before the delay
    assert np.abs(reaction[ages > 0.5 + 1e-9]).max() > 1e-3        # and a real reaction after it


@slow("slow (two discounts x two engines on a delayed two-agent model); set NOISESTATE_SLOW=1")
def test_the_discount_does_not_degrade_the_delayed_cross_engine_agreement():
    """A missing or mis-signed discount on a delayed term would show up as the stationary and finite
    engines agreeing at rho = 0 and drifting apart at rho > 0.  They do not: the residual gap is the
    ordinary discretisation difference between the two engines at this resolution, the same size
    either way.
    """
    import noisestate as ns
    gaps = {}
    for rho in (0.0, 0.5):
        stat = ns.solve(_cross_delayed(rho, window=3.0, nodes=14)).require_converged()
        m = _cross_delayed(rho).with_horizon(ns.Finite(T=3.0, discount=rho))
        fin = ns.solve(m, {"nodes": 4}).require_converged()
        s = float(stat.kernel("D1", "w0").at(0.5))
        f = float(fin.kernel("D1", "w0").at(1.5, 1.0))
        gaps[rho] = abs(s - f) / abs(s)
    assert max(gaps.values()) < 0.06, gaps
    assert abs(gaps[0.5] - gaps[0.0]) < 0.02, gaps      # the discount adds no error of its own
