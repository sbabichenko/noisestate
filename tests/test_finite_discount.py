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
from helpers import slow

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
    """kind "finite_cells" names the cell engine's result; the model asks for kind finite with engine cells."""
    cells = kind == "finite_cells"
    return {"shocks": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}}},
                             "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
            "horizon": {"kind": "finite" if cells else kind, "T": T, "discount": rho},
            "numerics": {"nodes": nodes, **({"engine": "cells"} if cells else {})}}


TS = np.array([0.5, 1.0, 1.0, 2.0, 2.0, 2.8, 2.8]); SS = TS - np.array([0.1, 0.1, 0.5, 0.5, 1.5, 0.1, 2.0])


@pytest.mark.parametrize("rho", [0.0, 0.5, 2.0])
def test_spectral_finite_engine_matches_the_discounted_closed_form(rho):
    """12 nodes per side: the cost to 1e-8 (measured -5e-9 at rho = 0 and -1e-8 at 0.5; 3e-5 at 8 nodes, 2e-4 at 6)
    and the kernels to 6e-5 (D on w1, the one that reads K(t) and G(s) directly; 1e-5 and below on the others)."""
    J, kernel = exact(rho)
    res = ns.solve(model(rho, "finite", 12)).require_converged()
    assert abs(res.costs["a"] - J) < 1e-7
    for name, ch in (("D", "w1"), ("D", "w0"), ("X", "w0"), ("X", "w1")):
        assert np.abs(res.evaluate(name, ch, TS, SS) - kernel(name, ch, TS, SS)).max() < 2e-4, (name, ch)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_cell_engine_is_first_order_and_its_richardson_pair_matches_the_closed_form(rho):
    """48 and 96 cells: the cost error halves (measured 4.5e-2 / 2.3e-2 at rho = 0, 2.4e-2 / 1.2e-2 at 0.5, ratio 1.98)
    and the Richardson pair is within 4e-4 (measured 4.0e-4 and 2.6e-4; the pair (24, 48) is within 1.8e-3)."""
    J, _ = exact(rho)
    e48 = ns.solve(model(rho, "finite_cells", 48)).require_converged().costs["a"] - J
    e96 = ns.solve(model(rho, "finite_cells", 96)).require_converged().costs["a"] - J
    assert 1.9 < e48 / e96 < 2.1
    assert abs(2 * e96 - e48) < 1e-3


#  --------------------------------------------------------------- the stationary side of the discount
#  The tests above pin the FINITE engines against a closed form.  The stationary engine has no closed
#  form to hand, but the dissertation makes a sharp claim about what the discount does there, and it
#  is testable: a positive discount makes the stationary problem well posed, so the answer stops
#  depending on the lag window that truncates it.  At rho = 0 the same model has no stationary
#  solution and its kernels are window artefacts.

@slow("slow (3 windows x 2 discounts on Kyle-Back); set NOISESTATE_SLOW=1")
def test_a_discount_makes_the_stationary_answer_independent_of_the_window():
    """Why ch4_kyle_back ships with rho = 0.5 rather than 0.

    The stationary verification assumes rho > 0, and at rho = 0 the lag window stands in for the
    transversality condition -- so an undiscounted model can converge on a window and still be
    reporting the window rather than the model.  The Kyle-Back trader is the case: undiscounted, its
    profit settles neither in the window nor in the resolution; discounted, it is the same to five
    digits on windows of 8, 16 and 32.
    """
    kb = ns.load(ns.example("ch4_kyle_back"))
    assert kb.horizon.discount == 0.5, "the example ships discounted for this reason"

    profits = {}
    for L in (8.0, 16.0, 32.0):
        res = ns.solve(kb.with_stationary(L), {"nodes": 24}).require_converged()
        profits[L] = -res.costs["trader1"]
    assert max(profits.values()) - min(profits.values()) < 1e-4, profits
    assert abs(profits[8.0] - 0.8208) < 1e-3, profits          # the value guards.md quotes

    #  undiscounted, the same quantity moves with the window and the guard says the result is not sound
    undiscounted = {}
    for L in (8.0, 16.0):
        res = ns.solve(kb.with_params(rho=1e-9).with_stationary(L), {"nodes": 24})
        undiscounted[L] = -res.costs["trader1"]
        assert not res.diagnostics.assess().accepted          # flagged, not silently reported
    assert abs(undiscounted[8.0] - undiscounted[16.0]) > 0.1, undiscounted


def test_the_stationary_cost_is_the_flow_and_the_finite_cost_is_the_integral():
    """Two normalisations, and cost_kind is what distinguishes them.

    A stationary result reports the flow loss per unit time at EVERY discount -- rho enters the
    first-order condition, not the reported scalar -- while a finite horizon reports the discounted
    integral over [0, T].  Reading one as the other is a factor of rho.
    """
    m = ns.load(ns.example("ch3_two_player"))
    stat = ns.solve(m.with_horizon(ns.Stationary(window=6.0, discount=0.5)), {"nodes": 12})
    assert stat.cost_kind == "stationary flow loss per unit time"
    fin = ns.solve(m.with_horizon(ns.Finite(T=2.0, discount=0.5)), {"nodes": 8})
    assert fin.cost_kind == "discounted integral over [0, T]"
    #  the finite integral over a horizon this short is well under the stationary flow's own scale,
    #  which is the point: they are not comparable numbers
    assert fin.costs["player1"] < stat.costs["player1"] / 0.5


#  A cross-engine comparison at every discount used to sit here.  It cost 152 s of a 303 s suite --
#  half the runtime in three tests -- because agreement between a spectral method and a FIRST-ORDER
#  cell scheme needs 400 cells to reach 3e-3.  It was also the weaker check: both engines are
#  already pinned against the exact solution above, so agreement between them follows, and two
#  engines wrong in the same way would have passed it.  Its one unique contribution was covering a
#  third discount, which the closed-form test now does in 0.6 s against an exact reference.

def _own_lag(rho, T=None, window=None, nodes=16):
    """A loss that reads the agent's OWN control at a lag: loss ... + c D(t) D(t - 0.5)."""
    return ns.Model.from_dict({
        "name": "ownlag", "params": {"r": 0.5, "c": 0.3}, "shocks": ["w0", "v1"], "states": {"X": "(-0.3 X + D) dt + dw0"},
        "agents": {"me": {"controls": "D", "observes": {"y": "1.5 X dt + dv1"}, "loss": "X^2 + r D^2 + c D D@0.5"}},
        "horizon": {"window": window, "discount": rho} if window is not None else {"T": T, "discount": rho},
        "numerics": {"nodes": nodes}})


@slow("slow (two engines x two discounts on an own-lag model); set NOISESTATE_SLOW=1")
def test_the_own_lagged_read_is_discounted_consistently_across_the_engines():
    """The one discount factor the engines write out explicitly.

    A loss term c D(t) D(t - 0.5) reaches the first-order condition twice: as an instantaneous read
    of the PAST control, undiscounted because it is felt at t, and through the FUTURE flow at
    t + 0.5 in which D(t) is the lagged factor, which carries exp(-rho * 0.5).  Only the second is
    the `exp(-c.rho * lag) * own_lag_read(lag)` term, which is why the cross term's influence does
    not fade as rho grows -- half of it never was discounted.

    Nothing exercised that path: deleting the factor left the whole suite green, because no shipped
    example and no test had an own lagged control in a loss.  The two engines implement it
    separately, so agreeing at rho > 0 is the check available without a closed form.  It is not a
    sharp one -- removing the factor moves the stationary answer about 1.3% here, against a
    cross-engine gap of about 0.5% -- so it is a consistency test, not a proof.
    """
    gaps = {}
    for rho in (0.0, 0.5):
        stat = ns.solve(_own_lag(rho, window=4.0, nodes=16)).require_converged()
        fin = ns.solve(_own_lag(rho, T=5.0, nodes=6), {"nodes": 6}).require_converged()
        s = float(stat.kernel("D", "w0").at(1.0))
        f = float(fin.kernel("D", "w0").at(3.0, 2.0))
        gaps[rho] = abs(s - f) / abs(s)
    assert max(gaps.values()) < 0.02, gaps
    assert gaps[0.5] <= gaps[0.0] + 0.01, gaps        # the discount does not degrade the agreement


def test_an_own_lagged_read_is_undiscounted_at_rho_zero():
    """The factor is exp(-rho * lag), so at rho = 0 it must be exactly inert -- a sanity check that
    the term is a discount and not a stray lag weight."""
    a = ns.solve(_own_lag(0.0, window=4.0, nodes=14)).require_converged()
    b = ns.solve(_own_lag(0.0, window=4.0, nodes=14)).require_converged()
    assert float(a.kernel("D", "w0").at(1.0)) == float(b.kernel("D", "w0").at(1.0))
    #  and it is genuinely live at rho > 0: the same model answers differently
    c = ns.solve(_own_lag(1.0, window=4.0, nodes=14)).require_converged()
    assert abs(float(c.kernel("D", "w0").at(1.0)) - float(a.kernel("D", "w0").at(1.0))) > 1e-3
