"""CARA beyond the finite engine's zero-mean, dt-only costs: the linear part of the cost (means), stochastic-integral loss terms
(`integrals`), and risk-averse agents on the stationary engine under consistent planning (noisestate/stationary_risk.py).

Pinned against brute-force discrete references independent of the package: extras/leqg_reference.py (MeanGame: affine
strategies with a target, a drift and an initial state; Game with xdw: the integral xdw int X sigma dW0 in the cost) and
extras/stationary_cara_reference.py (a stationary one-agent game whose every date's self minimises the entropic cost of its
own discounted continuation given the later selves' map; the date-0 best response in closed form)."""
import json
import os

import numpy as np
import pytest
import yaml

import noisestate as ns

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "refs")


def richardson(values, xs):
    A = np.array([[1.0 / x ** k for k in range(len(xs))] for x in xs])
    return float(np.linalg.solve(A, np.asarray(values, dtype=float))[0])


def ch1(theta=0.0, nodes=12, means=False, xdw=0.0):
    d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
    d["numerics"]["nodes"] = nodes
    if means:
        d["states"]["X"] = {"d": "(D1 + D2 + 0.5) dt + sigma dw0", "initial": 1.0}
    for a, u, r in (("player1", "D1", "r1"), ("player2", "D2", "r2")):
        if means:
            d["agents"][a]["loss"] = f"X^2 - 0.6 X + {r} {u}^2"
        if xdw:
            d["agents"][a]["integrals"] = [[xdw, "X", "w0"]]
        if theta:
            d["agents"][a]["risk_aversion"] = theta
    return ns.Model.from_dict(d)


# ------------------------------------------------------------------------------------------------ means
@pytest.fixture(scope="module")
def means_ref():
    with open(os.path.join(REFS, "leqg_ch1_means.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_means_match_the_reference(means_ref, theta):
    """A target, a constant drift and an initial state: the mean first-order condition gains theta <f_t, S k> (k the cost's
    linear part) and the entropic cost theta / 2 <k, S k>.  Against the Richardson limit of the discrete reference (n = 50 ..
    300): the entropic and expected costs, the mean paths of D1 and X, and D1's kernel.  Without the linear part's tilt the
    mean paths would be off by 0.04 (theta 0.5) to 0.09 (theta 1) at t = 0.1."""
    lim = means_ref["limit"][f"{theta:g}"]
    res = ns.solve(ch1(theta, means=True), diagnostics=False)
    assert res.converged
    t = np.array(means_ref["levels"][f"{theta:g}"][0]["times"])
    assert abs(res.entropic_costs["player1"] - lim["entropic"]) < 3e-7
    assert abs(res.costs["player1"] - lim["expected"]) < 1e-7
    assert np.abs(res.mean("D1", t) - lim["D1_mean"]).max() < 3e-6
    assert np.abs(res.mean("X", t) - lim["X_mean"]).max() < 3e-6
    assert np.abs(res.evaluate("D1", "w0", t, np.full(t.size, 0.2)) - lim["D1"]).max() < 2e-5
    neutral = means_ref["limit"]["0"]["D1_mean"][0]
    assert abs(lim["D1_mean"][0] - neutral) > 0.04


def test_means_leave_the_kernels_alone():
    """The kernels' condition does not see the linear part (k is deterministic): the maps with means are the maps without."""
    a = ns.solve(ch1(1.0, means=True), diagnostics=False)
    b = ns.solve(ch1(1.0), diagnostics=False)
    for n in a.maps:
        assert np.array_equal(a.maps[n], b.maps[n])
    assert a.risk["player1"]["linear_part"] > 0


# ------------------------------------------------------------------------------------------------ integrals
def test_integrals_at_zero_risk_aversion_change_nothing():
    """A stochastic integral has mean zero: a risk-neutral solve ignores it, to the bit."""
    a = ns.solve(ch1(0.0, xdw=1.0), diagnostics=False)
    b = ns.solve(ch1(0.0), diagnostics=False)
    assert a.costs == b.costs
    for n in a.maps:
        assert np.array_equal(a.maps[n], b.maps[n])


@pytest.fixture(scope="module")
def xdw_ref():
    with open(os.path.join(REFS, "leqg_ch1_xdw.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_integrals_match_the_reference(xdw_ref, theta):
    """int X sigma dW0 in both players' costs (the reference's Ito sum of X_k sigma sqrt(dt) g0_k): the entropic and expected
    costs and D1's response against the Richardson limit of the discrete reference."""
    case = xdw_ref["cases"][f"{theta:g}"]
    lim = case["limit"]
    res = ns.solve(ch1(theta, xdw=xdw_ref["xdw"]), diagnostics=False)
    assert res.converged
    t = np.array([0.3, 0.5, 0.8])
    assert abs(res.entropic_costs["player1"] - lim["entropic"][0]) < 1e-6
    assert abs(res.costs["player1"] - lim["expected"][0]) < 1e-6
    assert np.abs(res.evaluate("D1", "w0", t, np.full(3, 0.2)) - lim["D1"]).max() < 5e-5
    plain = ns.solve(ch1(theta), diagnostics=False)
    assert abs(plain.entropic_costs["player1"] - lim["entropic"][0]) > 1e-3


def test_the_wealth_identity_path_by_path():
    """Kyle-Back with a random-walk value, rho = 0: -D V + D P + eps D^2 with the integral -sigma_V Q dwV is minus the wealth
    liquidated at V_T (int D P dt - Q_T V_T) path by path, so the two forms have the same entropic equilibrium (up to the
    discretisation, which the random walk's start makes algebraic: 4e-5 on J at 8 nodes)."""
    def kyle(theta, form):
        tr = {"controls": "D1", "observes": {"v": "sigma_V dwV", "flow": "sigma_Z dwZ"}, "risk_aversion": theta}
        if form == "wealth":
            tr.update(loss="D1 P + eps D1^2", terminal="-Q V")
        else:
            tr.update(loss="-D1 V + D1 P + eps D1^2", integrals=[["-sigma_V", "Q", "wV"]])
        d = {"name": "kb", "params": {"eps": 0.2, "sigma_V": 1.0, "sigma_Z": 1.0}, "shocks": ["wV", "wZ"],
             "states": {"V": "sigma_V dwV", "Q": "D1 dt"},
             "agents": {"market_maker": {"controls": "P", "observes": {"flow": "D1 dt + sigma_Z dwZ"}, "loss": "P^2 - 2 P V",
                                         "myopic": True}, "trader1": tr},
             "horizon": {"T": 1.0}, "numerics": {"nodes": 8}}
        return ns.solve(ns.Model.from_dict(d), diagnostics=False)
    a, b = kyle(1.0, "wealth"), kyle(1.0, "dw")
    assert a.converged and b.converged
    assert abs(a.entropic_costs["trader1"] - b.entropic_costs["trader1"]) < 1e-4
    assert abs(a.entropic_costs["trader1"] - a.costs["trader1"]) > 0.1
    t = np.array([0.3, 0.5, 0.8])
    assert np.abs(a.evaluate("D1", "wV", t, np.full(3, 0.05)) - b.evaluate("D1", "wV", t, np.full(3, 0.05))).max() < 3e-3


def test_integral_grammar_is_checked():
    d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
    d["agents"]["player1"]["integrals"] = [[1.0, "X", "wQ"]]
    with pytest.raises(ValueError, match="not a shock"):
        ns.Model.from_dict(d)
    m = ch1(0.5, means=True).to_dict()
    m["agents"]["player1"]["integrals"] = [[1.0, "X", "w0"]]
    with pytest.raises(NotImplementedError, match="without means"):
        ns.solve(ns.Model.from_dict(m))


# ------------------------------------------------------------------------------------------------ stationary, consistent planning
def lq(theta=0.0, obs="noise", nodes=16):  # noqa: D401
    """One agent, dX = D dt + dW0, loss X^2 + 0.1 D^2, rho 0.5, window 8; it sees the state's shocks (noise) or a signal
    sqrt(3) X dt + dW1 (then a deviation moves what its later selves see, and they react)."""
    ag = {"controls": "D1", "observes": "sigma dw0" if obs == "noise" else "sqrt(p) X dt + dw1", "loss": "X^2 + r D1^2"}
    if theta:
        ag["risk_aversion"] = theta
    d = {"name": "lqs", "params": {"sigma": 1.0, "r": 0.1, **({"p": 3.0} if obs != "noise" else {})},
         "shocks": ["w0"] + ([] if obs == "noise" else ["w1"]), "states": {"X": "D1 dt + sigma dw0"}, "agents": {"p1": ag},
         "horizon": {"window": 8.0, "discount": 0.5}, "numerics": {"nodes": nodes}}
    return ns.Model.from_dict(d)


@pytest.fixture(scope="module")
def stat_ref():
    with open(os.path.join(REFS, "stationary_cara_lq.json")) as f:
        return json.load(f)


def test_stationary_zero_risk_aversion_is_the_risk_neutral_solve():
    """An explicit risk_aversion 0 on the stationary engine takes the risk-neutral path, to the bit."""
    a = ns.solve(lq(0.0), diagnostics=False)
    d = lq(0.0).to_dict()
    d["agents"]["p1"]["risk_aversion"] = 0.0
    b = ns.solve(ns.Model.from_dict(d), diagnostics=False)
    assert a.costs == b.costs and np.array_equal(a.maps["p1"], b.maps["p1"]) and b.risk == {}


@pytest.mark.parametrize("obs,nodes,thetas", [("noise", 16, (0.5, 1.0)), ("signal", 24, (0.25, 0.5))])
def test_stationary_consistent_planning_matches_the_reference(stat_ref, obs, nodes, thetas):
    """The equilibrium under consistent planning against the discrete reference's Richardson limit (h = 0.1 .. 0.0125 /
    0.025, stable to 1e-5 across subsets of the levels): the action's kernel on the state's shock at ages 0 .. 2, the flow
    cost, and the date-0 self's conditional entropic cost averaged over the past.  With the signal the later selves react
    to a deviation (f_0 with the own responses on); without the reaction terms the kernel would be off by about the
    risk-averse effect itself.  16 nodes on the signal model leave 2e-3 at age 2 (at theta 0 too): 24 nodes."""
    ref = stat_ref[obs]["limit"]
    ages = np.array(stat_ref[obs]["ages"])
    base = ns.solve(lq(0.0, obs, nodes), diagnostics=False)
    k0 = np.asarray(base.kernel("D1", "w0").at(ages)).ravel()
    assert np.abs(k0 - np.array(ref["0"]["D_w0"])).max() < 1e-4
    for th in thetas:
        res = ns.solve(lq(th, obs, nodes), diagnostics=False)
        assert res.converged
        k = np.asarray(res.kernel("D1", "w0").at(ages)).ravel()
        lim = np.array(ref[f"{th:g}"]["D_w0"])
        # the reference's own error: 3e-5 where it has the level h = 0.0125, 1.6e-4 without it (noise, theta 1)
        assert np.abs(k - lim).max() < (2e-4 if len(stat_ref[obs]["levels"][f"{th:g}"]) < 4 and obs == "noise" else 1e-4), (k, lim)
        assert np.abs(k - k0).max() > 0.03
        assert abs(res.costs["p1"] - ref[f"{th:g}"]["flow"]) < 3e-5
        rk = res.risk["p1"]
        excess = rk["entropic"] - rk["expected"]
        # a report, not the equilibrium: the lattice's excess converges slowly (the kernels' kinks on the diagonal), 1e-4
        # relative on the signal model, 1e-2 on the noise one at the default lattices (40, 80, 160)
        assert abs(excess - ref[f"{th:g}"]["cond_excess"]) < 1.2e-2 * ref[f"{th:g}"]["cond_excess"], (excess, ref[f"{th:g}"]["cond_excess"])
        assert rk["theta_mu_max"] < 1 and rk["richardson_gap"] < 1e-3


def kyle_stationary(theta=0.0, nodes=16, wealth=True):
    d = ns.load(ns.example("ch4_kyle_back")).to_dict()
    d["numerics"]["nodes"] = nodes
    if wealth:
        d["states"]["Q1"] = {"drift": {"D1": 1.0}, "noise": {}}
        d["agents"]["trader1"]["integrals"] = [["-sigma_V", "Q1", "wV"]]
    if theta:
        d["agents"]["trader1"]["risk_aversion"] = theta
    return ns.Model.from_dict(d)


def test_stationary_kyle_operators():
    """Chapter 4's insider on wealth, at the risk-neutral equilibrium: (i) the envelope identity, P (phi^on - phi^off) = 0,
    the FOC kernel with the own later reactions on and off agree on the rows (a closed loop that dropped the market maker's
    reaction to the spike broke it by 2 before the responses were rebuilt by the reaction fixed point); (ii) the lattice's
    K_0 and f_0 against direct quadrature for a smooth shock path g: <g, K_0 g> = 2 C_0(g) and <f_0, g> = the derivative of
    C_0 under the spike, both to h^2."""
    from noisestate.stationary import StationarySolver
    from noisestate.stationary_risk import StationaryTilt
    r = ns.solve(kyle_stationary(0.0), diagnostics=False)
    m = kyle_stationary(1.0)
    sol = StationarySolver(m)
    a = [x for x in m.agents if x.name == "trader1"][0]
    tl = StationaryTilt(sol, a, r.maps, 1.0)
    c = sol.c
    Zpass = sol._spikes(c, r.maps, a)[0]
    H = sol._projection_operator(a, *sol._passive_rows(a, Zpass))
    hp = lambda phi: H @ phi.T.reshape(-1)
    assert np.abs(hp(tl.phi_off[0])).max() < 1e-6                                  # the equilibrium's FOC residual
    assert np.abs(hp(tl.phi_on[0] - tl.phi_off[0])).max() < 1e-6 < 1e-2 < np.abs(tl.phi_on[0] - tl.phi_off[0]).max()
    assert abs(float((c.grid.interp([0.5]) @ tl.Ron[c.block("P"), 0])[0]) - 0.918) < 5e-3   # the market maker reacts
    # direct quadrature
    gr, L, rho, nW = c.grid, tl.L, tl.rho, tl.nW
    def gfun(u):
        u = np.asarray(u, float)
        return np.stack([np.exp(-((u - 1.0 - k) ** 2)) * (1 + 0.3 * k) for k in range(nW)])
    a_ = np.linspace(0, L, 2001); wa = np.full(a_.size, a_[1]); wa[[0, -1]] *= 0.5
    Ia = gr.interp(a_); Ia[-1] = gr.interp([L], side=-1)[0]
    za = np.einsum("an,jnc->ajc", Ia, tl.z); zxa = np.einsum("an,jnc->ajc", Ia, tl.zx)
    zeta = lambda t, zz: np.einsum("a,ajc,ca->j", wa, zz, gfun(t - a_))
    tau = np.linspace(0, L + 40.0, 3001); wt = np.full(tau.size, tau[1]); wt[[0, -1]] *= 0.5
    Z = np.array([zeta(t, za) for t in tau]); Zx = np.array([zeta(t, zxa) for t in tau])
    C0 = 0.5 * np.sum(wt * np.exp(-rho * tau) * np.einsum("tj,ji,ti->t", Z, tl.Q, Z)) \
        + np.sum(wt * np.exp(-rho * tau) * np.einsum("tj,jc,ct->t", Zx, tl.Lx, gfun(tau)))
    Rt = Ia @ tl.Rj[0].T; Rx = Ia @ tl.Rx[0].T
    Za = np.array([zeta(t, za) for t in a_])
    dC = np.sum(wa * np.exp(-rho * a_) * (np.einsum("tj,ji,ti->t", Rt, tl.Q, Za) + np.einsum("tj,jc,ct->t", Rx, tl.Lx, gfun(a_))))
    dC += (tl.Q @ zeta(0.0, za))[tl.atoms.index(("D1", 0.0))]
    errs = []
    for h in tl.hs:
        lat = tl._lattice(h)
        u = -L + h * np.arange(lat["Nu"]); G = gfun(u)
        wu = np.full(u.size, h); wu[0] *= 0.5
        errs.append((abs(0.5 * np.sum(wu * np.sum(G * tl._K(lat, G), axis=0)) - C0),
                     abs(np.sum(wu * np.sum(tl._f(lat, 0) * G, axis=0)) - dC)))
    # second order in the lattice step (0.02 -> 0.01: 2.1e-5 -> 4.3e-6 on C_0, 4.4e-5 -> 9.6e-6 on the derivative)
    assert errs[1][0] < errs[0][0] / 3 and errs[1][1] < errs[0][1] / 3, errs
    assert errs[1][0] < 2e-4 * abs(C0) and errs[1][1] < 2e-4 * abs(dC), (errs, C0, dC)


def test_discounted_wealth_integral_matches_the_kyle_reference():
    """Kyle-Back with a random-walk value on a finite horizon, discounted (rho 0.5): the insider's wealth written as the
    fundamental-valued flow plus the integral -sigma_V int e^{-rho t} Q dW_V (which no terminal form can write with a
    discount), against extras/kyle_reference.py's cost "dw" (Richardson over n = 40 .. 120,
    tests/refs/leqg_kyle_dw_rho.json).  The engine converges like N^-2 here (the random walk's start: the risk-neutral
    solve too), so its 20- and 24-node values are extrapolated in 1/N^2: the risk-averse effect J(theta) - J(0) agrees to
    1e-5, the costs to 1e-5."""
    with open(os.path.join(REFS, "leqg_kyle_dw_rho.json")) as f:
        lim = json.load(f)["limit"]

    def kyle(theta, nodes):
        tr = {"controls": "D1", "observes": {"v": "sigma_V dwV", "flow": "sigma_Z dwZ"}, "loss": "-D1 V + D1 P + eps D1^2",
              "integrals": [["-sigma_V", "Q", "wV"]]}
        if theta:
            tr["risk_aversion"] = theta
        d = {"name": "kb", "params": {"eps": 0.2, "sigma_V": 1.0, "sigma_Z": 1.0}, "shocks": ["wV", "wZ"],
             "states": {"V": "sigma_V dwV", "Q": "D1 dt"},
             "agents": {"market_maker": {"controls": "P", "observes": {"flow": "D1 dt + sigma_Z dwZ"}, "loss": "P^2 - 2 P V",
                                         "myopic": True}, "trader1": tr},
             "horizon": {"T": 1.0, "discount": 0.5}, "numerics": {"nodes": nodes}}
        return ns.Model.from_dict(d)
    J = {}
    for th in (0.0, 1.0, 2.0):
        v = [ns.solve(kyle(th, n), diagnostics=False).entropic_costs["trader1"] for n in (20, 24)]
        J[th] = v[1] + (v[1] - v[0]) * 20 ** 2 / (24 ** 2 - 20 ** 2)
        assert abs(J[th] - lim[f"{th:g}"]["entropic"]) < 1e-5, (th, J[th], lim[f"{th:g}"]["entropic"])
    for th in (1.0, 2.0):
        assert abs((J[th] - J[0.0]) - (lim[f"{th:g}"]["entropic"] - lim["0"]["entropic"])) < 1e-5
