"""A risk-averse (CARA) insider in Chapter 4's Kyle-Back market on a finite horizon: a quadratic cost with cross terms
(D V, D P: G^{DX} != 0, an indefinite loss form) and a past of initial shocks (the value drawn at 0-), solved on the
spectral finite engine (noisestate/risk.py).

The insider's realised cost is minus its wealth: it buys at P and liquidates its position Q at V_T, so the loss is
D P + eps D^2 with the terminal -Q V (dQ = D dt).  With a fixed value that is the chapter's flow -D (V - P) path by
path; with a moving value it differs from it by the mean-zero int Q dV, which a risk-averse trader prices.

Pinned against extras/kyle_reference.py, a brute-force discrete solver independent of the package (the insider's
rows on the shocks, the exact log-det entropic cost, L-BFGS best responses, the market maker's projection), Richardson-
extrapolated over n = 40, 60, 80, 100, 120 (tests/refs/leqg_kyle_prior.json, about twenty seconds to regenerate)."""
import json
import os

import numpy as np
import pytest

import noisestate as ns

REFS = os.path.join(os.path.dirname(__file__), "refs", "leqg_kyle_prior.json")


def kyle(theta=0.0, nodes=8, eps=0.2, prior=True, wealth=True, T=1.0):
    if prior:
        d = ns.load(ns.example("kyle_back_prior")).to_equations()
        d["params"].update(eps=eps, Sigma0=1.0)
        d["horizon"]["T"] = T
    else:                                   # a random-walk value the insider sees move, and the noise trades
        d = {"name": "kyle_back_walk", "params": {"eps": eps, "sigma_V": 1.0, "sigma_Z": 1.0}, "shocks": ["wV", "wZ"],
             "states": {"V": "sigma_V dwV"},
             "agents": {"market_maker": {"controls": "P", "observes": {"flow": "D1 dt + sigma_Z dwZ"}, "loss": "P^2 - 2 P V",
                                         "myopic": True},
                        "trader1": {"controls": "D1", "observes": {"v": "sigma_V dwV", "flow": "sigma_Z dwZ"},
                                    "loss": "-D1 V + D1 P + eps D1^2"}},
             "horizon": {"T": T}, "numerics": {}}
    d["numerics"]["nodes"] = nodes
    tr = d["agents"]["trader1"]
    if wealth:
        d["states"]["Q"] = "D1 dt"
        tr["loss"] = "D1 P + eps D1^2"
        tr["terminal"] = "-Q V"
    if theta:
        tr["risk_aversion"] = theta
    return ns.Model.from_dict(d)


@pytest.fixture(scope="module")
def refs():
    with open(REFS) as f:
        return json.load(f)


def test_zero_risk_aversion_with_a_past_is_the_risk_neutral_solve_bit_for_bit():
    d = kyle(0.0).to_equations()
    d["agents"]["trader1"]["risk_aversion"] = 0.0
    a, b = ns.solve(ns.Model.from_dict(d), diagnostics=False), ns.solve(kyle(0.0), diagnostics=False)
    assert a.costs == b.costs and a.evaluations == b.evaluations
    for n in a.maps:
        assert np.array_equal(a.maps[n], b.maps[n])
    assert a.risk == {}


def test_a_fixed_value_makes_the_wealth_the_flow():
    """With V drawn once, the terminal wealth and the fundamental-valued flow are the same random variable: the
    same equilibrium and the same entropic cost."""
    a, b = ns.solve(kyle(1.0), diagnostics=False), ns.solve(kyle(1.0, wealth=False), diagnostics=False)
    assert abs(a.entropic_costs["trader1"] - b.entropic_costs["trader1"]) < 1e-7      # the terminal read against the flow's
    assert abs(a.costs["trader1"] - b.costs["trader1"]) < 1e-7                          # quadrature at 8 nodes


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, 2.0, 4.0])
def test_matches_the_brute_force_reference(refs, theta):
    """The entropic and expected costs, the insider's order per unit of the value, its reaction and the price's to a
    noise trade at s = 0.05, and the market maker's error variance of V at t = 0.1, 0.3, 0.5, 0.8, against the
    reference's limit (its own spread over the levels' fits is below 1e-6 on the costs)."""
    ref = refs["limit"][f"{theta:g}"]
    res = ns.solve(kyle(theta, nodes=12), diagnostics=False)
    assert res.converged
    assert abs(res.entropic_costs["trader1"] - ref["entropic"]) < 2e-7
    assert abs(res.costs["trader1"] - ref["expected"]) < 2e-7
    t = np.array([0.1, 0.3, 0.5, 0.8]); s = np.full(4, 0.05)
    assert np.abs(res.evaluate("D1", "v0", t, np.zeros(4)) - ref["D_g0"]).max() < 2e-5
    assert np.abs(res.evaluate("D1", "wZ", t, s) - ref["D_wZ"]).max() < 3e-5
    assert np.abs(res.evaluate("P", "wZ", t, s) - ref["P_wZ"]).max() < 1e-5
    if theta:
        assert res.risk["trader1"]["lambda_max"] < 1e-3          # K <= 0: the insider's wealth is never a loss here


def test_risk_aversion_front_loads_the_trading_and_reveals_faster(refs):
    """Holden and Subrahmanyam (1994): a risk-averse insider trades more aggressively early and the price learns the
    value faster.  The order per unit of V falls over time (flat when risk neutral) and the market maker's error
    variance at t = 0.5 falls with theta."""
    lim = refs["limit"]
    flat = np.array(lim["0"]["D_g0"])
    assert np.ptp(flat) < 1e-5
    prev = None
    for th in ("1", "2", "4"):
        d = np.array(lim[th]["D_g0"])
        assert np.all(np.diff(d) < 0) and d[0] > flat[0] > d[-1]
        if prev is not None:
            assert lim[th]["Sigma_mm"][2] < prev
        prev = lim[th]["Sigma_mm"][2]


def _entropic(sol, agent, maps):
    Z = sol.c.closed_loop(maps)
    return sol.risk_report(agent, Z, sol.expected_cost(agent, Z))["entropic"]


def test_entropic_cost_is_stationary_in_the_insiders_strategy():
    """Independently of the first-order operators and of the reference: the entropic cost from the closed loop and K's
    spectrum has zero slope at the solution along three smooth directions of the insider's map (its discrete weights on
    the initial shock included), against slopes above 1e-3 at the risk-neutral equilibrium judged by the same objective."""
    m = kyle(2.0, nodes=12)
    res = ns.solve(m, diagnostics=False)
    sol = res.solver_class(m, **res.solver_kw)
    a = next(x for x in m.agents if x.name == "trader1")
    c = sol.c; nd = res.maps["trader1"].shape[2] - c.N
    t = np.concatenate([c.g.t, c.tm[:nd]]); s = np.concatenate([c.g.s, np.zeros(nd)])

    def slopes(maps, h=1e-4):
        g0 = maps["trader1"]; out = []
        for d in (g0, g0 * (t - 0.5)[None, None, :], g0 * np.cos(3 * s)[None, None, :]):
            up, dn = dict(maps), dict(maps)
            up["trader1"] = g0 + h * d; dn["trader1"] = g0 - h * d
            out.append((_entropic(sol, a, up) - _entropic(sol, a, dn)) / (2 * h))
        return np.abs(np.array(out))
    at = slopes(res.maps).max()
    far = slopes(ns.solve(kyle(0.0, nodes=12), diagnostics=False).maps).min()
    assert far > 1e-3 and at < 1e-6 * far, (at, far)


def test_a_moving_value_makes_the_wealth_riskier_than_the_flow():
    """With a random-walk value the wealth carries the inventory risk int Q dV that the fundamental-valued flow does
    not: the wealth's cost kernel has a positive eigenvalue (a breakdown exists), the flow's has none (K <= 0), and the
    risk-averse insider on its wealth trades less on the same news."""
    w = ns.solve(kyle(2.0, prior=False), diagnostics=False)
    f = ns.solve(kyle(2.0, prior=False, wealth=False), diagnostics=False)
    assert w.risk["trader1"]["lambda_max"] > 0.05 and f.risk["trader1"]["lambda_max"] < 1e-3
    t = np.array([0.3, 0.6]); s = np.full(2, 0.2)
    assert np.all(w.evaluate("D1", "wV", t, s) < f.evaluate("D1", "wV", t, s))


def test_the_undiscounted_stationary_market_is_refused():
    """The stationary engine solves a risk-averse insider under consistent planning (tests/test_cara_ext.py), which needs a
    discount: without one the continuation's entropic cost is not defined."""
    d = ns.load(ns.example("ch4_kyle_back")).to_equations()
    d["agents"]["trader1"]["risk_aversion"] = 0.5
    d["params"]["rho"] = 0.0
    st = ns.Model.from_dict(d)
    with pytest.raises(NotImplementedError, match="discount"):
        ns.solve(st)


def kyle2(theta=0.0, nodes=16, eps=0.2, T=1.0):
    """Two identical insiders who both see the value (Holden and Subrahmanyam 1992), each seeing the flow less its own
    orders, on their wealth."""
    ag = {"market_maker": {"controls": "P", "observes": {"flow": "D1 dt + D2 dt + sigma_Z dwZ"}, "loss": "P^2 - 2 P V", "myopic": True}}
    for i, j in ((1, 2), (2, 1)):
        tr = {"controls": f"D{i}", "observes": {"flow": f"D{j} dt + sigma_Z dwZ"}, "loss": f"D{i} P + eps D{i}^2", "terminal": f"-Q{i} V"}
        if theta:
            tr["risk_aversion"] = theta
        ag[f"trader{i}"] = tr
    return ns.Model.from_dict({
        "name": "kyle_back_prior2", "params": {"eps": eps, "Sigma0": 1.0, "sigma_Z": 1.0}, "shocks": ["wZ"],
        "states": {"V": "0", "Q1": "D1 dt", "Q2": "D2 dt"}, "agents": ag,
        "horizon": {"T": T, "past": [{"name": "v0", "loads": {"V": "sqrt(Sigma0)"}, "rows": {"trader1.flow": 1.0, "trader2.flow": 1.0}}],
                    "continuation": "end"},
        "numerics": {"nodes": nodes}})


@pytest.mark.parametrize("theta", [0.0, 2.0])
def test_two_insiders_match_the_brute_force_reference(theta):
    """Two risk-averse insiders against the reference's symmetric equilibrium, in which a rival reacts to a deviation
    through its map on the flow it sees (the engine's convention; holding the rival's orders on the shocks fixed instead
    is a different game, 0.037 away).  16 nodes: the costs to 1e-6, the kernels to 1e-4 (both converge with the nodes:
    3e-7 and 3e-5 at 20)."""
    with open(os.path.join(os.path.dirname(__file__), "refs", "leqg_kyle_prior2.json")) as f:
        ref = json.load(f)["limit"][f"{theta:g}"]
    res = ns.solve(kyle2(theta), diagnostics=False)
    assert res.converged
    for a in ("trader1", "trader2"):
        assert abs(res.entropic_costs[a] - ref["entropic"]) < 1e-6
        assert abs(res.costs[a] - ref["expected"]) < 1e-6
    t = np.array([0.1, 0.3, 0.5, 0.8]); s = np.full(4, 0.05)
    assert np.abs(res.evaluate("D1", "v0", t, np.zeros(4)) - ref["D_g0"]).max() < 1e-4
    assert np.abs(res.evaluate("P", "wZ", t, s) - ref["P_wZ"]).max() < 1e-4
