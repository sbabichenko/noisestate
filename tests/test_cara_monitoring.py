"""CARA with monitored deviations (Chapter 6) on the finite engine.

A risk-averse privy player's response to a deviation it knows solves f^xi_t(s) + theta <S f_t, K e_xi(s)> = 0: the
risk-neutral condition on the seed plus the cross term of its cost between the shocks and the seed, weighted by S = (I -
theta K)^-1 (risk.Tilt.seed_operator; the derivation is in docs/method.md).  No conditional covariance and no prior on the
seed enter.  The deviator's own blip continuation is such a response too.

Pinned against extras/leqg_reference.py MonitorGame, a brute-force discrete game in which player 2 is privy to player 1's
deviations: the privy players' response columns minimise their exact entropic objective of the seed world (log det plus
the linear part), nothing of the engine's derivation used (tests/refs/leqg_monitor.json, Richardson over n = 40 .. 120)."""
import json
import os

import numpy as np
import pytest

import noisestate as ns

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "refs")


def follower(th1=0.0, th2=0.0, nodes=14, monitored=True, udw=(0.0, 0.0)):
    """Player 1 regulates its own state X1 from a signal; player 2 tracks X1 with its own state X2 from a signal of the gap and
    is privy to player 1's deviations (MonitorGame's continuous limit).  udw: each player's cost gains udw_i int D_i dW, its own
    control against its own state's shock (an integral whose quantity is the agent's current control)."""
    a1 = {"controls": "D1", "observes": {"y1": "sqrt(p1) X1 dt + dw1"}, "loss": "X1^2 + r1 D1^2", "risk_aversion": th1}
    a2 = {"controls": "D2", "observes": {"y2": "sqrt(p2) X2 dt - sqrt(p2) X1 dt + dw3"},
          "loss": "X2^2 - 2 X1 X2 + X1^2 + r2 D2^2", "risk_aversion": th2}
    if udw[0]:
        a1["integrals"] = [[udw[0], "D1", "w0"]]
    if udw[1]:
        a2["integrals"] = [[udw[1], "D2", "w2"]]
    if monitored:
        a2["monitors"] = "p1"
    return ns.Model.from_dict({
        "name": "follower", "params": {"p1": 3.0, "p2": 3.0, "r1": 0.1, "r2": 0.1, "s2": 1.0},
        "shocks": ["w0", "w1", "w2", "w3"], "states": {"X1": "D1 dt + dw0", "X2": "D2 dt + s2 dw2"},
        "agents": {"p1": a1, "p2": a2},
        "horizon": {"T": 1.0}, "numerics": {"nodes": nodes}})


@pytest.fixture(scope="module")
def monitor_ref():
    with open(os.path.join(REFS, "leqg_monitor.json")) as f:
        return json.load(f)


def _responses(res, times):
    return res.deviation_response("p1", ["X1", "D1", "X2", "D2"]).over(np.asarray(times), np.full(len(times), 0.2))


@pytest.mark.parametrize("case, tol, tol_J", [("neutral", 1e-4, 3e-6), ("both 1", 1e-4, 3e-6), ("responder 1", 1e-4, 3e-6),
                                              ("deviator 1.5", 1e-4, 3e-6), ("both 1, own integrals", 1.5e-3, 3e-4)])
def test_privy_responses_match_the_brute_force(monitor_ref, case, tol, tol_J):
    """The response to a unit seed of player 1 at s = 0.2 (X1 and player 1's own blip continuation D1, X2 and player 2's
    response D2 at t = 0.3, 0.5, 0.8) and both players' entropic costs, at 14 nodes against the reference's limit (measured:
    1e-7 risk neutral; 5e-5 on D2 at t = 0.3, the steepest point, and 1e-6 on J with theta 1).  With the own-control integrals
    the seed is also a point mass in the integral's quantity (risk.Tilt.seed_constant) and the engine converges like N^-2.4
    (7e-4 on D1 and 1.3e-4 on J at 14 nodes, 3.5e-4 and 7e-5 at 18; the reference's limit is stable to 1e-6 when n = 160,
    200 are added)."""
    ref = monitor_ref[case]
    th = ref["params"]["theta"]
    res = ns.solve(follower(*th, udw=tuple(ref["params"].get("udw", (0.0, 0.0)))), diagnostics=False)
    assert res.converged
    lim = ref["limit"]
    got = _responses(res, ref["levels"][0]["times"])
    for i, k in enumerate(("X1", "D1", "X2", "D2")):
        assert np.abs(got[:, i] - np.array(lim[k])).max() < tol, k
    ent = [res.risk[a]["entropic"] if a in res.risk else res.costs[a] for a in ("p1", "p2")]
    assert np.abs(np.array(ent) - np.array(lim["entropic"])).max() < tol_J


def test_without_the_seed_term_the_responses_are_wrong(monitor_ref, monkeypatch):
    """The risk-neutral response condition f^xi = 0 with the risk-averse profile misses the reference by 0.1 to 1: the term
    is what makes the match."""
    from noisestate import risk
    monkeypatch.setattr(risk.Tilt, "seed_operator", lambda self, ui, Zw: np.zeros((self.geo.N, len(self.c.prim) * self.geo.N)))
    ref = monitor_ref["responder 1"]
    res = ns.solve(follower(*ref["params"]["theta"], nodes=10), diagnostics=False)
    got = _responses(res, ref["levels"][0]["times"])
    assert np.abs(got[:, 3] - np.array(ref["limit"]["D2"])).max() > 0.1


def test_a_risk_neutral_responder_needs_no_term():
    """theta = 0 takes the risk-neutral path: a model whose agents all have risk_aversion 0 solves as one without the key."""
    a = ns.solve(follower(0.0, 0.0, nodes=8), diagnostics=False)
    d = follower(0.0, 0.0, nodes=8).to_dict()
    for ag in d["agents"].values():
        ag.pop("risk_aversion", None)
    b = ns.solve(ns.Model.from_dict(d), diagnostics=False)
    for n in a.maps:
        assert np.array_equal(a.maps[n], b.maps[n])


def test_the_seed_spike_in_an_integral_is_counted(monitor_ref, monkeypatch):
    """With the own-control integrals the seed spike itself loads on the shock of its instant (seed_constant): without that
    term the deviator's own continuation misses the reference by 0.2 and more."""
    from noisestate import risk
    monkeypatch.setattr(risk.Tilt, "seed_constant", lambda self, ui, Zw, spread: np.zeros(self.geo.N))
    ref = monitor_ref["both 1, own integrals"]
    res = ns.solve(follower(*ref["params"]["theta"], nodes=10, udw=tuple(ref["params"]["udw"])), diagnostics=False)
    got = _responses(res, ref["levels"][0]["times"])
    assert np.abs(got[:, 0] - np.array(ref["limit"]["X1"])).max() > 0.1


# ------------------------------------------------------------------------------------------------ an own-control integral
@pytest.fixture(scope="module")
def udw_ref():
    with open(os.path.join(REFS, "leqg_ch1_udw.json")) as f:
        return json.load(f)


def ch1_udw(theta, nodes=16, udw=0.5):
    import yaml
    d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
    d["numerics"]["nodes"] = nodes
    for a, u in (("player1", "D1"), ("player2", "D2")):
        d["agents"][a]["integrals"] = [[udw, u, "w0"]]
        d["agents"][a]["risk_aversion"] = theta
    return ns.Model.from_dict(d)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_an_own_control_integral_matches_the_reference(udw_ref, theta):
    """udw int D_i sigma dW0 in both players' costs of Chapter 1's game (the reference's sum of D_{i,k} sigma sqrt(dt) g0_k: the
    control against the shock of its own instant, as a market maker's quote against the noise trades): the spike's point mass
    on the shock of its instant is carried by its correction (risk.Tilt.Ldelta).  At 16 nodes against the Richardson limit of
    the discrete reference (n = 40 .. 200): measured 4e-6 on J and 7e-6 on D1's response at theta 1 (N^-2.4: 2e-6 and 3e-6 at
    20 nodes).  Without the point mass D1 is off by 0.03 (theta 0.5) to 0.04 (theta 1)."""
    lim = udw_ref["cases"][f"{theta:g}"]["limit"]
    res = ns.solve(ch1_udw(theta), diagnostics=False)
    assert res.converged
    t = np.array([0.3, 0.5, 0.8])
    assert abs(res.entropic_costs["player1"] - lim["entropic"][0]) < 6e-6
    assert abs(res.costs["player1"] - lim["expected"][0]) < 2e-6
    assert np.abs(res.evaluate("D1", "w0", t, np.full(3, 0.2)) - lim["D1"]).max() < 1.5e-5


def test_an_own_control_integral_is_refused_on_the_stationary_engine():
    d = yaml_ch3()
    d["agents"]["player1"]["risk_aversion"] = 0.5
    d["agents"]["player1"]["integrals"] = [[0.5, "D1", "w0"]]
    with pytest.raises(NotImplementedError, match="current control of its own"):
        ns.solve(ns.Model.from_dict(d))


def yaml_ch3():
    import yaml
    d = yaml.safe_load(open(ns.example("ch3_two_player")))
    d.setdefault("horizon", {})["discount"] = 0.5
    return d
