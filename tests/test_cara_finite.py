"""Risk-averse (CARA) agents solved on the spectral finite engine (noisestate/risk.py): the entropic objective
theta^-1 log E exp(theta C), Chapter 1's appendix thm:risk_sensitive_appendix.

Pinned against extras/leqg_reference.py, a brute-force discrete-time solver independent of the package (linear maps on
each player's signal history, the exact log-det entropic cost, L-BFGS best responses): tests/refs/leqg_ch1.json holds
its equilibria at n = 50, 100, 150, 200, 300, 400 steps (`python extras/leqg_reference.py table tests/refs/leqg_ch1.json`,
about twenty minutes), whose Richardson limit (a polynomial in 1/n through every level) the engine must reach.  The first-order condition is checked independently of the
engine's first-order operators: the entropic cost of the solution, computed from its closed loop and the cost kernel's
spectrum alone, is stationary in the agent's own strategy."""
import json
import os

import numpy as np
import pytest

import noisestate as ns
from noisestate.risk import RiskBreakdown

from helpers import slow

REFS = os.path.join(os.path.dirname(__file__), "refs", "leqg_ch1.json")


def ch1(theta=(1.0, 1.0), nodes=12, p2=None, r2=None, **numerics):
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    d["numerics"]["nodes"] = nodes
    d["numerics"].update(numerics)
    if p2 is not None:
        d["params"]["p2"] = p2
    if r2 is not None:
        d["params"]["r2"] = r2
    for a, th in zip(("player1", "player2"), theta):
        if th:
            d["agents"][a]["risk_aversion"] = th
    return ns.Model.from_dict(d)


def richardson(values, ns_):
    A = np.array([[1.0 / n ** k for k in range(len(ns_))] for n in ns_])
    return float(np.linalg.solve(A, np.asarray(values, dtype=float))[0])


@pytest.fixture(scope="module")
def refs():
    with open(REFS) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ theta = 0 is today's solve
def test_zero_risk_aversion_is_the_risk_neutral_solve_bit_for_bit():
    """An explicit 0 takes the risk-neutral path: no correction, no continuation, the same maps to the bit."""
    a, b = ns.solve(ch1((0.0, 0.0))), ns.solve(ns.load(ns.example("ch1_two_player_finite")))
    assert a.costs == b.costs and a.evaluations == b.evaluations
    for n in a.maps:
        assert np.array_equal(a.maps[n], b.maps[n])
    assert a.risk == {} and a.entropic_costs == a.costs


# ------------------------------------------------------------------------------------------------ against the reference
@pytest.mark.parametrize("case,theta,kw", [("theta 1", (1.0, 1.0), {}), ("mixed", (1.0, 0.0), {}),
                                           ("asymmetric", (1.0, 0.5), {"p2": 1.0, "r2": 0.2})])
def test_matches_the_brute_force_reference(refs, case, theta, kw):
    """The entropic and expected costs of both players and D1's response to a state shock against the Richardson limit
    of the discrete reference (its discretisation error is O(1/n): 7e-3 relative at n = 100 on the entropic cost)."""
    ref = refs[case]
    ns_ = [r["n"] for r in ref["levels"]]
    res = ns.solve(ch1(theta, **kw), diagnostics=False)
    assert res.converged
    for i, a in enumerate(("player1", "player2")):
        J = richardson([r["entropic"][i] for r in ref["levels"]], ns_)
        EC = richardson([r["expected"][i] for r in ref["levels"]], ns_)
        assert abs(res.entropic_costs[a] - J) < 1e-6, (a, res.entropic_costs[a], J)
        assert abs(res.costs[a] - EC) < 1e-6, (a, res.costs[a], EC)
    t = np.array([0.3, 0.5, 0.8])
    D1 = res.evaluate("D1", "w0", t, np.full(3, 0.2))
    D1ref = [richardson([r["D1"][k] for r in ref["levels"]], ns_) for k in range(3)]
    assert np.max(np.abs(D1 - D1ref)) < 3e-5, (D1, D1ref)


def test_every_theta_of_the_symmetric_table(refs):
    """Along theta = 0 .. 2 the costs follow the reference's limit; risk aversion lowers the expected cost at first (the
    players push back harder, which counters their free riding on each other) and raises it later."""
    J, EC = [], []
    for th in (0.0, 0.5, 1.5, 2.0):
        ref = refs[f"theta {th:g}"]
        ns_ = [r["n"] for r in ref["levels"]]
        res = ns.solve(ch1((th, th)), diagnostics=False)
        J.append(res.entropic_costs["player1"]); EC.append(res.costs["player1"])
        assert abs(J[-1] - richardson([r["entropic"][0] for r in ref["levels"]], ns_)) < 1e-6
        assert abs(EC[-1] - richardson([r["expected"][0] for r in ref["levels"]], ns_)) < 1e-6
    assert EC[1] < EC[0] and EC[3] > EC[0] and np.all(np.diff(J) > 0)


# ------------------------------------------------------------------------------------------------ the first-order condition
def _entropic(sol, agent, maps):
    Z = sol.c.closed_loop(maps)
    return sol.risk_report(agent, Z, sol.expected_cost(agent, Z))["entropic"]


def _slopes(res, sol, agent, h=1e-4):
    g0 = res.maps[agent.name]; t, s = sol.c.g.t, sol.c.g.s
    out = []
    for d in (g0, g0 * (t - 0.5)[None, None, :], g0 * np.cos(3 * s)[None, None, :]):
        up, dn = dict(res.maps), dict(res.maps)
        up[agent.name] = g0 + h * d; dn[agent.name] = g0 - h * d
        out.append((_entropic(sol, agent, up) - _entropic(sol, agent, dn)) / (2 * h))
    return np.array(out)


@pytest.mark.parametrize("extra", [None, "terminal and discount"])
def test_entropic_cost_is_stationary_in_the_own_strategy(extra):
    """Independently of the first-order operators: the entropic cost from the closed loop, the expected cost and K's
    spectrum (risk_report) has zero slope at the solution along three directions of player 1's own map, against slopes
    of 1e-3 to 1e-1 at the risk-neutral equilibrium judged by the same objective; the slope falls with the nodes."""
    d = ch1((0.5, 0.5) if extra else (1.0, 1.0)).to_dict()
    if extra:
        d["horizon"]["discount"] = 0.7
        d["agents"]["player1"]["terminal"] = [[1.0, "X", "X"]]
    slopes = {}
    for nodes in (12, 16):
        d["numerics"]["nodes"] = nodes
        m = ns.Model.from_dict(d)
        res = ns.solve(m, diagnostics=False)
        sol = res.solver_class(m, **res.solver_kw)
        slopes[nodes] = np.abs(_slopes(res, sol, m.agents[0])).max()
    d0 = json.loads(json.dumps(d))
    for a in d0["agents"].values():
        a.pop("risk_aversion", None)
    r0 = ns.solve(ns.Model.from_dict(d0), diagnostics=False)
    far = np.abs(_slopes(r0, sol, m.agents[0])).min()
    assert far > 1e-3
    assert slopes[16] < 1e-5 * far and slopes[16] < slopes[12]


def test_krylov_solve_of_the_correction_is_the_assembled_one():
    """The best response's preconditioned GMRES (the risk-neutral system factored, the correction applied) against the
    whole system assembled column by column and solved directly."""
    from noisestate import finite_free
    m = ch1((1.0, 1.0))
    res = ns.solve(m, diagnostics=False)
    sol = res.solver_class(m, **res.solver_kw)
    a = m.agents[0]
    captured = {}
    orig = finite_free.FocSystem.solve

    def spy(self, x0=None):
        captured["sys"] = self
        return orig(self, x0)
    finite_free.FocSystem.solve = spy
    try:
        _, out = sol.best_response(a, res.maps)
    finally:
        finite_free.FocSystem.solve = orig
    system = captured["sys"]
    direct = np.linalg.solve(system.matrix(with_tilt=True), -system.bvec)
    assert np.max(np.abs(system.expand(direct).reshape(out["gamma"].shape) - out["gamma"])) < 1e-9 * np.abs(direct).max()


# ------------------------------------------------------------------------------------------------ breakdown and refusals
def test_breakdown_is_detected_and_named():
    """Beyond the breakdown (theta about 2.9 on Chapter 1's game at 8 nodes: E exp(theta C) infinite) a solve started from
    the risk-neutral equilibrium ends on strategies past it and raises RiskBreakdown, a ValueError, naming the agent and
    1 / lambda_max there."""
    d = ch1((0.0, 0.0), nodes=8)
    r0 = ns.solve(d, diagnostics=False)
    with pytest.raises(RiskBreakdown, match=r"player\d.*1 / lambda_max") as info:
        ns.solve(ch1((3.2, 3.2), nodes=8), diagnostics=False, start_from=r0.maps)
    assert isinstance(info.value, ValueError) and info.value.theta_star < 3.2 and info.value.reached is None
    assert info.value.theta * info.value.lam_max >= 1.0


@slow("slow (the continuation's 13 steps up to the breakdown, 10 s); set NOISESTATE_SLOW=1")
def test_the_continuation_ends_at_the_breakdown():
    """From no start the continuation in risk aversion climbs to the breakdown and says where the path met it: short of
    theta = 3.2, at about 2.9 on 8 nodes (the step whose equilibrium has theta lambda_max within reach of 1)."""
    with pytest.raises(RiskBreakdown, match="along risk aversion reach the breakdown") as info:
        ns.solve(ch1((3.2, 3.2), nodes=8), diagnostics=False)
    assert 2.7 < info.value.reached < 3.2


def test_the_report_refuses_a_profile_past_the_breakdown():
    """The uncontrolled world (every map zero) has lambda_max = 0.81 on Chapter 1's game (0.405 = 4 / pi^2 for the form
    W'MW, K being twice it), so theta = 1.5 is past its breakdown and theta = 1 is not."""
    m = ch1((1.5, 1.5), nodes=8)
    from noisestate.finite_spectral import SpectralFiniteSolver
    sol = SpectralFiniteSolver(m)
    Z = sol.c.closed_loop(sol.zero_maps())
    lam = sol._spectrum(m.agents[0], Z).lam_max
    assert abs(lam - 8 / np.pi ** 2) < 1e-3 * lam
    with pytest.raises(RiskBreakdown):
        sol.risk_report(m.agents[0], Z, sol.expected_cost(m.agents[0], Z))


def test_unsupported_combinations_are_refused():
    d = ch1((0.5, 0.5)).to_dict()
    d["agents"]["player1"]["loss"].append([-2.0, "X"])                  # a target: (X - 1)^2 less its constant: solved now
    d["agents"]["player1"]["integrals"] = [[1.0, "X", "w0"]]             # ... but not together with integrals
    with pytest.raises(NotImplementedError, match="without means"):
        ns.solve(ns.Model.from_dict(d))
    st = ns.load(ns.example("ch3_two_player")).to_dict()
    st["agents"][next(iter(st["agents"]))]["risk_aversion"] = 0.5
    st["horizon"]["discount"] = 0.0
    with pytest.raises(NotImplementedError, match="consistent planning.*discount"):
        ns.solve(ns.Model.from_dict(st))
    with pytest.raises(NotImplementedError, match="strategy"):
        ns.solve(ch1((0.5, 0.0), nodes=8)).strategy("D1")


# ------------------------------------------------------------------------------------------------ the result
def test_result_carries_the_entropic_cost_and_the_checks():
    res = ns.solve(ch1((1.0, 0.0)))
    rk = res.risk["player1"]
    assert set(rk) == {"risk_aversion", "entropic", "expected", "lambda_max", "theta_lambda_max"}
    assert rk["expected"] == res.costs["player1"] < rk["entropic"] == res.entropic_costs["player1"]
    assert "player2" not in res.risk and res.entropic_costs["player2"] == res.costs["player2"]
    assert "entropic" in res.summary()
    dec = res.foc["player1"]["D1"]
    assert np.allclose(dec["foc"], dec["physical"] + dec["wedge"] + dec["risk"])
    assert res.second_order["player1"]["ok"] and res.second_order["player1"]["bound"] == "entropic"
    assert res.diagnostics.assess().accepted
    payload = res.to_dict()
    from noisestate.schema import validate
    validate(payload, "payload")
    assert payload["risk"]["player1"]["entropic"] == rk["entropic"]


def test_entropic_cost_converges_in_the_basis():
    """The Galerkin part of the correction and the eigenvalues it misses (their second-order term through the exact
    tr K^2): the entropic cost moves by less than 1e-8 between the default basis and one two and a half times larger."""
    a = ns.solve(ch1((1.0, 1.0)), diagnostics=False)
    b = ns.solve(ch1((1.0, 1.0), settings={"risk_basis": 40}), diagnostics=False)
    assert abs(a.entropic_costs["player1"] - b.entropic_costs["player1"]) < 1e-8
    assert abs(a.costs["player1"] - b.costs["player1"]) < 1e-9


@slow("slow (150 evaluations at 16 nodes); set NOISESTATE_SLOW=1")
def test_near_the_breakdown():
    """theta = 2.5, theta lambda_max = 0.94 at the equilibrium: the continuation in risk aversion starts from the
    risk-neutral equilibrium (itself past the breakdown at this theta: theta lambda_max = 1.37 there) and the costs
    reach the reference's five-level limit (n = 50 is past the discrete game's own breakdown) at 16 nodes."""
    with open(REFS) as f:
        ref = json.load(f)["theta 2.5"]
    ns_ = [r["n"] for r in ref["levels"]]
    res = ns.solve(ch1((2.5, 2.5), nodes=16), diagnostics=False)
    assert res.converged and "continuation in risk aversion" in res.message
    assert 0.9 < res.risk["player1"]["theta_lambda_max"] < 1.0
    assert abs(res.entropic_costs["player1"] - richardson([r["entropic"][0] for r in ref["levels"]], ns_)) < 5e-6
    assert abs(res.costs["player1"] - richardson([r["expected"][0] for r in ref["levels"]], ns_)) < 2e-6


@slow("slow (the reference at n = 200); set NOISESTATE_SLOW=1")
def test_regenerated_reference_row_agrees():
    """The reference itself, recomputed at n = 50, 100, 200 for theta = 1: the stored table's levels to 1e-8, and its
    three-level limit against the engine.  The reference stops its fixed point at a map change of 1e-10 (L-BFGS inside),
    so a level reproduces only to about 1e-9 and moves with the BLAS threads: 1.28e-9 at 2 threads, under 1e-9 at 4.
    1e-8 still catches any change to the reference, four decades below the 2e-5 the engine is held to."""
    from leqg_reference import Game, equilibrium, record
    with open(REFS) as f:
        stored = {r["n"]: r for r in json.load(f)["theta 1"]["levels"]}
    recs = []
    for n in (50, 100, 200):
        g = Game(n=n, theta=(1.0, 1.0))
        Gs, _ = equilibrium(g)
        recs.append(record(g, Gs))
        assert abs(recs[-1]["entropic"][0] - stored[n]["entropic"][0]) < 1e-8
    J = richardson([r["entropic"][0] for r in recs], [50, 100, 200])
    res = ns.solve(ch1((1.0, 1.0)), diagnostics=False)
    assert abs(res.entropic_costs["player1"] - J) < 2e-5
