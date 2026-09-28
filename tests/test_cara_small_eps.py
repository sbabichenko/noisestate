"""Risk-averse Kyle-Back at a small trading cost: the retry with tighter best responses, the GMRES cycles, and the
second-order probe of an entropic cost whose loss Hessian is indefinite (finite_free._entropic_probe)."""
import numpy as np
import pytest

import noisestate as ns

from helpers import slow


def kyle(theta=0.0, nodes=16, eps=0.2):
    """One CARA insider on its wealth with a value drawn at 0- (the prior Kyle-Back of tests/test_cara_kyle.py)."""
    tr = {"controls": "D1", "observes": {"flow": "sigma_Z dwZ"}, "loss": "D1 P + eps D1^2", "terminal": "-Q1 V"}
    if theta:
        tr["risk_aversion"] = theta
    return ns.Model.from_dict({
        "name": "kyle1", "params": {"eps": eps, "Sigma0": 1.0, "sigma_Z": 1.0}, "shocks": ["wZ"],
        "states": {"V": "0", "Q1": "D1 dt"},
        "agents": {"market_maker": {"controls": "P", "observes": {"flow": "D1 dt + sigma_Z dwZ"}, "loss": "P^2 - 2 P V", "myopic": True},
                   "trader1": tr},
        "horizon": {"T": 1.0, "past": [{"name": "v0", "loads": {"V": "sqrt(Sigma0)"}, "rows": {"trader1.flow": 1.0}}],
                    "continuation": "end"},
        "numerics": {"nodes": nodes}})


def test_an_indefinite_loss_gets_the_entropic_probe():
    """Kyle's D P makes the loss Hessian indefinite: the second-order check probes the entropic cost itself along the
    expected cost's lowest and highest directions (it used to be skipped as "not checked")."""
    res = ns.solve(kyle(2.0, nodes=12))
    so = res.second_order["trader1"]
    assert res.converged and so["bound"] == "probe" and so["converged"] and so["ok"]
    assert so["min"] > 0 and "expected_min" in so


@slow()
def test_a_small_trading_cost_converges_with_the_retry():
    """eps = 0.05, theta 1.5 from the theta 1.25 equilibrium at 16 nodes: the first attempt stalls at the best responses'
    Krylov floor (residual 2e-5), the retry with 1e-6 converges to CE 0.53980 (the value 20 nodes give directly)."""
    prev = None
    for e in (0.2, 0.14, 0.1, 0.07):
        prev = ns.solve(kyle(0.0, eps=e), diagnostics=False, **({"start_from": prev.maps} if prev is not None else {}))
    for th in (1.0, 1.25):
        prev = ns.solve(kyle(th, eps=0.05), diagnostics=False, start_from=prev.maps)
        assert prev.converged
    res = ns.solve(kyle(1.5, eps=0.05), diagnostics=False, start_from=prev.maps)
    assert res.converged and "retried" in res.message
    assert -res.entropic_costs["trader1"] == pytest.approx(0.53980, abs=2e-5)


@pytest.mark.parametrize("planning", ["precommitment", "consistent"])
def test_the_proximal_best_response_has_the_same_fixed_point(planning):
    """The retry's proximal term mu (g - g_prev) vanishes at a fixed point: forced on from the start, the solve lands on the
    equilibrium of the plain one (Chapter 1's game, theta 1 for both players, 12 nodes)."""
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    for a in ("player1", "player2"):
        d["agents"][a]["risk_aversion"] = 1.0
    m = ns.Model.from_dict(d)
    num = {"settings": {"risk_planning": planning}}
    plain = ns.solve(m, num, diagnostics=False)
    S = ns.engines.spectral(m, settings=num["settings"])
    S._risk_prox = S.RISK_PROX
    prox = S.solve(diagnostics=False)
    assert plain.converged and prox.converged
    assert abs(prox.entropic_costs["player1"] - plain.entropic_costs["player1"]) < 1e-8
    assert np.abs(prox.maps["player1"] - plain.maps["player1"]).max() < 1e-6 * np.abs(plain.maps["player1"]).max()


@slow()
def test_the_proximal_retry_crosses_the_frozen_system_s_singularity():
    """eps 0.05 past theta 1.58, where the insider's frozen best-response system E^Q[C''] goes singular at the equilibrium
    (its smallest eigenvalue, relative to the largest: 3.1e-3 at theta 1.5, -2.9e-4 at 1.6, -3.6e-3 at 1.7) while the
    entropic cost stays convex: the plain iteration diverged from theta 1.5 to 1.6 (the market maker's system went singular
    at a wild iterate).  The retry's proximal best responses cross it.  The certainty equivalents against
    extras/kyle_reference.py (eps 0.05, Sigma0 1, n = 40 .. 120, Richardson over the five levels, reference fixed point to a
    map change of 1e-5 or better): 0.5206571 at theta 1.7 and 0.4953707 at 2 (the four-level limits differ by 4e-6)."""
    prev = None
    for e in (0.2, 0.14, 0.1, 0.07):
        prev = ns.solve(kyle(0.0, eps=e), diagnostics=False, **({"start_from": prev.maps} if prev is not None else {}))
    for th in (1.0, 1.5, 1.7, 2.0):
        prev = ns.solve(kyle(th, eps=0.05), diagnostics=False, start_from=prev.maps)
        assert prev.converged
        if th == 1.7:
            assert -prev.entropic_costs["trader1"] == pytest.approx(0.5206571, abs=1e-5)
    assert "proximal" in prev.message
    assert -prev.entropic_costs["trader1"] == pytest.approx(0.4953707, abs=1e-5)


def test_a_breakdown_at_a_non_converged_iterate_is_reported_not_raised(monkeypatch):
    """A solve that stops short ends at its best iterate, which can be anywhere; a breakdown there says nothing of the
    equilibrium, and raising it made the risk-averse solve give up before its proximal retry.  At a converged solve the
    breakdown is the equilibrium's and still raises."""
    from noisestate.finite_spectral import SpectralFiniteSolver
    from noisestate.risk import RiskBreakdown

    def past_it(self, agent, *a, **k):
        raise RiskBreakdown(agent.name, 1.0, 1.5)
    monkeypatch.setattr(SpectralFiniteSolver, "risk_report", past_it)
    m = kyle(1.0, nodes=8)
    base = ns.solve(kyle(0.0, nodes=8), diagnostics=False)
    res = ns.engines.spectral(m).solve(start_from=base.maps, max_evaluations=1, diagnostics=False)
    assert not res.converged and np.isnan(res.entropic_costs["trader1"]) and "past trader1's risk-sensitive breakdown" in res.message
    with pytest.raises(RiskBreakdown):                  # the same maps taken as converged: the breakdown is the equilibrium's
        ns.engines.spectral(m)._result(base.maps, converged=True, residual=0.0, evaluations=1, message="", solve_kw={},
                                       diagnostics=False)
