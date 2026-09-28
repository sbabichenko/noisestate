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
