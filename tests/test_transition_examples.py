"""The two shipped transition examples: the Chapter 3 precision change (a model file with a past solved on the
fly) and the Kyle-Back market started from a prior (initial shocks, the game ending at T), pinned by node
convergence."""
import numpy as np
import noisestate as ns
from helpers import example, example_path, slow


@slow("slow (9 s; its excess costs are tests/test_transition_result.py's at T = 9, its solve tests/test_baseline.py's); set NOISESTATE_SLOW=1")
def test_ch3_precision_change_example():
    """examples/ch3_precision_change.yaml (p1 = 3 to 10, T = 6, 12 nodes, 9 s): converged, the excess costs
    0.024027 and 0.028619 (the T = 9 values of tests/test_transition_result.py to 5e-8: the transition's cost
    does not depend on where the closure sits once it has settled), settled 4.8e-4 with the guard firing
    (T = 6 is short for a 1e-2 closed-loop decay; T = 9 gives 2.7e-6)."""
    res = ns.solve(example_path("ch3_precision_change"))
    assert res.converged and isinstance(res, ns.TransitionResult) and res.model.horizon.kind == "transition"
    assert res.past.provenance["name"] == "ch3_two_player" and res.continuation.compiled.grid.n == 12
    assert abs(res.excess_costs["player1"] - 0.024027) < 2e-6 and abs(res.excess_costs["player2"] - 0.028619) < 2e-6
    assert 4e-4 < res.settled < 6e-4 and "TRANSITION NOT SETTLED" in res.summary()
    assert abs(res.loss_path["player1"][0] - 0.43506) < 1e-4 and abs(res.new_flows["player1"] - 0.42174) < 1e-4


def lambda0(res):
    """The market maker's weight on the flow increment at age 0 at time 0+: the initial price impact."""
    g = res.grid; i0 = int(np.flatnonzero((g.t < 1e-12) & (g.a < 1e-12) & ~g.upper)[0])
    return float(res.maps["market_maker"][0, 0, i0])


def test_kyle_back_prior_example():
    """examples/kyle_back_prior.yaml (eps = 0.1, T = 2): lambda(0+) = 0.658872 at 8, 12 and 16 nodes (to 2e-6),
    the market maker's belief error of V falls monotonically from Sigma0 = 1 to 0.132 Sigma0 at T, the trader's
    profit is 1.11508; the initial shock's loading sqrt(Sigma0) is a parameter expression that survives
    with_params().  Back's price impact at eps = 0 is sqrt(Sigma0/T)/sigma_Z = 0.7071 at T = 2 (the identity
    test below; the README records the eps sweep toward it).  The trader's second-order check flags NOT A
    MINIMUM (-0.0102 relative to the form's largest curvature, the prior column's): the strip quadrature's
    error on the D P cross term at a small trading cost, with or without the prior (README, Limits)."""
    m = example("kyle_back_prior")
    assert m.horizon.past["initial"][0]["loads"] == {"V": 1.0} and m.with_params(eps=0.2).horizon.past["initial"][0]["loads"] == {"V": 1.0}
    assert m.with_params(Sigma0=4.0).horizon.past["initial"][0]["loads"] == {"V": 2.0}
    res = {n: ns.solve(m.with_horizon(nodes=n)) for n in (8, 12)}
    for n, r in res.items():
        assert r.converged and r.continuation is None and r.shocks == ["wZ", "v0"], n
        assert abs(lambda0(r) - 0.658872) < 2e-6, (n, lambda0(r))
        be = r.belief_error("market_maker", "V")
        assert abs(be[0] - 1.0) < 1e-10 and np.all(np.diff(be) <= 1e-9) and abs(be[-1] - 0.1318) < 5e-4
        assert abs(-r.costs["trader1"] - 1.11508) < 2e-5
    assert not res[12].second_order["trader1"]["ok"] and res[12].second_order["market_maker"]["ok"]
    assert abs(res[12].second_order["trader1"]["min"] + 0.010235) < 3e-4, res[12].second_order["trader1"]


def test_kyle_back_prior_martingale_identity():
    """The price is a martingale with a constant impact (the market maker's map is the constant lambda on every
    node of the strip), so lambda^2 sigma_Z^2 T = Sigma0 - Sigma_T with Sigma_T the market maker's belief error
    of V at T: at 12 nodes lambda = 0.614143 against sqrt((1 - Sigma_T)/2) = 0.614142 at eps 0.2 and 0.658872
    against 0.658865 at 0.1, pinned to 1e-4 (two solves, 2 s).  Back's eps = 0 limit is sqrt(Sigma0/T)/sigma_Z
    = 0.7071, which the README's eps sweep approaches from below."""
    m = example("kyle_back_prior").with_horizon(nodes=12)
    for eps, lam_ref in ((0.2, 0.614143), (0.1, 0.658872)):
        res = ns.solve(m.with_params(eps=eps))
        lam = lambda0(res); sig_T = res.belief_error("market_maker", "V")[-1]
        mm = res.maps["market_maker"][0, 0, :res.compiled.N][~res.grid.upper]
        assert res.converged and abs(lam - lam_ref) < 2e-6 and mm.max() - mm.min() < 1e-4, (eps, lam)
        assert abs(lam - np.sqrt((1.0 - sig_T) / 2.0)) < 1e-4 and lam < np.sqrt(0.5), (eps, lam, sig_T)
