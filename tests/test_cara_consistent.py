"""Consistent planning for risk-averse agents on the finite engine (settings.risk_planning = "consistent").

Every date's self minimises the entropic cost of its own continuation, J_t = theta^-1 log E[exp(theta C_t) | F_t], the later
selves playing the equilibrium map: P_t S_t f_t^on = 0 with S_t = (I - theta K_t)^-1, K_t the kernel of the continuation
cost from t (discounted from t) and f_t^on the FOC kernel with the agent's own later reactions on (risk.ConsistentTilt).
The default, precommitment, minimises J_0 over the whole strategy.

Pinned against extras/leqg_reference.py ConsistentGame, a brute-force discrete game solved by backward sweeps over the
date selves, each date's row the exact minimiser of its conditional entropic objective (a quadratic form in the action
and the seen signals after integrating the unseen shocks out), no gradient and no optimiser
(tests/refs/leqg_ch1_consistent.json, Richardson over n = 40 .. 120)."""
import json
import os

import numpy as np
import pytest

import noisestate as ns
from noisestate import finite_free

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "refs")
CONSISTENT = {"settings": {"risk_planning": "consistent"}}


def ch1(theta, nodes=14, **extra):
    import yaml
    d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
    d["numerics"]["nodes"] = nodes
    for a in ("player1", "player2"):
        if theta:
            d["agents"][a]["risk_aversion"] = theta
        d["agents"][a].update(extra)
    return ns.Model.from_dict(d)


@pytest.fixture(scope="module")
def cons_ref():
    with open(os.path.join(REFS, "leqg_ch1_consistent.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("theta", [0.5, 1.0, 1.5])
def test_consistent_planning_matches_the_brute_force(cons_ref, theta):
    """Chapter 1's game: the date-0 self's entropic cost, the expected cost and D1's response to a state shock at s = 0.2
    against the reference's Richardson limit, at 14 nodes (measured: J 3e-8 / 2e-7 / 1.2e-6 and D1 3e-7 / 8e-7 / 6e-6 at
    theta 0.5 / 1 / 1.5).  The precommitment equilibrium is 4e-3 / 1.4e-2 / 4.8e-2 (J) and 0.04 / 0.09 / 0.14 (D1) away: the
    two criteria are told apart."""
    lim = cons_ref[f"{theta:g}"]["limit"]
    res = ns.solve(ch1(theta), numerics=CONSISTENT, diagnostics=False)
    assert res.converged
    t = np.array([0.3, 0.5, 0.8])
    assert abs(res.risk["player1"]["entropic"] - lim["entropic"][0]) < 3e-6
    assert abs(res.costs["player1"] - lim["expected"][0]) < 1e-6
    assert np.abs(res.evaluate("D1", "w0", t, np.full(3, 0.2)) - lim["D1"]).max() < 2e-5
    pre = ns.solve(ch1(theta), diagnostics=False)
    assert np.abs(pre.evaluate("D1", "w0", t, np.full(3, 0.2)) - lim["D1"]).max() > 0.03


def test_consistent_planning_discounts_from_each_date(cons_ref):
    """With a discount (rho = 1) each date's self discounts its continuation from its own date, so the kernels discounted from
    zero enter with theta e^{rho t}: against the reference (tests/refs/leqg_ch1_consistent_rho1.json, n = 40 .. 100), measured
    4e-8 on J and 1e-6 on D1 at 14 nodes (precommitment is 5e-4 and 0.04 away)."""
    with open(os.path.join(REFS, "leqg_ch1_consistent_rho1.json")) as f:
        lim = json.load(f)["1"]["limit"]
    d = ch1(1.0).to_dict(); d["horizon"]["discount"] = 1.0
    res = ns.solve(ns.Model.from_dict(d), numerics=CONSISTENT, diagnostics=False)
    t = np.array([0.3, 0.5, 0.8])
    assert res.converged
    assert abs(res.risk["player1"]["entropic"] - lim["entropic"][0]) < 1e-6
    assert np.abs(res.evaluate("D1", "w0", t, np.full(3, 0.2)) - lim["D1"]).max() < 2e-5


def test_the_own_reactions_satisfy_the_envelope_identity():
    """At a risk-neutral equilibrium the FOC kernel with the agent's own later reactions on and the envelope one (reactions
    off) have the same projection on the seen rows: the reactions are optimal.  |P (phi^on - phi^off)| is 5.5e-9 at 14 nodes
    against |phi^on - phi^off| = 0.11 (the own reactions are 0.43)."""
    m = ch1(0.0)
    r = ns.solve(m, diagnostics=False)
    S = r._solver(); c = S.c; N, nP, ncol = c.N, len(c.prim), c.ncol
    Z = S._profile_world(r.maps).reshape(nP, N, ncol)
    a = m.agents[0]
    Zpass, R0 = S._spikes(c, r.maps, a)
    Zpass = S._passive_world(a, r.maps, Zpass, R0)
    proj = finite_free.ProjOps(S, a, *S._passive_rows(a, Zpass))
    Ron = S._responses_on(a, r.maps, R0)
    fon = finite_free.FocOps(S, a, Ron, envelope=False); foff = finite_free.FocOps(S, a, R0)
    d = fon.foc(0, fon.atoms_of(Z)) - foff.foc(0, foff.atoms_of(Z))
    assert np.abs(d).max() > 0.1 and np.abs(Ron - R0).max() > 0.4
    assert np.abs(proj.apply(d)).max() < 5e-8


def test_a_lone_agent_is_worse_off_at_date_zero_under_consistent_planning():
    """One agent against no one: precommitment minimises J_0 over the whole strategy, so the consistent plan's J_0 is no lower."""
    import yaml
    d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
    d["agents"].pop("player2"); d["shocks"].remove("w2"); d["params"].pop("p2"); d["params"].pop("r2")
    d["states"]["X"] = "D1 dt + sigma dw0"
    d["agents"]["player1"]["risk_aversion"] = 1.0
    d["numerics"]["nodes"] = 12
    m = ns.Model.from_dict(d)
    pre = ns.solve(m, diagnostics=False); con = ns.solve(m, numerics=CONSISTENT, diagnostics=False)
    assert pre.converged and con.converged
    assert con.risk["player1"]["entropic"] > pre.risk["player1"]["entropic"] + 1e-4


def test_consistent_planning_is_refused_where_it_is_not_built():
    import yaml

    def base():
        d = yaml.safe_load(open(ns.example("ch1_two_player_finite")))
        d["numerics"]["nodes"] = 8
        for a in ("player1", "player2"):
            d["agents"][a]["risk_aversion"] = 1.0
        return d
    with pytest.raises(ValueError, match="risk_planning"):
        ns.solve(ns.Model.from_dict(base()), numerics={"settings": {"risk_planning": "sometimes"}})
    d = base(); d["agents"]["player2"]["monitors"] = "player1"
    with pytest.raises(NotImplementedError, match="monitoring"):
        ns.solve(ns.Model.from_dict(d), numerics=CONSISTENT)
    d = base(); d["states"]["X"] = {"d": "(D1 + D2 + 0.5) dt + sigma dw0", "initial": 1.0}
    with pytest.raises(NotImplementedError, match="means"):
        ns.solve(ns.Model.from_dict(d), numerics=CONSISTENT)
