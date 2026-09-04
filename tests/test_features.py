"""Model-language features that only the (slow, opt-in) Chapter 5 test used to exercise."""
import os, numpy as np
import noisestate as ns
from noisestate.stationary import StationarySolver
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_two_firm_cycle_market_ties_definitions_and_two_controls():
    """Ties, definitions, lagged definitions and two controls per agent, in seconds.  The tied
    (symmetric) equilibrium must be a fixed point of the untied best-response map, and mirror
    firm 0 onto firm 1 channel by channel.  (The untied iteration may land on a different,
    asymmetric equilibrium of the same game; that is not a defect.)"""
    from make_ch5_cycle_market import build
    from noisestate.stationary import StationarySolver
    b = build(N=2, L=6.0, nodes=6, unit_range=3.0)
    tied = ns.solve(b.build(), tol=1e-8).check()
    d = b.to_dict(); d["ties"] = []
    untied_solver = StationarySolver(ns.Model.from_dict(d))
    new = untied_solver.response_map(tied.maps)
    assert max(np.abs(new[k] - tied.maps[k]).max() for k in tied.maps) < 1e-6
    ch = tied.channels

    import re

    def mirror(name):                    # swap the firm index only: w_0_2 <-> w_1_2, w_a0 <-> w_a1, w_eta0 <-> w_eta1
        m = re.match(r"^w_(\d)_(\d)$", name)
        if m:
            return f"w_{1 - int(m[1])}_{m[2]}"
        m = re.match(r"^w_(a|eta)(\d)$", name)
        if m:
            return f"w_{m[1]}{1 - int(m[2])}"
        return name
    perm = [ch.index(mirror(c)) for c in ch]
    for u0, u1 in (("P0", "P1"), ("o0", "o1")):
        assert np.abs(tied.kernel(u0) - tied.kernel(u1)[:, perm]).max() < 1e-8
    assert abs(tied.costs["firm0"] - tied.costs["firm1"]) < 1e-8


def test_lead_atoms_solve_and_are_optimal():
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["definitions"] = {"gap": {"X": 1.0, "X@-0.5": -0.5}}                 # a lead: X half a unit ahead
    d["agents"]["player1"]["loss"] = [[0.5, "gap", "gap"], ["0.5*r1", "D1", "D1"]]
    S = StationarySolver(ns.Model.from_dict(d)); res = S.solve().check()
    a = S.model.agents[0]; c = S.c; nW = c.nW
    g, out = S.best_response(a, res.maps)
    Zp = c.closed_loop(res.maps, excluded=a.name, impulse_controls=a.controls); Zpass, R = Zp[:, :nW], Zp[:, nW:]
    Resp = S._response_operators(a, R)[0]; ytil, yinst = S._passive_rows(a, Zpass); Gk = S._row_operator(ytil, yinst)
    cost = lambda cc: S.expected_loss(a, Zpass + Resp @ cc)
    c0 = out["action"][0]; L0 = cost(c0); rng = np.random.default_rng(2)
    for _ in range(5):
        gam = rng.standard_normal(Gk.shape[2]); dc = np.stack([Gk[k] @ gam for k in range(nW)], axis=1); dc *= 0.02 / np.abs(dc).max()
        assert cost(c0 + dc) >= L0 - 1e-10 and cost(c0 - dc) >= L0 - 1e-10


def test_lagged_state_feedback_in_both_engines():
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["states"]["X"]["drift"]["X@0.5"] = -0.3                                  # delayed mean reversion
    d["horizon"]["unit"] = 0.5
    ra = ns.solve(ns.Model.from_dict(d), variable="actions").check()
    rm = ns.solve(ns.Model.from_dict(d), variable="maps", method="newton").check()
    # with lagged state feedback the two paths agree only to first order in the node count
    # (1.6e-5 at 24 nodes per panel, 3.7e-6 at 48; see README "Limits")
    assert np.abs(ra.kernel("X") - rm.kernel("X")).max() < 1e-4
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); df["horizon"]["nodes"] = 6
    df["states"]["X"]["drift"]["X@0.25"] = -0.3
    rf = ns.solve(ns.Model.from_dict(df)).check()
    rfm = ns.solve(ns.Model.from_dict(df), variable="maps").check()
    assert abs(rf.costs["player1"] - rfm.costs["player1"]) < 1e-6


def test_previous_point_predictor_and_finite_discounting():
    from noisestate.sweep import sweep
    rows = sweep(os.path.join(EX, "ch3_two_player.yaml"), "p1", [3.0, 4.0, 5.0], predictor="previous")
    assert all(r["converged"] for r in rows)
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); df["horizon"]["nodes"] = 8; df["horizon"]["discount"] = 0.5
    r = ns.solve(ns.Model.from_dict(df)).check()
    assert r.costs["player1"] > 0
