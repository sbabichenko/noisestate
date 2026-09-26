"""Model-language features that only the (slow, opt-in) Chapter 5 test used to exercise."""
import os, numpy as np
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_two_firm_cycle_market_ties_definitions_and_two_controls():
    """Ties, definitions, lagged definitions and two controls per agent, in seconds.  The tied
    (symmetric) equilibrium must be a fixed point of the untied best-response map, and mirror
    firm 0 onto firm 1 channel by channel.  (The untied iteration may land on a different,
    asymmetric equilibrium of the same game; that is not a defect.)"""
    from make_ch5_cycle_market import build
    from noisestate.stationary import StationarySolver
    b = build(N=2, L=6.0, nodes=6, unit_range=3.0)
    tied = ns.solve(b, tol=1e-8).require_converged()
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


def test_lagged_state_feedback_in_both_engines():
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["states"]["X"]["drift"]["X@0.5"] = -0.3                                  # delayed mean reversion
    d.setdefault("numerics", {})["unit"] = 0.5
    ra = ns.solve(ns.Model.from_dict(d), {"variable": "actions"}).require_converged()
    rm = ns.solve(ns.Model.from_dict(d), {"variable": "maps"}).require_converged()
    # with lagged state feedback the two paths agree only to first order in the node count
    # (1.6e-5 at 24 nodes per panel, 3.7e-6 at 48; see README "Limits")
    assert np.abs(ra.kernel("X") - rm.kernel("X")).max() < 1e-4
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); df.setdefault("numerics", {})["nodes"] = 6
    df["states"]["X"]["drift"]["X@0.25"] = -0.3
    rf = ns.solve(ns.Model.from_dict(df)).require_converged()
    rfm = ns.solve(ns.Model.from_dict(df), {"variable": "maps"}).require_converged()
    assert abs(rf.costs["player1"] - rfm.costs["player1"]) < 1e-6


def test_sweep_and_finite_discounting():
    from noisestate.sweep import sweep
    rows = sweep(os.path.join(EX, "ch3_two_player.yaml"), "p1", [3.0, 4.0, 5.0])
    assert all(r.converged for r in rows)
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); df.setdefault("numerics", {})["nodes"] = 8; df["horizon"]["discount"] = 0.5
    r = ns.solve(ns.Model.from_dict(df)).require_converged()
    assert 0 < r.costs["player1"] < 0.3969      # 0.2926, below the undiscounted 0.39690577; the closed form is in test_finite_discount.py
