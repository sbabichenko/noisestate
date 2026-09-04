import os
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_resolution_flag_and_stability_on_the_two_firm_market():
    """At 6 nodes per panel the two-firm cycle market is under-resolved and the result says so; the
    map and action-kernel paths then disagree.  At 14 nodes they agree and the stability report is
    computed at a genuine fixed point."""
    from make_ch5_cycle_market import build
    coarse = ns.solve(build(N=2, L=6.0, nodes=6, unit_range=3.0).build(), tol=1e-8).check()
    assert not coarse.resolution_ok and "UNDER-RESOLVED" in coarse.summary()
    d = build(N=2, L=6.0, nodes=14, unit_range=3.0).to_dict(); d["ties"] = []
    ra = ns.solve(ns.Model.from_dict(d), tol=1e-8).check(); rm = ns.solve(ns.Model.from_dict(d), tol=1e-8, variable="maps").check()
    assert abs(ra.costs["firm0"] - rm.costs["firm0"]) < 1e-3
    st = rm.stability(); assert st["fixed_point_residual"] < 1e-6 and st["radius"] > 0 and "stability" in rm.to_dict()


def test_stability_of_the_chapter_3_game_and_finite_engine():
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml")).check(); s = r.stability()
    assert s["stable"] and s["radius"] < 1.0 and s["fixed_point_residual"] < 1e-8
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); df["horizon"]["nodes"] = 6
    rf = ns.solve(ns.Model.from_dict(df)).check(); sf = rf.stability()
    assert sf["stable"] and sf["radius"] < 1.0
