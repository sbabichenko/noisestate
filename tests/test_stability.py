import noisestate as ns
from helpers import example_path, example_dict

def test_resolution_flag_and_stability_on_the_two_firm_market():
    """At 6 nodes per panel the two-firm cycle market is under-resolved and the result says so; the
    map and action-kernel paths then disagree.  At 14 nodes they agree and the stability report is
    computed at a genuine fixed point."""
    from make_ch5_cycle_market import build
    coarse = ns.solve(build(N=2, L=6.0, nodes=6, unit_range=3.0).build(), tol=1e-8).check()
    assert not coarse.resolution_ok and "UNDER-RESOLVED" in coarse.summary()
    d = build(N=2, L=6.0, nodes=14, unit_range=3.0).to_dict(); d["ties"] = []
    ra = ns.solve(ns.Model.from_dict(d), tol=1e-8).check(); rm = ns.solve(ns.Model.from_dict(d), {"variable": "maps"}, tol=1e-8).check()
    assert abs(ra.costs["firm0"] - rm.costs["firm0"]) < 1e-3
    st = rm.stability(); assert st["fixed_point_residual"] < 1e-6 and st["radius"] > 0 and "stability" in rm.to_dict()
    # the two-firm market's negative curvature is the window's truncation of the lagged loss terms (README): the
    # same direction is positive on a window longer by two lags, so the check passes and reports the edge
    so = rm.second_order["firm0"]; assert so["ok"] and so["edge"] and so["min"] < -1e-4 and so["embedded"] > 0 and "NOT A MINIMUM" not in rm.summary()
    assert "window edge" in rm.summary() and any(d["name"] == "second_order_edge:firm0" for d in rm.diagnose())


def test_stability_of_the_chapter_3_game_and_finite_engine():
    r = ns.solve(example_path("ch3_two_player")).check(); s = r.stability()
    assert s["stable"] and s["radius"] < 1.0 and s["fixed_point_residual"] < 1e-8
    df = example_dict("ch1_two_player_finite"); df.setdefault("numerics", {})["nodes"] = 6
    rf = ns.solve(ns.Model.from_dict(df)).check(); sf = rf.stability()
    assert sf["stable"] and sf["radius"] < 1.0


def test_a_shrinking_negative_curvature_is_the_grid_not_a_saddle():
    """kyle_back_prior reports NOT A MINIMUM on trader1 at every resolution while the number it reports
    goes to zero (-1.45e-02 at 8 nodes to -5.63e-03 at 24, about n^-0.85): the offending direction sits
    on the diagonal a = t and alternates in sign between neighbouring age nodes, so it is the quadrature's
    and not a strategy.  refine() records the finer grid's curvature and a shrinking one clears the flag."""
    import noisestate as ns
    r = ns.solve(example_path("kyle_back_prior"))
    assert not r.second_order["trader1"]["ok"] and r.second_order["trader1"]["min"] < -1e-3
    assert any("NOT A MINIMUM" in f for f in r.status["flags"])
    rep = r.refine()
    cv = rep["second_order"]["trader1"]
    assert cv["shrinking"] and cv["min"] < cv["fine_min"] < 0        # less negative on the finer grid
    assert not any("NOT A MINIMUM" in f for f in r.status["flags"])  # the verdict is overturned
    assert any(d["name"] == "second_order_grid:trader1" and d["ok"] for d in r.diagnose())
