import noisestate as ns
from noisestate.diagnostics import Status
from helpers import example_path, example_dict

def test_resolution_flag_and_stability_on_the_two_firm_market():
    """At 6 nodes per panel the two-firm cycle market is under-resolved and the result says so; the
    map and action-kernel paths then disagree.  At 14 nodes they agree and the stability report is
    computed at a genuine fixed point."""
    from make_ch5_cycle_market import build
    coarse = ns.solve(build(N=2, L=6.0, nodes=6, unit_range=3.0).build(), tol=1e-8).require_converged()
    assert coarse.diagnostics.statuses["resolution"] is Status.FAILED and "UNDER-RESOLVED" in coarse.summary()
    d = build(N=2, L=6.0, nodes=14, unit_range=3.0).to_dict(); d["ties"] = []
    ra = ns.solve(ns.Model.from_dict(d), tol=1e-8).require_converged(); rm = ns.solve(ns.Model.from_dict(d), {"variable": "maps"}, tol=1e-8).require_converged()
    assert abs(ra.costs["firm0"] - rm.costs["firm0"]) < 1e-3
    st = rm.stability(); assert st["fixed_point_residual"] < 1e-6 and st["radius"] > 0 and "stability" in rm.to_dict()
    # the two-firm market's negative curvature is the window's truncation of the lagged loss terms (README): the
    # same direction is positive on a window longer by two lags, so the check passes and reports the edge
    so = rm.second_order["firm0"]; assert so["ok"] and so["edge"] and so["min"] < -1e-4 and so["embedded"] > 0 and "NOT A MINIMUM" not in rm.summary()
    assert "window edge" in rm.summary() and any(d["name"] == "second_order_edge:firm0" for d in rm.diagnostics.rows)


def test_stability_of_the_chapter_3_game_and_finite_engine():
    r = ns.solve(example_path("ch3_two_player")).require_converged(); s = r.stability()
    assert s["stable"] and s["radius"] < 1.0 and s["fixed_point_residual"] < 1e-8
    df = example_dict("ch1_two_player_finite"); df.setdefault("numerics", {})["nodes"] = 6
    rf = ns.solve(ns.Model.from_dict(df)).require_converged(); sf = rf.stability()
    assert sf["stable"] and sf["radius"] < 1.0


def test_a_shrinking_negative_curvature_is_the_grid_not_a_saddle():
    """kyle_back_prior reports NOT A MINIMUM on trader1 at every resolution while the number it reports
    goes to zero (-1.45e-02 at 8 nodes to -5.63e-03 at 24, about n^-0.85): the offending direction sits
    on the diagonal a = t and alternates in sign between neighbouring age nodes, so it is the quadrature's
    and not a strategy.  refine() records the finer grid's curvature and a shrinking one clears the flag."""
    import noisestate as ns
    r = ns.solve(example_path("kyle_back_prior"))
    assert not r.second_order["trader1"]["ok"] and r.second_order["trader1"]["min"] < -1e-3
    assert any("NOT A MINIMUM" in f for f in r.diagnostics.flags)
    rep = r.refine()
    cv = rep["second_order"]["trader1"]
    assert cv["shrinking"] and cv["min"] < cv["fine_min"] < 0        # less negative on the finer grid
    assert not any("NOT A MINIMUM" in f for f in r.diagnostics.flags)  # the verdict is overturned
    assert any(d["name"] == "second_order_grid:trader1" and d["ok"] for d in r.diagnostics.rows)


def test_a_long_window_finds_a_second_branch_that_only_the_guard_refuses():
    """docs/limits.md, "a stationary window much longer than the kernel's support".  ch3_two_player at
    64 nodes solves the equilibrium up to L = 15 (the state kernel decays to 3e-08 of its peak, cost
    0.4273); at L = 18 the solve still converges to its tolerance, onto a fixed point of the truncated
    problem whose kernel has not decayed at the edge and whose cost is seven times larger.  check()
    passes on it because it did converge -- only the window guard, and so require_ok(), refuses it."""
    import numpy as np, pytest, noisestate as ns
    base = ns.load(example_path("ch3_two_player"))

    good = ns.solve(base.with_horizon(window=15.0).with_numerics(nodes=64))
    a = np.asarray(good.ages); K = np.asarray(good.kernel("X"))
    assert good.diagnostics.assess().accepted and abs(good.costs["player1"] - 0.427295) < 1e-5
    assert np.abs(K[a > 0.95 * 15.0]).max() / np.abs(K).max() < 1e-6      # decayed by the edge

    spurious = ns.solve(base.with_horizon(window=18.0).with_numerics(nodes=64))
    assert spurious.converged and spurious.require_converged() is spurious            # it really did converge
    assert spurious.costs["player1"] > 3.0                                # and to the wrong thing
    Ks = np.asarray(spurious.kernel("X")); asp = np.asarray(spurious.ages)
    assert np.abs(Ks[asp > 0.95 * 18.0]).max() / np.abs(Ks).max() > 0.5   # a closed loop that does not stabilise
    assert not spurious.diagnostics.assess().accepted and any("WINDOW TOO SHORT" in f for f in spurious.diagnostics.flags)
    #  a guard failure is NOT a convergence failure: the solve converged.  The two are siblings,
    #  so `except ConvergenceError` must not catch this one.
    assert not issubclass(ns.DiagnosticsError, ns.ConvergenceError)
    assert issubclass(ns.DiagnosticsError, ns.ResultValidationError)
    with pytest.raises(ns.DiagnosticsError, match="converged, but"):
        spurious.require_ok()

    # both fixed points exist at L = 18: a cold start lands on the spurious one, and continuation in
    # the window reaches the equilibrium.  What separates them is best-response stability -- Anderson
    # and the Newton polish are root finders and will sit on a fixed point naive adjustment would flee.
    wider = base.with_horizon(window=18.0).with_numerics(nodes=64)
    warm = ns.solve(wider, init=ns.StationarySolver(wider).interpolate_maps(good)).require_ok()
    assert abs(warm.costs["player1"] - 0.427295) < 1e-5
    assert warm.stability()["radius"] < 0.9 < 1.0 < spurious.stability()["radius"]

    # the radius is computed unasked where it discriminates: the window guard failing is the only place
    # it is worth its best responses, and there it separates a branch from a truncation
    report = lambda r: getattr(r, "stability_report", None)     # only set once stability() has run
    assert report(spurious) is not None and report(spurious)["radius"] > 1.0
    assert "UNSTABLE" in spurious.summary() and "WINDOW TOO SHORT" in spurious.summary()
    assert report(good) is None                                 # a window that passes pays nothing
    shipped = ns.solve(example_path("ch3_two_player"))           # window 3: flagged, but a truncation
    assert report(shipped) is not None and report(shipped)["radius"] < 1.0
