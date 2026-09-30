import noisestate as ns
from noisestate import engines
from noisestate.diagnostics import Status
from helpers import example_path, example_dict

def test_resolution_flag_and_stability_on_the_two_firm_market():
    """At 6 nodes per panel the two-firm cycle market is under-resolved and the result says so; the
    map and action-kernel paths then disagree.  At 14 nodes they agree and the stability report is
    computed at a genuine fixed point."""
    from make_ch5_cycle_market import build
    coarse = ns.solve(build(N=2, L=6.0, nodes=6, unit_range=3.0), tol=1e-8).require_converged()
    assert coarse.diagnostics.statuses["resolution"] is Status.FAILED and "UNDER-RESOLVED" in coarse.summary()
    d = build(N=2, L=6.0, nodes=14, unit_range=3.0).to_dict(); d["ties"] = []
    ra = ns.solve(ns.Model.from_dict(d), tol=1e-8).require_converged(); rm = ns.solve(ns.Model.from_dict(d), {"variable": "maps"}, tol=1e-8).require_converged()
    assert abs(ra.costs["firm0"] - rm.costs["firm0"]) < 1e-3
    st = rm.stability(); assert st.fixed_point_residual < 1e-6 and st.radius > 0 and "stability" in rm.to_dict()
    # the two-firm market's negative curvature is the window's truncation of the lagged loss terms (README): the
    # same direction is positive on a window longer by two lags, so the check passes and reports the edge
    so = rm.second_order["firm0"]; assert so["ok"] and so["edge"] and so["min"] < -1e-4 and so["embedded"] > 0 and "NOT A MINIMUM" not in rm.summary()
    assert "window edge" in rm.summary() and any(d["name"] == "second_order_edge:firm0" for d in rm.diagnostics.rows)


def test_stability_of_the_chapter_3_game_and_finite_engine():
    r = ns.solve(example_path("ch3_two_player")).require_converged(); s = r.stability()
    assert s.stable and s.radius < 1.0 and s.fixed_point_residual < 1e-8
    df = example_dict("ch1_two_player_finite"); df.setdefault("numerics", {})["nodes"] = 6
    rf = ns.solve(ns.Model.from_dict(df)).require_converged(); sf = rf.stability()
    assert sf.stable and sf.radius < 1.0


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
    cv = rep.curvature["trader1"]
    assert cv["shrinking"] and cv["min"] < cv["fine_min"] < 0        # less negative on the finer grid
    assert not any("NOT A MINIMUM" in f for f in r.diagnostics.flags)  # the verdict is overturned
    assert any(d["name"] == "second_order_grid:trader1" and d["ok"] for d in r.diagnostics.rows)


def test_long_window_action_recovery_reaches_an_equilibrium():
    """The former L=18 'branch' had O(1) profitable deviations despite a 1e-10
    action residual. Recovery must agree with a direct map solve, even without diagnostics."""
    import numpy as np
    base = ns.load(example_path("ch3_two_player"))
    wider = base.with_stationary(18.0).with_numerics(nodes=64)
    reference = ns.solve(wider, variable="maps", diagnostics=False).require_converged()
    log = []
    recovered = ns.solve(wider, diagnostics=False, progress=log.append).require_converged()
    assert "map recovery" in recovered.message
    assert abs(recovered.costs["player1"] - 0.427294931719) < 1e-10
    assert abs(recovered.costs["player1"] - reference.costs["player1"]) < 1e-10
    solver = engines.stationary(wider)
    g = solver.pack(recovered.maps)
    assert np.linalg.norm(solver.pack(solver.response_map(recovered.maps)) - g) / max(1, np.linalg.norm(g)) < 1e-10
    assert [x["evaluation"] for x in log] == list(range(1, recovered.evaluations + 1))
    assert log[-1]["phase"].startswith("maps ")
    assert all(x["seconds"] <= y["seconds"] for x, y in zip(log, log[1:]))
    assert abs(log[-1]["residual"] - recovered.residual) < 1e-15

    # Insufficient recovery budget is a convergence failure, even though the action
    # iteration reached its tolerance. No uncounted best responses or fresh budget.
    action_evals = next(x["evaluation"] for x in log if x["phase"].startswith("maps ")) - 1
    for budget in (action_evals, action_evals + 2):
        events = []
        short = ns.solve(wider, max_evaluations=budget, diagnostics=False, progress=events.append)
        assert not short.converged and short.evaluations == len(events) == budget
        assert "action recovery error" in short.message and "evaluation budget" in short.message
        assert short.residual > 1e-8


def test_window_checks_do_not_request_stability(monkeypatch):
    """A spectral radius is expensive and does not certify equilibrium or window accuracy."""
    original = ns.Result.stability
    calls = []
    def counted(self, **kw):
        calls.append(self)
        return original(self, **kw)
    monkeypatch.setattr(ns.Result, "stability", counted)
    result = ns.solve(example_path("ch3_two_player"))
    assert result.diagnostics.statuses["window"] is Status.FAILED
    assert not calls and getattr(result, "stability_report", None) is None
    requested = ns.solve(example_path("ch3_two_player"), stability=True)
    assert calls == [requested] and requested.stability_report.radius < 1


def test_the_arnoldi_finds_a_cluster_of_equal_moduli():
    """stability()'s eigensolver (results._arnoldi_dominant) against numpy's eig: on a generic matrix it stops early with
    the dominant pair to 1e-9, and on one whose four largest eigenvalues share a modulus (Chapter 6's opaque market's
    Jacobian, on which ARPACK ran out of budget and the power iteration's fallback read 3.5% high) it runs to the
    Krylov space's dimension and returns that modulus."""
    import numpy as np
    from noisestate.results import _arnoldi_dominant
    rng = np.random.default_rng(5)
    n = 60
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    lam = np.linalg.eigvals(A); lam = lam[np.argsort(-np.abs(lam))]
    count = [0]
    def mv(x):
        count[0] += 1
        return A @ x
    v = rng.standard_normal(n)
    got = _arnoldi_dominant(mv, v, A @ v, 2, 1e-9)
    assert abs(np.abs(got).max() - np.abs(lam[0])) < 1e-9 and count[0] < n
    th = np.exp(1j * np.array([0.3, -0.3, 2.0, -2.0]))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    blocks = np.zeros((n, n))
    for i, t in enumerate(th[::2]):
        blocks[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 1.08 * np.array([[t.real, -t.imag], [t.imag, t.real]])
    blocks[4:, 4:] = np.diag(np.linspace(-0.9, 0.9, n - 4))
    B = Q @ blocks @ Q.T
    got = _arnoldi_dominant(lambda x: B @ x, v, B @ v, 2, 1e-6)
    assert abs(np.abs(got).max() - 1.08) < 1e-9
