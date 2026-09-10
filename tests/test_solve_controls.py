"""Bounding and watching a solve (2026-09-05 audit): max_evaluations, deadline, progress, diagnostics=False,
the stability budget, the Newton-Krylov inner budget, and a non-finite residual."""
import json, os, warnings, numpy as np, pytest
import noisestate as ns
from noisestate import engines
from noisestate.diagnostics import Status
from noisestate.accel import solve_fixed_point
from noisestate.cli import main
from noisestate.sweep import sweep
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")
CH3 = os.path.join(EX, "ch3_two_player.yaml")


def _ch3(**horizon):
    d = ns.read_yaml(CH3); d["numerics"] = {**d.get("numerics", {}), **{k: horizon.pop(k) for k in list(horizon) if k in ("nodes", "unit", "unit_range", "breakpoints")}}; d["horizon"].update(horizon)
    return ns.Model.from_dict(d)


def test_evaluation_budget_returns_the_best_iterate_unconverged():
    res = ns.solve(_ch3(), max_evaluations=3)
    assert not res.converged and res.evaluations == 3 and "evaluation budget" in res.message and "max_evaluations=3" in res.message
    assert "NOT converged" in res.summary() and "evaluation budget" in res.summary()
    row = [d for d in res.diagnostics.rows if d["name"] == "converged"][0]; assert row["ok"] is False and "evaluation budget" in row["advice"]
    assert res.solve_kw["max_evaluations"] == 3 and np.isfinite(res.costs["player1"])
    with pytest.raises(ns.ConvergenceError):
        res.require_converged()
    assert res.refine()["converged"]                                     # the bound is this solve's, not the refinement's
    # the budget counts the polish as well, and the best iterate comes back, not the last
    d = ns.read_yaml(CH3); d["params"]["p1"] = d["params"]["p2"] = 1e200        # Anderson stalls at 2.6e-7, then a long polish
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = []; r = ns.solve(ns.Model.from_dict(d), max_evaluations=100, progress=log.append)
    assert not r.converged and r.evaluations == len(log) == 100 and "newton polish" in r.message and "evaluation budget" in r.message
    assert {x["phase"] for x in log} == {"anderson", "newton"} and r.residual == min(x["residual"] for x in log)
    with pytest.raises(ValueError, match="at least 1"):
        ns.solve(_ch3(), max_evaluations=0)


def test_deadline_zero_returns_after_one_evaluation():
    res = ns.solve(_ch3(), deadline=0)
    assert not res.converged and res.evaluations == 1 and "deadline" in res.message and np.isfinite(res.residual)
    with pytest.raises(ValueError, match="deadline"):
        ns.solve(_ch3(), deadline=-1)


def test_progress_is_called_per_evaluation_and_can_cancel():
    log = []; res = ns.solve(_ch3(), progress=log.append)
    assert res.converged and len(log) == res.evaluations and [x["evaluation"] for x in log] == list(range(1, res.evaluations + 1))
    assert all(set(x) == {"evaluation", "residual", "phase", "seconds"} for x in log)
    assert all(x["phase"] == "anderson" for x in log) and np.isclose(log[-1]["residual"], res.residual, rtol=1e-12)
    assert all(log[i]["seconds"] <= log[i + 1]["seconds"] for i in range(len(log) - 1)) and log[-1]["seconds"] <= res.seconds

    class Cancelled(Exception):
        pass

    def cancel(info):
        if info["evaluation"] == 2:
            raise Cancelled()
    with pytest.raises(Cancelled):
        ns.solve(_ch3(), progress=cancel)
    log = []; res = ns.solve(_ch3(), start="coarse", progress=log.append)   # a coarse start reports its phase, on the one clock
    phases = [x["phase"] for x in log]
    assert phases[0] == "coarse anderson" and phases[-1] == "anderson" and res.converged and "coarse start" in res.message
    assert all(log[i]["seconds"] <= log[i + 1]["seconds"] for i in range(len(log) - 1))
    res = ns.solve(_ch3(), start="coarse", max_evaluations=2)                # both solves bounded
    assert not res.converged and res.evaluations == 2 and res.message.startswith("coarse start: 2 evaluations")


def test_diagnostics_off_skips_the_checks_and_their_best_responses(monkeypatch):
    S = engines.solver(_ch3(nodes=96, window=10.0)); full = S.solve(); w = full.maps
    off = S.solve(init=w, diagnostics=False)
    assert off.converged and off.second_order == {} and off.foc == {} and off.representation_error == {} and off.diagnostics.statuses["resolution"] is not Status.PASSED
    assert full.second_order and full.foc and (full.diagnostics.statuses["resolution"] is Status.PASSED)
    assert all(abs(off.costs[k] - full.costs[k]) < 1e-9 for k in full.costs) and off.solve_kw["diagnostics"] is False
    assert "diagnostics skipped" in off.summary() and any(d["name"] == "diagnostics" and d["ok"] is None for d in off.diagnostics.rows)
    assert "diagnostics skipped" not in full.summary() and "diagnostics" not in [d["name"] for d in full.diagnostics.rows]
    assert json.loads(json.dumps(off.to_dict()))["options"]["solve"]["diagnostics"] is False
    # the checks are the best responses after the fixed point (one per agent, with the decomposition, which
    # halves the time of this warm-started solve); with diagnostics=False none is made
    calls = []; best_response = type(S).best_response

    def counted(self, agent, maps, want_decomp=False):
        calls.append(want_decomp)
        return best_response(self, agent, maps, want_decomp)
    monkeypatch.setattr(type(S), "best_response", counted)
    r = S.solve(init=w, diagnostics=False); assert len(calls) == 2 * r.evaluations and not any(calls)
    calls.clear(); r = S.solve(init=w); assert len(calls) == 2 * r.evaluations + 2 and calls[-2:] == [True, True]
    # the cell engine (no checks of its own) accepts the option as well
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d["horizon"] = {"kind": "finite", "T": 1.0}; d["numerics"] = {"engine": "cells", "nodes": 12}
    assert ns.solve(ns.Model.from_dict(d), diagnostics=False).converged


def test_non_finite_best_response_stops_with_a_clear_error():
    # a map that returns a non-finite value stops the iteration at that evaluation, whichever it is
    calls = [0]

    def F(x):
        calls[0] += 1
        return np.full_like(x, np.nan) if calls[0] == 3 else -0.5 * x
    with pytest.raises(RuntimeError, match="evaluation 3"):
        solve_fixed_point(F, np.ones(4), tol=1e-12, max_newton=0)


def test_newton_polish_inner_budget_holds():
    """scipy's newton_krylov replaces LGMRES's outer loop by the Newton steps, so the inner_maxiter it was
    given bounded nothing; inner_m does: a step is at most 15 fresh Krylov vectors, plus the min(step - 1, 10)
    directions carried from earlier steps (scipy does not keep their products: one evaluation each) and the
    line search (one evaluation, two when it backtracks), with one evaluation at the start of the polish and
    one for the residual it reports."""
    def polish(A, b, steps):
        calls = [0]

        def F(x):
            calls[0] += 1
            return A @ x - b
        z, rn, ev, ok, msg = solve_fixed_point(F, np.zeros(len(b)), tol=1e-14, anderson_iters=1, max_newton=steps)
        assert ev == calls[0] and "newton polish" in msg
        return ev - 2                                            # Anderson made two evaluations before its cap

    def budget(steps, line_search):
        return 2 + sum(15 + min(j, 10) + line_search for j in range(steps))
    rng = np.random.default_rng(1); n = 120
    A = rng.standard_normal((n, n)) / np.sqrt(n) + 2.0 * np.eye(n); b = rng.standard_normal(n)
    # 48: the first step's inner loop meets its 1e-3 forcing tolerance after 10 vectors, the later steps run
    # all 15; the old inner_m=30 made 73
    assert budget(3, 1) - 14 <= polish(A, b, 3) <= budget(3, 2)
    # an ill-conditioned system runs every step's 15 fresh vectors, so the count is the formula itself, out
    # to twelve steps where ten directions are carried: 53 and 259 (98 and 439 with inner_m=30)
    rng = np.random.default_rng(2); n = 200
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))); A = Q @ np.diag(np.logspace(-4, 0, n)) @ Q.T; b = rng.standard_normal(n)
    assert budget(3, 1) <= polish(A, b, 3) <= budget(3, 2)
    assert budget(12, 1) <= polish(A, b, 12) <= budget(12, 2)


def test_stability_budget_bounds_the_best_response_rounds(monkeypatch):
    res = ns.solve(_ch3()).require_converged()
    monkeypatch.setattr(ns.Result, "STABILITY_MAX_EVALUATIONS", 5)
    st = res.stability()
    assert st["evaluations"] <= 5 and "evaluation budget" in st["method"] and 0 < st["radius"] < 1 and "power iteration" in res.summary()
    monkeypatch.setattr(ns.Result, "STABILITY_MAX_EVALUATIONS", 200)
    full = res.stability()
    assert full["method"] == "arnoldi" and full["evaluations"] < 200 and abs(full["radius"] - st["radius"]) < 0.1


def test_cli_and_sweep_forward_the_bounds(tmp_path, capsys):
    assert main(["solve", CH3, "--max-evaluations", "2"]) == 1
    assert "evaluation budget" in capsys.readouterr().out
    assert main(["solve", CH3, "--deadline", "0"]) == 1
    assert "deadline" in capsys.readouterr().out
    assert main(["solve", CH3, "--max-evaluations", "0"]) == 2
    out = tmp_path / "sw.json"
    assert main(["sweep", CH3, "p2", "3,5", "-o", str(out), "--max-evaluations", "2"]) == 1
    rows = json.load(open(out))          # JSON: dicts, not SweepPoints
    assert [r["evaluations"] for r in rows] == [2, 2] and not any(r["converged"] for r in rows)
    rows = sweep(CH3, "p2", [3.0, 5.0], solve_kw={"max_evaluations": 2}); assert all(r.evaluations == 2 and not r.converged for r in rows)


def test_lead_term_under_a_discount_neither_overflows_nor_is_silent():
    """The stationary lead term weights the flows before t that read the quantity after t by exp(rho v),
    v within the lead: taken on every node of the window it overflowed at rho * window > 709 (inf times
    the mask's zero is NaN in every first-order condition).  It is now taken within the lead only, so the
    model compiles and solves cleanly; and a discount that makes exp(rho tau) large is announced at
    compile, since those weights dominate the best-response system (docs/limits.md)."""
    d = ns.read_yaml(CH3); d["agents"]["player1"]["loss"].append([0.1, "D1", "X@-0.5"]); d["numerics"].update(unit=0.5, nodes=8)
    d["horizon"]["discount"] = 0.5                                        # exp(0.25) = 1.3: nothing to say
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mild = ns.solve(ns.Model.from_dict(d))
    assert mild.converged and np.isfinite(mild.costs["player1"])
    d["horizon"]["discount"] = 300.0                                      # rho * window = 900, exp(rho tau) = 1e65
    with pytest.warns(UserWarning, match="lead X@-0.5 under the discount rate 300"):
        S = engines.stationary(ns.Model.from_dict(d))
    agent = S.model.agents[0]; Ru = np.zeros((len(S.c.prim) * S.c.N, 1)); Ru[S.c.block("X")] = 1.0
    assert np.isfinite(S._lead_term(agent, Ru[:, 0], "X", -0.5)).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = S.solve(max_evaluations=60)                                 # clean either way: the myopic limit converges in 3
    assert np.isfinite(res.residual) and (not res.converged or np.isfinite(res.costs["player1"]))
    d["horizon"]["discount"] = 20.0                                       # exp(rho tau) = 2e4: the fixed point runs away, cleanly
    with pytest.warns(UserWarning, match="lead X@-0.5"):
        res = ns.solve(ns.Model.from_dict(d), max_evaluations=60)
    assert not res.converged and np.isfinite(res.residual) and "NOT converged" in res.summary()


def test_scipy_jacobian_failure_in_the_polish_is_a_non_convergence_not_a_value_error():
    """newton_krylov raises ValueError("Jacobian inversion yielded zero vector") when its Krylov step is
    zero; the polish reports that as non-convergence with the Anderson iterate.  A ValueError raised by the
    map itself (the singular best-response system) still propagates."""
    def F(x):                                         # a constant residual: no fixed point, a zero Jacobian
        return np.full_like(x, 0.5)
    z, rn, ev, ok, msg = solve_fixed_point(F, np.zeros(3), tol=1e-12, anderson_iters=5, max_newton=3)
    assert not ok and "newton polish failed" in msg and np.isfinite(z).all()
    calls = [0]

    def G(x):
        calls[0] += 1
        if calls[0] > 5:
            raise ValueError("the best-response system of a is singular")
        return np.full_like(x, 0.5)
    with pytest.raises(ValueError, match="singular"):
        solve_fixed_point(G, np.zeros(3), tol=1e-12, anderson_iters=5, max_newton=3)
