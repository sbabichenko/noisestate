"""The cell engine (extras/cells.py), the first-order uniform-cell cross-check that left the package in 1.1:
its own behaviour, and the cross-checks that pin the spectral engine against an independent discretisation.
The package's tests build these models; the engine is constructed here directly, FiniteSolver(model), on a
finite horizon with numerics.nodes cells.

    .venv/bin/python -m pytest -q extras/test_cells.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "examples"), os.path.join(HERE, "..", "tests"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import noisestate as ns                                             # noqa: E402
from noisestate import engines                                      # noqa: E402
from noisestate._settings import Settings                           # noqa: E402
from noisestate.diagnostics import Policy, Status                   # noqa: E402
from cells import FiniteSolver                                      # noqa: E402
from helpers import example                                         # noqa: E402

EX = os.path.join(HERE, "..", "examples")


def cells(model, nodes=None, settings=None, **solve_kw):
    """The cell engine on a finite model (a Model, a dict or a path) at `nodes` cells (the model's own when None)."""
    m = ns.as_model(model)
    if nodes is not None:
        m = m.with_numerics(nodes=nodes)
    return FiniteSolver(m, settings=settings).solve(**solve_kw)


def ch1(nodes):
    return ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).with_numerics(nodes=nodes)


# ------------------------------------------------------------------------------ the engine itself

@pytest.fixture(scope="module")
def ch1_cells():
    return cells(ch1(12))


def test_cell_engine_kernel_without_a_shock_is_the_stack_over_shocks(ch1_cells):
    K = ch1_cells.kernel("X"); N = ch1_cells.compiled.N
    assert K.shape == (N, N, len(ch1_cells.shocks))
    for k, ch in enumerate(ch1_cells.shocks):
        assert np.array_equal(K[..., k], ch1_cells.kernel("X", ch))
    assert ch1_cells.kernel("D1").shape == (N, N, len(ch1_cells.shocks))


def test_cell_engine_mean_solve_refuses_a_singular_mean_system():
    from ch1_mean_sweep import model as ch1_targets
    d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}
    d["numerics"] = {"nodes": 8}
    with pytest.raises(ValueError, match="the mean system is singular"):
        cells(d, settings={"mean_rcond": 1.0})                     # every system fails a threshold of 1


def test_ch1_finite_converges_and_matches_cost_to_first_order():
    res = cells(ch1(24))                                            # the first-order cell scheme
    assert res.converged and res.residual < 1e-8
    # spec_ch1 (spectral, 16x16 nodes) reports Jvar1 = 0.39664911 for this game
    assert abs(res.costs["player1"] - 0.39664911) < 0.01
    assert abs(res.costs["player1"] - res.costs["player2"]) < 1e-9        # symmetric game
    K = res.kernel("D1", "w1"); assert np.all(np.triu(K) == 0)            # strictly causal kernels


def test_the_kernel_is_an_n_by_n_matrix_read_at_the_nearest_cell():
    from expr_examples import EXAMPLES
    res = cells(EXAMPLES["ch1_two_player_finite"](), nodes=8, max_evaluations=2, diagnostics=False)
    kc = res.kernel("X", "w0")
    assert kc.shape == (8, 8) and "nearest" in kc.note
    assert kc.at(0.5, 0.2) == kc[np.abs(kc.axes["time"] - 0.5).argmin(), np.abs(kc.axes["time"] - 0.2).argmin()]


def test_a_cell_result_reads_like_the_others(ch1_cells):
    """A consumer reads a kernel with its axes without knowing the engine; the cell kernel is (N, N)."""
    r = ch1_cells
    K = r.kernel("X", "w1")
    node_axes = {k: v for k, v in r.axes.items() if k != "maps"}
    assert K.ndim == 2 and all(len(v) == K.shape[0] for v in node_axes.values())
    assert "time" in node_axes and "shock_time" in node_axes and r.times is not None and len(r.times) == len(r.paths["means"]["X"])
    assert r.axes["maps"]["player1"]["y1"]["map_time"].shape[0] == r.maps["player1"].shape[2]
    assert isinstance(r, ns.Result) and r.kind == "finite_cells"
    assert r.diagnostics.statuses["converged"] is Status.PASSED
    assert r.cost_kind.startswith("discounted") and set(r.cost_parts["player1"]) == {"variance", "mean"}


def test_the_zero_means_of_a_model_without_drivers():
    d = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}
    rc = cells(d, nodes=20).require_converged()
    assert all(np.array_equal(v, np.zeros(20)) for v in rc.means.values()) and rc.cost_parts["player1"]["mean"] == 0.0


def test_a_constant_drift_gives_mean_paths_on_the_cells():
    d = {"shocks": ["w0", "w1"], "states": {"X": {"drift": {"X": -1.0, "D": 1.0, "const": 0.3}, "noise": {"w0": 1.0}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w1": 1.0}}},
                          "loss": [[1.0, "X", "X"], [-2.0, "X"], [1.0, "D", "D"]]}},
         "horizon": {"kind": "finite", "T": 1.0}, "numerics": {"nodes": 4}}
    assert cells(d).require_converged().means["D"].shape == (4,)


def test_diagnostics_off_is_accepted():
    d = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}
    assert cells(d, nodes=12, diagnostics=False).converged


def test_every_predicate_name_returns_a_real_bool():
    res = cells(example("ch1_two_player_finite"), nodes=8, diagnostics=False)
    checked = 0
    for obj in (res, res.model):
        for n in dir(obj):
            if n.startswith(("is_", "has_", "drives_")) and not n.startswith("_"):
                value = getattr(obj, n)
                assert isinstance(value, bool), f"{type(obj).__name__}.{n} is {type(value).__name__}"
                checked += 1
    assert checked


def test_settings_reach_the_cell_engine():
    m = ch1(8)
    assert FiniteSolver(m, settings=Settings(map_ridge=1e-9)).settings.map_ridge == 1e-9
    assert FiniteSolver.DENSE_MAX == 200 and FiniteSolver.KRYLOV_RTOL == 1e-12


def test_a_transition_is_refused():
    d = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).to_dict()
    t = ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "x.yaml"}}})
    with pytest.raises(ValueError, match="spectral finite engine only"):
        FiniteSolver(t)


def test_wrong_grid_warm_start_is_an_error():
    rc = cells(ch1(8))
    with pytest.raises(ValueError, match="different grid"):
        FiniteSolver(ch1(16)).solve(start_from=rc.maps)


def test_dense_branch_with_delays_and_refine_doubles():
    d = ns.load(os.path.join(EX, "ch1_delayed_finite.yaml")).to_dict()
    r = cells(d, nodes=16).require_converged()
    assert r.diagnostics.statuses["resolution"] is Status.UNSUPPORTED          # the cell engine cannot compute it
    assert r.to_dict()["assessment"]["statuses"]["resolution"] == "unsupported"
    r.refine(); assert r.refinement.nodes == 32                             # doubled: lags stay aligned


def test_refine_reports_without_a_verdict():
    r = cells(ch1(12)); r.refine()
    assert r.refinement.resolved is None and "NOT RESOLVED" not in r.summary() and r.refinement.nodes == 24


def test_the_cell_plot_draws(tmp_path, ch1_cells):
    pytest.importorskip("matplotlib")
    path = tmp_path / "cells.png"
    ch1_cells.plot(str(path))
    assert path.exists() and path.stat().st_size > 1000


@pytest.mark.parametrize("nodes", [6, 24])
def test_a_singular_best_response_system_raises(nodes):
    """cells 6: the dense branch; 24: the Krylov branch."""
    import warnings
    from test_singular_foc import _kyle_back
    with pytest.warns(UserWarning, match="control P has no strictly positive quadratic term"):
        m = ns.Model.from_dict(_kyle_back("finite", nodes, [[-2.0, "P", "V"]]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="the best-response system of market_maker is singular"):
            FiniteSolver(m).solve()
        res = cells(_kyle_back("finite", nodes, [[1.0, "P", "P"], [-2.0, "P", "V"]])).require_converged()
    assert res.converged and abs(res.costs["market_maker"]) > 0.01


# ---------------------------------------------------- a check the engine cannot run is not a check passed

def _diag(nodes=8):
    from test_diagnostics import model
    return cells(model("finite", nodes=nodes))


def test_an_engine_that_cannot_run_a_required_check_is_refused_not_excused():
    """The cell engine computes neither a representation error nor a second-order form; before the
    diagnostics contract it passed the whole verdict while running two fewer checks than the others."""
    res = _diag()
    st = res.diagnostics.statuses
    assert st["resolution"] is Status.UNSUPPORTED and st["second_order"] is Status.UNSUPPORTED
    verdict = res.diagnostics.assess()
    assert not verdict.accepted and verdict.uncomputed == ("resolution", "second_order")
    assert {b.status for b in verdict.blocking} == {Status.UNSUPPORTED}
    with pytest.raises(ns.DiagnosticsError, match="cannot compute"):
        res.require_ok()


def test_a_weaker_policy_is_legitimate_and_must_be_named():
    res = _diag()
    assert not res.diagnostics.assess(Policy.PUBLICATION).accepted
    assert res.diagnostics.assess(Policy.EXPLORATORY).accepted
    assert res.require_ok(Policy.EXPLORATORY) is res
    assert res.diagnostics.assess(Policy.EXPLORATORY).policy == "exploratory"


def test_a_check_this_engine_cannot_build_is_unsupported_not_inapplicable():
    """Both checks are meaningful for the model, so they APPLY and the engine cannot run them; the same model
    on the spectral engine computes both."""
    from test_diagnostics import model
    res = _diag(nodes=10)
    assert res.diagnostics.statuses["second_order"] is Status.UNSUPPORTED
    assert res.diagnostics.statuses["resolution"] is Status.UNSUPPORTED
    with pytest.raises(ns.DiagnosticsError, match="second_order"):
        res.require_ok()
    assert res.require_ok(Policy.EXPLORATORY) is res
    assert ns.solve(model("finite", nodes=10)).diagnostics.statuses["second_order"] is Status.PASSED


def test_the_payload_carries_every_status_and_the_policy_that_judged_them():
    payload = _diag().to_dict()["assessment"]
    assert payload["policy"] == "publication" and payload["accepted"] is False
    assert payload["statuses"]["resolution"] == "unsupported"
    assert payload["uncomputed"] == ["resolution", "second_order"]


def test_the_cli_says_cannot_be_checked_rather_than_failed(capsys):
    """--require-ok on a result whose checks the engine cannot run: "cannot be checked here", not "failed", and
    a weaker standard is accepted once it is named (cli._exit_code, what `noisestate solve` returns)."""
    from noisestate.cli import _exit_code
    res = cells(example("ch1_two_player_finite"), nodes=10)
    assert _exit_code(res, SimpleNamespace(require_ok=True, policy="publication")) == 1
    err = capsys.readouterr().err
    assert "cannot be checked here" in err and "second_order" in err
    assert "failed diagnostic checks" not in err
    assert "no setting will change it" in err                  # it is the engine, not the tuning
    assert _exit_code(res, SimpleNamespace(require_ok=True, policy="exploratory")) == 0


# ------------------------------------------------------------- cross-checks against the spectral engine

@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_cell_engine_is_first_order_and_its_richardson_pair_matches_the_closed_form(rho):
    """48 and 96 cells: the cost error halves (measured 4.5e-2 / 2.3e-2 at rho = 0, 2.4e-2 / 1.2e-2 at 0.5, ratio 1.98)
    and the Richardson pair is within 4e-4 (measured 4.0e-4 and 2.6e-4; the pair (24, 48) is within 1.8e-3)."""
    from test_finite_discount import exact, model
    J, _ = exact(rho)
    e48 = cells(model(rho, "finite", 48)).require_converged().costs["a"] - J
    e96 = cells(model(rho, "finite", 96)).require_converged().costs["a"] - J
    assert 1.9 < e48 / e96 < 2.1
    assert abs(2 * e96 - e48) < 1e-3


def test_ch1_p10_mean_paths_close_on_the_spectral_ones_as_h_squared():
    """The cell engine, an independent first-order discretisation, solves the same mean system on its cells: its
    Richardson pairs (40, 80) and (80, 160) close on the spectral paths as h^2 (4.8e-3 then 1.2e-3 at T/2; the
    cost 1.4e-3 then 3.3e-4)."""
    from ch1_mean_sweep import model as ch1_targets
    sp = ns.solve(ch1_targets(10.0, nodes=12)).require_converged()
    tq = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9]); d1sp = sp.mean("D1", tq); Jsp = sp.cost_parts["player1"]["mean"]
    got = {}
    for N in (40, 80, 160):
        d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}
        res = cells(d, nodes=N).require_converged(); idx = np.round(tq / res.compiled.h).astype(int)
        got[N] = (res.means["D1"][idx], res.cost_parts["player1"]["mean"])
        assert res.mean_times.shape == (N,) and np.abs(res.means["D2"] + res.means["D1"]).max() < 1e-12
    assert np.abs(got[160][0] - d1sp).max() < np.abs(got[80][0] - d1sp).max() < np.abs(got[40][0] - d1sp).max()
    r1 = 2 * got[80][0] - got[40][0] - d1sp; r2 = 2 * got[160][0] - got[80][0] - d1sp
    assert np.abs(r2).max() < 1.5e-3 and np.abs(r2).max() < 0.5 * np.abs(r1).max()
    J1 = 2 * got[80][1] - got[40][1] - Jsp; J2 = 2 * got[160][1] - got[80][1] - Jsp
    assert abs(J2) < 5e-4 and abs(J2) < 0.5 * abs(J1)


def test_one_agent_mean_error_halves_with_the_cells():
    """The cell engine's error halves from 40 to 80 cells (first order): 1.5e-2 to 7.6e-3 on the state at
    (a, r, theta, rho) = (-0.5, 0.3, 2, 0.5), against the deterministic finite-horizon LQ closed form."""
    from test_means_finite import one_state, riccati
    a, r, theta, rho, x0 = -0.5, 0.3, 2.0, 0.5, -1.0
    errs = []
    for N in (40, 80):
        rc = cells(one_state(a, r, theta, rho, x0, nodes=N)).require_converged()
        xs, us, J, V0 = riccati(a, 1.0, 1.0, theta, r, rho, x0, 1.0, rc.mean_times)
        errs.append(np.abs(rc.means["X"] - xs[0]).max())
    assert 0.4 < errs[1] / errs[0] < 0.6 and errs[1] < 1e-2
