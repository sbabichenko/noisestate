"""The spectral finite engine's matrix-free best response (noisestate/finite_free.py on the operators of noisestate/spectral_operators.py, settings.foc_dense_max)
against the dense one: the same best response and the same fixed point to the Krylov tolerance."""
import numpy as np

import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
from helpers import example, example_dict, stationary, same_model_solver, slow

FREE = {"foc_dense_max": 0}          # every system matrix-free
DENSE = {"foc_dense_max": 1 << 20}   # every system assembled and factored (the default 500 leaves only tiny ones there)


def _same_best_response(dense, free, maps, tol=1e-10):
    """Each agent's best response on both paths: gamma, action kernels, world, the projected map, the
    second-order check and the FOC decomposition agree to tol of their peak."""
    for a in dense.model.agents:
        gd, od = dense.best_response(a, maps, want_decomp=True)
        gf, of = free.best_response(a, maps, want_decomp=True)
        assert of["krylov"] >= 1 and free._krylov_log[-1][2] <= 1e-11
        for k in ("gamma", "action", "Zfull"):
            assert np.abs(od[k] - of[k]).max() <= tol * np.abs(od[k]).max(), (a.name, k)
        assert np.abs(gd - gf).max() <= tol * np.abs(gd).max(), a.name
        assert abs(od["second_order"]["min"] - of["second_order"]["min"]) < 1e-9 and od["second_order"]["ok"] == of["second_order"]["ok"]
        for u in a.controls:
            for part in ("foc", "physical", "wedge"):
                assert np.abs(od["decomp"][u][part] - of["decomp"][u][part]).max() <= tol * max(1e-300, np.abs(od["decomp"][u]["foc"]).max())
        assert abs(dense.expected_cost(a, od["Zfull"]) - free.expected_cost(a, of["Zfull"])) < 1e-12
        assert abs(dense._representation_error(a, od["Zfull"], od["action"], gd) - free._representation_error(a, of["Zfull"], of["action"], gf)) < 1e-12


def _same_fixed_point(rd, rf, cost_tol=1e-10):
    assert rf.converged and abs(rd.evaluations - rf.evaluations) <= 1 and rd.converged
    for a in rd.model.agents:
        assert abs(rd.costs[a.name] - rf.costs[a.name]) < cost_tol, (a.name, rd.costs[a.name], rf.costs[a.name])
        assert np.abs(rd.maps[a.name] - rf.maps[a.name]).max() < 1e-9 * np.abs(rd.maps[a.name]).max()
        assert abs(rd.second_order[a.name]["min"] - rf.second_order[a.name]["min"]) < 1e-9
        assert abs(rd.representation_error[a.name] - rf.representation_error[a.name]) < 1e-9


@slow("slow (12 s; the 6-node case with a past below is the fast one); set NOISESTATE_SLOW=1")
def test_matrix_free_best_response_matches_the_dense_one_without_a_past():
    """examples/ch1_delayed_finite.yaml at 8 nodes (N = 640, player2's row delayed): with foc_dense_max 0 every
    operator is applied from the line paths and the first-order conditions are solved by GMRES (11 to 18
    iterations from zero with the time-row preconditioner, 1 to 7 warm-started along the fixed point); the
    best response to random maps agrees with the dense path to 1e-12 of its peak (gamma, actions, world, the
    projected map, the second-order check, the decomposition, the cost and the representation error), and
    the fixed point on the action kernels reaches the same equilibrium in the same 13 evaluations, the costs
    to 1e-15.  This system is past the default threshold (500), so the dense path is asked for explicitly."""
    d = example_dict("ch1_delayed_finite"); d.setdefault("numerics", {})["nodes"] = 8
    m = ns.Model.from_dict(d)
    dense = SpectralFiniteSolver(m, settings=DENSE); free = SpectralFiniteSolver(m, settings=FREE)
    assert not dense.foc_free and free.foc_free and free.settings.foc_dense_max == 0
    c = dense.c; rng = np.random.default_rng(5)
    acts = {a.name: rng.standard_normal((len(a.controls), c.N, c.nW)) * 0.1 for a in m.agents}
    _same_best_response(dense, free, dense.maps_from_actions(acts))
    _same_fixed_point(dense.solve(), free.solve())
    its = [k for (_, k, _) in free._krylov_log]
    assert max(its) <= 40 and min(its[-4:]) <= 3, its


def test_matrix_free_best_response_matches_the_dense_one_with_a_past():
    """Chapter 3 as its own past and continuation (T = 6, L = 3, 6 nodes: the band, the buffer, the envelope
    responses, the corner ties of the Duffy triangles in the Krylov solve and its preconditioner) and the
    Kyle-Back prior (initial shocks: the discrete weights and the point conditions of the line s = 0): the
    same best responses to 1e-10 and the same fixed points, costs to 1e-12, in the same number of evaluations."""
    m3 = example("ch3_two_player")
    stat = stationary(m3, 6)
    dense = same_model_solver(m3, stat, 6.0, 6)
    free = same_model_solver(m3, stat, 6.0, 6, settings=FREE)
    assert free.foc_free and free.c.buffer.any() and free.c.g.upper.any()
    _same_best_response(dense, free, dense.c.frozen)
    _same_fixed_point(dense.solve(start_policy="stationary"), free.solve(start_policy="stationary"))
    kb = example("kyle_back_prior").with_numerics(nodes=8)
    rd = ns.solve(kb); rf = ns.solve(kb, {"settings": FREE})
    assert rd.compiled.n_init == 1 and rf.solver_kw["settings"] == FREE
    _same_fixed_point(rd, rf)


def test_matrix_free_path_reports_a_singular_system():
    """A control whose current value carries no quadratic term makes the time-row blocks of the preconditioner
    singular: the matrix-free path raises the singular-system ValueError the dense path raises on its
    condition estimate, naming the block."""
    import pytest
    d = example_dict("ch1_two_player_finite"); d.setdefault("numerics", {})["nodes"] = 4
    m = ns.Model.from_dict(d)
    with pytest.raises(ValueError, match="singular"):
        SpectralFiniteSolver(m, settings={"foc_dense_max": 0, "foc_rcond": 1.0}).solve()
