"""Reference-free correctness properties.

1. Optimality: an agent's best response beats every random feasible perturbation (stationary and
   spectral finite engines).
2. The two iteration variables (raw maps, action kernels) reach the same equilibrium.
3. Relabelling channels or reordering agents does not change the equilibrium.
"""
import os, numpy as np
import noisestate as ns
from noisestate.stationary import StationarySolver
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def _feasible_perturbation(S, agent, Zpass, rng):
    ytil, yinst = S._passive_rows(agent, Zpass)
    Gk = S._row_operator(agent, ytil, yinst)
    gam = rng.standard_normal(Gk.shape[2])
    return np.stack([Gk[k] @ gam for k in range(S.c.nW)], axis=1)


def test_stationary_best_response_is_optimal_kyle_back():
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml")); d["params"]["rho"] = 0.0   # the flow loss is the objective only at rho = 0
    S = StationarySolver(ns.Model.from_dict(d)); res = S.solve().check()
    assert res.resolution_ok and max(res.representation_error.values()) < 1e-9     # well-resolved reference models
    a = S.model.agents[1]; c = S.c; nW = c.nW
    g, out = S.best_response(a, res.maps)
    Zp = c.closed_loop(res.maps, excluded=a.name, impulse_controls=a.controls); Zpass, R = Zp[:, :nW], Zp[:, nW:]
    Resp = S._response_operators(a, R)[0]
    cost = lambda cc: S.expected_loss(a, Zpass + Resp @ cc)
    c0 = out["action"][0]; L0 = cost(c0); rng = np.random.default_rng(0)
    for _ in range(8):
        dc = _feasible_perturbation(S, a, Zpass, rng); dc *= 0.02 / np.abs(dc).max()
        assert cost(c0 + dc) >= L0 - 1e-10 and cost(c0 - dc) >= L0 - 1e-10


def test_spectral_finite_best_response_is_optimal():
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d["horizon"]["nodes"] = 8
    S = SpectralFiniteSolver(ns.Model.from_dict(d)); res = S.solve().check()
    a = S.model.agents[0]; c = S.c; nW = c.nW
    maps = res.maps; g, out = S.best_response(a, maps)
    Zp = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls); Zpass, R = Zp[:, :nW], Zp[:, nW:]
    Resp = S._response_operators(a, R)[0]
    ytil, yinst = S._passive_rows(a, Zpass); Gk = S._row_operator(a, ytil, yinst)
    cost = lambda cc: S.expected_cost(a, Zpass + Resp @ cc)
    c0 = out["action"][0]; L0 = cost(c0); rng = np.random.default_rng(1)
    for _ in range(6):
        gam = rng.standard_normal(Gk.shape[2]); dc = np.stack([Gk[k] @ gam for k in range(nW)], axis=1); dc *= 0.02 / np.abs(dc).max()
        assert cost(c0 + dc) >= L0 - 1e-9 and cost(c0 - dc) >= L0 - 1e-9


def test_map_and_action_iterations_agree():
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    ra = StationarySolver(m).solve(variable="actions").check(); rm = StationarySolver(m).solve(variable="maps").check()
    assert np.abs(ra.action_kernel("D1") - rm.action_kernel("D1")).max() < 1e-8


def test_channel_relabelling_and_agent_order_invariance():
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml"))
    base = ns.solve(ns.Model.from_dict(d)).check()
    # rename and reorder channels
    ren = {"wV": "fund", "wZ": "noise_flow", "w1": "sig"}
    d2 = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml")); d2["channels"] = ["sig", "fund", "noise_flow"]
    for st in d2["states"].values():
        st["noise"] = {ren[k]: v for k, v in st["noise"].items()}
    for ag in d2["agents"].values():
        for row in ag["signals"].values():
            row["noise"] = {ren[k]: v for k, v in row["noise"].items()}
    # and put the trader before the market maker
    d2["agents"] = {"trader1": d2["agents"]["trader1"], "market_maker": d2["agents"]["market_maker"]}
    alt = ns.solve(ns.Model.from_dict(d2)).check()
    for name in ("V", "P", "D1"):
        for old, new in ren.items():
            assert np.abs(base.kernel(name)[:, base.compiled.channels.index(old)] - alt.kernel(name)[:, alt.compiled.channels.index(new)]).max() < 1e-9
    assert abs(base.costs["trader1"] - alt.costs["trader1"]) < 1e-10


def test_typos_in_model_files_are_rejected():
    import pytest
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml"))
    d["agents"]["trader1"]["signals"]["y1"]["dealy"] = 0.5
    with pytest.raises(ValueError, match="unknown key"):
        ns.Model.from_dict(d)
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml")); d["channels"].append("w_unused")
    with pytest.raises(ValueError, match="never loaded"):
        ns.Model.from_dict(d)


def test_naive_observers_remove_the_wedge():
    """If every other agent is naive to trader1's deviations, the information-wedge part of its
    first-order condition vanishes and its equilibrium equals the no-reaction one."""
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    res = StationarySolver(m, naive_observers={"trader1": ["market_maker"]}).solve().check()
    dec = res.foc["trader1"]["D1"]
    assert np.abs(dec["wedge"]).max() < 1e-9 * max(1.0, np.abs(dec["foc"]).max())
    base = StationarySolver(m).solve().check()
    assert np.abs(base.foc["trader1"]["D1"]["wedge"]).max() > 1e-3          # the wedge is real in the full game
    assert np.abs(base.action_kernel("D1") - res.action_kernel("D1")).max() > 1e-3


def test_model_round_trips_through_to_dict():
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    m2 = ns.Model.from_dict(m.to_dict())
    assert m2.to_dict() == m.to_dict()
    assert abs(ns.solve(m).costs["trader1"] - ns.solve(m2).costs["trader1"]) < 1e-12
