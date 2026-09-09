"""Settings: the tuning constants in one place (noisestate/settings.py), read by the engines and the results
through settings=, with the older class-attribute names kept as aliases."""
import json, os, pytest
import noisestate as ns
from noisestate import Settings
from noisestate.results import StationaryResult
from noisestate.engine import EngineBase
from noisestate.settings import DEFAULT
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_defaults_aliases_and_the_dict_form():
    assert Settings() == DEFAULT and DEFAULT.changed() == {} and Settings.of(None) is DEFAULT
    # every older name reads the same field, on the class (the defaults) and on an instance (its settings)
    aliases = {(EngineBase, "FOC_RCOND"): "foc_rcond", (EngineBase, "ANDERSON_M"): "anderson_m",
               (EngineBase, "SECOND_ORDER_TOL"): "second_order_tol", (EngineBase, "SECOND_ORDER_DENSE"): "second_order_dense",
               (ns.SpectralFiniteSolver, "MAP_RIDGE"): "map_ridge", (ns.SpectralFiniteSolver, "MEAN_RCOND"): "mean_rcond",
               (ns.StationarySolver, "MEAN_RCOND"): "mean_rcond", (ns.BaseResult, "RESOLUTION_TOL"): "resolution_tol",
               (ns.BaseResult, "WINDOW_TAIL_TOL"): "window_tail_tol", (ns.BaseResult, "MEAN_ZERO"): "mean_zero",
               (ns.BaseResult, "STABILITY_MAX_EVALUATIONS"): "stability_max_evaluations", (ns.BaseResult, "STABILITY_FALLBACK"): "stability_fallback",
               (ns.BaseResult, "REFINE_COST_TOL"): "refine_cost_tol", (ns.BaseResult, "REFINE_KERNEL_TOL"): "refine_kernel_tol"}
    for (cls, name), field in aliases.items():
        assert getattr(cls, name) == getattr(DEFAULT, field)
    assert (EngineBase.FOC_RCOND, ns.BaseResult.STABILITY_MAX_EVALUATIONS, ns.SpectralFiniteSolver.MAP_RIDGE) == (1e-10, 200, 1e-13)
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    S = ns.StationarySolver(m, settings={"foc_rcond": 1e-8})
    assert S.FOC_RCOND == S.settings.foc_rcond == 1e-8 and S.solver_kw["settings"] == {"foc_rcond": 1e-8}
    assert S.c.LEAD_WEIGHT_WARN == DEFAULT.lead_weight_warn and "settings" not in ns.StationarySolver(m).solver_kw
    with pytest.raises(TypeError, match="unknown settings"):
        Settings.of({"nope": 1})
    with pytest.raises(TypeError):
        ns.solve(m, {"settings": 3})
    with pytest.raises(TypeError):
        Settings(nope=1)


def test_settings_reach_the_checks_and_are_recorded():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    r0 = ns.solve(m).check()
    lo, hi = sorted(so["min"] for so in r0.second_order.values())          # both positive: the checks pass by default
    assert lo > 0 and r0.resolution_ok and all(so["ok"] for so in r0.second_order.values())
    # the thresholds of the checks: a curvature tolerance between the two agents' minima (a negative one, so a
    # positive curvature can fail it), a resolution tolerance below the representation error, a stability budget
    s = Settings(second_order_tol=-0.5 * (lo + hi), resolution_tol=1e-20, stability_max_evaluations=5)
    r = ns.solve(m, {"settings": s}, stability=True)
    assert r.settings is s and r.solver_kw["settings"] == s.changed() == {"second_order_tol": -0.5 * (lo + hi), "resolution_tol": 1e-20,
                                                                          "stability_max_evaluations": 5}
    assert r.costs == r0.costs and all((r.maps[a] == r0.maps[a]).all() for a in r.maps)      # the checks' thresholds do not touch the numbers
    assert [so["ok"] for so in r.second_order.values()].count(False) == 1 and "NOT A MINIMUM" in r.summary()
    assert r.resolution_ok is False and "UNDER-RESOLVED" in r.summary()
    assert r.stability_report["evaluations"] <= 5 and "evaluation budget" in r.stability_report["method"]
    assert [d["threshold"] for d in r.diagnose() if d["name"] == "resolution"] == [1e-20]
    assert [d["threshold"] for d in r.diagnose() if d["name"].startswith("second_order:")] == [0.5 * (lo + hi)] * 2
    # recorded in the payload, JSON-ready, and the engine a result rebuilds carries them
    d = r.to_dict(); json.dumps(d)
    assert d["options"]["solver"]["settings"] == s.changed()
    assert ns.make_solver(m, **d["options"]["solver"]).settings == s and r._make_solver(m).settings == s
    assert r._make_solver(m).RESULT is StationaryResult


def test_settings_reach_the_best_response_and_the_solve():
    m = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).with_horizon().with_numerics(nodes=4)
    r0 = ns.solve(m).check()
    with pytest.raises(ValueError, match="singular"):                        # every system fails a condition threshold of 1
        ns.solve(m, {"settings": Settings(foc_rcond=1.0)})
    r = ns.solve(m, {"settings": Settings(anderson_iters=2)})                    # two Anderson iterations, then the Newton polish
    assert "anderson: 3 evaluations" in r.message and "newton polish" in r.message
    assert r.converged and abs(r.costs["player1"] - r0.costs["player1"]) < 1e-9
    assert ns.SpectralFiniteSolver(m, settings=Settings(map_ridge=1e-9)).MAP_RIDGE == 1e-9
    assert ns.FiniteSolver(m.with_horizon(kind="finite").with_numerics(nodes=8, engine="cells"), settings={"cell_dense_max": 1}).settings.cell_dense_max == 1
