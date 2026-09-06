"""The API stage's surface: Numerics apart from the model, the explicit solve(), the engines namespace."""
import os

import pytest

import noisestate as ns
from noisestate import Numerics

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_numerics_block_and_the_deprecated_nesting_agree():
    """A numerics: block and the old keys under horizon: build the same model, the old form with a note;
    a nested key that disagrees with the block is an error; kind finite_cells is engine cells."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    assert d["numerics"] == {"nodes": 24} and "nodes" not in d["horizon"]
    m = ns.Model.from_dict(d)
    assert m.numerics == Numerics(nodes=24) and m.numerics.engine is None and not m.deprecations
    old = {**d, "horizon": {**d["horizon"], "nodes": 24}}; del old["numerics"]
    mo = ns.Model.from_dict(old)
    assert mo.horizon.nodes == 24 and any("deprecated: horizon.nodes" in n for n in mo.notes)
    assert mo.to_dict()["numerics"] == {"nodes": 24} and "nodes" not in mo.to_dict()["horizon"]
    with pytest.raises(ValueError, match="disagree"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "nodes": 8}})
    f = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); f["horizon"]["kind"] = "finite_cells"
    mc = ns.Model.from_dict(f)
    assert mc.horizon.kind == "finite" and mc.numerics.engine == "cells" and any("finite_cells" in n for n in mc.notes)
    assert mc.to_dict()["horizon"]["kind"] == "finite" and mc.to_dict()["numerics"]["engine"] == "cells"
    with pytest.raises(ValueError, match="numerics.engine 'cells'"):
        ns.Model.from_dict({**d, "numerics": {"nodes": 8, "engine": "cells"}})


def test_solve_takes_a_numerics_and_names_the_field_of_a_stray_keyword():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    r = ns.solve(m, Numerics(nodes=12, tol=1e-8))
    assert r.compiled.N == 12 and r.numerics.nodes == 12 and r.numerics.engine == "stationary" and r.numerics.tol == 1e-8
    assert r.to_dict()["options"]["numerics"]["nodes"] == 12 and r.model.horizon.nodes == 12 and m.horizon.nodes == 24
    r2 = ns.solve(m, {"nodes": 12}, tol=1e-6)                     # a dict; the keyword tol over the numerics'
    assert r2.numerics.tol == 1e-6 and r2.solve_kw["tol"] == 1e-6
    assert ns.solve(m, nodes=12).compiled.N == 12                  # the alias (until 0.6)
    assert ns.solve(m, settings={"anderson_m": 5}).settings.anderson_m == 5
    with pytest.raises(TypeError, match=r"Numerics\(unit_range=\.\.\.\)"):
        ns.solve(m, unit_range=2.0)
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(m, bogus=1)
    with pytest.raises(TypeError, match="not the stationary one"):
        ns.solve(m, past=r)
    r3 = ns.solve(m.with_numerics(nodes=8))
    assert r3.compiled.N == 8 and r3.numerics == r3.model.numerics.resolved("stationary").merged(Numerics(tol=1e-10, damping=0.6, max_newton=60, variable="actions"))
    fine = r3.refine(); assert fine["nodes"] == 12 and r3.refinement is fine


def test_engines_namespace_and_the_cell_engine_by_numerics():
    from noisestate import engines
    assert engines.stationary is ns.StationarySolver and engines.spectral is ns.SpectralFiniteSolver and engines.cells is ns.FiniteSolver
    assert set(ns.ENGINES) == {"stationary", "spectral", "cells"}
    m = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml"))
    S, num = engines.build(m, {"engine": "cells", "nodes": 8})
    assert isinstance(S, ns.FiniteSolver) and num.engine == "cells" and S.solve().kind == "finite_cells"
    with pytest.raises(TypeError, match="cell engine"):
        engines.build(m, {"engine": "cells"}, past=[])
    rows = ns.sweep(m, "p1", [3.0, 4.0], numerics={"nodes": 6})
    assert all(r["result"].compiled.g.nt == 6 for r in rows)


@pytest.mark.parametrize("numerics", [{"engine": "spectral", "nodes": 12}, {"engine": "cells", "nodes": 12}])
def test_one_result_reads_the_same_on_every_engine(numerics):
    """A consumer reads a kernel with its axes without knowing the engine: every axis in res.axes has the
    length of the kernel's node axis (or axes), the maps' axes are there too, and the aliases hold."""
    m = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml"))
    r = ns.solve(m, numerics).check()
    K = r.kernel("X", "w1")
    node_axes = {k: v for k, v in r.axes.items() if k != "maps"}
    assert K.ndim == (2 if numerics["engine"] == "cells" else 1) and all(len(v) == K.shape[0] for v in node_axes.values())
    assert "time" in node_axes and "shock_time" in node_axes and r.times is not None and len(r.times) == len(r.paths["means"]["X"])
    assert r.axes["maps"]["player1"]["y1"]["map_time"].shape[0] == r.maps["player1"].shape[2]
    assert r.evaluations == r.iterations and r.world is r.Z and isinstance(r, ns.Result) and isinstance(r, ns.BaseResult)
    assert r.status["ok"] is True and r.status["flags"] == [] and r.status["rows"] == r.diagnose()
    assert r.cost_kind.startswith("discounted") and set(r.cost_parts["player1"]) == {"variance", "mean"}
    p = r.to_dict()
    assert p["payload_version"] == 1 and p["engine"] == numerics["engine"] and set(p["axes"]) == set(node_axes) and p["status"]["ok"]


def test_the_stationary_result_and_a_transition_read_the_same_way():
    s = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8})
    assert list(s.axes) == ["age", "maps"] and s.times is None and s.paths == {} and "window_tail" in s.extra
    assert s.kernel("X").shape == (len(s.axes["age"]), 3) and s.status["ok"] is False and any("WINDOW" in f for f in s.status["flags"])
    t = ns.solve(os.path.join(EX, "ch3_precision_change.yaml"), {"nodes": 5, "continuation_nodes": 8}, max_evaluations=3)
    assert (t.axes["shock_time"] < 0).any() and set(t.paths) == {"means", "loss", "belief_error"}
    assert t.paths["loss"]["player1"].shape == t.times.shape and t.paths["belief_error"]("player2", "X").shape == t.times.shape
    assert {"past", "continuation", "settled", "old_flows", "new_flows", "excess_costs"} <= set(t.extra) and t.extra["settled"] == t.settled
    assert t.to_dict()["kind"] == "transition" and t.to_dict()["engine"] == "spectral"
