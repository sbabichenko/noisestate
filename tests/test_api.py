"""The API stage's surface: Numerics apart from the model, the explicit solve(), the engines namespace."""
import os

import pytest

import noisestate as ns
from noisestate import Numerics

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_the_numerics_block_is_the_only_place_the_grid_is_sized():
    """The grid lives in numerics: alone.  The spellings 0.5 nested under horizon: were removed in 0.6 and
    each is refused by the name that replaced it; the cell engine is numerics.engine, not a horizon kind."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    assert d["numerics"] == {"nodes": 24} and "nodes" not in d["horizon"]
    m = ns.Model.from_dict(d)
    assert m.numerics == Numerics(nodes=24) and m.numerics.engine is None and not m.deprecations
    old = {**d, "horizon": {**d["horizon"], "nodes": 24}}; del old["numerics"]
    with pytest.raises(ValueError, match="moved under numerics"):
        ns.Model.from_dict(old)
    f = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); f["horizon"]["kind"] = "finite_cells"
    with pytest.raises(ValueError, match="numerics.engine 'cells'"):
        ns.Model.from_dict(f)
    fc = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml"))
    fc.setdefault("numerics", {}).update(nodes=8, engine="cells")
    mc = ns.Model.from_dict(fc)
    assert mc.horizon.kind == "finite" and mc.numerics.engine == "cells" and not mc.deprecations
    with pytest.raises(ValueError, match="numerics.engine 'cells'"):
        ns.Model.from_dict({**d, "numerics": {"nodes": 8, "engine": "cells"}})


def test_solve_takes_a_numerics_and_names_the_field_of_a_stray_keyword():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    r = ns.solve(m, Numerics(nodes=12, tol=1e-8))
    assert r.compiled.N == 12 and r.numerics.nodes == 12 and r.numerics.engine == "stationary" and r.numerics.tol == 1e-8
    assert r.to_dict()["options"]["numerics"]["nodes"] == 12 and r.model.horizon.nodes == 12 and m.horizon.nodes == 24
    r2 = ns.solve(m, {"nodes": 12}, tol=1e-6)                     # a dict; the keyword tol over the numerics'
    assert r2.numerics.tol == 1e-6 and r2.solve_kw["tol"] == 1e-6
    assert ns.solve(m, Numerics(nodes=12, settings={"anderson_m": 5})).settings.anderson_m == 5
    with pytest.raises(TypeError, match=r"Numerics\(nodes=\.\.\.\)"):        # no bare aliases since 0.6
        ns.solve(m, nodes=12)
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
    assert isinstance(r, ns.Result) and isinstance(r, ns.BaseResult)
    assert not hasattr(r, "Z") and not hasattr(r, "iterations")      # the 0.5 spellings, removed in 0.6
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


def test_schema_accepts_the_shipped_files_and_names_the_path_of_an_error():
    import glob, json
    from noisestate.cli import main
    for f in sorted(glob.glob(os.path.join(EX, "*.yaml"))):
        assert ns.schema.validate(ns.read_yaml(f), "model") == [], f
    s = ns.schema("model")
    assert s["$schema"].endswith("2020-12/schema")
    assert "nodes" not in s["properties"]["horizon"]["properties"]          # the grid lives in numerics: alone
    assert s["properties"]["numerics"]["properties"]["nodes"]["type"] == "integer"
    bad = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); bad["numerics"]["nodes"] = 1; bad["agents"]["player1"]["extra"] = 1
    errs = ns.schema.validate(bad, "model")
    assert any(e.startswith("numerics.nodes:") for e in errs) and any(e.startswith("agents.player1.extra:") for e in errs)
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8})
    assert ns.schema.validate(json.loads(json.dumps(r.to_dict())), "payload") == []
    p = r.to_dict(); del p["costs"]; p["engine"] = "abacus"
    errs = ns.schema.validate(json.loads(json.dumps(p)), "payload")
    assert any("missing required key 'costs'" in e for e in errs) and any(e.startswith("engine:") for e in errs)
    with pytest.raises(ValueError, match="'model' or 'payload'"):
        ns.schema("grid")
    assert main(["schema", "payload"]) == 0


def test_cli_validate_transition_schema_and_plot(tmp_path, capsys):
    import json, yaml
    from noisestate.cli import main
    bad = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); bad["numerics"]["nodes"] = 1
    with open(tmp_path / "bad.yaml", "w") as fh:
        yaml.safe_dump(bad, fh)
    assert main(["validate", str(tmp_path / "bad.yaml")]) == 2 and "numerics.nodes: 1 is below the minimum 2" in capsys.readouterr().err
    assert main(["validate", os.path.join(EX, "ch3_precision_change.yaml")]) == 0
    new = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); new["params"]["p1"] = 10.0
    with open(tmp_path / "new.yaml", "w") as fh:
        yaml.safe_dump(new, fh)
    out = tmp_path / "change.json"
    code = main(["transition", os.path.join(EX, "ch3_two_player.yaml"), str(tmp_path / "new.yaml"), "--window", "6", "--nodes", "5",
                 "-o", str(out), "--max-evaluations", "2"])
    p = json.load(open(out))
    assert code == 1 and p["kind"] == "transition" and p["options"]["solve"]["start"] == "stationary" and p["numerics"]["nodes"] == 5
    assert ns.schema.validate(p, "payload") == []
    pytest.importorskip("matplotlib")
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8}, max_evaluations=3)
    with open(tmp_path / "res.json", "w") as fh:
        json.dump(r.to_dict(), fh)
    assert main(["plot", str(tmp_path / "res.json"), str(tmp_path / "res.png")]) == 1 and (tmp_path / "res.png").exists()


def test_the_zero_start_is_explicit_with_a_continuation():
    old = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8}).check()
    new = old.model.with_params(p1=10.0).with_horizon(kind="finite", window=6.0).with_numerics(nodes=5)
    d = ns.solve(new, past=old, continuation="stationary", max_evaluations=1)
    z = ns.solve(new, past=old, continuation="stationary", max_evaluations=1, start="zero")
    e = ns.solve(new, past=old, continuation="end", max_evaluations=1)
    assert d.solve_kw["start"] == "stationary" and z.solve_kw["start"] == "zero" and e.solve_kw["start"] == "zero"
    assert d.residual != z.residual
    rows = ns.sweep(new.with_horizon(kind="transition", past={"model": old.model.to_dict()}, continuation="stationary"), "p1",
                    [10.0], solve_kw={"max_evaluations": 1})
    assert rows[0]["result"].solve_kw["start"] == "stationary"
