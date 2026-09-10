"""The JSON payload carries its provenance and says how delayed-row maps are indexed (2026-09-05 audit)."""
import json, os, time, numpy as np, pytest
import noisestate as ns
from noisestate.schema import PAYLOAD_VERSION
from noisestate import engines
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_payload_round_trips_and_rebuilds_the_solve():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    res = ns.solve(m)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["version"] == ns.__version__ and d["name"] == "ch3_two_player" and d["engine"] == "stationary"
    assert d["params"] == {"p1": 3.0, "p2": 10.0, "r1": 1.0, "r2": 1.0}
    assert d["horizon"] == {"kind": "stationary", "discount": 0.0, "window": 3.0} and d["numerics"] == {"engine": "stationary", "nodes": 24, "tol": 1e-10, "damping": 0.6, "max_newton": 60, "variable": "actions"}
    assert d["payload_version"] == PAYLOAD_VERSION and d["axes"]["age"] == d["grid"]["ages"] and d["times"] is None and d["assessment"]["accepted"] is False
    assert d["options"]["solve"]["start_policy"] == "zero" and d["options"]["solver"]["verbose"] is False
    assert d["agents"]["player2"] == {"controls": ["D2"], "signals": {"y2": {"delay": 0.0, "map_age": d["grid"]["ages"]}}}
    m2 = ns.Model.from_dict(d["model"]); assert m2.to_dict() == m.to_dict()
    again = engines.solver(m2, **d["options"]["solver"]).solve(**d["options"]["solve"])
    assert all(abs(again.costs[k] - res.costs[k]) < 1e-12 for k in res.costs)


def test_delayed_row_map_axes_place_the_map():
    """A delayed row's map is not indexed like an undelayed one: the payload's map_age (stationary) and
    map_time (finite, where the map is stored at t - delay) say where each value belongs, and the map is
    zero exactly where those axes leave the window or the horizon."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); d["agents"]["player2"]["signals"]["y2"]["delay"] = 1.0; d.setdefault("numerics", {})["nodes"] = 8
    r = ns.solve(d).require_converged(); p = r.to_dict(); row = p["agents"]["player2"]["signals"]["y2"]
    g = np.array(p["maps"]["player2"]["D2"]["y2"]); age = np.array(row["map_age"]); L = p["horizon"]["window"]
    assert row["delay"] == 1.0 and np.allclose(age, r.ages + 1.0) and "map_age" in p["map_convention"]
    assert (age > L + 1e-9).sum() == 7 and np.abs(g[age > L + 1e-9]).max() == 0.0 and np.abs(g[age < L]).max() > 0.1
    d = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml")); d.setdefault("numerics", {})["nodes"] = 4
    r = ns.solve(d).require_converged(); p = json.loads(json.dumps(r.to_dict())); row = p["agents"]["player2"]["signals"]["y2"]
    g = np.array(p["maps"]["player2"]["D2"]["y2"]); t = np.array(row["map_time"]); T = p["horizon"]["T"]
    assert row["delay"] == 0.25 and np.allclose(t, r.grid.t + 0.25) and np.allclose(row["map_age"], r.grid.a + 0.25)
    assert (t > T + 1e-9).any() and np.abs(g[t > T + 1e-9]).max() == 0.0 and np.abs(g[t <= T]).max() > 0.1
    assert p["agents"]["player1"]["signals"]["y1"]["map_time"] == p["grid"]["t"]        # an undelayed row: the grid's own time


def test_solve_kw_records_the_start_option_and_repeats_the_solve():
    S = engines.solver(ns.load(os.path.join(EX, "ch3_two_player.yaml")))
    res = S.solve(start_policy="coarse")
    assert res.solve_kw["start_policy"] == "coarse" and json.dumps(res.solve_kw) and "coarse start" in res.message
    again = S.solve(**res.solve_kw)
    assert again.evaluations == res.evaluations and again.message == res.message
    assert S.solve().solve_kw["start_policy"] == "zero"


def test_seconds_include_the_diagnostics(monkeypatch):
    S = engines.solver(ns.load(os.path.join(EX, "ch3_two_player.yaml")))
    finish = type(S)._finish
    monkeypatch.setattr(type(S), "_finish", lambda self, res: (time.sleep(0.2), finish(self, res)))
    assert S.solve().seconds >= 0.2


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"noisestate {ns.__version__}"


def test_version_is_the_source_tree_s():
    # __version__ (the payload's `version`, --version) is pyproject.toml's when the package is imported from a
    # checkout, not the installed distribution's (an editable install bumped since it was installed said 0.2.4)
    import re
    text = open(os.path.join(HERE, "..", "pyproject.toml")).read()
    assert ns.__version__ == re.search(r'^version = "([^"]+)"', text, re.M).group(1) != "unknown"


def test_a_refined_payload_json_encodes():
    """A gap the suite had: 26 places encoded a payload and 10 called refine(), and no test did
    both -- so to_dict() putting the Refinement OBJECT into the payload went unnoticed.  A
    dataclass is not JSON-serialisable however many dict methods it carries, so nothing was
    covering for it; the encoding simply never ran with a refinement present."""
    res = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8})
    res.refine()
    payload = res.to_dict()
    assert isinstance(payload["refinement"], dict)          # data, not the object
    #  allow_nan=False: json.dumps otherwise emits bare NaN/Infinity, which are not valid JSON
    #  and which no conforming parser will read back.  A payload of numerical results is exactly
    #  where they arise -- an unconverged solve, a zero scale, an empty maximum.
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["refinement"]["nodes"] > 8
    assert ns.schema.validate(round_tripped, "payload") == []


def test_a_payload_with_stability_and_a_march_json_encodes():
    """The same gap for the other two dataclasses the payload carries."""
    path = os.path.join(EX, "ch3_two_player.yaml")
    old = ns.solve(path, {"nodes": 6}).require_converged()
    #  a settle march, so the payload actually carries MarchPoints; a fixed T has none
    res = ns.transition(old, ns.load(path).with_params(p1=10.0), settle=5e-2,
                        numerics={"nodes": 6}, max_evaluations=30)
    res.stability()
    payload = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert isinstance(payload["stability"], dict) and payload["stability"]["verified"] in (True, False)
    assert isinstance(payload["march"], list) and isinstance(payload["march"][0], dict)
    assert {"T", "gap", "evaluations", "seconds", "monitor"} <= set(payload["march"][0])
    assert ns.schema.validate(payload, "payload") == []


def test_the_payload_version_is_pinned_by_the_schema():
    """The version and the format move together: the schema pins payload_version as a const, so a
    key change that forgets to bump it fails validation here rather than in whatever reads the
    file.  Version 2 (0.8) renamed means_t to mean_times, replaced status with assessment, dropped
    resolution_ok, renamed options.solve.start, extended stability with its evidence, and made
    refinement a dict -- five of those six validated silently under version 1's open schema."""
    res = ns.solve(os.path.join(EX, "ch1_two_player_finite.yaml"), {"nodes": 8})
    d = res.to_dict()
    assert d["payload_version"] == PAYLOAD_VERSION and PAYLOAD_VERSION == 2
    assert "mean_times" in d and "means_t" not in d          # the attribute's name, not the old key
    assert "assessment" in d and "status" not in d and "resolution_ok" not in d
    assert "start_policy" in d["options"]["solve"] and "start" not in d["options"]["solve"]

    stale = json.loads(json.dumps(d)); stale["payload_version"] = 1
    assert any("payload_version" in e for e in ns.schema.validate(stale, "payload"))
