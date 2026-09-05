"""The JSON payload carries its provenance and says how delayed-row maps are indexed (2026-09-05 audit)."""
import json, os, time, numpy as np, pytest
import noisestate as ns
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_payload_round_trips_and_rebuilds_the_solve():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    res = ns.solve(m)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["version"] == ns.__version__ and d["name"] == "ch3_two_player" and d["engine"] == "stationary"
    assert d["params"] == {"p1": 3.0, "p2": 10.0, "r1": 1.0, "r2": 1.0}
    assert d["horizon"] == {"kind": "stationary", "discount": 0.0, "window": 3.0, "nodes": 24}
    assert d["options"]["solve"]["start"] == "zero" and d["options"]["solver"]["verbose"] is False
    assert d["agents"]["player2"] == {"controls": ["D2"], "signals": {"y2": {"delay": 0.0, "map_age": d["grid"]["ages"]}}}
    m2 = ns.Model.from_dict(d["model"]); assert m2.to_dict() == m.to_dict()
    again = ns.make_solver(m2, **d["options"]["solver"]).solve(**d["options"]["solve"])
    assert all(abs(again.costs[k] - res.costs[k]) < 1e-12 for k in res.costs)


def test_delayed_row_map_axes_place_the_map():
    """A delayed row's map is not indexed like an undelayed one: the payload's map_age (stationary) and
    map_time (finite, where the map is stored at t - delay) say where each value belongs, and the map is
    zero exactly where those axes leave the window or the horizon."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); d["agents"]["player2"]["signals"]["y2"]["delay"] = 1.0; d["horizon"]["nodes"] = 8
    r = ns.solve(d).check(); p = r.to_dict(); row = p["agents"]["player2"]["signals"]["y2"]
    g = np.array(p["maps"]["player2"]["D2"]["y2"]); age = np.array(row["map_age"]); L = p["horizon"]["window"]
    assert row["delay"] == 1.0 and np.allclose(age, r.ages + 1.0) and "map_age" in p["map_convention"]
    assert (age > L + 1e-9).sum() == 7 and np.abs(g[age > L + 1e-9]).max() == 0.0 and np.abs(g[age < L]).max() > 0.1
    d = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml")); d["horizon"]["nodes"] = 4
    r = ns.solve(d).check(); p = json.loads(json.dumps(r.to_dict())); row = p["agents"]["player2"]["signals"]["y2"]
    g = np.array(p["maps"]["player2"]["D2"]["y2"]); t = np.array(row["map_time"]); T = p["horizon"]["window"]
    assert row["delay"] == 0.25 and np.allclose(t, r.grid.t + 0.25) and np.allclose(row["map_age"], r.grid.a + 0.25)
    assert (t > T + 1e-9).any() and np.abs(g[t > T + 1e-9]).max() == 0.0 and np.abs(g[t <= T]).max() > 0.1
    assert p["agents"]["player1"]["signals"]["y1"]["map_time"] == p["grid"]["t"]        # an undelayed row: the grid's own time


def test_solve_kw_records_the_start_option_and_repeats_the_solve():
    S = ns.make_solver(ns.load(os.path.join(EX, "ch3_two_player.yaml")))
    res = S.solve(start="coarse")
    assert res.solve_kw["start"] == "coarse" and json.dumps(res.solve_kw) and "coarse start" in res.message
    again = S.solve(**res.solve_kw)
    assert again.iterations == res.iterations and again.message == res.message
    assert S.solve().solve_kw["start"] == "zero"


def test_seconds_include_the_diagnostics(monkeypatch):
    S = ns.make_solver(ns.load(os.path.join(EX, "ch3_two_player.yaml")))
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
