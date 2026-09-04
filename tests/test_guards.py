"""Guards against misleading results: refinement check, window tail, sweep continuity, unreferenced
parameters, convention notes, cost labels."""
import os, pytest
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_refinement_check_on_a_resolved_and_an_unresolved_model():
    res = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), refine=True)
    assert res.refinement["resolved"] and res.refinement["cost_change"] < 1e-8
    assert "refinement to" in res.summary() and res.to_dict()["refinement"]["nodes"] > 24
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); d["horizon"]["nodes"] = 4
    coarse = ns.solve(ns.Model.from_dict(d), refine=True)
    assert not coarse.refinement["resolved"] and "NOT RESOLVED" in coarse.summary()
    df = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml"))
    rf = ns.solve(ns.Model.from_dict(df), refine=True)                 # 12 nodes: resolved to 1e-5 in the kernels
    assert rf.refinement["resolved"] and rf.refinement["kernel_change"] < 1e-4
    df["horizon"]["nodes"] = 8
    assert not ns.solve(ns.Model.from_dict(df), refine=True).refinement["resolved"]


def test_window_tail_flag():
    short = ns.solve(os.path.join(EX, "ch3_two_player.yaml"))        # L = 3: the state kernel still moves 2.4% over the last tenth
    assert short.window_tail > 0.02 and "WINDOW TOO SHORT" in short.summary()
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); d["horizon"]["window"] = 10.0; d["horizon"]["nodes"] = 48
    long = ns.solve(ns.Model.from_dict(d))
    assert long.window_tail < 1e-3 and "WINDOW TOO SHORT" not in long.summary()


def test_sweep_reports_continuity():
    from noisestate.sweep import sweep
    rows = sweep(os.path.join(EX, "ch3_two_player.yaml"), "p2", [3.0, 4.0, 5.0, 6.0])
    assert rows[0]["change"] is None and all(r["change"] is not None for r in rows[1:])
    assert not any(r["jump"] for r in rows)


def test_unreferenced_parameter_is_an_error_and_notes_exist():
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml")); d["params"]["sigma_v"] = 1.0        # never used
    with pytest.raises(ValueError, match="never used"):
        ns.Model.from_dict(d)
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    notes = m.notes
    assert any("observes the control" in n for n in notes) and any("myopic" in n for n in notes) and any("flow loss" in n for n in notes)
    res = ns.solve(m); assert res.cost_kind.startswith("stationary") and "flow loss" in res.summary() and res.to_dict()["notes"]
