"""Guards against misleading results: refinement check, window tail, sweep continuity, unreferenced
parameters, convention notes, cost labels."""
import os, pytest
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def test_refinement_check_on_a_resolved_and_an_unresolved_model():
    res = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), refine=True)
    assert res.refinement.resolved and res.refinement.cost_change < 1e-8
    assert "refinement to" in res.summary() and res.to_dict()["refinement"]["nodes"] > 24
    d = ns.load(os.path.join(EX, "ch3_two_player.yaml")).to_dict(); d.setdefault("numerics", {})["nodes"] = 4
    coarse = ns.solve(ns.Model.from_dict(d), refine=True)
    assert not coarse.refinement.resolved and "NOT RESOLVED" in coarse.summary()
    df = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).to_dict()
    rf = ns.solve(ns.Model.from_dict(df), refine=True)                 # 12 nodes: resolved to 1e-5 in the kernels
    assert rf.refinement.resolved and rf.refinement.kernel_change < 1e-4
    df.setdefault("numerics", {})["nodes"] = 8
    assert not ns.solve(ns.Model.from_dict(df), refine=True).refinement.resolved


def test_window_tail_flag():
    short = ns.solve(os.path.join(EX, "ch3_two_player.yaml"))        # L = 3: the state kernel still moves 2.4% over the last tenth
    assert short.window_tail > 0.02 and "WINDOW TOO SHORT" in short.summary()
    trend = short.window_tail_extrapolation()
    assert trend["assessment"] == "decaying" and trend["predicted_at_double_window"] < short.window_tail
    assert trend["projection_range"] == [0.5 * trend["predicted_at_double_window"], 2.5 * trend["predicted_at_double_window"]]
    d = ns.load(os.path.join(EX, "ch3_two_player.yaml")).to_dict(); d["horizon"]["window"] = 10.0; d.setdefault("numerics", {})["nodes"] = 48
    long = ns.solve(ns.Model.from_dict(d))
    assert long.window_tail < 1e-3 and "WINDOW TOO SHORT" not in long.summary()


def test_sweep_reports_continuity():
    from noisestate.sweep import sweep
    rows = sweep(os.path.join(EX, "ch3_two_player.yaml"), "p2", [3.0, 4.0, 5.0, 6.0])
    assert rows[0].change is None and all(r.change is not None for r in rows[1:])
    assert not any(r.jump for r in rows)


def test_unreferenced_parameter_is_an_error_and_notes_exist():
    d = ns.load(os.path.join(EX, "ch4_kyle_back.yaml")).to_dict(); d["params"]["sigma_v"] = 1.0        # never used
    with pytest.raises(ValueError, match="never used"):
        ns.Model.from_dict(d)
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    notes = m.notes
    assert any("observes the control" in n for n in notes) and any("myopic" in n for n in notes) and any("flow loss" in n for n in notes)
    res = ns.solve(m); assert res.cost_kind.startswith("stationary") and "flow loss" in res.summary() and res.to_dict()["notes"]


def test_an_agent_with_no_signal_rows_is_refused_by_name():
    """Before 0.6.9 this reached the linear algebra: a RuntimeWarning about a divide, three
    ' ** On entry to DSYRK parameter number 10 had an illegal value' lines on stderr from LAPACK, and
    an IndexError out of numpy.  An agent that reads nothing has no strategy to solve for."""
    d = {"shocks": ["w"], "states": {"X": {"drift": {"D": 1.0}, "noise": {"w": 1.0}}},
         "agents": {"a": {"controls": ["D"], "loss": [[1.0, "X", "X"], [1.0, "D", "D"]]}},
         "horizon": {"kind": "stationary", "window": 3.0}, "numerics": {"nodes": 8}}
    with pytest.raises(ValueError, match="agent a has no signal rows"):
        ns.Model.from_dict(d).validate()


def test_guard_advice_names_keys_the_model_file_actually_accepts():
    """The under-resolved guard told every reader to "raise horizon.nodes", which the model file
    REFUSES -- `unknown key(s) ['nodes'] in horizon`.  The key is numerics.nodes.

    The advice drifted because model.horizon.nodes is a real PYTHON attribute: the resolved numerics
    are merged onto the horizon after loading.  So the sentence was true of the object and false of
    the file it was telling someone to edit, which is the file they have.

    Any `raise <block>.<key>` in a guard's flag or advice has to name a key from_dict() accepts.
    """
    import re
    import noisestate as ns
    model_schema = ns.schema("model")["properties"]
    coarse = {"ch3_precision_change": {"nodes": 5}, "ch5_cycle_market": {"nodes": 6}}
    seen = 0
    for name in ns.examples():
        res = ns.solve(ns.example(name), coarse.get(name, {"nodes": 8}))
        for row in res.diagnostics.rows:
            for text in (row.get("flag") or "", row.get("advice") or ""):
                for block, key in re.findall(r"raise (\w+)\.(\w+)", text):
                    seen += 1
                    assert block in model_schema, f"{name}/{row['name']}: no block {block!r} in a model file"
                    props = (model_schema[block].get("properties") or {})
                    assert key in props, (f"{name}/{row['name']}: advises {block}.{key}, which a model "
                                          f"file does not accept; {block} takes {sorted(props)}")
    assert seen > 0, "no guard advice was exercised, so this checked nothing"
