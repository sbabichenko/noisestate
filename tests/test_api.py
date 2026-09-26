"""The API stage's surface: Numerics apart from the model, the explicit solve(), the engines namespace."""
import json
import os
import warnings

import numpy as np
import pytest

import noisestate as ns
from noisestate import Numerics, Settings
from helpers import example, example_path
from noisestate.schema import PAYLOAD_VERSION
from noisestate.diagnostics import Status

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_the_numerics_block_is_the_only_place_the_grid_is_sized():
    """The grid lives in numerics: alone; a grid field under horizon: is an unknown key, and the cell engine is
    numerics.engine, not a horizon kind."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    assert d["numerics"] == {"nodes": 24} and "nodes" not in d["horizon"]
    m = ns.Model.from_dict(d)
    assert m.numerics == Numerics(nodes=24) and m.numerics.engine is None
    old = {**d, "horizon": {**d["horizon"], "nodes": 24}}; del old["numerics"]
    with pytest.raises(ValueError, match="unknown key"):
        ns.Model.from_dict(old)
    fc = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml"))
    fc.setdefault("numerics", {}).update(nodes=8, engine="cells")
    mc = ns.Model.from_dict(fc)
    assert mc.horizon.kind == "finite" and mc.numerics.engine == "cells"
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
    # a Numerics field given directly is laid over the numerics (solve(m, nodes=12) is solve(m, {"nodes": 12}))
    r3 = ns.solve(m, nodes=12)
    assert r3.numerics.nodes == 12 and r3.costs == ns.solve(m, {"nodes": 12}).costs
    assert ns.solve(m, {"nodes": 12}, unit_range=2.0).numerics.unit_range == 2.0
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(m, bogus=1)
    with pytest.raises(TypeError, match="not the stationary one"):
        ns.solve(m, past=r)
    r3 = ns.solve(m.with_numerics(nodes=8))
    assert r3.compiled.N == 8 and r3.numerics == r3.model.numerics.resolved("stationary").merged(Numerics(tol=1e-10, damping=0.6, max_newton=60, variable="actions"))
    fine = r3.refine(); assert fine.nodes == 12 and r3.refinement is fine


def test_signal_transforms_are_immutable_validated_and_serialisable():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    public = m.with_signal("public_flow", drift={"D1": 1, "D2": 1}, noise={"wf": 0.5})
    assert "wf" not in m.channels and "wf" in public.channels
    assert all("public_flow" not in [r.name for r in a.signals] for a in m.agents)
    assert all("public_flow" in [r.name for r in a.signals] for a in public.agents)
    assert ns.Model.from_dict(public.to_dict()).to_dict() == public.to_dict()

    one = m.with_signal("extra", drift={"X": 1}, noise={"we": 1}, audience="player1")
    assert "extra" in [r.name for r in one.agents[0].signals]
    assert "extra" not in [r.name for r in one.agents[1].signals]
    with pytest.raises(ValueError, match="unknown agent"):
        m.with_signal("extra", drift={"X": 1}, noise={"we": 1}, audience="nobody")
    with pytest.raises(ValueError, match="already has a signal"):
        m.with_signal("y1", drift={"X": 1}, noise={"we": 1}, audience="player1")


def test_category_verdict_excludes_by_root_not_by_full_name():
    """The equilibrium category holds `stability` beside the per-agent `second_order:<agent>` rows.
    Excluding on the full name would silently match nothing for the suffixed ones, so the exclusion
    is on the root -- the part before the ':', which is what the category itself is computed from."""
    m = ns.Model.from_dict({
        "name": "verdicts", "channels": ["w0", "w1"],
        "states": {"X": {"drift": {"X": -1, "D": 1}, "noise": {"w0": 1}}},
        "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1}, "noise": {"w1": 1}}},
                          "loss": [[1, "X", "X"], [1, "D", "D"]]}},
        "horizon": {"kind": "stationary", "window": 4}, "numerics": {"nodes": 8}})
    res = ns.solve(m)
    res.stability()
    names = [r["name"] for r in res.diagnostics.rows if r["category"] == "equilibrium"]
    assert "stability" in names and any(n.startswith("second_order:") for n in names)

    # the per-agent rows collapse to their ROOT, which is what a policy names and a status keys on
    assert res.diagnostics.statuses["second_order"] is Status.PASSED
    assert [r["name"] for r in res.diagnostics.by_category("equilibrium")] == names
    assert res.diagnostics.by_category("no such category") == ()


def test_compare_summary_keeps_the_assessment_and_dynamics_in_their_own_columns():
    m = ns.Model.from_dict({
        "name": "cols", "channels": ["w0", "w1"],
        "states": {"X": {"drift": {"X": -1, "D": 1}, "noise": {"w0": 1}}},
        "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1}, "noise": {"w1": 1}}},
                          "loss": [[1, "X", "X"], [1, "D", "D"]]}},
        "horizon": {"kind": "stationary", "window": 4}, "numerics": {"nodes": 8}})
    study = ns.compare({"base": m, "a considerably longer scenario name": m.with_signal(
        "public", drift={"X": 1}, noise={"wp": 1})}, baseline="base", stability=True)
    lines = study.summary().splitlines()
    head = lines[0]
    for column in ("assessment", "full response"):
        assert column in head
    # every column starts under its own header, whatever the scenario names and verdict words are
    for column in study.HEAD[3:]:
        at = head.index(column)
        assert all(line[at:at + 1] != " " for line in lines[1:]), f"{column} column is misaligned"


def test_compare_keeps_results_and_separates_costs_from_dynamics():
    base = ns.Model.from_dict({
        "name": "comparison", "channels": ["w0", "w1"],
        "states": {"X": {"drift": {"X": -1, "D": 1}, "noise": {"w0": 1}}},
        "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1}, "noise": {"w1": 1}}},
                          "loss": [[1, "X", "X"], [1, "D", "D"]]}},
        "horizon": {"kind": "stationary", "window": 4}, "numerics": {"nodes": 8},
    })
    more = base.with_signal("public", drift={"X": 1}, noise={"wp": 1})
    study = ns.compare({"base": base, "more information": more}, baseline="base", stability=True)
    assert study["base"].result.model is base
    assert study["base"].total_change == 0 and study["more information"].cost_changes["a"]["value"] == study["more information"].result.costs["a"]
    #  The classification is gated on VERIFICATION, not on convergence.  While D4 is open no point
    #  is verified, so the words are withheld -- and the EVIDENCE is carried in their place, which
    #  is the whole difference from the old behaviour of classifying whatever it was handed.
    dyn = study["base"].dynamics
    assert dyn["verified"] is False and dyn["full_response"] is None
    assert any("residual criterion" in why for why in dyn["unverified_reasons"])
    assert dyn["fixed_point_residual"] >= 0 and dyn["residual_norm"] and dyn["residual_tolerance"] is None
    assert dyn["radius"] < 1 and dyn["stable"] is True      # the spectrum itself is still reported
    payload = study.to_dict(include_results=False)
    assert payload["baseline"] == "base" and len(payload["scenarios"]) == 2
    import json
    json.dumps(study.to_dict())
    assert "scenario" in study.summary() and "not checked" not in study.summary()
    assert "not verified" in study.summary()               # stability ran; the point is not established
    with pytest.raises(ValueError, match="incompatible"):
        ns.compare({"stationary": base, "finite": base.with_finite(4)})


def test_engines_namespace_and_the_cell_engine_by_numerics():
    from noisestate import engines
    assert engines.stationary is engines.stationary and engines.spectral is engines.spectral and engines.cells is engines.cells
    assert set(engines.ENGINE_CLASSES) == {"stationary", "spectral", "cells"}
    m = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml"))
    #  the public factory returns the engine; the resolved Numerics come off the engine's model
    S = engines.solver(m, {"engine": "cells", "nodes": 8})
    num = S.model.numerics
    assert isinstance(S, engines.cells) and num.engine == "cells" and S.solve().kind == "finite_cells"
    with pytest.raises(TypeError, match="cell engine"):
        engines.solver(m, {"engine": "cells"}, past=[])
    rows = ns.sweep(m, "p1", [3.0, 4.0], numerics={"nodes": 6})
    assert all(r.result.compiled.g.nt == 6 for r in rows)


@pytest.mark.parametrize("numerics", [{"engine": "spectral", "nodes": 12}, {"engine": "cells", "nodes": 12}])
def test_one_result_reads_the_same_on_every_engine(numerics):
    """A consumer reads a kernel with its axes without knowing the engine: every axis in res.axes has the
    length of the kernel's node axis (or axes), the maps' axes are there too, and the aliases hold."""
    m = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml"))
    r = ns.solve(m, numerics).require_converged()
    K = r.kernel("X", "w1")
    node_axes = {k: v for k, v in r.axes.items() if k != "maps"}
    assert K.ndim == (2 if numerics["engine"] == "cells" else 1) and all(len(v) == K.shape[0] for v in node_axes.values())
    assert "time" in node_axes and "shock_time" in node_axes and r.times is not None and len(r.times) == len(r.paths["means"]["X"])
    assert r.axes["maps"]["player1"]["y1"]["map_time"].shape[0] == r.maps["player1"].shape[2]
    #  one public type; the concrete class is the engine's and is not part of the API
    assert isinstance(r, ns.Result) and type(r) is not ns.Result
    assert not hasattr(r, "Z") and not hasattr(r, "iterations")      # the 0.5 spellings, removed in 0.6
    assert r.diagnostics.flags == ()
    #  the rows carry the presentation fields; the statuses carry the verdict, keyed on the root
    assert all({"code", "category", "severity", "meaning"} <= set(row) for row in r.diagnostics.rows)
    assert r.diagnostics.statuses["converged"] is Status.PASSED
    #  Acceptance is NOT the same on every engine, and must not be: the cell engine computes
    #  neither a representation error nor a second-order form, so under PUBLICATION it reports
    #  UNSUPPORTED and is refused.  An engine must not pass a verdict because it cannot test it.
    verdict = r.diagnostics.assess()
    if numerics["engine"] == "cells":
        assert not verdict.accepted and verdict.uncomputed == ("resolution", "second_order")
        assert {b.status for b in verdict.blocking} == {Status.UNSUPPORTED}
        assert r.require_ok(ns.Policy.EXPLORATORY) is r          # a weaker use is legitimate, and named
    else:
        assert verdict.accepted is True and verdict.uncomputed == ()
    diagnostic = r.to_dict()["diagnostics"][0]
    assert diagnostic["code"] == "converged" and diagnostic["category"] == "solve"
    assert diagnostic["severity"] == "ok" and diagnostic["meaning"] and "suggested_options" in diagnostic
    assert r.cost_kind.startswith("discounted") and set(r.cost_parts["player1"]) == {"variance", "mean"}
    p = r.to_dict()
    assert p["payload_version"] == PAYLOAD_VERSION and p["engine"] == numerics["engine"] and set(p["axes"]) == set(node_axes)
    assert p["assessment"]["accepted"] is verdict.accepted and p["assessment"]["policy"] == "publication"


def test_the_stationary_result_and_a_transition_read_the_same_way():
    s = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8})
    assert list(s.axes) == ["age", "maps"] and s.times is None and s.paths == {} and "window_tail" in s.extra
    assert s.kernel("X").shape == (len(s.axes["age"]), 3) and s.diagnostics.assess().accepted is False and any("WINDOW" in f for f in s.diagnostics.flags)
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
    code = main(["transition", os.path.join(EX, "ch3_two_player.yaml"), str(tmp_path / "new.yaml"), "--T", "6", "--nodes", "5",
                 "-o", str(out), "--max-evaluations", "2"])
    p = json.load(open(out))
    assert code == 1 and p["kind"] == "transition" and p["options"]["solve"]["start_policy"] == "stationary" and p["numerics"]["nodes"] == 5
    assert ns.schema.validate(p, "payload") == []
    recovery = capsys.readouterr().out
    assert "Next: retry with" in recovery and "--past-window 6" in recovery
    continuation = next(d for d in p["diagnostics"] if d["name"] == "continuation window")
    assert continuation["trend"]["assessment"] == "inconclusive"       # do not infer nonexistence from a coarse continuation
    pytest.importorskip("matplotlib")
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8}, max_evaluations=3)
    with open(tmp_path / "res.json", "w") as fh:
        json.dump(r.to_dict(), fh)
    assert main(["plot", str(tmp_path / "res.json"), str(tmp_path / "res.png")]) == 1 and (tmp_path / "res.png").exists()

    from noisestate.plotting import plot_payload
    stationary = r.to_dict()
    stationary["kernels"]["D1"] = {ch: [0.0] * len(stationary["axes"]["age"]) for ch in stationary["channels"]}
    fig = plot_payload(stationary, str(tmp_path / "zero.png"))
    assert any(ax.get_title().startswith("D1:") for ax in fig.axes)       # a zero kernel remains visible
    assert "WARNING: failed checks" in fig._suptitle.get_text()           # an under-resolved plot cannot look authoritative

    #  The plain finite TRIANGLE, which is the default engine for a finite model and what the
    #  README's first workflow produces.  The suite plotted a stationary, a cells and a transition
    #  payload and never this one, so plot_payload read horizon["window"] for its time axis -- a key
    #  a finite horizon stopped carrying when 0.8 split T from the lag window -- and raised KeyError
    #  for every finite payload.  The transition case hid it by short-circuiting on kind first.
    tri = ns.solve(os.path.join(EX, "ch1_two_player_finite.yaml"), {"nodes": 6}, max_evaluations=1).to_dict()
    assert "window" not in tri["horizon"] and tri["horizon"]["T"] == 1.0
    fig = plot_payload(tri, str(tmp_path / "triangle.png"))
    kernel_axes = [ax for ax in fig.axes if " on " in ax.get_title()]
    assert kernel_axes and all(ax.lines for ax in kernel_axes)            # curves drawn, not an empty frame
    assert (tmp_path / "triangle.png").exists()

    cells = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml"))
    cells["numerics"] = {"engine": "cells", "nodes": 6}
    cp = ns.solve(cells, max_evaluations=1).to_dict()
    fig = plot_payload(cp, str(tmp_path / "cells.png"))
    image_axes = [ax for ax in fig.axes if ax.get_title()]
    assert len(image_axes) == len(cp["kernels"]) * len(cp["channels"])
    assert all(len(ax.images) == 1 for ax in image_axes)                  # every cell kernel has data, not an empty panel
    mask = np.ma.getmaskarray(image_axes[0].images[0].get_array())
    assert mask.any() and not mask[-1, 0]                                 # shocks after the observation time are visibly excluded

    fig = plot_payload(p, str(tmp_path / "transition.png"))
    titles = [ax.get_title() for ax in fig.axes]
    assert sum("expected loss" in x for x in titles) == len(p["agents"])
    assert sum("belief error variance" in x for x in titles) == len(p["agents"])
    kernel_axes = [ax for ax in fig.axes if " on " in ax.get_title()]
    assert kernel_axes and all(ax.patches for ax in kernel_axes)          # the pre-transition band is shaded
    assert all(line.get_marker() == "." for ax in kernel_axes for line in ax.lines
               if line.get_label().startswith("t="))                     # saved-node interpolation is visually explicit
    assert all(float(line.get_label().split("=")[1]) <= p["T"] + 1e-12
               for line in kernel_axes[0].lines if line.get_label().startswith("t="))


def test_the_zero_start_is_explicit_with_a_continuation():
    old = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8}).require_converged()
    new = old.model.with_params(p1=10.0).with_finite(6.0).with_numerics(nodes=5)
    d = ns.solve(new, past=old, continuation="stationary", max_evaluations=1)
    z = ns.solve(new, past=old, continuation="stationary", max_evaluations=1, start_policy="zero")
    e = ns.solve(new, past=old, continuation="end", max_evaluations=1)
    assert d.solve_kw["start_policy"] == "stationary" and z.solve_kw["start_policy"] == "zero" and e.solve_kw["start_policy"] == "zero"
    assert d.residual != z.residual
    rows = ns.sweep(new.with_transition(6.0, past={"model": old.model.to_dict()}, continuation="stationary"), "p1",
                    [10.0], solve_kw={"max_evaluations": 1})
    assert rows[0].result.solve_kw["start_policy"] == "stationary"


def test_a_saved_transition_keeps_its_past_beside_it():
    """A past under the saved file's own directory is written relative to it, so a copied pair reads the
    past beside the copy; a past elsewhere keeps the absolute path, which names that file and no other.
    Before 0.6.9 both were absolute, so a copied pair silently read the original past."""
    import shutil, tempfile, yaml
    d = tempfile.mkdtemp()
    ns.load(os.path.join(EX, "ch3_two_player.yaml")).save(os.path.join(d, "past.yaml"))
    tr = ns.load(os.path.join(EX, "ch3_precision_change.yaml"))._patch_horizon(past={"model": os.path.join(d, "past.yaml")})
    tr.save(os.path.join(d, "tr.yaml"))
    assert yaml.safe_load(open(os.path.join(d, "tr.yaml")))["horizon"]["past"] == {"model": "past.yaml"}

    moved = tempfile.mkdtemp()                       # the pair copied elsewhere reads the copy, not the original
    for f in ("past.yaml", "tr.yaml"):
        shutil.copy(os.path.join(d, f), moved)
    edited = yaml.safe_load(open(os.path.join(moved, "past.yaml"))); edited["params"]["p1"] = 99.0
    yaml.safe_dump(edited, open(os.path.join(moved, "past.yaml"), "w"))
    m = ns.load(os.path.join(moved, "tr.yaml"))
    assert ns.load(m.horizon.past["model"]).params["p1"] == 99.0

    elsewhere = tempfile.mkdtemp()                    # a past outside the directory stays absolute
    ns.load(os.path.join(EX, "ch3_precision_change.yaml")).save(os.path.join(elsewhere, "tr.yaml"))
    saved = yaml.safe_load(open(os.path.join(elsewhere, "tr.yaml")))["horizon"]["past"]["model"]
    assert os.path.isabs(saved) and os.path.exists(saved)


def test_a_signal_is_one_type_across_both_workflows():
    """PART 8.1: the expression form and the transform form say a row with the SAME object.
    Signal.compile() already produced the block with_signals consumes, so the object form is not a
    parallel path -- it compiles to the same thing the file form writes."""
    from noisestate import Signal, State, shocks
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    w, X = shocks("wf"), State("X")
    by_object = m.with_signal(Signal("flow", X + w.wf))
    by_name = m.with_signal("flow", drift={"X": 1.0}, noise={"wf": 1.0})
    by_keyword = m.with_signal(name="flow", drift={"X": 1.0}, noise={"wf": 1.0})
    assert by_object.to_dict(numeric=True) == by_name.to_dict(numeric=True) == by_keyword.to_dict(numeric=True)
    #  several at once, in either form
    assert m.with_signals([Signal("a", X + w.wf), Signal("b", X + w.wf)]).agents[0].signals[-1].name == "b"
    with pytest.raises(TypeError, match="a sequence must hold Signals"):
        m.with_signals([{"drift": {"X": 1.0}, "noise": {"wf": 1.0}}])


def test_with_signal_refuses_the_ambiguous_calls():
    from noisestate import Signal, State, shocks
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    w, X = shocks("wf"), State("X")
    with pytest.raises(TypeError, match="given twice"):
        m.with_signal("flow", name="flow", drift={"X": 1.0}, noise={"wf": 1.0})
    with pytest.raises(TypeError, match="a Signal OR name/drift/noise"):
        m.with_signal(Signal("f", X + w.wf), drift={"X": 1.0})
    with pytest.raises(TypeError, match="a Signal or the row's name"):
        m.with_signal(7)


def test_removing_a_row_is_the_inverse_of_adding_one():
    """with_signal refuses a name an agent already has rather than replacing it, so there has to be
    a way to say "remove, then add".  The two are inverses down to the CHANNELS: with_signals adds
    the ones a row loads, so without_signal drops the ones left unloaded -- the model refuses a
    channel nothing uses, and a round trip would not validate otherwise."""
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    added = m.with_signal("flow", drift={"X": 1.0}, noise={"wf": 1.0})
    assert "wf" in added.channels and "wf" not in m.channels
    back = added.without_signal("flow")
    #  BOTH forms: numeric=True resolves the parameter expressions to numbers, so on its own it
    #  would pass even if the round trip had lost `sqrt(p1)` and left 1.732... in its place.  The
    #  plain to_dict() is the one that says the SYMBOLIC model came back.
    assert back.to_dict(numeric=True) == m.to_dict(numeric=True)
    assert back.to_dict() == m.to_dict()
    assert "sqrt(p1)" in json.dumps(back.to_dict())                  # the expression, not its value
    assert "wf" not in back.channels

    with pytest.raises(ValueError, match="already has a signal"):
        added.with_signal("flow", drift={"X": 1.0}, noise={"wf": 1.0})
    with pytest.raises(ValueError, match="has no signal named"):
        m.without_signal("nope")
    with pytest.raises(ValueError, match="no rows at all"):
        m.without_signal("y1")
    with pytest.raises(ValueError, match="unknown agent"):
        m.without_signal("y1", audience="nobody")


def test_to_dict_carries_the_behaviour_and_omits_only_provenance():
    """What the file form promises, and what a to_dict() comparison therefore establishes.

    The signal round-trip test asserts two models are equal by comparing to_dict(); that is only
    an equivalence claim if to_dict() carries everything a solve depends on.  It does: definitions
    and ties are there when the model has them -- they are missing from a model with none, which
    is easy to mistake for an omission -- and what is left out is source, remarks and
    deprecations, none of which reach the solver.
    """
    import dataclasses
    rich = ns.load(os.path.join(EX, "ch5_cycle_market.yaml"))
    assert rich.definitions and rich.ties                        # a model that exercises both
    omitted = {f.name for f in dataclasses.fields(ns.Model)} - set(rich.to_dict())
    assert omitted == {"source"}
    back = ns.Model.from_dict(rich.to_dict())
    assert back.to_dict() == rich.to_dict()
    assert len(back.definitions) == len(rich.definitions) and len(back.ties) == len(rich.ties)


# ---------------------------------------------------------------- numerics given directly, and no warnings

def test_solve_takes_the_numerics_fields_directly_again():
    # solve(m, nodes=6) is solve(m, {"nodes": 6}) since 1.1 (refused 0.6 to 1.0, when two routes were one too many;
    # with ModelBuilder gone, the keyword is the plain way to say it)
    m = example("ch3_two_player")
    assert ns.solve(m, nodes=6).costs == ns.solve(m, Numerics(nodes=6)).costs
    assert ns.solve(m, Numerics(nodes=6), settings=Settings(anderson_m=3)).settings.anderson_m == 3   # settings is a Numerics field
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(m, Numerics(nodes=6), bogus=3)
    assert ns.solve(m, Numerics(nodes=6)).numerics.nodes == 6


def test_transition_takes_the_numerics_fields_directly_too():
    old = ns.solve(example("ch3_two_player").with_numerics(nodes=6)).require_converged()
    new = example("ch3_two_player").with_params(p1=10.0)
    assert ns.transition(old, new, 3.0, nodes=6).numerics.nodes == 6
    with pytest.raises(TypeError, match="unknown option"):
        ns.transition(old, new, 3.0, stationary={"nodes": 6})


def test_a_settings_field_passed_to_solve_names_the_settings_route():
    with pytest.raises(TypeError, match=r"Numerics\(settings=\{'anderson_m': \.\.\.\}\)"):
        ns.solve(example_path("ch3_two_player"), anderson_m=3)
    assert ns.solve(example_path("ch3_two_player"), nodes=6, damping=0.5).numerics.damping == 0.5   # a Numerics field is taken


def test_the_package_emits_no_deprecation_warning_in_normal_use():
    """A deprecated name used INSIDE noisestate warns code the user did not write.  This caught a
    real one before the layer was deleted: renaming sweep.make_solver to solver left
    Result._make_solver importing the old name, an ImportError on a path no other test reaches."""
    import dataclasses
    m = example("ch3_two_player").with_numerics(nodes=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        res = ns.solve(m)
        res.require_converged(); res.stability(); res.diagnostics.summary(); res.to_dict()
        assert dataclasses.replace(res, solver_class=None)._make_solver(m) is not None
