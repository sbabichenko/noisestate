"""The API stage's surface: Numerics apart from the model, the explicit solve(), the engines namespace."""
import os

import numpy as np
import pytest

import noisestate as ns
from noisestate.diagnostics import Status
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


def test_model_transforms_keep_the_naive_observers_the_file_does_not_carry():
    """naive_observers is not a model-file key, so a with_*() that rebuilt through from_dict() alone
    dropped it -- and Model.solve() reads it off the model, so the transformed copy silently solved a
    fully strategic game instead."""
    from noisestate import Param, shocks, State, Control, Agent, Signal
    eps, rho, g1, sV, sZ = Param.many(eps=0.2, rho=0.5, gamma1=1.0, sigma_V=1.0, sigma_Z=1.0)
    w = shocks("wV", "wZ", "w1")
    V = State("V"); P, D1 = Control("P"), Control("D1"); V.drift = sV * w.wV
    mm = Agent("market_maker", [P], [Signal("flow", D1 + sZ * w.wZ)], P**2 - 2 * P * V, myopic=True)
    tr = Agent("trader1", [D1], [Signal("y1", g1 * V - g1 * P + w.w1), Signal("flow", sZ * w.wZ)],
               -D1 * V + D1 * P + eps * D1**2, naive_observers=["market_maker"])
    game = ns.Model("kb", states=[V], agents=[mm, tr], horizon=ns.Stationary(window=8.0, discount=rho),
                    numerics={"nodes": 12})
    want = {"trader1": ["market_maker"]}
    assert game.naive_observers == want
    for made in (game.with_horizon(window=6.0), game.with_numerics(nodes=10), game.with_params(eps=0.3),
                 game.with_signal("extra", drift={"V": 1}, noise={"we": 1}), game.with_finite(4.0),
                 game.with_horizon(window=6.0).with_numerics(nodes=10)):
        assert made.naive_observers == want
    # the field is only the mechanism; what matters is that the solver is still told
    res = game.with_horizon(window=6.0).solve(max_evaluations=1, diagnostics=False)
    assert res.solver_kw.get("naive_observers") == want


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
    assert engines.stationary is ns.StationarySolver and engines.spectral is ns.SpectralFiniteSolver and engines.cells is ns.FiniteSolver
    assert set(ns.ENGINE_CLASSES) == {"stationary", "spectral", "cells"}
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
    assert p["payload_version"] == 1 and p["engine"] == numerics["engine"] and set(p["axes"]) == set(node_axes)
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
    code = main(["transition", os.path.join(EX, "ch3_two_player.yaml"), str(tmp_path / "new.yaml"), "--window", "6", "--nodes", "5",
                 "-o", str(out), "--max-evaluations", "2"])
    p = json.load(open(out))
    assert code == 1 and p["kind"] == "transition" and p["options"]["solve"]["start"] == "stationary" and p["numerics"]["nodes"] == 5
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

    from noisestate.results import plot_payload
    stationary = r.to_dict()
    stationary["kernels"]["D1"] = {ch: [0.0] * len(stationary["axes"]["age"]) for ch in stationary["channels"]}
    fig = plot_payload(stationary, str(tmp_path / "zero.png"))
    assert any(ax.get_title().startswith("D1:") for ax in fig.axes)       # a zero kernel remains visible
    assert "WARNING: failed checks" in fig._suptitle.get_text()           # an under-resolved plot cannot look authoritative

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
    assert all(float(line.get_label().split("=")[1]) <= p["window"] + 1e-12
               for line in kernel_axes[0].lines if line.get_label().startswith("t="))


def test_the_zero_start_is_explicit_with_a_continuation():
    old = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), {"nodes": 8}).require_converged()
    new = old.model.with_params(p1=10.0).with_finite(6.0).with_numerics(nodes=5)
    d = ns.solve(new, past=old, continuation="stationary", max_evaluations=1)
    z = ns.solve(new, past=old, continuation="stationary", max_evaluations=1, start="zero")
    e = ns.solve(new, past=old, continuation="end", max_evaluations=1)
    assert d.solve_kw["start"] == "stationary" and z.solve_kw["start"] == "zero" and e.solve_kw["start"] == "zero"
    assert d.residual != z.residual
    rows = ns.sweep(new.with_horizon(kind="transition", past={"model": old.model.to_dict()}, continuation="stationary"), "p1",
                    [10.0], solve_kw={"max_evaluations": 1})
    assert rows[0]["result"].solve_kw["start"] == "stationary"


def test_a_saved_transition_keeps_its_past_beside_it():
    """A past under the saved file's own directory is written relative to it, so a copied pair reads the
    past beside the copy; a past elsewhere keeps the absolute path, which names that file and no other.
    Before 0.6.9 both were absolute, so a copied pair silently read the original past."""
    import shutil, tempfile, yaml
    d = tempfile.mkdtemp()
    ns.load(os.path.join(EX, "ch3_two_player.yaml")).save(os.path.join(d, "past.yaml"))
    tr = ns.load(os.path.join(EX, "ch3_precision_change.yaml")).with_horizon(past={"model": os.path.join(d, "past.yaml")})
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
