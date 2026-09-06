"""The transition's API surface (stage 3): the model-file form (horizon.kind transition with its past,
continuation and stationary blocks), noisestate.transition(old, new, T), ModelBuilder.transition, the
validation of the blocks and the CLI round trip."""
import json
import os
import shutil
import numpy as np, pytest
import yaml
import noisestate as ns
from noisestate.cli import main
from helpers import EX, example, slow, prior_model as one_agent


@pytest.fixture(scope="module")
def regime():
    """Chapter 3, p1 = 3 (the shipped file) to p1 = 10, T = 6, 8 nodes: the keyword form from zero and its inputs."""
    m = example("ch3_two_player")
    old = ns.solve(m).check()
    new = m.with_params(p1=10.0).with_horizon(kind="finite", window=6.0, nodes=8)
    return {"m": m, "old": old, "new": new, "zero": ns.solve(new, past=old, continuation="stationary", start="zero")}


def test_file_form_equals_the_keyword_form_bit_for_bit(regime):
    """horizon: {kind: transition, past: {model: ch3_two_player.yaml}} solves the past on the fly and the
    continuation at horizon.nodes, exactly as solve(new, past=old, continuation="stationary")."""
    d = regime["m"].with_params(p1=10.0).to_dict()
    d["horizon"] = {"kind": "transition", "window": 6.0, "past": {"model": EX + "ch3_two_player.yaml"}}; d["numerics"] = {"nodes": 8}
    m = ns.Model.from_dict(d)
    assert m.horizon.kind == "transition" and m.horizon.past == {"model": EX + "ch3_two_player.yaml"} and m.horizon.continuation is None
    assert m.to_dict()["horizon"]["past"] == {"model": EX + "ch3_two_player.yaml"}
    res = ns.solve(m, start="zero"); z = regime["zero"]                # the file form, from zero like z
    assert np.array_equal(res.world, z.world) and res.costs == z.costs and res.evaluations == z.evaluations and res.settled == z.settled
    assert res.past.provenance["name"] == "ch3_two_player" and res.continuation.compiled.grid.n == 8
    assert np.array_equal(res.continuation.world, z.continuation.world)
    # the keyword overrides the file block: another past changes the answer, the same one repeats it
    same = ns.solve(m, past=regime["old"], start="zero")
    assert np.array_equal(same.world, z.world)
    other = ns.solve(m, past=regime["m"].with_params(p1=5.0), max_evaluations=2)
    assert not np.array_equal(other.world[:, :1], z.world[:, :1])
    # `stationary: {nodes: 6}` sizes the continuation's solve; its window must be the past's
    d6 = dict(d); d6["horizon"] = {**d["horizon"], "stationary": {"nodes": 6}}
    r6 = ns.solve(d6, max_evaluations=1)
    assert r6.continuation.compiled.grid.n == 6
    with pytest.raises(ValueError, match="must equal the past's window"):
        ns.solve({**d, "horizon": {**d["horizon"], "stationary": {"window": 4.0}}})
    # continuation: end in the file is the keyword "end"
    e = ns.solve({**d, "horizon": {**d["horizon"], "continuation": "end"}}, max_evaluations=2)
    k = ns.solve(regime["new"], past=regime["old"], continuation="end", max_evaluations=2)
    assert np.array_equal(e.world, k.world) and e.continuation is None


def test_transition_helper_starts_from_the_new_stationary_maps(regime):
    """transition(old, new, T, nodes) solves the past (a path) and the continuation, starts from the new
    stationary maps (start="stationary") and returns the result with past and stationary attached: bit for
    bit the keyword form with the same start, in fewer evaluations than from zero (21 against 23)."""
    m = regime["m"]; z = regime["zero"]
    res = ns.transition(EX + "ch3_two_player.yaml", m.with_params(p1=10.0), T=6.0, numerics={"nodes": 8})
    assert res.model.horizon.kind == "transition" and res.model.horizon.window == 6.0 and res.model.horizon.nodes == 8
    assert res.model.horizon.past == {"model": EX + "ch3_two_player.yaml"} and res.solve_kw["start"] == "stationary"
    assert res.past is not None and res.stationary is res.continuation and res.stationary.model.horizon.kind == "stationary"
    kw = ns.solve(regime["new"], past=regime["old"], continuation="stationary", start="stationary")
    assert np.array_equal(res.world, kw.world) and res.costs == kw.costs and res.evaluations == kw.evaluations
    assert res.evaluations < z.evaluations and res.converged
    assert np.abs(res.world - z.world).max() < 1e-6
    # a result as the old regime: its model is inlined in the block; a Past too
    r2 = ns.transition(regime["old"], m.with_params(p1=10.0), T=6.0, numerics={"nodes": 8}, max_evaluations=1)
    assert r2.model.horizon.past["model"]["name"] == "ch3_two_player" and r2.past.source is regime["old"]
    with pytest.raises(ValueError, match="start='stationary' needs a stationary continuation"):
        ns.solve(regime["new"], past=regime["old"], start="stationary")
    with pytest.raises(ValueError, match="start must be"):
        ns.solve(regime["new"], past=regime["old"], start="warm")
    assert z.solve_kw["start"] == "zero"                        # asked for; the default with a continuation is "stationary"


def test_builder_and_initial_shocks_in_the_file_form():
    """ModelBuilder.transition with a list of initial shocks (continuation "end") is the keyword form with
    past=[shocks] bit for bit; the helper with a list ends the game at T as well."""
    shock = {"name": "xi", "loads": {"X": 0.9}, "rows": {"a.y": 1.0}}
    d = one_agent()
    ref = ns.solve(d, past=[shock])
    b = (ns.ModelBuilder("prior").channel("w0", "w1").state("X", {"X": -0.3, "D": 1.0}, {"w0": 1.0})
         .agent("a", ["D"], [[1.0, "X", "X"], [0.5, "D", "D"]]).signal("a", "y", {"X": 1.5}, {"w1": 1.0})
         .transition(3.0, 12, past=[shock], continuation="end", discount=0.5))
    m = b.build()
    assert m.horizon.kind == "transition" and m.horizon.past == {"initial": [shock]} and m.horizon.continuation == "end"
    res = ns.solve(b)
    assert np.array_equal(res.world, ref.world) and res.costs == ref.costs and res.shocks == ["w0", "w1", "xi"]
    h = ns.transition([shock], d, T=3.0, numerics={"nodes": 12})
    assert np.array_equal(h.world, ref.world) and h.continuation is None and h.solve_kw["start"] == "zero"
    with pytest.raises(ValueError, match="needs a past with a window"):
        ns.solve({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"initial": [shock]}, "continuation": "stationary"}})


def test_transition_validation():
    """The named checks: a past block on kind finite or stationary, kind transition without one, an empty
    block, unknown keys, a bad continuation, a bad stationary block, and the cell engine refusing the kind."""
    d = one_agent()
    with pytest.raises(ValueError, match="belongs to horizon.kind 'transition'"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "past": {"initial": []}}})
    with pytest.raises(ValueError, match="belongs to horizon.kind 'transition'"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "stationary", "continuation": "end"}})
    with pytest.raises(ValueError, match="needs a past block"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition"}})
    with pytest.raises(ValueError, match="needs a `model`"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"initial": []}}})
    with pytest.raises(ValueError, match="unknown key"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"result": 1}}})
    with pytest.raises(ValueError, match="'stationary' or 'end'"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "x.yaml"}, "continuation": "tail"}})
    with pytest.raises(ValueError, match="continuation_nodes"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "x.yaml"}, "stationary": {"nodes": 1}}})
    with pytest.raises(ValueError, match="unknown key"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "x.yaml"}, "stationary": {"L": 3}}})
    with pytest.raises(ValueError, match="horizon.kind must be"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transitions"}})
    m = ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "missing.yaml"}, "continuation": "end"}})
    with pytest.raises(OSError):
        ns.solve(m)
    t = ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": "x.yaml"}}})
    for engine in (ns.FiniteSolver, ns.StationarySolver):
        with pytest.raises(ValueError, match="spectral finite engine only"):
            engine(t)
    with pytest.raises(TypeError, match="horizon.kind 'finite'"):
        ns.solve({**d, "horizon": {**d["horizon"], "kind": "transition", "past": {"model": {**d, "horizon": {"kind": "finite", "window": 1.0}}}}})


def test_cli_round_trip(tmp_path, capsys):
    """A transition file next to its past's file: `validate` prints the transition's structure, `solve -o`
    writes the payload with the past's provenance, and the relative past path is taken from the file's
    directory whatever the working directory."""
    shutil.copy(EX + "ch3_two_player.yaml", tmp_path / "old.yaml")
    d = example("ch3_two_player").with_params(p1=10.0).to_dict()
    d["horizon"] = {"kind": "transition", "window": 6.0, "past": {"model": "old.yaml"}, "continuation": "stationary"}
    d["numerics"] = {"nodes": 8, "continuation_nodes": 8}
    with open(tmp_path / "change.yaml", "w") as fh:
        yaml.safe_dump(d, fh)
    m = ns.load(str(tmp_path / "change.yaml"))
    assert m.horizon.past["model"] == str(tmp_path / "old.yaml")
    assert main(["validate", str(tmp_path / "change.yaml")]) == 0
    out = capsys.readouterr().out
    assert "horizon transition" in out and "past: the stationary model file" in out and "old.yaml" in out
    assert "continuation: the new model's stationary equilibrium" in out and "8 nodes per panel" in out
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path.parent)
        code = main(["solve", str(tmp_path / "change.yaml"), "-o", str(tmp_path / "out.json")])
    finally:
        os.chdir(cwd)
    assert code == 0
    with open(tmp_path / "out.json") as fh:
        payload = json.load(fh)
    assert payload["past"]["kind"] == "stationary" and payload["past"]["name"] == "ch3_two_player" and payload["settled"] is not None
    assert payload["horizon"]["kind"] == "transition" and payload["horizon"]["past"]["model"] == str(tmp_path / "old.yaml")
    assert payload["continuation"]["nodes"] == 8 and set(payload["kernels"]) == {"X", "D1", "D2"}
    assert "TRANSITION NOT SETTLED" in capsys.readouterr().out            # T = 6 is short for this change (settled 6e-4)
    with open(tmp_path / "bad.yaml", "w") as fh:
        yaml.safe_dump({**d, "horizon": {**d["horizon"], "continuation": "tail"}}, fh)
    assert main(["solve", str(tmp_path / "bad.yaml")]) == 2
    assert "horizon.continuation" in capsys.readouterr().err          # the schema reports it first, with its path


def test_transition_gap_is_the_one_shot_from_the_stationary_rules(regime):
    """transition_gap(old, new): the T = 0 pass of the settle march, one best response per agent from the new
    model's stationary rules with the old regime's shocks attached, on the smallest strip the engine builds:
    one unit (numerics.unit, else the smallest lag, else the past's window: Chapter 3 has no lags, so [0, L]
    without a unit, [0, 1] with unit 1).  The same model as its own past sits at the grid's one-shot floor
    (3.8e-6 and 1.3e-5 at 12 nodes on [0, 3]; 8e-11 and 5.5e-10 on the unit-cut [0, 1] strip at 12 nodes, 2.2e-6
    and 9.5e-6 at 8, whatever the past's nodes); the Chapter 3 precision change 3 -> 10 gives 0.585 for player1 and 0.061 for player2 (0.583
    and 0.061 on the unit strip); a tiny change 3 -> 3.03 gives 0.0033, 87 times below 3 -> 6 (0.287): the
    transient is linear in the mismatch."""
    m = regime["m"]; old = regime["old"]
    same = ns.transition_gap(old, m, numerics={"nodes": 12})
    assert set(same) == {"player1", "player2"} and all(1e-7 < v < 3e-5 for v in same.values()), same
    unit = ns.transition_gap(old, m, numerics={"nodes": 8, "unit": 1.0})
    assert all(v < 3e-5 for v in unit.values()), unit
    big = ns.transition_gap(old, m.with_params(p1=10.0), numerics={"nodes": 12})
    assert 0.5 < big["player1"] < 0.7 and 0.05 < big["player2"] < 0.08, big
    mid = ns.transition_gap(old, m.with_params(p1=6.0), numerics={"nodes": 12})
    tiny = ns.transition_gap(old, m.with_params(p1=3.03), numerics={"nodes": 12})
    assert 60 < mid["player1"] / tiny["player1"] < 120 and 60 < mid["player2"] / tiny["player2"] < 120, (mid, tiny)
    with pytest.raises(ValueError, match="needs a past with a window"):
        ns.transition_gap([{"name": "xi", "loads": {"X": 0.9}, "rows": {"a.y": 1.0}}], one_agent())
    big_u = ns.transition_gap(old, m.with_params(p1=10.0), numerics={"nodes": 8, "unit": 1.0})
    assert 0.5 < big_u["player1"] < 0.7 and 0.04 < big_u["player2"] < 0.09, big_u


@pytest.fixture(scope="module")
def marched(regime):
    """Chapter 3, 3 -> 10 at 12 nodes: transition(settle=1e-4), the march from the T = 0 pass (12 s)."""
    return ns.transition(regime["old"], regime["m"].with_params(p1=10.0), settle=1e-4, numerics={"nodes": 12})


@slow()
def test_settle_march_finds_the_window_and_equals_the_explicit_solve(regime, marched):
    """(b) transition(old, new, settle=1e-4) on Chapter 3's precision change 3 -> 10 at 12 nodes: the march
    starts at the T = 0 pass (gap 0.585 from the stationary rules on [0, L]), solves at T = 3 (20 evaluations
    from the stationary start), 6 and 9 (8 and 4 evaluations, warm-started from the previous maps on the new
    grid) and stops at T = 9 with the monitor, the best-response gap on the last window [T - L, T], at 9.3e-7:
    the gaps at T = 3, 6, 9 are 0.577, 9.2e-4, 9.3e-7, a factor of 625 then 990 per window (the closed-loop
    rate; the last one meets the 12-node floor).  T = 9 is the smallest T whose explicit solve settles under
    the tolerance (settled 2.7e-6; at T = 6, 4.8e-4 does not), and the monitor is of the order of `settled`
    (9.3e-7 against 2.7e-6: the best-response measure against the maps').  The final solution equals the
    explicit solve at T = 9 on the same grid and closure to 5e-8 on the maps at the default tol (3e-11 at tol
    1e-11: the next test); the march's 31 evaluations against 20 for the one explicit solve.  Slow (40 s): the
    6-node march in test_settle_march_same_model_and_max_window is the fast copy."""
    res = marched; m = regime["m"]; old = regime["old"]
    assert res.kind == "transition" and res.march_stop == "settled" and res.extra["window"] == 9.0 and res.march_settle == 1e-4
    rows = res.march
    assert [r["T"] for r in rows] == [0.0, 3.0, 6.0, 9.0] and rows[0]["evaluations"] == 0
    assert all(r["monitor"] == "[T - L, T]" for r in rows[1:])
    assert 0.5 < max(rows[0]["gap"].values()) < 0.7
    gaps = [max(r["gap"].values()) for r in rows[1:]]
    assert 0.5 < gaps[0] < 0.7 and 5e-4 < gaps[1] < 2e-3 and 2e-7 < gaps[2] < 3e-6
    factors = [gaps[0] / gaps[1], gaps[1] / gaps[2]]
    assert 300 < factors[0] < 1500 and 300 < factors[1] < 3000, factors
    assert rows[1]["evaluations"] > 2 * rows[2]["evaluations"] >= rows[3]["evaluations"] and 1 <= rows[3]["polish"] <= 3
    assert [r["unknowns"] for r in rows] == [0, 576, 864, 576]
    assert res.settled < 1e-4 and all(r["ok"] for r in res.diagnose() if r["name"] == "settled")
    assert 0.25 < gaps[2] / res.settled < 4
    # the explicit solve at the same T, grid and closure
    ex = ns.transition(old, m.with_params(p1=10.0), T=9.0, numerics={"nodes": 12}, continuation=res.continuation)
    assert ex.compiled.g.nt == res.compiled.g.nt and ex.compiled.N == res.compiled.N and ex.compiled.T == res.compiled.T
    assert max(np.abs(ex.maps[k] - res.maps[k]).max() / np.abs(res.maps[k]).max() for k in res.maps) < 2e-7
    assert max(abs(ex.costs[k] - res.costs[k]) for k in ex.costs) < 1e-7 and abs(ex.settled - res.settled) < 1e-7
    assert sum(r["evaluations"] for r in rows) < 2.5 * ex.evaluations
    e6 = ns.transition(old, m.with_params(p1=10.0), T=6.0, numerics={"nodes": 12}, continuation=res.continuation)
    assert ex.settled < 1e-4 < e6.settled and 0.25 < gaps[1] / e6.settled < 4, (e6.settled, ex.settled)
    d = res.to_dict()
    assert d["window"] == 9.0 and d["march_stop"] == "settled" and len(d["march"]) == 4 and d["march"][2]["gap"]["player1"] > 0


@slow()
def test_settle_march_equals_the_explicit_solve_at_a_tight_tol(regime):
    """(b) at tol 1e-11: the march's final maps and the explicit solve's agree to 4e-11 (each solve then takes
    12 to 28 evaluations).  The local step alone leaves 2e-8 behind (the step to T = 9 fixes the strategies before
    T - 2L = 3 at the T = 6 solve's, where the T = 6 handover, 9.2e-4 on [3, 6], had leaked back at the closed-loop
    rate); the polish on the whole strip, five evaluations at this tol (one at the default), removes it."""
    old = regime["old"]; new = regime["m"].with_params(p1=10.0); tol = 1e-11
    res = ns.transition(old, new, settle=1e-4, numerics={"nodes": 12}, tol=tol)
    ex = ns.transition(old, new, T=res.extra["window"], numerics={"nodes": 12}, continuation=res.continuation, tol=tol)
    assert max(np.abs(ex.maps[k] - res.maps[k]).max() / np.abs(res.maps[k]).max() for k in res.maps) < 1e-10
    assert 2 <= res.march[-1]["polish"] <= 8 and "polish" not in res.march[-2]


def test_settle_march_same_model_and_max_window(regime, tmp_path, capsys):
    """(a) The same model as its own past: the T = 0 pass is under the tolerance (1.3e-5 at 12 nodes on [0, 3];
    9.5e-6 at 8 nodes on the unit strip [0, 1] with step=1 and unit 1), so the march stops there without solving:
    the result is the continuation's stationary maps on the first strip built (window 0, one march row, no
    evaluation, settled 0, the costs the stationary flow's).  (b, the fast copy) 3 -> 10 at 6 nodes with settle
    5e-2, above that grid's floor (the same-model gap on the first strip [0, 3], res.march_floor: 3.3e-2 at 6
    nodes, 2.5e-3 at 8, 7.8e-6 at 12; the first strip's window holds the band's tip, where the one-shot floor is
    worst, so this floor is above the last window's at a later T: 3.7e-3 at 6 nodes, 1.8e-4 at 8): T = 3 then
    T = 6, where the gap is 4.0e-3, and the maps equal the explicit T = 6 solve's to 1e-6.  (c) A tolerance below the floor stops the march at once, march_stop "floor" with the stationary result
    and the `settle floor` row (settle 1e-4 at 8 nodes: nothing marched); max_window=1 with settle 3e-3 at 8 nodes
    stops at T = 3 with the gap at 0.58: the settled flag stays and names the march's stop.  (d) The CLI:
    `transition old new --settle TOL` prints the march; exactly one of --window and --settle.  The file form:
    horizon.settle in place of window (exactly one), solved by solve() through the march, bit for bit the
    helper's."""
    m = regime["m"]; old = regime["old"]; new = m.with_params(p1=10.0)
    same = ns.transition(old, m, settle=1e-4, numerics={"nodes": 12})
    assert same.march_stop == "settled at T = 0" and same.extra["window"] == 0.0 and len(same.march) == 1 and same.evaluations == 0
    assert max(same.march[0]["gap"].values()) < 1e-4 and same.settled == 0.0 and same.march_floor is None and same.converged
    assert all(abs(same.excess_costs[a]) < 1e-6 for a in same.costs) and same.to_dict()["window"] == 0.0
    assert all(np.array_equal(same.maps[a][:, :, :same.compiled.N], same.compiled.frozen[a]) for a in same.maps)
    same_u = ns.transition(old, m, settle=1e-4, step=1.0, numerics={"nodes": 8, "unit": 1.0})
    assert same_u.march_stop == "settled at T = 0" and same_u.extra["window"] == 0.0 and same_u.evaluations == 0 and same_u.compiled.T == 1.0
    small = ns.transition(old, new, settle=5e-2, numerics={"nodes": 6})
    assert [r["T"] for r in small.march] == [0.0, 3.0, 6.0] and small.march_stop == "settled" and small.extra["window"] == 6.0
    assert 0.5 < max(small.march[1]["gap"].values()) < 0.7 and 2e-3 < max(small.march[2]["gap"].values()) < 5e-3
    assert small.march[1]["evaluations"] > small.march[2]["evaluations"] and all(r["monitor"] == "[T - L, T]" for r in small.march[1:])
    assert 2e-2 < max(small.march_floor.values()) < 5e-2 and small.extra["settle_floor"] == small.march_floor
    assert [r["unknowns"] for r in small.march] == [0, 144, 216]
    ex6 = ns.transition(old, new, T=6.0, numerics={"nodes": 6}, continuation=small.continuation)
    assert max(np.abs(ex6.maps[k] - small.maps[k]).max() / np.abs(small.maps[k]).max() for k in small.maps) < 1e-6
    fl = ns.transition(old, new, settle=1e-4, numerics={"nodes": 8})
    assert fl.march_stop == "floor" and fl.extra["window"] == 0.0 and len(fl.march) == 1 and fl.evaluations == 0
    assert 2e-3 < max(fl.march_floor.values()) < 4e-3
    row = next(r for r in fl.diagnose() if r["name"] == "settle floor")
    assert not row["ok"] and "nothing was marched" in row["flag"] and row["advice"] == "raise numerics.nodes"
    cap = ns.transition(old, new, settle=3e-3, max_window=1, numerics={"nodes": 8})
    assert cap.march_stop == "max_window" and cap.extra["window"] == 3.0 and cap.settled > 1e-4 and [r["T"] for r in cap.march] == [0.0, 3.0]
    row = next(r for r in cap.diagnose() if r["name"] == "settled")
    assert not row["ok"] and "stopped at max_window" in row["flag"] and row["advice"] == "raise max_window"
    assert "TRANSITION NOT SETTLED" in cap.summary() or not cap.status["ok"]
    # validation
    with pytest.raises(ValueError, match="exactly one of T"):
        ns.transition(old, new, T=6.0, settle=1e-4)
    with pytest.raises(ValueError, match="exactly one of T"):
        ns.transition(old, new)
    with pytest.raises(ValueError, match="belong to the march"):
        ns.transition(old, new, T=6.0, step=3.0)
    with pytest.raises(ValueError, match="stationary continuation"):
        ns.transition(old, new, settle=1e-4, continuation="end")
    # the file form
    d = new.to_dict()
    d["horizon"] = {"kind": "transition", "settle": 5e-2, "past": {"model": EX + "ch3_two_player.yaml"}}; d["numerics"] = {"nodes": 6}
    fm = ns.Model.from_dict(d)
    assert fm.horizon.settle == 5e-2 and fm.to_dict()["horizon"] == {"kind": "transition", "discount": 0.0, "settle": 5e-2, "past": d["horizon"]["past"]}
    assert "settle" not in fm.with_horizon(window=6.0, settle=None).to_dict()["horizon"]
    fr = ns.solve(fm, max_evaluations=40)
    hr = ns.transition(EX + "ch3_two_player.yaml", new, settle=5e-2, numerics={"nodes": 6}, max_evaluations=40)
    assert fr.extra["window"] == hr.extra["window"] and np.array_equal(fr.world, hr.world) and [r["evaluations"] for r in fr.march] == [r["evaluations"] for r in hr.march]
    with pytest.raises(ValueError, match="exactly one of window"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "window": 6.0}})
    with pytest.raises(ValueError, match="belongs to horizon.kind 'transition'"):
        ns.Model.from_dict({**d, "horizon": {"kind": "finite", "settle": 1e-4}})
    with pytest.raises(ValueError, match="positive tolerance"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "settle": -1.0}})
    with pytest.raises(ValueError, match="needs continuation 'stationary'"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "continuation": "end"}})
    with pytest.raises(ValueError, match="init and start do not apply"):
        ns.solve(fm, start="zero")
    # the CLI
    shutil.copy(EX + "ch3_two_player.yaml", tmp_path / "old.yaml")
    with open(tmp_path / "new.yaml", "w") as fh:
        yaml.safe_dump(new.to_dict(), fh)
    assert main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml"), "--settle", "5e-2", "--nodes", "6",
                 "--max-window", "1", "-o", str(tmp_path / "out.json")]) == 0          # converged (the exit code is convergence; the flag is in the summary)
    out = capsys.readouterr().out
    assert "settle march: window 3 (max_window)" in out and "T = 0:" in out and "wrote" in out
    with open(tmp_path / "out.json") as fh:
        payload = json.load(fh)
    assert payload["window"] == 3.0 and payload["march_stop"] == "max_window" and len(payload["march"]) == 2
    assert payload["march"][1]["unknowns"] == 144 and 2e-2 < max(payload["settle_floor"].values()) < 5e-2
    with pytest.raises(SystemExit):
        main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml")])
    with pytest.raises(SystemExit):
        main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml"), "--window", "6", "--settle", "1e-4"])


@slow()
def test_settle_march_by_unit_steps(regime):
    """step=1 with unit 1 on 3 -> 10 at 6 nodes: the march visits T = 1, 2, 3 (the monitor's window [0, T] holds
    the initial transient, so the gap stays near 0.6) and max_window=1 stops it at T = 3; the default step is one
    window (T = 3 first).  settle 0.5 keeps the 6-node floor (3.3e-2 on [0, 3]) out of the way.  Slow (10 s):
    the unit-cut strips below the window are dear, which is why the default is a window."""
    old = regime["old"]; new = regime["m"].with_params(p1=10.0)
    units = ns.transition(old, new, settle=0.5, max_window=1, step=1.0, numerics={"nodes": 6, "unit": 1.0})
    assert [r["T"] for r in units.march] == [0.0, 1.0, 2.0, 3.0] and units.march_stop == "max_window"
    assert all(0.5 < max(r["gap"].values()) < 0.7 for r in units.march)
    assert units.march[0]["monitor"] == "[0, 1] from the stationary rules"
    assert [r["T"] for r in ns.transition(old, new, settle=0.5, max_window=1, numerics={"nodes": 6, "unit": 1.0}).march] == [0.0, 3.0]


def test_excess_cost_tail_and_the_floor_stop(regime):
    """The excess cost's tail past T: res.excess_windows[agent] are the discounted integrals of the excess loss
    over [T - L, T], [T - 2L, T - L], ... (they sum to res.excess_costs); the factor per window comes from the
    loss path's own decay over the last two windows (an explicit solve) or from the march's last two gaps, and
    the tail is E_last r / (1 - r), res.excess_costs_total the sum.  Chapter 3, 3 -> 10 at 6 nodes: the
    explicit T = 6 solve's second window sits at the coarse grid's floor (-1.6e-4 against 2.2e-2; at 12 nodes
    it is 5.85e-6, the slow test), so it has no loss-path factor and no tail, and neither has a single window
    (T = 3); the march (settle 5e-2, T = 3 then 6) has the gap factor and the tail by the formula.  The floor
    stop: settle 1e-6 at 8 nodes is below the grid's floor (the same-model gap 2.5e-3 on the first strip), so
    the march stops before solving, res.march_stop "floor" with the stationary result, the `settle floor` row
    flags the tolerance as below what the grid resolves, and the tail then comes from the loss path (none
    here: nothing marched).  A march that reaches the floor from above (a gap within a factor 2 of it) stops
    the same way with its last solve."""
    old = regime["old"]; new = regime["m"].with_params(p1=10.0)
    ex = ns.transition(old, new, T=6.0, numerics={"nodes": 6})
    for a, E in ex.excess_windows.items():
        assert len(E) == 2 and abs(sum(E) - ex.excess_costs[a]) < 1e-12 and E[1] > 0.02 and -1e-3 < E[0] < 0
    assert ex.excess_tail == {"source": "loss path", "factor": {}, "windows": [(3.0, 6.0), (0.0, 3.0)]}
    assert not ex.excess_costs_tail and not ex.excess_costs_total and "with the tail" not in ex.summary()
    one = ns.transition(old, new, T=3.0, numerics={"nodes": 6}, continuation=ex.continuation)
    assert all(len(E) == 1 and abs(E[0] - one.excess_costs[a]) < 1e-12 for a, E in one.excess_windows.items()) and not one.excess_costs_tail
    d = ex.to_dict(); assert d["excess_tail"]["source"] == "loss path" and d["excess_costs_total"] == {} and len(d["excess_windows"]["player1"]) == 2
    small = ns.transition(old, new, settle=5e-2, numerics={"nodes": 6}, continuation=ex.continuation)
    assert small.excess_tail["source"] == "march gaps" and "with the tail past T" in small.summary()
    for a in small.excess_costs:
        r = small.march[2]["gap"][a] / small.march[1]["gap"][a]
        assert 0 < r < 0.1 and abs(small.excess_tail["factor"][a] - r) < 1e-15
        assert abs(small.excess_costs_tail[a] - small.excess_windows[a][0] * r / (1 - r)) < 1e-15
        assert abs(small.excess_costs_total[a] - small.excess_costs[a] - small.excess_costs_tail[a]) < 1e-15
    fl = ns.transition(old, new, settle=1e-6, numerics={"nodes": 8})
    assert fl.march_stop == "floor" and fl.extra["window"] == 0.0 and [r["T"] for r in fl.march] == [0.0] and fl.settled == 0.0
    assert 2e-3 < max(fl.march_floor.values()) < 4e-3 and fl.evaluations == 0
    row = next(r for r in fl.diagnose() if r["name"] == "settle floor")
    assert row["ok"] is False and "SETTLE BELOW THE GRID'S FLOOR" in row["flag"] and row["advice"] == "raise numerics.nodes" and "8 nodes" in row["flag"]
    assert "SETTLE BELOW THE GRID'S FLOOR" in fl.summary() and fl.excess_tail["source"] == "loss path"
    assert "settle floor" not in [r["name"] for r in ex.diagnose()]


@slow()
def test_excess_cost_sequences_over_the_march_windows(regime, marched):
    """The excess-cost sequences on Chapter 3's 3 -> 10 at 12 nodes over the march's windows: untailed
    res.excess_costs 0.0240215, 0.0240273, 0.0240273 (player1) and 0.0286135, 0.0286190, 0.0286190 (player2) at
    T = 3, 6, 9; the excess per window falls by about 4000 per window (2.40e-2, 5.85e-6, then the floor at
    1e-9), faster than the maps' gap (625 then 990), so the untailed value is converged by T = 6 to 1e-9 and the
    tail is below the floor: 1.4e-9 at T = 6 from the loss path's factor 2.4e-4, -1e-12 at T = 9 from the
    march's gap factor 1.0e-3 (a negative last window at the floor).  The total at T = 6 is within 1e-8 of the
    T = 15 value, against the untailed T = 3 value's 5.8e-6."""
    old = regime["old"]; new = regime["m"].with_params(p1=10.0); cont = marched.continuation
    ex = {T: ns.transition(old, new, T=T, numerics={"nodes": 12}, continuation=cont) for T in (3.0, 6.0, 15.0)}
    for a in ("player1", "player2"):
        e3, e6, e15 = (ex[T].excess_costs[a] for T in (3.0, 6.0, 15.0))
        assert 1e-6 < abs(e3 - e15) < 1e-5 and abs(e6 - e15) < 1e-8 and abs(ex[6.0].excess_costs_total[a] - e15) < 1e-8
        E = ex[6.0].excess_windows[a]
        assert 0.02 < E[1] < 0.03 and 1e-6 < E[0] < 1e-5 and 1e-4 < ex[6.0].excess_tail["factor"][a] < 1e-3 and ex[6.0].excess_costs_tail[a] < 1e-8
        assert marched.excess_tail["source"] == "march gaps" and 1e-4 < marched.excess_tail["factor"][a] < 1e-2
        assert abs(marched.excess_costs_tail[a]) < 1e-10 and abs(marched.excess_costs_total[a] - e15) < 3e-8
