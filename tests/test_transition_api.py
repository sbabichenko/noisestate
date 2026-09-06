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
    model's stationary rules with the old regime's shocks attached, on the smallest strip the engine builds
    ([0, L]: the compile refuses T < L).  The same model as its own past sits at the grid's one-shot floor
    (4.2e-5 and 3.8e-5 at 12 nodes; 1.1e-5 at 16, 2.4e-6 at 24, whatever the past's nodes: the floor is the
    strip's, on the band's collapsing tip, not the solved identity's 2e-9 on the kernels); the Chapter 3
    precision change 3 -> 10 gives 0.585 for player1 and 0.061 for player2; a tiny change 3 -> 3.03 gives
    0.0033, 87 times below 3 -> 6 (0.287): the transient is linear in the mismatch."""
    m = regime["m"]; old = regime["old"]
    same = ns.transition_gap(old, m, numerics={"nodes": 12})
    assert set(same) == {"player1", "player2"} and all(1e-6 < v < 6e-5 for v in same.values()), same
    big = ns.transition_gap(old, m.with_params(p1=10.0), numerics={"nodes": 12})
    assert 0.5 < big["player1"] < 0.7 and 0.05 < big["player2"] < 0.08, big
    mid = ns.transition_gap(old, m.with_params(p1=6.0), numerics={"nodes": 12})
    tiny = ns.transition_gap(old, m.with_params(p1=3.03), numerics={"nodes": 12})
    assert 60 < mid["player1"] / tiny["player1"] < 120 and 60 < mid["player2"] / tiny["player2"] < 120, (mid, tiny)
    with pytest.raises(ValueError, match="needs a past with a window"):
        ns.transition_gap([{"name": "xi", "loads": {"X": 0.9}, "rows": {"a.y": 1.0}}], one_agent())


@pytest.fixture(scope="module")
def marched(regime):
    """Chapter 3, 3 -> 10 at 12 nodes: transition(settle=1e-4), the march from the T = 0 pass (17 s)."""
    return ns.transition(regime["old"], regime["m"].with_params(p1=10.0), settle=1e-4, numerics={"nodes": 12})


def test_settle_march_finds_the_window_and_equals_the_explicit_solve(regime, marched):
    """(b) transition(old, new, settle=1e-4) on Chapter 3's precision change 3 -> 10 at 12 nodes: the march
    starts at the T = 0 pass (gap 0.585 from the stationary rules on [0, L]), solves at T = 3 (20 evaluations
    from the stationary start), 6, 9 and 12 (8, 4 and 4 evaluations, warm-started from the previous maps on the
    new grid) and stops at T = 12 with the monitor, the best-response gap on the window before the last, at
    9.3e-7: the gaps on [0, 3], [3, 6], [6, 9] are 0.577, 9.2e-4, 9.3e-7, a factor of 625 then 990 per window
    (the closed-loop rate; the last one meets the 12-node floor).  The monitor certifies the solve one window
    back, so the march's T = 12 is one window past the smallest T whose explicit solve settles on its last
    window under the tolerance (T = 9: settled 2.7e-6; T = 6: 4.8e-4 does not), and each row's gap_last (the
    same pass on [T - L, T]) is of the order of what the explicit solve's `settled` measures (9.3e-7 against 2.7e-6 at
    T = 9, 9.2e-4 against 4.8e-4 at T = 6: the best-response measure against the maps').  The final solution equals the
    explicit solve at T = 12 on the same grid and closure to 5e-8 on the maps at the default tol (3e-11 at tol
    1e-11: the slow variant); the march's 36 evaluations against 20 for the one explicit solve from the
    stationary start (23 from zero)."""
    res = marched; m = regime["m"]; old = regime["old"]
    assert res.kind == "transition" and res.march_stop == "settled" and res.extra["window"] == 12.0 and res.march_settle == 1e-4
    rows = res.march
    assert [r["T"] for r in rows] == [0.0, 3.0, 6.0, 9.0, 12.0] and rows[0]["evaluations"] == 0
    assert [r["monitor"] for r in rows[1:]] == ["[0, T]", "[T - 2L, T - L]", "[T - 2L, T - L]", "[T - 2L, T - L]"]
    assert 0.5 < max(rows[0]["gap"].values()) < 0.7
    gaps = [max(r["gap"].values()) for r in rows[1:]]
    assert 0.5 < gaps[0] < 0.7 and abs(gaps[1] - gaps[0]) < 1e-3 * gaps[0]           # T = 6 measures [0, 3] again
    assert 5e-4 < gaps[2] < 2e-3 and 2e-7 < gaps[3] < 3e-6
    factors = [gaps[1] / gaps[2], gaps[2] / gaps[3]]
    assert 300 < factors[0] < 1500 and 300 < factors[1] < 3000, factors
    assert rows[1]["evaluations"] > 2 * rows[2]["evaluations"] >= rows[3]["evaluations"] and rows[4]["evaluations"] <= 6
    assert res.settled < 1e-4 and all(r["ok"] for r in res.diagnose() if r["name"] == "settled")
    # the explicit solve at the same T, grid and closure
    ex = ns.transition(old, m.with_params(p1=10.0), T=12.0, numerics={"nodes": 12}, continuation=res.continuation)
    assert ex.compiled.g.nt == res.compiled.g.nt and ex.compiled.N == res.compiled.N and ex.compiled.T == res.compiled.T
    assert max(np.abs(ex.maps[k] - res.maps[k]).max() / np.abs(res.maps[k]).max() for k in res.maps) < 2e-7
    assert max(abs(ex.costs[k] - res.costs[k]) for k in ex.costs) < 1e-7 and abs(ex.settled - res.settled) < 1e-7
    assert sum(r["evaluations"] for r in rows) < 2.5 * ex.evaluations
    assert abs(max(rows[-1]["gap_last"].values()) - ex.settled) < 3e-7          # the handover's gap is what settled measures
    d = res.to_dict()
    assert d["window"] == 12.0 and d["march_stop"] == "settled" and len(d["march"]) == 5 and d["march"][2]["gap_last"]["player1"] > 0


@slow()
def test_settle_march_equals_the_explicit_solve_at_a_tight_tol(regime, marched):
    """(b) at tol 1e-11: the march's final maps and the explicit solve's agree to 3e-11 (36 s; each solve then
    takes 19 to 28 evaluations, the warm start's saving being at the default tol).  And the smallest explicit T
    that settles under the tolerance on its last window is one window back from the march's (T = 9: settled
    2.7e-6; T = 6: 4.8e-4), each row's gap_last of the order of that `settled` (9.3e-7 against 2.7e-6 at T = 9,
    9.2e-4 against 4.8e-4 at T = 6: the best-response measure against the maps')."""
    old = regime["old"]; new = regime["m"].with_params(p1=10.0); tol = 1e-11
    res = ns.transition(old, new, settle=1e-4, numerics={"nodes": 12}, tol=tol)
    ex = ns.transition(old, new, T=res.extra["window"], numerics={"nodes": 12}, continuation=res.continuation, tol=tol)
    assert max(np.abs(ex.maps[k] - res.maps[k]).max() / np.abs(res.maps[k]).max() for k in res.maps) < 1e-10
    rows = marched.march
    e9 = ns.transition(old, new, T=9.0, numerics={"nodes": 12}, continuation=marched.continuation)
    e6 = ns.transition(old, new, T=6.0, numerics={"nodes": 12}, continuation=marched.continuation)
    assert e9.settled < 1e-4 < e6.settled, (e6.settled, e9.settled)
    for r, e in ((rows[3], e9), (rows[2], e6)):
        assert 0.25 < max(r["gap_last"].values()) / e.settled < 4, (r["gap_last"], e.settled)


def test_settle_march_same_model_and_max_window(regime, tmp_path, capsys):
    """(a) The same model as its own past: the T = 0 pass is under the tolerance (4.2e-5 at 12 nodes), so the
    march stops there with the smallest-window solve, T = L = 3, six evaluations from the stationary start.
    (c) max_window=2 on 3 -> 10 at 8 nodes stops at T = 6 with the gap at 0.58: the settled flag stays (6.4e-4
    against settled_tol) and names the march's stop.  (d) The CLI: `transition old new --settle 1e-4` prints the
    march; exactly one of --window and --settle.  The file form: horizon.settle in place of window (exactly one),
    solved by solve() through the march, bit for bit the helper's."""
    m = regime["m"]; old = regime["old"]; new = m.with_params(p1=10.0)
    same = ns.transition(old, m, settle=1e-4, numerics={"nodes": 12})
    assert same.march_stop == "settled at T = 0" and same.extra["window"] == 3.0 and len(same.march) == 2
    assert max(same.march[0]["gap"].values()) < 1e-4 and same.march[1]["evaluations"] <= 8 and same.settled < 1e-4
    cap = ns.transition(old, new, settle=1e-4, max_window=2, numerics={"nodes": 8})
    assert cap.march_stop == "max_window" and cap.extra["window"] == 6.0 and cap.settled > 1e-4
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
    d["horizon"] = {"kind": "transition", "settle": 1e-4, "past": {"model": EX + "ch3_two_player.yaml"}}; d["numerics"] = {"nodes": 8}
    fm = ns.Model.from_dict(d)
    assert fm.horizon.settle == 1e-4 and fm.to_dict()["horizon"] == {"kind": "transition", "discount": 0.0, "settle": 1e-4, "past": d["horizon"]["past"]}
    assert "settle" not in fm.with_horizon(window=6.0, settle=None).to_dict()["horizon"]
    fr = ns.solve(fm, max_evaluations=40)
    hr = ns.transition(EX + "ch3_two_player.yaml", new, settle=1e-4, numerics={"nodes": 8}, max_evaluations=40)
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
    assert main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml"), "--settle", "1e-4", "--nodes", "8",
                 "--max-window", "2", "-o", str(tmp_path / "out.json")]) == 0          # converged (the exit code is convergence; the flag is in the summary)
    out = capsys.readouterr().out
    assert "settle march: window 6 (max_window)" in out and "T = 0:" in out and "wrote" in out
    with open(tmp_path / "out.json") as fh:
        payload = json.load(fh)
    assert payload["window"] == 6.0 and payload["march_stop"] == "max_window" and len(payload["march"]) == 3
    with pytest.raises(SystemExit):
        main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml")])
    with pytest.raises(SystemExit):
        main(["transition", str(tmp_path / "old.yaml"), str(tmp_path / "new.yaml"), "--window", "6", "--settle", "1e-4"])
