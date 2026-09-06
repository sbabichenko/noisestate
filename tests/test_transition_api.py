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
from helpers import EX, example, prior_model as one_agent


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
