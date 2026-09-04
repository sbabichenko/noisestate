"""A tied symmetric game must reproduce the untied solution (catches channel-permutation mistakes)."""
import os, numpy as np, yaml
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__))

def test_tied_symmetric_game_matches_untied():
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch3_two_player.yaml")))
    d["params"]["p2"] = d["params"]["p1"]; d["params"]["r2"] = d["params"]["r1"]
    free = ns.solve(ns.Model.from_dict(d))
    d["ties"] = [["player1", "player2"]]
    tied = ns.solve(ns.Model.from_dict(d))
    assert free.converged and tied.converged
    assert np.abs(free.kernel("X") - tied.kernel("X")).max() < 1e-8
    # the tied copy mirrors the free solution channel for channel (own noise <-> own noise)
    ch = tied.compiled.channels; k1, k2 = tied.action_kernel("D1"), tied.action_kernel("D2")
    assert np.abs(k2[:, ch.index("w2")] - k1[:, ch.index("w1")]).max() < 1e-8
    assert np.abs(k2[:, ch.index("w1")] - k1[:, ch.index("w2")]).max() < 1e-8
    assert np.abs(k1 - free.action_kernel("D1")).max() < 1e-8
    assert abs(free.costs["player1"] - tied.costs["player2"]) < 1e-8
