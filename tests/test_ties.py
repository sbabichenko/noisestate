"""A tied symmetric game must reproduce the untied solution (catches channel-permutation mistakes)."""
import os, numpy as np
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__))

def test_tied_symmetric_game_matches_untied():
    d = ns.load(os.path.join(HERE, "..", "examples", "ch3_two_player.yaml")).to_dict()
    d["params"]["p2"] = d["params"]["p1"]; d["params"]["r2"] = d["params"]["r1"]
    free = ns.solve(ns.Model.from_dict(d))
    d["ties"] = [["player1", "player2"]]
    tied = ns.solve(ns.Model.from_dict(d))
    assert free.converged and tied.converged
    assert np.abs(free.kernel("X") - tied.kernel("X")).max() < 1e-8
    # the tied copy mirrors the free solution channel for channel (own noise <-> own noise)
    ch = tied.compiled.channels; k1, k2 = tied.kernel("D1"), tied.kernel("D2")
    assert np.abs(k2[:, ch.index("w2")] - k1[:, ch.index("w1")]).max() < 1e-8
    assert np.abs(k2[:, ch.index("w1")] - k1[:, ch.index("w2")]).max() < 1e-8
    assert np.abs(k1 - free.kernel("D1")).max() < 1e-8
    assert abs(free.costs["player1"] - tied.costs["player2"]) < 1e-8


def _neighbours(ties, m=3, horizon=None, coef=0.4):
    """A ring: player i controls its own state X_i and sees it, and sees its neighbour's X_{i+1} as well, so every state
    is read by two players and is public; its loss has a cross term in the neighbour's state."""
    d = {"name": "ring", "shocks": [], "states": {}, "agents": {},
         "horizon": horizon or {"window": 6.0, "discount": 0.2}, "numerics": {"nodes": 16}}
    for i in range(m):
        j = (i + 1) % m
        d["shocks"] += [f"w{i}", f"v{i}", f"u{i}"]
        d["states"][f"X{i}"] = {"drift": {f"X{i}": -0.5, f"D{i}": 1.0}, "noise": {f"w{i}": 1.0}}
        d["agents"][f"p{i}"] = {"controls": [f"D{i}"],
                                "signals": {"y": {"drift": {f"X{i}": 1.0}, "noise": {f"v{i}": 1.0}},
                                            "z": {"drift": {f"X{j}": 0.5}, "noise": {f"u{i}": 1.0}}},
                                "loss": [[1.0, f"X{i}", f"X{i}"], [1.0, f"D{i}", f"D{i}"], [coef, f"D{i}", f"X{j}"]]}
    if ties:
        d["ties"] = [[f"p{i}" for i in range(m)]]
    return d


def test_a_ring_whose_states_are_public_is_tied_by_its_cyclic_relabelling():
    """The tie signature compares public states by name, so a ring moving them (X0 -> X1 -> X2) was refused as "not
    structurally identical" though the relabelling carries every player's problem onto the next one's; a single tie group
    now falls back on that relabelling, found and verified by rebuilding the model (symmetry.find_cyclic_symmetry).  The
    tied solve is the untied one (costs to 3e-13 stationary, 2e-15 on [0, 1]), and a ring that is not symmetric is still
    refused."""
    free = ns.solve(ns.Model.from_dict(_neighbours(False)))
    tied = ns.solve(ns.Model.from_dict(_neighbours(True)))
    assert free.converged and tied.converged
    assert max(abs(tied.costs[k] - free.costs[k]) for k in free.costs) < 1e-9
    assert np.abs(tied.kernel("X1") - free.kernel("X1")).max() < 1e-8
    fin = {"kind": "finite", "T": 1.0, "discount": 0.2}
    a, b = ns.solve(ns.Model.from_dict(_neighbours(False, horizon=fin))), ns.solve(ns.Model.from_dict(_neighbours(True, horizon=fin)))
    assert max(abs(a.costs[k] - b.costs[k]) for k in a.costs) < 1e-9
    bad = _neighbours(True)
    bad["agents"]["p2"]["loss"][2][0] = 0.3                   # p2's cross term differs: no relabelling carries p1 onto p2
    import pytest
    with pytest.raises(ValueError, match="not structurally identical"):
        ns.Model.from_dict(bad)
