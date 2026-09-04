import os, numpy as np, pytest, yaml
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__))

def test_ch1_finite_converges_and_matches_cost_to_first_order():
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch1_two_player_finite.yaml")))
    d["horizon"]["nodes"] = 24
    res = ns.solve(ns.Model.from_dict(d))
    print(res.summary())
    assert res.converged and res.residual < 1e-8
    # spec_ch1 (spectral, 16x16 nodes) reports Jvar1 = 0.39664911 for this game
    assert abs(res.costs["player1"] - 0.39664911) < 0.01
    assert abs(res.costs["player1"] - res.costs["player2"]) < 1e-9        # symmetric game
    # strictly causal kernels
    K = res.kernel("D1", "w1"); assert np.all(np.triu(K) == 0)
