import os
import json, os, numpy as np, pytest
import noisestate as ns

REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
REF = os.path.join(REFS, "ch3_p3_p10_r1_r1.json")
HERE = os.path.dirname(os.path.abspath(__file__))

def test_ch3_matches_spectral_solver():
    ref = json.load(open(REF))
    res = ns.solve(os.path.join(HERE, "..", "examples", "ch3_two_player.yaml"), verbose=True)
    print(res.summary())
    lag = np.array(ref["lag"]); assert np.allclose(lag, res.ages, atol=1e-12)
    col = lambda d: np.stack([np.array(d[f"ch{k}"]) for k in range(3)], axis=1)
    x_ref = col(ref["x"]); d1_ref = col(ref["calD1"]); d2_ref = col(ref["calD2"])
    x = res.kernel("X"); d1 = res.action_kernel("D1"); d2 = res.action_kernel("D2")
    print("x    max|diff|", np.abs(x - x_ref).max(), " scale", np.abs(x_ref).max())
    print("calD1 max|diff|", np.abs(d1 - d1_ref).max(), " scale", np.abs(d1_ref).max())
    print("calD2 max|diff|", np.abs(d2 - d2_ref).max(), " scale", np.abs(d2_ref).max())
    assert res.converged
    # window L=3 truncates the tails differently in the two codes (x(L) ~ 0.07); at L=10 they agree to 1e-11
    assert np.abs(x - x_ref).max() < 1e-4
    assert np.abs(d1 - d1_ref).max() < 1e-4
    assert np.abs(d2 - d2_ref).max() < 1e-4


REF10 = REF.replace("ch3_p3_p10_r1_r1.json", "ch3_L10.json")

def test_ch3_wide_window_agrees_to_machine_precision():
    import yaml
    ref = json.load(open(REF10))
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch3_two_player.yaml")))
    d["horizon"]["nodes"] = 64; d["horizon"]["window"] = 10.0
    res = ns.solve(ns.Model.from_dict(d))
    col = lambda dd: np.stack([np.array(dd[f"ch{k}"]) for k in range(3)], axis=1)
    assert res.converged
    assert np.abs(res.kernel("X") - col(ref["x"])).max() < 1e-9
    assert np.abs(res.action_kernel("D1") - col(ref["calD1"])).max() < 1e-9
