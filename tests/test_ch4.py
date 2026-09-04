import json, os, numpy as np, pytest, yaml
import noisestate as ns

SP = "/tmp/scratch"
HERE = os.path.dirname(os.path.abspath(__file__))

def two_trader(d):
    d = json.loads(json.dumps(d))
    d["params"]["gamma2"] = 1.0
    d["channels"].append("w2")
    d["agents"]["market_maker"]["signals"]["flow"]["drift"]["D2"] = 1.0
    d["agents"]["trader1"]["signals"]["flow"]["drift"] = {"D2": 1.0}
    d["agents"]["trader2"] = {"controls": ["D2"],
                              "signals": {"y2": {"drift": {"V": "gamma2", "P": "-gamma2"}, "noise": {"w2": 1.0}},
                                          "flow": {"drift": {"D1": 1.0}, "noise": {"wZ": "sigma_Z"}}},
                              "loss": [[-1.0, "D2", "V"], [1.0, "D2", "P"], ["eps", "D2", "D2"]]}
    return d

CASES = [("ch4_N24_L8_e0.2_r0_q1_g1.json", 0.0, False), ("ch4_N24_L8_e0.2_r0.5_q1_g1.json", 0.5, False),
         ("ch4_N24_L8_e0.2_r0_q1_g1_1.json", 0.0, True)]

@pytest.mark.parametrize("fname,rho,two", CASES)
def test_ch4_matches_kb_spectral_q(fname, rho, two):
    path = os.path.join(SP, fname)
    if not os.path.exists(path):
        pytest.skip("reference not present")
    ref = json.load(open(path))
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch4_kyle_back.yaml")))
    d["params"]["rho"] = rho
    if two:
        d = two_trader(d)
    res = ns.solve(ns.Model.from_dict(d))
    print(res.summary())
    assert np.allclose(np.array(ref["lag"]), res.ages, atol=1e-12)
    nW = len(d["channels"])
    for j, tr in enumerate(ref["traders"]):
        c_ref = np.stack([np.array(tr["c"][k][0]) for k in range(nW)], axis=1)   # (N, channels)
        c = res.action_kernel(f"D{j+1}")
        print(f"trader {j+1}: max|diff| {np.abs(c - c_ref).max():.2e}  scale {np.abs(c_ref).max():.3f}")
        assert np.abs(c - c_ref).max() < 2e-4
    assert res.converged
