import json, os, numpy as np, pytest, yaml
import noisestate as ns

REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
SP = REFS
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
    assert os.path.exists(path), path
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
        if not two:
            assert np.abs(c - c_ref).max() < 5e-4      # window-edge discretization; 1e-5 at 48 nodes
    assert res.converged
    if two:
        # The C++ port kb_spectral_q disagrees here (2-6%) and its profile is not a best response.
        # The reference grid solver kb_multi.py, Richardson-extrapolated over n = 60..400, gives the
        # following values of trader 1's kernel at lags 0.5, 1, 2 on channels [wV, wZ, w1, w2].
        rich = np.array([[0.2628, -0.3526, 0.3296, -0.2014],
                         [0.2476, -0.1635, 0.0647, -0.1880],
                         [0.1122, -0.0366, -0.0254, -0.0839]])
        I = res.compiled.grid.interp([0.5, 1.0, 2.0])
        mine = I @ res.action_kernel("D1")
        print("vs Richardson limit of kb_multi:", np.abs(mine - rich).max())
        assert np.abs(mine - rich).max() < 2e-3


def test_cpp_cascade_replica_reproduces_published_two_trader_solution():
    """Pins the kb_spectral_q defect: residual-flow policy rows inside a total-flow feedback."""
    from noisestate.diagnostics import CppCascadeReplica
    ref = json.load(open(os.path.join(SP, "ch4_N24_L8_e0.2_r0_q1_g1_1.json")))
    d = two_trader(yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch4_kyle_back.yaml"))))
    res = CppCascadeReplica(ns.Model.from_dict(d)).solve()
    c_ref = np.stack([np.array(ref["traders"][0]["c"][k][0]) for k in range(4)], axis=1)
    assert res.converged
    assert np.abs(res.action_kernel("D1") - c_ref).max() < 1e-3
    assert abs(-res.costs["trader1"] - ref["traders"][0]["flow"]) < 2e-6
