import os
from noisestate.sweep import sweep
HERE = os.path.dirname(os.path.abspath(__file__))

def test_stationary_sweep_warm_starts_and_serialises():
    rows = sweep(os.path.join(HERE, "..", "examples", "ch4_kyle_back.yaml"), "eps", [0.2, 0.15, 0.1])
    assert all(r["converged"] for r in rows)
    assert rows[1]["evaluations"] < rows[0]["evaluations"]          # warm start pays
    d = rows[-1]["result"].to_dict()
    assert d["grid"]["kind"] == "stationary" and "D1" in d["kernels"] and "trader1" in d["foc"]
    import json; json.dumps(d)                                       # JSON-ready

def test_finite_sweep_warm_starts():
    rows = sweep(os.path.join(HERE, "..", "examples", "ch1_two_player_finite.yaml"), "p1", [3.0, 4.0, 5.0])
    assert all(r["converged"] for r in rows)
    assert rows[2]["evaluations"] <= rows[0]["evaluations"]
    d = rows[-1]["result"].to_dict(); assert d["grid"]["kind"] == "finite"
