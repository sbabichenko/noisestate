"""Pins the defect of the dissertation's C++ two-trader solver (kb_spectral_q): residual-flow policy
rows used inside a total-flow feedback cascade.  Not part of the package's test suite; run with
    .venv/bin/python -m pytest -q extras/test_cpp_cascade.py
"""
import os, sys, numpy as np
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from test_ch4 import two_trader                                   # noqa: E402


def test_cpp_cascade_replica_reproduces_published_two_trader_solution():
    """Pins the kb_spectral_q defect: residual-flow policy rows inside a total-flow feedback."""
    from cpp_cascade_replica import CppCascadeReplica
    ref = ns.read_json(os.path.join(HERE, "refs", "ch4_N24_L8_e0.2_r0_q1_g1_1.json"))
    d = two_trader(ns.read_yaml(os.path.join(HERE, "..", "examples", "ch4_kyle_back.yaml")))
    d["params"]["rho"] = 0.0                                          # the reference is the undiscounted case (r0); the example ships with rho 0.5
    res = CppCascadeReplica(ns.Model.from_dict(d)).solve()
    c_ref = np.stack([np.array(ref["traders"][0]["c"][k][0]) for k in range(4)], axis=1)
    assert res.converged
    assert np.abs(res.strategy_kernel("D1") - c_ref).max() < 1e-3
    assert abs(-res.costs["trader1"] - ref["traders"][0]["flow"]) < 2e-6
