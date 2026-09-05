"""Chapter 5 cycle market vs the recorded sweep point (slow: ~6 s at 4 threads, longer on a loaded machine). Run with NOISESTATE_SLOW=1."""
import os, numpy as np, pytest
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
REF = os.path.join(REFS, "ch5_s1_2.5_u16_endo.txt")

@pytest.mark.skipif(not os.environ.get("NOISESTATE_SLOW"), reason="slow; set NOISESTATE_SLOW=1")
def test_ch5_cycle_market_matches_recorded_sweep():
    res = ns.solve(os.path.join(HERE, "..", "examples", "ch5_cycle_market.yaml"), tol=1e-8)
    assert res.converged
    ref = np.loadtxt(REF); I = res.compiled.grid.interp(ref[:, 0]); g = res.maps["firm0"]
    for ci in range(2):
        for r in range(5):
            mine, theirs = I @ g[ci, r], ref[:, 1 + ci * 5 + r]
            assert np.abs(mine - theirs).max() / np.abs(theirs).max() < 0.06
