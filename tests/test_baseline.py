"""The current package against the record tests/refs/baseline_0.4.json (+ .npz, the cases' Z), written by
extras/compare_baseline.py: the five shipped examples, the Chapter 1 sweep point p = 10, Chapter 3 as its own
transition at 6 nodes and the Kyle-Back prior.  Costs to 1e-11 relative, the evaluation counts equal, Z within 1e-11 in
max |dZ| / max |Z| (every case's distance is reported as a warning: a last-bit change is allowed, a change of
the formulation is not).  Re-baseline with `python extras/compare_baseline.py write tests/refs/baseline_0.4.json`
when a step is meant to move the record."""
import os
import warnings

from compare_baseline import run, differences, load
from helpers import REFS, slow

REF = os.path.join(REFS, "baseline_0.4.json")


@slow("slow (12 s: every shipped case re-solved); set NOISESTATE_SLOW=1")
def test_current_package_matches_the_baseline_record():
    ref, Zref = load(REF)
    assert Zref is not None, "tests/refs/baseline_0.4.npz missing: write the record"
    fails, notes = differences(ref["cases"], run(), Zref)
    for line in notes:
        warnings.warn(f"baseline {ref.get('commit')}: {line}")
    assert not fails, "\n".join(fails)
