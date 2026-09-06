"""The current package against the record of master 222eb96 (0.4.0 + the size stage + the matrix-free best
response), tests/refs/baseline_0.4.json, written by extras/compare_baseline.py: the five shipped examples,
the Chapter 1 sweep point p = 10, Chapter 3 as its own transition at 6 nodes and the Kyle-Back prior.  Costs
to 1e-12, the evaluation counts equal, Z equal at 12 significant digits; a change of Z's raw bytes with the
12-digit SHA intact is reported as a warning, not asserted (BLAS rounding).  Re-baseline with
`python extras/compare_baseline.py write tests/refs/baseline_0.4.json` when a step is meant to move bits."""
import json
import os
import warnings

from compare_baseline import run, differences
from helpers import REFS, slow

REF = os.path.join(REFS, "baseline_0.4.json")


@slow("slow (12 s: every shipped case re-solved); set NOISESTATE_SLOW=1")
def test_current_package_matches_the_baseline_record():
    with open(REF) as fh:
        ref = json.load(fh)
    fails, notes = differences(ref["cases"], run())
    for line in notes:
        warnings.warn(f"baseline {ref.get('commit')}: {line}")
    assert not fails, "\n".join(fails)
