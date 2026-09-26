"""The cell engine: its kernel stack over the shocks and its mean solve's rcond guard (moved from
test_deprecations.py; they go to extras/ with the engine)."""
import numpy as np
import pytest
import noisestate as ns
from helpers import example


@pytest.fixture(scope="module")


def test_cell_engine_kernel_without_a_channel_is_the_stack_over_channels(cells):
    K = cells.kernel("X"); N = cells.compiled.N
    assert K.shape == (N, N, len(cells.channels))
    for k, ch in enumerate(cells.channels):
        assert np.array_equal(K[..., k], cells.kernel("X", ch))
    assert cells.kernel("D1").shape == (N, N, len(cells.channels))


def test_cell_engine_mean_solve_refuses_a_singular_mean_system():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from ch1_mean_sweep import model as ch1_targets
    d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}
    d["numerics"] = {"engine": "cells", "nodes": 8, "settings": {"mean_rcond": 1.0}}      # every system fails a threshold of 1
    with pytest.raises(ValueError, match="the mean system is singular"):
        ns.solve(d)
