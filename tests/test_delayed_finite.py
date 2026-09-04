import os, numpy as np, yaml
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__))

def test_delayed_finite_spectral_converges_and_matches_cells():
    """Control lag 0.25 in the state and a delayed observation for player 2 (Chapter 2 style).
    Reference: the first-order cell scheme, Richardson-extrapolated from 60 and 120 cells
    (cost 0.4839003 / 0.4767628, computed 2026-09-03)."""
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch1_delayed_finite.yaml")))
    d["horizon"]["nodes"] = 6
    S = SpectralFiniteSolver(ns.Model.from_dict(d)); res = S.solve()
    assert res.converged and res.residual < 1e-7
    assert len(S.c.g.pieces) == 10                         # 4 time panels of width 0.25
    assert abs(res.costs["player1"] - 0.4839003) < 2e-4 and abs(res.costs["player2"] - 0.4767628) < 2e-4
    # player 2 cannot respond to shocks younger than its observation delay
    t = np.array([0.6, 0.8]); assert np.abs(res.evaluate("D2", "w2", t, t - 0.1)).max() < 1e-10
    assert np.abs(res.evaluate("D2", "w2", t, t - 0.4)).max() > 0.05
