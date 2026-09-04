import os, numpy as np, yaml
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__))

def load(nodes):
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch1_two_player_finite.yaml")))
    d["horizon"]["nodes"] = nodes
    return ns.Model.from_dict(d)

def test_ch1_spectral_converges_exponentially():
    r12 = SpectralFiniteSolver(load(12)).solve(); r16 = SpectralFiniteSolver(load(16)).solve()
    assert r12.converged and r12.residual < 1e-7
    assert abs(r12.costs["player1"] - r16.costs["player1"]) < 1e-7        # already converged at 12 nodes per side
    assert abs(r12.costs["player1"] - 0.39690577) < 1e-6                   # value on record (spec_ch1 with Tikhonov: 0.39665)
    assert abs(r12.costs["player1"] - r12.costs["player2"]) < 1e-10

def test_ch1_spectral_diagonal_matches_cell_richardson():
    """The control's instantaneous response to its own signal noise, calD1(t, s=t) on w1:
    Richardson limit of the cell engine (N=60,120) computed 2026-09-03."""
    res = SpectralFiniteSolver(load(12)).solve()
    ts = np.array([0.02, 0.16, 0.30, 0.44, 0.58, 0.72, 0.86])
    rich = np.array([-0.1027, -0.7718, -1.2895, -1.6214, -1.7490, -1.6003, -1.0278])
    mine = res.evaluate("D1", "w1", ts, ts)
    assert np.abs(mine - rich).max() < 2e-3
