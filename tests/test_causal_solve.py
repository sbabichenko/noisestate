import numpy as np, yaml, os
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__))

def test_block_forward_substitution_matches_dense_solve():
    d = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "ch1_delayed_finite.yaml"))); d["horizon"]["nodes"] = 5
    S = SpectralFiniteSolver(ns.Model.from_dict(d)); c = S.c
    n = len(c.prim) * c.N
    rng = np.random.default_rng(3)
    # a random operator that respects causality: panel p only reads panels <= p
    panel = np.concatenate([np.full(pc.n, pc.p) for pc in c.g.pieces]); panel = np.tile(panel, len(c.prim))
    M = rng.standard_normal((n, n)) * 0.02
    M[panel[:, None] < panel[None, :]] = 0.0
    B = rng.standard_normal((n, 4))
    Z = c._solve_causal(M, B)
    assert np.allclose(Z, np.linalg.solve(np.eye(n) - M, B), atol=1e-10)
    # and the real closed loop agrees with a dense solve
    acts = {a.name: rng.standard_normal((len(a.controls), c.N, c.nW)) * 0.1 for a in S.model.agents}
    maps = S.maps_from_actions(acts)
    Zb = c.closed_loop(maps)
    c._solve_causal = lambda M_, B_: np.linalg.solve(np.eye(n) - M_, B_)
    Zd = c.closed_loop(maps)
    assert np.abs(Zb - Zd).max() < 1e-9 * max(1.0, np.abs(Zd).max())
