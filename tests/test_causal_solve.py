import numpy as np, os
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__))

def test_block_forward_substitution_matches_dense_solve():
    d = ns.read_yaml(os.path.join(HERE, "..", "examples", "ch1_delayed_finite.yaml")); d["horizon"]["nodes"] = 5
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


def test_per_panel_closed_loop_matches_the_dense_one():
    """The closed loop assembled one time panel at a time (settings.closed_loop_dense_max below n_prim N: the
    rows of each panel from the line paths and the sparse reads, the (n_prim N)^2 system never built) is the
    dense one to BLAS rounding: without a past (ch1_delayed, 5 nodes; an agent excluded with its impulse
    column) and with a past and a continuation (Chapter 3 as its own, T = 6, L = 3, 6 nodes: the band, the
    buffer, the excluded agent frozen or off on the buffer) the worlds agree to 1e-13 of their peak, while
    the dense path within the limit is the default (measured 2e-16 to 3e-13 on the shipped examples; the
    two are not bit-identical since a BLAS product of a row block rounds differently from the full one)."""
    d = ns.read_yaml(os.path.join(HERE, "..", "examples", "ch1_delayed_finite.yaml")); d["horizon"]["nodes"] = 5
    m = ns.Model.from_dict(d)
    dense = SpectralFiniteSolver(m); panel = SpectralFiniteSolver(m, settings={"closed_loop_dense_max": 0})
    assert not dense.c.per_panel and panel.c.per_panel and panel.c.N == dense.c.N
    c = dense.c; rng = np.random.default_rng(5)
    acts = {a.name: rng.standard_normal((len(a.controls), c.N, c.nW)) * 0.1 for a in m.agents}
    maps = dense.maps_from_actions(acts)
    for excl, imp in ((None, ()), ("player2", ("D2",))):
        Zd = dense.c.closed_loop(maps, excluded=excl, impulse_controls=imp)
        Zp = panel.c.closed_loop(maps, excluded=excl, impulse_controls=imp)
        assert Zd.shape == Zp.shape and np.abs(Zp - Zd).max() < 1e-13 * np.abs(Zd).max()
    m3 = ns.load(os.path.join(HERE, "..", "examples", "ch3_two_player.yaml"))
    stat = ns.solve(m3.with_horizon(nodes=6)).check()
    hz = m3.with_horizon(kind="finite", window=6.0, nodes=6)
    dense = SpectralFiniteSolver(hz, past=stat, continuation=stat)
    panel = SpectralFiniteSolver(hz, past=stat, continuation=stat, settings={"closed_loop_dense_max": 0})
    assert panel.c.per_panel and panel.c.buffer.any() and panel.c.g.upper.any()
    for a in m3.agents:
        for own_frozen in (True, False):
            Zd = dense.c.closed_loop(dense.c.frozen, excluded=a.name, impulse_controls=a.controls, own_frozen=own_frozen)
            Zp = panel.c.closed_loop(panel.c.frozen, excluded=a.name, impulse_controls=a.controls, own_frozen=own_frozen)
            assert np.abs(Zp - Zd).max() < 1e-13 * np.abs(Zd).max()
    Zd = dense.c.closed_loop(dense.c.frozen); Zp = panel.c.closed_loop(panel.c.frozen)
    assert np.abs(Zp - Zd).max() < 1e-13 * np.abs(Zd).max()
