"""The spectral finite engine's closed loop (noisestate.closed_loop.ClosedLoopRows): the one assembly, its rows
built one time panel at a time and solved by block forward substitution, against the dense system those rows stack
into."""
import numpy as np
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver, ClosedLoopRows
from helpers import example, example_dict, stationary


def stacked_solve(c, **kw):
    """The rows of every panel stacked into the dense (I - M) Z = B and solved at once; the block layout is
    checked on the way: a panel's rows read the nodes before the panel's end only."""
    L = ClosedLoopRows(c, **kw); nP = len(c.prim); N = c.N; n = nP * N
    M = np.zeros((nP, N, nP, N)); B = L.B.copy()
    for p, (lo, hi) in enumerate(c._panel_ranges):
        blk, forcing = L.panel(p)
        for (i, j), X in blk.items():
            assert X.shape == (hi - lo, hi)
            M[i, lo:hi, j, :hi] += X
        for bi, f in forcing.items():
            B[bi, lo:hi, :c.nW] += f
    return np.linalg.solve(np.eye(n) - M.reshape(n, n), B.reshape(n, -1))


def test_block_forward_substitution_matches_the_stacked_dense_solve():
    """ch1_delayed at 5 nodes (player2's row delayed): the closed loop under random maps, plain and with an agent
    excluded and its impulse column, is the dense solve of the stacked rows to 1e-10."""
    d = example_dict("ch1_delayed_finite"); d.setdefault("numerics", {})["nodes"] = 5
    S = SpectralFiniteSolver(ns.Model.from_dict(d)); c = S.c
    rng = np.random.default_rng(3)
    acts = {a.name: rng.standard_normal((len(a.controls), c.N, c.nW)) * 0.1 for a in S.model.agents}
    maps = S.maps_from_actions(acts)
    for excl, imp in ((None, ()), ("player2", ("D2",))):
        Zb = c.closed_loop(maps, excluded=excl, impulse_controls=imp)
        Zd = stacked_solve(c, maps=maps, excluded=excl, impulse_controls=imp)
        assert Zb.shape == Zd.shape and np.abs(Zb - Zd).max() < 1e-10 * max(1.0, np.abs(Zd).max())
    Zw = S.world_from_actions(acts)
    assert np.abs(Zw - stacked_solve(c, maps=None, actions=acts)).max() < 1e-10 * np.abs(Zw).max()


def test_closed_loop_with_a_past_and_a_continuation_matches_the_stacked_dense_solve():
    """Chapter 3 as its own past and continuation (T = 6, L = 3, 6 nodes): the band's forcing (the old shocks),
    the buffer's frozen rows, the excluded agent frozen or off on the buffer, the plain closed loop, each the
    dense solve of the stacked rows to 1e-12 of the world's peak."""
    m3 = example("ch3_two_player"); stat = stationary(m3, 6)
    S = SpectralFiniteSolver(m3.with_horizon(kind="finite", window=6.0, nodes=6), past=stat, continuation=stat); c = S.c
    assert c.buffer.any() and c.g.upper.any()
    cases = [dict(maps=c.frozen)]
    for a in m3.agents:
        for own_frozen in (True, False):
            cases.append(dict(maps=c.frozen, excluded=a.name, impulse_controls=a.controls, own_frozen=own_frozen))
    for kw in cases:
        Zb = c.closed_loop(**kw); Zd = stacked_solve(c, **kw)
        assert np.abs(Zb - Zd).max() < 1e-12 * np.abs(Zd).max(), kw
