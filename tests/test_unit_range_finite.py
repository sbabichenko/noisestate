"""horizon.unit_range on the triangle grid: within it the delay cuts stay at every multiple of the unit, beyond it
the panels grow geometrically (TriangleGrid.fill_geometric), in time and in age, since the kink at the k-th delay
line weakens with k; the required cuts are kept (T; T - k unit when the game ends at T, where a control is idle
within the last lag; the strip's L and the past's cuts within unit_range; the buffer's panels).  A lagged read
beyond unit_range is interpolated (map_shift), the panels are not closed under the lags.  Default: the window,
today's grid bit for bit.  Measured (examples/ch1_delayed_finite.yaml with player2's row undelayed, window 3,
6 nodes): unit_range 1.0 keeps the costs to 1.2e-8 and the kernels to 6.9e-6 with N 2808 -> 1980 (104 s -> 44 s),
0.5 to 3.2e-6 and 1.8e-4 with N 1008 (8.9 s); the transition of ch1_delayed as its own past (window 3, T = 6,
4 nodes) at unit_range 1.5 returns the stationary maps to 2.5e-3 / 6.0e-3 (the full grid 5.8e-4 / 4.0e-4) with
N 9408 -> 3168 (588 -> 198 pieces).  A delayed row without a past keeps its map on the action grid shifted by
the delay, read node to node on every piece, so unit_range below the window is refused there."""
import numpy as np, pytest
import noisestate as ns
from helpers import example, delayed_stationary, same_model_solver, one_shot_deviation, slow


def undelayed(**hz):
    """The delayed example without the delay; the grid keys go to with_numerics, the rest to with_horizon."""
    d = example("ch1_delayed_finite").to_dict()
    d["agents"]["player2"]["signals"]["y2"].pop("delay")
    num = {k: hz.pop(k) for k in list(hz) if k in ("nodes", "unit", "unit_range", "breakpoints")}
    return ns.Model.from_dict(d)._patch_horizon(**hz).with_numerics(**num)


def test_unit_range_at_the_window_is_the_grid_bit_for_bit():
    m = example("ch1_delayed_finite").with_numerics(nodes=5)
    a = ns.solve(m); b = ns.solve(m.with_numerics(unit_range=1.0))
    assert not b.compiled.coarse and [float(x) for x in b.grid.bp] == [float(x) for x in a.grid.bp] and b.compiled.N == a.compiled.N
    assert all(np.array_equal(a.maps[k], b.maps[k]) for k in a.maps) and a.costs == b.costs


@slow("slow (8 s; the bit-for-bit default and the guards stay fast); set NOISESTATE_SLOW=1")
def test_unit_range_below_the_window_coarsens_the_grid_within_the_measured_cost():
    """Window 2, 5 nodes, the controls lagged by 0.25 and both rows undelayed: unit_range 0.5 keeps the cuts at
    0.25 and 0.5, at T - 0.25 and T - 0.5, and one panel between: 21 pieces for 36, N 525 for 900; the costs
    move by 1.3e-5 relative and the state kernel by 6.4e-4 of its peak on a 41 x 41 lattice."""
    full = ns.solve(undelayed(T=2.0, nodes=5)); coarse = ns.solve(undelayed(T=2.0, nodes=5, unit_range=0.5))
    assert coarse.compiled.coarse and [float(b) for b in coarse.grid.bp] == [0.0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0]
    assert len(coarse.grid.pieces) == 21 and len(full.grid.pieces) == 36 and coarse.compiled.N == 525 and full.compiled.N == 900
    for k in full.costs:
        assert abs(coarse.costs[k] - full.costs[k]) < 5e-5 * abs(full.costs[k]), (k, coarse.costs[k], full.costs[k])
    tt, aa = np.meshgrid(np.linspace(0, 2, 41), np.linspace(0, 2, 41), indexing="ij"); keep = aa <= tt
    for nm in ("X", "D1"):
        kf = full.grid.interp(tt[keep], aa[keep]) @ full.kernel(nm); kc = coarse.grid.interp(tt[keep], aa[keep]) @ coarse.kernel(nm)
        assert np.abs(kc - kf).max() < 3e-3 * np.abs(kf).max(), (nm, np.abs(kc - kf).max())


def test_unit_range_on_a_transition_keeps_the_identity_within_the_measured_floor():
    """ch1_delayed as its own past and continuation (window 3, T = 6, 3 nodes) at unit_range 1.5: the strip's cuts
    are the unit ones to 1.5, then 2, 3 (L), 3.5, 4.5, 6 (T) and the buffer's; 198 pieces for 588; the cost of
    the stationary maps on the strip is T times the stationary flow to 2.2e-5 (the full grid's floor at 3 nodes
    2.0e-5; at 4 nodes 1.0e-6 against 1e-7, its Gram matrix too slow for the suite), and one best response
    returns the stationary maps to 9.2e-3 / 1.6e-2 (the full grid's floor at 3 nodes 7.6e-3 / 6.7e-3)."""
    stat = delayed_stationary(3, window=3.0); m = stat.model
    full = same_model_solver(m, stat, 6.0, 3)
    coarse = same_model_solver(m, stat, 6.0, 3, unit_range=1.5)
    c = coarse.c
    assert [float(b) for b in c.g.bp] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 3.5, 4.5, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 8.0, 9.0]
    assert len(c.g.pieces) == 198 and len(full.c.g.pieces) == 588 and c.N == 1782
    Zc = c.closed_loop(c.frozen); dev = one_shot_deviation(coarse)
    for a in m.agents:
        cf, cc = 6.0 * stat.costs[a.name], coarse.expected_cost(a, Zc)
        assert abs(cc - cf) < 5e-5 * abs(cf), (a.name, cc, cf)
        assert dev[a.name].max() < 3e-2, (a.name, dev[a.name].max())


def test_unit_range_guards_and_kind_change():
    m = example("ch1_delayed_finite")
    with pytest.raises(ValueError, match="shifted by the delay"):
        ns.SpectralFiniteSolver(m.with_finite(2.0).with_numerics(nodes=4, unit_range=0.5))
    with pytest.raises(ValueError, match="below the largest lag"):
        ns.SpectralFiniteSolver(undelayed(T=2.0, nodes=4, unit=0.125, unit_range=0.125))
    s = m.with_stationary(3.0).with_numerics(nodes=4, unit_range=2.0)
    assert s.with_finite(2.0).horizon.unit_range is None            # a stationary sizing is not a finite one
    assert s.with_finite(2.0).with_numerics(unit_range=1.0).horizon.unit_range == 1.0
    assert s.with_numerics(nodes=6).horizon.unit_range == 2.0
