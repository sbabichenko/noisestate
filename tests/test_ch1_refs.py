"""The Chapter 1 finite-horizon game against the two reference solutions in tests/refs, frozen from the dissertation's
own solvers, away from the diagonal.  Both references carry their own discretisation error, so the tolerances here
are theirs, not the package's: between 12 and 20 nodes per side the package's kernels move by at most 1.2e-5 and its
cost by 1e-11, and the cell engine's Richardson pairs (80, 160) and (160, 320) give 0.396956 and 0.396918 against the
spectral 0.39690577, closing as h^2 (measured 2026-09-05).  The tolerances are 1.5x the differences measured that day,
so a regression larger than a reference's own error fails here and nothing finer does; the finer checks are the
package's own convergence (tests/test_ch1_spectral.py, with the diagonal against the cell engine's Richardson limit)."""
import os, json, numpy as np, pytest
import noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__)); REFS = os.path.join(HERE, "refs")
CHANNELS = ["w0", "w1", "w2"]


@pytest.fixture(scope="module")
def res():
    d = ns.read_yaml(os.path.join(HERE, "..", "examples", "ch1_two_player_finite.yaml")); d["horizon"]["nodes"] = 12
    return SpectralFiniteSolver(ns.Model.from_dict(d)).solve().check()


def compare(res, name, ch, t, s, ref, tol):
    diff = res.evaluate(name, ch, t, s) - ref
    mx, rms = float(np.abs(diff).max()), float(np.sqrt(np.mean(diff ** 2)))
    assert mx < tol[0] and rms < tol[1], (name, ch, mx, rms)


def test_kernels_match_spec_ch1_off_the_diagonal(res):
    """spec_ch1 at 16 x 16 Chebyshev-Lobatto nodes (Duffy coordinates, Tikhonov penalty 1e-7): columns t, s, then
    X, calD1, calD2 on the three channels.  Its README reports weakly determined modes near the diagonal (the
    own-noise loading calD1/w1 is 0.35 from the package at lags below 0.1), so the comparison is at lags >= 0.1
    (132 of the 256 nodes).  Measured max / rms: X/w0 3.1e-3 / 1.3e-3, X/w1 and w2 1.0e-2 / 2.8e-3, calD1/w0
    6.5e-3 / 2.1e-3, calD1/w1 4.9e-2 / 1.2e-2, calD1/w2 1.9e-3 / 6.5e-4; calD2 mirrors calD1 with w1 and w2 swapped."""
    ref = np.loadtxt(os.path.join(REFS, "ch1_spec_p3_p3.txt")); t, s = ref[:, 2], ref[:, 3]
    sel = t - s >= 0.1 - 1e-9; assert sel.sum() == 132
    own, other, state = (7.5e-2, 1.8e-2), (3e-3, 1e-3), (1e-2, 3.5e-3)
    tol = {"X": [(5e-3, 2e-3), (1.5e-2, 4.5e-3), (1.5e-2, 4.5e-3)], "D1": [state, own, other], "D2": [state, other, own]}
    for name, col in (("X", 6), ("D1", 9), ("D2", 12)):
        for j, ch in enumerate(CHANNELS):
            compare(res, name, ch, t[sel], s[sel], ref[sel, col + j], tol[name][j])


def test_kernels_match_the_grid_solver_off_the_diagonal(res):
    """solve_interactive, the dissertation's first-order grid solver, on 160 nodes t_i = i / 159: X, calD1 and calD2
    per channel as (160, 160) arrays indexed [i, j] = (t_i, s_j), the state kernel carrying the unit impulse on its
    diagonal.  Compared at i - j >= 16 (lags >= 0.1, 10440 nodes); its error is first order in h = 1/159.  Measured
    max / rms: X/w0 4.2e-3 / 1.3e-3, X/w1 and w2 6.3e-3 / 3.4e-3, calD1/w0 5.2e-3 / 2.5e-3, calD1/w1 3.0e-2 / 9.4e-3,
    calD1/w2 3.2e-3 / 1.5e-3.  spec_ch1 is further from this file on every kernel but X/w0 (calD1/w1 6.8e-2 / 1.5e-2,
    X/w1 1.5e-2 / 5.0e-3).  Its J1 (4.1332) includes the mean part in another normalisation and is not used."""
    with open(os.path.join(REFS, "ch1_grid_N160.json")) as f:
        g = json.load(f)
    tg = np.asarray(g["t"]); N = len(tg); assert N == 160 and abs(tg[-1] - 1.0) < 1e-9
    I, J = np.tril_indices(N, -16); assert len(I) == 10440
    own, other, state = (4.5e-2, 1.4e-2), (5e-3, 2.5e-3), (8e-3, 4e-3)
    tol = {"X": [(6e-3, 2e-3), (1e-2, 5e-3), (1e-2, 5e-3)], "calD1": [state, own, other], "calD2": [state, other, own]}
    for key, name in (("X", "X"), ("calD1", "D1"), ("calD2", "D2")):
        for j, ch in enumerate(CHANNELS):
            G = np.asarray(g[key][f"ch{j}"], dtype=float)
            compare(res, name, ch, tg[I], tg[J], G[I, J], tol[key][j])
