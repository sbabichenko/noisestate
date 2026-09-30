"""Exact shifts: the grid's shift-aware operators against brute-force quadrature on panels a lag is not aligned with,
and against the resampled operators on aligned panels (where resampling is exact); the stationary engine's exact-shift
path against the resampling path on aligned panels (both exact there)."""
import numpy as np
import pytest
from scipy.integrate import quad

import noisestate as ns
import noisestate.stationary as st
from noisestate.grid import AgeGrid
from noisestate import grid_cache
from helpers import example_dict

BP = [0, 0.5, 1.0, 1.5, 2.25, 3.4, 5.0, 6.0]         # unit panels, then a tail no lag of 0.5 is aligned with


def _setup():
    rng = np.random.default_rng(0)
    g = AgeGrid(BP, 7)
    return g, rng.standard_normal(g.N), rng.standard_normal(g.N)


def _read(g, v, cls):
    """The kernel of class cls = (s, lo, hi) with nodal values v, as a function."""
    s, lo, hi = g.shift_class(cls)
    return lambda a: float((g.interp(np.array([a - s])) @ v)[0]) if lo <= a <= hi else 0.0


def _integral(fun, lo, hi, cuts):
    cuts = sorted(c for c in cuts if lo < c < hi)
    return quad(fun, lo, hi, points=cuts or None, limit=500, epsabs=1e-13, epsrel=1e-13)[0]


def test_alignment():
    g, _, _ = _setup()
    assert not g.aligned(0.5) and not g.aligned(-0.5) and g.aligned(0.0)
    u = AgeGrid([0, 0.5, 1.0, 1.5, 2.0], 5)
    assert u.aligned(0.5) and u.aligned(-0.5) and u.aligned(1.0)
    assert u.compose(-0.5, 0.5) == (0.0, 0.0, 1.5)            # a lead of a lag: the window cuts it at L - lead
    assert g.shift_class(0.5) == (0.5, 0.5, 6.0)


@pytest.mark.parametrize("c1,c2", [(0.5, 0.0), (0.5, 0.5), (-0.5, 0.5), (1.0, -0.5), ((0.0, 0.0, 5.5), 0.5)])
def test_mass(c1, c2):
    g, x, y = _setup()
    fx, fy = _read(g, x, c1), _read(g, y, c2)
    s1, s2 = g.shift_class(c1)[0], g.shift_class(c2)[0]
    ref = _integral(lambda a: fx(a) * fy(a), 0, g.L, [b + s1 for b in BP] + [b + s2 for b in BP])
    assert abs(x @ g.mass_shifted(c1, c2) @ y - ref) < 1e-12


@pytest.mark.parametrize("s", [0.5, 1.0, -0.5])
def test_convolutions(s):
    g, x, y = _setup()
    fx, fy = _read(g, x, 0.0), _read(g, y, s)
    C = g.conv_ops_shifted(y[:, None], s)[0]                  # on x, for the kernel y of class s
    Cl = g.conv_ops_left_shifted(x[:, None], s)[0]            # on y, for the kernel x
    for k in (5, 17, 30, g.N - 1):
        a = g.nodes[k]
        ref = _integral(lambda b: fx(b) * fy(a - b), 0, a, BP + [a - s - b for b in BP])
        assert abs((C @ x)[k] - ref) < 1e-12 and abs((Cl @ y)[k] - ref) < 1e-12


@pytest.mark.parametrize("rho,cw,cz", [(0.3, 0.5, 0.0), (0.0, 0.0, 0.5), (0.2, -0.5, 1.0), (0.1, 0.5, (0.0, 0.0, 5.5))])
def test_correlations(rho, cw, cz):
    g, x, y = _setup()
    fx, fy = _read(g, x, cw), _read(g, y, cz)
    sw, sz = g.shift_class(cw)[0], g.shift_class(cz)[0]
    C = g.corr_ops_shifted(x[:, None], rho, cw, cz)[0]
    for k in (3, 20, 33):
        a = g.nodes[k]
        ref = _integral(lambda t: np.exp(-rho * t) * fx(t) * fy(a + t), 0, g.L - a, [b + sw for b in BP] + [b + sz - a for b in BP])
        assert abs((C @ y)[k] - ref) < 1e-10                       # quad's own accuracy on these pieces


def test_aligned_panels_agree_with_resampling():
    rng = np.random.default_rng(1)
    g = AgeGrid([0, .5, 1, 1.5, 2, 2.5, 3], 6)
    x, y = rng.standard_normal(g.N), rng.standard_normal(g.N)
    S, Sm = g.shift(0.5), g.shift(-0.5)
    c = g.compose(-0.5, 0.5)
    assert np.abs(g.sample_class(c) - Sm @ S).max() == 0.0
    assert np.abs(g.conv_ops_shifted(y[:, None], 0.5)[0] - g.conv_ops((S @ y)[:, None])[0]).max() < 1e-14
    assert np.abs(g.conv_ops_left_shifted(x[:, None], 0.5)[0] - g.conv_ops_left(x[:, None])[0] @ S).max() < 1e-14
    assert np.abs(g.corr_ops_shifted(x[:, None], 0.4, 0.5, -0.5)[0] - g.corr_ops((S @ x)[:, None], 0.4)[0] @ Sm).max() < 1e-14
    assert np.abs(g.corr_ops_shifted(x[:, None], 0.3, 0.5, c)[0] - g.corr_ops((S @ x)[:, None], 0.3)[0] @ Sm @ S).max() < 1e-14
    assert np.abs(g.mass_shifted(0.5, -0.5) - S.T @ g.mass_matrix @ Sm).max() < 1e-14
    assert np.abs(g.discounted_mass_shifted(0.3, c) - g.discounted_mass(0.3) @ Sm @ S).max() < 1e-14


def test_shifted_propagator():
    g = AgeGrid(BP, 14)
    A = np.array([[-0.8]]); s = 0.5
    u = np.where(g.nodes < 1.0, 1.0, np.exp(1.0 - g.nodes))             # kinked at 1: its shift breaks inside a panel
    a = np.maximum(g.nodes - s, 0.0)
    # x' = -0.8 x + u(a - s): int_0^{a-s} e^{-0.8 (a - s - r)} u(r) dr in closed form
    lo = np.minimum(a, 1.0)
    first = (1 - np.exp(-0.8 * lo)) / 0.8 * np.exp(-0.8 * (a - lo))
    hi = np.maximum(a - 1.0, 0.0)
    second = np.where(a > 1.0, (np.exp(-hi) - np.exp(-0.8 * hi)) / (-0.2), 0.0)
    exact = first + second
    assert np.abs(g.propagator_shifted(A, s) @ u - exact).max() < 1e-10


def _ch5_short(window=6.0, nodes=6, unit_range=6.0):
    d = example_dict("ch5_cycle_market")
    d["horizon"]["window"] = window
    d["numerics"] = {"nodes": nodes, "unit": 0.5, "unit_range": unit_range}
    return ns.Model.from_dict(d)


def _solve(model, exact, **kw):
    old = st.Compiled.FORCE_EXACT
    try:
        st.Compiled.FORCE_EXACT = exact; grid_cache.clear()
        return ns.solve(model, **kw)
    finally:
        st.Compiled.FORCE_EXACT = old; grid_cache.clear()


def test_exact_path_equals_resampling_on_aligned_panels():
    m = _ch5_short(window=4.0, nodes=5, unit_range=4.0)
    r0, r1 = _solve(m, False, diagnostics=False), _solve(m, True, diagnostics=False)
    assert not r0.compiled.exact and r1.compiled.exact
    assert np.abs(r1.world - r0.world).max() < 1e-12 * np.abs(r0.world).max()
    assert abs(r1.costs["firm0"] - r0.costs["firm0"]) < 1e-12 * abs(r0.costs["firm0"])
    assert r1.evaluations == r0.evaluations


def test_automatic_age_panels_on_a_lagged_model():
    """No grid in the numerics: the panels are chosen (unit panels through the lags, then growing on the unit lattice, cut
    at L - tau and L - 2 tau), the result says so and carries the grid; any grid field given is the expert's, unchanged."""
    from make_ch5_cycle_market import build
    d = build(N=2, L=6.0).to_dict()
    assert d["numerics"] == {"unit": 0.5}
    res = ns.solve(ns.Model.from_dict(d), diagnostics=False)
    p = res.panels
    assert p is not None and p["resolved"] and p["route"] in ("a priori", "refined") and res.compiled.exact
    bp = p["breakpoints"]
    assert res.numerics.breakpoints == bp and res.numerics.nodes == p["nodes"]
    assert all(abs(b / 0.5 - round(b / 0.5)) < 1e-12 for b in bp)            # on the unit lattice
    assert {5.5, 5.0} <= set(bp) and bp[:3] == [0.0, 0.5, 1.0]
    assert max(p["tails"]) <= res.settings.auto_grid_tol
    given = ns.solve(ns.Model.from_dict({**d, "numerics": {"unit": 0.5, "nodes": 6, "unit_range": 3.0}}), diagnostics=False)
    assert given.panels is None and given.compiled.N == 6 * len(given.compiled.grid.breakpoints[:-1])
