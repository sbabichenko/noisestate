import numpy as np
import noisestate as ns
from noisestate.triangle import TriangleGrid

def kernel(t, a):      # smooth in t, kinked at age 0.5 (piecewise-analytic)
    return np.exp(-0.7 * a) * (1 + 0.3 * np.sin(2 * t)) + np.where(a > 0.5, (a - 0.5) ** 2, 0.0)

def test_interp_and_quadrature_with_delay_panels():
    g = TriangleGrid(TriangleGrid.breakpoints(2.0, [0.5]), nt=14, na=14)
    f = kernel(g.t, g.a)
    rng = np.random.default_rng(1)
    tt = rng.uniform(0, 2, 300); aa = rng.uniform(0, 1, 300) * tt
    assert np.abs(g.evaluate(f, tt, aa) - kernel(tt, aa)).max() < 1e-9
    # row integral at t = 1.3 and 2D mass against numerical reference
    from scipy.integrate import quad, dblquad
    t = 1.3
    ref = quad(lambda a: kernel(t, a), 0, 0.5)[0] + quad(lambda a: kernel(t, a), 0.5, t)[0]
    assert abs(g.row_weights(t) @ f - ref) < 1e-10
    ref2 = dblquad(lambda a, t: kernel(t, a), 0, 2, 0, lambda t: t, epsabs=1e-11)[0]
    assert abs(g.mass @ f - ref2) < 1e-8

def test_line_operators():
    g = TriangleGrid(TriangleGrid.breakpoints(2.0, [0.5]), nt=14, na=14)
    f = kernel(g.t, g.a)
    from scipy.integrate import quad
    lam = 0.8
    # (a) Volterra in time at fixed s: F(t, s) = int_s^t e^{-lam (t - r)} f(r, r - s) dr
    V = g.line_op(g.t, g.a, r_lo=g.s, r_hi=g.t,
                  point_fn=lambda k, r: (r, r - g.s[k]), weight_fn=lambda k, r: np.exp(-lam * (g.t[k] - r)))
    k = np.argmin(np.abs(g.t - 1.7) + np.abs(g.a - 0.9))
    t, s = g.t[k], g.s[k]
    ref = sum(quad(lambda r: np.exp(-lam * (t - r)) * kernel(r, r - s), lo, hi)[0]
              for lo, hi in ((s, s + 0.5), (s + 0.5, t)))
    assert abs((V @ f)[k] - ref) < 1e-10
    # (b) convolution over u at fixed (t, s): C(t, s) = int_s^t gam(t, t - u) y(u, u - s) du, y known
    y = lambda t_, a_: np.cos(t_) * np.exp(-a_)
    C = g.line_op(g.t, g.a, r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r),
                  weight_fn=lambda k, r: y(r, r - g.s[k]))
    ref = sum(quad(lambda u: kernel(t, t - u) * y(u, u - s), lo, hi)[0] for lo, hi in ((s, t - 0.5), (t - 0.5, t)))
    assert abs((C @ f)[k] - ref) < 1e-10
    # (c) continuation over tau at fixed (t, s): int_t^T e^{-rho (tau - t)} R(tau, tau - t) f(tau, tau - s) dtau
    rho = 0.3; R = lambda tau, a_: np.exp(-0.5 * a_) * (1 + 0.1 * tau)
    K = g.line_op(g.t, g.a, r_lo=g.t, r_hi=np.full(g.N, g.T), point_fn=lambda k, r: (r, r - g.s[k]),
                  weight_fn=lambda k, r: np.exp(-rho * (r - g.t[k])) * R(r, r - g.t[k]))
    cuts = sorted({t, g.T, s + 0.5, s + 1.0, s + 1.5, t + 0.5} & set(np.round(np.linspace(0, 10, 1), 13)) | {t, g.T} | {c for c in (s + 0.5, s + 1.0, s + 1.5, t + 0.5) if t < c < g.T})
    ref = sum(quad(lambda tau: np.exp(-rho * (tau - t)) * R(tau, tau - t) * kernel(tau, tau - s), lo, hi)[0] for lo, hi in zip(cuts[:-1], cuts[1:]))
    assert abs((K @ f)[k] - ref) < 1e-9
    # (d) projection over s at fixed (t, u): int_0^u phi(t, s) y(u, u - s) ds, output indexed by (t, u)
    P = g.line_op(g.t, g.a, r_lo=np.zeros(g.N), r_hi=g.s, point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r),
                  weight_fn=lambda k, r: y(g.s[k], g.s[k] - r))
    u = s
    cuts = sorted({0.0, u} | {c for c in (t - 0.5, t - 1.0, t - 1.5, u - 0.5, u - 1.0) if 0 < c < u})
    ref = sum(quad(lambda s_: kernel(t, t - s_) * y(u, u - s_), lo, hi)[0] for lo, hi in zip(cuts[:-1], cuts[1:]))
    assert abs((P @ f)[k] - ref) < 1e-10


def test_vectorised_path_matches_the_per_node_quadrature():
    """TriangleGrid.path cuts and quadratures every output node at once; against a per-node oracle written the
    way the loop was (the read point evaluated at the ends of the node's range, a cut at every breakpoint
    crossing of t and of a, of the triangles' diagonals, of the known's age grid and at the extra cuts, to
    1e-12 inside the range, rounded to 13 decimals, Gauss-Legendre on every interval) the rows, points and
    weights of every path family a solve builds are identical, and the read points are the per-node ones:
    examples/ch1_delayed_finite.yaml at 8 nodes and Chapter 3 as its own past and continuation at 6 nodes
    (the band, the buffer, the past's paths)."""
    from numpy.polynomial import legendre
    from noisestate import triangle
    from helpers import example, example_dict, stationary, same_model_solver
    calls = []; orig = triangle.TriangleGrid.path

    def rec(self, *a, **k):
        lp = orig(self, *a, **k); calls.append((self, a, k, lp)); return lp
    triangle.TriangleGrid.path = rec
    ns.clear_grid_cache()                     # the grids are shared across models and cache their paths: count fresh builds
    try:
        d = example_dict("ch1_delayed_finite"); d.setdefault("numerics", {})["nodes"] = 8
        ns.solve(ns.Model.from_dict(d))
        m3 = example("ch3_two_player"); stat = stationary(m3, 6)
        same_model_solver(m3, stat, 6.0, 6).solve(start_policy="stationary")
    finally:
        triangle.TriangleGrid.path = orig
    assert len(calls) >= 12

    def oracle(g, r_lo, r_hi, point_fn, known_fn, extra_cuts, known_grid, m):
        xg, wg = legendre.leggauss(m)
        rows, rq_all, wts, pts = [], [], [], []
        for k in range(len(r_lo)):
            lo, hi = float(r_lo[k]), float(r_hi[k])
            if hi - lo <= 1e-13:
                continue
            cuts = set()

            def crossings(fn):
                t0, a0 = fn(k, np.array([lo])); t1, a1 = fn(k, np.array([hi]))
                for b in g.bp:
                    for (v0, v1) in ((t0[0], t1[0]), (a0[0], a1[0])):
                        if abs(v1 - v0) > 1e-14:
                            r = lo + (b - v0) / (v1 - v0) * (hi - lo)
                            if lo + 1e-12 < r < hi - 1e-12:
                                cuts.add(round(r, 13))
                if g.L is not None:
                    for org in g.origins:
                        v0, v1 = a0[0] - (t0[0] - org), a1[0] - (t1[0] - org)
                        if abs(v1 - v0) > 1e-14:
                            r = lo + (0.0 - v0) / (v1 - v0) * (hi - lo)
                            if lo + 1e-12 < r < hi - 1e-12:
                                cuts.add(round(r, 13))
            crossings(point_fn)
            if known_fn is not None and known_grid is None:
                crossings(known_fn)
            elif known_fn is not None:
                x0 = float(known_fn(k, np.array([lo]))[0]); x1 = float(known_fn(k, np.array([hi]))[0])
                if abs(x1 - x0) > 1e-14:
                    for b in known_grid.breakpoints:
                        r = lo + (b - x0) / (x1 - x0) * (hi - lo)
                        if lo + 1e-12 < r < hi - 1e-12:
                            cuts.add(round(r, 13))
            if extra_cuts is not None:
                for r in extra_cuts(k):
                    if lo + 1e-12 < r < hi - 1e-12:
                        cuts.add(round(float(r), 13))
            edges = sorted([lo, hi] + list(cuts))
            for e0, e1 in zip(edges[:-1], edges[1:]):
                rq = 0.5 * (e1 - e0) * xg + 0.5 * (e0 + e1)
                rows.append(np.full(len(rq), k)); rq_all.append(rq); wts.append(0.5 * (e1 - e0) * wg)
                pts.append(np.stack(point_fn(k, rq)))
        if not rows:
            return None
        return np.concatenate(rows), np.concatenate(rq_all), np.concatenate(wts), np.concatenate(pts, axis=1)
    import inspect
    for (g, a, k, lp) in calls:
        b = inspect.signature(orig).bind(g, *a, **k); b.apply_defaults(); p = b.arguments
        ref = oracle(g, p["r_lo"], p["r_hi"], p["point_fn"], p["known_fn"], p["extra_cuts"], p["known_grid"], p["m"] or (max(g.nt, g.na) + 2))
        point_fn = p["point_fn"]
        if ref is None:
            assert lp.rows is None; continue
        rows, r, w, pts = ref
        assert np.array_equal(lp.rows, rows) and np.array_equal(lp.r, r) and np.array_equal(lp.w, w)
        pt, pa = point_fn(lp.rows, lp.r)                                       # the vector call gives the per-node values
        assert np.array_equal(np.broadcast_to(pt, r.shape), pts[0]) and np.array_equal(np.broadcast_to(pa, r.shape), pts[1])
