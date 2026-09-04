import numpy as np
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
