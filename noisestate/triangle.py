"""Piecewise-spectral representation of kernels K(t, s), 0 <= s <= t <= T.

Coordinates are time t and shock age a = t - s.  The domain 0 <= a <= t <= T is
cut by the same breakpoints in t and in a (multiples of the delays, and T),
which is where kernels kink: delay lines are the vertical lines a = k tau, and
the start of a delayed effect is the horizontal line t = k tau.  A piece with
age panel q below time panel p is a rectangle; the piece with q = p is the
triangle {t_p <= a <= t <= t_{p+1}}, parametrised by Duffy coordinates
a = t_p + theta (t - t_p), theta in [0, 1].  Each piece carries a tensor grid
of Chebyshev-Lobatto nodes, so a kernel that is analytic on every piece is
represented with spectral accuracy, and delays are exact.

Operators are dense matrices on the global nodal vector, built by Gauss
quadrature along lines with the integration range split at every piece
boundary the line crosses:

    interp(points)                    2-D barycentric interpolation at arbitrary (t, a)
    line_op(...)                      generic  F(node) = sum_q w_q phi(node, r_q) f(point(node, r_q))
    row_weights(t)                    quadrature over a on the row of pieces at time t
    mass                              2-D quadrature weights over the domain
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.polynomial import legendre

from .grid import bary_weights, cheb_lobatto, clenshaw_curtis


class Piece:
    def __init__(self, p: int, q: int, t0: float, t1: float, a0: float, a1: float, nt: int, na: int, offset: int):
        self.p, self.q = p, q
        self.t0, self.t1, self.a0, self.a1 = t0, t1, a0, a1
        self.triangle = (p == q)
        self.nt, self.na = nt, na
        self.offset = offset
        self.tn = cheb_lobatto(nt, t0, t1)
        self.xn = cheb_lobatto(na, 0.0, 1.0) if self.triangle else cheb_lobatto(na, a0, a1)   # theta or a
        self.wt, self.wx = bary_weights(nt), bary_weights(na)
        T, X = np.meshgrid(self.tn, self.xn, indexing="ij")          # (nt, na)
        if self.triangle:
            A = t0 + X * (T - t0)
        else:
            A = X
        self.t = T.reshape(-1); self.a = A.reshape(-1)
        self.n = nt * na

    def local_coords(self, t, a):
        """(t, x) with x = theta (triangle) or a (rectangle)."""
        if self.triangle:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(t - self.t0 > 1e-14, (a - self.t0) / (t - self.t0), 0.0)
            return t, np.clip(x, 0.0, 1.0)
        return t, a

    def contains(self, t, a, tol=1e-12):
        if self.triangle:
            return (t >= self.t0 - tol) & (t <= self.t1 + tol) & (a >= self.t0 - tol) & (a <= t + tol)
        return (t >= self.t0 - tol) & (t <= self.t1 + tol) & (a >= self.a0 - tol) & (a <= self.a1 + tol)


def _bary_rows(pts: np.ndarray, xs: np.ndarray, w: np.ndarray) -> np.ndarray:
    d = pts[:, None] - xs[None, :]
    exact = np.abs(d) < 1e-13
    with np.errstate(divide="ignore", invalid="ignore"):
        r = w[None, :] / d
    r[exact] = 0.0
    out = r / r.sum(axis=1, keepdims=True)
    rows = np.where(exact.any(axis=1))[0]
    out[rows] = 0.0
    out[rows, np.argmax(exact[rows], axis=1)] = 1.0
    return out


class TriangleGrid:
    def __init__(self, breakpoints, nt: int = 16, na: int = 16):
        bp = np.asarray(sorted(set(float(b) for b in breakpoints)))
        if bp[0] != 0.0 or len(bp) < 2:
            raise ValueError("breakpoints must start at 0 and end at T")
        self.bp = bp
        self.T = float(bp[-1])
        self.P = len(bp) - 1
        self.nt, self.na = nt, na
        self.pieces: List[Piece] = []
        off = 0
        for p in range(self.P):
            for q in range(p + 1):
                pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off)
                self.pieces.append(pc); off += pc.n
        self.N = off
        self.t = np.concatenate([pc.t for pc in self.pieces])
        self.a = np.concatenate([pc.a for pc in self.pieces])
        self.s = self.t - self.a
        self.a0 = np.concatenate([np.full(pc.n, pc.a0) for pc in self.pieces])      # age-panel start of each node's piece
        self.side_a = np.concatenate([np.where(np.abs(pc.a - pc.a1) < 1e-13, -1, 1) if not pc.triangle else np.ones(pc.n, dtype=int)
                                      for pc in self.pieces])
        self.side_t = np.concatenate([np.where(np.abs(pc.t - pc.t1) < 1e-13, -1, 1) for pc in self.pieces])
        self._mass = None
        self._piece_by_pq = {(pc.p, pc.q): pc for pc in self.pieces}

    # ------------------------------------------------------------ lookup
    def panel_of(self, x, side=+1):
        x = np.asarray(x, dtype=float)
        if side > 0:
            p = np.searchsorted(self.bp, x, side="right") - 1
        else:
            p = np.searchsorted(self.bp, x, side="left") - 1
        return np.clip(p, 0, self.P - 1)

    def interp(self, t, a, side_t=+1, side_a=+1) -> np.ndarray:
        """Matrix I (npts x N): values of a kernel at (t, a).  Points outside the domain
        (a < 0, a > t, t > T) get zero rows.  At a breakpoint, side selects the piece
        (+1: the piece above/right, -1: below/left) so one-sided limits are available."""
        t = np.atleast_1d(np.asarray(t, dtype=float)); a = np.atleast_1d(np.asarray(a, dtype=float))
        I = np.zeros((len(t), self.N))
        inside = (a >= -1e-12) & (a <= t + 1e-12) & (t <= self.T + 1e-12) & (t >= -1e-12)
        tc = np.clip(t, 0.0, self.T); ac = np.clip(a, 0.0, tc)
        st = np.broadcast_to(np.asarray(side_t), tc.shape); sa = np.broadcast_to(np.asarray(side_a), ac.shape)
        p = np.where(st > 0, self.panel_of(tc, +1), self.panel_of(tc, -1))
        q = np.where(sa > 0, self.panel_of(ac, +1), self.panel_of(ac, -1))
        q = np.minimum(q, p)
        for pc in self.pieces:
            sel = np.where(inside & (p == pc.p) & (q == pc.q))[0]
            if len(sel) == 0:
                continue
            tt, xx = pc.local_coords(tc[sel], ac[sel])
            Rt = _bary_rows(tt, pc.tn, pc.wt)                # (m, nt)
            Rx = _bary_rows(xx, pc.xn, pc.wx)                # (m, na)
            I[np.ix_(sel, np.arange(pc.offset, pc.offset + pc.n))] = (Rt[:, :, None] * Rx[:, None, :]).reshape(len(sel), -1)
        return I

    def evaluate(self, f, t, a):
        return self.interp(t, a) @ f

    # --------------------------------------------------------- quadrature
    def row_weights(self, t: float, side=+1) -> np.ndarray:
        """Weights w (N,) with int_0^t f(t, a) da = w @ f, for a time node t (exact on the
        row's polynomial spaces).  Uses Clenshaw-Curtis on each piece crossed at time t."""
        w = np.zeros(self.N)
        p = int(self.panel_of(t, side))
        for pc in self.pieces:
            if pc.p != p:
                continue
            if pc.triangle:
                lo, hi = pc.t0, t
                if hi - lo < 1e-14:
                    continue
                an = lo + pc.xn * (hi - lo)
                wa = clenshaw_curtis(pc.na, lo, hi)
            else:
                an = pc.xn; wa = clenshaw_curtis(pc.na, pc.a0, pc.a1)
            # values at (t, an): interpolate in t on this piece's nodes only
            tt, xx = pc.local_coords(np.full(pc.na, t), an)
            Rt = _bary_rows(tt, pc.tn, pc.wt)                  # (na, nt)
            Rx = _bary_rows(xx, pc.xn, pc.wx)                  # (na, na)
            W = (Rt[:, :, None] * Rx[:, None, :]).reshape(pc.na, -1)   # (na, n)
            w[pc.offset:pc.offset + pc.n] += wa @ W
        return w

    @property
    def mass(self) -> np.ndarray:
        """Weights (N,) with int_0^T int_0^t f(t, a) da dt = mass @ f."""
        if self._mass is None:
            m = np.zeros(self.N)
            for p in range(self.P):
                tq, wq = legendre.leggauss(self.nt + 2)
                lo, hi = self.bp[p], self.bp[p + 1]
                for x, w in zip(0.5 * (hi - lo) * tq + 0.5 * (hi + lo), 0.5 * (hi - lo) * wq):
                    m += w * self.row_weights(x)
            self._mass = m
        return self._mass

    # ------------------------------------------------------ helpers
    @staticmethod
    def breakpoints(T: float, delays, unit: Optional[float] = None) -> List[float]:
        delays = [float(d) for d in delays if d and d > 0]
        if not delays:
            return [0.0, T]
        unit = unit or min(delays)
        bp = list(np.arange(0.0, T - 1e-12, unit)) + [T]
        if T - bp[-2] < 0.25 * unit and len(bp) > 2:
            bp.pop(-2)
        return bp

    # ---------------------------------------------------------- line ops
    def path(self, out_t, out_a, r_lo, r_hi, point_fn: Callable, known_fn: Optional[Callable] = None,
             extra_cuts: Optional[Callable] = None, m: Optional[int] = None, side_t=+1, side_a=+1) -> "LinePath":
        """Quadrature structure of a family of line integrals (one per output node):
            F_k = int_{r_lo[k]}^{r_hi[k]} w(k, r) f(point_fn(k, r)) dr,
        cut at every r where the read point crosses a breakpoint (and at extra_cuts).
        The result caches the unknown's interpolation rows at the quadrature points and,
        if known_fn(k, r) -> (t, a) is given, the interpolation rows of a known kernel
        there, so operators for any weight are two sparse products (see LinePath)."""
        from scipy.sparse import csr_matrix
        out_t = np.atleast_1d(np.asarray(out_t, dtype=float)); out_a = np.atleast_1d(np.asarray(out_a, dtype=float))
        n_out = len(out_t)
        m = m or (max(self.nt, self.na) + 2)
        xg, wg = legendre.leggauss(m)
        rows, rq_all, wts = [], [], []
        for k in range(n_out):
            lo, hi = float(r_lo[k]), float(r_hi[k])
            if hi - lo <= 1e-13:
                continue
            t0, a0 = point_fn(k, np.array([lo])); t1, a1 = point_fn(k, np.array([hi]))
            cuts = set()
            for b in self.bp:
                for (v0, v1) in ((t0[0], t1[0]), (a0[0], a1[0])):
                    if abs(v1 - v0) > 1e-14:
                        r = lo + (b - v0) / (v1 - v0) * (hi - lo)
                        if lo + 1e-12 < r < hi - 1e-12:
                            cuts.add(round(r, 13))
            if known_fn is not None:
                kt0, ka0 = known_fn(k, np.array([lo])); kt1, ka1 = known_fn(k, np.array([hi]))
                for b in self.bp:
                    for (v0, v1) in ((kt0[0], kt1[0]), (ka0[0], ka1[0])):
                        if abs(v1 - v0) > 1e-14:
                            r = lo + (b - v0) / (v1 - v0) * (hi - lo)
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
        lp = LinePath(n_out, self.N)
        if not rows:
            return lp
        lp.rows = np.concatenate(rows); lp.r = np.concatenate(rq_all); lp.w = np.concatenate(wts)
        pt = np.empty_like(lp.r); pa = np.empty_like(lp.r)
        for k in np.unique(lp.rows):
            sel = lp.rows == k
            tt, aa = point_fn(int(k), lp.r[sel]); pt[sel] = tt; pa[sel] = aa
        lp.I = csr_matrix(self.interp(pt, pa, side_t=side_t, side_a=side_a))
        if known_fn is not None:
            kt = np.empty_like(lp.r); ka = np.empty_like(lp.r)
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                tt, aa = known_fn(int(k), lp.r[sel]); kt[sel] = tt; ka[sel] = aa
            lp.J = csr_matrix(self.interp(kt, ka))
        lp.out_t = out_t; lp.out_a = out_a
        lp.R = csr_matrix((np.ones(len(lp.rows)), (lp.rows, np.arange(len(lp.rows)))), shape=(n_out, len(lp.rows)))
        return lp

    def line_op(self, out_t, out_a, r_lo, r_hi, point_fn: Callable, weight_fn: Optional[Callable] = None,
                extra_cuts: Optional[Callable] = None, m: Optional[int] = None, side_t=+1, side_a=+1) -> np.ndarray:
        """Dense (n_out x N) operator for one weight function (convenience over path())."""
        lp = self.path(out_t, out_a, r_lo, r_hi, point_fn, None, extra_cuts, m, side_t, side_a)
        if lp.rows is None:
            return np.zeros((len(np.atleast_1d(out_t)), self.N))
        w = lp.w.copy().astype(complex if False else float)
        if weight_fn is not None:
            wf = np.empty(len(lp.r), dtype=complex)
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                wf[sel] = weight_fn(int(k), lp.r[sel])
            if np.abs(wf.imag).max() > 0:
                return lp.apply(wf)
            w = w * wf.real
            return lp.apply_weights(w)
        return lp.apply_weights(w)


class LinePath:
    """Cached quadrature of a family of line integrals; see TriangleGrid.path."""

    def __init__(self, n_out: int, N: int):
        self.n_out, self.N = n_out, N
        self.rows = None; self.r = None; self.w = None; self.I = None; self.J = None; self.R = None

    def apply_weights(self, wts: np.ndarray) -> np.ndarray:
        """Operator with total quadrature weights wts (already including the base weights)."""
        from scipy.sparse import diags
        if self.rows is None:
            return np.zeros((self.n_out, self.N))
        return (self.R @ (diags(wts) @ self.I)).toarray()

    def apply(self, factor: np.ndarray) -> np.ndarray:
        """Operator with weight = base weight x factor (factor evaluated at the quadrature points)."""
        if self.rows is None:
            return np.zeros((self.n_out, self.N), dtype=np.result_type(factor, float))
        from scipy.sparse import diags
        M = (self.R @ (diags(self.w * factor) @ self.I))
        return M.toarray()

    def with_known(self, kernel: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        """Operator whose weight is the known kernel read along the path (times `extra` per point)."""
        if self.rows is None:
            return np.zeros((self.n_out, self.N))
        f = self.J @ kernel
        if extra is not None:
            f = f * extra
        return self.apply(f)


