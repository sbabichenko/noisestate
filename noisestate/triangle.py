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

from typing import Callable, List, Optional

from functools import cached_property

import numpy as np
from numpy.polynomial import legendre

from .grid import bary_rows as _bary_rows, bary_weights, cheb_lobatto, clenshaw_curtis


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
        self._piece_by_pq = {(pc.p, pc.q): pc for pc in self.pieces}
        self.mass_matrices = {}              # rho -> mass_matrix(rho)
        self.read_cache = {}                 # (dt, da) -> read matrix (filled by the spectral engine)
        self.paths = {}                      # key -> LinePath (filled by the spectral engine)
        self._row_weights = {}               # (t, side) -> row_weights(t, side)

    # ------------------------------------------------------------ lookup
    def panel_of(self, x, side=+1):
        """Panel containing x; at a breakpoint the side decides (+1 above, -1 below).  A point within
        round-off of a breakpoint is snapped to it first, so the choice is the side's and never the sign
        of the round-off (a read time t - a with t and a on the same panel edge lands an ulp off)."""
        x = np.array(x, dtype=float)
        eps = 1e-12 * max(1.0, self.T)
        for b in self.bp:
            x[np.abs(x - b) <= eps] = b
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

    def interp_factors(self, t, a, side_t=+1, side_a=+1):
        """The interpolation at (t, a) as its factors: [(piece, point indices, Rt (m, nt), Rx (m, na))] over the
        pieces the points fall in (a point outside the domain is in no piece); the interpolation row of a point is
        the outer product Rt Rx' over the piece's nodes."""
        t = np.atleast_1d(np.asarray(t, dtype=float)); a = np.atleast_1d(np.asarray(a, dtype=float))
        inside = (a >= -1e-12) & (a <= t + 1e-12) & (t <= self.T + 1e-12) & (t >= -1e-12)
        tc = np.clip(t, 0.0, self.T); ac = np.clip(a, 0.0, tc)
        st = np.broadcast_to(np.asarray(side_t), tc.shape); sa = np.broadcast_to(np.asarray(side_a), ac.shape)
        p = np.where(st > 0, self.panel_of(tc, +1), self.panel_of(tc, -1))
        q = np.where(sa > 0, self.panel_of(ac, +1), self.panel_of(ac, -1))
        q = np.minimum(q, p)
        out = []
        for pc in self.pieces:
            sel = np.where(inside & (p == pc.p) & (q == pc.q))[0]
            if len(sel) == 0:
                continue
            tt, xx = pc.local_coords(tc[sel], ac[sel])
            out.append((pc, sel, _bary_rows(tt, pc.tn, pc.wt), _bary_rows(xx, pc.xn, pc.wx)))
        return out

    def interp_sparse(self, t, a, side_t=+1, side_a=+1, factors=None):
        """interp() as a CSR matrix built directly: each point touches one piece's nt x na nodes, so the
        dense form (npts x N) is wasteful for the tens of thousands of quadrature points of a path."""
        from scipy.sparse import csr_matrix
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if factors is None:
            factors = self.interp_factors(t, a, side_t, side_a)
        rows, cols, vals = [], [], []
        for pc, sel, Rt, Rx in factors:
            vals.append((Rt[:, :, None] * Rx[:, None, :]).reshape(len(sel), -1).ravel())
            rows.append(np.repeat(sel, pc.n)); cols.append(np.tile(np.arange(pc.offset, pc.offset + pc.n), len(sel)))
        if not rows:
            return csr_matrix((len(t), self.N))
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(t), self.N))

    def evaluate(self, f, t, a):
        return self.interp(t, a) @ f

    # --------------------------------------------------------- quadrature
    def row_weights(self, t: float, side=+1) -> np.ndarray:
        """Weights w (N,) with int_0^t f(t, a) da = w @ f, for a time node t (exact on the
        row's polynomial spaces).  Uses Clenshaw-Curtis on each piece crossed at time t.
        Cached per (t, side): the array is shared, so index or copy it rather than write to it."""
        key = (float(t), int(side))
        if key not in self._row_weights:
            self._row_weights[key] = self._row_weights_at(float(t), side)
        return self._row_weights[key]

    def _row_weights_at(self, t: float, side=+1) -> np.ndarray:
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

    @cached_property
    def mass(self) -> np.ndarray:
        """Weights (N,) with int_0^T int_0^t f(t, a) da dt = mass @ f."""
        m = np.zeros(self.N)
        for p in range(self.P):
            tq, wq = legendre.leggauss(self.nt + 2)
            lo, hi = self.bp[p], self.bp[p + 1]
            for x, w in zip(0.5 * (hi - lo) * tq + 0.5 * (hi + lo), 0.5 * (hi - lo) * wq):
                m += w * self.row_weights(x)
        return m

    # ------------------------------------------------------ helpers
    @staticmethod
    def breakpoints(T: float, lags, unit: Optional[float] = None) -> List[float]:
        """Uniform time/age panels of width `unit` (default: the smallest lag) up to T, then T.  When T is
        not a multiple of the unit the last panel is the remainder, however short: nothing is merged, since
        a trailing breakpoint can itself be a lag (a delay of 0.9 on a window of 1), and the compile closes
        the panels under the lags anyway, which adds T - k unit back.  Only a remainder within round-off of
        the unit (T a multiple of it to 1e-9) is dropped."""
        lags = [float(d) for d in lags if d and d > 0]
        if not lags:
            return [0.0, T]
        unit = unit or min(lags)
        bp = list(np.arange(0.0, T - 1e-12, unit)) + [T]
        if T - bp[-2] < 1e-9 * max(1.0, T):
            bp.pop(-2)
        return bp

    def mass_matrix(self, rho: float = 0.0) -> np.ndarray:
        """Exact Gram matrix M_ij = int_0^T e^{-rho t} int_0^t l_i l_j da dt of the nodal basis
        (tensor Gauss quadrature on every piece; Duffy Jacobian on the triangles)."""
        key = round(float(rho), 12)
        if key in self.mass_matrices:
            return self.mass_matrices[key]
        weight_t = (lambda t: np.exp(-rho * t)) if rho else None
        M = np.zeros((self.N, self.N))
        xg, wg = legendre.leggauss(max(self.nt, self.na) + 2)
        for pc in self.pieces:
            tq = 0.5 * (pc.t1 - pc.t0) * xg + 0.5 * (pc.t1 + pc.t0); tw = 0.5 * (pc.t1 - pc.t0) * wg
            for t, wt in zip(tq, tw):
                if pc.triangle:
                    lo, hi = pc.t0, t
                else:
                    lo, hi = pc.a0, pc.a1
                if hi - lo <= 1e-14:
                    continue
                aq = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); aw = 0.5 * (hi - lo) * wg
                I = self.interp(np.full_like(aq, t), aq)
                w = wt * aw * (weight_t(t) if weight_t is not None else 1.0)
                M += (I * w[:, None]).T @ I
        self.mass_matrices[key] = M
        return M

    # ---------------------------------------------------------- line ops
    def path(self, out_t, out_a, r_lo, r_hi, point_fn: Callable, known_fn: Optional[Callable] = None,
             extra_cuts: Optional[Callable] = None, m: Optional[int] = None, side_t=+1, side_a=+1) -> "LinePath":
        """Quadrature structure of a family of line integrals (one per output node):
            F_k = int_{r_lo[k]}^{r_hi[k]} w(k, r) f(point_fn(k, r)) dr,
        cut at every r where the read point crosses a breakpoint (and at extra_cuts).
        side_t may be one side per output node: a read at a breakpoint time (the node's own time, or its
        shift by a delay) then takes that node's side.
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
            cuts = self._crossings(point_fn, k, lo, hi)
            if known_fn is not None:
                cuts |= self._crossings(known_fn, k, lo, hi)
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
        # a read at a breakpoint time (the output node's own time, or its shift by a delay) takes the
        # node's side of it (side_t per node), so a node on the top edge of its piece reads the time
        # row of its own limit, not the next panel's; every other read is interior to the cuts
        side_t = np.asarray(side_t)
        at_bp = lambda tt: np.min(np.abs(tt[:, None] - self.bp[None, :]), axis=1) < 1e-12
        st_I = np.where(at_bp(pt), side_t[lp.rows], +1) if side_t.ndim == 1 else side_t
        lp.I = self.interp_sparse(pt, pa, side_t=st_I, side_a=side_a)
        if known_fn is not None:
            kt = np.empty_like(lp.r); ka = np.empty_like(lp.r)
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                tt, aa = known_fn(int(k), lp.r[sel]); kt[sel] = tt; ka[sel] = aa
            st_J = np.where(at_bp(kt), side_t[lp.rows], +1) if side_t.ndim == 1 else +1
            lp.Jf = self.interp_factors(kt, ka, side_t=st_J)            # the known kernel is read through its factors
            lp.J = self.interp_sparse(kt, ka, side_t=st_J, factors=lp.Jf)
        lp.out_t = out_t; lp.out_a = out_a
        lp.R = csr_matrix((np.ones(len(lp.rows)), (lp.rows, np.arange(len(lp.rows)))), shape=(n_out, len(lp.rows)))
        return lp

    def _crossings(self, fn: Callable, k: int, lo: float, hi: float) -> set:
        """Parameter values r in (lo, hi) at which the read point fn(k, r) (linear in r) crosses a breakpoint."""
        t0, a0 = fn(k, np.array([lo])); t1, a1 = fn(k, np.array([hi]))
        cuts = set()
        for b in self.bp:
            for (v0, v1) in ((t0[0], t1[0]), (a0[0], a1[0])):
                if abs(v1 - v0) > 1e-14:
                    r = lo + (b - v0) / (v1 - v0) * (hi - lo)
                    if lo + 1e-12 < r < hi - 1e-12:
                        cuts.add(round(r, 13))
        return cuts

    def line_op(self, out_t, out_a, r_lo, r_hi, point_fn: Callable, weight_fn: Optional[Callable] = None,
                extra_cuts: Optional[Callable] = None, m: Optional[int] = None, side_t=+1, side_a=+1) -> np.ndarray:
        """Dense (n_out x N) operator for one weight function (convenience over path())."""
        lp = self.path(out_t, out_a, r_lo, r_hi, point_fn, None, extra_cuts, m, side_t, side_a)
        if lp.rows is None:
            return np.zeros((len(np.atleast_1d(out_t)), self.N))
        w = lp.w.astype(float)
        if weight_fn is not None:
            wf = np.empty(len(lp.r))
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                wf[sel] = weight_fn(int(k), lp.r[sel])
            w = w * wf
            return lp.apply_weights(w)
        return lp.apply_weights(w)


class LinePath:
    """Cached quadrature of a family of line integrals; see TriangleGrid.path."""

    def __init__(self, n_out: int, N: int):
        self.n_out, self.N = n_out, N
        self.rows = None; self.r = None; self.w = None; self.I = None; self.J = None; self.R = None; self.Jf = None

    def read(self, kernels: np.ndarray) -> np.ndarray:
        """J @ kernels (nq, m) for kernels (N, m): the known kernels at the quadrature points, read piece by piece
        through the interpolation factors as dense products Rt (K_piece) Rx' (the same sums as the sparse product,
        associated over the piece's time nodes first; identical to round-off)."""
        K = kernels if kernels.ndim == 2 else kernels[:, None]
        m = K.shape[1]
        F = np.zeros((self.J.shape[0], m))
        for pc, sel, Rt, Rx in self.Jf:
            KB = K[pc.offset:pc.offset + pc.n].reshape(pc.nt, pc.na * m)
            F[sel] = np.einsum("qjc,qj->qc", (Rt @ KB).reshape(len(sel), pc.na, m), Rx)
        return F if kernels.ndim == 2 else F[:, 0]

    def swapped(self) -> "LinePath":
        """The same path with the unknown's and the known's read matrices exchanged: the integral of a
        product of two kernels along the same line, with the other factor as the unknown.  Shares every
        array (nothing is copied), so the two operators of a geometry cost one set of read matrices."""
        lp = LinePath(self.n_out, self.N)
        lp.rows, lp.r, lp.w, lp.R = self.rows, self.r, self.w, self.R
        lp.I, lp.J = self.J, self.I
        lp.out_t = getattr(self, "out_t", None); lp.out_a = getattr(self, "out_a", None)
        return lp

    def apply_weights(self, wts: np.ndarray) -> np.ndarray:
        """Operator with total quadrature weights wts (already including the base weights)."""
        if self.rows is None:
            return np.zeros((self.n_out, self.N))
        return self._weighted_sum(np.asarray(wts, dtype=float))

    @cached_property
    def _sum_layout(self):
        """Layout of the weighted row sum R (diag(d) I): the quadrature points are grouped by output node in
        increasing order, so the sum is the CSR matrix with I's data scaled row by row and the row pointers
        of each node's group collapsed into one row, made dense (duplicate columns add in point order, which
        is also the order the sparse product accumulates in: the two agree to the bit).  Returns (indptr of
        the collapsed rows, per-entry row index or None when every row of I has the same count)."""
        if not np.all(np.diff(self.rows) >= 0):
            return None
        nq = len(self.rows)
        starts = np.searchsorted(self.rows, np.arange(self.n_out + 1))
        indptr = self.I.indptr[starts]
        counts = np.diff(self.I.indptr)
        uniform = counts.size > 0 and bool(np.all(counts == counts[0]))
        rowidx = None if uniform else np.repeat(np.arange(nq), counts)
        return indptr, rowidx

    def _weighted_sum(self, d: np.ndarray) -> np.ndarray:
        """R @ (diag(d) @ I) as a dense (n_out, N) array."""
        layout = self._sum_layout
        if layout is None or not np.issubdtype(d.dtype, np.floating):
            from scipy.sparse import diags
            return (self.R @ (diags(d) @ self.I)).toarray()
        indptr, rowidx = layout
        I = self.I
        if rowidx is None:
            data = (I.data.reshape(len(d), -1) * d[:, None]).reshape(-1)
        else:
            data = I.data * d[rowidx]
        from scipy.sparse import _sparsetools
        out = np.zeros((self.n_out, self.N))
        _sparsetools.csr_todense(self.n_out, self.N, indptr, I.indices, data, out)
        return out

    def apply(self, factor: np.ndarray) -> np.ndarray:
        """Operator with weight = base weight x factor (factor evaluated at the quadrature points)."""
        if self.rows is None:
            return np.zeros((self.n_out, self.N), dtype=np.result_type(factor, float))
        return self._weighted_sum(self.w * factor)

    def with_known(self, kernel: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        """Operator whose weight is the known kernel read along the path (times `extra` per point)."""
        if self.rows is None:
            return np.zeros((self.n_out, self.N))
        f = self.read(kernel) if self.Jf is not None else self.J @ kernel
        if extra is not None:
            f = f * extra
        return self.apply(f)

    def with_known_many(self, kernels: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
        """with_known for every column of kernels (N, m) at once: (m, n_out, N).  One sparse read of all the
        kernels (the sparse product accumulates each column exactly as the single-kernel read does)."""
        m = kernels.shape[1]
        if self.rows is None:
            return np.zeros((m, self.n_out, self.N))
        F = self.read(kernels) if self.Jf is not None else self.J @ kernels     # (nq, m)
        if extra is not None:
            F = F * extra[:, None]
        return np.stack([self.apply(F[:, i]) for i in range(m)])


