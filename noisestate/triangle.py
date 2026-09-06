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

With a window L (a transition from a known past) the domain is the strip
[0, T] x [0, L]: below the diagonal a = t today's pieces, unchanged in nodes and
order and cut off at age L (a shock is forgotten at age L, as on the stationary
grid); above it, t < a <= L, the shocks born before time 0 (s = t - a < 0): in
each time panel p with t_{p+1} <= L an upper Duffy triangle {t_p <= t <= a <=
t_{p+1}}, a = t + theta (t_{p+1} - t), and rectangles (p, q) with q > p, listed
after the panel's lower pieces so every panel's nodes stay contiguous.  The two
triangles of a square keep separate nodes on the diagonal, where kernels may jump
(s = 0+ against s = 0-): reads there take the lower side unless side_d says
otherwise.  Without a window nothing of this exists and the grid is today's.

With a buffer (a transition continued past its horizon T_b by one window, on which
the maps are frozen) the time panels from T_b on are the age panels shifted by T_b,
and their pieces are the strip's on [0, L] with the origin at T_b: below the second
diagonal a = t - T_b the shocks born after T_b, above it those born before, which
the excluded agent's passive world (its own strategy off before T_b, frozen after)
kinks along.  A switch of the maps at a time t0 (the regime change at 0, the passive
world's at T_b) reaches a state through a lag tau only at t0 + tau, so the kernels
kink, more weakly at each step, along every line s = t0 - k tau as well: above a
region's diagonal each square piece is therefore split along its own diagonal
a - a0 = t - t0 into two Duffy triangles (Piece.origin = t0 - a0), rectangles
remaining only where the panels are not square.  `upper` on a piece means above
its own diagonal; on the grid, `upper` marks the band (s < 0), `above` the nodes
above their piece's diagonal.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from functools import cached_property

import numpy as np
from numpy.polynomial import legendre

from .grid import bary_rows as _bary_rows, bary_weights, cheb_lobatto, clenshaw_curtis


class Piece:
    def __init__(self, p: int, q: int, t0: float, t1: float, a0: float, a1: float, nt: int, na: int, offset: int,
                 upper: bool = False, origin: float = 0.0, triangle: Optional[bool] = None, band: Optional[bool] = None):
        self.p, self.q = p, q
        self.t0, self.t1, self.a0, self.a1 = t0, t1, a0, a1
        self.origin = origin                 # the piece's diagonal is a = t - origin (0 today; t0 - a0 on a split square)
        self.triangle = (p == q) if triangle is None else triangle
        self.upper = upper                   # above the piece's diagonal; the upper triangle is a = t + theta (t1 - t) at origin 0
        self.band = upper if band is None else band      # holds shocks born before zero (s < 0)
        self.nt, self.na = nt, na
        self.offset = offset
        self.tn = cheb_lobatto(nt, t0, t1)
        self.xn = cheb_lobatto(na, 0.0, 1.0) if self.triangle else cheb_lobatto(na, a0, a1)   # theta or a
        self.wt, self.wx = bary_weights(nt), bary_weights(na)
        T, X = np.meshgrid(self.tn, self.xn, indexing="ij")          # (nt, na)
        if self.triangle and upper:                                  # a in [t - origin, a1]; a1 = t1 at the origin 0
            A = (T - origin) + X * (a1 - (T - origin))
        elif self.triangle:                                          # a in [a0, t - origin]; a0 = t0 at the origin 0
            A = a0 + X * (T - origin - a0)
        else:
            A = X
        self.t = T.reshape(-1); self.a = A.reshape(-1)
        self.n = nt * na

    def local_coords(self, t, a):
        """(t, x) with x = theta (triangle) or a (rectangle)."""
        if self.triangle and self.upper:
            tr = t - self.origin
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(self.a1 - tr > 1e-14, (a - tr) / (self.a1 - tr), 0.0)
            return t, np.clip(x, 0.0, 1.0)
        if self.triangle:
            tr = t - self.origin
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(tr - self.a0 > 1e-14, (a - self.a0) / (tr - self.a0), 0.0)
            return t, np.clip(x, 0.0, 1.0)
        return t, a

    def contains(self, t, a, tol=1e-12):
        if self.triangle and self.upper:
            return (t >= self.t0 - tol) & (t <= self.t1 + tol) & (a >= t - self.origin - tol) & (a <= self.a1 + tol)
        if self.triangle:
            return (t >= self.t0 - tol) & (t <= self.t1 + tol) & (a >= self.a0 - tol) & (a <= t - self.origin + tol)
        return (t >= self.t0 - tol) & (t <= self.t1 + tol) & (a >= self.a0 - tol) & (a <= self.a1 + tol)


class TriangleGrid:
    def __init__(self, breakpoints, nt: int = 16, na: int = 16, T: Optional[float] = None, window: Optional[float] = None,
                 buffer: Optional[float] = None):
        """breakpoints: the shared sequence in t and in a, from 0 to max(T, L).  T (default the last
        breakpoint) ends the time axis, window L (None: no old shocks, no truncation) the age axis; both
        must be breakpoints.  buffer (with a window): the time T_b from which the panels are the age
        panels shifted (T = T_b + L) and the pieces have their origin at T_b (see the module docstring)."""
        bp = np.asarray(sorted(set(float(b) for b in breakpoints)))
        if bp[0] != 0.0 or len(bp) < 2:
            raise ValueError("breakpoints must start at 0 and end at T")
        self.bp = bp
        self.T = float(bp[-1]) if T is None else float(T)
        self.L = None if window is None else float(window)
        at = lambda x: int(np.argmin(np.abs(bp - x)))
        if abs(bp[at(self.T)] - self.T) > 1e-12 * max(1.0, bp[-1]):
            raise ValueError(f"the horizon T = {self.T} is not a breakpoint of {[float(b) for b in bp]}")
        self.P = at(self.T)                              # time panels
        if self.L is None:
            self.PL = self.P                             # age panels below the diagonal: all of them
        else:
            if self.L <= 0 or abs(bp[at(self.L)] - self.L) > 1e-12 * max(1.0, bp[-1]):
                raise ValueError(f"the window L = {self.L} is not a positive breakpoint of {[float(b) for b in bp]}")
            self.PL = at(self.L)
        self.Tb = None if buffer is None else float(buffer)
        self.P_T = self.P                                # time panels before the buffer
        if self.Tb is not None:
            if self.L is None:
                raise ValueError("a buffer needs a window")
            self.P_T = at(self.Tb)
            eps = 1e-9 * max(1.0, bp[-1])
            if abs(bp[self.P_T] - self.Tb) > eps or self.P_T + self.PL != self.P \
                    or any(abs(bp[self.P_T + q] - (self.Tb + bp[q])) > eps for q in range(self.PL + 1)):
                raise ValueError(f"the buffer's time panels must be the age panels shifted by {self.Tb:g}: breakpoints "
                                 f"{[float(b) for b in bp]}, window {self.L:g}")
        self.nt, self.na = nt, na
        self.pieces: List[Piece] = []
        eps = 1e-9 * max(1.0, bp[-1])
        self._split = np.zeros((self.P, max(self.PL, 1)), dtype=bool)        # (p, q) is a square above its region's diagonal, split
        off = 0
        for p in range(self.P):
            if self.L is None:
                for q in range(p + 1):
                    pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off)
                    self.pieces.append(pc); off += pc.n
                continue
            buf = self.Tb is not None and p >= self.P_T
            pr, org = (p - self.P_T, self.Tb) if buf else (p, 0.0)
            for q in range(min(pr, self.PL - 1) + 1):                      # below the region's diagonal
                pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off, origin=org, triangle=(q == pr), band=False)
                self.pieces.append(pc); off += pc.n
            for q in range(pr, self.PL):                                    # above it: the old shocks (band) or those born before T_b
                if q == pr:
                    pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off, upper=True, origin=org, triangle=True, band=not buf)
                    self.pieces.append(pc); off += pc.n
                elif abs((bp[p + 1] - bp[p]) - (bp[q + 1] - bp[q])) < eps:   # a square: split along its diagonal
                    self._split[p, q] = True
                    for up in (False, True):
                        pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off, upper=up, origin=bp[p] - bp[q],
                                   triangle=True, band=not buf)
                        self.pieces.append(pc); off += pc.n
                else:
                    pc = Piece(p, q, bp[p], bp[p + 1], bp[q], bp[q + 1], nt, na, off, upper=True, origin=org, triangle=False, band=not buf)
                    self.pieces.append(pc); off += pc.n
        self.N = off
        self.t = np.concatenate([pc.t for pc in self.pieces])
        self.a = np.concatenate([pc.a for pc in self.pieces])
        self.s = self.t - self.a
        self.a0 = np.concatenate([np.full(pc.n, pc.a0) for pc in self.pieces])      # age-panel start of each node's piece
        self.a1 = np.concatenate([np.full(pc.n, pc.a1) for pc in self.pieces])      # age-panel end of each node's piece
        self.above = np.concatenate([np.full(pc.n, pc.upper) for pc in self.pieces])   # nodes above their piece's diagonal
        self.origin = np.concatenate([np.full(pc.n, pc.origin) for pc in self.pieces])
        self.upper = np.concatenate([np.full(pc.n, pc.band) for pc in self.pieces])    # the band: nodes with s < 0
        self.origins = sorted({float(pc.origin) for pc in self.pieces if pc.triangle})   # the diagonals' origins, cut by paths
        self.side_d = np.where(self.above, 1, -1)                                     # side of the diagonal a node reads
        # a node on its piece's top age edge reads from below.  Without a window a lower triangle's nodes read from
        # above (its top corner (t1, t1) is on the diagonal; kept for bit identity); with one every piece's top
        # corner is at an age breakpoint where kernels may jump (the lower sub-triangle of a split square), and
        # its value is the limit from within the piece
        self.side_a = np.concatenate([np.where(np.abs(pc.a - pc.a1) < 1e-13, -1, 1) if not pc.triangle or pc.upper or self.L is not None
                                      else np.ones(pc.n, dtype=int) for pc in self.pieces])
        self.side_t = np.concatenate([np.where(np.abs(pc.t - pc.t1) < 1e-13, -1, 1) for pc in self.pieces])
        self._piece_by_pq = {(pc.p, pc.q): pc for pc in self.pieces if not pc.upper}
        self._upper_by_pq = {(pc.p, pc.q): pc for pc in self.pieces if pc.upper}
        self.mass_matrices = {}              # rho -> mass_matrix(rho)
        self.read_cache = {}                 # (dt, da) -> read matrix (filled by the spectral engine)
        self.paths = {}                      # key -> LinePath (filled by the spectral engine)
        self._row_weights = {}               # (t, side) -> row_weights(t, side)
        self._row_quad = {}                  # (t, side) -> row_quadrature(t, side)

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
        return np.clip(p, 0, len(self.bp) - 2)

    def _locate(self, t, a, side_t, side_a, side_d):
        """Where the points (t, a) fall: (inside, time panel, age panel, above the diagonal, clipped t, clipped a).
        At a breakpoint the sides select the piece (+1: the piece above/right, -1: below/left); side_a None
        is the plain read: from the right, and at the window's edge a = L the edge value (the only one
        there).  With a window and side_a given, two rules for the reads a node makes with its own sides.
        On a diagonal (a region's a = t - origin, or a split square's own) the side is the one (side_t,
        side_a) imply when they are unambiguous: t+ with a- is the limit s -> s0+, the triangle below the
        diagonal; t- with a+ is s -> s0-, the one above; only when they agree does side_d (+1: the upper
        triangle, -1: the lower one, the default) decide.  And a point at age exactly L with side_a = +1
        is outside the window (the limit from beyond L: the shock is gone) and gets no piece.  Both
        matter to the shifted read of a lagged control, D(t - lag, a - lag) from every node: a node
        (T - lag, a - lag) reading (T, a) with side_t = -1 must land in the piece of the shock it reads, not
        the one side_d names for the reader, which sits on another diagonal, and the read of a node at
        age L - lag with side_a = +1 is zero, not the corner value from inside the window.  Without a
        window nothing here applies."""
        t = np.atleast_1d(np.asarray(t, dtype=float)); a = np.atleast_1d(np.asarray(a, dtype=float))
        st = np.broadcast_to(np.asarray(side_t), t.shape)
        sa = np.broadcast_to(np.asarray(+1 if side_a is None else side_a), a.shape)
        if self.L is None:
            inside = (a >= -1e-12) & (a <= t + 1e-12) & (t <= self.T + 1e-12) & (t >= -1e-12)
            tc = np.clip(t, 0.0, self.T); ac = np.clip(a, 0.0, tc)
            p = np.where(st > 0, self.panel_of(tc, +1), self.panel_of(tc, -1))
            q = np.where(sa > 0, self.panel_of(ac, +1), self.panel_of(ac, -1))
            return inside, p, np.minimum(q, p), np.zeros(len(t), dtype=bool), tc, ac
        sd = np.broadcast_to(np.asarray(side_d), t.shape)
        inside = (a >= -1e-12) & (a <= self.L + 1e-12) & (t <= self.T + 1e-12) & (t >= -1e-12)
        if side_a is not None:
            ss = np.where((st > 0) & (sa < 0), 1, np.where((st < 0) & (sa > 0), -1, 0))   # the s-side the sides imply
            sd = np.where(ss == 0, sd, -ss)
            inside = inside & ~((np.abs(a - self.L) <= 1e-12 * max(1.0, self.T)) & (sa > 0))
        tc = np.clip(t, 0.0, self.T); ac = np.clip(a, 0.0, self.L)
        p = np.where(st > 0, self.panel_of(tc, +1), self.panel_of(tc, -1))
        if self.Tb is None:
            pr, tr = p, tc
        else:                                                # the buffer's panels count from T_b, its diagonal is a = t - T_b
            pr = np.where(p >= self.P_T, p - self.P_T, p); tr = np.where(p >= self.P_T, tc - self.Tb, tc)
        up = (ac > tr + 1e-12) | ((np.abs(ac - tr) <= 1e-12) & (sd > 0))
        ac = np.where(up, ac, np.minimum(ac, tr))
        q = np.where(sa > 0, self.panel_of(ac, +1), self.panel_of(ac, -1))
        q = np.where(up, np.maximum(q, pr), np.minimum(q, pr))
        # a square above the region's diagonal is two triangles along its own diagonal a - a0 = t - t0
        qc = np.minimum(q, self._split.shape[1] - 1)
        split = up & (q > pr) & self._split[p, qc]
        if split.any():
            d2 = (ac - self.bp[qc]) - (tc - self.bp[p])
            up = np.where(split, (d2 > 1e-12) | ((np.abs(d2) <= 1e-12) & (sd > 0)), up)
        return inside, p, q, up, tc, ac

    def interp(self, t, a, side_t=+1, side_a=None, side_d=-1) -> np.ndarray:
        """Matrix I (npts x N): values of a kernel at (t, a).  Points outside the domain
        (a < 0, a > t without a window, a > L with one, t > T) get zero rows.  At a breakpoint, side
        selects the piece (+1: the piece above/right, -1: below/left) so one-sided limits are available;
        on the diagonal side_d picks the lower (-1, default) or the upper (+1) triangle, unless the sides
        given settle it (see _locate); side_a None reads from the right and, at a = L, the edge value."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        I = np.zeros((len(t), self.N))
        inside, p, q, up, tc, ac = self._locate(t, a, side_t, side_a, side_d)
        for pc in self.pieces:
            sel = np.where(inside & (p == pc.p) & (q == pc.q) & (up == pc.upper))[0]
            if len(sel) == 0:
                continue
            tt, xx = pc.local_coords(tc[sel], ac[sel])
            Rt = _bary_rows(tt, pc.tn, pc.wt)                # (m, nt)
            Rx = _bary_rows(xx, pc.xn, pc.wx)                # (m, na)
            I[np.ix_(sel, np.arange(pc.offset, pc.offset + pc.n))] = (Rt[:, :, None] * Rx[:, None, :]).reshape(len(sel), -1)
        return I

    def interp_factors(self, t, a, side_t=+1, side_a=None, side_d=-1):
        """The interpolation at (t, a) as its factors: [(piece, point indices, Rt (m, nt), Rx (m, na))] over the
        pieces the points fall in (a point outside the domain is in no piece); the interpolation row of a point is
        the outer product Rt Rx' over the piece's nodes."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        inside, p, q, up, tc, ac = self._locate(t, a, side_t, side_a, side_d)
        out = []
        for pc in self.pieces:
            sel = np.where(inside & (p == pc.p) & (q == pc.q) & (up == pc.upper))[0]
            if len(sel) == 0:
                continue
            tt, xx = pc.local_coords(tc[sel], ac[sel])
            out.append((pc, sel, _bary_rows(tt, pc.tn, pc.wt), _bary_rows(xx, pc.xn, pc.wx)))
        return out

    def interp_sparse(self, t, a, side_t=+1, side_a=None, factors=None, side_d=-1):
        """interp() as a CSR matrix built directly: each point touches one piece's nt x na nodes, so the
        dense form (npts x N) is wasteful for the tens of thousands of quadrature points of a path."""
        from scipy.sparse import csr_matrix
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if factors is None:
            factors = self.interp_factors(t, a, side_t, side_a, side_d)
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
        """Weights w (N,) with int_0^t f(t, a) da = w @ f (int_0^L with a window), for a time node t (exact on
        the row's polynomial spaces).  Uses Clenshaw-Curtis on each piece crossed at time t.
        Cached per (t, side): the array is shared, so index or copy it rather than write to it."""
        key = (float(t), int(side))
        if key not in self._row_weights:
            self._row_weights[key] = self._row_weights_at(float(t), side)
        return self._row_weights[key]

    def row_quadrature(self, t: float, side=+1):
        """(I, w): the interpolation I (nq, N) of a kernel at the Gauss points (t, a_q) of every piece crossed at
        time t (na + 2 per piece) and the weights w (nq,) with int f(t, a) da = w @ (I @ f), exact for the product
        of two interpolants (row_weights integrates the interpolant of nodal values).  Cached per (t, side)."""
        key = (float(t), int(side))
        if key not in self._row_quad:
            self._row_quad[key] = self._row_weights_at(float(t), side, points=True)
        return self._row_quad[key]

    def _row_weights_at(self, t: float, side=+1, points: bool = False):
        w = np.zeros(self.N)
        p = int(self.panel_of(t, side))
        Is, ws = [], []
        for pc in self.pieces:
            if pc.p != p:
                continue
            if pc.triangle:
                lo, hi = (t - pc.origin, pc.a1) if pc.upper else (pc.a0, t - pc.origin)
                if hi - lo < 1e-14:
                    continue
            else:
                lo, hi = pc.a0, pc.a1
            if points:                                        # Gauss points: exact for products of two interpolants
                xg, wg = legendre.leggauss(pc.na + 2)
                an = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); wa = 0.5 * (hi - lo) * wg
            elif pc.triangle:
                an = lo + pc.xn * (hi - lo); wa = clenshaw_curtis(pc.na, lo, hi)
            else:
                an = pc.xn; wa = clenshaw_curtis(pc.na, pc.a0, pc.a1)
            # values at (t, an): interpolate in t on this piece's nodes only
            tt, xx = pc.local_coords(np.full(len(an), t), an)
            Rt = _bary_rows(tt, pc.tn, pc.wt)                  # (nq, nt)
            Rx = _bary_rows(xx, pc.xn, pc.wx)                  # (nq, na)
            W = (Rt[:, :, None] * Rx[:, None, :]).reshape(len(an), -1)   # (nq, n)
            if points:
                I = np.zeros((len(an), self.N)); I[:, pc.offset:pc.offset + pc.n] = W
                Is.append(I); ws.append(wa)
            else:
                w[pc.offset:pc.offset + pc.n] += wa @ W
        if points:
            return (np.concatenate(Is), np.concatenate(ws)) if Is else (np.zeros((0, self.N)), np.zeros(0))
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
    def breakpoints(T: float, lags, unit: Optional[float] = None, unit_range: Optional[float] = None) -> List[float]:
        """Uniform time/age panels of width `unit` (default: the smallest lag) up to T, then T.  When T is
        not a multiple of the unit the last panel is the remainder, however short: nothing is merged, since
        a trailing breakpoint can itself be a lag (a delay of 0.9 on a window of 1), and the compile closes
        the panels under the lags anyway, which adds T - k unit back.  Only a remainder within round-off of
        the unit (T a multiple of it to 1e-9) is dropped.  With unit_range below T the unit panels stop
        there and geometrically growing ones (fill_geometric) reach T: the kink at the k-th delay line
        weakens with k, as on the stationary grid."""
        lags = [float(d) for d in lags if d and d > 0]
        if not lags:
            return [0.0, T]
        unit = unit or min(lags)
        if unit_range is not None and unit_range < T - 1e-12:
            return TriangleGrid.fill_geometric(set(np.arange(0.0, unit_range + 1e-12, unit)) | {T}, unit)
        bp = list(np.arange(0.0, T - 1e-12, unit)) + [T]
        if T - bp[-2] < 1e-9 * max(1.0, T):
            bp.pop(-2)
        return bp

    @staticmethod
    def fill_geometric(points, unit: float, growth: float = 2.0) -> List[float]:
        """The sorted points with every gap wider than the unit filled by panels growing geometrically from
        the unit (the stationary grid's rule: a last panel shorter than a quarter of the next width is
        merged into it), so the required breakpoints stay and the stretches between them are coarse."""
        pts = sorted(set(round(float(b), 12) for b in points))
        out = [pts[0]]
        for b in pts[1:]:
            w = unit
            while out[-1] < b - 1e-12:
                w *= growth
                nxt = min(b, out[-1] + w)
                if b - nxt < 0.25 * w:
                    nxt = b
                out.append(round(nxt, 12))
        return out

    def mass_matrix(self, rho: float = 0.0, t_lo: Optional[float] = None, t_hi: Optional[float] = None) -> np.ndarray:
        """Exact Gram matrix M_ij = int_0^T e^{-rho t} int_0^t l_i l_j da dt of the nodal basis (over the
        strip [0, T] x [0, L] with a window; tensor Gauss quadrature on every piece; Duffy Jacobian on the
        triangles).  With t_lo / t_hi (breakpoints) the integral runs over the time panels between them
        only (a transition's cost over [0, T] and over its buffer [T, T + L] separately)."""
        key = round(float(rho), 12) if t_lo is None and t_hi is None else (round(float(rho), 12), t_lo, t_hi)
        if key in self.mass_matrices:
            return self.mass_matrices[key]
        weight_t = (lambda t: np.exp(-rho * t)) if rho else None
        M = np.zeros((self.N, self.N))
        xg, wg = legendre.leggauss(max(self.nt, self.na) + 2)
        eps = 1e-12 * max(1.0, self.T)
        for pc in self.pieces:
            if (t_lo is not None and pc.t1 <= t_lo + eps) or (t_hi is not None and pc.t0 >= t_hi - eps):
                continue
            tq = 0.5 * (pc.t1 - pc.t0) * xg + 0.5 * (pc.t1 + pc.t0); tw = 0.5 * (pc.t1 - pc.t0) * wg
            for t, wt in zip(tq, tw):
                if pc.triangle:
                    lo, hi = (t - pc.origin, pc.a1) if pc.upper else (pc.a0, t - pc.origin)
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
             extra_cuts: Optional[Callable] = None, m: Optional[int] = None, side_t=+1, side_a=+1,
             known_grid=None, side_d=-1) -> "LinePath":
        """Quadrature structure of a family of line integrals (one per output node):
            F_k = int_{r_lo[k]}^{r_hi[k]} w(k, r) f(point_fn(k, r)) dr,
        cut at every r where the read point crosses a breakpoint (and at extra_cuts).
        side_t may be one side per output node: a read at a breakpoint time (the node's own time, or its
        shift by a delay) then takes that node's side; side_d likewise for reads on the diagonal.
        The result caches the unknown's interpolation rows at the quadrature points and,
        if known_fn(k, r) -> (t, a) is given, the interpolation rows of a known kernel
        there, so operators for any weight are two sparse products (see LinePath).
        With known_grid (an AgeGrid, a known past's), known_fn(k, r) -> ages, the known is a
        kernel on that grid (its J has the grid's N columns) and the quadrature is also cut where
        the age crosses one of its breakpoints."""
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
            if known_fn is not None and known_grid is None:
                cuts |= self._crossings(known_fn, k, lo, hi)
            elif known_fn is not None:
                cuts |= self._crossings_1d(known_fn, k, lo, hi, known_grid.breakpoints)
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
        side_d = np.asarray(side_d)
        sd_I = side_d[lp.rows] if side_d.ndim == 1 else side_d
        lp.I = self.interp_sparse(pt, pa, side_t=st_I, side_a=side_a, side_d=sd_I)
        if known_fn is not None and known_grid is not None:
            ages = np.empty_like(lp.r)
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                ages[sel] = known_fn(int(k), lp.r[sel])
            lp.J = known_grid.interp(ages)                              # dense (nq, N_past): the known past kernel
        elif known_fn is not None:
            kt = np.empty_like(lp.r); ka = np.empty_like(lp.r)
            for k in np.unique(lp.rows):
                sel = lp.rows == k
                tt, aa = known_fn(int(k), lp.r[sel]); kt[sel] = tt; ka[sel] = aa
            st_J = np.where(at_bp(kt), side_t[lp.rows], +1) if side_t.ndim == 1 else +1
            # the known kernel is read through its factors; a known read on the diagonal (an impulse response
            # at (r, r) from a node at t = 0) is the new-shock side: the node's side_d is the unknown's
            lp.Jf = self.interp_factors(kt, ka, side_t=st_J)
            lp.J = self.interp_sparse(kt, ka, side_t=st_J, factors=lp.Jf)
        lp.out_t = out_t; lp.out_a = out_a
        lp.R = csr_matrix((np.ones(len(lp.rows)), (lp.rows, np.arange(len(lp.rows)))), shape=(n_out, len(lp.rows)))
        return lp

    def _crossings(self, fn: Callable, k: int, lo: float, hi: float) -> set:
        """Parameter values r in (lo, hi) at which the read point fn(k, r) (linear in r) crosses a breakpoint
        (or, with a window, the diagonal a = t)."""
        t0, a0 = fn(k, np.array([lo])); t1, a1 = fn(k, np.array([hi]))
        cuts = set()
        for b in self.bp:
            for (v0, v1) in ((t0[0], t1[0]), (a0[0], a1[0])):
                if abs(v1 - v0) > 1e-14:
                    r = lo + (b - v0) / (v1 - v0) * (hi - lo)
                    if lo + 1e-12 < r < hi - 1e-12:
                        cuts.add(round(r, 13))
        if self.L is not None:
            for org in self.origins:                                       # the diagonals a = t - origin of the triangles
                v0, v1 = a0[0] - (t0[0] - org), a1[0] - (t1[0] - org)
                if abs(v1 - v0) > 1e-14:
                    r = lo + (0.0 - v0) / (v1 - v0) * (hi - lo)
                    if lo + 1e-12 < r < hi - 1e-12:
                        cuts.add(round(r, 13))
        return cuts

    @staticmethod
    def _crossings_1d(fn: Callable, k: int, lo: float, hi: float, breakpoints) -> set:
        """Parameter values r in (lo, hi) at which the age fn(k, r) (linear in r) crosses a breakpoint of a 1-D grid."""
        x0 = float(fn(k, np.array([lo]))[0]); x1 = float(fn(k, np.array([hi]))[0])
        cuts = set()
        if abs(x1 - x0) > 1e-14:
            for b in breakpoints:
                r = lo + (b - x0) / (x1 - x0) * (hi - lo)
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

    @cached_property
    def _starts(self):
        """First quadrature point of every output node (and the total), the points being grouped by node."""
        return np.searchsorted(self.rows, np.arange(self.n_out + 1))

    def points(self, rows):
        """The quadrature-point range [i0, i1) of the output nodes [lo, hi) = rows."""
        lo, hi = rows
        return int(self._starts[lo]), int(self._starts[hi])

    def _weighted_sum(self, d: np.ndarray, rows=None) -> np.ndarray:
        """R @ (diag(d) @ I) as a dense (n_out, N) array; with rows = (lo, hi) the rows of the output nodes
        [lo, hi) only, (hi - lo, N), from d at their quadrature points (d given at every point, or at
        theirs).  The rows are the same sums in the same order whether or not the others are built."""
        layout = self._sum_layout
        if rows is None:
            lo, hi = 0, self.n_out
        else:
            lo, hi = rows
        if layout is None or not np.issubdtype(d.dtype, np.floating):
            from scipy.sparse import diags
            R = self.R if rows is None else self.R[lo:hi]
            return (R @ (diags(d) @ self.I)).toarray()
        indptr, rowidx = layout
        I = self.I
        i0, i1 = (0, len(self.rows)) if rows is None else self.points(rows)
        if i1 == i0:
            return np.zeros((hi - lo, self.N))
        if len(d) != i1 - i0:
            d = d[i0:i1]
        j0, j1 = int(indptr[lo]), int(indptr[hi])
        if rowidx is None:
            data = (I.data[j0:j1].reshape(i1 - i0, -1) * d[:, None]).reshape(-1)
        else:
            data = I.data[j0:j1] * d[rowidx[j0:j1] - i0]
        from scipy.sparse import _sparsetools
        out = np.zeros((hi - lo, self.N))
        _sparsetools.csr_todense(hi - lo, self.N, indptr[lo:hi + 1] - j0, I.indices[j0:j1], data, out)
        return out

    def apply(self, factor: np.ndarray, rows=None) -> np.ndarray:
        """Operator with weight = base weight x factor (factor evaluated at the quadrature points); with
        rows = (lo, hi) the rows of those output nodes only, from the factor at every point or at theirs."""
        if self.rows is None:
            n = self.n_out if rows is None else rows[1] - rows[0]
            return np.zeros((n, self.N), dtype=np.result_type(factor, float))
        if rows is None:
            return self._weighted_sum(self.w * factor)
        i0, i1 = self.points(rows)
        f = factor if len(factor) == i1 - i0 else factor[i0:i1]
        return self._weighted_sum(self.w[i0:i1] * f, rows)

    def bilinear(self, f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """The integrals with both factors known: (n_out, m) for the unknown's nodal vector f (N,) and the
        known's kernel (N_known, m) (or (N_known,) -> (n_out,)); every value is the same sum as
        with_known(kernel[:, j]) @ f, associated pointwise."""
        if self.rows is None:
            return np.zeros((self.n_out,) + kernel.shape[1:])
        g = self.I @ f
        K = self.read(kernel) if self.Jf is not None else self.J @ kernel
        prod = (self.w * g)[:, None] * K if K.ndim == 2 else self.w * g * K
        return self.R @ prod

    def with_known(self, kernel: np.ndarray, extra: Optional[np.ndarray] = None, rows=None) -> np.ndarray:
        """Operator whose weight is the known kernel read along the path (times `extra` per point); with
        rows = (lo, hi) the rows of those output nodes only, (hi - lo, N), the known read at their points."""
        if self.rows is None:
            n = self.n_out if rows is None else rows[1] - rows[0]
            return np.zeros((n, self.N))
        if rows is not None:
            i0, i1 = self.points(rows)
            f = self.J[i0:i1] @ kernel
            if extra is not None:
                f = f * extra[i0:i1]
            return self.apply(f, rows)
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


