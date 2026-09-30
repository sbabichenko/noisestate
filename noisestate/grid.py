"""Piecewise-Chebyshev representation of kernels in shock age on [0, L].

A kernel f(a) is stored by its values at the Chebyshev–Lobatto nodes of a set
of panels [b_0, b_1], ..., [b_{P-1}, b_P] with b_0 = 0 and b_P = L.  Every
panel carries both its endpoints, so a function may jump at a breakpoint (the
node at the end of one panel holds the left limit, the node at the start of
the next holds the right limit).  Delays that are sums of panel widths are
therefore exact shifts.

All operators are dense matrices acting on the global nodal vector:

    interp(points)            values at arbitrary ages (zero outside [0, L])
    shift(tau)                f(a - tau) 1{a >= tau}       (lag,  tau > 0)
    shift(-tau)               f(a + tau) 1{a + tau <= L}   (lead, tau > 0)
    mass                      quadrature weights, int_0^L f(a) da = mass @ f
    propagator(A)             x(a) = e^{A a} x0 + int_0^a e^{A(a-s)} u(s) ds
    conv_tensor               T[a, i, j] = int_0^a l_i(b) l_j(a - b) db
    corr_tensor(rho)          T[a, i, j] = int_0^{L-a} e^{-rho s} l_i(s) l_j(a + s) ds

where l_i are the global nodal basis functions.  The two tensors give every
convolution and correlation of two nodal kernels as a bilinear form; the
quadrature splits each integral at the breakpoints of both factors, so it is
spectrally accurate for kernels that are smooth within panels.
"""
from __future__ import annotations

from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.polynomial import legendre


_GAUSS: dict = {}


class _Pieces:
    """A tensor T[a, i, j] (N, N, N) on the panels in block form: its nonzero n x n blocks K[g] at (node a[g], panel pu[g] of
    i, panel pv[g] of j), the groups sorted by (a, pu, pv).  A shifted convolution or correlation has about 2 P blocks per
    node (the quadrature's pieces), N P n^2 numbers against the dense N^3, and its products with a kernel batch read
    those.  on_u(Y): the operators (m, N, N) with C[a, i] = sum_j T[a, i, j] Y[j] (Y contracted on the v side); on_v(W):
    C[a, j] = sum_i W[i] T[a, i, j]."""

    def __init__(self, n: int, P: int, N: int, a, pu, pv, K):
        self.n, self.P, self.N = n, P, N
        self.a, self.pu, self.pv, self.K = a, pu, pv, np.ascontiguousarray(K)
        self._seg = {}

    def _segments(self, side: str):
        """(order, starts, rows, panels): the groups ordered by (a, output panel) and the starts of their runs."""
        hit = self._seg.get(side)
        if hit is None:
            out = self.pu if side == "u" else self.pv
            key = self.a * self.P + out
            order = np.argsort(key, kind="stable")
            ks = key[order]
            starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if ks.size else np.zeros(0, int)
            Ko = np.ascontiguousarray(self.K[order] if side == "u" else self.K[order].transpose(0, 2, 1))
            inp = (self.pv if side == "u" else self.pu)[order]
            hit = self._seg[side] = (Ko, inp, starts, self.a[order][starts], out[order][starts])
        return hit

    def _apply(self, X: np.ndarray, side: str) -> np.ndarray:
        n, P, N = self.n, self.P, self.N
        m = X.shape[1]
        out = np.zeros((N, P, n, m))
        if self.K.shape[0]:
            Ko, inp, starts, rows, panels = self._segments(side)
            Z = np.matmul(Ko, X.reshape(P, n, m)[inp])                          # every block on its input panel
            out[rows, panels] = np.add.reduceat(Z, starts, axis=0)             # summed per output block
        return np.ascontiguousarray(out.reshape(N, N, m).transpose(2, 0, 1))

    def on_u(self, Y: np.ndarray) -> np.ndarray:
        return self._apply(Y, "u")

    def on_v(self, W: np.ndarray) -> np.ndarray:
        return self._apply(W, "v")


def _leggauss(m: int):
    """legendre.leggauss(m), made once per m."""
    hit = _GAUSS.get(m)
    if hit is None:
        hit = _GAUSS[m] = legendre.leggauss(m)
    return hit


def cheb_lobatto(n: int, lo: float, hi: float) -> np.ndarray:
    k = np.arange(n)
    return lo + 0.5 * (hi - lo) * (1.0 - np.cos(np.pi * k / (n - 1)))


def bary_weights(n: int) -> np.ndarray:
    w = np.ones(n)
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def bary_rows(pts: np.ndarray, xs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Barycentric interpolation rows: R[i, j] = weight of node xs[j] in the value at pts[i]."""
    d = pts[:, None] - xs[None, :]
    exact = np.abs(d) < 1e-13 * max(1.0, float(np.abs(xs).max()))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = w[None, :] / d
        r[exact] = 0.0
        out = r / r.sum(axis=1, keepdims=True)
    rows = np.where(exact.any(axis=1))[0]
    out[rows] = 0.0
    out[rows, np.argmax(exact[rows], axis=1)] = 1.0
    return out


def clenshaw_curtis(n: int, lo: float, hi: float) -> np.ndarray:
    """Weights integrating a degree n-1 interpolant on Lobatto nodes over [lo, hi]."""
    if n == 1:
        return np.array([hi - lo])
    N = n - 1
    k = np.arange(n)
    theta = np.pi * k / N
    w = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1, N // 2 + 1):
            bj = 1.0 if 2 * j != N else 0.5
            s += bj / (4 * j * j - 1) * np.cos(2 * j * theta[i])
        ci = 1.0 if i in (0, N) else 2.0
        w[i] = ci / N * (1.0 - 2.0 * s)
    return w * 0.5 * (hi - lo)


def integration_matrix(n: int, lo: float, hi: float) -> np.ndarray:
    """S with (S f)_k = int_{lo}^{x_k} f, f given on Lobatto nodes (spectral)."""
    x = cheb_lobatto(n, lo, hi)
    V = np.polynomial.chebyshev.chebvander(2 * (x - lo) / (hi - lo) - 1.0, n - 1)
    C = np.linalg.solve(V, np.eye(n))          # nodal values -> Chebyshev coefficients
    S = np.zeros((n, n))
    for j in range(n):
        cj = np.zeros(n); cj[j] = 1.0
        ij = np.polynomial.chebyshev.chebint(cj, lbnd=-1.0)
        S[:, j] = np.polynomial.chebyshev.chebval(2 * (x - lo) / (hi - lo) - 1.0, ij)
    return 0.5 * (hi - lo) * S @ C


class AgeGrid:
    def __init__(self, breakpoints, nodes_per_panel: int = 16):
        bp = np.asarray(sorted(set(float(b) for b in breakpoints)), dtype=float)
        if bp[0] != 0.0 or len(bp) < 2:
            raise ValueError("breakpoints must start at 0 and contain L")
        self.breakpoints = bp
        self.L = float(bp[-1])
        self.P = len(bp) - 1
        self.n = int(nodes_per_panel)
        self.N = self.P * self.n
        self.nodes = np.concatenate([cheb_lobatto(self.n, bp[p], bp[p + 1]) for p in range(self.P)])
        self.panel_of_node = np.repeat(np.arange(self.P), self.n)
        self._bw = bary_weights(self.n)
        self.mass = np.concatenate([clenshaw_curtis(self.n, bp[p], bp[p + 1]) for p in range(self.P)])
        self._corr = {}                      # rho -> corr_tensor
        self._corr_flat = {}                 # rho -> corr_tensor as [(a, j), i]
        self._corr_parts = {}                # rho -> (core, tail slab, panel factors) on uniform panels
        self._shift_cache = {}               # tau -> shift matrix
        self._dmass = {}                     # rho -> discounted quadrature weights

    # ------------------------------------------------------------ basics
    def panel_index(self, a: np.ndarray, side: int = +1) -> np.ndarray:
        """Panel containing a; ties at a breakpoint go right (side=+1) or left (-1)."""
        a = np.asarray(a, dtype=float)
        if side > 0:
            p = np.searchsorted(self.breakpoints, a, side="right") - 1
        else:
            p = np.searchsorted(self.breakpoints, a, side="left") - 1
        return np.clip(p, 0, self.P - 1)

    def interp(self, points, side: int = +1) -> np.ndarray:
        """Matrix I with (I f)(points) = interpolant of f; zero outside [0, L]."""
        pts = np.atleast_1d(np.asarray(points, dtype=float))
        I = np.zeros((len(pts), self.N))
        eps = 1e-14 * max(1.0, self.L)
        inside = (pts >= -eps) & (pts <= self.L + eps)
        p = self.panel_index(np.clip(pts, 0.0, self.L), side)
        for q in range(self.P):
            sel = np.where(inside & (p == q))[0]
            if len(sel) == 0:
                continue
            xs = self.nodes[q * self.n:(q + 1) * self.n]
            I[np.ix_(sel, np.arange(q * self.n, (q + 1) * self.n))] = self._bary_rows(pts[sel], xs)
        return I

    def _bary_rows(self, pts: np.ndarray, xs: np.ndarray) -> np.ndarray:
        return bary_rows(pts, xs, self._bw)

    def diff(self) -> np.ndarray:
        """(N, N) the derivative in age, panel by panel (the barycentric differentiation matrix of each panel's
        Chebyshev-Lobatto nodes): exact for a kernel that is a polynomial of degree n - 1 on every panel.  A kernel
        that jumps at a breakpoint has its jump left out (the level rows that use this refuse lagged models)."""
        if getattr(self, "_diff", None) is None:
            D = np.zeros((self.N, self.N))
            w = self._bw
            for p in range(self.P):
                sl = slice(p * self.n, (p + 1) * self.n)
                x = self.nodes[sl]
                dx = x[:, None] - x[None, :]
                np.fill_diagonal(dx, 1.0)
                Dp = (w[None, :] / w[:, None]) / dx
                np.fill_diagonal(Dp, 0.0)
                np.fill_diagonal(Dp, -Dp.sum(axis=1))
                D[sl, sl] = Dp
            self._diff = D
        return self._diff

    def node_sides(self) -> np.ndarray:
        """+1 for the first node of a panel, -1 for the last, 0 for interior nodes."""
        k = np.arange(self.N) % self.n
        s = np.zeros(self.N, dtype=int); s[k == 0] = 1; s[k == self.n - 1] = -1
        return s

    def shift(self, tau: float) -> np.ndarray:
        """Lag (tau>0): f(a-tau) 1{a>=tau}.  Lead (tau<0): f(a+|tau|) 1{a+|tau|<=L}.
        A node at a breakpoint reads the shifted point from the side of its own panel (the last node of
        a panel reads the left limit, the first node the right limit), so a kernel that jumps at a
        breakpoint is shifted exactly: the lower copy of the node at tau reads f(0-) = 0 and the upper
        copy f(0+), and the lead is the transpose of the lag on the duplicated nodes.  Interior nodes
        read from above for a lag and from below for a lead (they never sit on a breakpoint when the
        panels are aligned to the lags).
        The rows are the shifted kernel's values at the nodes, exact point samples on any panels; as a
        kernel on the panels (its interpolant) it is exact only where the panels are aligned with tau
        (aligned(tau)), and the exact-shift operators below take the shift as an argument instead."""
        if abs(tau) < 1e-15:
            return np.eye(self.N)
        return self.shifted_reads(self.nodes, self.node_sides(), tau)

    def shifted_reads(self, nodes: np.ndarray, sides: np.ndarray, tau: float) -> np.ndarray:
        """(len(nodes), N): the values at `nodes` of a kernel read at the lag tau (a lead for tau < 0), each node
        on its side of a breakpoint (sides +1: the right limit, -1: the left limit, 0: interior); shift(tau) is
        this at the grid's own nodes.  Another grid's nodes whose panels are cut at the shifted breakpoints read
        a shifted kernel exactly (propagator_shifted)."""
        eps = 1e-14 * max(1.0, self.L)
        pts = np.asarray(nodes, dtype=float) - tau
        sides = np.asarray(sides)
        for b in self.breakpoints:                       # a shifted node that lands within round-off of a breakpoint
            pts[np.abs(pts - b) <= 1e-12 * max(1.0, self.L)] = b     # is on it (non-dyadic units: 0.9 - 0.3)
        inner = +1 if tau > 0 else -1
        M = np.zeros((len(pts), self.N))
        for s in (+1, -1):
            sel = (sides == s) | ((sides == 0) & (s == inner))
            if sel.any():
                M[sel] = self.interp(pts[sel], side=s)
        below = (pts < -eps) | ((pts <= eps) & (sides == -1))          # f(0-) = 0
        above = (pts > self.L + eps) | ((pts >= self.L - eps) & (sides == +1))   # f(L+) = 0
        M[below | above] = 0.0
        return M

    def shift_cached(self, tau: float) -> np.ndarray:
        key = round(float(tau), 12)
        if key not in self._shift_cache:
            self._shift_cache[key] = self.shift(key)
        return self._shift_cache[key]

    def evaluate(self, f: np.ndarray, points) -> np.ndarray:
        return self.interp(points) @ f

    @cached_property
    def mass_matrix(self) -> np.ndarray:
        """Exact Gram matrix M_ij = int_0^L l_i(a) l_j(a) da of the nodal basis (Gauss quadrature,
        exact for products of two interpolants; the lumped `mass` is exact only for one)."""
        M = np.zeros((self.N, self.N))
        xg, wg = legendre.leggauss(self.n + 2)
        for p in range(self.P):
            lo, hi = self.breakpoints[p], self.breakpoints[p + 1]
            xs = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); ws = 0.5 * (hi - lo) * wg
            I = self.interp(xs, side=+1)
            M += (I * ws[:, None]).T @ I
        return M

    def discounted_mass(self, rho: float) -> np.ndarray:
        """Weights w with int_0^L e^{-rho a} f(a) da = w @ f for a nodal kernel f: the Gauss rule of the
        correlation tensors on every panel (exact for an interpolant at rho = 0, where it is `mass`)."""
        key = float(rho)
        if key not in self._dmass:
            xg, wg = legendre.leggauss(self.n + 2)
            w = np.zeros(self.N)
            for p in range(self.P):
                lo, hi = self.breakpoints[p], self.breakpoints[p + 1]
                xs = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo); ws = 0.5 * (hi - lo) * wg
                w += (ws * np.exp(-key * xs)) @ self.interp(xs, side=+1)
            self._dmass[key] = w
        return self._dmass[key]

    # ------------------------------------------------------------ exact shifts
    #  A kernel read at a shift s (a lag s > 0, a lead s < 0) is f(a - s) for a in [0, L], zero where a - s leaves
    #  [0, L]: a piecewise polynomial on the panels moved by s.  shift(s) resamples it onto the grid's panels, which
    #  is exact when the panels are aligned with s (aligned(s)) and on any other panels interpolates a function that
    #  breaks at b + s inside a panel: an error no number of nodes removes, of the size of the kernel's
    #  non-smoothness at b (Chapter 5's market: the representation error 5.6e-12 on unit panels, 5e-7 with a
    #  geometric tail).  The operators here take the shift as an argument: their quadratures split at the shifted
    #  breakpoints as well, so they integrate products of shifted nodal kernels exactly on any panels.
    #
    #  A shift CLASS is (s, lo, hi): the nodal kernel x standing for x(a - s) on ages a in [lo, hi] and zero elsewhere,
    #  with lo >= max(0, s) and hi <= min(L, L + s) (the window: `natural` bounds when not narrower).  Composed shifts
    #  narrow it: a lead by d of a kernel read at the lag l reads x at a + d - l only for a <= L - d (compose).  An
    #  argument given as a float s is the class (s, natural bounds).

    def shift_class(self, s, lo: Optional[float] = None, hi: Optional[float] = None) -> Tuple[float, float, float]:
        """The normalized class (s, lo, hi) (every bound rounded to 12 digits): lo, hi default to the window's."""
        if isinstance(s, tuple):
            s, lo, hi = s
        s = round(float(s), 12)
        L = self.L
        nlo, nhi = max(0.0, s), min(L, L + s)
        lo = nlo if lo is None else max(nlo, round(float(lo), 12))
        hi = nhi if hi is None else min(nhi, round(float(hi), 12))
        return (s, lo, hi)

    def compose(self, outer: float, cls) -> Tuple[float, float, float]:
        """The class of a kernel of class `cls` read again at the shift `outer`: f(a - outer) of f = x(. - s) on [lo, hi]."""
        s, lo, hi = self.shift_class(cls)
        o = float(outer)
        return self.shift_class(s + o, max(lo + o, o, 0.0), min(hi + o, self.L + o, self.L))

    def is_natural(self, cls) -> bool:
        s, lo, hi = self.shift_class(cls)
        return (lo, hi) == (max(0.0, s), min(self.L, self.L + s))

    def aligned(self, s) -> bool:
        """Whether shift(s) is exact as a kernel on the panels: every breakpoint b with b + s inside (0, L) is a
        breakpoint (the panels move onto unions of panels).  A class (s, lo, hi) is aligned when s is and its bounds
        are breakpoints."""
        if isinstance(s, tuple):
            s, lo, hi = self.shift_class(s)
            return self.aligned(s) and all(np.min(np.abs(self.breakpoints - x)) <= 1e-12 * max(1.0, self.L) for x in (lo, hi))
        s = round(float(s), 12)
        memo = self.__dict__.setdefault("_aligned", {})
        hit = memo.get(s)
        if hit is None:
            tol = 1e-12 * max(1.0, self.L)
            bp = self.breakpoints
            moved = bp + s
            moved = moved[(moved > tol) & (moved < self.L - tol)]
            hit = memo[s] = bool(np.all(np.min(np.abs(moved[:, None] - bp[None, :]), axis=1) <= tol)) if moved.size else True
        return hit

    def sample_class(self, cls) -> np.ndarray:
        """(N, N): the values at the nodes of a kernel of class cls (shift(s) with the rows outside [lo, hi] zero, a
        node on a bound reading the inside only from its own side): exact point values on any panels."""
        s, lo, hi = self.shift_class(cls)
        S = self.shift_cached(s)
        if self.is_natural((s, lo, hi)):
            return S
        memo = self.__dict__.setdefault("_sample_class", {})
        hit = memo.get((s, lo, hi))
        if hit is None:
            tol = 1e-12 * max(1.0, self.L)
            sides = self.node_sides(); a = self.nodes
            out = (a < lo - tol) | ((a <= lo + tol) & (sides == -1)) | (a > hi + tol) | ((a >= hi - tol) & (sides == +1))
            hit = S.copy(); hit[out] = 0.0
            memo[(s, lo, hi)] = hit
        return hit

    @property
    def _zero_compact(self) -> bool:
        """Whether the shifted operators take the class 0 from the uniform panels' compact form (conv_ops, corr_ops): when
        those panels hold most of the nodes; on a geometric grid the compact form's dense tail slab is the whole tensor,
        and the block form is the cheaper one."""
        return self._compact and 2 * self.uniform_panels * self.n >= self.N

    def _local_reads(self, x: np.ndarray, side: int):
        """(panel index (Q,), local barycentric rows (Q, n)): the reads of the points x on the panels (zero outside [0, L],
        where the panel index is -1)."""
        x = np.asarray(x, dtype=float)
        eps = 1e-14 * max(1.0, self.L)
        inside = (x >= -eps) & (x <= self.L + eps)
        p = self.panel_index(np.clip(x, 0.0, self.L), side)
        R = np.zeros((len(x), self.n))
        for q in np.unique(p[inside]):
            sel = np.where(inside & (p == q))[0]
            R[sel] = self._bary_rows(x[sel], self.nodes[q * self.n:(q + 1) * self.n])
        return np.where(inside, p, -1), R

    _PIECE_OUTER_BYTES = 64 * 1024**2

    def _pieces(self, points):
        # Keep the vectorized path for small quadratures. Above 64 MiB of
        # temporary outer products, direct panel contractions are faster and
        # keep memory proportional to the point reads and final blocks.
        if sum(len(x) for x, y, w, sx, sy in points) * self.n**2 * 8 > self._PIECE_OUTER_BYTES:
            return self._pieces_blockwise(points)
        return self._pieces_outer(points)

    def _pieces_blockwise(self, points):
        """Reduce quadrature directly into panel blocks, without a Q*n*n temporary.

        Resolving a weak oscillation can require thousands of integration points;
        their outer products would dwarf the final operator. Matrix products per
        panel pair keep storage proportional to Q*n and the finished blocks.
        """
        rows, pu_all, pv_all, blocks = [], [], [], []
        for a, (x, y, weights, sx, sy) in enumerate(points):
            if not len(x):
                continue
            pu, U = self._local_reads(x, sx)
            pv, V = self._local_reads(y, sy)
            keep = (pu >= 0) & (pv >= 0)
            pu, pv, U, V, weights = pu[keep], pv[keep], U[keep], V[keep], weights[keep]
            keys = pu * self.P + pv
            for key in np.unique(keys):
                select = keys == key
                blocks.append((U[select] * weights[select, None]).T @ V[select])
                rows.append(a); pu_all.append(key // self.P); pv_all.append(key % self.P)
        return _Pieces(self.n, self.P, self.N, np.asarray(rows, int), np.asarray(pu_all, int),
                       np.asarray(pv_all, int), np.asarray(blocks).reshape(-1, self.n, self.n))

    def _pieces_outer(self, points):
        """The block form of a tensor T[a, i, j] = sum_q w_q u_i(x_q) v_j(y_q) built from its quadrature: `points` gives per
        node a (ascending) the arrays (x, y, w, side_x, side_y); every Gauss piece lies in one panel pu of the u side and one
        pv of the v side, so T is the sum over the groups (a, pu, pv) of the n x n blocks K = sum w U' V.  Returns _Pieces."""
        A, X, Y, W, SX, SY = [], [], [], [], [], []
        for a, (x, y, w, sx, sy) in enumerate(points):
            if len(x):
                A.append(np.full(len(x), a)); X.append(x); Y.append(y); W.append(w); SX.append(np.full(len(x), sx)); SY.append(np.full(len(x), sy))
        n, P = self.n, self.P
        if not A:
            return _Pieces(n, P, self.N, np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros((0, n, n)))
        A, X, Y, W = np.concatenate(A), np.concatenate(X), np.concatenate(Y), np.concatenate(W)
        SX, SY = np.concatenate(SX), np.concatenate(SY)
        pu = np.zeros(len(X), int); U = np.zeros((len(X), n)); pv = np.zeros(len(X), int); V = np.zeros((len(X), n))
        for sd in (+1, -1):
            s = SX == sd
            if s.any():
                pu[s], U[s] = self._local_reads(X[s], sd)
            s = SY == sd
            if s.any():
                pv[s], V[s] = self._local_reads(Y[s], sd)
        ok = (pu >= 0) & (pv >= 0)
        A, pu, pv, U, V, W = A[ok], pu[ok], pv[ok], U[ok], V[ok], W[ok]
        key = (A * P + pu) * P + pv
        order = np.argsort(key, kind="stable")
        key, A, pu, pv, U, V, W = key[order], A[order], pu[order], pv[order], U[order], V[order], W[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        K = np.add.reduceat((U * W[:, None])[:, :, None] * V[:, None, :], starts, axis=0)          # (G, n, n)
        return _Pieces(n, P, self.N, A[starts], pu[starts], pv[starts], K)

    def _conv_pieces(self, cls) -> "_Pieces":
        """The shifted convolution T[a, i, j] = int_0^a l_i(b) y_j(a - b) db, y_j the basis kernel of class cls, in block form
        (u = i at b, read from the right; v = j at a - s - b, from the left).  Cached per class."""
        cls = self.shift_class(cls)
        memo = self.__dict__.setdefault("_conv_piece", {})
        hit = memo.get(cls)
        if hit is None:
            s, lo, hi = cls
            bp = list(self.breakpoints); m = self.n + 2
            pts = []
            for k in range(self.N):
                a = self.nodes[k]
                b0, b1 = max(0.0, a - hi), min(a, a - lo)
                if b1 - b0 <= 1e-14:
                    pts.append((np.zeros(0), np.zeros(0), np.zeros(0), +1, -1)); continue
                xs, ws = self._gauss_pieces(b0, b1, bp + [a - s - b for b in bp], m)
                pts.append((xs, a - s - xs, ws, +1, -1))
            hit = memo[cls] = self._pieces(pts)
        return hit

    def conv_ops_shifted(self, Y: np.ndarray, cls) -> np.ndarray:
        """(m, N, N): conv_ops for the m kernels Y (N, m) of class cls: (C g)(a) = int_0^a g(b) y(a - b - s) db over a - b in
        [lo, hi]."""
        cls = self.shift_class(cls)
        if cls == (0.0, 0.0, self.L) and self._zero_compact:
            return self.conv_ops(Y)
        return self._conv_pieces(cls).on_u(np.asarray(Y, dtype=float))

    def conv_ops_left_shifted(self, G: np.ndarray, cls) -> np.ndarray:
        """(m, N, N): conv_ops_left on a kernel of class cls: (C y)(a) = int_0^a g(b) y(a - b - s) db for the m fixed g (N, m)."""
        cls = self.shift_class(cls)
        if cls == (0.0, 0.0, self.L) and self._zero_compact:
            return self.conv_ops_left(G)
        return self._conv_pieces(cls).on_v(np.asarray(G, dtype=float))

    def _corr_pieces(self, rho: float, cw, cz) -> "_Pieces":
        """The correlation T[a, i, j] = int_0^{L-a} e^{-rho x} w_i(x) z_j(a + x) dx of a class-cw kernel (u = i) with a class-cz
        one (v = j), in block form.  Cached per (rho, cw, cz)."""
        cw, cz = self.shift_class(cw), self.shift_class(cz)
        key = (float(rho), cw, cz)
        memo = self.__dict__.setdefault("_corr_piece", {})
        hit = memo.get(key)
        if hit is None:
            (sw, lw, hw), (sz, lz, hz) = cw, cz
            bp = list(self.breakpoints); m = self.n + 2; L = self.L
            pts = []
            for k in range(self.N):
                a = self.nodes[k]
                lo, hi = max(0.0, lw, lz - a), min(L - a, hw, hz - a)
                if hi - lo <= 1e-14:
                    pts.append((np.zeros(0), np.zeros(0), np.zeros(0), +1, +1)); continue
                xs, ws = self._gauss_pieces(lo, hi, [b + sw for b in bp] + [b + sz - a for b in bp], m)
                pts.append((xs - sw, a + xs - sz, ws * np.exp(-float(rho) * xs), +1, +1))
            hit = memo[key] = self._pieces(pts)
        return hit

    def corr_ops_shifted(self, W: np.ndarray, rho: float, cw, cz) -> np.ndarray:
        """(m, N, N): corr_ops for the m kernels W (N, m) of class cw acting on a kernel z of class cz:
        (C z)(a) = int_0^{L-a} e^{-rho x} w(x - sw) z(a + x - sz) dx within both classes' bounds.  Cached per (rho, cw, cz)."""
        cw, cz = self.shift_class(cw), self.shift_class(cz)
        if cw == (0.0, 0.0, self.L) and cz == (0.0, 0.0, self.L) and self._zero_compact:
            return self.corr_ops(W, rho)
        return self._corr_pieces(rho, cw, cz).on_v(np.asarray(W, dtype=float))

    def mass_shifted(self, c1, c2) -> np.ndarray:
        """(N, N): M[i, j] = int_0^L u_i(a) v_j(a) da, u_i the basis kernel of class c1 and v_j of class c2: the Gram of a
        kernel of class c1 with one of class c2 (mass_matrix at the classes 0)."""
        c1, c2 = self.shift_class(c1), self.shift_class(c2)
        if c1 == c2 == (0.0, 0.0, self.L):
            return self.mass_matrix
        memo = self.__dict__.setdefault("_mass_shift", {})
        hit = memo.get((c1, c2))
        if hit is None:
            back = memo.get((c2, c1))
            if back is not None:
                hit = back.T
            else:
                (s1, l1, h1), (s2, l2, h2) = c1, c2
                lo, hi = max(l1, l2), min(h1, h2)
                if hi - lo <= 1e-14:
                    hit = np.zeros((self.N, self.N))
                else:
                    bp = list(self.breakpoints)
                    xs, ws = self._gauss_pieces(lo, hi, [b + s1 for b in bp] + [b + s2 for b in bp], self.n + 2)
                    I1 = self.interp(xs - s1, side=+1); I2 = self.interp(xs - s2, side=+1)
                    hit = (I1 * ws[:, None]).T @ I2
            memo[(c1, c2)] = hit
        return hit

    def discounted_mass_shifted(self, rho: float, cls) -> np.ndarray:
        """(N,): w with int_0^L e^{-rho a} f(a) da = w @ x for the kernel f of class cls with the nodal values x."""
        cls = self.shift_class(cls)
        if cls == (0.0, 0.0, self.L):
            return self.discounted_mass(rho)
        memo = self.__dict__.setdefault("_dmass_shift", {})
        key = (float(rho), cls)
        hit = memo.get(key)
        if hit is None:
            s, lo, hi = cls
            hit = np.zeros(self.N)
            if hi - lo > 1e-14:
                xs, ws = self._gauss_pieces(lo, hi, [b + s for b in self.breakpoints], self.n + 2)
                hit = (ws * np.exp(-float(rho) * xs)) @ self.interp(xs - s, side=+1)
            memo[key] = hit
        return hit

    def propagator_shifted(self, A: np.ndarray, s: float) -> np.ndarray:
        """P_s (N m x N m), (node, component) ordering: the state x' = A x + u(a - s), x(0) = 0, driven by an input of
        class s, at the nodes (propagator's P at s = 0).  Solved on the panels cut at the shifted breakpoints as well
        (there the input is a polynomial on every panel, read exactly by shifted_reads), then read at this grid's nodes."""
        s = round(float(s), 12)
        if s == 0.0:
            return self.propagator(A)[1]
        A = np.atleast_2d(np.asarray(A, dtype=float)); m = A.shape[0]
        memo = self.__dict__.setdefault("_prop_shift", {})
        key = (s, A.tobytes())
        hit = memo.get(key)
        if hit is None:
            tol = 1e-12 * max(1.0, self.L)
            cuts = [b + s for b in self.breakpoints if tol < b + s < self.L - tol]
            fine = AgeGrid(list(self.breakpoints) + cuts, self.n)
            E = self.shifted_reads(fine.nodes, fine.node_sides(), s)              # (fine N, N): exact
            sides = self.node_sides(); I = np.zeros((self.N, fine.N))
            for sd in (+1, -1):
                sel = (sides == sd) | ((sides == 0) & (sd == +1))
                I[sel] = fine.interp(self.nodes[sel], side=sd)
            _, PF = fine.propagator(A)
            Im = np.eye(m)
            hit = memo[key] = np.kron(I, Im) @ PF @ np.kron(E, Im)
        return hit

    # ---------------------------------------------------------- propagator
    def propagator(self, A: np.ndarray):
        """Returns (P0, P) with x = P0 @ x0 + P @ u for the vector ODE
        dx/da = A x + u(a) on [0, L], x(0) = x0, nodal ordering (node, component).
        Jumps at breakpoints are handled by jump_injector()."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        m = A.shape[0]
        n = self.n
        P0 = np.zeros((self.N * m, m))
        P = np.zeros((self.N * m, self.N * m))
        # state at the left end of the current panel, as affine maps of (x0, u)
        E0 = np.eye(m)                     # x(b_p) = E0 @ x0 + EU @ u
        EU = np.zeros((m, self.N * m))
        for p in range(self.P):
            S = integration_matrix(n, self.breakpoints[p], self.breakpoints[p + 1])
            K = np.eye(n * m) - np.kron(S, A)
            Kinv = np.linalg.inv(K)
            sl = slice(p * n * m, (p + 1) * n * m)
            # x_panel = Kinv @ (1 (x) x(b_p) + kron(S, I) u_panel)
            ones = np.kron(np.ones((n, 1)), np.eye(m))
            P0[sl] = Kinv @ ones @ E0
            P[sl] = Kinv @ ones @ EU
            P[sl, sl] += Kinv @ np.kron(S, np.eye(m))
            # end of panel
            last = slice((p + 1) * n * m - m, (p + 1) * n * m)
            E0 = P0[last].copy()
            EU = P[last].copy()
        return P0, P

    def jump_injector(self, A: np.ndarray, tau: float) -> np.ndarray:
        """Matrix J (N*m x m): response of x to a jump v in x at age tau (x(tau+) = x(tau-) + v).
        Equals the homogeneous propagation of v started at the breakpoint tau (must be a breakpoint)."""
        A = np.atleast_2d(np.asarray(A, dtype=float))
        m = A.shape[0]
        idx = np.where(np.abs(self.breakpoints - tau) < 1e-12 * max(1.0, self.L))[0]
        if len(idx) == 0:
            raise ValueError(f"jump age {tau} is not a panel breakpoint {self.breakpoints}")
        p0 = int(idx[0])
        J = np.zeros((self.N * m, m))
        n = self.n
        E0 = np.eye(m)
        for p in range(p0, self.P):
            S = integration_matrix(n, self.breakpoints[p], self.breakpoints[p + 1])
            Kinv = np.linalg.inv(np.eye(n * m) - np.kron(S, A))
            ones = np.kron(np.ones((n, 1)), np.eye(m))
            sl = slice(p * n * m, (p + 1) * n * m)
            J[sl] = Kinv @ ones @ E0
            E0 = J[(p + 1) * n * m - m:(p + 1) * n * m].copy()
        return J

    # ------------------------------------------------------------ tensors
    def _gauss_pieces(self, lo: float, hi: float, cuts, m: int):
        """Gauss–Legendre points/weights on [lo, hi] split at the cuts inside it."""
        edges = np.array(sorted(set([lo, hi] + [c for c in cuts if lo + 1e-13 < c < hi - 1e-13])))
        xg, wg = _leggauss(m)
        e0, e1 = edges[:-1, None], edges[1:, None]
        return (0.5 * (e1 - e0) * xg + 0.5 * (e0 + e1)).ravel(), (0.5 * (e1 - e0) * wg).ravel()

    def _conv_rows(self, ks) -> np.ndarray:
        """conv_tensor rows of the nodes ks: T[a, i, j] = int_0^a l_i(b) l_j(a - b) db, (len(ks), N, N)."""
        T = np.zeros((len(ks), self.N, self.N))
        bp = list(self.breakpoints)
        m = self.n + 2
        for t, k in enumerate(ks):
            a = self.nodes[k]
            if a <= 1e-14:
                continue
            cuts = bp + [a - b for b in bp]
            xs, ws = self._gauss_pieces(0.0, a, cuts, m)
            Li = self.interp(xs, side=+1)          # l_i(b)
            Lj = self.interp(a - xs, side=-1)      # l_j(a-b): approach from the left
            T[t] = (Li * ws[:, None]).T @ Lj
        return T

    def _corr_rows(self, rho: float, ks, jlo: int = 0) -> np.ndarray:
        """corr_tensor rows of the nodes ks for the columns j >= jlo: T[a, i, j] = int e^{-rho s} l_i(s) l_j(a + s) ds,
        (len(ks), N, N - jlo).  The integral runs over the s at which a + s is in the panel of node jlo or beyond
        (below it every l_j with j >= jlo vanishes); the pieces are those of the full integral, so the entries agree
        with the full one to round-off."""
        T = np.zeros((len(ks), self.N, self.N - jlo))
        if jlo >= self.N:
            return T
        bp = list(self.breakpoints)
        m = self.n + 2
        start = self.breakpoints[self.panel_of_node[jlo]] if jlo else 0.0
        for t, k in enumerate(ks):
            a = self.nodes[k]
            top = self.L - a; lo = max(0.0, start - a)
            if top - lo <= 1e-14:
                continue
            cuts = bp + [b - a for b in bp]
            xs, ws = self._gauss_pieces(lo, top, cuts, m)
            Li = self.interp(xs, side=+1)          # l_i(s)
            Lj = self.interp(a + xs, side=+1)      # l_j(a+s)
            w = ws * np.exp(-rho * xs)
            T[t] = (Li * w[:, None]).T @ Lj[:, jlo:]
        return T

    @cached_property
    def conv_tensor(self) -> np.ndarray:
        """T[a, i, j] = int_0^a l_i(b) l_j(a - b) db  (node a).  Dense; the operators below do not need it
        on a grid with uniform leading panels (see uniform_panels)."""
        return self._conv_rows(range(self.N))

    def corr_tensor(self, rho: float = 0.0) -> np.ndarray:
        """T[a, i, j] = int_0^{L-a} e^{-rho s} l_i(s) l_j(a + s) ds  (node a).  Dense, as conv_tensor."""
        key = float(rho)
        if key not in self._corr:
            self._corr[key] = self._corr_rows(key, range(self.N))
        return self._corr[key]

    # ------------------------------------------- uniform panels: core and tail
    @cached_property
    def uniform_panels(self) -> int:
        """Number of leading panels of equal width w.  On them both tensors are block-Toeplitz in the panel
        index: for node a in panel p with local index alpha, i in panel q (local beta), j in panel r (gamma),

            conv_tensor[a, i, j] = K[p - q - r, alpha, beta, gamma],                      p - q - r in {0, 1},
            corr_tensor[a, i, j] = exp(-rho q w) K_rho[r - p - q, alpha, beta, gamma],    r - p - q in {0, 1},

        whenever a and j are in uniform panels (i is then uniform as well: b <= a for the convolution, and
        the correlation reads i at the lag between a and the older j), and zero at every other panel triple.
        The cores K, K_rho (2, n, n, n) are integrals over the two pieces of one panel (_unit_core); what
        involves the tail (the rows of tail nodes a for the convolution, the columns of tail nodes j for the
        correlation) is kept dense.  Below two uniform panels the dense tensors are used throughout."""
        widths = np.diff(self.breakpoints)
        off = np.where(np.abs(widths - widths[0]) > 1e-12 * widths[0])[0]
        return int(off[0]) if len(off) else self.P

    @property
    def _compact(self) -> bool:
        return self.uniform_panels >= 2

    def _unit_core(self, rho: Optional[float]) -> np.ndarray:
        """The core on one panel [0, w] with local nodes xi.  rho None (convolution):
            K[0, al] = int_0^{xi_al} l_beta(s) l_gamma(xi_al - s) ds,   K[1, al] = int_{xi_al}^w l_beta(s) l_gamma(w + xi_al - s) ds;
        rho a float (correlation):
            K[0, al] = int_0^{w - xi_al} e^{-rho s} l_beta(s) l_gamma(xi_al + s) ds,   K[1, al] = int_{w - xi_al}^w e^{-rho s} l_beta(s) l_gamma(xi_al + s - w) ds.
        The Gauss rule and the pieces are those of the dense quadrature on that panel, so the entries agree to round-off."""
        n = self.n; w = float(self.breakpoints[1]); xi = cheb_lobatto(n, 0.0, w)
        xg, wg = legendre.leggauss(n + 2)
        K = np.zeros((2, n, n, n))
        for al in range(n):
            cut = xi[al] if rho is None else w - xi[al]
            if cut <= 1e-13:
                pieces = [(0.0, w, 1)]
            elif cut >= w - 1e-13:
                pieces = [(0.0, w, 0)]
            else:
                pieces = [(0.0, cut, 0), (cut, w, 1)]
            for lo, hi, d in pieces:
                xs = 0.5 * (hi - lo) * xg + 0.5 * (lo + hi); ws = 0.5 * (hi - lo) * wg
                Li = bary_rows(xs, xi, self._bw)
                if rho is None:
                    Lj = bary_rows(xi[al] + d * w - xs, xi, self._bw)
                else:
                    Lj = bary_rows(xi[al] + xs - d * w, xi, self._bw); ws = ws * np.exp(-rho * xs)
                K[d, al] = (Li * ws[:, None]).T @ Lj
        return K

    @cached_property
    def _conv_core(self) -> np.ndarray:
        return self._unit_core(None)

    @cached_property
    def _conv_core_flat(self) -> np.ndarray:
        return self._conv_core.reshape(2, self.n * self.n, self.n)                                        # [d, (alpha, beta), gamma]

    @cached_property
    def _conv_core_flat_left(self) -> np.ndarray:
        return np.ascontiguousarray(self._conv_core.transpose(0, 1, 3, 2)).reshape(2, self.n * self.n, self.n)   # [d, (alpha, gamma), beta]

    @cached_property
    def _conv_tail_flat(self) -> np.ndarray:
        Nu = self.uniform_panels * self.n
        return self._conv_rows(range(Nu, self.N)).reshape((self.N - Nu) * self.N, self.N)             # [(a, i), j], tail a

    @cached_property
    def _conv_tail_flat_left(self) -> np.ndarray:
        Nu = self.uniform_panels * self.n
        T = self._conv_tail_flat.reshape(self.N - Nu, self.N, self.N)
        return np.ascontiguousarray(T.transpose(0, 2, 1)).reshape((self.N - Nu) * self.N, self.N)      # [(a, j), i], tail a

    def _corr_compact(self, rho: float):
        """(core [d, (alpha, gamma), beta], slab [(a, j), i] over the tail j, panel factors exp(-rho w q))."""
        key = float(rho)
        if key not in self._corr_parts:
            n, Nu = self.n, self.uniform_panels * self.n
            KL = np.ascontiguousarray(self._unit_core(key).transpose(0, 1, 3, 2)).reshape(2, n * n, n)
            slab = self._corr_rows(key, range(self.N), jlo=Nu)                                          # (a, i, tail j)
            flat = np.ascontiguousarray(slab.transpose(0, 2, 1)).reshape(self.N * (self.N - Nu), self.N)
            fac = np.exp(-key * float(self.breakpoints[1]) * np.arange(self.uniform_panels))
            self._corr_parts[key] = (KL, flat, fac)
        return self._corr_parts[key]

    def _panel_blocks(self, KK: np.ndarray, X: np.ndarray, fac: Optional[np.ndarray] = None) -> np.ndarray:
        """M (m, Pu, n, n) with M[Delta] = KK[0] X[Delta] + KK[1] X[Delta - 1]: the core contracted with the
        uniform panels of the kernels X (Pu n, m), each panel scaled by fac when given."""
        n, Pu = self.n, self.uniform_panels; m = X.shape[1]
        Xp = X.reshape(Pu, n, m)
        if fac is not None:
            Xp = Xp * fac[:, None, None]
        A = (KK.reshape(2 * n * n, n) @ Xp.transpose(1, 0, 2).reshape(n, Pu * m)).reshape(2, n, n, Pu, m)
        M = A[0].copy(); M[:, :, 1:] += A[1][:, :, :-1]
        return np.ascontiguousarray(M.transpose(3, 2, 0, 1))

    def _toeplitz_fill(self, C: np.ndarray, M: np.ndarray, lower: bool) -> None:
        """Write M[Delta] into every panel block (p, q) of C (m, N, N) with p - q = Delta (lower) or q - p = Delta."""
        m, N, _ = C.shape; n, Pu = self.n, self.uniform_panels
        s0, s1, s2 = C.strides; flat = C.reshape(-1)
        for D in range(Pu):
            off = D * n * (N if lower else 1)
            V = as_strided(flat[off:], shape=(m, Pu - D, n, n), strides=(s0, n * (s1 + s2), s1, s2))
            V[...] = M[:, D, None]

    def _conv_assemble(self, KK: np.ndarray, tail: np.ndarray, Y: np.ndarray) -> np.ndarray:
        N, Nu = self.N, self.uniform_panels * self.n; m = Y.shape[1]
        C = np.zeros((m, N, N))
        self._toeplitz_fill(C, self._panel_blocks(KK, Y[:Nu]), lower=True)
        if N > Nu:
            C[:, Nu:, :] = (tail @ Y).reshape(N - Nu, N, m).transpose(2, 0, 1)
        return C

    # ----------------------------------------------------------- operators
    def conv_op(self, y: np.ndarray) -> np.ndarray:
        """Matrix C with (C g)(a) = int_0^a g(b) y(a-b) db for the fixed nodal kernel y."""
        return self.conv_ops(y[:, None])[0]

    def conv_ops(self, Y: np.ndarray) -> np.ndarray:
        """Batched conv_op: Y (N, m) -> (m, N, N).  Uniform panels: block lower-triangular Toeplitz from the
        core and the tail rows from the stored slab; otherwise one BLAS product with the dense tensor."""
        Y = np.asarray(Y, dtype=float); N = self.N
        if not self._compact:
            return (self._conv_flat @ Y).reshape(N, N, -1).transpose(2, 0, 1)
        return self._conv_assemble(self._conv_core_flat, self._conv_tail_flat, Y)

    @cached_property
    def _conv_flat(self) -> np.ndarray:
        return self.conv_tensor.reshape(self.N * self.N, self.N)              # [(a, i), j]

    def corr_ops(self, W: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Batched corr_op: W (N, m) -> (m, N, N).  Uniform panels: block upper-triangular Toeplitz from the
        core (the panels of W scaled by exp(-rho w q)) and the tail columns from the stored slab."""
        W = np.asarray(W, dtype=float); N = self.N
        key = float(rho)
        if not self._compact:
            if key not in self._corr_flat:
                T = self.corr_tensor(rho)
                self._corr_flat[key] = np.ascontiguousarray(T.transpose(0, 2, 1)).reshape(N * N, N)   # [(a, j), i]
            return (self._corr_flat[key] @ W).reshape(N, N, -1).transpose(2, 0, 1)
        KL, slab, fac = self._corr_compact(key)
        Nu = self.uniform_panels * self.n; m = W.shape[1]
        C = np.zeros((m, N, N))
        self._toeplitz_fill(C, self._panel_blocks(KL, W[:Nu], fac), lower=False)
        if N > Nu:
            C[:, :, Nu:] = (slab @ W).reshape(N, N - Nu, m).transpose(2, 0, 1)
        return C

    def conv_op_left(self, g: np.ndarray) -> np.ndarray:
        """Matrix C with (C y)(a) = int_0^a g(b) y(a-b) db for the fixed nodal kernel g."""
        return self.conv_ops_left(g[:, None])[0]

    def conv_ops_left(self, G: np.ndarray) -> np.ndarray:
        """Batched conv_op_left: G (N, m) -> (m, N, N), as conv_ops with the roles of i and j exchanged."""
        G = np.asarray(G, dtype=float); N = self.N
        if not self._compact:
            return (self._conv_flat_left @ G).reshape(N, N, -1).transpose(2, 0, 1)
        return self._conv_assemble(self._conv_core_flat_left, self._conv_tail_flat_left, G)

    @cached_property
    def _conv_flat_left(self) -> np.ndarray:
        return np.ascontiguousarray(self.conv_tensor.transpose(0, 2, 1)).reshape(self.N * self.N, self.N)   # [(a, j), i]

    def corr_op(self, w: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Matrix C with (C z)(a) = int_0^{L-a} e^{-rho s} w(s) z(a+s) ds for fixed w."""
        return self.corr_ops(w[:, None], rho)[0]

    def corr_op_right(self, z: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Matrix C with (C w)(a) = int_0^{L-a} e^{-rho s} w(s) z(a+s) ds for fixed z."""
        return np.einsum("aij,j->ai", self.corr_tensor(rho), z)

    @staticmethod
    def breakpoints_from_delays(L: float, delays, unit: float | None = None,
                                unit_range: float | None = None, growth: float = 2.0):
        """Uniform panels of width `unit` (default: the smallest delay) up to
        `unit_range` (default: 8 units), then geometrically growing panels to L."""
        delays = [float(d) for d in delays if d and d > 0]
        if unit is None:
            unit = min(delays) if delays else L
        if unit_range is None:
            unit_range = min(L, 8 * unit) if delays else L
        bp = list(np.arange(0.0, unit_range + 1e-12, unit))
        # the window's edge less each lag is a cut as well: a lagged read at an age above L - d reads past the window,
        # where the kernels are cut off, so they jump there (1e-5 of the peak on the Chapter 5 market at L = 24), and a
        # jump inside a geometric panel is an error no number of nodes removes (the representation error's floor of
        # about 1.2e-6 at 12, 14 and 16 nodes, at ages near L); within unit_range it is a multiple of the unit already
        stops = sorted({round(L - d, 12) for d in delays if bp[-1] + 1e-12 < L - d < L - 1e-12} | {L})
        w = unit
        for stop in stops:
            while bp[-1] < stop - 1e-12:
                w *= growth
                nxt = min(stop, bp[-1] + w)
                if stop - nxt < 0.25 * w:
                    nxt = stop
                bp.append(nxt)
        return bp
