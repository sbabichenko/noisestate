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

import numpy as np
from numpy.polynomial import legendre


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
        self._corr_flat = {}                 # rho -> corr_tensor as [(a, j), i] (the only stored form; corr_tensor is a view of it)
        self._corr_blocks = {}               # rho -> panel blocks of corr_tensor (see _tensor_blocks)
        self._shift_cache = {}               # tau -> shift matrix

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
        panels are aligned to the lags)."""
        if abs(tau) < 1e-15:
            return np.eye(self.N)
        eps = 1e-14 * max(1.0, self.L)
        sides = self.node_sides()
        pts = self.nodes - tau
        for b in self.breakpoints:                       # a shifted node that lands within round-off of a breakpoint
            pts[np.abs(pts - b) <= 1e-12 * max(1.0, self.L)] = b     # is on it (non-dyadic units: 0.9 - 0.3)
        inner = +1 if tau > 0 else -1
        M = np.zeros((self.N, self.N))
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
        edges = sorted(set([lo, hi] + [c for c in cuts if lo + 1e-13 < c < hi - 1e-13]))
        xg, wg = legendre.leggauss(m)
        xs, ws = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            xs.append(0.5 * (e1 - e0) * xg + 0.5 * (e0 + e1))
            ws.append(0.5 * (e1 - e0) * wg)
        return np.concatenate(xs), np.concatenate(ws)

    @cached_property
    def conv_tensor(self) -> np.ndarray:
        """T[a, i, j] = int_0^a l_i(b) l_j(a - b) db  (node a)."""
        T = np.zeros((self.N, self.N, self.N))
        bp = list(self.breakpoints)
        m = self.n + 2
        for k, a in enumerate(self.nodes):
            if a <= 1e-14:
                continue
            cuts = bp + [a - b for b in bp]
            xs, ws = self._gauss_pieces(0.0, a, cuts, m)
            Li = self.interp(xs, side=+1)          # l_i(b)
            Lj = self.interp(a - xs, side=-1)      # l_j(a-b): approach from the left
            T[k] = (Li * ws[:, None]).T @ Lj
        return T

    def corr_tensor(self, rho: float = 0.0) -> np.ndarray:
        """T[a, i, j] = int_0^{L-a} e^{-rho s} l_i(s) l_j(a + s) ds  (node a).  A (non-contiguous) view of the
        stored [(a, j), i] layout that every operator uses; nothing else is kept."""
        return self._corr_flat_for(rho).reshape(self.N, self.N, self.N).transpose(0, 2, 1)

    def _corr_flat_for(self, rho: float) -> np.ndarray:
        """corr_tensor(rho) stored as [(a, j), i]: each node's N x N slab is computed as before and written
        transposed into place, so the raw [a, i, j] tensor is never materialised (one slab at a time)."""
        key = float(rho)
        if key not in self._corr_flat:
            N = self.N
            flat = np.zeros((N * N, N))
            bp = list(self.breakpoints)
            m = self.n + 2
            for k, a in enumerate(self.nodes):
                top = self.L - a
                if top <= 1e-14:
                    continue
                cuts = bp + [b - a for b in bp]
                xs, ws = self._gauss_pieces(0.0, top, cuts, m)
                Li = self.interp(xs, side=+1)          # l_i(s)
                Lj = self.interp(a + xs, side=+1)      # l_j(a+s)
                w = ws * np.exp(-rho * xs)
                flat[k * N:(k + 1) * N] = ((Li * w[:, None]).T @ Lj).T
            self._corr_flat[key] = flat
        return self._corr_flat[key]

    def conv_op(self, y: np.ndarray) -> np.ndarray:
        """Matrix C with (C g)(a) = int_0^a g(b) y(a-b) db for the fixed nodal kernel y."""
        return self.conv_ops(y[:, None])[0]

    # The tensors are block sparse by panel: a convolution at an age in panel p reads only nodes of
    # panels <= p in both factors, a correlation only older ages in the kernel and younger ones in the
    # operator's argument.  Each batched product is therefore done per panel of the output age, over the
    # bounding index ranges of that panel's nonzero entries (taken from the tensor itself, so any zero
    # pattern is handled, and the ranges only ever include zeros, never drop a nonzero): the same sums
    # with the exactly zero terms left out, so the result agrees with the full product to round-off.
    def _tensor_blocks(self, T: np.ndarray, contract: int):
        """Per output-age panel, (a0, a1, o0, o1, c0, c1, block) with the block [(a, out), contracted] of the
        entries T[a, :, :] restricted to the nonzero ranges of the output index (o) and the contracted index (c)."""
        out_axis = 3 - contract
        blocks = []
        for p in range(self.P):
            a0, a1 = p * self.n, (p + 1) * self.n
            nz = T[a0:a1] != 0
            o_any = nz.any(axis=(0, contract)); c_any = nz.any(axis=(0, out_axis))
            if not o_any.any():
                continue
            oi = np.where(o_any)[0]; ci = np.where(c_any)[0]
            o0, o1, c0, c1 = int(oi[0]), int(oi[-1]) + 1, int(ci[0]), int(ci[-1]) + 1
            blk = T[a0:a1, o0:o1, c0:c1] if out_axis == 1 else T[a0:a1, c0:c1, o0:o1].transpose(0, 2, 1)
            blocks.append((a0, a1, o0, o1, c0, c1, np.ascontiguousarray(blk).reshape((a1 - a0) * (o1 - o0), c1 - c0)))
        return blocks

    @staticmethod
    def _apply_blocks(blocks, N: int, Y: np.ndarray) -> np.ndarray:
        """(m, N, N) with out[k, a, o] = sum_c T[a, o, c] Y[c, k] from the panel blocks."""
        m = Y.shape[1]; YT = Y.T
        out = np.zeros((m, N, N))
        for a0, a1, o0, o1, c0, c1, blk in blocks:                          # the product transposed: rows of the
            out[:, a0:a1, o0:o1] = (YT[:, c0:c1] @ blk.T).reshape(m, a1 - a0, o1 - o0)     # output written contiguously
        return out

    def conv_ops(self, Y: np.ndarray) -> np.ndarray:
        """Batched conv_op: Y (N, m) -> (m, N, N), out[k, a, j] = sum_i T[a, i, j] Y[i, k]."""
        return self._apply_blocks(self._conv_blocks, self.N, Y)

    @cached_property
    def _conv_blocks(self):
        return self._tensor_blocks(self.conv_tensor, contract=2)

    def corr_ops(self, W: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Batched corr_op: W (N, m) -> (m, N, N), out[k, a, j] = sum_i T[a, i, j] W[i, k]."""
        key = float(rho)
        if key not in self._corr_blocks:
            self._corr_blocks[key] = self._tensor_blocks(self.corr_tensor(rho), contract=1)
        return self._apply_blocks(self._corr_blocks[key], self.N, W)

    def conv_op_left(self, g: np.ndarray) -> np.ndarray:
        """Matrix C with (C y)(a) = int_0^a g(b) y(a-b) db for the fixed nodal kernel g."""
        return self.conv_ops_left(g[:, None])[0]

    def conv_ops_left(self, G: np.ndarray) -> np.ndarray:
        """Batched conv_op_left: G (N, m) -> (m, N, N), out[k, a, i] = sum_j T[a, i, j] G[j, k]."""
        return self._apply_blocks(self._conv_blocks_left, self.N, G)

    @cached_property
    def _conv_blocks_left(self):
        return self._tensor_blocks(self.conv_tensor, contract=1)

    def corr_op(self, w: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Matrix C with (C z)(a) = int_0^{L-a} e^{-rho s} w(s) z(a+s) ds for fixed w."""
        return self.corr_ops(w[:, None], rho)[0]

    def corr_op_right(self, z: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """Matrix C with (C w)(a) = int_0^{L-a} e^{-rho s} w(s) z(a+s) ds for fixed z."""
        return np.einsum("aji,j->ai", self._corr_flat_for(rho).reshape(self.N, self.N, self.N), z)   # from the [(a, j), i] layout

    # ------------------------------------------------------------- helpers
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
        w = unit
        while bp[-1] < L - 1e-12:
            w *= growth
            nxt = min(L, bp[-1] + w)
            if L - nxt < 0.25 * w:
                nxt = L
            bp.append(nxt)
        return bp
