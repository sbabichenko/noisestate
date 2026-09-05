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
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided
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

    def footprint(self) -> int:
        """Bytes held by the cached operator arrays (tensors, cores, tail slabs, flats, correlation parts)."""
        owners = {}
        arrays = [v for v in self.__dict__.values() if isinstance(v, np.ndarray)]
        arrays += [v for v in self._corr.values()] + [v for v in self._corr_flat.values()]
        arrays += [a for parts in self._corr_parts.values() for a in parts]
        for v in arrays:
            o = v if v.base is None else v.base
            owners[id(o)] = o.nbytes
        return int(sum(owners.values()))

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
        w = unit
        while bp[-1] < L - 1e-12:
            w *= growth
            nxt = min(L, bp[-1] + w)
            if L - nxt < 0.25 * w:
                nxt = L
            bp.append(nxt)
        return bp
