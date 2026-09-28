"""Risk-averse (CARA) agents on the spectral finite engine: the entropic objective J = theta^-1 log E e^{theta C}.

Chapter 1's appendix (thm:risk_sensitive_appendix): an agent that minimises the entropic cost of its realised cost C
has the risk-neutral first-order condition evaluated at the risk-adjusted noise-state W^theta = (I - theta Sigma_t K)^-1
(W_hat + theta Sigma_t k), Sigma_t its conditional covariance of all the shocks (past and future), K the quadratic kernel of C
over the whole horizon [0, T]^2 (C = c0 + 1/2 <W, K W>, the Hessian convention of the engine's loss) and k its
linear part.  W^theta is where the derivative is evaluated -- the average of the (linear) marginal cost over the
outcomes weighted by e^{theta C} / E e^{theta C} -- not a belief: the agent's filter and its information are
unchanged.  Without means (k = 0) the condition is, for every map node (t, s) of the agent's seen rows,

    < S f_t , Y_s > = 0,      S = (I - theta K)^-1,

with f_t the FOC kernel of a spike at t over ALL the shocks (its future part too: certainty equivalence fails) and
Y_s the rows: when P_t (I - theta K Sigma_t)^-1 f_t = 0 the transformed kernel is Sigma_t-invariant, so Sigma_t drops out and
one operator S, the same at every t, carries it.  (It is the stationarity condition of E^Q C with Q the weighting
of outcomes e^{theta C} dP / E e^{theta C}, under which the shocks have covariance S: a weighting of the
derivative's outcomes, not what the agent expects.)  The engine keeps its
risk-neutral projection and adds the correction

    Delta_t = S f_t - f_t = theta K S f_t      (a kernel on the triangle, u <= t),

projected on the rows like the FOC kernel itself.  K = A' G A with (A g)(tau) = int_0^tau zeta(tau, v) g(v) dv the
loss atoms' exposure and G = e^{-rho tau} Q dtau (plus the terminal loss at T), all in the world of the current
profile (everyone's current strategy, the agent's own included: a CARA equilibrium is one fixed point, and the
kernels, the responses and K all come from the risk-averse closed loop).  With h = Delta_t, h = theta K f + theta K h:

    * theta K f_t on the nodes exactly, by line integrals on the triangle: A f_t(tau) as two nodal kernels (AL for
      tau <= t, AU for tau >= t, through the projection and response paths), the future part of f_t as the nodal
      kernel fU(v, t) = f_t(v), v > t (the continuation path with the roles of the spike and the shock exchanged),
      then K f_t = A' G (A f_t) along the response and continuation paths;
    * the rest by Galerkin on an orthonormal Legendre basis per time panel (Sloan's iterate): with E = A Phi and
      K_G = E' G E, c_t = theta (I - theta K_G)^-1 Phi' K f_t and Delta_t = theta K f_t + theta (K Phi) c_t.

K_G's eigenvalues (Rayleigh-Ritz, from below) give the entropic cost J = E C + (2 theta)^-1 sum(-log(1 - theta
lambda) - theta lambda), the eigenvalues the basis misses through their second-order term and the exact tr K^2
(trace_K2), and the breakdown: E e^{theta C} is finite iff theta lambda_max < 1, and a solution past it raises
RiskBreakdown (an iterate past it inside a solve is answered at a smaller theta, clip).  theta = 0 skips all of this: the risk-neutral solve, bit for bit.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.polynomial import legendre

from .grid import bary_rows, bary_weights, cheb_lobatto


CLIP = 0.9          # an iterate past the breakdown responds at theta_eff = CLIP / lambda_max (Tilt, clip=True)


class RiskBreakdown(ValueError):
    """A risk-averse agent's entropic cost is infinite: theta times the largest eigenvalue of its cost kernel K is at least
    one (E e^{theta C} diverges) at the strategies of a solution, or, with `reached`, the equilibria along the
    continuation in risk aversion (solve) come to the breakdown at theta = reached, short of the model's theta.
    theta_star = 1 / lambda_max is the breakdown value of theta at the strategies it was measured at."""

    def __init__(self, agent: str, theta: float, lam_max: float, reached: Optional[float] = None):
        self.agent, self.theta, self.lam_max, self.reached = agent, float(theta), float(lam_max), reached
        self.theta_star = 1.0 / lam_max if lam_max > 0 else float("inf")
        if reached is None:
            msg = (f"risk-sensitive breakdown for {agent}: risk_aversion {theta:g} is at or beyond 1 / lambda_max = "
                   f"{self.theta_star:.4g} of its cost kernel at the solution's strategies (theta lambda_max = {theta * lam_max:.4f} "
                   ">= 1), so E exp(theta C) is infinite and the entropic cost is not defined; lower risk_aversion below the "
                   "breakdown value")
        else:
            msg = (f"risk-sensitive breakdown for {agent}: the equilibria along risk aversion reach the breakdown (E exp(theta C) "
                   f"infinite) at risk_aversion {reached:.4g}, short of the model's {theta:g}; no equilibrium with a finite "
                   "entropic cost was found there; lower risk_aversion below the breakdown value")
        super().__init__(msg)


def _bmm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b batched over the first axis, for many tiny matrices: einsum's loop for one column (2.6 times faster than
    matmul at 22000 3 x 2 blocks), matmul for several (4 times faster than einsum at 64 columns)."""
    return np.einsum("qij,qjb->qib", a, b) if b.shape[-1] == 1 else np.matmul(a, b)


def _segments(lo: float, hi: float, cuts, eps: float):
    """The pieces [a, b] of [lo, hi] cut at every value of `cuts` strictly inside, each longer than eps."""
    pts = sorted({lo, hi} | {float(x) for x in cuts if lo + eps < x < hi - eps})
    return [(a, b) for a, b in zip(pts[:-1], pts[1:]) if b - a > eps]


def _gauss(segs, m: int):
    """Gauss-Legendre points and weights on every segment."""
    xg, wg = legendre.leggauss(m)
    if not segs:
        return np.zeros(0), np.zeros(0)
    a = np.array([s[0] for s in segs]); b = np.array([s[1] for s in segs])
    h = 0.5 * (b - a)
    return (h[:, None] * xg[None, :] + (0.5 * (a + b))[:, None]).ravel(), (h[:, None] * wg[None, :]).ravel()


class _Interp:
    """An interpolation at fixed points (interp_sparse's) kept as its factors and applied through them (LinePath's
    read): the rows of a point are the outer product of its time and age rows, so the factors hold nt + na numbers
    a point where the sparse matrix holds nt na (and its indices), and the dense products are faster."""

    def __init__(self, g, t, a, side_t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        self.factors = g.interp_factors(t, a, side_t=side_t)
        self.rows = np.empty(t.size)                  # LinePath._through reads only the number of points from it
        self.shape = (t.size, g.N)

    def __matmul__(self, kernels: np.ndarray) -> np.ndarray:
        from .triangle import LinePath
        return LinePath._through(self, self.factors, kernels)


class RiskGeometry:
    """The map-independent quadrature of the correction on one compiled model (built once, cached on it): a
    piecewise-Chebyshev representation of functions of one time on the time panels (n1 nodes each, x1), the
    Legendre basis (nb per panel and channel), and the reads each step makes at fixed points."""

    def __init__(self, c, nb: int, n1: int, mq: int):
        g = c.g; self.c = c; self.g = g
        self.P = P = g.P; self.T = T = float(c.T); self.N = N = c.N; self.nW = nW = c.nW
        self.rho = float(c.rho)
        self.nb, self.n1, self.mq = nb, n1, mq
        bp = [float(b) for b in g.bp[:P + 1]]
        self.bp = bp
        eps = 1e-12 * max(1.0, T)
        self.eps = eps
        self.nV = P * nb * nW
        # the one-time representation: n1 Chebyshev-Lobatto nodes per panel, read from within the panel
        self.x1 = np.concatenate([cheb_lobatto(n1, bp[p], bp[p + 1]) for p in range(P)])
        self.side1 = np.tile(np.where(np.arange(n1) == n1 - 1, -1, 1), P)
        self.w1 = bary_weights(n1)
        # every quadrature below is cut at the panels (bp) and where the read's age crosses one (x - bp, x + bp): a kernel
        # read along a line kinks only there
        # E(tau) = int_0^tau zeta(tau, v) phi(v) dv at the one-time nodes tau = x1
        tq, vq, wq, iq, sq = [], [], [], [], []
        for i, (tau, s) in enumerate(zip(self.x1, self.side1)):
            v, w = _gauss(_segments(0.0, tau, bp + [tau - b for b in bp], eps), mq)
            tq.append(np.full(v.size, tau)); vq.append(v); wq.append(w); iq.append(np.full(v.size, i)); sq.append(np.full(v.size, s))
        tq, vq, wq, iq, sq = map(np.concatenate, (tq, vq, wq, iq, sq))
        self.IE = _Interp(g, tq, tq - vq, sq)
        self.BE = self._basis_sum(iq, vq, wq, len(self.x1))                       # (n1 P P nb, npts)
        # the time quadrature of K_G: Gauss points per panel, E read there from its one-time nodes
        tg, wg = _gauss([(bp[p], bp[p + 1]) for p in range(P)], mq)
        self.tg, self.wg = tg, wg * np.exp(-self.rho * tg)
        self.Bg = self._bary(tg, np.repeat(np.arange(P), mq))
        self.iT = len(self.x1) - 1                                                  # the node tau = T (the last panel's end)

        # Psi(u) = (K phi)(u) = int_u^T zeta(tau, u)' G E(tau) dtau (+ the terminal loss) at the one-time nodes u = x1
        tq, uq, wq, iq = [], [], [], []
        for i, u in enumerate(self.x1):
            tau, w = _gauss(_segments(u, T, bp + [u + b for b in bp], eps), mq)
            tq.append(tau); uq.append(np.full(tau.size, u)); wq.append(w * np.exp(-self.rho * tau)); iq.append(np.full(tau.size, i))
        tq, uq, wq, iq = map(np.concatenate, (tq, uq, wq, iq))
        self.IPsi = _Interp(g, tq, tq - uq, np.ones(tq.size))
        self.BPsi = self._bary(tq, np.clip(g.panel_of(tq), 0, P - 1))
        from scipy.sparse import csr_matrix
        self.SPsi = csr_matrix((wq, (iq, np.arange(iq.size))), shape=(len(self.x1), iq.size))
        self.IPsiT = g.interp_sparse(np.full(len(self.x1), T), T - self.x1, side_t=-np.ones(len(self.x1)))
        # Psi at the nodes' shock times
        self.Bnode = self._bary(g.s, np.clip(g.panel_of(g.s), 0, P - 1))

        # per time row: int_0^T Psi(v)' f_t(v) dv, the past part read from phi at (t, t - v), the future part from
        # the nodal kernel fU at (v, v - t)
        rows = sorted(c.trow_by_pit.items())
        self.node_row = np.zeros(N, dtype=int)
        tp, ap, wp, rp, sp, vp = [], [], [], [], [], []
        tf, af, wf, rf, vf = [], [], [], [], []
        for ri, ((p, it), idx) in enumerate(rows):
            self.node_row[idx] = ri
            t = float(g.t[idx[0]]); side = int(g.side_t[idx[0]])
            v, w = _gauss(_segments(0.0, t, bp + [t - b for b in bp], eps), mq)
            tp.append(np.full(v.size, t)); ap.append(t - v); wp.append(w); rp.append(np.full(v.size, ri)); sp.append(np.full(v.size, side)); vp.append(v)
            v, w = _gauss(_segments(t, T, bp + [t + b for b in bp], eps), mq)
            tf.append(v); af.append(v - t); wf.append(w); rf.append(np.full(v.size, ri)); vf.append(v)
        tp, ap, wp, rp, sp, vp = map(np.concatenate, (tp, ap, wp, rp, sp, vp))
        tf, af, wf, rf, vf = map(np.concatenate, (tf, af, wf, rf, vf))
        self.n_rows = len(rows)
        self.Ipast = _Interp(g, tp, ap, sp)
        self.Ifut = _Interp(g, tf, af, np.ones(tf.size))
        self.Bpast = self._bary(vp, np.clip(g.panel_of(vp), 0, P - 1))
        self.Bfut = self._bary(vf, np.clip(g.panel_of(vf), 0, P - 1))
        # int w Psi(v)' f(v) dv over a row's points is sum_n Psi(x1_n)' (sum_i w_i B[i, n] f(v_i)): the inner sums, one row of
        # (row, one-time node) per pair, are a sparse product with f at the points (never Psi at the points)
        self.Mpast = self._row_nodes(self.Bpast, rp, wp)
        self.Mfut = self._row_nodes(self.Bfut, rf, wf)

        # the line paths of the engine's own operators (the same keys and geometry, so they are shared)
        self.lp_proj = c._path(("projection", 0.0), r_lo=np.zeros(N), r_hi=g.s,
                               point_fn=lambda k, r_: (np.full_like(r_, g.t[k]), g.t[k] - r_),
                               known_fn=lambda k, r_: (np.full_like(r_, g.s[k]), g.s[k] - r_))
        self.lp_resp = c._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                               known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        self.lp_cont = c._path(("continuation",), r_lo=g.t, r_hi=np.full(N, c.Tg), point_fn=lambda k, r: (r, r - g.s[k]),
                               known_fn=lambda k, r: (r, r - g.t[k]))
        # the terminal reads: a kernel at (T, T - s_k) and at (T, T - t_k)
        self.IZT = g.interp_sparse(np.full(N, T), T - g.s, side_t=-np.ones(N))
        self.IRT = g.interp_sparse(np.full(N, T), T - g.t, side_t=-np.ones(N))
        self._lag_reads: Dict[float, object] = {}
        # the initial shocks of a past (n0 of them, the world's columns after the channels): each is one coordinate of the
        # shocks beside the paths, its kernels functions of time on the line s = 0 (c.diag, on the time nodes tm).  They
        # join the basis as unit vectors (exact), read on the one-time representation from the time nodes panel by panel
        self.n0 = int(c.ncol - nW)
        if self.n0:
            if c.g.L is not None or len(c.diag) != self.n_rows:
                raise NotImplementedError("risk-averse agents with initial shocks are solved without a window (a past of initial "
                                          "shocks only, on a finite horizon)")
            nt = g.nt
            Btm = np.zeros((len(self.x1), c.Nt))
            for p in range(P):
                Btm[p * n1:(p + 1) * n1, p * nt:(p + 1) * nt] = bary_rows(self.x1[p * n1:(p + 1) * n1], c.tm[p * nt:(p + 1) * nt],
                                                                          bary_weights(nt))
            self.Btm = Btm                                                          # (n1 P, Nt): time nodes -> one-time nodes
            self.wg_plain = wg                                                      # Gauss weights without the discount
            self.diag = np.asarray(c.diag)

    def _row_nodes(self, B, rows, w):
        """(n_rows P n1, npts) CSR: entry (r, n; i) = w_i B[i, n] for the points i of row r."""
        from scipy.sparse import csr_matrix
        B = B.tocoo()
        n1P = self.P * self.n1
        return csr_matrix((w[B.row] * B.data, (rows[B.row] * n1P + B.col, B.row)), shape=(self.n_rows * n1P, B.shape[0]))

    def lag_read(self, lag: float):
        """(N x N, CSR): a kernel at (s_k + lag, s_k + lag - t_k) for every node k: b(t + lag, v) at t = s_k, v = t_k
        (zero where v > t + lag or t + lag > T: outside the triangle)."""
        key = round(float(lag), 12)
        if key not in self._lag_reads:
            g = self.g
            self._lag_reads[key] = self.g.interp_sparse(g.s + lag, g.s + lag - g.t, side_t=np.ones(self.N))
        return self._lag_reads[key]

    def _bary(self, x: np.ndarray, panel: np.ndarray):
        """(len(x), P n1) CSR: the one-time representation read at x, each point from the nodes of its panel."""
        from scipy.sparse import csr_matrix
        n1 = self.n1
        rows, cols, vals = [], [], []
        for p in range(self.P):
            sel = np.where(panel == p)[0]
            if sel.size == 0:
                continue
            R = bary_rows(x[sel], self.x1[p * n1:(p + 1) * n1], self.w1)
            rows.append(np.repeat(sel, n1)); cols.append(np.tile(p * n1 + np.arange(n1), sel.size)); vals.append(R.ravel())
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(x.size, self.P * n1))

    def basis(self, v: np.ndarray) -> np.ndarray:
        """(len(v), P, nb): the orthonormal Legendre basis of every panel at v (zero off the panel)."""
        out = np.zeros((v.size, self.P, self.nb))
        pan = np.clip(self.g.panel_of(v), 0, self.P - 1)
        for p in range(self.P):
            sel = pan == p
            if not sel.any():
                continue
            h = self.bp[p + 1] - self.bp[p]
            x = 2.0 * (v[sel] - self.bp[p]) / h - 1.0
            V = legendre.legvander(x, self.nb - 1)
            out[sel, p] = V * np.sqrt((2 * np.arange(self.nb) + 1) / h)[None, :]
        return out

    def _basis_sum(self, iq, vq, wq, n_out):
        """(n_out P nb, npts) CSR: row (i, p, d) sums w phi_{p d}(v) over the points of output i."""
        from scipy.sparse import csr_matrix
        B = self.basis(vq)                                              # (npts, P, nb)
        pan = np.clip(self.g.panel_of(vq), 0, self.P - 1)
        nb, P = self.nb, self.P
        rows = ((iq[:, None] * P + pan[:, None]) * nb + np.arange(nb)[None, :]).ravel()
        cols = np.repeat(np.arange(vq.size), nb)
        vals = (wq[:, None] * B[np.arange(vq.size), pan]).ravel()
        return csr_matrix((vals, (rows, cols)), shape=(n_out * P * nb, vq.size))


def geometry(c, settings) -> RiskGeometry:
    """The compiled model's RiskGeometry (built on first use, cached with its other map-independent structures).  The
    one-time representation must resolve the exposures E(tau) = int_0^tau zeta(tau, v) phi(v) dv, whose degree in tau is
    the basis's plus the kernel's, so it carries nb + max(nt, na) + 2 nodes per panel, and the Gauss rules two more."""
    k = max(c.g.nt, c.g.na)
    nb = int(settings.risk_basis) or (k + 4)
    n1 = nb + k + 2
    mq = n1 + 2
    key = ("risk_geometry", nb, n1, mq)
    if key not in c._disc:
        c._disc[key] = RiskGeometry(c, nb, n1, mq)
    return c._disc[key]


class Tilt:
    """What one best response of a risk-averse agent freezes at the profile (everyone's current strategy): the loss
    atoms' kernels zeta in that world, K_G and its spectrum, (I - theta K_G)^-1, Psi = K Phi at the nodes and at the
    quadrature points, and the fixed reads of zeta and of the spike responses along the paths.  delta(world) is
    then the correction Delta = theta K S f, linear in the world the FOC kernel f is taken in."""

    def __init__(self, geo: RiskGeometry, foc, agent, theta: float, Zprof: np.ndarray, R: Optional[np.ndarray], chunk: int = 64,
                 spectrum_only: bool = False, clip: bool = False):
        c = geo.c; self.geo = geo; self.c = c; self.foc = foc; self.agent = agent
        self.theta = th = float(theta)
        N, nW, P = geo.N, geo.nW, geo.P
        self.chunk = chunk
        m = foc.m_flow; self.m = m; self.mt = len(foc.atoms) - m
        Q = np.asarray(foc.Q, dtype=float)
        self.Qf, self.QT = Q[:m, :m], Q[m:, m:]
        nP = len(c.prim)
        n0 = self.n0 = geo.n0
        if n0:
            zfull = foc.atoms_of(Zprof.reshape(nP, N, c.ncol))                  # (m + mt, N, ncol)
            zp = np.ascontiguousarray(zfull[:, :, :nW])
            # the atoms' kernels on the initial shocks: functions of time on the line s = 0, at the one-time nodes
            z0 = zfull[:, geo.diag, nW:]                                        # (ma, Nt, n0)
            self.Z0x = np.einsum("xt,jti->xji", geo.Btm, z0)                    # (n1 P, ma, n0)
            self.z0T = (c.terminal_point @ zfull[m:, :, nW:].transpose(1, 0, 2).reshape(N, -1)).reshape(self.mt, n0) if self.mt else None
        else:
            zp = foc.atoms_of(Zprof[:, :nW].reshape(nP, N, nW))                 # (m + mt, N, nW)
        self.zf = zf = zp[:m]; self.zT = zT = zp[m:]
        ma = m + self.mt
        # E at the one-time nodes (n1 P, m + mt, nV), the channel innermost in the basis index
        Zpts = geo.IE @ zp.transpose(1, 0, 2).reshape(N, -1)                    # (npts, ma nW)
        # the basis index a = (p, d, k): the Legendre function phi_{p d} placed on channel k, so E_{j a}(tau) = int zeta_j(tau, v)_k phi_{p d}(v) dv
        E = (geo.BE @ Zpts).reshape(len(geo.x1), P, geo.nb, ma, nW).transpose(0, 3, 1, 2, 4).reshape(len(geo.x1), ma, -1)
        self.nVb = E.shape[2]                                                   # the paths' basis functions; the initial shocks follow
        if n0:
            # the initial shocks as unit vectors of the basis: their exposure is the atoms' kernel itself, zeta^0(tau)
            E = np.concatenate([E, self.Z0x], axis=2)
        self.E = E
        Eg = (geo.Bg @ E[:, :m].reshape(len(geo.x1), -1)).reshape(geo.tg.size, m, -1)
        KG = np.einsum("q,qja,jl,qlb->ab", geo.wg, Eg, self.Qf, Eg, optimize=True)
        if self.mt:
            ET = E[geo.iT, m:]                                                  # (mt, nV): the terminal atoms at T
            if n0:
                ET = np.concatenate([ET[:, :self.nVb], self.z0T], axis=1)       # the initial shocks' columns at T by the corner read
            KG += np.exp(-geo.rho * geo.T) * (ET.T @ self.QT @ ET)
            self.ET = ET
        KG = 0.5 * (KG + KG.T)
        # the initial shocks' rows of K on the basis, [Psi_xi, K_xixi] = <e_xi, K Phi> (n0, nV): the Galerkin form's rows
        self.Cxi = KG[self.nVb:] if n0 else None
        lam, V = np.linalg.eigh(KG)
        self.lam = lam
        self.lam_max = float(lam[-1]) if lam.size else 0.0
        self.clipped = False
        if th * self.lam_max >= 1.0:
            if not clip:
                raise RiskBreakdown(agent.name, th, self.lam_max)
            # an iterate of the fixed point beyond the breakdown (the zero start is the uncontrolled world): its best
            # response is taken at theta_eff = CLIP / lambda_max, which an equilibrium never uses (the solve checks
            # the equilibrium's own spectrum at the end and raises there)
            self.theta = th = CLIP / self.lam_max
            self.clipped = True
        self.Sg = (V / (1.0 - th * lam)[None, :]) @ V.T                        # (I - theta K_G)^-1
        if spectrum_only and not n0:
            return
        # Psi = K Phi at the one-time nodes u, (n1 P, nW, nV)
        # Psi(u)_k = sum_i w_i sum_j zeta_j(tau_i, u)_k (Q E)_j(tau_i), (Q E)(tau_i) = sum_n BPsi[i, n] (Q E)(x1_n): contract the
        # points first, T[u, k, j, n] = sum_i SPsi[u, i] zeta_j(tau_i, u)_k BPsi[i, n] (small), never (Q E) at the points
        QE = np.einsum("jl,ila->ija", self.Qf, E[:, :m])                          # (n1 P, m, nV)
        Zq = (geo.IPsi @ zf.transpose(1, 0, 2).reshape(N, -1)).reshape(-1, m, nW)
        from scipy.sparse import diags
        n1P = len(geo.x1)
        Tpsi = np.empty((n1P, nW, m, n1P))
        for j in range(m):
            for k in range(nW):
                Tpsi[:, k, j, :] = (geo.SPsi @ diags(Zq[:, j, k]) @ geo.BPsi).toarray()
        Psi = np.einsum("ukjn,nja->uka", Tpsi, QE, optimize=True)
        if self.mt:
            QET = self.QT @ self.ET                                                # (mt, nV)
            ZT = (geo.IPsiT @ zT.transpose(1, 0, 2).reshape(N, -1)).reshape(-1, self.mt, nW)
            Psi = Psi + np.exp(-geo.rho * geo.T) * np.einsum("ujk,ja->uka", ZT, QET)
        self.Psi = Psi
        self.Psi_node = (geo.Bnode @ Psi.reshape(len(geo.x1), -1)).reshape(N, nW, -1)
        self.PsiT = np.ascontiguousarray(Psi.transpose(2, 0, 1).reshape(Psi.shape[2], -1))    # (nV, n1 P nW)
        if spectrum_only:                                   # with initial shocks tr K^2 needs k = K e_xi, Psi's last columns
            return
        # the fixed reads of zeta along the paths (flow atoms; the projection path reads the terminal atoms too)
        zall = zp.transpose(1, 0, 2).reshape(N, -1)
        lp, lr, lc = geo.lp_proj, geo.lp_resp, geo.lp_cont
        self.proj_J = lp.read(zall).reshape(-1, ma, nW) if lp.rows is not None else None          # zeta at (s_k, s_k - r): AL
        self.proj_I = lp.read_unknown(zall).reshape(-1, ma, nW) if lp.rows is not None else None  # zeta at (t_k, t_k - r): AU, first part
        if lr.rows is not None:
            self.resp_J = lr.read(zall).reshape(-1, ma, nW)                                       # zeta at (t_k, t_k - r): AU, second part
            resp_I = lr.read_unknown(zf.transpose(1, 0, 2).reshape(N, -1)) * (lr.w * np.exp(-geo.rho * lr.r))[:, None]
            self.resp_It = np.ascontiguousarray(resp_I.reshape(-1, m, nW).transpose(0, 2, 1))             # (nq, nW, m)
        if lc.rows is not None:
            cont_I = lc.read_unknown(zf.transpose(1, 0, 2).reshape(N, -1)) * (lc.w * np.exp(-geo.rho * lc.r))[:, None]
            self.cont_It = np.ascontiguousarray(cont_I.reshape(-1, m, nW).transpose(0, 2, 1))             # (nq, nW, m)
        if self.mt:
            self.zT_at = (geo.IZT @ zT.transpose(1, 0, 2).reshape(N, -1)).reshape(N, self.mt, nW) * np.exp(-geo.rho * geo.T)
        # the spike responses of the continuation (the envelope), per control: read at (r, r - s_k) with the weight
        # e^{-rho (r - s_k)}, and at (T, T - s_k) for the terminal atoms
        self.per_control = []
        g = geo.g
        for ui in range(len(agent.controls)):
            j0, cont, lags = foc.per_control[ui]
            item = {"lags": lags}
            if cont is not None and lc.rows is not None:
                js = cont[0]
                Rj = np.stack([foc.AO[j][1] @ R[c.block(foc.atoms[j][0]), ui] for j in js], axis=1)
                item["js"] = js
                item["Rq"] = lc.read_unknown(Rj) * (lc.w * np.exp(-geo.rho * (lc.r - g.s[lc.rows])))[:, None]
            if foc.per_terminal[ui]:
                jts = [j for (j, _) in foc.per_terminal[ui]]
                RT = np.stack([geo.IZT @ (foc.AO[j][1] @ R[c.block(foc.atoms[j][0]), ui]) for j in jts], axis=1)
                item["jts"] = jts
                item["RT"] = RT * np.exp(-geo.rho * (geo.T - g.s))[:, None]
            self.per_control.append(item)

    # ------------------------------------------------------------------ the correction
    def delta(self, ui: int, Zw: np.ndarray, a: Optional[np.ndarray] = None, phi: Optional[np.ndarray] = None) -> np.ndarray:
        """Delta (N, nW[, B]) of control ui for the world Zw (nP, N, nW[, B]) the FOC kernel is taken in.  The caller that
        has them already passes the atoms' kernels a = foc.atoms_of(Zw) and the FOC kernel phi = foc.foc(ui, a) (the
        same arrays, so the same bits), which are then not recomputed."""
        single = Zw.ndim == 3
        if single:
            Zw = Zw[..., None]
            a = None if a is None else a[..., None]
            phi = None if phi is None else phi[..., None]
        B = Zw.shape[3]
        out = np.empty((self.geo.N, self.geo.nW + self.n0, B))
        for b0 in range(0, B, self.chunk):
            sl = slice(b0, b0 + self.chunk)
            out[:, :, sl] = self._delta(ui, Zw[:, :, :, sl], None if a is None else a[..., sl], None if phi is None else phi[..., sl])
        return out[:, :, 0] if single else out

    def _delta(self, ui: int, Zw: np.ndarray, a: Optional[np.ndarray] = None, phi: Optional[np.ndarray] = None) -> np.ndarray:
        geo = self.geo; foc = self.foc; th = self.theta
        N, nW = geo.N, geo.nW
        m, mt = self.m, self.mt
        B = Zw.shape[3]
        n0 = self.n0
        if a is None:
            a = foc.atoms_of(Zw)                                                  # (ma, N, ncol, B)
        bQ = np.tensordot(foc.Q, a, axes=1)                                       # Q zeta
        if phi is None:
            phi = foc.on_qzeta(ui, bQ)                                            # (N, ncol, B): foc.foc(ui, a)
        if n0:
            # the FOC kernel on the initial shocks, f_xi(t) on the line s = 0 per time row, and the paths' part
            fxi = phi[geo.diag, nW:]                                              # (rows, n0, B)
            phi = np.ascontiguousarray(phi[:, :nW]); bQ = np.ascontiguousarray(bQ[:, :, :nW])
        fU = self._future(ui, bQ, B)                                              # (N, nW, B): f_t(v), v > t, at (v, t)
        lp, lr, lc = geo.lp_proj, geo.lp_resp, geo.lp_cont
        ma = m + mt

        def pair(path, z, f):          # sum over the path's points of w zeta_j(.)_k f(.)_k per output node: (N, ma, B)
            return (path.R @ (path.w[:, None, None] * _bmm(z, f)).reshape(-1, ma * B)).reshape(N, ma, B)
        # A f_t(tau): AL (tau <= t, at the node (t, tau)) and AU (tau >= t, at the node (tau, t)), (N, ma, B)
        AL = np.zeros((N, ma, B)); AU = np.zeros((N, ma, B))
        if lp.rows is not None:
            AL += pair(lp, self.proj_J, lp.read_unknown(phi.reshape(N, -1)).reshape(-1, nW, B))
            AU += pair(lp, self.proj_I, lp.read(phi.reshape(N, -1)).reshape(-1, nW, B))
        if lr.rows is not None:
            AU += pair(lr, self.resp_J, lr.read_unknown(fU.reshape(N, -1)).reshape(-1, nW, B))
        # K f_t at the nodes (t, u): int_u^T zeta(tau, u)' G A f_t(tau) dtau
        Kf = np.zeros((N, nW, B))
        if lr.rows is not None:
            QAL = np.einsum("jl,nlb->njb", self.Qf, AL[:, :m])
            J = lr.read(QAL.reshape(N, -1)).reshape(-1, m, B)
            Kf += (lr.R @ _bmm(self.resp_It, J).reshape(-1, nW * B)).reshape(N, nW, B)
        if lc.rows is not None:
            QAU = np.einsum("jl,nlb->njb", self.Qf, AU[:, :m])
            J = lc.read(QAU.reshape(N, -1)).reshape(-1, m, B)
            Kf += (lc.R @ _bmm(self.cont_It, J).reshape(-1, nW * B)).reshape(N, nW, B)
        if mt:
            AT = (geo.IRT @ AU[:, m:].reshape(N, -1)).reshape(N, mt, B)            # A_T f_t(T) at t = t_k
            Kf += np.einsum("njk,jl,nlb->nkb", self.zT_at, self.QT, AT, optimize=True)
        # Phi' K f_t per time row, then c_t = theta (I - theta K_G)^-1 Phi' K f_t
        fp = (geo.Ipast @ phi.reshape(N, -1)).reshape(-1, nW, B)
        ff = (geo.Ifut @ fU.reshape(N, -1)).reshape(-1, nW, B)
        # the Galerkin coefficients of K f_t: int Psi(v)' f_t(v) dv per time row (Psi = K Phi, K's own kernel on the basis)
        n1P = len(geo.x1)
        g = (geo.Mpast @ fp.reshape(fp.shape[0], -1) + geo.Mfut @ ff.reshape(ff.shape[0], -1)).reshape(geo.n_rows, n1P * nW, B)
        F = np.matmul(self.PsiT, g)                                                # (rows, nV, B)
        if n0:
            # the initial shocks' coordinates: K f_t gains k(u) f_xi(t) on the paths (k = K e_xi, the last columns of Psi) and
            # has the part <k, f_t> + K_xixi f_xi(t) on them (F's last rows once C_xi' f_xi joins it); Phi' K f_t gains
            # C_xi' f_xi(t) (the basis's initial rows being the unit vectors)
            F = F + np.einsum("ia,rib->rab", self.Cxi, fxi)
            Kf = Kf + np.einsum("nki,nib->nkb", self.Psi_node[:, :, self.nVb:], fxi[geo.node_row])
        Cc = th * np.matmul(self.Sg, F)                                          # (rows, nV, B)
        D = th * Kf + th * np.matmul(self.Psi_node, Cc[geo.node_row])
        if n0:
            Dxi = th * F[:, self.nVb:] + th * np.einsum("ia,rab->rib", self.Cxi, Cc)   # (rows, n0, B): the correction on the initial shocks
            return np.concatenate([D, Dxi[geo.node_row]], axis=1)                # carried along each time row, read on s = 0
        return D

    def _future(self, ui: int, bQ: np.ndarray, B: int) -> np.ndarray:
        """fU (N, nW, B): the FOC kernel of a spike at t = s_k on the shock at v = t_k > t (the node's time), the part
        of the continuation the risk-neutral condition drops: int_v^T e^{-rho (tau - t)} R_j(tau, t) (Q zeta)_j(tau, v)
        dtau over the atoms the spike moves, the terminal loss's at T and the agent's own lagged reads."""
        geo = self.geo; N, nW = geo.N, geo.nW
        item = self.per_control[ui]
        out = np.zeros((N, nW, B))
        lc = geo.lp_cont
        if "js" in item:
            acc = None
            for i, j in enumerate(item["js"]):
                Jb = lc.read(bQ[j].reshape(N, -1))                                 # (nq, nW B): b_j at (r, r - t_k)
                term = item["Rq"][:, i][:, None] * Jb
                acc = term if acc is None else acc + term
            out += (lc.R @ acc).reshape(N, nW, B)
        if "jts" in item:
            for i, j in enumerate(item["jts"]):
                out += item["RT"][:, i][:, None, None] * (geo.IRT @ bQ[j].reshape(N, -1)).reshape(N, nW, B)
        for (j, w, _) in item["lags"]:
            lag = float(self.foc.atoms[j][1])
            out += w * (geo.lag_read(lag) @ bQ[j].reshape(N, -1)).reshape(N, nW, B)
        # the part of the kernel on the node's own shock time (v = t) is the past's (phi): fU is the limit v -> t+
        return out

    # ------------------------------------------------------------------ the entropic cost
    def trace_K2(self) -> float:
        """tr K^2 = ||K||_HS^2 = 2 int_{u < v} |K(u, v)|_F^2, exactly (not through the basis): K(u, v) = int_v^T zeta(tau, u)' G
        zeta(tau, v) dtau (+ the terminal loss) is a kernel on the triangle (the node (v, u) = (t_k, s_k), read along the
        continuation path), squared under the triangle's Gram."""
        geo = self.geo; N, nW = geo.N, geo.nW; m = self.m
        lc = geo.lp_cont
        KN = np.zeros((N, nW, nW))
        if lc.rows is not None:
            zf = self.zf.transpose(1, 0, 2).reshape(N, -1)
            I = lc.read_unknown(zf).reshape(-1, m, nW) * (lc.w * np.exp(-geo.rho * lc.r))[:, None, None]
            J = lc.read(zf).reshape(-1, m, nW)
            KN += (lc.R @ np.einsum("qjc,jl,qld->qcd", I, self.Qf, J, optimize=True).reshape(-1, nW * nW)).reshape(N, nW, nW)
        if self.mt:
            zT = self.zT.transpose(1, 0, 2).reshape(N, -1)
            a = (geo.IZT @ zT).reshape(N, self.mt, nW); b = (geo.IRT @ zT).reshape(N, self.mt, nW)
            KN += np.exp(-geo.rho * geo.T) * np.einsum("njc,jl,nld->ncd", a, self.QT, b, optimize=True)
        M = geo.g.mass_sparse(rho=0.0)
        F = KN.reshape(N, -1)
        out = float(2.0 * np.sum(F * (M @ F)))
        if self.n0:
            # the initial shocks' blocks: 2 sum_i ||k_i||^2 (k_i = K e_xi on the paths, both off-diagonal blocks) + ||K_xixi||_F^2
            if not hasattr(self, "Psi"):
                raise RuntimeError("trace_K2 with initial shocks needs the full Tilt (not spectrum_only)")
            kg = (geo.Bg @ self.Psi[:, :, self.nVb:].reshape(len(geo.x1), -1))            # (Gauss points, nW n0)
            out += 2.0 * float(geo.wg_plain @ np.sum(kg * kg, axis=1)) + float(np.sum(self.Cxi[:, self.nVb:] ** 2))
        return out

    def entropic_excess(self) -> float:
        """J - E C = (2 theta)^-1 sum over K's eigenvalues of (-log(1 - theta lambda) - theta lambda): the Ritz values of the
        basis, and for the eigenvalues it misses (the small ones, lambda_i ~ i^-2 on each channel) their second-order term
        theta / 4 (tr K^2 - sum of the Ritz values squared), tr K^2 exact (trace_K2): the truncation error is then of third
        order in the missing eigenvalues."""
        th = self.theta
        if th == 0.0:
            return 0.0
        x = th * self.lam
        ritz = float(np.sum(-np.log1p(-x) - x) / (2.0 * th))
        tail = max(0.0, self.trace_K2() - float(np.sum(self.lam ** 2)))
        return ritz + 0.25 * th * tail
