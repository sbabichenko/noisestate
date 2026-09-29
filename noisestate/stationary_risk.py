"""Risk-averse (CARA) agents on the stationary engine, under consistent planning.

Each date's self minimises the entropic cost of its own discounted continuation, J_t = theta^-1 log E[exp(theta C_t) |
F_t], C_t = int_t^inf e^{-rho (s - t)} c_s ds (+ the stochastic-integral terms int_t^inf e^{-rho (s - t)} x_s' L dW_s), the
later selves playing the stationary strategy.  The tilt argument of risk.py at one date (Ch1 appendix,
thm:risk_sensitive_appendix, with K the kernel of the date's own continuation) gives, at date 0 and by stationarity at
every date,

    P_0 S f_0 = 0,    S = (I - theta K_0)^-1  on the shocks born in (-L, inf),

K_0 the kernel of C_0 (C_0 = c + 1/2 <W, K_0 W>): K_0(u, v) = int_{max(u, v, 0)} e^{-rho tau} z(tau - u)' Q z(tau - v) dtau
over the loss atoms' lag kernels z (on [0, L]), plus the integrals' e^{-rho v} z_x(v - u)' L for u < v, v > 0.  f_0 is the
FOC kernel of a spike at date 0 over all those shocks, the past ones (ages 0 .. L, the engine's phi) and the future ones
(0 < u < L: the spike's responses against Q z, and e^{-rho u} L' R_x(u) from the integrals).  Unlike the risk-neutral
condition, f_0 is taken with the agent's OWN later reactions on: the later selves minimise their own J_t, not J_0, so a
date-0 deviation's effect through them is not zero to first order (no envelope); at theta = 0 the two conditions agree
at an equilibrium, and theta = 0 never comes here.  W^theta = S f_0 is where the derivative is taken (a weighting of
outcomes by e^{theta C_0}), not a belief: the filter and the information are the risk-neutral ones.

The engine keeps its risk-neutral best response and adds, frozen at the current profile (everyone's current strategy,
the agent's own included), the shift

    Delta = (S f_0^on - f_0^off) on the past  =  (phi^on - phi^off) + theta K_0 Sigma g,   g = (I - theta K_0 Sigma)^-1 f_0^on,

phi^on / phi^off the engine's FOC kernel with the own reactions on / off (the envelope responses it uses), Sigma the
projector off what the agent has seen (its passive rows on the lattice's past): g = S f_0 wherever P_0 g = 0, and the
Sigma form is well conditioned whenever the conditional entropic cost is finite, which (I - theta K_0) is not (K_0's
seen-shock block can be large).  At a fixed
point the profile is the agent's own best response, so the equilibrium solves P_0 S f_0^on = 0 exactly; off it the
shift is an explicit (Picard) term, which the engine's Anderson mixing iterates away.

Nystrom on a uniform lattice of step h (u = -L + i h, tau = k h, both lattices through the age L so the kernels' cut
at ages 0 and L falls on lattice points, taken at half their one-sided value: the trapezoid rule, second order): A g
(the atoms' exposure) and A' by FFT convolutions, (I - theta K_0 Sigma) g = f_0 by GMRES, and theta K_0 Sigma g read at the
engine's age nodes with the tau integral cut exactly at the window (the last partial interval by the trapezoid on the
exact kernel and the lattice interpolant of A g).  Two lattices (h, h / 2) and Richardson's h^2 step give the shift.
The own reactions' responses are rebuilt by a fixed point (_responses_on): the closed loop carries a control's spike to
the other agents' rows only while its owner is switched off.  Checked (tests/test_cara_ext.py) against direct
quadrature of C_0 and of its spike derivative on Chapter 4's wealth model, the envelope identity P (phi^on - phi^off) = 0
at a risk-neutral equilibrium, and extras/stationary_cara_reference.py.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _staircase_q(A: np.ndarray, reach: np.ndarray, bs: Optional[int] = None) -> np.ndarray:
    """The orthonormal factor Q (n, k = min(n, q)) of the Householder QR of A (n, q) whose column j is zero below row
    reach[j] (nondecreasing): a blocked QR whose reflectors and updates touch only the rows a panel's columns reach
    (LAPACK dgeqrf on the panel, dormqr on the trailing columns up to k, then Q by applying the panels backwards to the
    identity), a fraction of the dense QR's work on a staircase (the information basis's time-ordered rows and columns:
    about 1/10 on Kyle-Back's insider).  The same basis as numpy.linalg.qr's in exact arithmetic (up to the columns'
    signs, which the projector Q Q' does not see); in floating point another one of the same span."""
    from scipy.linalg.lapack import dgeqrf, dormqr
    n, q = A.shape
    k = min(n, q)
    if bs is None:                                  # panels of 32 columns on a staircase of slope about 1 (as many rows as
        bs = 32 if reach[k - 1] <= 1.25 * k else 64     # columns per step), 64 on a steeper one (measured: dormqr's speed)
    A = np.asfortranarray(A[:, :k])
    end = np.minimum(n, np.maximum.accumulate(np.maximum(np.asarray(reach[:k]), np.arange(1, k + 1))))
    panels = []
    for j0 in range(0, k, bs):
        j1 = min(k, j0 + bs); r1 = int(end[j1 - 1])
        qr_, tau, _, info = dgeqrf(A[j0:r1, j0:j1])
        if info != 0:
            raise ValueError(f"the information basis's QR failed (dgeqrf info {info})")
        if j1 < k:
            cq, _, info = dormqr("L", "T", qr_, tau, A[j0:r1, j1:], lwork=max(1, (k - j1) * 64))
            if info != 0:
                raise ValueError(f"the information basis's QR failed (dormqr info {info})")
            A[j0:r1, j1:] = cq
        panels.append((j0, r1, qr_, tau))
    Q = np.zeros((n, k), order="F")
    Q[np.arange(k), np.arange(k)] = 1.0
    for j0, r1, qr_, tau in reversed(panels):
        cq, _, info = dormqr("L", "N", qr_, tau, Q[j0:r1, j0:], lwork=max(1, (k - j0) * 64))
        if info != 0:
            raise ValueError(f"the information basis's QR failed (dormqr info {info})")
        Q[j0:r1, j0:] = cq
    return Q


def _upper_toeplitz(sym: np.ndarray) -> np.ndarray:
    """The dense block upper triangular Toeplitz matrix (Na a, Na b) whose block (i, m) is sym[m - i] (a x b) for m >= i:
    sym (Na, a, b), one strided view of the zero-padded symbol and one copy."""
    from numpy.lib.stride_tricks import sliding_window_view
    Na, a, b = sym.shape
    E = np.concatenate([np.zeros((Na - 1, a, b)), sym])                 # E[s] = sym[s - (Na - 1)]
    W = sliding_window_view(E, Na, axis=0)                              # W[s, :, :, t] = E[s + t]: block (i, t) at s = Na - 1 - i
    return np.ascontiguousarray(W[::-1].transpose(0, 1, 3, 2)).reshape(Na * a, Na * b)


class StationaryTilt:
    """What one best response of a risk-averse agent freezes at the profile: the atoms' lag kernels, the spike responses
    with and without the own reactions, and the lattice operators; shift() is Delta (nU, N, nW)."""

    def __init__(self, solver, agent, maps: Dict[str, np.ndarray], theta: float, h: Optional[float] = None,
                 horizon: Optional[float] = None):
        c = solver.c; g = c.grid
        self.solver, self.c, self.agent, self.theta = solver, c, agent, float(theta)
        self.N, self.nW, self.L, self.rho = c.N, c.nW, float(g.L), float(c.rho)
        if self.rho <= 0:
            raise NotImplementedError("a risk-averse agent on the stationary engine needs a discount (consistent planning of the "
                                      "discounted continuation)")
        c.use_maps(maps)
        self._maps = maps
        self._qpast: Dict[float, np.ndarray] = {}
        nW = self.nW
        nU = len(agent.controls)
        self.Z = Z = c.closed_loop(maps)[:, :nW]
        self.Ron = Ron = self._responses_on(solver, maps)
        atoms, Q, q = c.loss[agent.name]
        self.atoms, self.Q = list(atoms), np.asarray(Q, dtype=float)
        self.z = np.stack([c.shift(lag) @ Z[c.block(nm)] for nm, lag in atoms])                    # (m, N, nW)
        self.Rj = np.stack([np.stack([c.shift(lag) @ Ron[c.block(nm), ui] for nm, lag in atoms], axis=0) for ui in range(nU)])   # (nU, m, N)
        integ = (getattr(c, "integrals", None) or {}).get(agent.name)
        self.mx = 0
        if integ:
            xatoms, L = integ
            self.Lx = np.asarray(L, dtype=float); self.mx = len(xatoms)
            self.zx = np.stack([c.shift(lag) @ Z[c.block(nm)] for nm, lag in xatoms])
            self.Rx = np.stack([np.stack([c.shift(lag) @ Ron[c.block(nm), ui] for nm, lag in xatoms]) for ui in range(nU)])
        # the lattice: h dividing L; the future reach where e^{-rho tau} is negligible
        h = h or getattr(solver, "RISK_LATTICE", 0.02)
        n_age = int(np.ceil(self.L / min(h, self.L / 50)))
        self.hs = [self.L / n_age, self.L / (2 * n_age)]
        self.U = float(horizon) if horizon else self.L + 20.0 / self.rho               # e^{-20} of the discounted weight beyond
        # the FOC kernels (past part) with the own reactions on and off
        R0 = solver._spikes(c, maps, agent)[1]
        Roff = solver._impulse_responses(agent, maps, R0)
        Fon = solver._foc_operators(agent, Ron, envelope=False)
        Foff = solver._foc_operators(agent, Roff)
        self.phi_on = np.stack([Fon[ui] @ Z for ui in range(nU)])                                   # (nU, N, nW)
        self.phi_off = np.stack([Foff[ui] @ Z for ui in range(nU)])
        self.gmres = []

    def _responses_on(self, solver, maps) -> np.ndarray:
        """(n_prim N, nU): the world's response to a unit spike of each of the agent's controls with its own later reactions
        ON.  The closed loop carries a control's impulse to the other agents' rows only when its owner is switched off
        (closed_loop's impulse columns), so the responses start from those (R0: the others reacting) and add the agent's own
        reaction kernels c, the fixed point c = G Y (R0 + Resp c): Y the agent's passive rows' reads of the world
        (row_blocks with its own controls excluded: its own orders are known to it), G its maps (the grid's convolution by
        each row's map) and Resp the world's response to its action kernels (the others reacting)."""
        c, agent = self.c, self.agent
        N, nP = c.N, len(c.prim)
        nU = len(agent.controls)
        R0 = solver._spikes(c, maps, agent)[1]                                     # (nP N, nU), own reactions off
        Resp = solver._response_operators(agent, R0)                               # per control (nP N, N), own block I
        excl = frozenset(agent.controls)
        A = []                                                                     # per row: (N, nP N) the regular read
        for r in range(len(agent.signals)):
            blocks, _ = c.row_blocks(agent.name, r, excl)
            op = np.zeros((N, nP * N))
            for nm, B in blocks.items():
                op[:, c.block(nm)] += B
            A.append(op)
        g = maps[agent.name]                                                       # (nU, nR, N)
        M = np.zeros((nU * N, nP * N))
        for vi in range(nU):
            Cs = c.grid.conv_ops_left(g[vi].T)                                     # per row (N, N): row kernel -> action kernel
            for r in range(len(A)):
                M[vi * N:(vi + 1) * N] += Cs[r] @ A[r]
        RespAll = np.concatenate(Resp, axis=1)                                     # (nP N, nU N)
        lhs = np.eye(nU * N) - M @ RespAll
        out = np.empty_like(R0)
        for ui in range(nU):
            cvec = np.linalg.solve(lhs, M @ R0[:, ui])
            out[:, ui] = R0[:, ui] + RespAll @ cvec
        return out

    # ------------------------------------------------------------------ one lattice
    def _lattice(self, h: float):
        Na = int(round(self.L / h))
        ages = h * np.arange(Na + 1)
        I = self._geo(("I", round(h, 12)), lambda: self._lattice_interp(ages))
        half = np.ones(Na + 1); half[0] = half[-1] = 0.5             # the cut at ages 0 and L: half the one-sided value
        zt = np.einsum("an,jnc->ajc", I, self.z) * half[:, None, None]                                  # (Na + 1, m, nW)
        Rt = np.einsum("an,ujn->uaj", I, self.Rj)                                                        # (nU, Na + 1, m)
        out = {"h": h, "Na": Na, "ages": ages, "I": I, "half": half, "zt": zt, "Rt": Rt}
        if self.mx:
            out["zxt"] = np.einsum("an,jnc->ajc", I, self.zx) * half[:, None, None]
            out["Rxt"] = np.einsum("an,ujn->uaj", I, self.Rx)
        Nf = int(np.ceil(self.U / h))
        out["Nu"] = Na + Nf + 1                                  # u_i = -L + i h, i = 0 .. Na + Nf
        out["Nt"] = Nt = Nf + 1                                       # tau_k = k h, k = 0 .. Nf
        tau = h * np.arange(Nt)
        wt = np.ones(Nt); wt[0] = 0.5
        out["dtau"] = h * wt * np.exp(-self.rho * tau)                # the tau quadrature with the discount
        out["tau"] = tau
        return out

    def _geo(self, key, make):
        """What depends on the grid, the window and the lattice only (not on the profile), kept across the best responses'
        tilts (one StationaryTilt per best response) on the compiled model, whose grid, window and rate it is: the
        lattice's interpolation rows, _read's per-age quadrature.  Read only."""
        memo = self.c.__dict__.setdefault("_tilt_geo", {})
        if key not in memo:
            memo[key] = make()
        return memo[key]

    def _lattice_interp(self, ages):
        g = self.c.grid
        I = g.interp(ages)
        I[-1] = g.interp([self.L], side=-1)[0]                        # the age L from the left
        return I

    def _spectra(self, lat, key: str, n_in: int, rev: bool):
        """(f, {(j, c): rfft of the lag kernel lat[key][:, j, c], reversed for A', at length f}) over the nonzero kernels, made
        once per lattice: f = next_fast_len(n_in + Na), the length scipy.signal.fftconvolve takes for a convolution of an
        input of length n_in with them, so the convolutions below are fftconvolve's to the bit (its transforms one line at
        a time, the kernel's made once instead of per call, the input's once for every kernel it meets)."""
        memo = lat.setdefault("spectra", {})
        if (key, rev, n_in) not in memo:
            from scipy import fft as sf
            zt = lat[key]
            f = sf.next_fast_len(n_in + lat["Na"], True)
            memo[key, rev, n_in] = (f, {(j, cc): sf.rfft(zt[::-1, j, cc] if rev else zt[:, j, cc], f)
                                        for j in range(zt.shape[1]) for cc in range(self.nW) if np.any(zt[:, j, cc])})
        return memo[key, rev, n_in]

    def _A(self, lat, zt, G, key: str = "zt"):
        """(A g)_j(tau_k) = int z_j(tau - u)' g(u) du on the lattice, (m, Nt): G (nW, Nu); zt = lat[key]."""
        from scipy import fft as sf
        h, Na, Nt = lat["h"], lat["Na"], lat["Nt"]
        f, S = self._spectra(lat, key, G.shape[1], False)
        out = np.zeros((zt.shape[1], Nt))
        SG = {}
        for (j, cc), Sz in S.items():                                                    # j major, c minor: the sums' order
            if cc not in SG:
                SG[cc] = sf.rfft(G[cc], f)
            out[j] += sf.irfft(SG[cc] * Sz, f)[Na:Na + Nt]
        return h * out

    def _At(self, lat, zt, B, key: str = "zt"):
        """(A' b)_c(u_i) = sum_k z(tau_k - u_i)' b(tau_k) (the weights already in b), (nW, Nu): B (m, Nt); zt = lat[key]."""
        from scipy import fft as sf
        Nu = lat["Nu"]
        f, S = self._spectra(lat, key, B.shape[1], True)
        out = np.zeros((self.nW, Nu))
        SB = {}
        for (j, cc), Sz in S.items():
            if j not in SB:
                SB[j] = sf.rfft(B[j], f) if np.any(B[j]) else None
            if SB[j] is not None:
                out[cc] += sf.irfft(SB[j] * Sz, f)[:Nu]
        return out

    def _K(self, lat, G):
        """K_0 g on the lattice, (nW, Nu)."""
        Na, Nt = lat["Na"], lat["Nt"]
        Ag = self._A(lat, lat["zt"], G)
        out = self._At(lat, lat["zt"], lat["dtau"][None, :] * (self.Q @ Ag))
        if self.mx:
            Gf = G[:, Na:Na + Nt]                                                        # g at u = tau >= 0
            out += self._At(lat, lat["zxt"], lat["dtau"][None, :] * (self.Lx @ Gf), "zxt")
            Axg = self._A(lat, lat["zxt"], G, "zxt")                                            # (mx, Nt)
            ind = np.ones(Nt); ind[0] = 0.5                                              # 1_{u >= 0}, half at u = 0
            out[:, Na:Na + Nt] += (ind * np.exp(-self.rho * lat["tau"]))[None, :] * (self.Lx.T @ Axg)
        return out

    def _f(self, lat, ui: int):
        """f_0 on the lattice (nW, Nu): the past part phi^on at the ages L - i h, the future part at u = i' h."""
        h, Na, Nu = lat["h"], lat["Na"], lat["Nu"]
        F = np.zeros((self.nW, Nu))
        past = lat["I"] @ self.phi_on[ui]                                                # (Na + 1, nW) at ages 0 .. L
        F[:, :Na + 1] = past[::-1].T                                                     # u = -L .. 0
        # future part: f(u) = sum_j int_u^L e^{-rho tau} R_j(tau) (Q z)_j(tau - u) dtau, u = i' h in (0, L]: the trapezoid on
        # the lattice points tau_k, k = i' .. Na (z read at its own value, the ends at half weight)
        Rt = lat["Rt"][ui]                                                               # (Na + 1, m)
        Qz = np.einsum("ji,aic->ajc", self.Q, np.einsum("an,jnc->ajc", lat["I"], self.z))  # (Na + 1, m, nW), unhalved
        disc = np.exp(-self.rho * lat["ages"])
        fut = np.zeros((Na + 1, self.nW))
        m = Rt.shape[1]
        if Na * m * self.nW <= 2500:
            # small lattices: fut[ip] = sum over k = ip .. Na and j of (w_ip(k) Rt[k, j]) Qz[k - ip, j, :] (w_ip = h disc with
            # its ends halved) for a block of ip at once, the terms laid out (t = k - ip, j, ip, c) with zeros past Na and
            # reduced over their leading axis, which numpy sums in order: the einsum below, one ip at a time, to the bit
            # (its sum from zero: + 0.0).  Above the size the per-point einsum is faster (the terms' memory traffic)
            t = np.arange(Na + 1)
            step = max(1, (2 << 20) // max(1, (Na + 1) * m * self.nW))              # <= 16 MB of terms
            for i0 in range(0, Na, step):
                ips = np.arange(i0, min(Na, i0 + step))
                kk = ips[None, :] + t[:, None]                                           # (t, ip): k = ip + t
                kc = np.minimum(kk, Na)
                w = np.where(kk <= Na, h * disc[kc], 0.0)
                w[0] *= 0.5
                w[Na - ips, np.arange(ips.size)] *= 0.5                                  # k = Na
                terms = (w[:, None, :] * Rt[kc].transpose(0, 2, 1))[:, :, :, None] * Qz[:, :, None, :]   # (t, j, ip, c)
                fut[ips] = np.add.reduce(terms.reshape(-1, ips.size, self.nW), axis=0) + 0.0
        else:
            for ip in range(0, Na):
                k = np.arange(ip, Na + 1)
                w = h * disc[k]; w[0] *= 0.5; w[-1] *= 0.5
                fut[ip] = np.einsum("k,kj,kjc->c", w, Rt[k], Qz[k - ip])
        if self.mx:
            fut += np.exp(-self.rho * lat["ages"])[:, None] * (lat["Rxt"][ui] @ self.Lx)            # u = 0 .. L (0+ at u = 0)
            fut[Na] *= 0.5                                                               # u = L: the cut, half its value
        F[:, Na + 1:2 * Na + 1] = fut[1:].T
        F[:, Na] = 0.5 * (past[0] + fut[0])                                              # u = 0: the average of the two sides
        return F

    def _solve(self, lat, F, key=None):
        """g = (I - theta K_0 Sigma)^-1 f_0: the tilt of what the agent has not seen (Sigma), which is S f_0 wherever the
        condition holds (P_0 g = 0) and, unlike (I - theta K_0)^-1, well conditioned whenever the conditional entropic cost is
        finite (K_0's past-past block, the seen shocks', can be large: a Kyle insider's inventory)."""
        from scipy.sparse.linalg import LinearOperator, gmres
        nW, Nu, th = self.nW, lat["Nu"], self.theta
        op = LinearOperator((nW * Nu, nW * Nu), matvec=lambda x: x - th * self._K(lat, self._sigma(lat, x.reshape(nW, Nu))).ravel(),
                            dtype=float)
        b = F.ravel()
        # started from the last solution of the same (agent, control, lattice): the fixed point's iterates and the stability
        # matvecs' perturbations move it little, and GMRES's tolerance is relative to b, not to the start's residual
        memo = self.solver.__dict__.setdefault("_tilt_x0", {}) if key is not None else {}
        x0 = memo.get(key)
        x, info = gmres(op, b, x0=x0 if x0 is not None and x0.shape == b.shape else None, rtol=1e-12, atol=0.0, restart=200, maxiter=50)
        res = float(np.linalg.norm(op @ x - b) / max(np.linalg.norm(b), 1e-300))
        self.gmres.append((info, res))
        if res > 1e-8:
            raise ValueError(f"the stationary risk correction's solve (I - theta K_0 Sigma) g = f_0 did not converge (residual {res:.1e})")
        if key is not None:
            memo[key] = x.copy()
        return x.reshape(nW, Nu)

    def _read(self, lat, G, ages=None):
        """theta (K_0 g)(u = -a) at the engine's age nodes a: sum over tau in [0, L - a] of e^{-rho tau} z(tau + a)' (Q A g)(tau)
        (+ the integrals' z_x(tau + a)' L g(tau)), the tau integral cut exactly at L - a: the trapezoid on the lattice up to
        the last point below it and the partial interval by the trapezoid with the lattice interpolant of A g."""
        c = self.c; gr = c.grid; h, Na = lat["h"], lat["Na"]
        b = self.Q @ self._A(lat, lat["zt"], G)                                          # (m, Nt)
        bx = self.Lx @ G[:, Na:] if self.mx else None
        ages = gr.nodes if ages is None else np.asarray(ages, dtype=float)
        out = np.zeros((ages.size, self.nW))
        rows, IL = self._geo(("read", round(h, 12), ages.tobytes()), lambda: self._read_rows(h, ages))
        zL = np.einsum("n,jnc->jc", IL, self.z)
        zxL = np.einsum("n,jnc->jc", IL, self.zx) if self.mx else None
        for n, (K, wd, frac, Iz, s, d_last, d_up) in enumerate(rows):
            zk = np.einsum("kn,jnc->kjc", Iz, self.z)
            val = np.einsum("k,kjc,jk->c", wd, zk, b[:, :K + 1])
            if frac > 1e-14:
                bu = (1 - s) * b[:, K] + s * b[:, K + 1]
                val += 0.5 * frac * (d_last * np.einsum("jc,j->c", zk[-1], b[:, K])
                                     + d_up * np.einsum("jc,j->c", zL, bu))
            if self.mx:
                zxk = np.einsum("kn,jnc->kjc", Iz, self.zx)
                val += np.einsum("k,kjc,jk->c", wd, zxk, bx[:, :K + 1])
                if frac > 1e-14:
                    bxu = (1 - s) * bx[:, K] + s * bx[:, K + 1]
                    val += 0.5 * frac * (d_last * np.einsum("jc,j->c", zxk[-1], bx[:, K])
                                         + d_up * np.einsum("jc,j->c", zxL, bxu))
            out[n] = self.theta * val
        return out

    def _read_rows(self, h: float, ages):
        """_read's quadrature at each age a (profile-free): (K, the weights with the discount, the partial interval frac,
        the interpolation rows at tau_k + a, frac / h, e^{-rho tau_K}, e^{-rho (L - a)}), and the row reading age L."""
        gr = self.c.grid
        IL = gr.interp([self.L], side=-1)[0]
        rows = []
        for a in ages:
            up = self.L - a
            K = int(np.floor(up / h + 1e-12))
            tk = h * np.arange(K + 1)
            w = h * np.ones(K + 1); w[0] = 0.5 * h; w[-1] = 0.5 * h
            if K == 0:
                w[:] = 0.0
            frac = up - K * h
            Iz = gr.interp(tk + a)
            Iz[tk + a >= self.L - 1e-12] = IL
            rows.append((K, w * np.exp(-self.rho * tk), frac, Iz, frac / h, np.exp(-self.rho * tk[-1]), np.exp(-self.rho * up)))
        return rows, IL

    def risk_part(self, ui: int, h: float) -> np.ndarray:
        """theta K_0 S f_0^on on the past, at the age nodes (N, nW), on the lattice of step h."""
        lat = self._lattice(h)
        G = self._solve(lat, self._f(lat, ui), key=(self.agent.name, ui, round(h, 12), self.theta))
        return self._read(lat, self._sigma(lat, G))                                      # theta K_0 Sigma g = g - f_0

    def shift(self) -> np.ndarray:
        """Delta (nU, N, nW) = (phi^on - phi^off) + theta K_0 S f_0^on (Richardson over the two lattices)."""
        out = []
        for ui in range(len(self.agent.controls)):
            d1, d2 = (self.risk_part(ui, h) for h in self.hs)
            out.append(self.phi_on[ui] - self.phi_off[ui] + (4.0 * d2 - d1) / 3.0)
            self.richardson_gap = float(np.abs(d2 - d1).max())
        return np.stack(out)

    # ------------------------------------------------------------------ the entropic cost of the date-0 continuation
    def _info_basis(self, lat, full: bool = True) -> np.ndarray:
        """An orthonormal basis (columns) of what the agent has seen by date 0, in the lattice's coordinates x = sqrt(h) g: each
        past step s = -L .. -h of each row r, h y_r(s - u) (its drift part, the passive row kernel) plus its point loadings
        (the noise it sees at s).  The rows are the passive ones (its own orders known and taken out)."""
        solver, c = self.solver, self.c
        h, Na, Nu, nW = lat["h"], lat["Na"], lat["Nu"], self.nW
        key = round(h, 12)
        if key not in self._qpast:
            Zpass = solver._spikes(c, self._maps, self.agent)[0]
            ytil, yinst = solver._passive_rows(self.agent, Zpass)
            # the step s = u_m = -L + m h of row r: h y_r at the ages (m - i) h of the u_i <= s, plus its point loadings at u_m
            nR = len(ytil)
            n, q = nW * (Na + 1), nR * Na
            hyls = []
            for r, y in enumerate(ytil):
                if any(age != 0.0 for (_, age, _) in yinst[r]):
                    raise NotImplementedError("a risk-averse agent on the stationary engine with a delayed point loading on its rows")
                hyls.append(h * (lat["I"] @ y[:, :nW]))                                  # (Na + 1, nW) at the ages
            if 2 * q >= n:
                # at least half as many columns as rows (rows at least half the channels): with the rows in time order
                # (i nW + c) and the columns by step (m nR + r) the matrix is a staircase, column m reaching the rows of the
                # steps i <= m only (the reversed row kernel, one slice per column, built F-ordered for the QR), and the
                # staircase QR does a fraction of the dense one's work (1/10 on Kyle-Back's insider); on a tall matrix (a row
                # seeing many channels) the dense QR is as fast, and is kept (the panels' overhead)
                A = np.zeros((n, q), order="F")
                for r, hyl in enumerate(hyls):
                    Hf = hyl[::-1].ravel()                                               # the ages L .. 0, channel minor
                    for m in range(Na):
                        A[:(m + 1) * nW, m * nR + r] = Hf[(Na - m) * nW:]
                    for (k, age, w) in yinst[r]:
                        A[np.arange(Na) * nW + k, np.arange(Na) * nR + r] += w
                reach = nW * (np.arange(q) // nR + 1)                                    # column m nR + r: rows of the steps <= m
                Qt = _staircase_q(A, reach)                                              # (n, min(n, q)), time-major rows
                Qp = Qt.reshape(Na + 1, nW, -1).transpose(1, 0, 2).reshape(n, -1)
            else:                                                                        # rows (c, i), columns r Na + m
                lag = np.arange(Na)[None, :] - np.arange(Na + 1)[:, None]              # (i, m): m - i
                seen = lag >= 0
                cols = []
                for r, hyl in enumerate(hyls):
                    V = np.where(seen[None], hyl.T[:, np.where(seen, lag, 0)], 0.0)      # (nW, Na + 1, Na)
                    for (k, age, w) in yinst[r]:
                        V[k, np.arange(Na), np.arange(Na)] += w
                    cols.append(V.reshape(n, Na))
                Qp = np.linalg.qr(np.concatenate(cols, axis=1))[0]
            self._qpast[key] = np.ascontiguousarray(Qp)                                  # (nW (Na + 1), q) on the past block
        Qp = self._qpast[key]
        if not full:
            return Qp
        Q = np.zeros((nW, Nu, Qp.shape[1]))
        Q[:, :Na + 1] = Qp.reshape(nW, Na + 1, -1)
        return Q.reshape(nW * Nu, -1)

    GRAPH_BOUND = 1e6                   # _graph: the pivot block's condition and the symbol's l1 norm past which the QR is used

    def _graph(self, lat):
        """The information's span as a graph over pivot coordinates, or None (then _info_basis's QR serves).

        The basis matrix A of _info_basis (rows (i, c): step i, channel c; columns (m, r): step m, row r) is block upper
        triangular Toeplitz: A[(i, c), (m, r)] = a[m - i][c, r], a[0] carrying the rows' point loadings, and its last step's
        rows (i = Na, the age 0) are zero.  Taking nR pivot channels P (where a[0] is best conditioned, a column-pivoted QR
        of a[0]') and the others F, A = [A_P; A_F] (rows reordered) with A_P square block triangular Toeplitz, invertible
        with a[0][P], so

            span(A) = span([I; X]),    X = A_F A_P^-1,

        X again block upper triangular Toeplitz, its symbol x = a_F * a_P^-1 (the causal convolution with the series
        inverse: a Volterra resolvent, computed by one unit-triangular solve).  The projector off the span is then

            Sigma g = [-X' s; s],  (I + X X') s = g_F - X g_P          (the complement form, when |F| <= |P|), or
            Sigma g = g - [v; X v],  (I + X' X) v = g_P + X' g_F        (the primal form),

        a Cholesky of the smaller Gram, whose entries are lagged sums of the symbol (Toeplitz: each is a partial sum
        over one lag, O(Na^2) in all instead of a GEMM).  The same projector as the QR's (1e-15 on the stationary CARA
        models) at a fraction of its cost: the QR of the staircase was most of a risk-averse best response.  A pivot
        block that is ill conditioned, or a growing resolvent (a row whose noise loading is small next to its drift),
        leaves the graph steep; past GRAPH_BOUND the QR is used."""
        from scipy.linalg import qr, solve_triangular, cholesky
        from numpy.lib.stride_tricks import sliding_window_view
        key = round(lat["h"], 12)
        memo = self.__dict__.setdefault("_gpast", {})
        if key in memo:
            return memo[key]
        memo[key] = None
        h, Na, nW = lat["h"], lat["Na"], self.nW
        solver, c = self.solver, self.c
        Zpass = solver._spikes(c, self._maps, self.agent)[0]
        ytil, yinst = solver._passive_rows(self.agent, Zpass)
        nR = len(ytil)
        if nR == 0 or nR >= nW or Na < 1:
            return None
        a = np.zeros((Na, nW, nR))                                      # a[d][c, r]
        for r, y in enumerate(ytil):
            if any(age != 0.0 for (_, age, _) in yinst[r]):
                raise NotImplementedError("a risk-averse agent on the stationary engine with a delayed point loading on its rows")
            a[:, :, r] = h * (lat["I"][:Na] @ y[:, :nW])
            for (k, age, w) in yinst[r]:
                a[0, k, r] += w
        _, _, piv = qr(a[0].T, pivoting=True, mode="economic")
        P = np.sort(piv[:nR]); F = np.setdiff1d(np.arange(nW), P)
        nF = F.size
        a0 = a[0][P]
        if not np.all(np.isfinite(a)) or np.linalg.cond(a0) > self.GRAPH_BOUND:
            return None
        a0i = np.linalg.inv(a0)
        # the series inverse b of a_P (b[0] = a0^-1): the last block column of A_P^-1, from the unit upper triangular system
        # blockdiag(a0^-1) A_P z = blockdiag(a0^-1) e_last (column m of step m: rows i <= m)
        cP = np.einsum("pq,dqr->dpr", a0i, a[:, P, :])                  # a0^-1 a_P[d], cP[0] = I
        if Na > 1:
            rhs = np.zeros((Na * nR, nR)); rhs[-nR:] = a0i
            z = solve_triangular(_upper_toeplitz(cP), rhs, lower=False, unit_diagonal=True, check_finite=False)
            b = z.reshape(Na, nR, nR)[::-1]                              # b[d] = block (Na - 1 - d) of the last column
        else:
            b = a0i[None]
        # x = a_F * b (causal, the first Na lags): one direct convolution per entry, summed over the pivot rows
        x = np.zeros((Na, nF, nR))
        aF = a[:, F, :]
        for ci in range(nF):
            for r in range(nR):
                acc = np.zeros(Na)
                for p in range(nR):
                    acc += np.convolve(aF[:, ci, p], b[:, p, r])[:Na]
                x[:, ci, r] = acc
        if not np.all(np.isfinite(x)) or float(np.abs(x).sum(axis=0).max()) > self.GRAPH_BOUND:
            return None
        # X dense (F coordinates (i, c) x P coordinates (m, p), time major): X[(i, c), (m, p)] = x[m - i][c, p], m >= i
        X = _upper_toeplitz(x)
        comp = nF <= nR
        # the Gram's lagged sums: complement (I + X X')[(i, c), (j, c')] (i <= j) = sum_{t = 0}^{Na-1-j} x[t + j - i] x[t]';
        # primal (I + X' X)[(m, p), (m', p')] (m <= m') = sum_{s = 0}^{m} x[s]' x[s + m' - m]
        xp = np.concatenate([x, np.zeros_like(x)])                          # x[k] = 0 for k >= Na
        lagged = sliding_window_view(xp, Na, axis=0)[:Na]                   # lagged[d, :, :, t] = x[t + d]
        if comp:
            prod = np.einsum("dcrt,tkr->dtck", lagged, x)                   # x[t + d] x[t]'
        else:
            prod = np.einsum("tcp,dcqt->dtpq", x, lagged)                   # x[t]' x[t + d]
        S = np.cumsum(prod, axis=1)                                         # S[d, T] = the partial sum to T
        del prod
        k = nF if comp else nR
        # the lower triangle only (the Cholesky reads no other): block (i + d, i) is block (i, i + d)', S[d, Na - 1 - d - i]
        # (complement) or S[d, i] (primal), written one block diagonal at a time through the flat (Na Na, k, k) view
        G5 = np.zeros((Na, Na, k, k))
        flat = G5.reshape(Na * Na, k, k)
        for d in range(Na):
            vals = S[d, Na - 1 - d::-1] if comp else S[d, :Na - d]
            flat[d * Na::Na + 1][:Na - d] = vals.transpose(0, 2, 1)
        del S
        Gm = G5.transpose(0, 2, 1, 3).reshape(Na * k, Na * k)
        del G5, flat
        Gm[np.diag_indices_from(Gm)] += 1.0
        try:
            Lc = cholesky(Gm, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return None
        out = {"P": P, "F": F, "X": X, "L": Lc, "comp": comp, "Na": Na}
        memo[key] = out
        return out

    def _sigma(self, lat, G):
        """Sigma g: the part of g (nW, Nu) the agent has not seen (its information's span taken out on the past block)."""
        Na = lat["Na"]
        gr = self._graph(lat)
        if gr is not None:
            from scipy.linalg import solve_triangular
            P, F, X, Lc = gr["P"], gr["F"], gr["X"], gr["L"]
            out = G.copy()
            gP = G[P, :Na].T.ravel(); gF = G[F, :Na].T.ravel()             # time major (i, c); the step Na is not touched
            if gr["comp"]:
                s = solve_triangular(Lc, gF - X @ gP, lower=True, check_finite=False)
                s = solve_triangular(Lc, s, lower=True, trans="T", check_finite=False)
                oP, oF = -(X.T @ s), s
            else:
                v = solve_triangular(Lc, gP + X.T @ gF, lower=True, check_finite=False)
                v = solve_triangular(Lc, v, lower=True, trans="T", check_finite=False)
                oP, oF = gP - v, gF - X @ v
            out[P, :Na] = oP.reshape(Na, P.size).T
            out[F, :Na] = oF.reshape(Na, F.size).T
            return out
        Qp = self._info_basis(lat, full=False)
        out = G.copy()
        gp = G[:, :Na + 1].ravel()
        out[:, :Na + 1] -= (Qp @ (Qp.T @ gp)).reshape(self.nW, Na + 1)
        return out

    def _K_lags(self, lat) -> np.ndarray:
        """K_0 on the lattice (the operator _K applies) by lags: Kl[i, d] (Nu, Na + 1, nW, nW) the block K[i, i - d] between
        u_i and u_{i - d} (zero past |i - i'| > Na: the atoms' lag kernels live on [0, L]),

            K[i, i - d] = h sum_a dtau(i - Na + a) zt[a]' Q zt[a + d]     (+ the integrals' dtau(i - Na) Lx' zxt[d], i >= Na)

        (a the age of the later exposure; the sum over a for all rows i at once is one product with the Toeplitz matrix of
        the discount weights), the same sums as _K's convolutions without their FFT round-off, Q symmetrised as the dense
        form's 0.5 (K + K') did; the diagonal blocks (d = 0) symmetric."""
        from numpy.lib.stride_tricks import sliding_window_view
        nW, h, Na, Nu = self.nW, lat["h"], lat["Na"], lat["Nu"]
        zt = lat["zt"]                                                                   # (Na + 1, m, nW)
        ztQ = np.einsum("ajc,jk->akc", zt, 0.5 * (self.Q + self.Q.T))
        P = np.zeros((Na + 1, Na + 1, nW, nW))                                           # P[a, d] = zt[a]' Q zt[a + d]
        for d in range(Na + 1):
            P[:Na + 1 - d, d] = np.einsum("akc,ake->ace", ztQ[:Na + 1 - d], zt[d:])
        ext = np.concatenate([np.zeros(Na), lat["dtau"], np.zeros(Na)])
        T = np.ascontiguousarray(sliding_window_view(ext, Na + 1)[:Nu])                 # T[i, a] = dtau(i - Na + a)
        Kl = (h * (T @ P.reshape(Na + 1, -1))).reshape(Nu, Na + 1, nW, nW)
        if self.mx:
            X = np.einsum("jc,dje->dce", self.Lx, lat["zxt"])                            # (Na + 1, nW, nW): Lx' zxt[d]
            wk = lat["dtau"][:Nu - Na]                                                   # rows i = Na + k
            Kl[Na:] += wk[:, None, None, None] * X[None]
            Kl[Na:, 0] += wk[:, None, None] * X[0].T[None]                               # the diagonal block: X[0] + X[0]'
        return Kl

    @staticmethod
    def _lag_block(Kl, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """The dense block of K (time-major rows i nW + c) between the steps [r0, r1) and [c0, c1) with r0 >= c0 (on or
        below the diagonal): Kl[i, i - i'] where 0 <= i - i' <= Na, its transpose above the diagonal."""
        nW, Na = Kl.shape[2], Kl.shape[1] - 1
        i = np.arange(r0, r1)[:, None]; ip = np.arange(c0, c1)[None, :]
        d = i - ip
        low = (d >= 0) & (d <= Na)
        out = np.zeros((r1 - r0, c1 - c0, nW, nW))
        ii, jj = np.nonzero(low)
        out[ii, jj] = Kl[i[ii, 0], d[ii, jj]]
        up = (d < 0) & (d >= -Na)
        if up.any():
            ii, jj = np.nonzero(up)
            out[ii, jj] = Kl[ip[0, jj], -d[ii, jj]].transpose(0, 2, 1)
        return out.transpose(0, 2, 1, 3).reshape((r1 - r0) * nW, (c1 - c0) * nW)

    def cond_excess(self, h: float):
        """E[J_0 | F_0] - E C_0 averaged over the past, on the lattice of step h: with Pi the projector on the agent's
        information and Sigma = I - Pi, (theta / 2) tr(Pi K B K Pi) + (2 theta)^-1 sum (-log(1 - theta mu) - theta mu), mu the
        eigenvalues of Sigma K Sigma and B = Sigma (I - theta Sigma K Sigma)^-1 Sigma (the tilt of the unseen shocks given the
        seen ones).  Returns (excess, the largest theta mu: the breakdown at 1).

        In time-major coordinates (i nW + c) K is block tridiagonal in blocks of Na + 1 steps (_K_lags: it couples steps at
        most Na apart), and Pi lives on the past block, the first of them, so Sigma K Sigma differs from K only on the first
        two blocks (the steps K couples to the past) and is block tridiagonal too.  With its block Cholesky factor
        C C' = I - theta Sigma K Sigma: the sum over mu is -log det - theta tr, read row by row without cancellation
        (C_ii^2 = 1 - y_i, y_i = theta A_ii + s_i, s_i = sum_k<i C_ik^2: the sum is sum_i (-log(1 - y_i) - y_i) + s_i, every
        part >= 0); the first term is (theta / 2) |C^-1 Sigma K Q|^2 (Q the basis of Pi: tr(Q'K Sigma (I - theta Sigma K
        Sigma)^-1 Sigma K Q)), a block forward substitution; the largest mu by Lanczos on the blocks.  The same quantities
        as the eigen-decomposition of the dense n x n form, O(n (nW Na)^2) instead of O(n^3), no n x n matrix."""
        from scipy.linalg import solve_triangular
        from scipy.linalg.lapack import dpotrf
        from scipy.sparse.linalg import LinearOperator, eigsh
        lat = self._lattice(h)
        nW, Na, Nu = self.nW, lat["Na"], lat["Nu"]
        n = nW * Nu
        Kl = self._K_lags(lat)
        s0 = list(range(0, Nu, Na + 1)) + [Nu]                                           # the blocks' first steps
        nb = len(s0) - 1
        D = [self._lag_block(Kl, s0[k], s0[k + 1], s0[k], s0[k + 1]) for k in range(nb)]
        D = [0.5 * (Dk + Dk.T) for Dk in D]                                              # symmetric to the bit
        E = [None] + [self._lag_block(Kl, s0[k], s0[k + 1], s0[k - 1], s0[k]) for k in range(1, nb)]   # E[k]: rows k, cols k - 1
        del Kl
        # Sigma K Sigma on the first two blocks (the past is block 0)
        Qp = self._info_basis(lat, full=False)
        q, bp = Qp.shape[1], nW * (Na + 1)
        Qt = Qp.reshape(nW, Na + 1, q).transpose(1, 0, 2).reshape(bp, q)               # time-major rows
        two = nb > 1
        K2 = np.block([[D[0], E[1].T], [E[1], D[1]]]) if two else D[0].copy()
        KQ = K2[:, :bp] @ Qt                                                             # K Q (zero past block 1)
        QKQ = Qt.T @ KQ[:bp]
        K2[:bp] -= Qt @ KQ.T
        K2[:, :bp] -= KQ @ Qt.T
        K2[:bp, :bp] += Qt @ (QKQ @ Qt.T)
        K2 = 0.5 * (K2 + K2.T)
        D[0] = K2[:bp, :bp].copy()
        if two:
            E[1] = K2[bp:, :bp].copy(); D[1] = K2[bp:, bp:].copy()
        del K2
        KQ[:bp] -= Qt @ QKQ                                                              # Sigma K Q, on the first two blocks
        th = self.theta
        off = np.cumsum([0] + [Dk.shape[0] for Dk in D])

        def mv(x):
            x = np.ravel(x); y = np.empty(n)
            for k in range(nb):
                a, b = off[k], off[k + 1]
                yk = D[k] @ x[a:b]
                if k:
                    yk += E[k] @ x[off[k - 1]:a]
                if k + 1 < nb:
                    yk += E[k + 1].T @ x[b:off[k + 2]]
                y[a:b] = yk
            return y
        mu_max = float(eigsh(LinearOperator((n, n), matvec=mv, dtype=float), k=1, which="LA", v0=np.ones(n) / np.sqrt(n),
                             return_eigenvectors=False, tol=0.0)[0])
        if th * mu_max >= 1.0:
            return float("inf"), float(th * mu_max)
        # the block Cholesky factor of I - theta Sigma K Sigma, the rows' s_i and the forward substitution as it goes
        y_d, s_all, t1 = [], [], 0.0
        Lprev = Yprev = None
        for k in range(nb):
            M = -th * D[k]
            dg = -np.diag(M).copy()                                                     # theta A_ii
            M[np.diag_indices_from(M)] += 1.0
            s = np.zeros(M.shape[0])
            if k:
                F = solve_triangular(Lprev, (-th * E[k]).T, lower=True, check_finite=False).T   # C_{k,k-1} = M_{k,k-1} C_{k-1}^-T
                M -= F @ F.T
                s += np.sum(F ** 2, axis=1)
            Lk, info = dpotrf(M, lower=1, clean=1, overwrite_a=1)
            if info != 0:
                return float("inf"), float(th * mu_max)
            s += np.sum(np.tril(Lk, -1) ** 2, axis=1)
            y_d.append(dg + s); s_all.append(s)
            rhs = KQ[off[k]:off[k + 1]] if k < 2 else np.zeros((Lk.shape[0], q))
            if k:
                rhs = rhs - F @ Yprev
            Yprev = solve_triangular(Lk, rhs, lower=True, check_finite=False)
            t1 += float(np.sum(Yprev ** 2))
            Lprev = Lk
        y_d = np.concatenate(y_d); s_all = np.concatenate(s_all)
        term1 = 0.5 * th * t1
        term2 = float(np.sum(-np.log1p(-y_d) - y_d) + np.sum(s_all)) / (2.0 * th)
        return term1 + term2, float(th * mu_max)
