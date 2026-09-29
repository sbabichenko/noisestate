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
        g = self.c.grid
        Na = int(round(self.L / h))
        ages = h * np.arange(Na + 1)
        I = g.interp(ages)
        I[-1] = g.interp([self.L], side=-1)[0]                        # the age L from the left
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

    @staticmethod
    def _conv(a, b):
        from scipy.signal import fftconvolve
        return fftconvolve(a, b)

    def _A(self, lat, zt, G):
        """(A g)_j(tau_k) = int z_j(tau - u)' g(u) du on the lattice, (m, Nt): G (nW, Nu)."""
        h, Na, Nt = lat["h"], lat["Na"], lat["Nt"]
        m = zt.shape[1]
        out = np.zeros((m, Nt))
        for j in range(m):
            for cc in range(self.nW):
                if np.any(zt[:, j, cc]):
                    out[j] += self._conv(G[cc], zt[:, j, cc])[Na:Na + Nt]
        return h * out

    def _At(self, lat, zt, B):
        """(A' b)_c(u_i) = sum_k z(tau_k - u_i)' b(tau_k) (the weights already in b), (nW, Nu): B (m, Nt)."""
        Nu = lat["Nu"]
        out = np.zeros((self.nW, Nu))
        for j in range(zt.shape[1]):
            if not np.any(B[j]):
                continue
            for cc in range(self.nW):
                if np.any(zt[:, j, cc]):
                    out[cc] += self._conv(B[j], zt[::-1, j, cc])[:Nu]
        return out

    def _K(self, lat, G):
        """K_0 g on the lattice, (nW, Nu)."""
        Na, Nt = lat["Na"], lat["Nt"]
        Ag = self._A(lat, lat["zt"], G)
        out = self._At(lat, lat["zt"], lat["dtau"][None, :] * (self.Q @ Ag))
        if self.mx:
            Gf = G[:, Na:Na + Nt]                                                        # g at u = tau >= 0
            out += self._At(lat, lat["zxt"], lat["dtau"][None, :] * (self.Lx @ Gf))
            Axg = self._A(lat, lat["zxt"], G)                                            # (mx, Nt)
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

    def _solve(self, lat, F):
        """g = (I - theta K_0 Sigma)^-1 f_0: the tilt of what the agent has not seen (Sigma), which is S f_0 wherever the
        condition holds (P_0 g = 0) and, unlike (I - theta K_0)^-1, well conditioned whenever the conditional entropic cost is
        finite (K_0's past-past block, the seen shocks', can be large: a Kyle insider's inventory)."""
        from scipy.sparse.linalg import LinearOperator, gmres
        nW, Nu, th = self.nW, lat["Nu"], self.theta
        op = LinearOperator((nW * Nu, nW * Nu), matvec=lambda x: x - th * self._K(lat, self._sigma(lat, x.reshape(nW, Nu))).ravel(),
                            dtype=float)
        b = F.ravel()
        x, info = gmres(op, b, rtol=1e-12, atol=0.0, restart=200, maxiter=50)
        res = float(np.linalg.norm(op @ x - b) / max(np.linalg.norm(b), 1e-300))
        self.gmres.append((info, res))
        if res > 1e-8:
            raise ValueError(f"the stationary risk correction's solve (I - theta K_0 Sigma) g = f_0 did not converge (residual {res:.1e})")
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
        for n, a in enumerate(ages):
            up = self.L - a
            K = int(np.floor(up / h + 1e-12))
            tk = h * np.arange(K + 1)
            w = h * np.ones(K + 1); w[0] = 0.5 * h; w[-1] = 0.5 * h
            if K == 0:
                w[:] = 0.0
            frac = up - K * h
            Iz = gr.interp(tk + a)
            Iz[tk + a >= self.L - 1e-12] = gr.interp([self.L], side=-1)[0]
            zk = np.einsum("kn,jnc->kjc", Iz, self.z)
            val = np.einsum("k,kjc,jk->c", w * np.exp(-self.rho * tk), zk, b[:, :K + 1])
            if frac > 1e-14:
                s = frac / h
                bu = (1 - s) * b[:, K] + s * b[:, K + 1]
                zL = np.einsum("n,jnc->jc", gr.interp([self.L], side=-1)[0], self.z)
                val += 0.5 * frac * (np.exp(-self.rho * tk[-1]) * np.einsum("jc,j->c", zk[-1], b[:, K])
                                     + np.exp(-self.rho * up) * np.einsum("jc,j->c", zL, bu))
            if self.mx:
                zxk = np.einsum("kn,jnc->kjc", Iz, self.zx)
                val += np.einsum("k,kjc,jk->c", w * np.exp(-self.rho * tk), zxk, bx[:, :K + 1])
                if frac > 1e-14:
                    s = frac / h
                    bxu = (1 - s) * bx[:, K] + s * bx[:, K + 1]
                    zxL = np.einsum("n,jnc->jc", gr.interp([self.L], side=-1)[0], self.zx)
                    val += 0.5 * frac * (np.exp(-self.rho * tk[-1]) * np.einsum("jc,j->c", zxk[-1], bx[:, K])
                                         + np.exp(-self.rho * up) * np.einsum("jc,j->c", zxL, bxu))
            out[n] = self.theta * val
        return out

    def risk_part(self, ui: int, h: float) -> np.ndarray:
        """theta K_0 S f_0^on on the past, at the age nodes (N, nW), on the lattice of step h."""
        lat = self._lattice(h)
        G = self._solve(lat, self._f(lat, ui))
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
        Zpass = solver._spikes(c, self._maps, self.agent)[0]
        ytil, yinst = solver._passive_rows(self.agent, Zpass)
        key = round(h, 12)
        if key not in self._qpast:
            rows = []
            for r, y in enumerate(ytil):
                if any(age != 0.0 for (_, age, _) in yinst[r]):
                    raise NotImplementedError("a risk-averse agent on the stationary engine with a delayed point loading on its rows")
                yl = lat["I"] @ y[:, :nW]                                                # (Na + 1, nW) at the ages
                for m in range(Na):                                                      # s = u_m = -L + m h
                    v = np.zeros((nW, Na + 1))
                    ages_idx = m - np.arange(0, m + 1)                                   # u_i <= s: age (m - i) h
                    v[:, :m + 1] = h * yl[ages_idx].T
                    for (k, age, w) in yinst[r]:
                        v[k, m] += w
                    rows.append(v.ravel())
            self._qpast[key] = np.linalg.qr(np.array(rows).T)[0]                         # (nW (Na + 1), q) on the past block
        Qp = self._qpast[key]
        if not full:
            return Qp
        Q = np.zeros((nW, Nu, Qp.shape[1]))
        Q[:, :Na + 1] = Qp.reshape(nW, Na + 1, -1)
        return Q.reshape(nW * Nu, -1)

    def _sigma(self, lat, G):
        """Sigma g: the part of g (nW, Nu) the agent has not seen (its information's span taken out on the past block)."""
        Na = lat["Na"]
        Qp = self._info_basis(lat, full=False)
        out = G.copy()
        gp = G[:, :Na + 1].ravel()
        out[:, :Na + 1] -= (Qp @ (Qp.T @ gp)).reshape(self.nW, Na + 1)
        return out

    def cond_excess(self, h: float):
        """E[J_0 | F_0] - E C_0 averaged over the past, on the lattice of step h: with Pi the projector on the agent's
        information and Sigma = I - Pi, (theta / 2) tr(Pi K B K Pi) + (2 theta)^-1 sum (-log(1 - theta mu) - theta mu), mu the
        eigenvalues of Sigma K Sigma and B = Sigma (I - theta Sigma K Sigma)^-1 Sigma (the tilt of the unseen shocks given the
        seen ones).  Returns (excess, the largest theta mu: the breakdown at 1)."""
        lat = self._lattice(h)
        n = self.nW * lat["Nu"]
        Km = np.empty((n, n)); E = np.zeros(n)
        for j in range(n):
            E[j] = 1.0
            Km[:, j] = self._K(lat, E.reshape(self.nW, -1)).ravel()
            E[j] = 0.0
        K = 0.5 * (Km + Km.T)
        Q = self._info_basis(lat)
        S = np.eye(n) - Q @ Q.T
        SKS = S @ K @ S
        mu, V = np.linalg.eigh(0.5 * (SKS + SKS.T))
        th = self.theta
        if th * mu.max() >= 1.0:
            return float("inf"), float(th * mu.max())
        KQ = K @ Q
        SV = S @ V
        Y = SV.T @ KQ                                                                    # (n, q)
        term1 = 0.5 * th * float(np.sum((Y ** 2) / (1.0 - th * mu)[:, None]))
        term2 = float(np.sum(-np.log1p(-th * mu) - th * mu) / (2.0 * th))
        return term1 + term2, float(th * mu.max())
