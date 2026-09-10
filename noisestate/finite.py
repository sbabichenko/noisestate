"""Finite-horizon equilibrium in noise-state linear strategies (uniform time cells).

CROSS-CHECK ENGINE.  The production finite-horizon engine is finite_spectral.py; this first-order
scheme is kept as an independent discretisation for validation (Richardson-extrapolated), selected
with numerics.engine = "cells" (its result reports kind "finite_cells").

Time [0, T] is cut into N cells of length h.  Shocks are the cell increments
dW_j (variance h).  A kernel K[i, j] is the response at cell i (state at t_i,
control over cell i) to a unit increment in cell j; controls are predictable,
so every kernel is strictly lower triangular.  An agent's strategy is a raw
map g[u][r][i, v]: its control over cell i is sum_r sum_{v<i} g[u][r][i, v]
dY_r,v.  Everything else follows the stationary engine: closed loop by a
forward march, passive-world best response (the per-cell first-order
condition is affine in the map on the passive rows and is solved with a
Krylov method), raw map by projection, and an Anderson fixed point over the raw maps.

The scheme is first order in h (Euler state step, cell-averaged controls);
refine N or Richardson-extrapolate for high accuracy.  The means (targets,
constant drifts, initial states) are solved on the cells at the end, the same
first-order condition on the mean paths (the hooks of EngineBase's mean layer, FiniteSolver._mean_conditions).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

from .engine import EngineBase, singular_system_message
from .compile import CompiledBase, reject_leads
from .results import CellResult
from .spec import Agent, Atom, Model


class FiniteCompiled(CompiledBase):
    def __init__(self, model: Model):
        super().__init__(model)
        reject_leads(model, 'cell engine')
        hz = model.horizon
        self.T = float(hz.extent)
        self.N = int(hz.nodes)
        self.h = self.T / self.N
        self.rho = float(hz.discount)
        self.times = np.arange(self.N) * self.h
        st = self.st
        # observation delays in cells
        self.rows = {a: [(n, d, E, int(round(delay / self.h))) for (n, d, E, delay) in rr] for a, rr in st.rows.items()}
        for l in model.all_lags():
            if abs(l / self.h - round(l / self.h)) > 1e-9:
                raise ValueError(f"lag {l} is not a multiple of the cell length {self.h}; choose nodes so that it is")

    # kernels are arrays K[prim, i, col]; lagged atom = shift along i
    def lag_cells(self, lag: float) -> int:
        return int(round(lag / self.h))

    def atom_kernel(self, Z: np.ndarray, atom: Atom) -> np.ndarray:
        """Kernel (N, ncol) of name@lag from primary kernels Z (n_prim, N, ncol)."""
        name, lag = atom
        K = Z[self.index[name]]
        d = self.lag_cells(lag)
        out = np.zeros_like(K)
        if d == 0:
            return K.copy()
        if d > 0:
            out[d:] = K[:-d]
        else:
            out[:d] = K[-d:]
        return out

    def expr_kernel(self, Z: np.ndarray, expr: Dict[Atom, float]) -> np.ndarray:
        out = np.zeros(Z.shape[1:])
        for atom, c in expr.items():
            out += c * self.atom_kernel(Z, atom)
        return out

    # ---------------------------------------------------------- closed loop
    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                    impulse_controls=()) -> np.ndarray:
        """Forward march.  maps[agent]: (n_ctrl, n_rows, N, N) raw maps g[u][r][i, v], v < i.
        Columns: nW Brownian channels (unit increment in cell j) followed, for each impulse
        control, by N columns (unit mass of that control in cell j).  Returns Z of shape
        (n_prim, N, ncol)."""
        N, h, nW = self.N, self.h, self.nW
        imp = list(impulse_controls)
        ncol = nW * N + len(imp) * N
        Z = np.zeros((len(self.prim), N, ncol))
        agents = [a for a in self.model.agents if a.name != excluded]
        # signal-row kernels seen by each agent, filled as we go: yk[agent][r] (N, ncol)
        yk = {a.name: [np.zeros((N, ncol)) for _ in a.signals] for a in self.model.agents}

        def col_brown(k, j):
            return k * N + j

        def col_imp(u, j):
            return nW * N + imp.index(u) * N + j

        for i in range(N):
            # 1. controls over cell i from maps (rows up to cell i-1)
            for a in agents:
                g = maps[a.name]
                for ui, u in enumerate(a.controls):
                    p = self.index[u]
                    for r, (rname, drift, E, dly) in enumerate(self.rows[a.name]):
                        # seen row at cell v is row at cell v - dly
                        if i - 1 - dly < 0:
                            continue
                        gv = g[ui, r, i, :i]                       # v = 0..i-1
                        Yseen = np.zeros((i, ncol))
                        lo = max(0, dly)
                        Yseen[lo:i] = yk[a.name][r][0:i - dly] if dly else yk[a.name][r][:i]
                        Z[p, i] += gv @ Yseen
            # impulse controls: unit mass in cell i -> rate 1/h over cell i
            for u in imp:
                Z[self.index[u], i, col_imp(u, i)] += 1.0 / h
            # 2. signal rows at cell i (drift over cell i, predictable controls) + noise
            for a in self.model.agents:
                for r, (rname, drift, E, dly) in enumerate(self.rows[a.name]):
                    row = np.zeros(ncol)
                    for (n, l), c in drift.items():
                        d = self.lag_cells(l)
                        if i - d < 0 or d < 0:
                            continue
                        row += c * h * Z[self.index[n], i - d]
                    for k in range(nW):
                        if E[k] != 0.0:
                            row[col_brown(k, i)] += E[k]
                    yk[a.name][r][i] = row
            # 3. state step to cell i+1
            if i + 1 < N and self.nX:
                Xi = Z[:self.nX, i]
                inp = np.zeros((self.nX, ncol))
                for si, (n, l), c in self.state_inputs:
                    d = self.lag_cells(l)
                    if i - d < 0 or d < 0:
                        continue
                    inp[si] += c * Z[self.index[n], i - d]
                Z[:self.nX, i + 1] = Xi + h * (self.A @ Xi + inp)
                for k in range(nW):
                    Z[:self.nX, i + 1, col_brown(k, i)] += self.sigma[:, k]
        return Z

    def rows_seen(self, agent: str, Z: np.ndarray, own_off: bool):
        """Seen signal-row kernels (list over rows of (N, ncol)) computed from primary kernels."""
        N, h, nW = self.N, self.h, self.nW
        ncol = Z.shape[2]
        a = next(x for x in self.model.agents if x.name == agent)
        out = []
        for (rname, drift, E, dly) in self.rows[agent]:
            Y = np.zeros((N, ncol))
            for (n, l), c in drift.items():
                if own_off and n in a.controls:
                    continue
                d = self.lag_cells(l)
                if d < 0:
                    continue
                Y[d:] += c * h * Z[self.index[n], :N - d]
            for k in range(nW):
                if E[k] != 0.0:
                    for i in range(N):
                        Y[i, k * N + i] += E[k]
            S = np.zeros_like(Y)
            if dly:
                S[dly:] = Y[:N - dly]
            else:
                S = Y
            out.append(S)
        return out


class FiniteSolver(EngineBase):
    RESULT = CellResult
    TOL, DAMPING, MAX_NEWTON = 1e-8, 0.5, 60
    ACTIONS = False
    def __init__(self, model: Model, verbose: bool = False, settings=None):
        """The cell grid's compiled model and the map shapes (nU, nR, N, N): g[u][r][i, v], the weight the
        control in cell i puts on the seen increment of cell v < i.  settings: the tuning constants
        (noisestate.Settings, or a dict of its fields; the defaults when None)."""
        if model.horizon.kind == "transition":
            raise ValueError(f"horizon.kind 'transition' ({model.name!r}) runs on the spectral finite engine only "
                             "(noisestate.solve routes it there; this engine has no past)")
        super().__init__(model, verbose, settings=settings)
        self.c = FiniteCompiled(model)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N, self.c.N) for a in model.agents}
        self._warm = {}                                          # last Krylov solution per agent (warm start)
        N = self.c.N
        self.tri = np.tril(np.ones((N, N), dtype=bool), -1)     # v < i

    def pack(self, maps):                                   # only the causal (strictly lower) triangle is free
        """The representatives' maps as one vector: only the causal (strictly lower) triangle v < i of each
        (N, N) map is free, so the vector has nU nR N (N - 1) / 2 entries per representative."""
        return np.concatenate([maps[n][:, :, self.tri].reshape(-1) for n in self.c.reps])

    def unpack(self, z):
        """The inverse of pack: every agent's (nU, nR, N, N) maps, zero above the diagonal, tied agents filled."""
        maps, pos = {}, 0
        for n in self.c.reps:
            nu, nr, N, _ = self.shapes[n]
            size = nu * nr * int(self.tri.sum())
            g = np.zeros(self.shapes[n]); g[:, :, self.tri] = z[pos:pos + size].reshape(nu, nr, -1); pos += size
            maps[n] = g
        return self._fill_ties(maps)

    # ---------------------------------------------------------- best response
    def best_response(self, agent: Agent, maps):
        """The agent's best response to `maps` on the cells: (raw map (nU, nR, N, N), {"gamma", "action",
        "Zfull"}) with "action" (nU, N, nW N) and "Zfull" (n_prim, N, nW N).  Replaces the base's best
        response wholesale (the base's kernel-algebra pieces do not apply to the cell layout): the same
        passive-world first-order condition, affine in the map on the passive rows, solved densely up to
        200 unknowns and by LGMRES above, then the raw map by one projection per cell.  No want_decomp
        argument: this engine computes no decomposition or second-order check, and its _diagnostics is
        the base's no-op.  A singular system raises the ValueError of singular_system_message (dense
        branch: the condition estimate; Krylov branch: one probe per control)."""
        c = self.c
        N, h, nW = c.N, c.h, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        NB = nW * N
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass = Zp[:, :, :NB]                                          # (n_prim, N, NB)
        R = {u: Zp[:, :, NB + ui * N:NB + (ui + 1) * N] for ui, u in enumerate(agent.controls)}   # (n_prim, N, N): [., tau, i]
        ytil = c.rows_seen(agent.name, Zpass, own_off=True)             # list (N, NB)
        atoms, Q, q = c.loss[agent.name]
        tri = self.tri
        nfree = int(tri.sum())

        def action_from_gamma(gam):        # gam: (nU, nR, N, N) -> c_u (nU, N, NB)
            out = np.zeros((nU, N, NB))
            for ui in range(nU):
                for r in range(nR):
                    out[ui] += (gam[ui, r] * tri) @ ytil[r]
            return out

        def full_world(cact):              # cact (nU, N, NB) -> Z (n_prim, N, NB)
            Z = Zpass.copy()
            for ui, u in enumerate(agent.controls):
                # Z[p, tau, :] += sum_i h R[p, tau, i] c[i, :]
                Z += h * np.einsum("pti,ic->ptc", R[u], cact[ui])
                Z[c.index[u]] = cact[ui]                    # own control is the action itself
            return Z

        # discounted future weights dm[i, tau] = e^{-rho (tau - i) h} for tau > i (fixed per best response)
        dm = np.exp(-c.rho * h * (np.arange(N)[None, :] - np.arange(N)[:, None])) * np.triu(np.ones((N, N)), 1)
        Rj_cache = {u: {at: (c.atom_kernel(R[u], at).T * dm) for at in atoms
                        if at[0] not in agent.controls} for u in agent.controls}

        def foc(Z):                        # -> (nU, N, NB) FOC kernels
            zeta = np.stack([c.atom_kernel(Z, at) for at in atoms])          # (m, N, NB)
            Qz = np.einsum("jk,ktc->jtc", Q, zeta)                            # (m, N, NB)
            out = np.zeros((nU, N, NB))
            for ui, u in enumerate(agent.controls):
                if (u, 0.0) in atoms:
                    out[ui] += Qz[atoms.index((u, 0.0))]
                if agent.myopic:
                    continue
                for j, at in enumerate(atoms):
                    name, lag = at
                    if name in agent.controls:
                        if name == u and lag > 0:
                            d = c.lag_cells(lag)
                            out[ui, :N - d] += np.exp(-c.rho * lag) * Qz[j, d:]
                        continue
                    # continuation: sum_{tau > i} h e^{-rho (tau-i) h} Rj[tau, i] Qz[j, tau, :]
                    out[ui] += h * (Rj_cache[u][at] @ Qz[j])
            return out

        # projection H: for each row r and v < i: sum_j FOC[i, j] ytil_r[v, j] = 0
        def project(F):                    # F (nU, N, NB) -> residual (nU, nR, N, N) on tri
            out = np.zeros((nU, nR, N, N))
            for ui in range(nU):
                for r in range(nR):
                    out[ui, r] = (F[ui] @ ytil[r].T) * tri
            return out

        def affine(gvec):
            gam = np.zeros((nU, nR, N, N)); gam[:, :, tri] = gvec.reshape(nU, nR, nfree)
            return project(foc(full_world(action_from_gamma(gam))))[:, :, tri].reshape(-1)

        n = nU * nR * nfree
        b = affine(np.zeros(n))
        op = LinearOperator((n, n), matvec=lambda v: affine(v) - b)
        # a row seen with a delay of d cells is zero at cells v < d, so the map there multiplies nothing:
        # those unknowns have zero rows and columns and stay at zero (see stationary)
        keep = np.zeros((nU, nR, N, N), dtype=bool)
        for r, (rname, drift, E, dly) in enumerate(c.rows[agent.name]):
            keep[:, r] = tri & (np.arange(N)[None, :] >= dly)
        keep = keep[:, :, tri].reshape(-1)
        st = self.settings
        if n <= st.cell_dense_max:
            M = np.column_stack([op.matvec(e) for e in np.eye(n)])
            gvec = np.zeros(n)
            gvec[keep] = self._solve_regular(agent, M[np.ix_(keep, keep)], -b[keep])
        else:
            # no matrix to factor here: one random probe per control catches a control whose whole block
            # the operator annihilates (no quadratic term in itself and nothing it moves in the loss); a
            # partial deficiency inside a block is not caught, and an inconsistent singular system then
            # ends as the non-converged solve below
            rng = np.random.default_rng(0); resp = {}
            for ui, u in enumerate(agent.controls):
                g = np.zeros(n); blk = slice(ui * nR * nfree, (ui + 1) * nR * nfree)
                g[blk] = rng.standard_normal(nR * nfree) * keep[blk]
                if keep[blk].any():                          # a control with no kept unknown has nothing to probe
                    resp[u] = np.linalg.norm(op.matvec(g)) / np.linalg.norm(g)
            if resp and not min(resp.values()) > self.FOC_RCOND * max(resp.values()):
                u = min(resp, key=resp.get)
                raise ValueError(singular_system_message(agent.name) + f" (its first-order condition does not respond to its strategy for {u})")
            x0 = self._warm.get(agent.name)
            if x0 is not None and x0.shape[0] != n:
                x0 = None
            gvec, info = lgmres(op, -b, x0=x0, rtol=st.cell_krylov_rtol, atol=0, maxiter=st.cell_krylov_maxiter)
            if info != 0:
                gvec, info = lgmres(op, -b, x0=gvec, rtol=st.cell_krylov_rtol, atol=0, maxiter=st.cell_krylov_retry)
            if info != 0:
                raise RuntimeError(f"cell engine: the best-response linear solve did not converge (lgmres info {info}); "
                                   f"the system of {agent.name} may be singular (a control with no quadratic term in itself)")
            self._warm[agent.name] = gvec.copy()
        gam = np.zeros((nU, nR, N, N)); gam[:, :, tri] = gvec.reshape(nU, nR, nfree)
        cact = action_from_gamma(gam)
        Zfull = full_world(cact)
        # raw map: project each action kernel on the closed-loop seen rows
        yraw = c.rows_seen(agent.name, Zfull, own_off=False)
        g = np.zeros((nU, nR, N, N))
        for i in range(1, N):
            B = np.concatenate([yraw[r][:i] for r in range(nR)], axis=0)      # (nR*i, NB)
            G = B @ B.T
            if np.trace(G) <= 0:
                continue                                   # no information yet (delayed rows): map stays zero
            G += st.map_ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            for ui in range(nU):
                sol = np.linalg.solve(G, B @ cact[ui, i])
                g[ui, :, i, :i] = sol.reshape(nR, i)
        return g, {"gamma": gam, "action": cact, "Zfull": Zfull}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """The variance part of the agent's discounted cost over [0, T] in the world Z (n_prim, N, nW N), the
        cell layout closed_loop returns: h sum_t e^{-rho t} 1/2 sum Q_ij <zeta_i, zeta_j>, first order in h."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_kernel(Z, at) for at in atoms])          # (m, N, NB)
        disc = np.exp(-c.rho * c.times)
        G = np.einsum("itc,jtc,t->ij", zeta, zeta, disc) * c.h * c.h    # sum_t h e^{-rho t} sum_j h zeta zeta
        return float(0.5 * np.sum(Q * G))

    # ------------------------------------------------------------ means
    def _shift(self, lag: float) -> np.ndarray:
        """(N x N) reading a cell path at cell i - lag / h, zero before the first cell (a negative lag reads ahead, zero past T)."""
        d = self.c.lag_cells(lag); N = self.c.N
        return np.eye(N, k=-d)

    # ------------------------------------------------------------ means (the hooks of EngineBase's mean layer)
    def _mean_times(self) -> np.ndarray:
        return self.c.times

    def _mean_weights(self) -> np.ndarray:
        return self.c.h * np.exp(-self.c.rho * self.c.times)

    def _mean_dynamics(self):
        """The states' rows of the mean system on the cells: the Euler march of the mean dynamics from x0, first
        order in h like the kernels."""
        c = self.c; N, h, nX, nP = c.N, c.h, c.nX, len(c.prim)
        blk = lambda i: slice(i * N, (i + 1) * N)
        Mx = np.zeros((nX * N, nP * N)); bx = np.zeros(nX * N)
        step = np.eye(N) - np.eye(N, k=-1)                                 # x[i] - x[i-1] on rows i >= 1; x[0] on row 0
        for i in range(nX):
            Mx[blk(i), blk(i)] = step
            for j in range(nX):
                Mx[blk(i), blk(j)] -= h * c.A[i, j] * np.eye(N, k=-1)
            bx[blk(i)][0] = c.x0[i]; bx[blk(i)][1:] = h * c.const[i]
        for si, (nm, lag), coef in c.state_inputs:
            Mx[blk(si), blk(c.index[nm])] -= h * coef * (np.eye(N, k=-1) @ self._shift(lag))
        return Mx, bx, []

    def _mean_conditions(self, agent: Agent, maps: Dict[str, np.ndarray]):
        """The agent's mean first-order condition in every cell, one block per control: the best response's per-cell
        condition (the instantaneous derivative of 1/2 z'Qz + q'z in the control, the discounted own lagged reads,
        the continuation h sum_{tau > i} e^{-rho (tau - i) h} R[tau, i] g[tau] through the passive-world impulse
        columns) applied to the mean paths with no information constraint and the targets q as the driver."""
        c = self.c; N, h, nW, nP = c.N, c.h, c.nW, len(c.prim); NB = nW * N; a = agent
        blk = lambda i: slice(i * N, (i + 1) * N)
        dm = np.exp(-c.rho * h * (np.arange(N)[None, :] - np.arange(N)[:, None])) * np.triu(np.ones((N, N)), 1)
        atoms, Q, q = c.loss[a.name]
        Mu = np.zeros((len(a.controls) * N, nP * N)); bu = np.zeros(len(a.controls) * N)
        Zp = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls)
        for ui, u in enumerate(a.controls):
            R = Zp[:, :, NB + ui * N:NB + (ui + 1) * N]                   # (n_prim, N, N): [., tau, i]
            Ms = np.zeros((len(atoms), N, N))                            # the operator on each atom's mean path
            if (u, 0.0) in atoms:
                Ms[atoms.index((u, 0.0))] += np.eye(N)
            if not a.myopic:
                for j, (nm, lag) in enumerate(atoms):
                    if nm in a.controls:
                        if nm == u and lag > 0:
                            Ms[j] += np.exp(-c.rho * lag) * self._shift(-lag)
                        continue
                    Ms[j] += h * (c.atom_kernel(R, (nm, lag)).T * dm)
            row = blk(ui)
            for j, (nm_j, lag_j) in enumerate(atoms):
                MQ = sum((Q[j, i] * Ms[i] for i in range(len(atoms)) if Q[j, i]), np.zeros((N, N)))
                if MQ.any():
                    Mu[row, blk(c.index[nm_j])] += MQ @ self._shift(lag_j)
                if q[j]:
                    bu[row] -= q[j] * Ms[j].sum(axis=1)
        return Mu, bu

    def _mean_atoms(self, zbar: np.ndarray, atoms) -> np.ndarray:
        """The loss atoms' mean paths (m, N) from the primaries' cell paths: a lagged atom is the path shifted."""
        c = self.c; N = c.N
        return np.array([self._shift(lag) @ zbar[c.index[nm] * N:(c.index[nm] + 1) * N] for (nm, lag) in atoms]).reshape(len(atoms), N)
