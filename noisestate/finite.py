"""Finite-horizon equilibrium in noise-state linear strategies (uniform time cells).

CROSS-CHECK ENGINE.  The production finite-horizon engine is finite_spectral.py; this first-order
scheme is kept as an independent discretisation for validation (Richardson-extrapolated), selected
with horizon.kind = "finite_cells".

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
refine N or Richardson-extrapolate for high accuracy.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

from .accel import solve_fixed_point
from .compile import compile_structure
from .results import CellResult as FiniteResult
from .spec import Agent, Atom, Model


class FiniteCompiled:
    def __init__(self, model: Model):
        model.validate()
        self.model = model
        for a in model.agents:
            for term in a.loss:
                for atom in term[1:]:
                    if any(l < 0 for (n, l) in model.expand({atom: 1.0})):
                        raise NotImplementedError(f"agent {a.name}: lead atoms in loss terms ({atom}) are not supported by "
                                                  "the finite-horizon engines yet; the stationary engine supports them")
        hz = model.horizon
        self.T = float(hz.window)
        self.N = int(hz.nodes)
        self.h = self.T / self.N
        self.rho = float(hz.discount)
        self.times = np.arange(self.N) * self.h
        st = compile_structure(model)
        self.channels, self.nW = st.channels, st.nW
        self.prim, self.index, self.nX, self.nU = st.prim, st.index, st.nX, st.nU
        self.A, self.state_inputs, self.sigma = st.A, st.state_inputs, st.sigma
        self.rows, self.loss, self.rep, self.reps = st.rows, st.loss, st.rep, st.reps
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


class FiniteSolver:
    def __init__(self, model: Model, verbose: bool = False):
        self.c = FiniteCompiled(model)
        self.model = model
        self.verbose = verbose
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N, self.c.N) for a in model.agents}
        self._warm = {}                                          # last Krylov solution per agent (warm start)
        N = self.c.N
        self.tri = np.tril(np.ones((N, N), dtype=bool), -1)     # v < i

    def pack(self, maps):
        return np.concatenate([maps[n][:, :, self.tri].reshape(-1) for n in self.c.reps])

    def unpack(self, z):
        maps, pos = {}, 0
        for n in self.c.reps:
            nu, nr, N, _ = self.shapes[n]
            size = nu * nr * int(self.tri.sum())
            g = np.zeros(self.shapes[n]); g[:, :, self.tri] = z[pos:pos + size].reshape(nu, nr, -1); pos += size
            maps[n] = g
        for a in self.model.agents:
            if a.name not in maps:
                maps[a.name] = maps[self.c.rep[a.name]]
        return maps

    def zero_maps(self):
        return {a.name: np.zeros(self.shapes[a.name]) for a in self.model.agents}

    # ---------------------------------------------------------- best response
    def best_response(self, agent: Agent, maps):
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

        b = affine(np.zeros(nU * nR * nfree))
        op = LinearOperator((nU * nR * nfree, nU * nR * nfree), matvec=lambda v: affine(v) - b)
        if nU * nR * nfree <= 200:
            M = np.column_stack([op.matvec(e) for e in np.eye(nU * nR * nfree)])
            gvec = np.linalg.lstsq(M, -b, rcond=1e-13)[0]       # delayed rows give zero columns (see stationary)
        else:
            x0 = self._warm.get(agent.name)
            if x0 is not None and x0.shape[0] != nU * nR * nfree:
                x0 = None
            gvec, info = lgmres(op, -b, x0=x0, rtol=1e-12, atol=0, maxiter=400)
            if info != 0:
                gvec, info = lgmres(op, -b, x0=gvec, rtol=1e-12, atol=0, maxiter=1000)
            if info != 0:
                raise RuntimeError(f"cell engine: the best-response linear solve did not converge (lgmres info {info})")
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
            G += 1e-13 * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            for ui in range(nU):
                sol = np.linalg.solve(G, B @ cact[ui, i])
                g[ui, :, i, :i] = sol.reshape(nR, i)
        return g, {"gamma": gam, "action": cact, "Zfull": Zfull}

    def response_map(self, maps):
        new = {}
        for a in self.model.agents:
            if self.c.rep[a.name] == a.name:
                new[a.name] = self.best_response(a, maps)[0]
        for a in self.model.agents:
            if a.name not in new:
                new[a.name] = new[self.c.rep[a.name]]
        return new

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_kernel(Z, at) for at in atoms])          # (m, N, NB)
        disc = np.exp(-c.rho * c.times)
        G = np.einsum("itc,jtc,t->ij", zeta, zeta, disc) * c.h * c.h    # sum_t h e^{-rho t} sum_j h zeta zeta
        return float(0.5 * np.sum(Q * G))

    def solve(self, init=None, tol: float = 1e-8, damping: float = 0.5, max_newton: int = 60) -> FiniteResult:
        """Anderson on the raw maps, then a Newton-Krylov polish.  init: raw maps."""
        t0 = time.time()
        maps = init if init is not None else self.zero_maps()
        z = self.pack(maps)
        hist, evals = [], [0]

        def F(zz):
            evals[0] += 1
            return self.pack(self.response_map(self.unpack(zz))) - zz

        z, resid, nev, converged, message = solve_fixed_point(F, z, tol=tol, verbose=self.verbose, damping=damping,
                                                     max_newton=max_newton)
        hist.append(resid)
        maps = self.unpack(z)
        Z = self.c.closed_loop(maps)
        res = FiniteResult(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                           iterations=evals[0], seconds=time.time() - t0, history=hist, message=message,
                           solver_class=type(self), solver_kw={"verbose": self.verbose},
                           solve_kw={"tol": tol, "damping": damping, "max_newton": max_newton})
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, Z)
        return res
