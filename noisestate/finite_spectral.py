"""Finite-horizon equilibrium on the piecewise-spectral triangle (see triangle.py).

Every kernel K(t, s) lives on the triangle grid in (t, age) coordinates cut by
the delays.  Strategies are raw maps g[u][r](t, b): the control at t is the
sum over signal rows of int_0^t g(t, b) dY_r^seen(t - b), with b the age of the
observation increment.  The construction is the stationary engine's: closed
loop as one linear system in the nodal kernels; best response in the agent's
passive world with the per-date first-order condition (instantaneous term,
discounted continuation through the impulse responses, delayed reads) affine
in the map on the passive rows; raw map by projection, one Gram per time
node; Anderson fixed point over the action kernels (or the raw maps), Newton-Krylov polish.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from functools import cached_property

import numpy as np

from .engine import EngineBase
from .compile import CompiledBase, reject_leads
from .results import TriangleResult
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid
from .grid_cache import triangle_grid


class SpectralCompiled(CompiledBase):
    def __init__(self, model: Model):
        super().__init__(model)
        reject_leads(model, 'spectral finite engine')
        hz = model.horizon
        self.T = float(hz.window)
        lags = model.all_lags()
        bp = list(hz.breakpoints) if hz.breakpoints else TriangleGrid.breakpoints(self.T, lags, hz.unit)
        for l in lags:
            if not any(abs(l - b) < 1e-9 * max(1.0, self.T) for b in bp):
                raise ValueError(f"lag {l} is not a breakpoint of the time/age partition {bp}; set horizon.unit "
                                 "so that every lag is a multiple of it")
        self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes)   # shared
        g = self.g
        self.N = g.N
        self.rho = float(hz.discount)
        # state propagation operators (matrix exponentials of A)
        if self.nX:
            # state propagation e^{A(t-r)} along the Volterra path; entrywise weights from expm, which
            # is exact for defective A too (an eigen-decomposition would not be)
            lp = g.path(g.t, g.a, r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]))
            self.Vol = np.zeros((self.nX, self.nX, self.N, self.N))
            if lp.rows is not None:
                d = g.t[lp.rows] - lp.r
                E = self._expm_batch(d)                                        # (nq, nX, nX)
                for i in range(self.nX):
                    for j in range(self.nX):
                        self.Vol[i, j] = lp.apply(E[:, i, j])
        # time rows (for per-time projections)
        rows = {}
        for pc in g.pieces:
            for it in range(pc.nt):
                idx = pc.offset + it * pc.na + np.arange(pc.na)
                rows.setdefault((pc.p, it), []).append(idx)
        self.trows: List[Tuple[int, np.ndarray]] = [(k[0], np.concatenate(v)) for k, v in sorted(rows.items())]

    # ------------------------------------------------------------ reads
    def _expm_batch(self, ds: np.ndarray) -> np.ndarray:
        """e^{A d} for every d in ds: (len, nX, nX).  Uses the eigen-decomposition when it is well
        conditioned, otherwise expm per distinct d."""
        from scipy.linalg import expm
        A = self.A
        lam, V = np.linalg.eig(A)
        if np.linalg.cond(V) < 1e8:
            W = np.linalg.inv(V)
            return np.einsum("im,km,mj->kij", V, np.exp(np.outer(ds, lam)), W).real
        uniq, inv = np.unique(np.round(ds, 12), return_inverse=True)
        Es = np.stack([expm(A * d) for d in uniq])
        return Es[inv]

    def expA(self, ages: np.ndarray) -> np.ndarray:
        """e^{A a} at the given ages: (len, nX, nX)."""
        return self._expm_batch(np.asarray(ages, dtype=float))

    def read(self, dt: float, da: float) -> np.ndarray:
        """Matrix reading a kernel at (t - dt, a - da) from nodal values, with one-sided
        limits chosen by the node's position in its piece; zero where the read age is
        negative (for da > 0 this is exact on delay-aligned pieces)."""
        key = (round(dt, 12), round(da, 12))
        cache = self.g.read_cache
        if key in cache:
            return cache[key]
        g = self.g
        if dt == 0.0 and da == 0.0:
            M = np.eye(self.N)
        else:
            M = g.interp(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a)
            if da > 0:
                M[g.a0 < da - 1e-12] = 0.0
        cache[key] = M
        return M

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_op(self, atom: Atom) -> np.ndarray:
        name, lag = atom
        M = np.zeros((self.N, len(self.prim) * self.N))
        M[:, self.block(name)] = self.read(lag, lag)
        return M

    def expr_op(self, expr) -> np.ndarray:
        M = np.zeros((self.N, len(self.prim) * self.N))
        for (name, lag), c in expr.items():
            M[:, self.block(name)] += c * self.read(lag, lag)
        return M

    def row_op(self, agent: str, r: int, excluded: set):
        """Seen row r of `agent` as (regular operator on Z, {source: [(age, weight)]}),
        the regular part already shifted by the observation delay."""
        name, drift, E, delay = self.rows[agent][r]
        S = self.read(delay, delay)
        reg = np.zeros((self.N, len(self.prim) * self.N))
        deltas: Dict[str, List[Tuple[float, float]]] = {}
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
            else:
                reg[:, self.block(n)] += c * (S @ self.read(l, l))
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
                deltas.setdefault(ch, []).append((delay, E[k]))
        return reg, deltas

    # ------------------------------------------------------ line operators
    # Each family of line integrals is a cached quadrature structure (triangle.LinePath);
    # an operator for a given known kernel is then two sparse products.
    @cached_property
    def _panel_idx(self):
        """Indices of every primary's unknowns on each time panel, in block order."""
        panel = np.concatenate([np.full(pc.n, pc.p) for pc in self.g.pieces])
        return [np.concatenate([q * self.N + np.where(panel == p)[0] for q in range(len(self.prim))]) for p in range(self.g.P)]

    def _path(self, key, **kw):
        if key not in self.g.paths:
            self.g.paths[key] = self.g.path(self.g.t, self.g.a, **kw)
        return self.g.paths[key]

    def conv_left(self, gker: np.ndarray, delay: float) -> np.ndarray:
        """(C y)(t, s) = int_{s+delay}^{t} g(t, t - u) y(u, s) du  for a fixed map kernel g."""
        g = self.g
        lp = self._path(("conv_left", delay), r_lo=g.s + delay, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        return lp.with_known(gker)

    def conv_right(self, yker: np.ndarray, delay: float) -> np.ndarray:
        """(C g)(t, s) = int_{s+delay}^{t} g(t, t - u) y_seen(u, s) du  for a fixed seen row y_seen."""
        g = self.g
        lp = self._path(("conv_right", delay), r_lo=g.s + delay, r_hi=g.t,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r), known_fn=lambda k, r: (r, r - g.s[k]))
        return lp.with_known(yker)

    def response_op(self, rker: np.ndarray) -> np.ndarray:
        """(C c)(t, s) = int_s^t R(t, t - r) c(r, s) dr  for a fixed impulse-response kernel R."""
        g = self.g
        lp = self._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        return lp.with_known(rker)

    def continuation_op(self, rker: np.ndarray) -> np.ndarray:
        """(C z)(t, s) = int_t^T e^{-rho (tau - t)} R(tau, tau - t) z(tau, s) dtau."""
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.T), point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-self.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        return lp.with_known(rker, disc)

    def projection_op(self, yker: np.ndarray, delay: float) -> np.ndarray:
        """(H phi)(t, b) = int_0^{u - delay} phi(t, s) y_raw(u - delay, s) ds with u = t - b."""
        g = self.g
        u = g.s
        lp = self._path(("projection", delay), r_lo=np.zeros(self.N), r_hi=np.maximum(u - delay, 0.0),
                        point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r),
                        known_fn=lambda k, r: (np.full_like(r, u[k] - delay), u[k] - delay - r))
        return lp.with_known(yker)

    # ------------------------------------------------ kernel algebra (see EngineBase)
    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        out = np.zeros((Y.shape[1], self.N, self.N))
        for k in range(Y.shape[1]):
            if np.abs(Y[:, k]).max() > 0:
                out[k] = self.conv_right(Y[:, k], delay)
        return out

    def instant(self, age: float) -> np.ndarray:
        return self.read(0.0, age)

    def instant_adjoint(self, age: float) -> np.ndarray:
        return self.read(0.0, -age)

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        N = self.N; Cu = np.zeros((len(self.prim) * N, N))
        for p in range(len(self.prim)):
            if p != own and np.abs(Ru[p]).max() > 0:
                Cu[p * N:(p + 1) * N] = self.response_op(Ru[p])
        return Cu

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        out = np.zeros((Rj.shape[1], self.N, self.N))
        for j in range(Rj.shape[1]):
            if np.abs(Rj[:, j]).max() > 0:
                out[j] = self.continuation_op(Rj[:, j])
        return out

    def own_lag_read(self, lag: float) -> np.ndarray:
        return self.read(-lag, -lag)

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        N = self.N; H = np.zeros((N, self.nW * N))
        for k in range(self.nW):
            yk = Y[:, k]
            if np.abs(yk).max() > 0:
                raw = self.read(-delay, -delay) @ yk if delay else yk
                H[:, k * N:(k + 1) * N] = self.projection_op(raw, delay)
        return H

    # ------------------------------------------------------- closed loop
    row = row_op                                          # the engines' common name

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None, impulse_controls=()):
        """maps[agent]: (n_ctrl, n_rows, N) nodal raw maps g(t, b).  Columns: Brownian channels,
        then one impulse column per control in impulse_controls (unit mass at the shock time).
        Returns Z (n_prim N, ncol)."""
        g = self.g; N = self.N; nW = self.nW
        imp = list(impulse_controls)
        n = len(self.prim) * N; ncol = nW + len(imp)
        M = np.zeros((n, n)); B = np.zeros((n, ncol))
        excl = set(next(a for a in self.model.agents if a.name == excluded).controls) if excluded else set()
        # states
        if self.nX:
            EA = self.expA(g.a)                                             # (N, nX, nX)
            for k in range(nW):
                v = self.sigma[:, k]
                for i in range(self.nX):
                    B[self.block(self.prim[i]), k] += EA[:, i, :] @ v
            inp = np.zeros((self.nX, N, n))
            for si, (nm, lag), c in self.state_inputs:
                if nm in excl:
                    continue
                inp[si] += c * self.atom_op((nm, lag))
            for i in range(self.nX):
                for j in range(self.nX):
                    M[self.block(self.prim[i])] += self.Vol[i, j] @ inp[j]
            for col, u in enumerate(imp):
                for si, (nm, lag), c in self.state_inputs:
                    if nm != u:
                        continue
                    v = np.zeros(self.nX); v[si] = c
                    EAd = self.expA(np.maximum(g.a - lag, 0.0)) if lag else EA
                    on = (g.a0 >= lag - 1e-12) if lag else np.ones(N, dtype=bool)
                    for i in range(self.nX):
                        B[self.block(self.prim[i]), nW + col] += on * (EAd[:, i, :] @ v)
        # controls from maps
        for a in self.model.agents:
            if a.name == excluded:
                continue
            gm = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = self.block(u)
                for r, (rname, drift, E, delay) in enumerate(self.rows[a.name]):
                    reg, deltas = self.row_op(a.name, r, excl)
                    gker = gm[ui, r]
                    M[bl] += self.conv_left(gker, delay) @ reg
                    for src, dl in deltas.items():
                        if src in self.channels:
                            col = self.channels.index(src)
                        elif src in imp:
                            col = nW + imp.index(src)
                        else:
                            continue
                        for (age, w) in dl:
                            B[bl, col] += w * (self.read(0.0, age) @ gker)
        return self._solve_causal(M, B)

    def _solve_causal(self, M: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve (I - M) Z = B exploiting causality: a kernel value at time panel p depends only on
        values at panels <= p, so with nodes grouped by panel the system is block lower triangular
        and is solved by block forward substitution (one dense solve per panel)."""
        Z = np.zeros_like(B)
        for p, idx in enumerate(self._panel_idx):
            rhs = B[idx].copy()
            for q in range(p):
                jdx = self._panel_idx[q]
                rhs -= (-M[np.ix_(idx, jdx)]) @ Z[jdx]        # (I - M) has -M off the diagonal blocks
            Z[idx] = np.linalg.solve(np.eye(len(idx)) - M[np.ix_(idx, idx)], rhs)
        return Z


class SpectralFiniteSolver(EngineBase):
    RESULT = TriangleResult
    TOL, DAMPING, MAX_NEWTON = 1e-8, 0.5, 8
    MAP_RIDGE = 1e-13      # ridge of the per-time-row map projection, relative to the best-identified row's Gram
    RIDGE = 1e-11          # relative Tikhonov term on the best-response system: needed with delayed rows, 2e-13 effect without

    def __init__(self, model: Model, verbose: bool = False):
        super().__init__(model, verbose)
        self.c = SpectralCompiled(model)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    # ------------------------------------------------ best-response pieces
    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        scale = np.abs(Amat).max()
        return np.linalg.solve(Amat + self.RIDGE * scale * np.eye(Amat.shape[0]), -bvec)

    def _project(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        return self.maps_from_world(agent, Zfull, cact)

    def maps_from_world(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps of `agent` reproducing its action kernels cact (nU, N, nW) given the closed-loop
        primary kernels Zfull: one weighted least-squares projection per time row."""
        c = self.c; g = c.g; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        gmap = np.zeros((nU, nR, N))
        systems = []
        for (p, idx) in c.trows:
            tv = g.t[idx[0]]
            w = g.row_weights(tv, side=(-1 if tv >= g.bp[p + 1] - 1e-12 else +1))[idx]
            cols = np.concatenate([r * N + idx for r in range(nR)])
            Bsub = Bk[:, idx][:, :, cols]
            G = sum((Bsub[k] * w[:, None]).T @ Bsub[k] for k in range(nW))
            systems.append((idx, w, Bsub, G))
        # one ridge for every time row, relative to the best-identified row: just after a delay a row's
        # history is short and its Gram tiny, and a ridge relative to that row's own Gram regularises
        # nothing, leaving the map there to round-off (the map iteration then stalls near 1e-6)
        scale = max((np.trace(G) / G.shape[0] for (_, _, _, G) in systems if np.trace(G) > 0), default=0.0)
        for (idx, w, Bsub, G) in systems:
            if np.trace(G) <= 0:
                continue
            G = G + self.MAP_RIDGE * scale * np.eye(G.shape[0])
            for ui in range(nU):
                rhs = sum((Bsub[k] * w[:, None]).T @ cact[ui, idx, k] for k in range(nW))
                sol = np.linalg.solve(G, rhs)
                for r in range(nR):
                    gmap[ui, r, idx] = sol[r * len(idx):(r + 1) * len(idx)]
        return gmap

    def world_from_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Closed-loop primary kernels when every agent's action kernels are given."""
        c = self.c; N, nW = c.N, c.nW
        n = len(c.prim) * N
        Z = np.zeros((n, nW))
        for a in self.model.agents:
            for ui, u in enumerate(a.controls):
                Z[c.block(u)] = actions[a.name][ui]
        if c.nX:
            EA = c.expA(c.g.a)
            X0 = np.zeros((c.nX * N, nW))                                       # homogeneous part, (comp, node)
            for k in range(nW):
                for i in range(c.nX):
                    X0[i * N:(i + 1) * N, k] = EA[:, i, :] @ c.sigma[:, k]
            inp = np.zeros((c.nX * N, nW)); Lx = np.zeros((c.nX * N, c.nX * N))
            for si, (nm, lag), coef in c.state_inputs:
                if nm in c.model.state_names:
                    j = c.model.state_names.index(nm)
                    Lx[si * N:(si + 1) * N, j * N:(j + 1) * N] += coef * c.read(lag, lag)
                else:
                    inp[si * N:(si + 1) * N] += coef * (c.read(lag, lag) @ Z[c.block(nm)])
            V = np.zeros((c.nX * N, c.nX * N))
            for i in range(c.nX):
                for j in range(c.nX):
                    V[i * N:(i + 1) * N, j * N:(j + 1) * N] = c.Vol[i, j]
            rhs = X0 + V @ inp
            X = np.linalg.solve(np.eye(c.nX * N) - V @ Lx, rhs) if Lx.any() else rhs
            Z[:c.nX * N] = X
        return Z

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        Z = self.world_from_actions(actions)
        return {a.name: self.maps_from_world(a, Z, actions[a.name]) for a in self.model.agents}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])                  # (m, N, nW)
        G = np.einsum("ink,nm,jmk->ij", zeta, c.g.mass_matrix(rho=c.rho), zeta)
        return float(0.5 * np.sum(Q * G))

    def _finish(self, res) -> None:
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, res.Z)
            g, out = self.best_response(a, res.maps)
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)

