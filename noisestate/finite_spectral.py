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

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .accel import solve_fixed_point
from .engine import EngineBase
from .compile import compile_structure, reject_leads
from .results import TriangleResult
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid
from .grid_cache import triangle_grid


class SpectralCompiled:
    def __init__(self, model: Model):
        model.validate()
        self.model = model
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
        st = compile_structure(model)
        self.channels, self.nW = st.channels, st.nW
        self.prim, self.index, self.nX, self.nU = st.prim, st.index, st.nX, st.nU
        self.A, self.state_inputs, self.sigma = st.A, st.state_inputs, st.sigma
        self.rows, self.loss, self.rep, self.reps = st.rows, st.loss, st.rep, st.reps
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
        cache = self.g.__dict__.setdefault("_read_cache", {})
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
    def _path(self, key, **kw):
        cache = self.g.__dict__.setdefault("_paths", {})
        if key not in cache:
            g = self.g
            cache[key] = g.path(g.t, g.a, **kw)
        return cache[key]

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
        N = self.N
        if not hasattr(self, "_panel_idx"):
            panel = np.concatenate([np.full(pc.n, pc.p) for pc in self.g.pieces])
            self._panel_idx = [np.concatenate([q * N + np.where(panel == p)[0] for q in range(len(self.prim))])
                               for p in range(self.g.P)]
        Z = np.zeros_like(B)
        for p, idx in enumerate(self._panel_idx):
            rhs = B[idx].copy()
            for q in range(p):
                jdx = self._panel_idx[q]
                rhs -= (-M[np.ix_(idx, jdx)]) @ Z[jdx]        # (I - M) has -M off the diagonal blocks
            Z[idx] = np.linalg.solve(np.eye(len(idx)) - M[np.ix_(idx, idx)], rhs)
        return Z


class SpectralFiniteSolver(EngineBase):
    RIDGE = 1e-11          # relative Tikhonov term on the best-response system: needed with delayed rows, 2e-13 effect without

    def __init__(self, model: Model, verbose: bool = False):
        self.c = SpectralCompiled(model)
        self.model = model
        self.verbose = verbose
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    # ------------------------------------------------ best-response pieces
    def _row_operator(self, agent: Agent, rows, inst):
        """Per channel, the operator (N x nR N) from stacked row kernels gamma to the action kernel."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        delays = [c.rows[agent.name][r][3] for r in range(nR)]
        Gk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            for k in range(nW):
                yk = rows[r][:, k]
                if np.abs(yk).max() > 0:
                    Gk[k, :, r * N:(r + 1) * N] += c.conv_right(yk, delays[r])
            for (k, age, w) in inst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.read(0.0, age)
        return Gk

    def _response_operators(self, agent: Agent, R: np.ndarray):
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        out = []
        for ui, u in enumerate(agent.controls):
            Ru = R[:, ui].reshape(len(c.prim), N)
            Cu = np.zeros((n_prim, N))
            for p in range(len(c.prim)):
                if np.abs(Ru[p]).max() > 0 and c.prim[p] != u:
                    Cu[p * N:(p + 1) * N] = c.response_op(Ru[p])
            Cu[c.block(u)] = np.eye(N)
            out.append(Cu)
        return out

    def _foc_operators(self, agent: Agent, R: np.ndarray):
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atoms, Q, q = c.loss[agent.name]
        AO = np.concatenate([c.atom_op(at) for at in atoms], axis=0)
        QZ = np.kron(Q, np.eye(N))
        Fu = []
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0)); op += QZ[j0 * N:(j0 + 1) * N] @ AO
            if not agent.myopic:
                for j, at in enumerate(atoms):
                    name, lag = at
                    Qj = QZ[j * N:(j + 1) * N] @ AO
                    if name in agent.controls:
                        if name == u and lag > 0:
                            op += np.exp(-c.rho * lag) * c.read(-lag, -lag) @ Qj
                        continue
                    rj = c.atom_op(at) @ R[:, ui]
                    if np.abs(rj).max() > 0:
                        op += c.continuation_op(rj) @ Qj
            Fu.append(op)
        return Fu

    def _projection_operator(self, agent: Agent, rows, inst):
        """H (nR N x nW N): E[phi_t dY_r^seen(u)] at every (t, b) node, from the FOC kernels."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        delays = [c.rows[agent.name][r][3] for r in range(nR)]
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            for k in range(nW):
                yk = rows[r][:, k]
                if np.abs(yk).max() > 0:
                    raw = c.read(-delays[r], -delays[r]) @ yk if delays[r] else yk
                    H[r * N:(r + 1) * N, k * N:(k + 1) * N] += c.projection_op(raw, delays[r])
            for (k, age, w) in inst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.read(0.0, -age)
        return H

    # ------------------------------------------------------ best response
    def best_response(self, agent: Agent, maps):
        c = self.c; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(agent, ytil, yinst)
        Resp = self._response_operators(agent, R)
        Fu = self._foc_operators(agent, R)
        H = self._projection_operator(agent, ytil, yinst)
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        for ui in range(nU):
            rowsl = slice(ui * nR * N, (ui + 1) * nR * N)
            for vi in range(nU):
                FR = Fu[ui] @ Resp[vi]
                colsl = slice(vi * nR * N, (vi + 1) * nR * N)
                for k in range(nW):
                    Amat[rowsl, colsl] += H[:, k * N:(k + 1) * N] @ (FR @ Gk[k])
            for k in range(nW):
                bvec[rowsl] += H[:, k * N:(k + 1) * N] @ (Fu[ui] @ Zpass[:, k])
        scale = np.abs(Amat).max()
        gamma = np.linalg.solve(Amat + self.RIDGE * scale * np.eye(nG), -bvec).reshape(nU, nR, N)
        cact = np.zeros((nU, N, nW))
        for ui in range(nU):
            for k in range(nW):
                cact[ui, :, k] = Gk[k] @ gamma[ui].reshape(-1)
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp[ui] @ cact[ui]
        gmap = self.maps_from_world(agent, Zfull, cact)
        return gmap, {"gamma": gamma, "action": cact, "Zfull": Zfull}


    def maps_from_world(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps of `agent` reproducing its action kernels cact (nU, N, nW) given the closed-loop
        primary kernels Zfull: one weighted least-squares projection per time row."""
        c = self.c; g = c.g; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        gmap = np.zeros((nU, nR, N))
        for (p, idx) in c.trows:
            tv = g.t[idx[0]]
            w = g.row_weights(tv, side=(-1 if tv >= g.bp[p + 1] - 1e-12 else +1))[idx]
            cols = np.concatenate([r * N + idx for r in range(nR)])
            Bsub = Bk[:, idx][:, :, cols]
            G = sum((Bsub[k] * w[:, None]).T @ Bsub[k] for k in range(nW))
            tr = np.trace(G)
            if tr <= 0:
                continue
            G += 1e-13 * tr / G.shape[0] * np.eye(G.shape[0])
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
        if not hasattr(self, "_mass_rho"):
            self._mass_rho = c.g.mass_matrix(rho=c.rho)
        G = np.einsum("ink,nm,jmk->ij", zeta, self._mass_rho, zeta)
        return float(0.5 * np.sum(Q * G))

    def solve(self, init=None, tol: float = 1e-8, damping: float = 0.5, max_newton: int = 8,
              variable: str = "actions") -> TriangleResult:
        """variable="actions": iterate on the agents' action kernels, raw maps derived by projection
        (robust where early-time maps are ill-determined).  variable="maps": iterate on raw maps.
        init: action kernels (nU, N, nW) or raw maps (nU, nR, N) per agent; either is accepted and
        converted to the iteration variable."""
        t0 = time.time()
        hist, evals = [], [0]
        shapes = self.action_shapes
        packa, unpacka = self.pack_actions, self.unpack_actions
        if variable == "actions" and self.model.ties:
            # tied agents' action kernels differ by a channel permutation the symmetry implies; raw maps
            # (on each agent's own rows) carry over verbatim, so iterate on maps when ties are present
            variable = "maps"
        kind = self.init_kind(init) if init is not None else None
        if variable == "actions":
            if kind is None:
                acts0 = {a.name: np.zeros(shapes[a.name]) for a in self.model.agents}
            elif kind == "actions":
                acts0 = init
            else:
                Z0 = self.c.closed_loop(init)
                acts0 = {a.name: np.stack([Z0[self.c.block(u)] for u in a.controls]) for a in self.model.agents}
            z = packa(acts0)

            def F(zz):
                evals[0] += 1
                return packa(self.response_actions(unpacka(zz))) - zz
        else:
            maps = self.zero_maps() if kind is None else (init if kind == "maps" else self.maps_from_actions(init))
            z = self.pack(maps)

            def F(zz):
                evals[0] += 1
                return self.pack(self.response_map(self.unpack(zz))) - zz
        z, resid, nev, converged, message = solve_fixed_point(F, z, tol=tol, verbose=self.verbose, damping=damping,
                                                     max_newton=max_newton)
        hist.append(resid)
        maps = self.maps_from_actions(unpacka(z)) if variable == "actions" else self.unpack(z)
        Z = self.c.closed_loop(maps)
        res = TriangleResult(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                             iterations=evals[0], seconds=time.time() - t0, history=hist, message=message,
                             solver_class=type(self), solver_kw={"verbose": self.verbose},
                             solve_kw={"tol": tol, "damping": damping, "max_newton": max_newton, "variable": variable})
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, Z)
            g, out = self.best_response(a, maps)
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)
        return res
