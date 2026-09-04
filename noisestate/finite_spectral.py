"""Finite-horizon equilibrium on the piecewise-spectral triangle (see triangle.py).

Every kernel K(t, s) lives on the triangle grid in (t, age) coordinates cut by
the delays.  Strategies are raw maps g[u][r](t, b): the control at t is the
sum over signal rows of int_0^t g(t, b) dY_r^seen(t - b), with b the age of the
observation increment.  The construction is the stationary engine's: closed
loop as one linear system in the nodal kernels; best response in the agent's
passive world with the per-date first-order condition (instantaneous term,
discounted continuation through the impulse responses, delayed reads) affine
in the map on the passive rows; raw map by projection, one Gram per time
node; Newton-Krylov fixed point over all raw maps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence

from .accel import solve_fixed_point
from .compile import compile_structure
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid


class SpectralCompiled:
    def __init__(self, model: Model):
        model.validate()
        self.model = model
        hz = model.horizon
        self.T = float(hz.window)
        lags = model.all_lags()
        bp = list(hz.breakpoints) if hz.breakpoints else TriangleGrid.breakpoints(self.T, lags, hz.unit)
        self.g = TriangleGrid(bp, hz.nodes, hz.nodes)
        g = self.g
        self.N = g.N
        self.rho = float(hz.discount)
        st = compile_structure(model); self.st = st
        self.channels, self.nW = st.channels, st.nW
        self.prim, self.index, self.nX, self.nU = st.prim, st.index, st.nX, st.nU
        self.A, self.state_inputs, self.sigma = st.A, st.state_inputs, st.sigma
        self.rows, self.loss, self.rep, self.reps = st.rows, st.loss, st.rep, st.reps
        self._read_cache: Dict[Tuple[float, float], np.ndarray] = {}
        self._paths: Dict[tuple, object] = {}
        # state propagation operators (eigen-decomposition of A)
        if self.nX:
            lam, V = np.linalg.eig(self.A); W = np.linalg.inv(V)
            self._lam, self._V, self._W = lam, V, W
            ops = {}
            for l in set(np.round(lam, 12)):
                ops[l] = g.line_op(g.t, g.a, r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                                   weight_fn=lambda k, r, l=l: np.exp(l * (g.t[k] - r)))
            self.Vol = np.zeros((self.nX, self.nX, self.N, self.N), dtype=complex)
            for i in range(self.nX):
                for j in range(self.nX):
                    for m_ in range(self.nX):
                        self.Vol[i, j] += V[i, m_] * W[m_, j] * ops[np.round(lam[m_], 12)]
            self.Vol = self.Vol.real
        # time rows (for per-time projections)
        rows = {}
        for pc in g.pieces:
            for it in range(pc.nt):
                idx = pc.offset + it * pc.na + np.arange(pc.na)
                rows.setdefault((pc.p, it), []).append(idx)
        self.trows: List[Tuple[int, np.ndarray]] = [(k[0], np.concatenate(v)) for k, v in sorted(rows.items())]

    # ------------------------------------------------------------ reads
    def expA(self, ages: np.ndarray) -> np.ndarray:
        """e^{A a} at the given ages: (len, nX, nX)."""
        E = np.einsum("im,km,mj->kij", self._V, np.exp(np.outer(ages, self._lam)), self._W)
        return E.real

    def read(self, dt: float, da: float) -> np.ndarray:
        """Matrix reading a kernel at (t - dt, a - da) from nodal values, with one-sided
        limits chosen by the node's position in its piece; zero where the read age is
        negative (for da > 0 this is exact on delay-aligned pieces)."""
        key = (round(dt, 12), round(da, 12))
        if key in self._read_cache:
            return self._read_cache[key]
        g = self.g
        if dt == 0.0 and da == 0.0:
            M = np.eye(self.N)
        else:
            M = g.interp(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a)
            if da > 0:
                M[g.a0 < da - 1e-12] = 0.0
        self._read_cache[key] = M
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
        if key not in self._paths:
            g = self.g
            self._paths[key] = g.path(g.t, g.a, **kw)
        return self._paths[key]

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
        return np.linalg.solve(np.eye(n) - M, B)


@dataclass
class SpectralResult:
    model: Model
    compiled: SpectralCompiled
    maps: Dict[str, np.ndarray]
    Z: np.ndarray
    converged: bool
    residual: float
    iterations: int
    seconds: float
    costs: Dict[str, float] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)

    @property
    def grid(self) -> TriangleGrid:
        return self.compiled.g

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        c = self.compiled
        K = c.expr_op(c.model.expand({name: 1.0})) @ self.Z
        return K if channel is None else K[:, c.channels.index(channel)]

    def evaluate(self, name: str, channel: str, t, s) -> np.ndarray:
        """Kernel value at (t, s) points (response at t to a unit shock at s)."""
        t = np.asarray(t, dtype=float); s = np.asarray(s, dtype=float)
        return self.grid.interp(t, t - s) @ self.kernel(name, channel)

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {'converged' if self.converged else 'NOT converged'} residual {self.residual:.2e} "
                 f"in {self.iterations} evaluations, {self.seconds:.1f}s; triangle grid {c.g.P} panels, "
                 f"{len(c.g.pieces)} pieces x {c.g.nt}x{c.g.na} nodes = {c.N} nodes on [0, {c.T}], rho={c.rho}"]
        for a in self.model.agents:
            lines.append(f"  {a.name}: E[cost] = {self.costs.get(a.name, float('nan')):+.8f}")
        return "\n".join(lines)


class SpectralFiniteSolver:
    def __init__(self, model: Model, verbose: bool = False, ridge: float = 1e-11):
        self.c = SpectralCompiled(model)
        self.model = model
        self.verbose = verbose
        self.ridge = ridge
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    def pack(self, maps):
        return np.concatenate([maps[n].reshape(-1) for n in self.c.reps])

    def unpack(self, z):
        maps, pos = {}, 0
        for n in self.c.reps:
            sh = self.shapes[n]; size = int(np.prod(sh)); maps[n] = z[pos:pos + size].reshape(sh); pos += size
        for a in self.model.agents:
            if a.name not in maps:
                maps[a.name] = maps[self.c.rep[a.name]]
        return maps

    def zero_maps(self):
        return {a.name: np.zeros(self.shapes[a.name]) for a in self.model.agents}

    # ------------------------------------------------------ best response
    def best_response(self, agent: Agent, maps):
        c = self.c; g = c.g; N, nW = c.N, c.nW
        rows = c.rows[agent.name]; nR, nU = len(rows), len(agent.controls)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]
        # passive seen rows: regular kernels (N, nW) and instantaneous entries [(k, age, w)]
        ytil, yinst = [], []
        for r in range(nR):
            reg, deltas = c.row_op(agent.name, r, set(agent.controls))
            ytil.append(reg @ Zpass)
            yinst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        delays = [rows[r][3] for r in range(nR)]
        # action from gamma: c_u[:, k] = sum_r (Conv[ytil_rk] + E_rk read(0, delay)) gamma_ur
        Gk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            for k in range(nW):
                yk = ytil[r][:, k]
                if np.abs(yk).max() > 0:
                    Gk[k, :, r * N:(r + 1) * N] += c.conv_right(yk, delays[r])
            for (k, age, w) in yinst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.read(0.0, age)
        # full world from the action: Z = Zpass + sum_u Resp_u c_u, own block := c_u
        n_prim = len(c.prim) * N
        Resp = []
        for ui, u in enumerate(agent.controls):
            Ru = R[:, ui].reshape(len(c.prim), N)
            Cu = np.zeros((n_prim, N))
            for p in range(len(c.prim)):
                if np.abs(Ru[p]).max() > 0 and c.prim[p] != u:
                    Cu[p * N:(p + 1) * N] = c.response_op(Ru[p])
            Cu[c.block(u)] = np.eye(N)
            Resp.append(Cu)
        # FOC operators: phi_u = F_u Z (per channel column)
        atoms, Q, q = c.loss[agent.name]
        AO = np.concatenate([c.atom_op(at) for at in atoms], axis=0)
        QZ = np.kron(Q, np.eye(N))
        Fu = []
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0)); op += QZ[j0 * N:(j0 + 1) * N] @ AO
            if not agent.myopic:
                Ru = R[:, ui]
                for j, at in enumerate(atoms):
                    name, lag = at
                    if name in agent.controls:
                        if name == u and lag > 0:
                            op += np.exp(-c.rho * lag) * c.read(-lag, -lag) @ (QZ[j * N:(j + 1) * N] @ AO)
                        continue
                    rj = c.atom_op(at) @ Ru
                    if np.abs(rj).max() > 0:
                        op += c.continuation_op(rj) @ (QZ[j * N:(j + 1) * N] @ AO)
            Fu.append(op)
        # projection H (nR N x nW N)
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            for k in range(nW):
                yk = ytil[r][:, k]
                if np.abs(yk).max() > 0:
                    # ytil is the seen row (already shifted); projection integrates the raw row at u - delay:
                    H[r * N:(r + 1) * N, k * N:(k + 1) * N] += c.projection_op(c.read(-delays[r], -delays[r]) @ yk if delays[r] else yk, delays[r])
            for (k, age, w) in yinst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.read(0.0, -age)
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        for ui in range(nU):
            rowsl = slice(ui * nR * N, (ui + 1) * nR * N)
            for k in range(nW):
                Hk = H[:, k * N:(k + 1) * N]
                bvec[rowsl] += Hk @ (Fu[ui] @ Zpass[:, k])
                for vi in range(nU):
                    colsl = slice(vi * nR * N, (vi + 1) * nR * N)
                    Amat[rowsl, colsl] += Hk @ (Fu[ui] @ (Resp[vi] @ Gk[k]))
        scale = np.abs(Amat).max()
        gamma = np.linalg.solve(Amat + self.ridge * scale * np.eye(nG), -bvec).reshape(nU, nR, N)
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
        rows = c.rows[agent.name]; nR, nU = len(rows), len(agent.controls)
        delays = [rows[r][3] for r in range(nR)]
        yraw, yrinst = [], []
        for r in range(nR):
            reg, deltas = c.row_op(agent.name, r, set())
            yraw.append(reg @ Zfull)
            yrinst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        Bk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            for k in range(nW):
                yk = yraw[r][:, k]
                if np.abs(yk).max() > 0:
                    Bk[k, :, r * N:(r + 1) * N] += c.conv_right(yk, delays[r])
            for (k, age, w) in yrinst[r]:
                Bk[k, :, r * N:(r + 1) * N] += w * c.read(0.0, age)
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
            if any(nm in c.model.state_names for _, (nm, _), _ in c.state_inputs):
                raise NotImplementedError("lagged state inputs: use variable='maps'")
            EA = c.expA(c.g.a)
            X = np.zeros((c.nX, N, nW))
            for k in range(nW):
                for i in range(c.nX):
                    X[i, :, k] += EA[:, i, :] @ c.sigma[:, k]
            inp = np.zeros((c.nX, N, nW))
            for si, (nm, lag), coef in c.state_inputs:
                inp[si] += coef * (c.read(lag, lag) @ Z[c.block(nm)])
            for i in range(c.nX):
                for j in range(c.nX):
                    X[i] += c.Vol[i, j] @ inp[j]
            for i in range(c.nX):
                Z[c.block(c.prim[i])] = X[i]
        return Z

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        Z = self.world_from_actions(actions)
        return {a.name: self.maps_from_world(a, Z, actions[a.name]) for a in self.model.agents}

    def response_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        maps = self.maps_from_actions(actions)
        new = {}
        for a in self.model.agents:
            if self.c.rep[a.name] == a.name:
                new[a.name] = self.best_response(a, maps)[1]["action"]
        for a in self.model.agents:
            if a.name not in new:
                new[a.name] = new[self.c.rep[a.name]]
        return new

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
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])                  # (m, N, nW)
        w = c.g.mass * np.exp(-c.rho * c.g.t)
        G = np.einsum("ink,jnk,n->ij", zeta, zeta, w)
        return float(0.5 * np.sum(Q * G))

    def solve(self, init=None, tol: float = 1e-8, damping: float = 0.5, pre_iterations: int = 10,
              max_newton: int = 8, pre_tol: float = 1e-3, variable: str = "actions") -> SpectralResult:
        """variable="actions": iterate on the agents' action kernels, raw maps derived by projection
        (robust where early-time maps are ill-determined).  variable="maps": iterate on raw maps."""
        t0 = time.time()
        hist, evals = [], [0]
        shapes = {a.name: (len(a.controls), self.c.N, self.c.nW) for a in self.model.agents}
        reps = self.c.reps

        def packa(acts):
            return np.concatenate([acts[n].reshape(-1) for n in reps])

        def unpacka(z):
            acts, pos = {}, 0
            for n in reps:
                size = int(np.prod(shapes[n])); acts[n] = z[pos:pos + size].reshape(shapes[n]); pos += size
            for a in self.model.agents:
                if a.name not in acts:
                    acts[a.name] = acts[self.c.rep[a.name]]
            return acts
        if variable == "actions":
            acts0 = init if init is not None else {a.name: np.zeros(shapes[a.name]) for a in self.model.agents}
            z = packa(acts0)

            def F(zz):
                evals[0] += 1
                return packa(self.response_actions(unpacka(zz))) - zz
        else:
            maps = init if init is not None else self.zero_maps()
            z = self.pack(maps)

            def F(zz):
                evals[0] += 1
                return self.pack(self.response_map(self.unpack(zz))) - zz
        z, resid, nev, converged = solve_fixed_point(F, z, tol=tol, verbose=self.verbose, damping=damping,
                                                     max_newton=max_newton)
        hist.append(resid)
        maps = self.maps_from_actions(unpacka(z)) if variable == "actions" else self.unpack(z)
        Z = self.c.closed_loop(maps)
        res = SpectralResult(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                             iterations=evals[0], seconds=time.time() - t0, history=hist)
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, Z)
        return res
