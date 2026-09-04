"""Stationary equilibrium in noise-state linear strategies.

Every process is a kernel in shock age a on [0, L] per Brownian channel.  An
agent's strategy is a set of raw-observation maps g[u][r](b): its control u at
time t is the sum over its signal rows r of int_0^L g[u][r](b) dY_r(t-b).

Given all maps the closed loop is one linear system in the nodal kernels of
the states and controls, solved for every channel at once.  An agent's best
response is computed in its passive world (its own maps switched off): its
information is the history of its passive signals, which does not depend on
its own strategy, so parametrising the control by kernels on those rows makes
the per-date first-order condition (instantaneous derivative + discounted
continuation through the physical state and through the other agents'
reactions) affine in the unknown, and the best response is one linear solve.
The raw map is then recovered by projecting the resulting action kernel onto
the agent's closed-loop signal rows.  The equilibrium is the fixed point of
the best-response map over all raw maps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize._nonlin import NoConvergence

from .grid import AgeGrid
from .accel import solve_fixed_point
from .spec import Agent, Atom, Model, parse_atom


# ------------------------------------------------------------------ helpers
@dataclass
class Delta:
    """A unit impulse of quantity `name` at age `age` with weight `w` (an instantaneous entry)."""
    age: float
    w: float


class Compiled:
    """Grid, index maps and constant operators for a stationary model."""

    def __init__(self, model: Model):
        model.validate()
        self.model = model
        hz = model.horizon
        lags = model.all_lags()
        if hz.breakpoints:
            bp = list(hz.breakpoints)
        elif lags or hz.unit:
            bp = AgeGrid.breakpoints_from_delays(hz.window, lags, hz.unit, hz.unit_range)
        else:
            bp = [0.0, hz.window]
        for l in lags:
            if not any(abs(l - b) < 1e-12 for b in bp):
                raise ValueError(f"lag {l} is not a panel breakpoint; set horizon.unit so every lag is a multiple")
        self.grid = AgeGrid(bp, hz.nodes)
        self.N = self.grid.N
        self.rho = float(hz.discount)
        self.channels = list(model.channels)
        self.nW = len(self.channels)
        self.prim = model.state_names + model.control_names       # primary quantities
        self.nX = len(model.state_names)
        self.nU = len(model.control_names)
        self.index = {n: i for i, n in enumerate(self.prim)}
        self.ctrl_agent = {u: a for a in model.agents for u in a.controls}
        self._shift_cache: Dict[float, np.ndarray] = {}
        # state dynamics: A (nX x nX) contemporaneous, plus per-atom inputs
        self.A = np.zeros((self.nX, self.nX))
        self.state_inputs: List[Tuple[int, Atom, float]] = []   # (state index, atom, coef) for controls/lagged
        for i, s in enumerate(model.states):
            for (n, l), c in model.expand(s.drift).items():
                if n in model.state_names and l == 0:
                    self.A[i, model.state_names.index(n)] += c
                else:
                    self.state_inputs.append((i, (n, l), c))
        self.sigma = np.zeros((self.nX, self.nW))
        for i, s in enumerate(model.states):
            for ch, c in s.noise.items():
                self.sigma[i, self.channels.index(ch)] = c
        self.P0, self.Pin = self.grid.propagator(self.A) if self.nX else (np.zeros((0, 0)), np.zeros((0, 0)))
        # per agent: row structure and loss form
        self.rows: Dict[str, List[Tuple[str, dict, np.ndarray, float]]] = {}
        for a in model.agents:
            rr = []
            for r in a.signals:
                E = np.zeros(self.nW)
                for ch, c in r.noise.items():
                    E[self.channels.index(ch)] = c
                rr.append((r.name, model.expand(r.drift), E, float(r.delay)))
            self.rows[a.name] = rr
        self.loss: Dict[str, Tuple[List[Atom], np.ndarray, np.ndarray]] = {}
        for a in model.agents:
            atoms: List[Atom] = []
            terms = []
            for term in a.loss:
                coef = float(term[0])
                ex = [model.expand({s: 1.0}) for s in term[1:]]
                for e in ex:
                    for k in e:
                        if k not in atoms:
                            atoms.append(k)
                terms.append((coef, ex))
            m = len(atoms)
            Q = np.zeros((m, m)); q = np.zeros(m)
            for coef, ex in terms:
                if len(ex) == 1:
                    for k, c in ex[0].items():
                        q[atoms.index(k)] += coef * c
                else:
                    for k1, c1 in ex[0].items():
                        for k2, c2 in ex[1].items():
                            i, j = atoms.index(k1), atoms.index(k2)
                            Q[i, j] += coef * c1 * c2
                            Q[j, i] += coef * c1 * c2     # so that loss = 1/2 z'Qz + q'z
            self.loss[a.name] = (atoms, Q, q)
        # tie groups -> representative
        self.rep: Dict[str, str] = {a.name: a.name for a in model.agents}
        for group in model.ties:
            for n in group:
                self.rep[n] = group[0]
        self.reps = [a.name for a in model.agents if self.rep[a.name] == a.name]

    # ------------------------------------------------------------- operators
    def shift(self, tau: float) -> np.ndarray:
        key = round(float(tau), 12)
        if key not in self._shift_cache:
            self._shift_cache[key] = self.grid.shift(key)
        return self._shift_cache[key]

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_op(self, atom: Atom) -> np.ndarray:
        """N x (n_prim N) matrix giving the kernel of `name@lag` from the primary vector."""
        name, lag = atom
        M = np.zeros((self.N, len(self.prim) * self.N))
        M[:, self.block(name)] = self.shift(lag)
        return M

    def expr_op(self, expr: Dict[Atom, float]) -> np.ndarray:
        M = np.zeros((self.N, len(self.prim) * self.N))
        for atom, c in expr.items():
            M[:, self.block(atom[0])] += c * self.shift(atom[1])
        return M

    # ------------------------------------------------------- closed loop
    def row_seen(self, agent: str, r: int, excluded: set) -> Tuple[np.ndarray, Dict[str, List[Delta]]]:
        """Regular part of row r as seen by the agent (delayed), as an operator on the
        primary vector, and the instantaneous entries per source: channel names for
        Brownian noise, control names for observed-control impulses.  Controls in
        `excluded` (the agent whose reaction is switched off) contribute impulses
        through their own impulse channel instead of through a kernel."""
        name, drift, E, delay = self.rows[agent][r]
        S = self.shift(delay)
        regular = np.zeros((self.N, len(self.prim) * self.N))
        deltas: Dict[str, List[Delta]] = {}
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append(Delta(delay + l, c))
            else:
                regular[:, self.block(n)] += c * (S @ self.shift(l))
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
                deltas.setdefault(ch, []).append(Delta(delay, E[k]))
        return regular, deltas

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                    impulse_controls: Sequence = ()):
        """Solve the closed loop for the Brownian channels and for unit impulses in
        `impulse_controls` (whose owning agent's reactions are switched off when it is
        `excluded`).  maps[agent] has shape (n_controls, n_rows, N).
        Returns Z with shape (n_prim N, nW + len(impulse_controls))."""
        n = len(self.prim) * self.N
        ncol = self.nW + len(impulse_controls)
        M = np.zeros((n, n))
        B = np.zeros((n, ncol))
        excl = set(self.model.agents[[a.name for a in self.model.agents].index(excluded)].controls) if excluded else set()
        # states
        if self.nX:
            U = np.zeros((self.nX * self.N, n))           # input to the propagator, (node, comp) ordering
            for i, (nm, lag), c in self.state_inputs:
                if nm in excl:
                    continue
                U[i::self.nX, :] += c * self.atom_op((nm, lag))
            xs = slice(0, self.nX * self.N)
            # propagator rows are (node, comp); primary vector is (comp, node): permute
            perm = np.arange(self.nX * self.N).reshape(self.N, self.nX).T.reshape(-1)   # prim index -> (node,comp) index
            PX = self.Pin[perm]            # (comp,node) rows
            P0X = self.P0[perm]
            M[xs, :] += PX @ U
            for k in range(self.nW):
                B[xs, k] += P0X @ self.sigma[:, k]
            for j, u in enumerate(impulse_controls):
                col = self.nW + j
                for i, (nm, lag), c in self.state_inputs:
                    if nm != u:
                        continue
                    v = np.zeros(self.nX); v[i] = c
                    if lag == 0:
                        B[xs, col] += P0X @ v
                    else:
                        B[xs, col] += self.grid.jump_injector(self.A, lag)[perm] @ v
        # controls
        for a in self.model.agents:
            if a.name == excluded:
                continue
            g = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = self.block(u)
                for r in range(len(a.signals)):
                    regular, deltas = self.row_seen(a.name, r, excl)
                    gur = g[ui, r]
                    M[bl, :] += self.grid.conv_op_left(gur) @ regular
                    for src, dl in deltas.items():
                        if src in self.channels:
                            col = self.channels.index(src)
                        elif src in impulse_controls:
                            col = self.nW + list(impulse_controls).index(src)
                        else:
                            continue
                        for d in dl:
                            B[bl, col] += d.w * (self.shift(d.age) @ gur)
        Z = np.linalg.solve(np.eye(n) - M, B)
        return Z


# ------------------------------------------------------------------ results
@dataclass
class Result:
    model: Model
    compiled: Compiled
    maps: Dict[str, np.ndarray]                 # raw maps g[agent] (n_ctrl, n_rows, N)
    Z: np.ndarray                               # closed-loop primary kernels (n_prim N, nW)
    converged: bool
    residual: float
    iterations: int
    seconds: float
    foc: Dict[str, dict] = field(default_factory=dict)   # per agent: decomposition of the FOC kernels
    costs: Dict[str, float] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)

    @property
    def ages(self) -> np.ndarray:
        return self.compiled.grid.nodes

    def kernel(self, name: str, lag: float = 0.0) -> np.ndarray:
        """Closed-loop kernel of a quantity: array (N, nW), one column per channel."""
        c = self.compiled
        expr = c.model.expand({f"{name}@{lag}" if lag else name: 1.0})
        return c.expr_op(expr) @ self.Z

    def action_kernel(self, control: str) -> np.ndarray:
        return self.Z[self.compiled.block(control)]

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {'converged' if self.converged else 'NOT converged'} "
                 f"residual {self.residual:.2e} in {self.iterations} evaluations, {self.seconds:.1f}s; "
                 f"grid {c.grid.P} panels x {c.grid.n} nodes on [0, {c.grid.L}], rho={c.rho}"]
        for a in self.model.agents:
            lines.append(f"  {a.name}: E[loss] = {self.costs.get(a.name, float('nan')):+.6f}")
            for u in a.controls:
                k = self.action_kernel(u)
                lines.append(f"    {u}(0+) on channels: " + ", ".join(f"{ch}={k[0, j]:+.4f}" for j, ch in enumerate(c.channels)))
        return "\n".join(lines)


# ------------------------------------------------------------------- solver
class StationarySolver:
    def __init__(self, model: Model, verbose: bool = False, nonreactors: Optional[Dict[str, List[str]]] = None):
        self.c = Compiled(model)
        self.model = model
        self.verbose = verbose
        self.nonreactors = nonreactors or {}     # agent -> agents whose maps ignore its deviations (naive observers)
        c = self.c
        self.shapes = {a.name: (len(a.controls), len(a.signals), c.N) for a in model.agents}

    # ------------------------------------------------------------ packing
    def pack(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([maps[n].reshape(-1) for n in self.c.reps])

    def unpack(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        maps = {}
        pos = 0
        for n in self.c.reps:
            sh = self.shapes[n]
            size = int(np.prod(sh))
            maps[n] = z[pos:pos + size].reshape(sh)
            pos += size
        for a in self.model.agents:
            if a.name not in maps:
                maps[a.name] = maps[self.c.rep[a.name]]
        return maps

    def zero_maps(self) -> Dict[str, np.ndarray]:
        return {a.name: np.zeros(self.shapes[a.name]) for a in self.model.agents}

    # ------------------------------------------------------ best response
    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        c = self.c
        N, nW = c.N, c.nW
        rows = c.rows[agent.name]
        nR, nU = len(rows), len(agent.controls)
        # passive world + impulse responses (own reactions off)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]                                  # R: (n_prim N, nU)
        if getattr(self, "reaction_hook", None) is not None:
            R = self.reaction_hook(agent, maps)                       # experiment hook: alternative reaction model
        if getattr(self, "passive_hook", None) is not None:
            Zpass = self.passive_hook(agent, maps, Zpass, R)          # experiment hook: alternative passive world
        frozen = self.nonreactors.get(agent.name, [])
        if frozen:
            mz = {k: (np.zeros_like(v) if k in frozen else v) for k, v in maps.items()}
            R = c.closed_loop(mz, excluded=agent.name, impulse_controls=agent.controls)[:, nW:]
        # passive rows: regular kernels per channel (N, nW) and instantaneous entries
        ytil, yinst = [], []
        for r in range(nR):
            regular, deltas = c.row_seen(agent.name, r, set(agent.controls))
            ytil.append(regular @ Zpass)
            yinst.append([(c.channels.index(src), d) for src, dl in deltas.items() if src in c.channels for d in dl])
        # action kernels from gamma: c_u[:, k] = sum_r (Conv[ytil_r,k] + E_rk S_delta) gamma_ur
        # Build G_k: (N, nR N) per channel, shared across controls
        Gk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            for k in range(nW):
                Gk[k, :, r * N:(r + 1) * N] += c.grid.conv_op(ytil[r][:, k])
            for (k, d) in yinst[r]:
                Gk[k, :, r * N:(r + 1) * N] += d.w * c.shift(d.age)
        # full world primary kernels as affine in gamma: Z = Zpass + sum_u Conv[R_u] c_u, then own block := c_u
        n_prim = len(c.prim) * N
        # per control u and channel k: Z[:, k] = Zpass[:, k] + Ru_conv @ c_u[:, k]
        Ru_conv = []
        for ui, u in enumerate(agent.controls):
            Ru = R[:, ui].reshape(len(c.prim), N)
            Cu = np.zeros((n_prim, N))
            for p in range(len(c.prim)):
                Cu[p * N:(p + 1) * N] = c.grid.conv_op(Ru[p])
            Cu[c.block(u)] = np.eye(N)     # own control kernel is the action itself
            Ru_conv.append(Cu)
        # unknown vector gamma: (nU, nR, N) flattened
        nG = nU * nR * N
        # affine map gamma -> Z_k (n_prim N) for each channel: Z_k = Zpass_k + sum_u Ru_conv[u] @ Gk[k] @ gamma_u
        # FOC kernels per control u and channel k, affine in gamma
        atoms, Q, q = c.loss[agent.name]
        atom_ops = [c.atom_op(at) for at in atoms]                          # N x n_prim N
        AO = np.concatenate(atom_ops, axis=0)                               # (m N) x (n_prim N), zeta stacked
        m = len(atoms)
        QZ = np.kron(Q, np.eye(N))                                          # (mN x mN): (Q zeta)
        # instantaneous derivative wrt u: row selector of atom (u, 0)
        Fu_ops = []     # per control: operator (N x n_prim N) mapping Z_k -> phi_u,k  (regular continuation included)
        Fu_phys = []    # physical part (all reactions off), for the decomposition
        if want_decomp or True:
            Zphys = c.closed_loop(self.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, nW:]
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim))
            op_phys = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0))
                op += (QZ[j0 * N:(j0 + 1) * N] @ AO)
                op_phys += (QZ[j0 * N:(j0 + 1) * N] @ AO)
            if not agent.myopic:
                # regular continuation: sum_j corr(r_uj, rho) (Q zeta)_j
                Ru = R[:, ui]
                Rp = Zphys[:, ui]
                for j, at in enumerate(atoms):
                    name, lag = at
                    if name in agent.controls:
                        # own controls: no reaction (envelope); a lagged read of u itself is a delta at s = lag
                        if name == u and lag > 0:
                            op += np.exp(-c.rho * lag) * c.shift(-lag) @ (QZ[j * N:(j + 1) * N] @ AO)
                            op_phys += np.exp(-c.rho * lag) * c.shift(-lag) @ (QZ[j * N:(j + 1) * N] @ AO)
                        continue
                    r_j = atom_ops[j] @ Ru
                    r_jp = atom_ops[j] @ Rp
                    op += c.grid.corr_op(r_j, c.rho) @ (QZ[j * N:(j + 1) * N] @ AO)
                    op_phys += c.grid.corr_op(r_jp, c.rho) @ (QZ[j * N:(j + 1) * N] @ AO)
            Fu_ops.append(op); Fu_phys.append(op_phys)
        # projection operator H: (nR N) x (nW N) acting on phi stacked by channel
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            for k in range(nW):
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += c.grid.corr_op(ytil[r][:, k], 0.0)
            for (k, d) in yinst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += d.w * c.shift(-d.age)
        # assemble linear system A gamma = -b:  H phi_u = 0 for each u
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        for ui in range(nU):
            for k in range(nW):
                # phi_{u,k} = Fu_ops[ui] @ Z_k;  Z_k = Zpass_k + sum_v Ru_conv[v] @ Gk[k] @ gamma_v
                const = Fu_ops[ui] @ Zpass[:, k]
                rowsl = slice(ui * nR * N, (ui + 1) * nR * N)
                Hk = H[:, k * N:(k + 1) * N]
                bvec[rowsl] += Hk @ const
                for vi in range(nU):
                    colsl = slice(vi * nR * N, (vi + 1) * nR * N)
                    Amat[rowsl, colsl] += Hk @ (Fu_ops[ui] @ (Ru_conv[vi] @ Gk[k]))
        gamma = np.linalg.solve(Amat, -bvec).reshape(nU, nR, N)
        # action kernels and full world
        cact = np.zeros((nU, N, nW))
        for ui in range(nU):
            for k in range(nW):
                cact[ui, :, k] = Gk[k] @ gamma[ui].reshape(-1)
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Ru_conv[ui] @ cact[ui]
        # raw map: project each action kernel on the agent's closed-loop rows
        yraw, yrinst = [], []
        for r in range(nR):
            regular, deltas = c.row_seen(agent.name, r, set())
            yraw.append(regular @ Zfull)
            yrinst.append([(c.channels.index(src), d) for src, dl in deltas.items() if src in c.channels for d in dl])
        Bk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            for k in range(nW):
                Bk[k, :, r * N:(r + 1) * N] += c.grid.conv_op(yraw[r][:, k])
            for (k, d) in yrinst[r]:
                Bk[k, :, r * N:(r + 1) * N] += d.w * c.shift(d.age)
        W = c.grid.mass
        Gram = sum((Bk[k] * W[:, None]).T @ Bk[k] for k in range(nW))
        g = np.zeros((nU, nR, N))
        for ui in range(nU):
            rhs = sum((Bk[k] * W[:, None]).T @ cact[ui, :, k] for k in range(nW))
            g[ui] = np.linalg.solve(Gram + 1e-14 * np.trace(Gram) / Gram.shape[0] * np.eye(nR * N), rhs).reshape(nR, N)
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            dec = {}
            for ui, u in enumerate(agent.controls):
                phi = np.stack([Fu_ops[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                phi_phys = np.stack([Fu_phys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
            out["decomp"] = dec
        return g, out

    # ------------------------------------------------ maps from kernels
    def maps_from_kernels(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Raw maps that reproduce given closed-loop primary kernels Z (n_prim N, nW):
        each agent's action kernels projected on its closed-loop signal rows."""
        c = self.c
        N, nW = c.N, c.nW
        W = c.grid.mass
        maps = {}
        for a in self.model.agents:
            nR = len(a.signals)
            Bk = np.zeros((nW, N, nR * N))
            for r in range(nR):
                regular, deltas = c.row_seen(a.name, r, set())
                y = regular @ Z
                for k in range(nW):
                    Bk[k, :, r * N:(r + 1) * N] += c.grid.conv_op(y[:, k])
                for src, dl in deltas.items():
                    if src in c.channels:
                        k = c.channels.index(src)
                        for d in dl:
                            Bk[k, :, r * N:(r + 1) * N] += d.w * c.shift(d.age)
            Gram = sum((Bk[k] * W[:, None]).T @ Bk[k] for k in range(nW))
            g = np.zeros((len(a.controls), nR, N))
            for ui, u in enumerate(a.controls):
                cu = Z[c.block(u)]
                rhs = sum((Bk[k] * W[:, None]).T @ cu[:, k] for k in range(nW))
                g[ui] = np.linalg.solve(Gram + 1e-14 * np.trace(Gram) / Gram.shape[0] * np.eye(nR * N), rhs).reshape(nR, N)
            maps[a.name] = g
        return maps

    # ------------------------------------------------------ fixed point
    def response_map(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new = {}
        for a in self.model.agents:
            if self.c.rep[a.name] != a.name:
                continue
            g, _ = self.best_response(a, maps)
            new[a.name] = g
        for a in self.model.agents:
            if a.name not in new:
                new[a.name] = new[self.c.rep[a.name]]
        return new

    def solve(self, init: Optional[Dict[str, np.ndarray]] = None, tol: float = 1e-10, damping: float = 0.3,
              pre_iterations: int = 20, max_newton: int = 60, pre_tol: float = 1e-3) -> Result:
        t0 = time.time()
        maps = init if init is not None else self.zero_maps()
        z = self.pack(maps)
        hist = []
        evals = [0]

        def F(zz):
            evals[0] += 1
            return self.pack(self.response_map(self.unpack(zz))) - zz

        # damped pre-phase
        z, resid, nev, converged = solve_fixed_point(F, z, tol=tol, verbose=self.verbose, damping=damping,
                                                     max_newton=max_newton, method="newton", pre_iterations=pre_iterations, pre_tol=pre_tol)
        hist.append(resid)
        maps = self.unpack(z)
        Z = self.c.closed_loop(maps)
        res = Result(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                     iterations=evals[0], seconds=time.time() - t0, history=hist)
        # decomposition and costs
        for a in self.model.agents:
            _, out = self.best_response(a, maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            res.costs[a.name] = self.expected_loss(a, Z)
        return res

    def expected_loss(self, agent: Agent, Z: np.ndarray) -> float:
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])          # (m, N, nW)
        W = c.grid.mass
        G = np.einsum("ink,jnk,n->ij", zeta, zeta, W)                # <zeta_i, zeta_j>
        return float(0.5 * np.sum(Q * G))
