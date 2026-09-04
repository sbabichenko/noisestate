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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid import AgeGrid
from .grid_cache import age_grid
from .accel import solve_fixed_point
from .compile import compile_structure
from .results import StationaryResult as Result
from .spec import Agent, Atom, Model


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
        self.grid = age_grid(tuple(round(float(b), 12) for b in bp), hz.nodes)   # shared, with its operator caches
        self.N = self.grid.N
        self.rho = float(hz.discount)
        st = compile_structure(model)
        self.channels, self.nW = st.channels, st.nW
        self.prim, self.index, self.nX, self.nU = st.prim, st.index, st.nX, st.nU
        self.A, self.state_inputs, self.sigma = st.A, st.state_inputs, st.sigma
        self.rows, self.loss, self.rep, self.reps = st.rows, st.loss, st.rep, st.reps
        self._atom_cache: Dict[tuple, np.ndarray] = {}
        self.P0, self.Pin = self.grid.propagator(self.A) if self.nX else (np.zeros((0, 0)), np.zeros((0, 0)))

    # ------------------------------------------------------------- operators
    def shift(self, tau: float) -> np.ndarray:
        return self.grid.shift_cached(tau)

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_op(self, atom: Atom) -> np.ndarray:
        """N x (n_prim N) matrix giving the kernel of `name@lag` from the primary vector (cached)."""
        key = (atom[0], round(float(atom[1]), 12))
        if key not in self._atom_cache:
            name, lag = atom
            M = np.zeros((self.N, len(self.prim) * self.N))
            M[:, self.block(name)] = self.shift(lag)
            self._atom_cache[key] = M
        return self._atom_cache[key]

    def expr_op(self, expr: Dict[Atom, float]) -> np.ndarray:
        M = np.zeros((self.N, len(self.prim) * self.N))
        for atom, c in expr.items():
            M[:, self.block(atom[0])] += c * self.shift(atom[1])
        return M

    # ------------------------------------------------------- closed loop
    def row_seen(self, agent: str, r: int, excluded: set) -> Tuple[np.ndarray, Dict[str, List[Tuple[float, float]]]]:
        """Regular part of row r as seen by the agent (delayed), as an operator on the
        primary vector, and the instantaneous entries per source: channel names for
        Brownian noise, control names for observed-control impulses.  Controls in
        `excluded` (the agent whose reaction is switched off) contribute impulses
        through their own impulse channel instead of through a kernel."""
        name, drift, E, delay = self.rows[agent][r]
        S = self.shift(delay)
        regular = np.zeros((self.N, len(self.prim) * self.N))
        deltas: Dict[str, List[Tuple[float, float]]] = {}      # source -> [(age, weight)]
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
            else:
                regular[:, self.block(n)] += c * (S @ self.shift(l))
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
                deltas.setdefault(ch, []).append((delay, E[k]))
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
                        for (age, w) in dl:
                            B[bl, col] += w * (self.shift(age) @ gur)
        Z = np.linalg.solve(np.eye(n) - M, B)
        return Z


# ------------------------------------------------------------------- solver
class StationarySolver:
    def __init__(self, model: Model, verbose: bool = False, naive_observers: Optional[Dict[str, List[str]]] = None):
        """naive_observers: {agent: [observers]} lists agents whose strategies do NOT react to that agent's
        deviations (Chapter 6's naive observers); every other observer is privy and reacts through its map."""
        self.c = Compiled(model)
        self.model = model
        self.verbose = verbose
        self.naive_observers = naive_observers or {}
        self._rphys: Dict[str, np.ndarray] = {}
        self._qa: Dict[str, np.ndarray] = {}
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

    # -------------------------------------------- overridable model pieces
    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """Responses of the primary kernels to a unit impulse of each of the agent's controls, with the
        agent's own reaction switched off.  Naive observers do not react to this agent."""
        naive = self.naive_observers.get(agent.name, [])
        if not naive:
            return R
        mz = {k: (np.zeros_like(v) if k in naive else v) for k, v in maps.items()}
        return self.c.closed_loop(mz, excluded=agent.name, impulse_controls=agent.controls)[:, self.c.nW:]

    def _passive_world(self, agent: Agent, maps, Zpass: np.ndarray, R: np.ndarray) -> np.ndarray:
        """The agent's passive world (its own strategy off); subclasses may replace it (see diagnostics.py)."""
        return Zpass

    # ------------------------------------------------ best-response pieces
    def _passive_rows(self, agent: Agent, Zpass: np.ndarray):
        """Seen signal rows of `agent` in its passive world: regular kernels (N, nW) per row and the
        instantaneous entries [(channel, age, weight)] per row."""
        c = self.c
        ytil, yinst = [], []
        for r in range(len(agent.signals)):
            regular, deltas = c.row_seen(agent.name, r, set(agent.controls))
            ytil.append(regular @ Zpass)
            yinst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        return ytil, yinst

    def _row_operator(self, rows, inst):
        """Per channel, the operator mapping stacked row kernels gamma (nR N) to the action kernel:
        c_k = sum_r (Conv[y_rk] + E_rk S_delta) gamma_r."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        Gk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            Gk[:, :, r * N:(r + 1) * N] += c.grid.conv_ops(rows[r])           # all channels at once
            for (k, age, w) in inst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.shift(age)
        return Gk

    def _response_operators(self, agent: Agent, R: np.ndarray):
        """Per control, the operator (n_prim N x N) giving the primary kernels' response to that
        control's action kernel: Z = Zpass + Resp_u c_u, with the own block equal to the action."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        out = []
        for ui, u in enumerate(agent.controls):
            Ru = R[:, ui].reshape(len(c.prim), N)
            Cu = c.grid.conv_ops(Ru.T).reshape(n_prim, N)            # all primaries at once
            Cu[c.block(u)] = np.eye(N)
            out.append(Cu)
        return out

    def _foc_operators(self, agent: Agent, R: np.ndarray, Rphys: np.ndarray):
        """Per control, operators (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses, delayed reads of own lagged controls.  Also the physical-only version
        (all reactions off) for the wedge decomposition."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atoms, Q, q = c.loss[agent.name]
        atom_ops = [c.atom_op(at) for at in atoms]
        if agent.name not in self._qa:                    # (Q zeta) as an operator on the primary kernels: map-independent
            AO = np.concatenate(atom_ops, axis=0)
            self._qa[agent.name] = np.kron(Q, np.eye(N)) @ AO
        QA = self._qa[agent.name]
        Fu, Fphys = [], []
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim)); op_phys = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0))
                op += QA[j0 * N:(j0 + 1) * N]; op_phys += QA[j0 * N:(j0 + 1) * N]
            if not agent.myopic:
                # impulse responses of every loss atom, batched: (N, m) for R and for Rphys
                Rj = np.stack([atom_ops[j] @ R[:, ui] for j in range(len(atoms))], axis=1)
                Rjp = np.stack([atom_ops[j] @ Rphys[:, ui] for j in range(len(atoms))], axis=1)
                CR = c.grid.corr_ops(Rj, c.rho); CRp = c.grid.corr_ops(Rjp, c.rho)
                for j, at in enumerate(atoms):
                    name, lag = at
                    Qj = QA[j * N:(j + 1) * N]
                    if name in agent.controls:
                        if name == u and lag > 0:          # delayed read of the control itself
                            op += np.exp(-c.rho * lag) * c.shift(-lag) @ Qj
                            op_phys += np.exp(-c.rho * lag) * c.shift(-lag) @ Qj
                        continue                            # own reactions: envelope
                    op += CR[j] @ Qj
                    op_phys += CRp[j] @ Qj
                    if lag < 0:
                        # a lead: flows at dates t - |lag| <= tau < t also read the quantity after t, so the
                        # derivative of the discounted objective has a further term over those past dates:
                        #   int_0^{|lag|} e^{rho v} r(|lag| - v) (Q zeta)_j(a - v) dv   (convolution with k(v))
                        for (Rsrc, target) in ((R, "op"), (Rphys, "op_phys")):
                            rx = Rsrc[c.block(name), ui]
                            v = c.grid.nodes
                            kv = np.exp(c.rho * v) * (c.grid.interp(-lag - v) @ rx) * (v <= -lag + 1e-12)
                            if target == "op":
                                op += c.grid.conv_op(kv) @ Qj
                            else:
                                op_phys += c.grid.conv_op(kv) @ Qj
            Fu.append(op); Fphys.append(op_phys)
        return Fu, Fphys

    def _projection_operator(self, rows, inst):
        """H (nR N x nW N): E[phi_t dY_r(t-b)] for every row r and age b, from the FOC kernels."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            ops = c.grid.corr_ops(rows[r], 0.0)                                  # (nW, N, N)
            H[r * N:(r + 1) * N] += ops.transpose(1, 0, 2).reshape(N, nW * N)
            for (k, age, w) in inst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.shift(-age)
        return H

    def _project_maps(self, agent: Agent, Z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Raw maps (nU, nR, N) reproducing the action kernels `actions` (nU, N, nW) on the agent's
        closed-loop seen rows, by weighted least squares."""
        c = self.c; N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        rows, inst = [], []
        for r in range(nR):
            regular, deltas = c.row_seen(agent.name, r, set())
            rows.append(regular @ Z)
            inst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        Bk = self._row_operator(rows, inst)
        W = c.grid.mass
        Gram = sum((Bk[k] * W[:, None]).T @ Bk[k] for k in range(nW))
        Gram += 1e-14 * np.trace(Gram) / Gram.shape[0] * np.eye(nR * N)
        g = np.zeros((nU, nR, N))
        for ui in range(nU):
            rhs = sum((Bk[k] * W[:, None]).T @ actions[ui, :, k] for k in range(nW))
            g[ui] = np.linalg.solve(Gram, rhs).reshape(nR, N)
        return g

    # ------------------------------------------------------ best response
    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        c = self.c; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        # passive world (own strategy off) and impulse responses (own reaction off)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]
        R = self._impulse_responses(agent, maps, R)
        Zpass = self._passive_world(agent, maps, Zpass, R)
        if agent.name not in self._rphys:                                   # map-independent: cache
            self._rphys[agent.name] = c.closed_loop(self.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, nW:]
        Rphys = self._rphys[agent.name]
        # operators: gamma -> action, action -> world, world -> FOC, FOC -> projection
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(ytil, yinst)
        Resp = self._response_operators(agent, R)
        Fu, Fphys = self._foc_operators(agent, R, Rphys)
        H = self._projection_operator(ytil, yinst)
        # the FOC is affine in gamma: solve H (Fu (Zpass + sum_v Resp_v Gk gamma_v)) = 0 for all controls
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        for ui in range(nU):
            rowsl = slice(ui * nR * N, (ui + 1) * nR * N)
            for vi in range(nU):
                FR = Fu[ui] @ Resp[vi]                                       # channel-independent factor
                colsl = slice(vi * nR * N, (vi + 1) * nR * N)
                FRG = (FR @ Gk).reshape(nW * N, nR * N)                      # FR applied per channel, stacked
                Amat[rowsl, colsl] += H @ FRG                                # one product over all channels
            bvec[rowsl] += H @ (Fu[ui] @ Zpass).T.reshape(-1)
        gamma = np.linalg.solve(Amat, -bvec).reshape(nU, nR, N)
        cact = np.zeros((nU, N, nW))
        for ui in range(nU):
            for k in range(nW):
                cact[ui, :, k] = Gk[k] @ gamma[ui].reshape(-1)
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp[ui] @ cact[ui]
        g = self._project_maps(agent, Zfull, cact)
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            dec = {}
            for ui, u in enumerate(agent.controls):
                phi = np.stack([Fu[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                phi_phys = np.stack([Fphys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
            out["decomp"] = dec
        return g, out

    # ------------------------------------------------ maps from kernels
    def world_from_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Closed-loop primary kernels (n_prim N, nW) when every agent's action kernels (nU, N, nW) are
        given: the states follow from the propagator, no strategy maps needed."""
        c = self.c; N, nW = c.N, c.nW
        Z = np.zeros((len(c.prim) * N, nW))
        for a in self.model.agents:
            for ui, u in enumerate(a.controls):
                Z[c.block(u)] = actions[a.name][ui]
        if c.nX:
            perm = np.arange(c.nX * N).reshape(N, c.nX).T.reshape(-1)
            PX, P0X = c.Pin[perm], c.P0[perm]
            U = np.zeros((c.nX * N, nW))                       # inputs from controls
            Lx = np.zeros((c.nX * N, c.nX * N))                # inputs from lagged states (linear in X)
            for i, (nm, lag), coef in c.state_inputs:
                if nm in c.model.state_names:
                    j = c.model.state_names.index(nm)
                    Lx[i::c.nX, j * N:(j + 1) * N] += coef * c.shift(lag)     # input (node, comp i) <- X block j
                else:
                    U[i::c.nX] += coef * (c.shift(lag) @ Z[c.block(nm)])
            rhs = PX @ U + P0X @ c.sigma
            X = np.linalg.solve(np.eye(c.nX * N) - PX @ Lx, rhs) if Lx.any() else rhs
            Z[:c.nX * N] = X
        return Z

    def response_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        maps = self.maps_from_kernels(self.world_from_actions(actions))
        new = {}
        for a in self.model.agents:
            if self.c.rep[a.name] == a.name:
                new[a.name] = self.best_response(a, maps)[1]["action"]
        for a in self.model.agents:
            if a.name not in new:
                new[a.name] = new[self.c.rep[a.name]]
        return new

    def maps_from_kernels(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Raw maps that reproduce given closed-loop primary kernels Z (n_prim N, nW); tied agents
        share the representative's projection."""
        maps = {}
        for a in self.model.agents:
            if self.c.rep[a.name] == a.name:
                maps[a.name] = self._project_maps(a, Z, np.stack([Z[self.c.block(u)] for u in a.controls]))
        for a in self.model.agents:
            if a.name not in maps:
                maps[a.name] = maps[self.c.rep[a.name]]
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
              pre_iterations: int = 20, max_newton: int = 60, pre_tol: float = 1e-3, method: str = "anderson",
              variable: str = "actions") -> Result:
        """method: "newton" (damped pre-phase, then Newton-Krylov) or "anderson" (regularised Anderson,
        Newton polish).  variable: "maps" iterates on the raw strategies; "actions" on the action
        kernels (raw maps by projection), which is better conditioned when strategies are weakly
        identified.  init: maps or actions accordingly."""
        t0 = time.time()
        hist, evals = [], [0]
        shapes_a = {a.name: (len(a.controls), self.c.N, self.c.nW) for a in self.model.agents}
        reps = self.c.reps

        def packa(acts):
            return np.concatenate([acts[n].reshape(-1) for n in reps])

        def unpacka(zz):
            acts, pos = {}, 0
            for n in reps:
                size = int(np.prod(shapes_a[n])); acts[n] = zz[pos:pos + size].reshape(shapes_a[n]); pos += size
            for a in self.model.agents:
                if a.name not in acts:
                    acts[a.name] = acts[self.c.rep[a.name]]
            return acts
        if variable == "actions" and self.model.ties:
            # tied agents' action kernels differ by a channel permutation the symmetry implies; raw maps
            # (on each agent's own rows) carry over verbatim, so iterate on maps when ties are present
            variable = "maps"
        if variable == "actions":
            if init is not None and all(v.ndim == 3 and v.shape[1] == self.c.N and v.shape[2] == self.c.nW for v in init.values()):
                acts0 = init
            elif init is not None:                       # maps given: convert to actions
                Z0 = self.c.closed_loop(init); acts0 = {a.name: np.stack([Z0[self.c.block(u)] for u in a.controls]) for a in self.model.agents}
            else:
                acts0 = {a.name: np.zeros(shapes_a[a.name]) for a in self.model.agents}
            z = packa(acts0)

            def F(zz):
                evals[0] += 1
                return packa(self.response_actions(unpacka(zz))) - zz
        else:
            maps = init if init is not None else self.zero_maps()
            if init is not None and all(v.ndim == 3 and v.shape[2] == self.c.nW for v in init.values()):
                maps = self.maps_from_kernels(self.world_from_actions(init))      # actions given: project
            z = self.pack(maps)

            def F(zz):
                evals[0] += 1
                return self.pack(self.response_map(self.unpack(zz))) - zz

        z, resid, nev, converged, message = solve_fixed_point(F, z, tol=tol, verbose=self.verbose, damping=damping,
                                                     max_newton=max_newton, method=method, pre_iterations=pre_iterations, pre_tol=pre_tol)
        hist.append(resid)
        maps = self.maps_from_kernels(self.world_from_actions(unpacka(z))) if variable == "actions" else self.unpack(z)
        Z = self.c.closed_loop(maps)
        res = Result(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                     iterations=evals[0], seconds=time.time() - t0, history=hist, message=message)
        # decomposition, costs, and how well the raw rows represent each best response
        for a in self.model.agents:
            g, out = self.best_response(a, maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            res.costs[a.name] = self.expected_loss(a, Z)
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)
        return res

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        """Relative residual of the best-response action kernels after projection on the agent's raw
        rows.  Zero in exact arithmetic; on the grid it measures how well products of kernels are
        resolved, so a value above about 1e-6 means the equilibrium is under-resolved: raise
        horizon.nodes."""
        c = self.c; nW = c.nW; nR = len(agent.signals)
        rows, inst = [], []
        for r in range(nR):
            reg, deltas = c.row_seen(agent.name, r, set())
            rows.append(reg @ Zfull)
            inst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        Bk = self._row_operator(rows, inst)
        worst = 0.0
        for ui in range(len(agent.controls)):
            recon = np.stack([Bk[k] @ g[ui].reshape(-1) for k in range(nW)], axis=1)
            worst = max(worst, float(np.abs(recon - actions[ui]).max() / max(1e-300, np.abs(actions[ui]).max())))
        return worst

    def expected_loss(self, agent: Agent, Z: np.ndarray) -> float:
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])          # (m, N, nW)
        M = c.grid.mass_matrix                                        # exact <l_i, l_j>
        G = np.einsum("ink,nm,jmk->ij", zeta, M, zeta)                # <zeta_i, zeta_j> over ages and channels
        return float(0.5 * np.sum(Q * G))
