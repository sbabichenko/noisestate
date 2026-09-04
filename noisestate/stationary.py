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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid import AgeGrid
from .grid_cache import age_grid
from .engine import EngineBase
from .compile import CompiledBase
from .results import StationaryResult
from .spec import Agent, Atom, Model


class Compiled(CompiledBase):
    """Grid, index maps and constant operators for a stationary model."""

    def __init__(self, model: Model):
        super().__init__(model)
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
                raise ValueError(f"lag {l} is not a panel breakpoint {[round(b, 6) for b in bp]}; set horizon.unit so every lag is a "
                                 f"multiple of it, and horizon.unit_range at least {max(lags)} so the unit panels reach the largest lag")
        self.grid = age_grid(tuple(round(float(b), 12) for b in bp), hz.nodes)   # shared, with its operator caches
        self.N = self.grid.N
        self.rho = float(hz.discount)
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

    row = row_seen                                        # the engines' common name

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
class StationarySolver(EngineBase):
    RESULT = StationaryResult
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.3, 60       # 0.3: Kyle-Back converges, 0.5 does not
    # the objective is truncated at the window, so a strategy can push a little loss past the edge: curvatures
    # within this fraction of the largest are treated as that truncation, not as a saddle
    SECOND_ORDER_TOL = 1e-4

    def __init__(self, model: Model, verbose: bool = False, naive_observers: Optional[Dict[str, List[str]]] = None):
        self.solver_kw = {"verbose": verbose, "naive_observers": naive_observers}
        """naive_observers: {agent: [observers]} lists agents whose strategies do NOT react to that agent's
        deviations (Chapter 6's naive observers); every other observer is privy and reacts through its map."""
        self.c = Compiled(model)
        self.model = model
        self.verbose = verbose
        self.naive_observers = naive_observers or {}
        names = [a.name for a in model.agents]
        if not isinstance(self.naive_observers, dict):
            raise TypeError("naive_observers must be a dict {agent: [observers that do not react to it]}")
        for k, v in self.naive_observers.items():
            if k not in names:
                raise ValueError(f"naive_observers: {k!r} is not an agent (agents: {names})")
            if isinstance(v, str) or not all(isinstance(x, str) for x in v):
                raise TypeError(f"naive_observers[{k!r}] must be a list of agent names")
            bad = [x for x in v if x not in names]
            if bad:
                raise ValueError(f"naive_observers[{k!r}]: {bad} are not agents (agents: {names})")
            if k in v:
                raise ValueError(f"naive_observers[{k!r}] lists the agent itself")
        self._rphys: Dict[str, np.ndarray] = {}
        self._qa: Dict[str, np.ndarray] = {}
        c = self.c
        self.shapes = {a.name: (len(a.controls), len(a.signals), c.N) for a in model.agents}

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
    def _row_operator(self, agent: Agent, rows, inst):
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

    def _foc_operators(self, agent: Agent, R: np.ndarray):
        """Per control, the operator (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses R, delayed reads of own lagged controls, and the past-date term of a lead.
        Called with the physical impulse responses (all reactions off) for the wedge decomposition."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atoms, Q, q = c.loss[agent.name]
        atom_ops = [c.atom_op(at) for at in atoms]
        if agent.name not in self._qa:                    # (Q zeta) as an operator on the primary kernels: map-independent
            AO = np.concatenate(atom_ops, axis=0)
            self._qa[agent.name] = np.kron(Q, np.eye(N)) @ AO
        QA = self._qa[agent.name]
        Fu = []
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0))
                op += QA[j0 * N:(j0 + 1) * N]
            if not agent.myopic:
                Rj = np.stack([atom_ops[j] @ R[:, ui] for j in range(len(atoms))], axis=1)   # impulse responses of every atom
                CR = c.grid.corr_ops(Rj, c.rho)
                for j, (name, lag) in enumerate(atoms):
                    Qj = QA[j * N:(j + 1) * N]
                    if name in agent.controls:
                        if name == u and lag > 0:          # delayed read of the control itself
                            op += np.exp(-c.rho * lag) * c.shift(-lag) @ Qj
                        continue                            # own reactions: envelope
                    op += CR[j] @ Qj
                    if lag < 0:
                        # a lead: flows at dates t - |lag| <= tau < t also read the quantity after t, so the
                        # derivative of the discounted objective has a further term over those past dates:
                        #   int_0^{|lag|} e^{rho v} r(|lag| - v) (Q zeta)_j(a - v) dv   (convolution with k(v))
                        v = c.grid.nodes
                        kv = np.exp(c.rho * v) * (c.grid.interp(-lag - v) @ R[c.block(name), ui]) * (v <= -lag + 1e-12)
                        op += c.grid.conv_op(kv) @ Qj
            Fu.append(op)
        return Fu

    def _projection_operator(self, agent: Agent, rows, inst):
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
        rows, inst = self._seen_rows(agent, Z, set())
        Bk = self._row_operator(agent, rows, inst)
        W = c.grid.mass
        keep = self._identified(agent)                                       # delayed rows: zero where they read nothing
        Gram = sum((Bk[k] * W[:, None]).T @ Bk[k] for k in range(nW))[np.ix_(keep, keep)]
        Gram += 1e-14 * np.trace(Gram) / Gram.shape[0] * np.eye(Gram.shape[0])
        g = np.zeros((nU, nR * N))
        for ui in range(nU):
            rhs = sum((Bk[k] * W[:, None]).T @ actions[ui, :, k] for k in range(nW))[keep]
            g[ui, keep] = np.linalg.solve(Gram, rhs)
        return g.reshape(nU, nR, N)

    def _identified(self, agent: Agent) -> np.ndarray:
        """Mask over the stacked map nodes (row-major over rows) of the ages at which the map on each
        row reads something within the window: all ages for an undelayed row, ages below L - delay
        for a row observed with a delay."""
        c = self.c; N = c.N
        keep = np.ones(len(agent.signals) * N, dtype=bool)
        for r in range(len(agent.signals)):
            d = c.rows[agent.name][r][3]
            if d > 0:
                keep[r * N:(r + 1) * N] = c.grid.nodes < c.grid.L - d - 1e-12
        return keep

    # ------------------------------------------------------ best response
    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        c = self.c; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        # passive world (own strategy off) and impulse responses (own reaction off)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]
        R = self._impulse_responses(agent, maps, R)
        Zpass = self._passive_world(agent, maps, Zpass, R)
        # operators: gamma -> action, action -> world, world -> FOC, FOC -> projection
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(agent, ytil, yinst)
        Resp = self._response_operators(agent, R)
        Fu = self._foc_operators(agent, R)
        H = self._projection_operator(agent, ytil, yinst)
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
        # a row observed with delay d is uninformative about shocks younger than d, so the map on it at
        # ages b > L - d reads nothing within the window: those unknowns (and their projection rows,
        # which are zero) are removed, and the map is zero there.  A plain solve on the full system
        # would be singular.
        keep = np.tile(self._identified(agent), nU)
        gamma = np.zeros(nG)
        if max(c.rows[agent.name][r][3] for r in range(nR)):
            # with a delayed row the seen kernels jump at the delay, and the map values at the panel
            # breakpoints (one of each duplicated node) drop out of the discrete system exactly; a few
            # more modes sit within 1e-8 of zero relative to the largest.  Least squares with that cutoff
            # takes the minimum-norm values there; the equilibrium is insensitive to the cutoff (costs move
            # by 1e-7 between 1e-9 and 1e-6 on the Chapter 3 game with one delayed row, window 6)
            gamma[keep] = np.linalg.lstsq(Amat[np.ix_(keep, keep)], -bvec[keep], rcond=1e-8)[0]
        else:
            try:
                gamma[keep] = np.linalg.solve(Amat[np.ix_(keep, keep)], -bvec[keep])
            except np.linalg.LinAlgError:
                raise ValueError(f"the best-response system of {agent.name} is singular: two of its rows may carry the "
                                 "same information, a control may have no quadratic term in its current value (a "
                                 "quadratic in a lagged read, D@tau, leaves the strategy free at ages above "
                                 "window - tau), or a row's noise loading may be zero") from None
        gamma = gamma.reshape(nU, nR, N)
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
            out["second_order"] = self._second_order(agent, Resp, Gk, keep)
            if agent.name not in self._rphys:                               # physical impulse responses: map-independent
                self._rphys[agent.name] = c.closed_loop(self.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, nW:]
            Fphys = self._foc_operators(agent, self._rphys[agent.name])
            dec = {}
            for ui, u in enumerate(agent.controls):
                phi = np.stack([Fu[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                phi_phys = np.stack([Fphys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
                dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
            out["decomp"] = dec
        return g, out

    def _second_order(self, agent: Agent, Resp, Gk, keep) -> Optional[dict]:
        """Second-order condition of the best response: the agent's objective is a quadratic form in its
        strategy, and a first-order condition is a minimum only if that form is positive on the
        feasible strategies (those its rows can express).  With rho = 0 the objective is the flow
        loss, integrated exactly, so the form is computed exactly: J(delta) = 1/2 delta' M delta with
        M = T' G T, T the map from a strategy to the world it produces and G the loss form.  Its
        extreme eigenvalues come from Lanczos on matvecs.  Returns {"min", "max", "ok"} with
        min/max the eigenvalues of M scaled by max; None when rho > 0 (the discounted objective is
        not a quadratic form in the stationary kernel)."""
        c = self.c
        if c.rho > 0:
            return None
        from scipy.sparse.linalg import LinearOperator, eigsh
        N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        atoms, Q, q = c.loss[agent.name]
        AO = np.concatenate([c.atom_op(at) for at in atoms], axis=0)           # (m N, n_prim N)
        QM = np.kron(Q, c.grid.mass_matrix)                                      # the loss form on the atoms
        GAO = AO.T @ (QM @ AO)                                                   # symmetric loss form on the world
        idx = np.where(keep)[0]

        def T(delta_full):                     # strategy -> world, per channel: (n_prim N, nW)
            Zd = np.zeros((GAO.shape[0], nW))
            for ui in range(nU):
                du = delta_full[ui * nR * N:(ui + 1) * nR * N]
                for k in range(nW):
                    Zd[:, k] += Resp[ui] @ (Gk[k] @ du)
            return Zd

        def Tt(Zd):                            # its transpose
            out = np.zeros(nU * nR * N)
            for ui in range(nU):
                RZ = Resp[ui].T @ Zd                                            # (N, nW)
                for k in range(nW):
                    out[ui * nR * N:(ui + 1) * nR * N] += Gk[k].T @ RZ[:, k]
            return out

        def matvec(v):
            full = np.zeros(nU * nR * N); full[idx] = np.asarray(v, dtype=float).ravel()
            return Tt(GAO @ T(full))[idx]
        n = idx.size
        if n <= 400:
            Mfull = np.column_stack([matvec(e) for e in np.eye(n)])
            w = np.linalg.eigvalsh((Mfull + Mfull.T) / 2)
            lo, hi = float(w[0]), float(w[-1])
        else:
            op = LinearOperator((n, n), matvec=matvec, dtype=float)
            try:
                hi = float(eigsh(op, k=1, which="LA", tol=1e-6, maxiter=300, return_eigenvectors=False)[0])
                lo = float(eigsh(op, k=1, which="SA", tol=1e-6, maxiter=300, return_eigenvectors=False)[0])
            except Exception as exc:                          # Lanczos did not settle: say so rather than stay silent
                return {"min": None, "max": None, "ok": None, "converged": False, "message": f"{type(exc).__name__}: {exc}"[:120]}
        scale = max(abs(lo), abs(hi), 1e-300)
        return {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -self.SECOND_ORDER_TOL * scale), "converged": True}

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

    def maps_from_kernels(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Raw maps that reproduce given closed-loop primary kernels Z (n_prim N, nW); tied agents
        share the representative's projection."""
        return self._over_representatives(lambda a: self._project_maps(a, Z, np.stack([Z[self.c.block(u)] for u in a.controls])))

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.maps_from_kernels(self.world_from_actions(actions))

    # ------------------------------------------------------ fixed point
    def _finish(self, res) -> None:
        """Costs, the first-order-condition decomposition, the second-order check and the representation
        error of every agent's best response at the equilibrium."""
        for a in self.model.agents:
            g, out = self.best_response(a, res.maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            res.costs[a.name] = self.expected_cost(a, res.Z)
            if out["second_order"] is not None:
                res.second_order[a.name] = out["second_order"]
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """Stationary flow loss per unit time of the agent in the world Z (exact Gram quadrature)."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])          # (m, N, nW)
        M = c.grid.mass_matrix                                        # exact <l_i, l_j>
        G = np.einsum("ink,nm,jmk->ij", zeta, M, zeta)                # <zeta_i, zeta_j> over ages and channels
        return float(0.5 * np.sum(Q * G))

    expected_loss = expected_cost                       # the older name
