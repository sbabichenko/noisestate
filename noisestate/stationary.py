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
from scipy.linalg import lu_factor, lu_solve

from .grid import AgeGrid
from .grid_cache import age_grid
from .engine import EngineBase
from .compile import CompiledBase, close_under_delays
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
        # the map on a row observed with delay d is read by the action at age b + d: for the map's panels to be
        # the action's panels shifted by d (the instantaneous entry node to node, the map's window edge L - d a
        # panel edge, no map mode the action cannot see) the breakpoints are closed under subtraction of every
        # row delay.  With geometric panels beyond unit_range this makes the panels uniform.
        delays = sorted({float(r[3]) for rr in self.rows.values() for r in rr if r[3] > 0})
        if delays:
            bp = close_under_delays(bp, delays)
        self.grid = age_grid(tuple(round(float(b), 12) for b in bp), hz.nodes)   # shared, with its operator caches
        self.N = self.grid.N
        self.rho = float(hz.discount)
        self._atom_cache: Dict[tuple, np.ndarray] = {}
        self._elim: Dict[frozenset, tuple] = {}
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
    def row_blocks(self, agent: str, r: int, excluded: set):
        """Regular part of row r as seen by the agent (delayed), as (N x N) operators on the primary
        kernels it reads, {primary: operator}, and the instantaneous entries per source: channel
        names for Brownian noise, control names for observed-control impulses.  Controls in
        `excluded` (the agent whose reaction is switched off) contribute impulses through their own
        impulse channel instead of through a kernel."""
        name, drift, E, delay = self.rows[agent][r]
        S = self.shift(delay)
        blocks: Dict[str, np.ndarray] = {}
        deltas: Dict[str, List[Tuple[float, float]]] = {}      # source -> [(age, weight)]
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
            else:
                op = c * (S @ self.shift(l)) if l else c * S
                blocks[n] = blocks[n] + op if n in blocks else op
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
                deltas.setdefault(ch, []).append((delay, E[k]))
        return blocks, deltas

    def row_seen(self, agent: str, r: int, excluded: set) -> Tuple[np.ndarray, Dict[str, List[Tuple[float, float]]]]:
        """row_blocks assembled as one operator (N x n_prim N) on the primary vector."""
        blocks, deltas = self.row_blocks(agent, r, excluded)
        regular = np.zeros((self.N, len(self.prim) * self.N))
        for n, op in blocks.items():
            regular[:, self.block(n)] += op
        return regular, deltas

    row = row_seen                                        # the engines' common name

    # ------------------------------------------------ kernel algebra (see EngineBase)
    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        return self.grid.conv_ops(Y)                      # the seen row is already shifted by the delay

    def instant(self, age: float) -> np.ndarray:
        return self.shift(age)

    def instant_adjoint(self, age: float) -> np.ndarray:
        return self.shift(-age)

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        return self.grid.conv_ops(Ru.T).reshape(len(self.prim) * self.N, self.N)      # all primaries at once

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        return self.grid.corr_ops(Rj, self.rho)

    def own_lag_read(self, lag: float) -> np.ndarray:
        return self.shift(-lag)

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        return self.grid.corr_ops(Y, 0.0).transpose(1, 0, 2).reshape(self.N, self.nW * self.N)

    def cost_mass(self) -> np.ndarray:
        """The Gram matrix under which expected_cost integrates products of kernels."""
        return self.grid.mass_matrix

    def _state_elimination(self, excl: frozenset):
        """The map-independent part of the closed loop, per set of excluded controls: with the states
        eliminated, Z_X = W Z_U + G B_X where G = (I - P U_X)^{-1} carries the lagged-state feedback
        and W = G P U_U the controls' effect on the states.  Cached: only the control rows of the
        closed loop depend on the strategies, so each solve is of size n_controls N, not n_prim N."""
        key = excl
        if key not in self._elim:
            nX, N, n = self.nX, self.N, len(self.prim) * self.N
            U = np.zeros((nX * N, n))                     # input to the propagator, (node, comp) ordering
            for i, (nm, lag), c in self.state_inputs:
                if nm in excl:
                    continue
                U[i::nX, :] += c * self.atom_op((nm, lag))
            perm = np.arange(nX * N).reshape(N, nX).T.reshape(-1)     # prim index -> (node, comp) index
            PX, P0X = self.Pin[perm], self.P0[perm]
            PU = PX @ U
            lu = lu_factor(np.eye(nX * N) - PU[:, :nX * N])
            W = lu_solve(lu, PU[:, nX * N:])
            self._elim[key] = (lu, W, P0X, perm)
        return self._elim[key]

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                    impulse_controls: Sequence = ()):
        """Solve the closed loop for the Brownian channels and for unit impulses in
        `impulse_controls` (whose owning agent's reactions are switched off when it is
        `excluded`).  maps[agent] has shape (n_controls, n_rows, N).
        Returns Z with shape (n_prim N, nW + len(impulse_controls)).  The states are eliminated
        through the cached propagator part (see _state_elimination); the solve is over the controls."""
        nX, nU, N = self.nX, self.nU, self.N
        n = len(self.prim) * N; nxs = nX * N
        ncol = self.nW + len(impulse_controls)
        excl = frozenset(self.model.agents[[a.name for a in self.model.agents].index(excluded)].controls) if excluded else frozenset()
        B = np.zeros((n, ncol))
        if nX:
            lu, W, P0X, perm = self._state_elimination(excl)
            for k in range(self.nW):
                B[:nxs, k] += P0X @ self.sigma[:, k]
            for j, u in enumerate(impulse_controls):
                col = self.nW + j
                for i, (nm, lag), c in self.state_inputs:
                    if nm != u:
                        continue
                    v = np.zeros(nX); v[i] = c
                    B[:nxs, col] += (P0X @ v) if lag == 0 else (self.grid.jump_injector(self.A, lag)[perm] @ v)
            GB = lu_solve(lu, B[:nxs])
        # control rows: the strategies
        MU = np.zeros((nU * N, n))
        for a in self.model.agents:
            if a.name == excluded:
                continue
            g = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = slice(self.block(u).start - nxs, self.block(u).stop - nxs)
                for r in range(len(a.signals)):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gur = g[ui, r]
                    C = self.grid.conv_op_left(gur)
                    for nm, op in blocks.items():                       # only the primaries the row reads
                        MU[bl, self.block(nm)] += C @ op
                    for src, dl in deltas.items():
                        if src in self.channels:
                            col = self.channels.index(src)
                        elif src in impulse_controls:
                            col = self.nW + list(impulse_controls).index(src)
                        else:
                            continue
                        for (age, w) in dl:
                            B[nxs + bl.start:nxs + bl.stop, col] += w * (self.shift(age) @ gur)
        Z = np.zeros((n, ncol))
        if nX:
            MUX, MUU = MU[:, :nxs], MU[:, nxs:]
            Z[nxs:] = np.linalg.solve(np.eye(nU * N) - MUU - MUX @ W, B[nxs:] + MUX @ GB)
            Z[:nxs] = W @ Z[nxs:] + GB
        else:
            Z[:] = np.linalg.solve(np.eye(nU * N) - MU, B)
        return Z

    def closed_loop_dense(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                          impulse_controls: Sequence = ()):
        """closed_loop as one dense solve over every primary kernel (the reference for the elimination; tests)."""
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
    SECOND_ORDER_QUADRATIC = False   # the flow loss is a quadratic form in the stationary strategy only at rho = 0

    def __init__(self, model: Model, verbose: bool = False, naive_observers: Optional[Dict[str, List[str]]] = None):
        """naive_observers: {agent: [observers]} lists agents whose strategies do NOT react to that agent's
        deviations (Chapter 6's naive observers); every other observer is privy and reacts through its map."""
        super().__init__(model, verbose, naive_observers=naive_observers)
        self.c = Compiled(model)
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
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    # -------------------------------------------- overridable model pieces
    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """Responses of the primary kernels to a unit impulse of each of the agent's controls, with the
        agent's own reaction switched off.  Naive observers do not react to this agent."""
        naive = self.naive_observers.get(agent.name, [])
        if not naive:
            return R
        mz = {k: (np.zeros_like(v) if k in naive else v) for k, v in maps.items()}
        return self.c.closed_loop(mz, excluded=agent.name, impulse_controls=agent.controls)[:, self.c.nW:]

    def _lead_term(self, agent: Agent, Ru: np.ndarray, name: str, lag: float) -> np.ndarray:
        # flows at dates t - |lag| <= tau < t also read the quantity after t, so the derivative of the
        # discounted objective has a further term over those past dates:
        #   int_0^{|lag|} e^{rho v} r(|lag| - v) (Q zeta)_j(a - v) dv   (a convolution with k(v))
        c = self.c; v = c.grid.nodes
        kv = np.exp(c.rho * v) * (c.grid.interp(-lag - v) @ Ru[c.block(name)]) * (v <= -lag + 1e-12)
        return c.grid.conv_op(kv)

    def _project(self, agent: Agent, Z: np.ndarray, actions: np.ndarray) -> np.ndarray:
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
        c = self.c; N = c.N; g = c.grid
        keep = np.ones(len(agent.signals) * N, dtype=bool)
        last = (np.arange(N) % g.n) == g.n - 1
        for r in range(len(agent.signals)):
            d = c.rows[agent.name][r][3]
            if d > 0:          # ages below L - d, and the lower copy of the node at L - d (the action at age L reads it)
                edge = g.L - d
                keep[r * N:(r + 1) * N] = (g.nodes < edge - 1e-12) | (last & (np.abs(g.nodes - edge) <= 1e-12))
        return keep

    # ------------------------------------------------------ best response
    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        # a row observed with delay d is uninformative about shocks younger than d, so the map on it at
        # ages b > L - d reads nothing within the window: those unknowns (and their projection rows,
        # which are zero) are removed, and the map is zero there.  A plain solve on the full system
        # would be singular.
        keep = np.tile(self._identified(agent), len(agent.controls))
        gamma = np.zeros(Amat.shape[0])
        # with a delayed row the unknowns removed by `keep` have exactly zero rows and columns (the seen
        # row is zero below the delay, the lead reads nothing beyond the window), so the reduced system
        # is as well conditioned as an undelayed one and is solved directly
        try:
            gamma[keep] = np.linalg.solve(Amat[np.ix_(keep, keep)], -bvec[keep])
        except np.linalg.LinAlgError:
            raise ValueError(f"the best-response system of {agent.name} is singular: two of its rows may carry the "
                             "same information, a control may have no quadratic term in its current value (a "
                             "quadratic in a lagged read, D@tau, leaves the strategy free at ages above "
                             "window - tau), or a row's noise loading may be zero") from None
        return gamma

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
        return self._over_representatives(lambda a: self._project(a, Z, np.stack([Z[self.c.block(u)] for u in a.controls])))

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.maps_from_kernels(self.world_from_actions(actions))

    # ------------------------------------------------------ fixed point
    def _finish(self, res) -> None:
        """Costs, the first-order-condition decomposition, the second-order check and the representation
        error of every agent's best response at the equilibrium."""
        self._second_order_cache.clear()                               # the equilibrium's own check, not a stale one
        order = [a for a in self.model.agents if self.c.rep[a.name] == a.name] + [a for a in self.model.agents if self.c.rep[a.name] != a.name]
        for a in order:
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
        G = np.einsum("ink,nm,jmk->ij", zeta, c.cost_mass(), zeta)   # <zeta_i, zeta_j> over ages and channels
        return float(0.5 * np.sum(Q * G))

    expected_loss = expected_cost                       # the older name
