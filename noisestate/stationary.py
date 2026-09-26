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

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.linalg import lu_factor, lu_solve

from .grid import AgeGrid
from .grid_cache import age_grid
from .engine import EngineBase, _is_eye, dense_curvature_form, singular_system_message, symmetrize
from .compile import CompiledBase, close_under_delays
from .symmetry import find_cyclic_symmetry
from .results import StationaryResult
from ._settings import Settings, tunable
from .spec import Agent, Atom, Model




class Compiled(CompiledBase):
    """Grid, index maps and constant operators for a stationary model."""
    LEAD_WEIGHT_WARN = tunable("lead_weight_warn")     # warn when a lead's past flows outweigh the current one by more than this (settings)

    def __init__(self, model: Model, settings=None):
        super().__init__(model)
        self.settings = Settings.of(settings)
        hz = model.horizon
        lags = model.all_lags()
        if model.numerics.breakpoints:
            bp = list(model.numerics.breakpoints)
        elif lags or model.numerics.unit:
            bp = AgeGrid.breakpoints_from_delays(hz.extent, lags, model.numerics.unit, model.numerics.unit_range)
        else:
            bp = [0.0, hz.extent]
        for l in lags:
            if not any(abs(l - b) < 1e-12 for b in bp):
                raise ValueError(f"lag {l} is not a panel breakpoint {[round(b, 6) for b in bp]}; set numerics.unit so every lag is a "
                                 f"multiple of it, and numerics.unit_range at least {max(lags)} so the unit panels reach the largest lag")
        # the map on a row observed with delay d is read by the action at age b + d: for the map's panels to be
        # the action's panels shifted by d (the instantaneous entry node to node, the map's window edge L - d a
        # panel edge, no map mode the action cannot see) the breakpoints are closed under subtraction of every
        # row delay.  With geometric panels beyond unit_range this makes the panels uniform.
        delays = sorted({float(r[3]) for rr in self.rows.values() for r in rr if r[3] > 0})
        if delays:
            bp = close_under_delays(bp, delays)
        self.grid = age_grid(tuple(round(float(b), 12) for b in bp), model.numerics.nodes)   # shared, with its operator caches
        self.N = self.grid.N
        self.rho = float(hz.discount)
        if self.rho > 0:
            leads = {(n, -l) for a in model.agents for term in a.loss for atom in term[1:] for (n, l) in model.expand({atom: 1.0}) if l < 0}
            for n, tau in sorted(leads):
                if self.rho * tau > np.log(self.LEAD_WEIGHT_WARN):
                    warnings.warn(f"lead {n}@-{tau:g} under the discount rate {self.rho:g}: the flows before t that read the quantity "
                                  f"after t enter the first-order condition weighted by up to exp(rho tau) = {np.exp(self.rho * tau):.1e} "
                                  "relative to the current flow, which dominates the best-response system (README, Limits)", stacklevel=2)
        self._elim: Dict[frozenset, tuple] = {}
        self._elim_wzero: Dict[frozenset, bool] = {}
        self.sym = find_cyclic_symmetry(model)
        self._modes = None
        self.P0, self.Pin = self.grid.propagator(self.A) if self.nX else (np.zeros((0, 0)), np.zeros((0, 0)))

    # ------------------------------------------------------------- operators
    def _add_point_columns(self, B, rows, deltas, gur, impulse_controls) -> None:
        """A row's POINT observations, added to the forcing columns of the closed-loop system.

        `deltas` is {source: [(age, weight)]}: what the row sees as an impulse rather than through a
        kernel.  A source is a Brownian channel (its own column) or an impulse control (a column
        AFTER the channels, at nW + its position), and anything else is not forced here.

        Written out three times in this class -- in the dense, per-panel and symmetric closed loops --
        which is three copies of that column mapping, where the nW offset is the easy thing to get
        wrong.  The row selector is what actually differed between them, so it is the argument.
        """
        for src, dl in deltas.items():
            if src in self.channels:
                col = self.channels.index(src)
            elif src in impulse_controls:
                col = self.nW + list(impulse_controls).index(src)
            else:
                continue
            for (age, w) in dl:
                B[rows, col] += w * (self.shift(age) @ gur)

    def shift(self, tau: float) -> np.ndarray:
        return self.grid.shift_cached(tau)

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_block(self, atom: Atom):
        """(primary index, the N x N block): where atom_op's one nonzero block sits and what it is.  The
        position is the atom's own primary, so nothing has to look for it."""
        return self.index[atom[0]], self.shift(atom[1])

    def atom_op(self, atom: Atom) -> np.ndarray:
        """N x (n_prim N) matrix giving the kernel of `name@lag` from the primary vector.  Not cached, and the
        engine's own code uses atom_block or shift and block instead: the matrix is one N x N block in zeros
        (48 MB per compiled model for Chapter 5's market, held as long as any result)."""
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

    # ------------------------------------------------ kernel algebra (algebra.KernelAlgebra)
    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(m, N, N) convolution operators of the m seen row kernels Y (N, m): map on the row -> action kernel.
        The seen row is already shifted by the delay, so `delay` is not used here."""
        return self.grid.conv_ops(Y)                      # the seen row is already shifted by the delay

    def instant(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N) shift by `age`: the action's read of the map at the age of an instantaneous entry."""
        return self.shift(age)

    def instant_adjoint(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N) shift by -age, the adjoint of instant on the FOC kernel."""
        return self.shift(-age)

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        """(n_prim N, N) convolution with every primary's impulse response Ru (n_prim, N): action -> world.
        The own block is included as computed; the base overwrites it with the identity."""
        return self.grid.conv_ops(Ru.T).reshape(len(self.prim) * self.N, self.N)      # all primaries at once

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        """(m, N, N) discounted correlation operators of the m atom responses Rj (N, m) at the rate rho."""
        return self.grid.corr_ops(Rj, self.rho)

    def own_lag_read(self, lag: float) -> np.ndarray:
        """(N, N) shift by -lag: the FOC term of the control's own read `lag` later."""
        return self.shift(-lag)

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(N, m N), columns (channel, node): the undiscounted correlation of the FOC kernel with the m seen row
        kernels Y (N, m), E[phi_t dY_r(t - b)] at every map node b."""
        return self.grid.corr_ops(Y, 0.0).transpose(1, 0, 2).reshape(self.N, Y.shape[1] * self.N)

    def causal_chunks(self, target: int = 4):
        """Node ranges of about `target` groups of whole panels: a correlation operator (projection_rows)
        is zero from a node to any node of an earlier panel, so the products from a chunk's ages need
        only the nodes of its own and later chunks."""
        P, n = self.grid.P, self.grid.n
        edges = sorted({0, P} | {int(round(P * i / target)) for i in range(1, target)})
        return [(a * n, b * n) for a, b in zip(edges[:-1], edges[1:])]

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
                U[i::nX, self.block(nm)] += c * self.shift(lag)
            perm = np.arange(nX * N).reshape(N, nX).T.reshape(-1)     # prim index -> (node, comp) index
            PX, P0X = self.Pin[perm], self.P0[perm]
            PU = PX @ U
            lu = lu_factor(np.eye(nX * N) - PU[:, :nX * N])
            W = lu_solve(lu, PU[:, nX * N:])
            self._elim[key] = (lu, W, P0X, perm)
            self._elim_wzero[key] = not W.any()          # no state driven by a control: the products with W are skipped
        return self._elim[key]

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                    impulse_controls: Sequence = ()):
        """Solve the closed loop for the Brownian channels and for unit impulses in
        `impulse_controls` (whose owning agent's reactions are switched off when it is
        `excluded`).  maps[agent] has shape (n_controls, n_rows, N).
        Returns Z with shape (n_prim N, nW + len(impulse_controls)).  The states are eliminated
        through the cached propagator part (see _state_elimination); the solve is over the controls.
        `excluded` may also be a tuple of agents, every one of them switched off (a deviation's privy set)."""
        if isinstance(excluded, (tuple, list)) and len(excluded) == 1:
            excluded = excluded[0]
        if self.sym is not None and not self.instant_loads and not isinstance(excluded, (tuple, list)) and self._maps_symmetric(maps):
            return self.closed_loop_symmetric(maps, excluded, impulse_controls)
        return self._closed_loop_eliminated(maps, excluded, impulse_controls)

    def _closed_loop_eliminated(self, maps, excluded=None, impulse_controls=()):
        nX, nU, N = self.nX, self.nU, self.N
        n = len(self.prim) * N; nxs = nX * N
        ncol = self.nW + len(impulse_controls)
        off = set(excluded) if isinstance(excluded, (tuple, list)) else ({excluded} if excluded else set())
        excl = frozenset(u for a in self.model.agents if a.name in off for u in a.controls)
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
            if a.name in off:
                continue
            g = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = slice(self.block(u).start - nxs, self.block(u).stop - nxs)
                Cs = self.grid.conv_ops_left(g[ui].T)                          # the rows' maps at once
                for r in range(len(a.signals)):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gur = g[ui, r]
                    C = Cs[r]
                    for nm, op in blocks.items():                       # only the primaries the row reads
                        MU[bl, self.block(nm)] += C @ op
                    self._add_point_columns(B, slice(nxs + bl.start, nxs + bl.stop), deltas, gur, impulse_controls)
        # instant observations: an observer's control moves with the level it sees, contemporaneously (its map covers
        # the rest of its action, see _map_part)
        for v, loads in (self.instant_loads or {}).items():
            if v in excl or any(v in a.controls for a in self.model.agents if a.name in off):
                continue
            bl = slice(self.block(v).start - nxs, self.block(v).stop - nxs)
            for u, h in loads.items():
                MU[bl, self.block(u)] += h * np.eye(N)
        Z = np.zeros((n, ncol))
        if nX:
            MUX, MUU = MU[:, :nxs], MU[:, nxs:]
            Z[nxs:] = np.linalg.solve(np.eye(nU * N) - MUU - MUX @ W, B[nxs:] + MUX @ GB)
            Z[:nxs] = W @ Z[nxs:] + GB
        else:
            Z[:] = np.linalg.solve(np.eye(nU * N) - MU, B)
        return Z

    # ------------------------------------------------ cyclic symmetry
    def _maps_symmetric(self, maps) -> bool:
        """Tied agents carry the same raw maps (the solver keeps them so); a user-supplied dict may not."""
        ags = self.sym.agents
        return all(np.array_equal(maps[a], maps[ags[0]]) for a in ags[1:])

    def _mode_structure(self):
        """Index structures of the control orbits: per orbit j and member s the node slice of that
        control in the control block, the fixed controls, the state-orbit permutations, and the DFT."""
        if self._modes is None:
            sym = self.sym; m = sym.order; N = self.N; nxs = self.nX * N; nU = self.nU
            ctrl_index = {u: i for i, u in enumerate(self.model.control_names)}
            state_index = {x: i for i, x in enumerate(self.model.state_names)}
            corb = [o for o in sym.orbits if o[0] in ctrl_index]                 # control orbits (cycle order)
            sorb = [o for o in sym.orbits if o[0] in state_index]                # state orbits
            fixed_c = [u for u in self.model.control_names if all(u not in o for o in corb)]
            # column indices of control-orbit j member s (in the control block), and of the fixed controls
            cols = [[np.arange(ctrl_index[o[s]] * N, (ctrl_index[o[s]] + 1) * N) for s in range(m)] for o in corb]
            fcols = np.concatenate([np.arange(ctrl_index[u] * N, (ctrl_index[u] + 1) * N) for u in fixed_c]) if fixed_c else np.zeros(0, dtype=int)
            # state-block row permutation by s steps of the cycle: entry i of the permuted vector is the
            # (s steps back) image, so that MUX_rep @ GB[perm[s]] gives the rows of firm s
            perms = []                     # GB[perms[s]] read at orbit member t gives GB at member t + s
            for sh in range(m):
                p = np.arange(nxs)
                for o in sorb:
                    for t in range(m):
                        src, dst = state_index[o[t]], state_index[o[(t + sh) % m]]
                        p[src * N:(src + 1) * N] = np.arange(dst * N, (dst + 1) * N)
                perms.append(p)
            # control columns shifted by s: entry at orbit j member t reads member t + s
            cperms = []
            for sh in range(m):
                p = np.arange(nU * N)
                for j in range(len(corb)):
                    for t in range(m):
                        p[cols[j][t]] = cols[j][(t + sh) % m]
                cperms.append(p)
            omega = np.exp(2j * np.pi / m)
            csl = [[slice(int(cc[0]), int(cc[-1]) + 1) for cc in cj] for cj in cols]     # the same columns as slices
            self._modes = {"m": m, "corb": corb, "cols": cols, "csl": csl, "fcols": fcols, "perms": perms, "cperms": cperms,
                           "omega": omega, "member": {o[t]: (j, t) for j, o in enumerate(corb) for t in range(m)}}
        return self._modes

    def _control_rows(self, maps, controls, excl: frozenset, impulse_controls, B, regular: bool = True):
        """The control-row operator MU (len(controls) N x n) for the given controls (their owners' maps),
        filling the instantaneous entries of B on the way (regular=False: only those entries)."""
        N = self.N; n = len(self.prim) * N
        owner = {u: a for a in self.model.agents for u in a.controls}
        MU = np.zeros((len(controls) * N, n))
        for i, u in enumerate(controls):
            a = owner[u]; ui = a.controls.index(u); g = maps[a.name]
            rows_here = slice(i * N, (i + 1) * N); bl = self.block(u)
            Cs = self.grid.conv_ops_left(g[ui].T) if regular else None       # the rows' maps at once
            for r in range(len(a.signals)):
                blocks, deltas = self.row_blocks(a.name, r, excl)
                gur = g[ui, r]
                if regular:
                    C = Cs[r]
                    for nm, op in blocks.items():
                        MU[rows_here, self.block(nm)] += C @ op
                self._add_point_columns(B, bl, deltas, gur, impulse_controls)
        return MU

    def closed_loop_symmetric(self, maps, excluded=None, impulse_controls=()):
        """closed_loop for a cyclically symmetric model with symmetric maps: the reduced control-row
        operator commutes with the cyclic relabelling, so in the Fourier basis over the cycle it is
        block diagonal, one block per mode, built from the representative agent's rows alone.
        Switching one agent off (the passive world) breaks the symmetry by a low-rank change of that
        agent's rows, applied with the Woodbury identity on top of the symmetric solve."""
        st = self._mode_structure(); m, omega = st["m"], st["omega"]
        nX, nU, N = self.nX, self.nU, self.N; n = len(self.prim) * N; nxs = nX * N
        ncol = self.nW + len(impulse_controls)
        ex_agent = self.model.agents[[a.name for a in self.model.agents].index(excluded)] if excluded else None
        B = np.zeros((n, ncol))
        lu, W, P0X, perm = self._state_elimination(frozenset())          # the symmetric world: nobody excluded
        wzero = self._elim_wzero[frozenset()]
        for k in range(self.nW):
            B[:nxs, k] += P0X @ self.sigma[:, k]
        for j, u in enumerate(impulse_controls):
            for i, (nm, lag), c in self.state_inputs:
                if nm == u:
                    v = np.zeros(nX); v[i] = c
                    B[:nxs, self.nW + j] += (P0X @ v) if lag == 0 else (self.grid.jump_injector(self.A, lag)[perm] @ v)
        GB = lu_solve(lu, B[:nxs])
        corb, cols, csl, fcols, perms, cperms = st["corb"], st["cols"], st["csl"], st["fcols"], st["perms"], st["cperms"]
        # representative rows of the reduced operator (states eliminated): R = MUU + MUX W
        rep_controls = [o[0] for o in corb] + [u for u in self.model.control_names if all(u not in o for o in corb)]
        Bsym = np.zeros((n, ncol))
        MU_rep = self._control_rows(maps, rep_controls, frozenset(), impulse_controls, Bsym)
        R_rep = MU_rep[:, nxs:] if wzero else MU_rep[:, nxs:] + MU_rep[:, :nxs] @ W       # (n_rep N, nU N)
        # every control's instantaneous entries (with the excluded agent's controls as impulses, which is
        # how the others' reactions to its impulse enter): cheap, no regular part
        all_controls = self.model.control_names
        excl = frozenset(ex_agent.controls) if ex_agent else frozenset()
        self._control_rows(maps, all_controls, excl, impulse_controls, B, regular=False)
        # right-hand side rows: the representative's state rows applied to the states shifted by s
        rhs = B[nxs:].copy(); MUX_rep = MU_rep[:, :nxs]
        for j, o in enumerate(corb):
            for sh in range(m):
                rhs[csl[j][sh]] += MUX_rep[j * N:(j + 1) * N] @ GB[perms[sh]]
        if fcols.size:
            rhs[fcols] += MUX_rep[len(corb) * N:] @ GB
        # mode blocks
        r = len(corb); nf = fcols.size
        # modes k and m - k are complex conjugates (the operator and the right-hand sides are real), so only
        # k <= m/2 is factorised and solved; the conjugate mode's contribution is the conjugate's
        kmax = m // 2
        blocks = []
        for k in range(kmax + 1):
            Ak = np.zeros((r * N + (nf if k == 0 else 0),) * 2, dtype=complex)
            for i in range(r):
                for j in range(r):
                    for d in range(m):
                        Ak[i * N:(i + 1) * N, j * N:(j + 1) * N] += (omega ** (d * k)) * R_rep[i * N:(i + 1) * N, csl[j][d]]
            if k == 0 and nf:
                for i in range(r):
                    Ak[i * N:(i + 1) * N, r * N:] += np.sqrt(m) * R_rep[i * N:(i + 1) * N][:, fcols]
                    Ak[r * N:, i * N:(i + 1) * N] += np.sqrt(m) * R_rep[r * N:][:, cols[i][0]]
                Ak[r * N:, r * N:] += R_rep[r * N:][:, fcols]
            blocks.append(lu_factor(np.eye(Ak.shape[0]) - Ak))

        def solve_sym(rhs_u):
            """(I - R) Z_U = rhs_u for the symmetric operator, by modes k <= m/2 and their conjugates."""
            out = np.zeros_like(rhs_u)
            for k in range(kmax + 1):
                size = r * N + (nf if k == 0 else 0)
                bk = np.zeros((size, rhs_u.shape[1]), dtype=complex)
                for j in range(r):
                    for sh in range(m):
                        bk[j * N:(j + 1) * N] += (omega ** (-sh * k)) * rhs_u[csl[j][sh]] / np.sqrt(m)
                if k == 0 and nf:
                    bk[r * N:] = rhs_u[fcols]
                zk = lu_solve(blocks[k], bk)
                weight = 1.0 if (k == 0 or 2 * k == m) else 2.0                  # the pair (k, m - k) or a self-conjugate mode
                for j in range(r):
                    for sh in range(m):
                        out[csl[j][sh]] += weight * ((omega ** (sh * k)) * zk[j * N:(j + 1) * N]).real / np.sqrt(m)
                if k == 0 and nf:
                    out[fcols] += zk[r * N:].real
            return out
        if ex_agent is None:
            ZU = solve_sym(rhs)
        else:
            # Woodbury: the excluded agent's control rows become identity rows (its strategy off)
            rows0 = np.concatenate([np.arange(all_controls.index(u) * N, (all_controls.index(u) + 1) * N) for u in ex_agent.controls])
            R0 = np.zeros((rows0.size, nU * N))                                  # the excluded agent's rows of the symmetric
            for i, u in enumerate(ex_agent.controls):                             # operator: the representative's, columns shifted
                if u in st["member"]:
                    j, sh = st["member"][u]; R0[i * N:(i + 1) * N] = R_rep[j * N:(j + 1) * N][:, cperms[(-sh) % m]]
                else:
                    fi = [x for x in rep_controls if x not in st["member"]].index(u)
                    R0[i * N:(i + 1) * N] = R_rep[(r + fi) * N:(r + fi + 1) * N]
            E0 = np.zeros((nU * N, rows0.size)); E0[rows0, np.arange(rows0.size)] = 1.0
            rhs_ex = rhs.copy(); rhs_ex[rows0] = 0.0                                # B rows of the excluded agent are zero
            Y = solve_sym(np.concatenate([rhs_ex, E0], axis=1)); Yb, Ye = Y[:, :ncol], Y[:, ncol:]
            cap = np.eye(rows0.size) + R0 @ Ye
            ZU = Yb - Ye @ np.linalg.solve(cap, R0 @ Yb)
            ZU[rows0] = 0.0
        Z = np.zeros((n, ncol)); Z[nxs:] = ZU; Z[:nxs] = GB if wzero else W @ ZU + GB
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
                U[i::self.nX, self.block(nm)] += c * self.shift(lag)
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
                    self._add_point_columns(B, bl, deltas, gur, impulse_controls)
        Z = np.linalg.solve(np.eye(n) - M, B)
        return Z



# ------------------------------------------------------------------- solver
class StationarySolver(EngineBase):
    RESULT = StationaryResult
    MONITORING = True                   # monitored deviations (Chapter 6): _impulse_responses below
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.6, 60       # with Anderson memory 15 (0.3 was needed at memory 6 for Kyle-Back)
    #  The second-order verdict does not depend on the discount, so this engine can check a
    #  discounted model.  The dissertation writes the discounted stationary objective (Chapter
    #  "Stationary infinite-horizon LQG", eq. stationary-objective) as
    #
    #      J = 1/2 E int_0^inf e^{-rho t} [X' G^XX X + 2 G^X' X + 2 D' G^DX X + D' G^DD D] dt
    #
    #  "with the joint running Hessian positive semidefinite".  That Hessian is TIME-LOCAL and
    #  carries no rho: the discount enters only as the strictly positive weight e^{-rho t}, which
    #  cannot change the sign of a form that is semidefinite pointwise in t.  So the check may be
    #  made on the average-cost system and its answer holds at every rho >= 0 -- which is what the
    #  Kyle-Back chapter does: "The second-order checks are made on the average-cost system and do
    #  not rely on the rho > 0 hypothesis", reporting the exact quadratic form positive definite
    #  with smallest eigenvalue 2 eps.  cost_mass() is that average-cost Gram at every rho.

    def __init__(self, model: Model, verbose: bool = False, settings=None):
        """settings: the tuning constants (noisestate.Settings, or a dict of its fields; the defaults when None)."""
        if model.horizon.kind == "transition":
            raise ValueError(f"horizon.kind 'transition' ({model.name!r}) runs on the spectral finite engine only "
                             "(noisestate.solve routes it there; this engine has no past)")
        super().__init__(model, verbose, settings=settings)
        self.c = Compiled(model, settings=self.settings)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}
        self._monitored = None                          # (maps key, {agent: monitored R}, {origin: seed world}, residual) of the last maps
        self._atom_blocks: Dict[str, tuple] = {}        # agent -> the loss atoms' blocks and which are the identity
        self._kernel_residual = 0.0                     # the inner solve's residual at the last maps

    # -------------------------------------------- overridable model pieces
    # ------------------------------------------- the best response (the kernel algebra of the age grid)
    def _row_support(self, agent: Agent, rows):
        """Which (row, channel) regular kernels are not identically zero, (nR, nW); the operators of the
        zero ones are zero and are skipped (a channel the agent's rows never carry, a row that reads
        nothing regular).  Rows are grouped by observation delay so one batched kernel call serves all
        rows of a delay."""
        sup = np.stack([np.any(y != 0, axis=0) for y in rows]) if rows else np.zeros((0, self.c.nW), dtype=bool)
        groups = {}
        for r in range(len(rows)):
            groups.setdefault(float(self.c.rows[agent.name][r][3]), []).append(r)
        return sup, groups

    def _row_operator(self, agent: Agent, rows, inst):
        """Per channel, the operator (N x nR N) mapping stacked row maps gamma to the action kernel:
        c_k = sum_r (Conv[y_rk] + E_rk S_delta) gamma_r."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        Gk = np.zeros((nW, N, nR * N))
        sup, groups = self._row_support(agent, rows)
        for d, rs in groups.items():
            pairs = [(r, k) for r in rs for k in np.where(sup[r])[0]]
            if pairs:
                ops = c.conv_rows(np.stack([rows[r][:, k] for r, k in pairs], axis=1), d)     # one call per delay
                for i, (r, k) in enumerate(pairs):
                    Gk[k, :, r * N:(r + 1) * N] = ops[i]
        for r in range(nR):
            for (k, age, w) in inst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.instant(age, c.rows[agent.name][r][3])
        return Gk

    def _response_operators(self, agent: Agent, R: np.ndarray):
        """Per control, the operator (n_prim N x N) giving the primary kernels' response to that
        control's action kernel: Z = Zpass + Resp_u c_u, with the own block equal to the action."""
        c = self.c; N = c.N
        out = []
        for ui, u in enumerate(agent.controls):
            Cu = c.response(R[:, ui].reshape(len(c.prim), N), c.prim.index(u))
            Cu[c.block(u)] = np.eye(N)
            for v, coef in (c.composite or {}).get(u, {}).items():   # an instant reaction moves with the action itself
                if v != u:
                    Cu[c.block(v)] += coef * np.eye(N)
            out.append(Cu)
        return out

    def _foc_operators(self, agent: Agent, R: np.ndarray, atoms: bool = False):
        """Per control, the operator (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses R, delayed reads of own lagged controls, and the past-date term of a lead.
        Called with the physical impulse responses (all reactions off) for the wedge decomposition.
        With atoms=True returns (Fu, Ms), Ms the per-control operators M (n_atoms, N, N) on the loss
        atoms' kernels that Fu contracts with Q (the mean part applies them to the targets q)."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atms, Q, q = c.loss[agent.name]
        # an atom operator is one N x N block, the shift into the atom's own primary: the position is the
        # atom's, so ask for the block rather than build the full-width operator and scan its zeros for it
        if agent.name not in self._atom_blocks:            # map-independent: built once per agent
            blocks = [[c.atom_block(at)] for at in atms]
            self._atom_blocks[agent.name] = (blocks, [[_is_eye(blk) for p, blk in bl] for bl in blocks])
        AO_blocks, AO_eye = self._atom_blocks[agent.name]       # AO_eye: an undelayed atom reads through the identity
        Fu, Ms = [], []
        for ui, u in enumerate(agent.controls):
            # op = sum_j M_j (Q zeta)_j with M_j the operator on atom j: identity for the instantaneous
            # term, continuation, delayed own read, lead term; contracted as sum_i (sum_j Q_ji M_j) AO_i
            # so the products are N x N x N per atom block instead of N x N x n_prim N per atom
            M = np.zeros((len(atms), N, N))
            for v, coef in (c.composite or {}).get(u, {u: 1.0}).items():   # the control and the instant reactions it draws
                if (v, 0.0) in atms:
                    M[atms.index((v, 0.0))] += coef * np.eye(N)
            if not agent.myopic:
                # impulse responses of every atom: its block against its own primary's rows of R
                Rj = np.stack([sum(A @ R[p * N:(p + 1) * N, ui] for p, A in AO_blocks[j]) for j in range(len(atms))], axis=1)
                CR = c.continuation(Rj)
                for j, (name, lag) in enumerate(atms):
                    if name in agent.controls:
                        if name == u and lag > 0:          # delayed read of the control itself
                            M[j] += np.exp(-c.rho * lag) * c.own_lag_read(lag)
                        continue                            # own reactions: envelope
                    M[j] += CR[j]
                    if lag < 0:
                        M[j] += self._lead_term(agent, R[:, ui], name, lag)
            MQ = np.tensordot(Q.T, M, axes=1)              # MQ[i] = sum_j Q[j, i] M_j
            op = np.zeros((N, n_prim))
            for i in range(len(atms)):
                if np.any(MQ[i]):
                    for (p, blk), eye in zip(AO_blocks[i], AO_eye[i]):
                        op[:, p * N:(p + 1) * N] += MQ[i] if eye else MQ[i] @ blk
            Fu.append(op); Ms.append(M)
        return (Fu, Ms) if atoms else Fu

    def _projection_operator(self, agent: Agent, rows, inst):
        """H (nR N x nW N): E[phi_t dY_r(t - b)] for every row r and lag b, from the FOC kernels."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            H[r * N:(r + 1) * N] += c.projection_rows(rows[r], c.rows[agent.name][r][3])
            for (k, age, w) in inst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.instant_adjoint(age, c.rows[agent.name][r][3])
        return H

    def _decompose(self, agent: Agent, out: dict, Fu, Resp, Gk, maps=None) -> None:
        """The second-order check and the FOC decomposition (instantaneous/physical/wedge).  Tied agents
        share the second-order check of their representative (the same problem up to relabelling)."""
        c = self.c; nW = c.nW; Zfull = out["Zfull"]
        out["second_order"] = self._shared_second_order(
            agent, lambda: self._second_order(agent, Resp, Gk, np.tile(self._identified(agent), len(agent.controls)), maps))
        Fphys = self._foc_operators(agent, self._physical_responses(agent, c.nW))
        dec = {}
        for ui, u in enumerate(agent.controls):
            phi = np.stack([Fu[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            phi_phys = np.stack([Fphys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
        out["decomp"] = dec

    def _second_order(self, agent: Agent, Resp, Gk, keep, maps=None) -> Optional[dict]:
        """Second-order condition of the best response: the agent's objective is a quadratic form in its
        strategy, and a first-order condition is a minimum only if that form is positive on the
        feasible strategies (those its rows can express).  The form is computed exactly from the
        cost's own Gram matrix: J(delta) = 1/2 delta' M delta with M = T' G T, T the map from a
        strategy to the world it produces and G the loss form.  Its extreme eigenvalues come from
        Lanczos on matvecs.  With a past the world has the initial shocks' columns after the channels',
        each under the point form of the line s = 0 (_loss_form(agent, start_from=True)), as expected_cost
        integrates them.  Returns {"min", "max", "ok", "converged"} with min/max the eigenvalues
        of M scaled by max.  The objective is a quadratic form in the strategy at every discount,
        because it enters the objective only as the strictly positive weight e^{-rho t} on a
        time-local Hessian, so the form's SIGN -- which is the whole verdict -- is the same at every
        rho and the check is made on the average-cost system.
        The objective is truncated at the window, so a strategy can push a little loss past the edge:
        curvatures within SECOND_ORDER_TOL of the largest are treated as that, not as a saddle.
        When the form is not positive and the engine defines _embedded_curvature(agent, maps, idx,
        vmin) (an optional hook, the stationary engine's), that is asked for the curvature of the
        offending direction on a longer window: a float, or None when it cannot say; a positive
        value turns the verdict into ok with "edge" and "embedded" recorded."""
        c = self.c
        N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        GAO = self._loss_form(agent)                                             # symmetric loss form on the world
        ncol = Gk.shape[0]                                                      # the channels, then a past's initial shocks
        forms = [(GAO, slice(0, nW))]                                           # (loss form, its columns of the world)
        if ncol > nW:
            forms.append((self._loss_form(agent, start_from=True), slice(nW, ncol)))
        idx = np.where(keep)[0]
        Nm = Gk.shape[2] // nR if nR else N                                     # a row's map block (N, plus discrete weights with a past)

        def T(delta_full):                     # strategy -> world, per column: (n_prim N, ncol)
            Zd = np.zeros((GAO.shape[0], ncol))
            for ui in range(nU):
                du = delta_full[ui * nR * Nm:(ui + 1) * nR * Nm]
                for k in range(ncol):
                    Zd[:, k] += Resp[ui] @ (Gk[k] @ du)
            return Zd

        def Tt(Zd):                            # its transpose
            out = np.zeros(nU * nR * Nm)
            for ui in range(nU):
                RZ = Resp[ui].T @ Zd                                            # (N, ncol)
                for k in range(ncol):
                    out[ui * nR * Nm:(ui + 1) * nR * Nm] += Gk[k].T @ RZ[:, k]
            return out

        def GT(Zd):                            # the loss form, column by column
            out = np.empty_like(Zd)
            for G, sl in forms:
                out[:, sl] = G @ Zd[:, sl]
            return out

        def matvec(v):
            full = np.zeros(nU * nR * Nm); full[idx] = np.asarray(v, dtype=float).ravel()
            return Tt(GT(T(full)))[idx]
        n = idx.size
        if n <= self.SECOND_ORDER_DENSE:
            # the form explicitly, M = sum_k T_k' G_(k) T_k with T_k = [Resp_u G_k]_u and G_(k) the column's loss
            # form, associated as M[u, v] = sum_k G_k' (Resp_u' G_(k) Resp_v) G_k: the inner form H_uv is N x N
            # (through the responding primaries' nodes only), and the sum over the columns of one loss form is
            # one product of the stacked row operators, restricted per column to the rows whose block of G_k is
            # not identically zero
            # eigenvalues only (no n x n eigenvectors and their workspace); the lowest direction is
            # computed only when the embedding below needs it
            Mfull = symmetrize(dense_curvature_form(Resp, Gk, forms, nU, nR, Nm, idx))
            w = np.linalg.eigvalsh(Mfull)
            lo, hi = float(w[0]), float(w[-1])
            vmin = lambda: sla.eigh(Mfull, subset_by_index=[0, 0])[1][:, 0]
        else:
            vmin = None
            res = self._lanczos_extremes(matvec, n)
            if "message" in res:
                return res
            lo, hi = res["lo"], res["hi"]
        scale = max(abs(lo), abs(hi), 1e-300)
        out = {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -self.SECOND_ORDER_TOL * scale), "converged": True}
        if not out["ok"] and vmin is not None and maps is not None and hasattr(self, "_embedded_curvature"):
            # the windowed objective omits the flows past the edge that read the strategy within the last lag:
            # a negative direction is a truncation artefact if the same direction, zero-extended onto a window
            # longer by two lags, has positive curvature under the same maps
            emb = self._embedded_curvature(agent, maps, idx, vmin())
            if emb is not None:
                out["embedded"] = float(emb / scale)
                out["edge"] = bool(emb >= 0.0)
                out["ok"] = out["edge"]
        return out

    def _loss_form(self, agent: Agent, start_from: bool = False) -> np.ndarray:
        """The loss form on the primary kernels, AO' kron(Q, mass) AO for the stacked atom operators AO,
        assembled block by block over the atoms' primary blocks (each atom reads one primary through one
        N x N block; an undelayed atom through the identity, whose products are skipped).  Map-independent,
        cached per agent.  With start_from=True the form of a past's initial-shock column: the same atoms under
        the point mass of the line s = 0 (_init_mass), as expected_cost integrates those columns."""
        key = (agent.name, start_from)
        if key not in self._loss_forms:
            c = self.c; N = c.N; n = len(c.prim) * N
            atoms, Q, q = c.loss[agent.name]
            if start_from:
                raise NotImplementedError("the form of an initial-shock column is the spectral finite engine's (finite_free)")
            mass = c.cost_mass()
            blocks = [[c.atom_block(at)] for at in atoms]
            GAO = np.zeros((n, n))
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    if Q[i, j] == 0.0:
                        continue
                    W = Q[i, j] * mass
                    for p, Ai in blocks[i]:
                        for p2, Aj in blocks[j]:
                            WA = W if _is_eye(Aj) else W @ Aj
                            GAO[p * N:(p + 1) * N, p2 * N:(p2 + 1) * N] += WA if _is_eye(Ai) else Ai.T @ WA
            self._loss_forms[key] = GAO
        return self._loss_forms[key]

    def _causal_chunks(self):
        """Node ranges [(lo, hi)] in increasing age such that the regular projection operator of any row is
        zero from ages in one chunk to nodes in an earlier one (a correlation reads only older ages); the
        products over those blocks are skipped (c.causal_chunks; the stationary Compiled declares them)."""
        return self.c.causal_chunks()

    def _foc_system(self, agent: Agent, rows, inst, Zpass: np.ndarray, Resp, Fu):
        """The first-order-condition system Amat gamma = -bvec on the passive rows,
        Amat[u, v] = sum_k H_k (Fu_u Resp_v) G_k, bvec[u] = sum_k H_k (Fu_u Zpass)_k, with G_k the row
        operator and H_k the projection operator of channel k.  Both split into a regular part (the
        convolution and correlation with the row kernels, zero for every (row, channel) whose kernel is
        zero) and the instantaneous entries (scaled shifts on one row block); the regular parts are
        assembled over the nonzero rows and channels only, with the projection's columns ordered
        (node, channel) so the product Fu Resp G comes out in the right layout without a transpose, and
        the instantaneous terms are added block by block.  Identical to the dense assembly to round-off."""
        c = self.c; N = c.N; nR, nU = len(rows), len(Fu)
        sup, groups = self._row_support(agent, rows)
        delay = [c.rows[agent.name][r][3] for r in range(nR)]
        Rn = np.where(sup.any(axis=1))[0]; Kn = np.where(sup.any(axis=0))[0]
        nRn, nKn = len(Rn), len(Kn)
        kpos = {int(k): i for i, k in enumerate(Kn)}
        # regular parts: Hs[(ri, a), (j, ki)] and Gs[a, (ki, ri, j)]
        Hs = np.zeros((nRn, N, N, nKn)); Gs = np.zeros((N, nKn, nRn, N))
        for d, rs in groups.items():
            pairs = [(ri, int(k)) for ri, r in enumerate(Rn) if r in rs for k in np.where(sup[r])[0]]
            if not pairs:
                continue
            Y = np.stack([rows[Rn[ri]][:, k] for ri, k in pairs], axis=1)
            Hp = c.projection_rows(Y, d).reshape(N, len(pairs), N)             # (a, pair, j)
            Gp = c.conv_rows(Y, d)                                              # (pair, a, j)
            for i, (ri, k) in enumerate(pairs):
                Hs[ri, :, :, kpos[k]] = Hp[:, i, :]
                Gs[:, kpos[k], ri, :] = Gp[i]
            del Y, Hp, Gp                                                       # two (N pairs N) arrays: not held through the assembly
        Hs = Hs.reshape(nRn * N, N * nKn); Gs = Gs.reshape(N, nKn * nRn * N)
        chunks = self._causal_chunks() if nKn else []
        Hc = [np.ascontiguousarray(Hs.reshape(nRn, N, N * nKn)[:, lo:hi, lo * nKn:]).reshape(nRn * (hi - lo), (N - lo) * nKn)
              for lo, hi in chunks]                                             # rows of ages in the chunk, columns of nodes not younger
        # instantaneous entries: (row, channel, weighted shift on the row operator, on the projection); a shift that is the
        # identity (an undelayed row's own noise) is applied as the scaling by its weight, which is what the product gives
        ent = [(r, k, w * c.instant(age, delay[r]), w * c.instant_adjoint(age, delay[r])) for r in range(nR) for (k, age, w) in inst[r]]
        ent = [(r, k, Sg, Sh, (w if _is_eye(c.instant(age, delay[r])) else None), (w if _is_eye(c.instant_adjoint(age, delay[r])) else None))
               for (r, k, Sg, Sh), (r_, k_, age, w) in zip(ent, [(r, k, age, w) for r in range(nR) for (k, age, w) in inst[r]])]
        rmul = lambda X, S, w: X * w if w is not None else X @ S            # X @ S with S = w I
        lmul = lambda S, w, X: w * X if w is not None else S @ X            # S @ X with S = w I
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        A6 = Amat.reshape(nU, nR, N, nU, nR, N); B3 = bvec.reshape(nU, nR, N)
        for ui in range(nU):
            phi = Fu[ui] @ Zpass                                                # (N, nW): the FOC of the passive world
            if nKn:
                B3[ui][Rn] += (Hs @ phi[:, Kn].reshape(-1)).reshape(nRn, N)
            for (r, k, Sg, Sh, wg, wh) in ent:
                B3[ui, r] += lmul(Sh, wh, phi[:, k])
            for vi in range(nU):
                FR = Fu[ui] @ Resp[vi]
                FRG = (FR @ Gs).reshape(N, nKn, nRn * N) if nKn else None      # [(j, ki), (ri, j')]
                for (lo, hi), H_ in zip(chunks, Hc):
                    T = (H_ @ FRG[lo:].reshape((N - lo) * nKn, nRn * N)).reshape(nRn, hi - lo, nRn, N)
                    for ri, r in enumerate(Rn):
                        A6[ui, r, lo:hi, vi][:, Rn, :] += T[ri]
                for (r, k, Sg, Sh, wg, wh) in ent:
                    if k in kpos:                                               # the channel also has regular kernels
                        ki = kpos[k]
                        X = (Hs.reshape(nRn * N, N, nKn)[:, :, ki] @ rmul(FR, Sg, wg)).reshape(nRn, N, N)     # H_reg FR G_inst
                        for ri, rr in enumerate(Rn):
                            A6[ui, rr, :, vi, r] += X[ri]
                        A6[ui, r, :, vi][:, Rn, :] += lmul(Sh, wh, FRG[:, ki, :]).reshape(N, nRn, N)        # H_inst FR G_reg
                    for (r2, k2, Sg2, Sh2, wg2, wh2) in ent:
                        if k2 == k:
                            A6[ui, r, :, vi, r2] += lmul(Sh, wh, rmul(FR, Sg2, wg2))                       # H_inst FR G_inst
        return Amat, bvec

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False, project: bool = True):
        """The agent's best response to `maps`: (raw map, {"gamma", "action", "Zfull", ...}).  The
        agent's information is the passive signal history, so its first-order condition is affine
        in its map on the passive rows: one linear solve.
        Hook (the cell engine overrides it wholesale, with the signature (agent, maps)).  Receives
        every agent's raw maps in self.shapes; must return the agent's raw map (its shape in
        self.shapes) and a dict with "gamma" (the FOC unknown), "action" (the action kernels,
        (nU, N, nW), what response_actions iterates on) and "Zfull" (the world with the response
        in).  With want_decomp=True the dict also carries "second_order" (the check, or None) and
        "decomp" (control -> {"foc", "physical", "wedge"} kernels (N, nW)), which _diagnostics
        reads; an engine without them must not call the base _diagnostics.  With project=False the
        raw map is None and its projection is skipped (response_actions needs the action kernels
        only).  A singular system raises the ValueError of singular_system_message."""
        c = self.c; N = c.N
        nR, nU = len(agent.signals), len(agent.controls)
        Zpass, R0 = self._spikes(c, maps, agent)
        # two uses of the impulse responses, kept apart: the on-path world is the passive world plus the agent's
        # actions as every other player sees them on the path (their filters, R0); the first-order condition and
        # the second-order form are about the agent's deviations, to which the players privy to it respond
        # through their response kernels (the monitored R; R0 itself without monitoring)
        R = self._impulse_responses(agent, maps, R0)
        Zpass = self._passive_world(agent, maps, Zpass, R0)
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(agent, ytil, yinst)
        Resp0 = self._response_operators(agent, R0)
        Resp = Resp0 if R is R0 else self._response_operators(agent, R)
        Fu = self._foc_operators(agent, R)
        # the FOC is affine in gamma: solve H (Fu (Zpass + sum_v Resp0_v Gk gamma_v)) = 0 for all controls
        Amat, bvec = self._foc_system(agent, ytil, yinst, Zpass, Resp0, Fu)
        gamma = self._solve_foc(agent, Amat, bvec).reshape(nU, nR, N)
        del Amat, bvec                                          # (nU nR N)^2: not kept through the diagnostics
        cact = np.stack([(Gk @ gamma[ui].reshape(-1)).T for ui in range(nU)])
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp0[ui] @ cact[ui]
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            self._decompose(agent, out, Fu, Resp, Gk, maps)
        return (self._project(agent, Zfull, self._map_part(agent, Zfull, cact)) if project else None), out

    def _map_part(self, agent: Agent, Z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The part of the agent's action kernels its map carries: the action less its instant loadings on the
        levels it sees (h times their kernels in Z), which the closed loop adds contemporaneously."""
        loads = self.c.instant_loads or {}
        if not any(u in loads for u in agent.controls):
            return actions
        out = np.array(actions, dtype=float, copy=True)
        for ui, v in enumerate(agent.controls):
            for u, h in loads.get(v, {}).items():
                out[ui] -= h * Z[self.c.block(u)]
        return out

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        """Relative residual of the best-response action kernels after projection on the agent's raw
        rows.  Zero in exact arithmetic; on the grid it measures how well products of kernels are
        resolved, so a value above about 1e-6 means the equilibrium is under-resolved: raise
        numerics.nodes."""
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        actions = self._map_part(agent, Zfull, actions)
        worst = 0.0
        for ui in range(len(agent.controls)):
            recon = np.stack([Bk[k] @ g[ui].reshape(-1) for k in range(self.c.nW)], axis=1)
            worst = max(worst, float(np.abs(recon - actions[ui]).max() / max(1e-300, np.abs(actions[ui]).max())))
        return worst

    # ------------------------------------------------------- monitored deviations (Chapter 6)
    #  With a monitoring relation, a unit deviation seed of agent i (a control impulse) is resolved by the players
    #  privy to it, P_i (i itself among them, which resumes play after the blip): they do not filter its effects
    #  through their maps but respond through response kernels D^{v<-i}, one per privy control v, in the seed's age.
    #  The naive players filter it as always.  By linearity the seed world is
    #      W_i = Z0_i[:, i] + sum_v C_v D^{v<-i},
    #  Z0_i the closed loop with every privy player's map off and an impulse column per privy control, C_v the
    #  convolution with v's impulse column (and the identity on v's own block: v's kernel is D^{v<-i}).  Each privy
    #  player j's first-order condition on W_i vanishes (Definition 6.3 (ii), sequential rationality), with j's
    #  continuation through its own monitored impulse responses R_j^mon, the responses to a seed of j with j's own
    #  reaction frozen (Lemma 6.6: the first-order condition is the same under the blip convention) and P_j \ {j}
    #  responding through their kernels D^{.<-j}.  The kernels for different origins depend on each other through
    #  R^mon, so they are found by iterating: kernels from the R^mon, R^mon from the kernels.  Every agent's best
    #  response then uses its R^mon (Definition 6.3 (i)).

    def _spikes(self, c, maps, agent: Agent, excluded=None):
        """(Zpass, R): the closed loop with `excluded` (default the agent) switched off, and the responses to a spike
        of each of the agent's controls together with the instant reactions it draws (c.composite: a trader seeing
        the quote trades at once), (n_prim N, nU).  Without instant observations the spikes are the controls' own."""
        comp = c.composite or {}
        extra = [v for u in agent.controls for v in comp.get(u, {u: 1.0}) if v not in agent.controls]
        ctrls = list(agent.controls) + list(dict.fromkeys(extra))
        Zp = c.closed_loop(maps, excluded=agent.name if excluded is None else excluded, impulse_controls=ctrls)
        cols = Zp[:, c.nW:]; k = {v: i for i, v in enumerate(ctrls)}
        R = np.stack([sum(coef * cols[:, k[v]] for v, coef in comp.get(u, {u: 1.0}).items()) for u in agent.controls], axis=1)
        return Zp[:, :c.nW], R

    def _maps_key(self, maps) -> bytes:
        import hashlib
        h = hashlib.sha1()
        for a in self.model.agents:
            h.update(np.ascontiguousarray(maps[a.name]).tobytes())
        return h.digest()

    def _seed_setup(self, maps, origin: str):
        """(ctrls, Z0, C): the privy controls (origin's first); Z0 (n_prim N, len(ctrls)), the closed loop with the
        privy players' maps off, whose columns are the spikes of the privy controls with the instant reactions each
        draws (c.composite); and per privy control the stacked convolution (n_prim N x N) with its spike's column,
        plus the identity on its own block and on the blocks of the controls reacting to it at once."""
        c = self.c; N = c.N; nP = len(c.prim)
        comp = c.composite or {}
        owner = {a.name: a for a in self.model.agents}
        P = self.model.privy(origin)
        ctrls = [u for n in P for u in owner[n].controls]
        need = list(dict.fromkeys(ctrls + [w for v in ctrls for w in comp.get(v, {v: 1.0})]))
        cols = c.closed_loop(maps, excluded=tuple(P), impulse_controls=need)[:, c.nW:]
        k = {v: i for i, v in enumerate(need)}
        spike = {v: sum(coef * cols[:, k[w]] for w, coef in comp.get(v, {v: 1.0}).items()) for v in ctrls}
        Z0 = np.stack([spike[v] for v in ctrls], axis=1)
        C = {}
        for v in ctrls:
            Cv = np.zeros((nP * N, N))
            for p in range(nP):
                Cv[p * N:(p + 1) * N] = c.grid.conv_op(spike[v][p * N:(p + 1) * N])
            for w, coef in comp.get(v, {v: 1.0}).items():
                Cv[c.block(w)] = coef * np.eye(N) if w == v else Cv[c.block(w)] + coef * np.eye(N)
            C[v] = Cv
        return ctrls, Z0, C

    def _monitoring(self, maps):
        """({agent: R^mon (n_prim N, nU)}, {origin: W (n_prim N, n origin controls)}) for the agents with privy
        others, the response kernels found by the iteration described above, from the naive responses, to 1e-10
        relative or until 10 rounds pass without improving on the best, which is kept (the plain iteration reaches
        rounding and then only wanders).  (Starting from the previous call's kernels carried a trial point's kernels into the next
        evaluation and broke the market of Chapter 6; Anderson on this iteration found other roots.  Neither is used.)"""
        key = self._maps_key(maps)
        if self._monitored is not None and self._monitored[0] == key:
            self._kernel_residual = self._monitored[3]
            return self._monitored[1], self._monitored[2]
        c = self.c; N = c.N
        owner = {a.name: a for a in self.model.agents}
        origins = [a.name for a in self.model.agents if len(self.model.privy(a.name)) > 1]
        setup = {i: self._seed_setup(maps, i) for i in origins}
        responders = {m for i in origins for m in self.model.privy(i)}
        Rmon = {n: self._spikes(c, maps, owner[n])[1] for n in responders}      # the naive responses
        shapes = {i: (len(owner[i].controls), len(setup[i][0]), N) for i in origins}
        Fu = {n: self._foc_operators(owner[n], Rmon[n]) for n in responders if n not in origins}   # fixed within the call
        eqs = {i: [(n, ui) for n in self.model.privy(i) for ui in range(len(owner[n].controls))] for i in origins}   # one FOC per privy control
        # the rows of the non-origin responders are fixed within the call: their blocks of A and B once
        fixed = {}
        for i in origins:
            ctrls, Z0, C = setup[i]
            for r, (n, ui) in enumerate(eqs[i]):
                if n not in origins:
                    fixed[(i, r)] = ([Fu[n][ui] @ C[v] for v in ctrls], Fu[n][ui] @ Z0[:, :shapes[i][0]])
        D = {}
        change = np.inf
        best = (np.inf, None, None)                     # (change, D, Rmon) of the best round
        best_round = 0
        for it in range(200):
            Fu.update({n: self._foc_operators(owner[n], Rmon[n]) for n in responders if n in origins})
            change = 0.0
            for i in origins:
                ctrls, Z0, C = setup[i]
                nO, nC = shapes[i][0], len(ctrls)
                A = np.empty((len(eqs[i]) * N, nC * N)); B = np.empty((len(eqs[i]) * N, nO))
                for r, (n, ui) in enumerate(eqs[i]):
                    rows = slice(r * N, (r + 1) * N)
                    blocks, rhs = fixed[(i, r)] if (i, r) in fixed else ([Fu[n][ui] @ C[v] for v in ctrls], Fu[n][ui] @ Z0[:, :nO])
                    for k, blk in enumerate(blocks):
                        A[rows, k * N:(k + 1) * N] = blk
                    B[rows] = -rhs
                X = np.linalg.solve(A, B)                                           # one factorisation, every origin control
                Di = X.T.reshape(shapes[i])
                change = max(change, float(np.abs(Di - D[i]).max() / max(1e-300, np.abs(Di).max())) if i in D else np.inf)
                D[i] = Di
            for j in origins:
                Rmon[j] = self._frozen_responses(j, setup[j], D[j], shapes[j][0])
            if change < best[0]:
                best = (change, dict(D), dict(Rmon)); best_round = it
            # the iteration reaches rounding (about 1e-11) and then only wanders: stop at 1e-10, or once 10 rounds
            # have not improved on the best, and keep the best round
            if change < 1e-10 or it - best_round > 10:
                break
        change, D, Rmon = best
        W = {}                                          # the seed worlds of the best round's kernels
        for i in origins:
            ctrls, Z0, C = setup[i]
            X = D[i].reshape(shapes[i][0], -1).T
            W[i] = Z0[:, :shapes[i][0]] + sum(C[v] @ X[k * N:(k + 1) * N] for k, v in enumerate(ctrls))
        # at a trial point of the outer iteration far from the equilibrium the kernels may not settle; the best
        # round is used there and its change kept, and _finish requires them settled at the equilibrium itself
        self._kernel_residual = float(change)
        self._monitored = (key, Rmon, W, self._kernel_residual)
        return Rmon, W

    def _frozen_responses(self, j: str, setup, Dj: np.ndarray, own: int) -> np.ndarray:
        """R^mon_j (n_prim N, own): the responses to a frozen spike of each of j's controls, the players privy to j
        responding.  They read j's control path under the blip convention, a seed followed by j's continuation
        D^{j<-j}, so a frozen spike (no continuation) is the seeds sigma with sigma + D^{j<-j} * sigma = delta: a
        spike at 0 and, after it, the seeds s that cancel the continuation, s_u + sum_o' D^{u<-j,o'} * s_o' =
        -D^{u<-j,o} (a Volterra equation of the second kind in the seed's age).  The privy controls v respond to
        sigma, D^{v<-j,o} + sum_o' D^{v<-j,o'} * s_o'; j's own controls are the spike alone."""
        c = self.c; N = c.N
        ctrls, Z0, C = setup
        conv = [[c.grid.conv_op(Dj[o2, u]) for o2 in range(own)] for u in range(own)]      # conv[u][o'] g = D^{u<-j,o'} * g
        M = np.eye(own * N) + np.block([[conv[u][o2] for o2 in range(own)] for u in range(own)])
        out = np.zeros((len(c.prim) * N, own))
        for o in range(own):
            s = np.linalg.solve(M, -np.concatenate([Dj[o, u] for u in range(own)])).reshape(own, N)
            col = Z0[:, o].copy()
            for k, v in enumerate(ctrls):
                if k < own:
                    continue
                x = Dj[o, k] + sum(c.grid.conv_op(Dj[o2, k]) @ s[o2] for o2 in range(own))
                col += C[v] @ x
            out[:, o] = col
        return out

    def _finish(self, res) -> None:
        """The base's, after requiring the monitored response kernels settled at the equilibrium's maps (at trial
        points of the fixed point they may not be, _monitoring): a result whose kernels did not settle is not
        converged."""
        if any(len(self.model.privy(a.name)) > 1 for a in self.model.agents):
            self._monitoring(res.maps)                  # the last evaluation's, when it was at these maps
            if self._kernel_residual > 1e-10:
                res.converged = False
                res.message += f"; the monitored response kernels did not settle at the equilibrium (residual {self._kernel_residual:.1e})"
        super()._finish(res)

    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """With a monitoring relation, the agent's impulse responses with the players privy to its deviations
        responding through their response kernels (Chapter 6); without one, R."""
        if len(self.model.privy(agent.name)) == 1:
            return R
        return self._monitoring(maps)[0][agent.name]

    def _lead_term(self, agent: Agent, Ru: np.ndarray, name: str, lag: float) -> np.ndarray:
        """(N, N) operator on the led atom's (Q zeta) kernel: the past-date term of a lead (see EngineBase)."""
        # flows at dates t - |lag| <= tau < t also read the quantity after t, so the derivative of the
        # discounted objective has a further term over those past dates:
        #   int_0^{|lag|} e^{rho v} r(|lag| - v) (Q zeta)_j(a - v) dv   (a convolution with k(v))
        c = self.c; v = c.grid.nodes
        m = v <= -lag + 1e-12                          # only the dates t - v within the lead (the exponential
        kv = np.zeros(c.N)                             # of rho v at the far end of the window would overflow)
        kv[m] = np.exp(c.rho * v[m]) * (c.grid.interp(-lag - v[m]) @ Ru[c.block(name)])
        return c.grid.conv_op(kv)

    def _project(self, agent: Agent, Z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Raw maps (nU, nR, N) reproducing the action kernels `actions` (nU, N, nW) on the agent's
        closed-loop seen rows, by weighted least squares."""
        c = self.c; N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        rows, inst = self._seen_rows(agent, Z, set())
        Bk = self._row_operator(agent, rows, inst)
        W = c.grid.mass
        keep = self._identified(agent)                                       # delayed rows: zero where they read nothing
        # the Gram sum_k Bk' W Bk over the action nodes in causal chunks: the row operators are convolutions (and
        # lagged instantaneous reads), so the action at an age reads only map nodes of its own and earlier panels,
        # and a chunk's part of the Gram is one symmetric rank-k update (dsyrk on the sqrt(W)-scaled block) on the
        # map nodes up to the chunk's top age on every row: the same sums as the full product without its zero terms
        from scipy.linalg.blas import dsyrk
        sw = np.sqrt(W)
        B4 = Bk.reshape(nW, N, nR, N)
        Gram = np.zeros((nR * N, nR * N)); G4 = Gram.reshape(nR, N, nR, N)
        chunks = self._causal_chunks()
        if any(np.any(B4[:, lo:hi, :, hi:]) for lo, hi in chunks):           # a lead among the instantaneous reads: no causal chunks
            chunks = [(0, N)]
        for lo, hi in chunks:
            X = (B4[:, lo:hi, :, :hi] * sw[None, lo:hi, None, None]).reshape(nW * (hi - lo), nR * hi)
            G4[:, :hi, :, :hi] += dsyrk(1.0, X, trans=1).reshape(nR, hi, nR, hi)     # upper triangle (local order = global order)
        Gram = np.triu(Gram); Gram = Gram + Gram.T - np.diag(np.diagonal(Gram))
        rhs = Bk.reshape(nW * N, nR * N).T @ (actions * W[None, :, None]).transpose(2, 1, 0).reshape(nW * N, nU)   # column ui: sum_k Bk' W actions[ui, :, k]
        if not keep.all():
            Gram = Gram[np.ix_(keep, keep)]; rhs = rhs[keep]
        Gram += self.settings.stationary_map_ridge * np.trace(Gram) / Gram.shape[0] * np.eye(Gram.shape[0])
        g = np.zeros((nU, nR * N))
        g[:, keep] = np.linalg.solve(Gram, rhs).T                              # one factorisation for every control
        return g.reshape(nU, nR, N)

    def _embedded_curvature(self, agent: Agent, maps, idx, vec):
        """The quadratic form of a strategy direction (vec over the kept stacked map nodes idx) on a window
        longer by two lags, with the same maps zero-extended: positive means the negative curvature on this
        window was its truncation (flows past the edge that read the strategy within the last lag).  One
        operator build on the longer grid, no fixed point."""
        c = self.c; N = c.N; nR, nU = len(agent.signals), len(agent.controls)
        lags = [l for (nm, l) in c.loss[agent.name][0] if l > 0] + [r.delay for a in self.model.agents for r in a.signals if r.delay > 0]
        if not lags:
            return None
        ext = c.grid.L + 2.0 * max(lags)
        bp = [float(b) for b in c.grid.breakpoints] + [ext]
        try:
            S2 = type(self)(self.model._patch_horizon(window=ext).with_numerics(breakpoints=bp), **self.solver_kw)
        except Exception:
            return None
        c2 = S2.c; N2 = c2.N; g2 = c2.grid
        sides = g2.node_sides(); I = np.zeros((N2, N))
        for sd in (+1, -1):
            sel = (sides == sd) | ((sides == 0) & (sd == +1))
            I[sel] = c.grid.interp(g2.nodes[sel], side=sd)                  # zero beyond the old window
        maps2 = {a: np.einsum("fn,urn->urf", I, m) for a, m in maps.items()}
        full = np.zeros(nU * nR * N); full[idx] = vec
        d2 = np.concatenate([(I @ full[u * nR * N:(u + 1) * nR * N].reshape(nR, N).T).T.reshape(-1) for u in range(nU)])
        Zpass, R = S2._spikes(c2, maps2, agent)
        R = S2._impulse_responses(agent, maps2, R)
        ytil, yinst = S2._passive_rows(agent, Zpass)
        Gk2 = S2._row_operator(agent, ytil, yinst); Resp2 = S2._response_operators(agent, R)
        GAO2 = S2._loss_form(agent)
        value = 0.0
        for k in range(c2.nW):
            zd = np.zeros(GAO2.shape[0])
            for u in range(nU):
                zd += Resp2[u] @ (Gk2[k] @ d2[u * nR * N2:(u + 1) * nR * N2])
            value += float(zd @ (GAO2 @ zd))
        return value

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
        """gamma (nU nR N,) solving the FOC system on the identified nodes with np.linalg.solve (its exact
        singularity test, not the condition estimate of _solve_regular: see there), zero elsewhere; an exactly
        singular system raises the ValueError of singular_system_message."""
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
            if keep.all():                                    # no delayed row: the system as assembled, no copy
                gamma[:] = np.linalg.solve(Amat, -bvec)
            else:
                gamma[keep] = np.linalg.solve(Amat[np.ix_(keep, keep)], -bvec[keep])
        except np.linalg.LinAlgError:
            raise ValueError(singular_system_message(agent.name)) from None
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
        return self._over_representatives(lambda a: self._project(a, Z, self._map_part(a, Z, np.stack([Z[self.c.block(u)] for u in a.controls]))))

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Every agent's raw maps (nU, nR, N) reproducing its action kernels (nU, N, nW) in the world those
        actions generate (the states from the propagator, then one projection per representative)."""
        return self.maps_from_kernels(self.world_from_actions(actions))

    # ------------------------------------------------------ fixed point
    def interpolate_maps(self, coarse) -> Dict[str, np.ndarray]:
        """The coarse result's raw maps read at this grid's nodes, each node from its own panel's side."""
        g, gc = self.c.grid, coarse.compiled.grid
        sides = g.node_sides(); I = np.zeros((g.N, gc.N))
        for sd in (+1, -1):
            sel = (sides == sd) | ((sides == 0) & (sd == +1))
            I[sel] = gc.interp(g.nodes[sel], side=sd)
        return {a.name: np.einsum("fn,urn->urf", I, coarse.maps[a.name]) for a in self.model.agents}

    def _diagnostics(self, res) -> None:
        """The first-order-condition decomposition, the second-order check and the representation error of
        every agent's best response at the equilibrium."""
        self._fill_diagnostics(res)

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """Stationary flow loss per unit time of the agent in the world Z (exact Gram quadrature): the
        variance part, from the shocks; the mean part is mean_cost."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.shift(lag) @ Z[c.block(nm)] for nm, lag in atoms])   # (m, N, nW)
        MZ = np.tensordot(zeta, c.cost_mass(), axes=([1], [0]))        # (m, nW, N): the mass applied to every atom kernel
        G = np.tensordot(MZ, zeta, axes=([1, 2], [2, 1]))              # <zeta_i, zeta_j> over ages and channels
        return float(0.5 * np.sum(Q * G))


    # ------------------------------------------------------------ means (the hooks of EngineBase's mean layer)
    def _mean_start(self) -> np.ndarray:
        """The stationary means have no initial condition."""
        return np.zeros(self.c.nX)

    def _mean_dynamics(self):
        """The states' rows of the stationary mean system: A xbar + (control and lagged inputs at their constants)
        + const = 0 (a lag of a constant is the constant); a random walk with no inputs, whose row is identically
        zero, is pinned at 0, since the stationary model does not carry its level."""
        c = self.c; nX, nP = c.nX, len(c.prim)
        Mx = np.zeros((nX, nP)); bx = np.zeros(nX)
        Mx[:, :nX] = c.A
        for i, (nm, lag), coef in c.state_inputs:
            Mx[i, c.index[nm]] += coef
        bx[:] = -c.const
        pinned = []
        for i, s in enumerate(self.model.states):
            if not Mx[i].any():
                if bx[i] != 0:
                    raise ValueError(f"state {s.name}: a constant drift with no feedback in its dynamics has no stationary mean")
                Mx[i, i] = 1.0; pinned.append(i)                # a random walk with no inputs: its level is taken as 0
        return Mx, bx, pinned

    def _mean_conditions(self, agent: Agent, maps: Dict[str, np.ndarray]):
        """The agent's mean first-order conditions, one row per control.  With g = Q zbar + q on the loss atoms
        it is sum_j m_j g_j = 0: m_j = 1 on the control's current value, e^{-rho tau} on its own read at lag
        tau, and for every other atom the discounted DC gain int_0^L e^{-rho a} R_j(a) da of the atom's
        passive-world impulse response (the other agents reacting through their kernels, the agent's own
        control passive; a lead reads the whole response of its primary, weighted by e^{rho tau})."""
        c = self.c; nP = len(c.prim); rho = c.rho; a = agent
        dc = c.grid.discounted_mass(rho)
        atoms, Q, q = c.loss[a.name]
        Mu = np.zeros((len(a.controls), nP)); bu = np.zeros(len(a.controls))
        P = np.zeros((len(atoms), nP))                        # atom means from the primaries' means
        for j, (nm, lag) in enumerate(atoms):
            P[j, c.index[nm]] = 1.0
        R = None
        for ui, u in enumerate(a.controls):
            m = np.zeros(len(atoms))
            for v, coef in (c.composite or {}).get(u, {u: 1.0}).items():      # the control and the instant reactions it draws
                if (v, 0.0) in atoms:
                    m[atoms.index((v, 0.0))] += coef
            if not a.myopic:
                if R is None:
                    R = self._spikes(c, maps, a)[1]
                    R = self._impulse_responses(a, maps, R)
                for j, (nm, lag) in enumerate(atoms):
                    if nm in a.controls:
                        if nm == u and lag > 0:               # delayed read of the control itself
                            m[j] += np.exp(-rho * lag)
                        continue                              # own reactions: envelope
                    if lag >= 0:
                        m[j] += dc @ (c.shift(lag) @ R[c.block(nm), ui])
                    else:
                        m[j] += np.exp(-rho * lag) * (dc @ R[c.block(nm), ui])
            Mu[ui] = m @ Q @ P; bu[ui] = -(m @ q)
        return Mu, bu
