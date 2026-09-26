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
from scipy.linalg import lu_factor, lu_solve

from .grid import AgeGrid
from .grid_cache import age_grid
from .engine import EngineBase, singular_system_message
from .compile import CompiledBase, close_under_delays
from .symmetry import find_cyclic_symmetry
from .results import StationaryResult
from ._settings import Settings, tunable
from .spec import Agent, Atom, Model



WITHDRAWN_NAIVE = ("naive_observers was withdrawn in 1.1: it did not compute Chapter 6's naive or privy equilibria "
                   "(its 'naive' was the chapter's privy, and its solves failed their own first-order conditions). "
                   "Monitored deviations will come back as a model's monitoring relation.")

class Compiled(CompiledBase):
    """Grid, index maps and constant operators for a stationary model."""
    LEAD_WEIGHT_WARN = tunable("lead_weight_warn")     # warn when a lead's past flows outweigh the current one by more than this (settings)

    def __init__(self, model: Model, settings=None):
        super().__init__(model)
        self.settings = Settings.of(settings)
        hz = model.horizon
        lags = model.all_lags()
        if hz.breakpoints:
            bp = list(hz.breakpoints)
        elif lags or hz.unit:
            bp = AgeGrid.breakpoints_from_delays(hz.extent, lags, hz.unit, hz.unit_range)
        else:
            bp = [0.0, hz.extent]
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
        if self.rho > 0:
            leads = {(n, -l) for a in model.agents for term in a.loss for atom in term[1:] for (n, l) in model.expand({atom: 1.0}) if l < 0}
            for n, tau in sorted(leads):
                if self.rho * tau > np.log(self.LEAD_WEIGHT_WARN):
                    warnings.warn(f"lead {n}@-{tau:g} under the discount rate {self.rho:g}: the flows before t that read the quantity "
                                  f"after t enter the first-order condition weighted by up to exp(rho tau) = {np.exp(self.rho * tau):.1e} "
                                  "relative to the current flow, which dominates the best-response system (README, Limits)", stacklevel=2)
        self._atom_cache: Dict[tuple, np.ndarray] = {}
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
                U[i::nX, :] += c * self.atom_op((nm, lag))
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
        through the cached propagator part (see _state_elimination); the solve is over the controls."""
        if self.sym is not None and self._maps_symmetric(maps):
            return self.closed_loop_symmetric(maps, excluded, impulse_controls)
        return self._closed_loop_eliminated(maps, excluded, impulse_controls)

    def _closed_loop_eliminated(self, maps, excluded=None, impulse_controls=()):
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
                Cs = self.grid.conv_ops_left(g[ui].T)                          # the rows' maps at once
                for r in range(len(a.signals)):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gur = g[ui, r]
                    C = Cs[r]
                    for nm, op in blocks.items():                       # only the primaries the row reads
                        MU[bl, self.block(nm)] += C @ op
                    self._add_point_columns(B, slice(nxs + bl.start, nxs + bl.stop), deltas, gur, impulse_controls)
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
                    self._add_point_columns(B, bl, deltas, gur, impulse_controls)
        Z = np.linalg.solve(np.eye(n) - M, B)
        return Z



# ------------------------------------------------------------------- solver
class StationarySolver(EngineBase):
    RESULT = StationaryResult
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

    def __init__(self, model: Model, verbose: bool = False, settings=None, **withdrawn):
        """settings: the tuning constants (noisestate.Settings, or a dict of its fields; the defaults when None)."""
        if "naive_observers" in withdrawn:
            raise TypeError(WITHDRAWN_NAIVE)
        if withdrawn:
            raise TypeError(f"unknown option(s) {sorted(withdrawn)} for the stationary engine")
        if model.horizon.kind == "transition":
            raise ValueError(f"horizon.kind 'transition' ({model.name!r}) runs on the spectral finite engine only "
                             "(noisestate.solve routes it there; this engine has no past)")
        super().__init__(model, verbose, settings=settings)
        self.c = Compiled(model, settings=self.settings)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    # -------------------------------------------- overridable model pieces
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
        for lo, hi in self._causal_chunks():
            if np.any(B4[:, lo:hi, :, hi:]):                                 # a lead among the instantaneous reads: no causal chunks
                lo, hi = 0, N
            X = (B4[:, lo:hi, :, :hi] * sw[None, lo:hi, None, None]).reshape(nW * (hi - lo), nR * hi)
            G4[:, :hi, :, :hi] += dsyrk(1.0, X, trans=1).reshape(nR, hi, nR, hi)     # upper triangle (local order = global order)
            if hi == N and lo == 0:
                break
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
        Zp = c2.closed_loop(maps2, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :c2.nW], Zp[:, c2.nW:]
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
        return self._over_representatives(lambda a: self._project(a, Z, np.stack([Z[self.c.block(u)] for u in a.controls])))

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
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])          # (m, N, nW)
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
            if (u, 0.0) in atoms:
                m[atoms.index((u, 0.0))] += 1.0
            if not a.myopic:
                if R is None:
                    R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls)[:, c.nW:]
                    R = self._impulse_responses(a, maps, R)
                for j, (nm, lag) in enumerate(atoms):
                    if nm in a.controls:
                        if nm == u and lag > 0:               # delayed read of the control itself
                            m[j] += np.exp(-rho * lag)
                        continue                              # own reactions: envelope
                    if lag >= 0:
                        m[j] += dc @ (c.atom_op((nm, lag)) @ R[:, ui])
                    else:
                        m[j] += np.exp(-rho * lag) * (dc @ R[c.block(nm), ui])
            Mu[ui] = m @ Q @ P; bu[ui] = -(m @ q)
        return Mu, bu
