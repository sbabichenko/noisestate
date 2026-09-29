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
from .monitoring import MonitoredDeviations, compose_spikes, spike_controls
from .compile import CompiledBase, close_under_delays
from .symmetry import find_cyclic_symmetry
from .results import StationaryResult
from ._settings import Settings, tunable
from .spec import Agent, Atom, Model


def _lu_solve(lu, b: np.ndarray, trans: int = 0) -> np.ndarray:
    """scipy.linalg.lu_solve(lu, b), its finiteness check included, straight on LAPACK's getrs: the same routine without
    the wrapper's layers, which cost more than the solve at the closed loop's sizes (thousands of calls in a
    monitored market's solve)."""
    b = np.asarray_chkfinite(b)
    getrs, = sla.get_lapack_funcs(("getrs",), (lu[0], b))
    x, info = getrs(lu[0], lu[1], b, trans=trans)
    if info != 0:
        raise ValueError(f"illegal value in {-info}th argument of internal getrs")
    return x


def _pgmres(A, Minv, b: np.ndarray, tol: float, maxit: int, x0=None):
    """(x, iterations) with ||b - A x|| <= tol ||b||, by GMRES right-preconditioned with Minv (a fixed preconditioner,
    so x = x0 + Minv(V y)), classical Gram-Schmidt with one reorthogonalisation; restarted from the true residual while
    the budget of maxit iterations lasts.  (None, maxit) when the tolerance is not reached."""
    n = b.size; nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return np.zeros(n), 0
    if x0 is None:
        x, r = np.zeros(n), b.copy()
    else:
        x = np.array(x0, dtype=float); r = b - A(x)
    its = 0
    while its < maxit:
        beta = float(np.linalg.norm(r))
        if beta <= tol * nb:
            return x, its
        m = maxit - its
        V = np.empty((m + 1, n)); H = np.zeros((m + 1, m))
        V[0] = r / beta
        k = 0
        for j in range(m):
            w = A(Minv(V[j]))
            h = V[:j + 1] @ w; w -= h @ V[:j + 1]
            h2 = V[:j + 1] @ w; w -= h2 @ V[:j + 1]
            H[:j + 1, j] = h + h2
            H[j + 1, j] = hn = float(np.linalg.norm(w))
            k = j + 1
            e1 = np.zeros(j + 2); e1[0] = beta
            y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
            est = float(np.linalg.norm(e1 - H[:j + 2, :j + 1] @ y))
            if hn == 0.0 or est <= 0.5 * tol * nb:
                break
            V[j + 1] = w / hn
        its += k
        x = x + Minv(y @ V[:k])
        r = b - A(x)
    return (x, its) if float(np.linalg.norm(r)) <= tol * nb else (None, its)


_LOWER: Dict[int, np.ndarray] = {}


def _strict_lower(m: int) -> np.ndarray:
    """The (m, m) mask of the strict lower triangle, made once per size."""
    M = _LOWER.get(m)
    if M is None:
        M = _LOWER[m] = np.tri(m, m, -1, dtype=bool)
    return M


class _BlockForm:
    """A loss form on the world (n x n, n = n_prim N) held as its nonzero N x N blocks {(p, p2): block}: the second-order
    form reads it only on the responding nodes (form[np.ix_(nz, nz)], a copy of those entries, the same as the dense
    array's), so the n x n array is made only for a product with it (__matmul__: the Lanczos path, the embedded
    curvature), once."""

    def __init__(self, n: int, N: int, blocks: Dict[Tuple[int, int], np.ndarray]):
        self.shape, self.N, self.blocks, self._dense = (n, n), N, blocks, None

    def dense(self) -> np.ndarray:
        if self._dense is None:
            N = self.N
            D = np.zeros(self.shape)
            for (p, p2), B in self.blocks.items():
                D[p * N:(p + 1) * N, p2 * N:(p2 + 1) * N] = B
            self._dense = D
        return self._dense

    def __matmul__(self, X):
        return self.dense() @ X

    def __getitem__(self, key):
        r, c = (np.asarray(k).ravel() for k in key)                   # np.ix_'s pair
        N = self.N
        out = np.zeros((r.size, c.size))
        rb, cb = r // N, c // N
        for (p, p2), B in self.blocks.items():
            ri = np.where(rb == p)[0]; ci = np.where(cb == p2)[0]
            if ri.size and ci.size:
                out[np.ix_(ri, ci)] = B[np.ix_(r[ri] - p * N, c[ci] - p2 * N)]
        return out


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
        self._row_ops: Dict[tuple, tuple] = {}             # (agent, row, excluded) -> row_blocks (map-independent)
        self._row_ops_bytes = 0
        self._state_forcing: Dict[tuple, tuple] = {}       # (excluded, impulse controls) -> (B_X, G B_X) of the closed loop
        self._elim_wzero: Dict[frozenset, bool] = {}
        self.sym = find_cyclic_symmetry(model)
        self._composite_static = self.composite            # the loss's instant reactions; use_maps adds the level rows
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

    def use_maps(self, maps) -> None:
        """Set the instant reactions to the strategies `maps`: a spike of a quantity seen on a level row draws, besides
        the loss's reaction (h, compile._composite), the observer's map on the row at age 0 (a jump of the level is
        an increment the map reads at once).  Without level rows the composite is the loss's alone."""
        self._level_g0 = []                              # (q, observer's control v, observer, g0): the level rows' part
        if not self.levels:
            return
        comp = {u: dict(v) for u, v in (self._composite_static or {}).items()}
        owner = {a.name: a for a in self.model.agents}
        for an, rows in self.levels.items():
            g = maps.get(an) if maps is not None else None
            if g is None:
                continue
            for r, q in rows.items():
                if q not in self.model.control_names:
                    continue                                 # a state is never spiked
                for ui, v in enumerate(owner[an].controls):
                    g0 = float(g[ui, r, 0])                  # node 0 is age 0
                    if g0 != 0.0:
                        comp.setdefault(q, {q: 1.0})
                        comp[q][v] = comp[q].get(v, 0.0) + g0
                        self._level_g0.append((q, v, an, g0))
        self.composite = comp

    def seed_composite(self, origin: str) -> Dict[str, Dict[str, float]]:
        """The composite for `origin`'s seed worlds (the monitoring iteration): a player privy to the origin knows its
        spike for what it is, so its instant reaction is the loss's alone, not the belief update its map on a level row
        would add (which the on-path composite, use_maps, keeps: on the path it reacts through its map)."""
        lg0 = getattr(self, "_level_g0", [])
        memo = self.__dict__.setdefault("_seed_comp", {})     # per origin, while use_maps has not been called again
        hit = memo.get(origin)
        if hit is not None and hit[0] is self.composite and hit[1] is lg0:
            return hit[2]
        privy = set(self.model.privy(origin))
        if not any(an in privy for (_, _, an, _) in lg0):
            comp = self.composite
        else:
            comp = {u: dict(v) for u, v in (self.composite or {}).items()}
            for q, v, an, g0 in lg0:
                if an in privy:
                    comp[q][v] -= g0
        memo[origin] = (self.composite, lg0, comp)
        return comp

    # ------------------------------------------------------- closed loop
    def row_blocks(self, agent: str, r: int, excluded: set):
        """Regular part of row r as seen by the agent (delayed), as (N x N) operators on the primary
        kernels it reads, {primary: operator}, and the instantaneous entries per source: channel
        names for Brownian noise, control names for observed-control impulses.  Controls in
        `excluded` (the agent whose reaction is switched off) contribute impulses through their own
        impulse channel instead of through a kernel.  Map-independent, so cached per (agent, row, excluded
        controls); the operators are shared and must not be written to."""
        key = (agent, r, frozenset(excluded))
        hit = self._row_ops.get(key)
        if hit is None:
            hit = self._row_blocks(agent, r, excluded)
            size = sum(op.nbytes for op in hit[0].values())
            if self.N <= 96 and self._row_ops_bytes + size <= (2 << 20):    # small grids only (the monitored markets
                self._row_ops[key] = hit                        # call it thousands of times): on a large grid the copies
                self._row_ops_bytes += size                     # cost more (memory, the allocator) than they save
        blocks, deltas = hit
        return blocks, {k: list(v) for k, v in deltas.items()}

    def _row_blocks(self, agent: str, r: int, excluded: set):
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
            del U
            k = nX * N                                    # I - P U_X built F-ordered and factored in place (no eye, no copy):
            A = np.subtract(0.0, PU[:, :k], order="F")    # 0 - x off the diagonal and 1 - x on it, the entries eye - x has
            A[np.arange(k), np.arange(k)] = 1.0 - PU[np.arange(k), np.arange(k)]
            lu = lu_factor(A, overwrite_a=True)
            del A
            W = lu_solve(lu, PU[:, k:])
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
        if self.sym is not None and not self.instant_loads and not self.levels and not isinstance(excluded, (tuple, list)) and self._maps_symmetric(maps):
            return self.closed_loop_symmetric(maps, excluded, impulse_controls)
        return self._closed_loop_eliminated(maps, excluded, impulse_controls)

    def _interp0(self) -> np.ndarray:
        """(N,) the read of a kernel at age 0+ (grid.interp([0.0])[0]), made once."""
        if getattr(self, "_at0", None) is None:
            self._at0 = self.grid.interp([0.0])[0]
        return self._at0

    def _closed_loop_eliminated(self, maps, excluded=None, impulse_controls=()):
        nX, nU, N = self.nX, self.nU, self.N
        n = len(self.prim) * N; nxs = nX * N
        ncol = self.nW + len(impulse_controls)
        off = set(excluded) if isinstance(excluded, (tuple, list)) else ({excluded} if excluded else set())
        excl = frozenset(u for a in self.model.agents if a.name in off for u in a.controls)
        B = np.zeros((n, ncol))
        if nX:
            lu, W, P0X, perm = self._state_elimination(excl)
            # the state rows of the forcing and their elimination G B_X are map-independent: once per (excluded,
            # impulse controls) (the monitored markets solve the closed loop thousands of times)
            skey = (excl, tuple(impulse_controls))
            hit = self._state_forcing.get(skey)
            if hit is None:
                Bx = np.zeros((nxs, ncol))
                for k in range(self.nW):
                    Bx[:, k] += P0X @ self.sigma[:, k]
                for j, u in enumerate(impulse_controls):
                    col = self.nW + j
                    for i, (nm, lag), c in self.state_inputs:
                        if nm != u:
                            continue
                        v = np.zeros(nX); v[i] = c
                        Bx[:, col] += (P0X @ v) if lag == 0 else (self.grid.jump_injector(self.A, lag)[perm] @ v)
                hit = (Bx, _lu_solve(lu, Bx))
                if len(self._state_forcing) < 64:
                    self._state_forcing[skey] = hit
            B[:nxs] = hit[0]
            GB = hit[1]
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
                    q = (self.levels or {}).get(a.name, {}).get(r)
                    if q is not None:                            # a level row: the map on the quantity's increments,
                        if q not in excl:                        # g * q' + g q(0+), in the kernel of q
                            MU[bl, self.block(q)] += C @ self.grid.diff() + np.outer(gur, self._interp0())
                        elif q in impulse_controls:              # a spike of q: g' after it (the block g(0+) it draws at
                            col = self.nW + list(impulse_controls).index(q)   # once is in the composite, use_maps)
                            B[nxs + bl.start:nxs + bl.stop, col] += self.grid.diff() @ gur
        # instant observations: an observer's control moves with the level it sees, contemporaneously (its map covers
        # the rest of its action, see _map_part)
        for v, loads in (self.instant_loads or {}).items():
            if v in excl or any(v in a.controls for a in self.model.agents if a.name in off):
                continue
            bl = slice(self.block(v).start - nxs, self.block(v).stop - nxs)
            for u, h in loads.items():
                blk = MU[bl, self.block(u)]
                blk[np.arange(N), np.arange(N)] += h                    # h I
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
        Switching one agent off (the passive world) breaks the symmetry: the others' rows, the representative's with
        their columns shifted, are then one real system on the remaining controls, solved directly."""
        st = self._mode_structure(); m, omega = st["m"], st["omega"]
        nX, N = self.nX, self.N; n = len(self.prim) * N; nxs = nX * N
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
        GB = _lu_solve(lu, B[:nxs])
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
        r = len(corb); nf = fcols.size
        if ex_agent is not None:
            # one agent switched off (the passive world): its controls are zero and the others' rows, the representative's
            # with their columns shifted along the cycle, form one real system on the remaining controls, solved directly
            # (a Woodbury correction on the mode solves took nW + |its controls| N complex right-hand sides)
            others = [u for u in all_controls if u not in ex_agent.controls]
            idx_o = np.concatenate([np.arange(all_controls.index(u) * N, (all_controls.index(u) + 1) * N) for u in others])
            fixed = [x for x in rep_controls if x not in st["member"]]
            Aoo = np.empty((idx_o.size, idx_o.size), order="F")
            for i, u in enumerate(others):
                if u in st["member"]:
                    j, sh = st["member"][u]
                    rows_u = R_rep[j * N:(j + 1) * N][:, cperms[(-sh) % m][idx_o]]
                else:
                    fi = fixed.index(u)
                    rows_u = R_rep[(r + fi) * N:(r + fi + 1) * N][:, idx_o]
                np.subtract(0.0, rows_u, out=Aoo[i * N:(i + 1) * N])
            Aoo[np.arange(idx_o.size), np.arange(idx_o.size)] += 1.0
            ZU = np.zeros_like(rhs)
            ZU[idx_o] = _lu_solve(lu_factor(Aoo, overwrite_a=True, check_finite=False), rhs[idx_o])
            Z = np.zeros((n, ncol)); Z[nxs:] = ZU; Z[:nxs] = GB if wzero else W @ ZU + GB
            return Z
        # mode blocks
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
                zk = _lu_solve(blocks[k], bk)
                weight = 1.0 if (k == 0 or 2 * k == m) else 2.0                  # the pair (k, m - k) or a self-conjugate mode
                for j in range(r):
                    for sh in range(m):
                        out[csl[j][sh]] += weight * ((omega ** (sh * k)) * zk[j * N:(j + 1) * N]).real / np.sqrt(m)
                if k == 0 and nf:
                    out[fcols] += zk[r * N:].real
            return out
        ZU = solve_sym(rhs)
        Z = np.zeros((n, ncol)); Z[nxs:] = ZU; Z[:nxs] = GB if wzero else W @ ZU + GB
        return Z

    def closed_loop_dense(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None,
                          impulse_controls: Sequence = ()):
        """closed_loop as one dense solve over every primary kernel (the reference for the elimination; tests)."""
        if self.levels:
            raise NotImplementedError("closed_loop_dense does not assemble level rows; use closed_loop")
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
class StationarySolver(MonitoredDeviations, EngineBase):
    RESULT = StationaryResult
    MONITORING = True                   # monitored deviations (Chapter 6): monitoring.MonitoredDeviations
    RISK_SENSITIVE = True               # risk-averse agents under consistent planning (stationary_risk.py), see __init__
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.6, 60       # with Anderson memory 15 (0.3 was needed at memory 6 for Kyle-Back)
    #  The second-order verdict does not depend on the discount, so this engine can check a
    #  discounted model.  The dissertation writes the discounted stationary objective (Chapter
    #  "Stationary infinite-horizon LQG", eq. stationary-objective) as
    #
    #      J = 1/2 E int_0^inf e^{-rho t} [X' G^XX X + 2 G^X' X + 2 D' G^DX X + D' G^DD D] dt
    #
    #  "with the joint running Hessian positive semidefinite".  Where that Hessian is semidefinite
    #  the form is at every rho; where it is not (a trader's D (P - V)) the sign depends on rho, and
    #  the second-order check is made on the discounted objective's own form, the responses weighted
    #  by e^{-rho tau / 2} (_half_discounted; at rho = 0 the average-cost form).

    def __init__(self, model: Model, verbose: bool = False, settings=None):
        """settings: the tuning constants (noisestate.Settings, or a dict of its fields; the defaults when None)."""
        if model.horizon.kind == "transition":
            raise ValueError(f"horizon.kind 'transition' ({model.name!r}) runs on the spectral finite engine only "
                             "(noisestate.solve routes it there; this engine has no past)")
        super().__init__(model, verbose, settings=settings)
        self.c = Compiled(model, settings=self.settings)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}
        self._monitored = None                          # (maps key, {agent: monitored R}, {origin: seed world}, residual) of the last maps
        self._mean_tails: Dict[str, float] = {}        # agent -> its passive response's level at L where the means use the window's integral (_passive_dc)
        self._spike_cache = (None, {})                  # (maps key, {(agent, excluded): (Zpass, R)}) of the last maps (_spikes)
        self._atom_blocks: Dict[str, tuple] = {}        # agent -> the loss atoms' blocks and which are the identity
        self._kernel_residual = 0.0                     # the inner solve's residual at the last maps
        self._risk_scale = 1.0                          # the continuation's step in risk aversion (solve): theta times this
        self._risk_info: Dict[str, dict] = {}           # agent -> the last correction's diagnostics
        averse = [a for a in model.agents if a.risk_aversion]
        if averse:
            why = []
            if not self.c.rho > 0:
                why.append("a discount (horizon.discount > 0)")
            if any(len(model.privy(a.name)) > 1 for a in model.agents) or any(a.instant for a in model.agents) or self.c.levels:
                why.append("no monitoring, instant observations or level rows")
            if any(l != 0 for a in averse for term in a.loss for at in term[1:] for (n, l) in model.expand({at: 1.0})):
                why.append("no lagged or leading atoms in a risk-averse agent's loss")
            if any(l != 0 for a in averse for (n, l) in (self.c.integrals or {}).get(a.name, ([], None))[0]):
                why.append("no lagged atoms in its integrals")
            if any(n in a.controls for a in averse for (n, l) in (self.c.integrals or {}).get(a.name, ([], None))[0]):
                why.append("no current control of its own in its integrals (its spike would load on the shock of its own instant)")
            if self._mean_driven():
                why.append("no means")
            if why:
                raise NotImplementedError(f"risk-averse agents ({', '.join(a.name for a in averse)}) on the stationary engine (consistent "
                                          "planning, stationary_risk.py) need " + "; ".join(why))

    # -------------------------------------------- overridable model pieces
    # ------------------------------------------- the best response (the kernel algebra of the age grid)
    def _row_support(self, agent: Agent, rows):
        """Which (row, channel) regular kernels are not identically zero, (nR, nW); the operators of the
        zero ones are zero and are skipped (a channel the agent's rows never carry, a row that reads
        nothing regular).  Rows are grouped by observation delay so one batched kernel call serves all
        rows of a delay."""
        sup = np.stack([y.any(axis=0) for y in rows]) if rows else np.zeros((0, self.c.nW), dtype=bool)
        memo = self.__dict__.setdefault("_row_groups", {})         # map-independent: once per (agent, number of rows)
        key = (agent.name, len(rows))
        groups = memo.get(key)
        if groups is None:
            groups = {}
            for r in range(len(rows)):
                groups.setdefault(float(self.c.rows[agent.name][r][3]), []).append(r)
            memo[key] = groups
        return sup, groups

    def _row_operator(self, agent: Agent, rows, inst):
        """Per channel, the operator (N x nR N) mapping stacked row maps gamma to the action kernel:
        c_k = sum_r (Conv[y_rk] + E_rk S_delta) gamma_r."""
        return self._row_operator_parts(agent, rows, inst)[0]

    def _row_operator_parts(self, agent: Agent, rows, inst):
        """(Gk, pre): _row_operator's result and, per (row, channel) block that also has instantaneous entries, a copy
        of its regular part before they are added, {(r, k): (N, N)} (_foc_system reads the regular blocks from them)."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        Gk = np.zeros((nW, N, nR * N))
        sup, groups = self._row_support(agent, rows)
        for d, rs in groups.items():
            pairs = [(r, k) for r in rs for k in np.where(sup[r])[0]]
            if pairs:
                ops = c.conv_rows(np.stack([rows[r][:, k] for r, k in pairs], axis=1), d)     # one call per delay
                for i, (r, k) in enumerate(pairs):
                    Gk[k, :, r * N:(r + 1) * N] = ops[i]
        pre = {}
        for r in range(nR):
            for (k, age, w) in inst[r]:
                S = c.instant(age, c.rows[agent.name][r][3])
                blk = Gk[k, :, r * N:(r + 1) * N]
                if (r, k) not in pre:
                    pre[(r, k)] = blk.copy()
                if self._shift_is_eye(S):
                    blk[np.arange(N), np.arange(N)] += w                # w I
                else:
                    blk += w * S
        return Gk, pre

    def _response_operators(self, agent: Agent, R: np.ndarray, keep_own: bool = False):
        """Per control, the operator (n_prim N x N) giving the primary kernels' response to that
        control's action kernel: Z = Zpass + Resp_u c_u, with the own block equal to the action (keep_own: the action
        plus what R's own block carries, an instant observer's own reactions in a monitored R)."""
        c = self.c; N = c.N
        out = []
        for ui, u in enumerate(agent.controls):
            Cu = c.response(R[:, ui].reshape(len(c.prim), N), c.prim.index(u))
            if keep_own:
                Cu[c.block(u)] += np.eye(N)
            else:
                Cu[c.block(u)] = np.eye(N)
            for v, coef in (c.composite or {}).get(u, {}).items():   # an instant reaction moves with the action itself
                if v != u:
                    Cu[c.block(v)] += coef * np.eye(N)
            out.append(Cu)
        return out

    def _foc_static(self, agent: Agent):
        """The map-independent pieces of _foc_operators, built once per agent: per atom its block (primary index,
        N x N shift) and whether the shift is the identity; Q' contiguous; per control the atoms its continuation
        enters (the others' quantities), its own delayed reads, the leads and its own controls' atoms (whose continuation
        only envelope=False adds)."""
        st = self._atom_blocks.get(agent.name)
        if st is None:
            c = self.c
            atms, Q, q = c.loss[agent.name]
            # an atom operator is one N x N block, the shift into the atom's own primary: the position is the
            # atom's, so ask for the block rather than build the full-width operator and scan its zeros for it
            AO = [c.atom_block(at) for at in atms]
            eye = [_is_eye(blk) for p, blk in AO]
            idx = {at: j for j, at in enumerate(atms)}
            per_u = []
            for u in agent.controls:
                cont = np.array([j for j, (name, lag) in enumerate(atms) if name not in agent.controls], dtype=np.intp)
                lead = [(j, name, lag) for j, (name, lag) in enumerate(atms) if name not in agent.controls and lag < 0]
                own_lag = [(j, lag) for j, (name, lag) in enumerate(atms) if name == u and lag > 0]
                own = np.array([j for j, (name, lag) in enumerate(atms) if name in agent.controls], dtype=np.intp)
                per_u.append((cont, lead, own_lag, own))
            # the identity atoms alone on their primary's block: their block of the operator is 0 + MQ_i, written at
            # once (a zero MQ_i writes the zeros the block holds); the other atoms are added one by one in order
            ps = [p for p, _ in AO]
            sole = [i for i in range(len(atms)) if eye[i] and ps.count(ps[i]) == 1]
            rest = np.array([i for i in range(len(atms)) if i not in sole], dtype=np.intp)
            sole = (np.array(sole, dtype=np.intp), np.array([ps[i] for i in sole], dtype=np.intp))
            st = self._atom_blocks[agent.name] = (AO, eye, idx, np.ascontiguousarray(Q.T), per_u, sole, rest)
        return st

    def _foc_operators(self, agent: Agent, R: np.ndarray, atoms: bool = False, envelope: bool = True):
        """Per control, the operator (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses R, delayed reads of own lagged controls, and the past-date term of a lead.
        Called with the physical impulse responses (all reactions off) for the wedge decomposition.
        With atoms=True returns (Fu, Ms), Ms the per-control operators M (n_atoms, N, N) on the loss
        atoms' kernels that Fu contracts with Q (the mean part applies them to the targets q)."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atms = c.loss[agent.name][0]; na = len(atms)
        AO, AO_eye, aidx, QT, per_u, (sole, sole_p), rest = self._foc_static(agent)   # AO_eye: an undelayed atom reads through the identity
        comp = c.seed_composite(agent.name) if hasattr(c, "seed_composite") else c.composite   # its own deviations
        Fu, Ms = [], []
        for ui, u in enumerate(agent.controls):
            # op = sum_j M_j (Q zeta)_j with M_j the operator on atom j: identity for the instantaneous
            # term, continuation, delayed own read, lead term; contracted as sum_i (sum_j Q_ji M_j) AO_i
            # so the products are N x N x N per atom block instead of N x N x n_prim N per atom
            cont, lead, own_lag, own = per_u[ui]
            M = np.zeros((na, N, N)); Mf = M.reshape(na, N * N)
            for v, coef in (comp or {}).get(u, {u: 1.0}).items():   # the control and the instant reactions it draws
                j = aidx.get((v, 0.0))
                if j is not None:
                    Mf[j, ::N + 1] += coef                          # + coef I
            if not agent.myopic:
                # impulse responses of every atom: its block against its own primary's rows of R (an identity
                # block reads the rows themselves; + 0.0 as the sum from 0 the products were added to)
                Rj = np.empty((N, na))
                for j, (p, A) in enumerate(AO):
                    Rj[:, j] = R[p * N:(p + 1) * N, ui] if AO_eye[j] else A @ R[p * N:(p + 1) * N, ui]
                Rj += 0.0
                CR = c.continuation(Rj)
                for j, lag in own_lag:                              # delayed read of the control itself
                    M[j] += np.exp(-c.rho * lag) * c.own_lag_read(lag)
                if len(cont):                                       # the others' quantities' continuation
                    M[cont] += CR[cont]
                if not envelope and len(own):                       # the own later reactions (R with the own map on:
                    M[own] += CR[own]                               # stationary_risk); by default the envelope drops them
                elif len(own) and c.instant_loads:
                    # the agent's own instant reactions in R (a player privy to it answers its spike, and the agent reacts
                    # at once to the level it sees): a fixed reaction, not a choice the envelope drops; its continuation
                    # through the others' quantities is in R already, so its own atoms join it (the whole derivative)
                    live = own[np.abs(Rj[:, own]).max(axis=0) > 0.0]
                    if len(live):
                        M[live] += CR[live]
                for j, name, lag in lead:
                    M[j] += self._lead_term(agent, R[:, ui], name, lag)
            MQ = np.dot(QT, Mf)                                     # MQ[i] = sum_j Q[j, i] M_j (tensordot's product)
            nz = MQ.any(axis=1)
            MQ = MQ.reshape(na, N, N)
            op = np.zeros((N, n_prim))
            if sole.size:
                op.reshape(N, -1, N)[:, sole_p, :] = (MQ[sole] + 0.0).transpose(1, 0, 2)
            for i in rest[nz[rest]]:
                p, blk = AO[i]
                op[:, p * N:(p + 1) * N] += MQ[i] if AO_eye[i] else MQ[i] @ blk
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

    def _decompose(self, agent: Agent, out: dict, Fu, Resp, Gk, maps=None, Resp_so=None) -> None:
        """The second-order check and the FOC decomposition (instantaneous/physical/wedge).  Tied agents
        share the second-order check of their representative (the same problem up to relabelling).  Resp_so: the
        second-order form's response operators (_half_discounted at rho > 0) [Resp]."""
        c = self.c; nW = c.nW; Zfull = out["Zfull"]
        Rso = Resp if Resp_so is None else Resp_so
        out["second_order"] = self._shared_second_order(
            agent, lambda: self._second_order(agent, Rso, Gk, np.tile(self._identified(agent), len(agent.controls)), maps))
        Fphys = self._foc_operators(agent, self._physical_responses(agent, c.nW))
        dec = {}
        for ui, u in enumerate(agent.controls):
            phi = np.stack([Fu[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            phi_phys = np.stack([Fphys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
            if out.get("risk_shift") is not None:
                # a risk-averse agent: the kernel whose projection vanishes is S f^on (stationary_risk.py), the risk-neutral
                # one plus the frozen shift; physical + wedge + risk = foc
                dec[u]["risk"] = out["risk_shift"][ui]
                dec[u]["foc"] = phi + out["risk_shift"][ui]
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
        of M scaled by max.  At a discount rho > 0, Resp are the responses weighted by e^{-rho tau / 2}
        (_half_discounted) and the loss form's lagged atoms likewise: the discounted objective's own form,
        the average-cost form at rho = 0.
        The objective is truncated at the window, so a strategy can push a little loss past the edge:
        curvatures within SECOND_ORDER_TOL of the largest are treated as that, not as a saddle.
        When the form is not positive and the engine defines _embedded_curvature(agent, maps, idx,
        vmin) (an optional hook, the stationary engine's), that is asked for the curvature of the
        offending direction on a longer window: a float, or None when it cannot say; a positive
        value turns the verdict into ok with "edge" and "embedded" recorded."""
        c = self.c
        N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        GAO = self._loss_form(agent, half_discount=c.rho > 0)                    # symmetric loss form on the world
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

    def _loss_form(self, agent: Agent, start_from: bool = False, half_discount: bool = False) -> "_BlockForm":
        """The loss form on the primary kernels, AO' kron(Q, mass) AO for the stacked atom operators AO,
        assembled block by block over the atoms' primary blocks (each atom reads one primary through one
        N x N block; an undelayed atom through the identity, whose products are skipped) and held as those blocks
        (_BlockForm: the (n_prim N)^2 array only for a product with it).  Map-independent, cached per agent.  With start_from=True the form of a past's initial-shock column: the same atoms under
        the point mass of the line s = 0 (_init_mass), as expected_cost integrates those columns.  With
        half_discount=True the second-order form's (_half_discounted): an atom read at lag l is a response older by l,
        so Q_ij carries e^{-rho (l_i + l_j) / 2} (a lead's negative lag raises it)."""
        key = (agent.name, start_from) if not half_discount else (agent.name, start_from, "half")
        if key not in self._loss_forms:
            c = self.c; N = c.N; n = len(c.prim) * N
            atoms, Q, q = c.loss[agent.name]
            if start_from:
                raise NotImplementedError("the form of an initial-shock column is the spectral finite engine's (finite_free)")
            mass = c.cost_mass()
            if half_discount:
                lg = np.array([float(l) for (_, l) in atoms])
                Q = Q * np.exp(-0.5 * c.rho * (lg[:, None] + lg[None, :]))
            blocks = [[c.atom_block(at)] for at in atoms]
            GB: Dict[Tuple[int, int], np.ndarray] = {}                 # the nonzero blocks, summed from zeros as the dense array was
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    if Q[i, j] == 0.0:
                        continue
                    W = Q[i, j] * mass
                    for p, Ai in blocks[i]:
                        for p2, Aj in blocks[j]:
                            WA = W if _is_eye(Aj) else W @ Aj
                            if (p, p2) not in GB:
                                GB[(p, p2)] = np.zeros((N, N))
                            GB[(p, p2)] += WA if _is_eye(Ai) else Ai.T @ WA
            self._loss_forms[key] = _BlockForm(n, N, GB)
        return self._loss_forms[key]

    def _causal_chunks(self):
        """Node ranges [(lo, hi)] in increasing age such that the regular projection operator of any row is
        zero from ages in one chunk to nodes in an earlier one (a correlation reads only older ages); the
        products over those blocks are skipped (c.causal_chunks; the stationary Compiled declares them)."""
        return self.c.causal_chunks()

    def _foc_system(self, agent: Agent, rows, inst, Zpass: np.ndarray, Resp, Fu, shift=None, Gk=None, Gpre=None, lazy: bool = False, Gfull=None):
        """The first-order-condition system Amat gamma = -bvec on the passive rows,
        Amat[u, v] = sum_k H_k (Fu_u Resp_v) G_k, bvec[u] = sum_k H_k (Fu_u Zpass)_k, with G_k the row
        operator and H_k the projection operator of channel k.  Both split into a regular part (the
        convolution and correlation with the row kernels, zero for every (row, channel) whose kernel is
        zero) and the instantaneous entries (scaled shifts on one row block); the regular parts are
        assembled over the nonzero rows and channels only, with the projection's columns ordered
        (node, channel) so the product Fu Resp G comes out in the right layout without a transpose, and
        the instantaneous terms are added block by block.  Identical to the dense assembly to round-off.
        Gk, when given, is _row_operator's result on the same rows and Gpre its blocks' regular parts where it added
        instantaneous entries (_row_operator_parts): the convolution operators this would build again (the same call
        on the same kernels) are read from them instead; Gpre is emptied once read.
        With lazy=True returns (assemble, bvec, matvec) instead: assemble() builds Amat, matvec(gamma) is Amat @ gamma
        from the factors (Gfull, the whole row operator (nW, N, nR N), applied to every channel at once), which the
        best response's Krylov solve uses without Amat (_foc_gamma)."""
        c = self.c; N = c.N; nR, nU = len(rows), len(Fu)
        sup, groups = self._row_support(agent, rows)
        delay = [c.rows[agent.name][r][3] for r in range(nR)]
        Rn = np.where(sup.any(axis=1))[0]; Kn = np.where(sup.any(axis=0))[0]
        nRn, nKn = len(Rn), len(Kn)
        kpos = {int(k): i for i, k in enumerate(Kn)}
        # regular parts: Hs[(ri, a), (j, ki)] and Gs[a, (ki, ri, j)]; with the row operator given, Gs is read from it when
        # Amat is assembled (lazy: only then)
        Hs = np.zeros((nRn, N, N, nKn)); Gs = np.zeros((N, nKn, nRn, N)) if Gk is None else None
        read = []                                                               # (ri, k) of the row operator's blocks
        for d, rs in groups.items():
            pairs = [(ri, int(k)) for ri, r in enumerate(Rn) if r in rs for k in np.where(sup[r])[0]]
            if not pairs:
                continue
            Y = np.stack([rows[Rn[ri]][:, k] for ri, k in pairs], axis=1)
            Hp = c.projection_rows(Y, d).reshape(N, len(pairs), N)             # (a, pair, j)
            if Gk is None:
                Gp = c.conv_rows(Y, d)                                          # (pair, a, j)
            for i, (ri, k) in enumerate(pairs):
                Hs[ri, :, :, kpos[k]] = Hp[:, i, :]
                if Gk is None:
                    Gs[:, kpos[k], ri, :] = Gp[i]
                else:
                    read.append((ri, k))
            del Y, Hp                                                           # (N pairs N) arrays: not held through the assembly
            Gp = None

        def regular_G():
            """Gs from the row operator's blocks (the same product), the regular parts where it added instantaneous entries."""
            G = np.zeros((N, nKn, nRn, N))
            for ri, k in read:
                r = int(Rn[ri])
                G[:, kpos[k], ri, :] = Gpre[(r, k)] if (r, k) in Gpre else Gk[k][:, r * N:(r + 1) * N]
            if Gpre:
                Gpre.clear()                                                    # read into Gs: not held through the assembly
            return G.reshape(N, nKn * nRn * N)
        Hs = Hs.reshape(nRn * N, N * nKn)
        if Gs is not None:
            Gs = Gs.reshape(N, nKn * nRn * N)
        chunks = self._causal_chunks() if nKn else []
        # instantaneous entries: (row, channel, weighted shift on the row operator, on the projection); a shift that is the
        # identity (an undelayed row's own noise) is applied as the scaling by its weight, which is what the product gives
        ent = []
        for r in range(nR):
            for (k, age, w) in inst[r]:
                G, H = c.instant(age, delay[r]), c.instant_adjoint(age, delay[r])
                eg, eh = self._shift_is_eye(G), self._shift_is_eye(H)
                ent.append((r, k, None if eg else w * G, None if eh else w * H, (w if eg else None), (w if eh else None)))
        rmul = lambda X, S, w: X * w if w is not None else X @ S            # X @ S with S = w I
        lmul = lambda S, w, X: w * X if w is not None else S @ X            # S @ X with S = w I
        nG = nU * nR * N
        Hs3 = Hs.reshape(nRn * N, N, nKn)

        def project(phi, out):
            """out (nR, N) += H phi for a FOC kernel phi (N, nW): the regular rows and the instantaneous entries."""
            if nKn:
                out[Rn] += (Hs @ phi[:, Kn].reshape(-1)).reshape(nRn, N)
            for (r, k, Sg, Sh, wg, wh) in ent:
                out[r] += lmul(Sh, wh, phi[:, k])
        bvec = np.zeros(nG); B3 = bvec.reshape(nU, nR, N)
        for ui in range(nU):
            phi = Fu[ui] @ Zpass                                                # (N, nW): the FOC of the passive world
            if shift is not None:
                phi = phi + shift[ui]                                           # a risk-averse agent's frozen correction
            project(phi, B3[ui])
        FR = [[Fu[ui] @ Resp[vi] for vi in range(nU)] for ui in range(nU)]

        def assemble():
            Gs_ = Gs if Gs is not None else regular_G()
            Hc = [np.ascontiguousarray(Hs.reshape(nRn, N, N * nKn)[:, lo:hi, lo * nKn:]).reshape(nRn * (hi - lo), (N - lo) * nKn)
                  for lo, hi in chunks]                                         # rows of ages in the chunk, columns of nodes not younger
            Amat = np.zeros((nG, nG))
            A6 = Amat.reshape(nU, nR, N, nU, nR, N)
            for ui in range(nU):
                for vi in range(nU):
                    FRuv = FR[ui][vi]
                    FRG = (FRuv @ Gs_).reshape(N, nKn, nRn * N) if nKn else None     # [(j, ki), (ri, j')]
                    for (lo, hi), H_ in zip(chunks, Hc):
                        T = (H_ @ FRG[lo:].reshape((N - lo) * nKn, nRn * N)).reshape(nRn, hi - lo, nRn, N)
                        for ri, r in enumerate(Rn):
                            A6[ui, r, lo:hi, vi][:, Rn, :] += T[ri]
                    for (r, k, Sg, Sh, wg, wh) in ent:
                        if k in kpos:                                               # the channel also has regular kernels
                            ki = kpos[k]
                            X = (Hs3[:, :, ki] @ rmul(FRuv, Sg, wg)).reshape(nRn, N, N)     # H_reg FR G_inst
                            for ri, rr in enumerate(Rn):
                                A6[ui, rr, :, vi, r] += X[ri]
                            A6[ui, r, :, vi][:, Rn, :] += lmul(Sh, wh, FRG[:, ki, :]).reshape(N, nRn, N)        # H_inst FR G_reg
                        for (r2, k2, Sg2, Sh2, wg2, wh2) in ent:
                            if k2 == k:
                                A6[ui, r, :, vi, r2] += lmul(Sh, wh, rmul(FRuv, Sg2, wg2))                       # H_inst FR G_inst
            return Amat
        if not lazy:
            return assemble(), bvec
        nWk = Gfull.shape[0]
        Gfull = Gfull.reshape(-1, nR * N)

        def matvec(gamma):
            """Amat @ gamma without Amat: sum_k H_k (sum_v FR_uv (G_k gamma_v)), the row operator Gk applied to every
            channel at once (its regular and instantaneous parts together, as Amat's four terms)."""
            g3 = gamma.reshape(nU, nR * N)
            C = (Gfull @ g3.T).reshape(nWk, N, nU)                               # (k, node, v): the action kernels
            out = np.zeros(nG); O3 = out.reshape(nU, nR, N)
            for ui in range(nU):
                phi = FR[ui][0] @ C[:, :, 0].T
                for vi in range(1, nU):
                    phi += FR[ui][vi] @ C[:, :, vi].T
                project(phi, O3[ui])
            return out
        return assemble, bvec, matvec

    def _shift_is_eye(self, S: np.ndarray) -> bool:
        """_is_eye of one of the compiled model's cached shifts (the same array every call), remembered per array."""
        memo = self.__dict__.setdefault("_eye_memo", {})
        hit = memo.get(id(S))
        if hit is None or hit[0] is not S:
            hit = memo[id(S)] = (S, _is_eye(S))
        return hit[1]

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
        c.use_maps(maps)
        Zpass, R0 = self._spikes(c, maps, agent)
        # two uses of the impulse responses, kept apart: the on-path world is the passive world plus the agent's
        # actions as every other player sees them on the path (their filters, R0); the first-order condition and
        # the second-order form are about the agent's deviations, to which the players privy to it respond
        # through their response kernels (the monitored R; R0 itself without monitoring)
        R = self._impulse_responses(agent, maps, R0)
        Zpass = self._passive_world(agent, maps, Zpass, R0)
        Resp0 = self._response_operators(agent, R0)
        close = self._instant_closure(agent, Resp0)
        if close is not None:
            # the agent's own instant reactions (h times the levels it sees) are part of its passive world: its map is
            # off, the reaction the loss fixes is not; a best response is its map on the rows of that world
            Zpass = close(Zpass)
            Resp0 = [close(Rv) for Rv in Resp0]
        # the deviations' responses: the monitored R carries the agent's own instant reactions to what the privy players
        # answer (their seed worlds close it), its own block among them
        Resp = Resp0 if R is R0 else self._response_operators(agent, R, keep_own=close is not None)
        ytil, yinst = self._passive_rows(agent, Zpass)
        if c.levels and agent.name in c.levels:
            self._silent = getattr(self, "_silent", {})
            scale = max([1e-300] + [float(np.abs(y[:, :c.nW]).max()) for y in ytil])
            self._silent[agent.name] = {r for r in c.levels[agent.name]            # relative: the zero start leaves round-off
                                        if max([0.0] + [abs(w) for (_, _, w) in yinst[r]]) <= 1e-9 * scale
                                        and float(np.abs(ytil[r][:, :c.nW]).max()) <= 1e-9 * scale}
        if type(self)._row_operator is StationarySolver._row_operator:
            Gk, Gpre = self._row_operator_parts(agent, ytil, yinst)
        else:                                                   # an overridden row operator: _foc_system builds its own
            Gk, Gpre = self._row_operator(agent, ytil, yinst), None
        Fu = self._foc_operators(agent, R)
        # the FOC is affine in gamma: solve H (Fu (Zpass + sum_v Resp0_v Gk gamma_v)) = 0 for all controls
        shift = None
        th = float(agent.risk_aversion) * self._risk_scale
        if th:
            from .stationary_risk import StationaryTilt
            tl = StationaryTilt(self, agent, maps, th)
            shift = tl.shift()
            self._risk_info[agent.name] = {"richardson_gap": tl.richardson_gap, "gmres_residual": max(r for (_, r) in tl.gmres)}
        gamma = self._foc_gamma(agent, ytil, yinst, Zpass, Resp0, Fu, shift, Gk, Gpre).reshape(nU, nR, N)
        cact = np.stack([(Gk @ gamma[ui].reshape(-1)).T for ui in range(nU)])
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp0[ui] @ cact[ui]
        if close is not None:                                   # the action: the map's part and the instant reaction
            cact = np.stack([Zfull[c.block(u)] for u in agent.controls])
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if shift is not None:
            out["risk_shift"] = shift
        if want_decomp:
            self._decompose(agent, out, Fu, Resp, Gk, maps, self._half_discounted(agent, R0, R) if c.rho > 0 else Resp)
        return (self._project(agent, Zfull, self._map_part(agent, Zfull, cact)) if project else None), out

    def _half_discounted(self, agent: Agent, R0: np.ndarray, R: np.ndarray):
        """The response operators of the second-order form at a discount rate rho > 0: every response to the agent's
        action weighted by e^{-rho tau / 2} at its age tau (the action itself, tau = 0, by one), then closed under the
        agent's own instant reactions and, with monitoring, taken from the monitored R, as best_response does.

        The objective is E int_0^inf e^{-rho t} l(t) dt from a date on which the agent's deviation starts, so its second
        variation weighs a pair of responses to actions at s and s', seen at t, by e^{-rho t}; with the deviation written
        e^{rho s / 2} x_s, x stationary (a map on the rows, times a deterministic factor: as feasible as x), that is
        e^{-rho (t - s) / 2} e^{-rho (t - s') / 2}, a weight on each response's own age, and the form per unit time of x
        is the average-cost form with the responses so weighted.  This family of deviations is the whole of it: the form
        on [0, inf) in the variable x is a Toeplitz form whose symbol is this form's, so it is positive on every feasible
        deviation exactly when this form is.  At rho = 0 it is the average-cost form.  (The average-cost form itself --
        the responses unweighted -- is the curvature of the flow loss of a deviation made at every date, the past's
        included, which no player can make; with a loss that is not positive semidefinite, a trader's, its sign is
        not the discounted objective's: Chapter 6's trader at window 8 reads -0.062 there and +0.0059 here.)  Exponential
        weights commute with the convolutions of the closure, so weighting R before closing is weighting the closed
        responses."""
        c = self.c
        w = np.tile(np.exp(-0.5 * c.rho * c.grid.nodes), len(c.prim))[:, None]
        Resp0 = self._response_operators(agent, w * R0)
        close = self._instant_closure(agent, Resp0)
        if close is not None:
            Resp0 = [close(Rv) for Rv in Resp0]
        return Resp0 if R is R0 else self._response_operators(agent, w * R, keep_own=close is not None)

    def _instant_closure(self, agent: Agent, Resp):
        """For an agent with instant observations, the map Z -> Z' closing a world Z (computed with the agent's controls
        off) under the agent's own instant reactions: its control v moves by sum_u h_vu Z'[u] (compile.instant_loads),
        which moves the world through Resp_v, the response operators of its controls (the own block the action).
        Z' = Z + A x, x = (I - B A)^-1 B Z, with A = [Resp_v] and B the loadings on the seen levels.  None without
        instant loadings (the world is closed already)."""
        c = self.c; N = c.N
        loads = c.instant_loads or {}
        pairs = [(vi, u, h) for vi, v in enumerate(agent.controls) for u, h in loads.get(v, {}).items() if h]
        if not pairs:
            return None
        nU = len(agent.controls)
        BA = np.zeros((nU * N, nU * N))
        for vi, u, h in pairs:
            for wi in range(nU):
                BA[vi * N:(vi + 1) * N, wi * N:(wi + 1) * N] += h * Resp[wi][c.block(u)]
        lu = lu_factor(np.eye(nU * N) - BA)

        def close(Z):
            BZ = np.zeros((nU * N,) + Z.shape[1:])
            for vi, u, h in pairs:
                BZ[vi * N:(vi + 1) * N] += h * Z[c.block(u)]
            x = lu_solve(lu, BZ)
            out = Z.copy()
            for wi in range(nU):
                out += Resp[wi] @ x[wi * N:(wi + 1) * N]
            return out
        return close

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
    #  The iteration and its algebra are monitoring.py's (MonitoredDeviations); the pieces on the age grid follow.

    def _spikes(self, c, maps, agent: Agent, excluded=None):
        """(Zpass, R): the closed loop with `excluded` (default the agent) switched off, and the responses to a spike
        of each of the agent's controls together with the instant reactions it draws (c.composite: a trader seeing
        the quote trades at once), (n_prim N, nU).  Without instant observations the spikes are the controls' own."""
        if hasattr(c, "use_maps"):
            c.use_maps(maps)                                 # level rows: the reactions a spike draws depend on the maps
        # the same closed loop is asked for more than once at one set of maps (an agent's best response, then the
        # monitored responses' naive start for every player privy to someone): kept for the last maps, copies out
        ck = None
        if c is self.c and (excluded is None or isinstance(excluded, str)):
            key = self._maps_key(maps)
            if self._spike_cache[0] != key:
                self._spike_cache = (key, {})
            ck = (agent.name, excluded)
            hit = self._spike_cache[1].get(ck)
            if hit is not None:
                return hit[0].copy(), hit[1].copy()
        comp = c.composite or {}
        ctrls = spike_controls(agent.controls, comp)
        Zp = c.closed_loop(maps, excluded=agent.name if excluded is None else excluded, impulse_controls=ctrls)
        spike = compose_spikes(Zp[:, c.nW:], ctrls, agent.controls, comp)
        R = np.stack([spike[u] for u in agent.controls], axis=1)
        Zpass = Zp[:, :c.nW]
        if ck is not None:
            self._spike_cache[1][ck] = (Zpass.copy(), R.copy())
        return Zpass, R

    def _seed_setup(self, maps, origin: str):
        """(ctrls, Z0, C): the privy controls (origin's first); Z0 (n_prim N, len(ctrls)), the closed loop with the
        privy players' maps off, whose columns are the spikes of the privy controls with the instant reactions each
        draws (c.composite); and per privy control the stacked convolution (n_prim N x N) with its spike's column,
        plus the identity on its own block and on the blocks of the controls reacting to it at once."""
        c = self.c; N = c.N; nP = len(c.prim)
        c.use_maps(maps)
        comp = c.seed_composite(origin) or {}
        owner = {a.name: a for a in self.model.agents}
        P = self.model.privy(origin)
        ctrls = [u for n in P for u in owner[n].controls]
        need = spike_controls(ctrls, comp)
        spike = compose_spikes(c.closed_loop(maps, excluded=tuple(P), impulse_controls=need)[:, c.nW:], need, ctrls, comp)
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

    def _monitor_focs(self, agent: Agent, maps, R: np.ndarray, origins, seed_consts) -> list:
        """The agent's first-order-condition operators on the monitored responses R (MonitoredDeviations._monitoring);
        no seed constants (a risk-averse agent with monitoring is refused here)."""
        return self._foc_operators(agent, R)

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
        M = conv[0][0].copy() if own == 1 else np.block([[conv[u][o2] for o2 in range(own)] for u in range(own)])
        M.flat[::own * N + 1] += 1.0                                          # I + the blocks (0 + x off the diagonal)
        out = np.zeros((len(c.prim) * N, own))
        for o in range(own):
            rhs = -(Dj[o, 0] if own == 1 else np.concatenate([Dj[o, u] for u in range(own)]))
            try:
                s = np.linalg.solve(M, rhs).reshape(own, N)
            except np.linalg.LinAlgError:              # a trial point far from the equilibrium can make the discretised
                s = np.linalg.lstsq(M, rhs, rcond=None)[0].reshape(own, N)   # Volterra operator singular; the
                                                        # iteration keeps its best round (_monitoring) and moves on
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
        self._require_settled(res)
        super()._finish(res)
        for a in self.model.agents:
            if a.risk_aversion and self._risk_scale:
                # the entropic cost of the date-0 continuation C_0 = int_0^inf e^{-rho t} c_t dt (+ the integrals), unconditional:
                # E C_0 = the flow cost / rho, plus the conditional excess of the date-0 self (stationary_risk.StationaryTilt.cond_excess)
                from .stationary_risk import StationaryTilt
                th = float(a.risk_aversion) * self._risk_scale
                tl = StationaryTilt(self, a, res.maps, th)
                # the lattice's excess converges like h (the kernels' kinks on the diagonal) plus h^2: the polynomial through the
                # levels in h, read at h = 0.  The steps are L / ENTROPIC_LATTICE up to a window of 8 and capped at
                # ENTROPIC_STEP beyond: steps that grew with the window grew the lattice's error with it (the gap between the two
                # finest levels 6e-4 at L = 4, 2.5e-3 at 8, 9.3e-3 at 16 on a one-agent signal model, whose extrapolated excess
                # moved by 1% between L = 8 and 16 on the coarse levels and by 0.1% on the capped ones), while the kernels whose
                # kinks it resolves live on the model's time scales, not the window's
                hs = self.entropic_steps(tl.L)
                vals = [tl.cond_excess(h) for h in hs]
                ex = float(np.linalg.solve(np.array([[x ** k for k in range(len(hs))] for x in hs]), np.array([v[0] for v in vals]))[0]) \
                    if all(np.isfinite(v[0]) for v in vals) else float("inf")
                EC0 = float(res.costs[a.name]) / self.c.rho
                res.risk[a.name] = {"risk_aversion": th, "expected": EC0, "entropic": EC0 + ex, "theta_mu_max": vals[-1][1],
                                    "entropic_lattice_gap": abs(vals[-1][0] - vals[-2][0]), "flow_expected": float(res.costs[a.name]),
                                    **self._risk_info.get(a.name, {})}
                if not np.isfinite(ex):
                    # the date-0 self's conditional entropic cost is infinite at these strategies (theta mu_max >= 1): the first-order
                    # condition holds but is no optimum, so this is not an equilibrium of the entropic game.  A converged solve raises
                    # RiskBreakdown, as the finite engine does at a solution past its breakdown; a solve that did not converge ends
                    # at its best iterate, which says nothing about the equilibrium, and is reported not converged
                    mu = max(v[1] for v in vals)
                    if res.converged:
                        from .risk import RiskBreakdown
                        raise RiskBreakdown(a.name, th, mu / th)
                    res.converged = False
                    res.message += (f"; {a.name}'s conditional entropic cost is infinite at the last iterate (theta mu_max = "
                                    f"{mu:.3g} >= 1): no equilibrium of the entropic game there")

    RISK_STEPS = (0.0, 0.5, 1.0)        # the continuation in risk aversion from no start: theta scaled by these in turn ...
    RISK_STEP = 0.9                     # ... while theta mu_max at the full theta, estimated at the last step's equilibrium, is
    RISK_GAIN = 0.6                     # at most this; else each step closes this fraction of the gap to the breakdown,
    RISK_EDGE = 1e-3                    # 1 - theta mu_max (the finite engine's rule), and a step within this of it short of the
    RISK_MAX_STEPS = 40                 # model's theta, or past this many steps, ends the path there: RiskBreakdown(reached)
    RISK_HALVINGS = 3                   # steps whose fixed point is past the breakdown are halved at most this many times in all
    RISK_JUMP = 0.5                     # from the risk-neutral equilibrium straight to the model's theta when theta mu_max there, estimated at it, is at most this
    RISK_STEP_EVALUATIONS = 100         # evaluations of a step between those of RISK_STEPS, unless solve() was bounded
    ENTROPIC_LATTICE = (40, 80, 160)    # lattice steps (L / these) of the date-0 continuation's entropic cost (res.risk) ...
    ENTROPIC_STEP = 0.2                 # ... the coarsest capped at this (and the others in proportion): windows above 8
    RISK_LATTICE = 0.02                 # the correction's lattice step (and half of it, Richardson): stationary_risk.py

    @classmethod
    def entropic_steps(cls, L: float):
        """The lattice steps of the date-0 entropic cost on a window L: L / ENTROPIC_LATTICE while the coarsest is at most
        ENTROPIC_STEP, else the same levels refined by the integer factor that brings it there (lattices through L)."""
        scale = max(1, int(np.ceil(L / (cls.ENTROPIC_STEP * cls.ENTROPIC_LATTICE[0]) - 1e-9)))
        return [L / (scale * na) for na in cls.ENTROPIC_LATTICE]

    def solve(self, start_from=None, **kw):
        """EngineBase.solve; with risk-averse agents and no start, by continuation in risk aversion: the risk-neutral
        equilibrium first (theta scaled by 0, the risk-neutral path), then theta scaled up (RISK_STEPS), each step started
        from the last.  The correction is frozen at the current profile inside a best response (stationary_risk.py), so a
        start far from the equilibrium is where it is least accurate.  Past the 0.5 step the path follows the finite engine's
        rule (RISK_STEP, RISK_GAIN, RISK_EDGE): straight to the model's theta while the breakdown measure theta mu_max there
        is at most 0.9, else in steps closing 0.6 of the gap; a step whose fixed point lies past the breakdown is halved
        (RISK_HALVINGS); a path that reaches the breakdown short of the model's theta raises RiskBreakdown with `reached`,
        and a converged solution past it raises RiskBreakdown (_finish), as on the finite engine."""
        if start_from is not None or not any(a.risk_aversion for a in self.model.agents):
            self._risk_scale = 1.0
            return super().solve(start_from, **kw)
        from .risk import RiskBreakdown
        diag = kw.pop("diagnostics", True)
        averse = [a for a in self.model.agents if a.risk_aversion]
        maps, res, done = None, None, []
        steps = list(self.RISK_STEPS)
        good, failed, cap = 0.0, 0, 1.0               # the last scale solved, the steps that failed, the longest step allowed
        try:
            while steps:
                sc = steps.pop(0)
                self._risk_scale = sc
                last = sc >= 1.0
                step_kw = kw
                if not last and sc not in self.RISK_STEPS and kw.get("max_evaluations") is None:
                    step_kw = {**kw, "max_evaluations": self.RISK_STEP_EVALUATIONS}   # a step of the path only: a stall ends it early
                try:
                    res = super().solve(maps, diagnostics=diag if last else False, **step_kw)
                    if not last and not res.converged and any(not np.isfinite(res.risk.get(a.name, {}).get("entropic", 0.0)) for a in averse):
                        # a step that did not converge and ended past the breakdown: as a converged one there (below)
                        worst = max(averse, key=lambda a: res.risk[a.name]["theta_mu_max"])
                        raise RiskBreakdown(worst.name, res.risk[worst.name]["risk_aversion"],
                                            res.risk[worst.name]["theta_mu_max"] / res.risk[worst.name]["risk_aversion"])
                except ValueError as exc:
                    # a converged fixed point past the breakdown (_finish), or an iterate whose correction's solve fails there (the
                    # tilt's operator singular): a step too long for the path, which lands on a spurious branch beyond it; the
                    # step is halved from the last scale solved, and past RISK_HALVINGS the path ends there
                    if not sc or not (isinstance(exc, RiskBreakdown) or "risk correction" in str(exc)):
                        raise
                    if not isinstance(exc, RiskBreakdown):
                        a0 = averse[0]
                        exc = RiskBreakdown(a0.name, a0.risk_aversion * sc, float("inf"))
                    failed += 1
                    if failed > self.RISK_HALVINGS or (sc - good) < self.RISK_EDGE:
                        if not good:
                            raise
                        raise RiskBreakdown(exc.agent, next(a.risk_aversion for a in averse if a.name == exc.agent), exc.lam_max,
                                            reached=good * next(a.risk_aversion for a in averse if a.name == exc.agent)) from None
                    cap = 0.5 * (sc - good)                     # this step halved, and no later step longer
                    steps = [good + cap]
                    continue
                done.append(sc)
                if not res.converged or last:
                    break
                maps, good = res.maps, sc
                if not sc and steps and self.RISK_JUMP > 0:
                    # the breakdown measure at the model's theta estimated at the risk-neutral equilibrium (theta mu_max scales
                    # with theta; one lattice, the coarsest): well inside the breakdown, straight to the model's theta, the
                    # intermediate steps being warm starts only
                    from .stationary_risk import StationaryTilt
                    try:
                        est = max(StationaryTilt(self, a, res.maps, float(a.risk_aversion)).cond_excess(self.entropic_steps(self.c.grid.L)[0])[1]
                                  for a in averse)
                    except (ValueError, np.linalg.LinAlgError):
                        est = float("inf")
                    if np.isfinite(est) and est <= self.RISK_JUMP:
                        steps = [1.0]
                if sc:
                    # the next step: the model's theta while the breakdown measure there, estimated at this step's equilibrium
                    # (theta mu_max scales with theta), is at most RISK_STEP; else a step closing RISK_GAIN of the gap to the
                    # breakdown.  A step straight to a theta near it can land on a spurious branch past it: the one-agent signal
                    # model at theta 1 went from 0.5 to 1 onto a fixed point with theta mu_max 8.95 (reported not converged),
                    # where the steps 0.783, 0.942, 1 reach its equilibrium, theta mu_max 0.886
                    worst = max(averse, key=lambda a: res.risk[a.name]["theta_mu_max"])
                    x = res.risk[worst.name]["theta_mu_max"] / sc                  # theta mu_max at the full theta
                    reached = sc * x
                    if x <= self.RISK_STEP:
                        steps = [1.0]
                    elif 1.0 - reached < self.RISK_EDGE or len(done) > self.RISK_MAX_STEPS:
                        raise RiskBreakdown(worst.name, worst.risk_aversion, x / worst.risk_aversion, reached=sc * worst.risk_aversion)
                    else:
                        steps = [min(1.0, (reached + self.RISK_GAIN * (1.0 - reached)) / x)]
                    steps = [min(steps[0], sc + cap)]
        finally:
            self._risk_scale = 1.0
        if done[-1] < 1.0 or not res.converged:
            res.converged = False
            res.message += f"; the continuation in risk aversion stopped at theta scaled by {done[-1]:g}"
        elif len(done) > len(self.RISK_STEPS):
            res.message = f"continuation in risk aversion, theta scaled by {', '.join(f'{s:.3g}' for s in done)}: " + res.message
        return res

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
        rhs = Bk.reshape(nW * N, nR * N).T @ (actions * W[None, :, None]).transpose(2, 1, 0).reshape(nW * N, nU)   # column ui: sum_k Bk' W actions[ui, :, k]
        if not keep.all():
            rhs = rhs[keep]
        reuse = nR * N >= self.PROJ_REUSE_MIN and not getattr(self, "_reuse_off", False)
        memo = self.__dict__.setdefault("_proj_chol", {})
        hit = memo.get(agent.name) if reuse else None
        if hit is not None and not hit[2] and np.array_equal(hit[0], keep):
            # the Gram changes with the iterate's step: Krylov on its matvec (two products with the row operator, the
            # ridge from its diagonal, the Gram's trace), preconditioned by the Cholesky factor of the last one built
            B2 = Bk.reshape(nW * N, nR * N)
            if not keep.all():
                B2 = B2[:, keep]
            Wr = np.tile(W, nW)
            ridge = self.settings.stationary_map_ridge * float(np.einsum("i,ij,ij->", Wr, B2, B2)) / B2.shape[1]
            A = lambda v: B2.T @ (Wr * (B2 @ v)) + ridge * v
            cf = hit[1]
            g = np.zeros((nU, nR * N)); ok = True; worst = 0
            for ui in range(nU):
                x, its = _pgmres(A, lambda v: sla.cho_solve(cf, v, check_finite=False), rhs[:, ui], self.FOC_KRYLOV_TOL,
                                 self.FOC_KRYLOV_MAXIT, hit[3][ui])
                if x is None:
                    ok = False
                    break
                g[ui, keep] = x; worst = max(worst, its)
            if ok:
                memo[agent.name] = (hit[0], cf, worst > self.FOC_REFRESH_ITS, g[:, keep])
                return g.reshape(nU, nR, N)
        memo.pop(agent.name, None)                                  # the old factor is not held through the new Gram
        hit = cf = None
        sw = np.sqrt(W)
        B4 = Bk.reshape(nW, N, nR, N)
        Gram = np.zeros((nR * N, nR * N)); G4 = Gram.reshape(nR, N, nR, N)
        chunks = self._causal_chunks()
        if any(np.any(B4[:, lo:hi, :, hi:]) for lo, hi in chunks):           # a lead among the instantaneous reads: no causal chunks
            chunks = [(0, N)]
        for lo, hi in chunks:
            X = (B4[:, lo:hi, :, :hi] * sw[None, lo:hi, None, None]).reshape(nW * (hi - lo), nR * hi)
            G4[:, :hi, :, :hi] += dsyrk(1.0, X, trans=1).reshape(nR, hi, nR, hi)     # upper triangle (local order = global order)
        # the lower triangle from the upper (dsyrk leaves it zero), a block of rows at a time: the entries of
        # triu(G) + triu(G)' - diag(G), without its three full temporaries (a diagonal block by that expression)
        n = Gram.shape[0]
        for i0 in range(0, n, 256):
            i1 = min(n, i0 + 256)
            Gram[i0:i1, :i0] = Gram[:i0, i0:i1].T
            # the diagonal block: its strict lower triangle (zero) from the upper, which is triu(D) + triu(D)' - diag(D)
            # to the bit (x + 0 - 0 above, 0 + x below, 2d - d on the diagonal; the Gram, summed from zeros, holds no -0)
            np.copyto(Gram[i0:i1, i0:i1], Gram[i0:i1, i0:i1].T, where=_strict_lower(i1 - i0))
        if not keep.all():
            Gram = Gram[np.ix_(keep, keep)]
        Gram.flat[::Gram.shape[0] + 1] += self.settings.stationary_map_ridge * np.trace(Gram) / Gram.shape[0]   # the ridge
        g = np.zeros((nU, nR * N))
        if reuse:
            try:
                cf = sla.cho_factor(Gram.T, overwrite_a=True, check_finite=False)       # symmetric: F-ordered in place
            except np.linalg.LinAlgError:
                cf = None
            if cf is not None:
                g[:, keep] = sla.cho_solve(cf, rhs, check_finite=False).T
                memo[agent.name] = (keep, cf, False, g[:, keep])            # the factor, and the maps: the next start
                return g.reshape(nU, nR, N)
            memo.pop(agent.name, None)
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
        Zpass, R0 = S2._spikes(c2, maps2, agent)
        R = S2._impulse_responses(agent, maps2, R0)
        ytil, yinst = S2._passive_rows(agent, Zpass)
        Gk2 = S2._row_operator(agent, ytil, yinst)
        Resp2 = S2._half_discounted(agent, R0, R) if c2.rho > 0 else S2._response_operators(agent, R)
        GAO2 = S2._loss_form(agent, half_discount=c2.rho > 0)
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
        memo = self.__dict__.setdefault("_delayed_masks", {})     # map-independent: once per agent
        hit = memo.get(agent.name)
        if hit is None:
            base = np.ones(len(agent.signals) * N, dtype=bool); delayed = set()
            last = (np.arange(N) % g.n) == g.n - 1
            for r in range(len(agent.signals)):
                d = c.rows[agent.name][r][3]
                if d > 0:      # ages below L - d, and the lower copy of the node at L - d (the action at age L reads it)
                    edge = g.L - d
                    base[r * N:(r + 1) * N] = (g.nodes < edge - 1e-12) | (last & (np.abs(g.nodes - edge) <= 1e-12))
                    delayed.add(r)
            hit = memo[agent.name] = (base, delayed)
        base, delayed = hit
        keep = base.copy()
        for r in getattr(self, "_silent", {}).get(agent.name, ()):   # a level row of a quantity that does not move
            if r not in delayed:                                     # (the zero start): nothing to read, its map is zero
                keep[r * N:(r + 1) * N] = False                      # (a delayed row's mask is its delay's)
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

    PROJ_REUSE_MIN = 400         # map nodes nR N from which the projection's Gram is solved the same way (_project)
    FOC_REUSE_MIN = 600          # unknowns from which a best response's FOC system is solved by Krylov on a frozen factorisation
    FOC_KRYLOV_TOL = 1e-14       # ... to this residual relative to the right-hand side (the direct solve's is about 5e-16)
    FOC_KRYLOV_MAXIT = 30        # ... in at most this many iterations, else the system is built and factored afresh
    FOC_REFRESH_ITS = 5          # ... and a solve that needed more than this many refactors at the next best response

    def _foc_gamma(self, agent: Agent, rows, inst, Zpass, Resp, Fu, shift, Gk, Gpre) -> np.ndarray:
        """gamma (nU nR N,) solving the best response's first-order conditions Amat gamma = -bvec (_foc_system).

        Small systems: Amat is built and solved directly (_solve_foc).  From FOC_REUSE_MIN unknowns the fixed point's
        consecutive systems differ by the iterate's step, so the LU of the last one built is kept per agent and the new
        system is solved by GMRES preconditioned with it, on Amat's matvec from its factors (no Amat: the build is
        O(N^3 nR^2 nW) and the factorisation O((nU nR N)^3), a matvec O(nW N nR N nU)).  Near the fixed point the
        preconditioned operator is the identity to the outer residual and GMRES takes one to three iterations; far from it,
        or when it does not reach FOC_KRYLOV_TOL within FOC_KRYLOV_MAXIT, the system is built and factored afresh (its
        LU the next preconditioner).  The solution is the direct solve's to its rounding (the system's condition times
        the tolerance), so the fixed point is the same to its tolerance.  A singular system is detected by the
        factorisation, as before."""
        c = self.c; N = c.N; nR, nU = len(rows), len(Fu)
        nG = nU * nR * N
        keep = np.tile(self._identified(agent), nU)
        direct = (nG < self.FOC_REUSE_MIN or type(self)._row_operator is not StationarySolver._row_operator
                  or getattr(self, "_reuse_off", False))
        if direct:
            Amat, bvec = self._foc_system(agent, rows, inst, Zpass, Resp, Fu, shift, None if Gpre is None else Gk, Gpre)
            return self._solve_foc(agent, Amat, bvec)
        assemble, bvec, matvec = self._foc_system(agent, rows, inst, Zpass, Resp, Fu, shift, Gk, Gpre, lazy=True, Gfull=Gk)
        memo = self.__dict__.setdefault("_foc_lu", {})
        hit = memo.get(agent.name)
        gamma = np.zeros(nG)
        rhs = -bvec[keep]
        if hit is not None and not hit[2] and np.array_equal(hit[0], keep):
            lu = hit[1]
            if keep.all():
                A = matvec
            else:
                def A(x):
                    full = np.zeros(nG); full[keep] = x
                    return matvec(full)[keep]
            x, its = _pgmres(A, lambda v: _lu_solve(lu, v, trans=1), rhs, self.FOC_KRYLOV_TOL, self.FOC_KRYLOV_MAXIT, hit[3])
            if x is not None:
                memo[agent.name] = (hit[0], lu, its > self.FOC_REFRESH_ITS, x)  # stale past the threshold: the next refactors
                gamma[keep] = x
                return gamma
        memo.pop(agent.name, None)                                  # the old factors are not held through the new ones
        hit = lu = None
        Amat = assemble()
        Ak = Amat if keep.all() else Amat[np.ix_(keep, keep)]
        del Amat
        # factored as its transpose, which is the C-ordered array read F-ordered (in place, no copy): A x = b is the
        # transposed solve with those factors
        getrf, = sla.get_lapack_funcs(("getrf",), (Ak,))
        lu, piv, info = getrf(Ak.T, overwrite_a=True)
        del Ak
        if info > 0 or not np.isfinite(lu).all():
            raise ValueError(singular_system_message(agent.name))
        gamma[keep] = x = _lu_solve((lu, piv), rhs, trans=1)
        memo[agent.name] = (keep, (lu, piv), False, x)                     # the factors, and the solution: the next start
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
        every agent's best response at the equilibrium.  A cyclically symmetric tie at symmetric maps (the closed loop's
        own condition for its mode solve; risk-neutral, unmonitored): the followers' are the representative's relabelled
        along the cycle -- a follower t steps on is the representative's problem with every name moved t steps, so its
        FOC kernels are the representative's on the moved channels and its curvature and representation error are the
        representative's -- instead of a best response each."""
        self.__dict__.pop("_foc_lu", None); self.__dict__.pop("_proj_chol", None)
        self._reuse_off = True                  # direct solves, no factors kept, for these one-off best responses
        try:
            self._diagnostics_inner(res)
        finally:
            self._reuse_off = False

    def _diagnostics_inner(self, res) -> None:
        c = self.c; sym = c.sym
        if (sym is None or c.instant_loads or c.levels or not c._maps_symmetric(res.maps)
                or any(a.risk_aversion or a.monitors for a in self.model.agents)
                or any(c.rep[f] != sym.agents[0] for f in sym.agents)):
            self._fill_diagnostics(res)
            return
        rep = sym.agents[0]
        self._fill_diagnostics(res, agents=[a.name for a in self.model.agents if a.name not in sym.agents[1:]])
        ch = {w: i for i, w in enumerate(c.channels)}
        names = sym.names
        for t, f in enumerate(sym.agents[1:], start=1):
            def step(x, t=t):                                       # a name moved t steps along the cycle
                for _ in range(t):
                    x = names.get(x, x)
                return x
            perm = np.array([ch[step(w)] for w in c.channels], dtype=np.intp)   # follower channel perm[k] carries rep's k
            dec = {}
            for u, d in res.foc[rep].items():
                dd = {}
                for key, K in d.items():
                    out = np.empty_like(K); out[:, perm] = K
                    dd[key] = out
                dec[step(u)] = dd
            res.foc[f] = dec
            if rep in res.second_order:
                res.second_order[f] = res.second_order[rep]
            res.representation_error[f] = res.representation_error[rep]
            self._diagnostics_extra(res, next(a for a in self.model.agents if a.name == f))

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
        tau, and for every other atom the discounted DC gain int_0^inf e^{-rho a} R_j(a) da of the atom's
        passive-world impulse response (the other agents reacting through their kernels, the agent's own
        control passive; a lagged or lead atom weighted by e^{-rho tau}): exact over all ages (_passive_dc), or
        where that does not apply its integral over the window, int_0^L."""
        c = self.c; nP = len(c.prim); rho = c.rho; a = agent
        dc = c.grid.discounted_mass(rho)
        atoms, Q, q = c.loss[a.name]
        Mu = np.zeros((len(a.controls), nP)); bu = np.zeros(len(a.controls))
        P = np.zeros((len(atoms), nP))                        # atom means from the primaries' means
        for j, (nm, lag) in enumerate(atoms):
            P[j, c.index[nm]] = 1.0
        R = None
        H = None
        if not a.myopic:
            R = self._impulse_responses(a, maps, self._spikes(c, maps, a)[1])
            H = self._passive_dc(a, maps, R, atoms)
        for ui, u in enumerate(a.controls):
            m = np.zeros(len(atoms))
            for v, coef in (c.composite or {}).get(u, {u: 1.0}).items():      # the control and the instant reactions it draws
                if (v, 0.0) in atoms:
                    m[atoms.index((v, 0.0))] += coef
            if a.myopic:
                pass
            else:
                for j, (nm, lag) in enumerate(atoms):
                    if nm in a.controls:
                        if nm == u and lag > 0:               # delayed read of the control itself
                            m[j] += np.exp(-rho * lag)
                        continue                              # own reactions: envelope
                    if H is not None:                         # the whole response (_passive_dc)
                        m[j] += np.exp(-rho * lag) * H[c.index[nm], ui]
                    elif lag >= 0:
                        m[j] += dc @ (c.shift(lag) @ R[c.block(nm), ui])
                    else:
                        m[j] += np.exp(-rho * lag) * (dc @ R[c.block(nm), ui])
            Mu[ui] = m @ Q @ P; bu[ui] = -(m @ q)
        return Mu, bu

    PASSIVE_DC_RCOND = 1e-12

    def _passive_dc(self, agent: Agent, maps: Dict[str, np.ndarray], R: np.ndarray, atoms) -> Optional[np.ndarray]:
        """The discounted DC gains of the agent's passive world over ALL ages, H (n_prim, nU) with
        H[p, u] = int_0^inf e^{-rho a} R_p(a) da for a unit impulse of the agent's control u, the others reacting
        through `maps`: the passive world's Laplace transform at s = rho, one small linear system on the primaries,
        (s - A) x - sum inputs e^{-s lag} = 0 on the states, v = sum_r g_vr(s) sum c e^{-s (delay + lag)} on every
        other control (g_vr(s) = int_0^L e^{-s a} g_vr(a) da: a map lives on the window, so this is exact), the
        agent's own controls the impulse.

        The mean first-order condition needs the whole response: a constant is felt at every age, and the passive
        world (one reaction fewer than the equilibrium) can decay much more slowly than the equilibrium's kernels,
        whose tail the window check measures.  Its integral over [0, L] made the means window-dependent where the
        kernels were not (a two-player tracking game with conflicting targets at window 3: player 1's passive
        response still 42% of its peak at L, the mean control 7.24 against 8.40 at L = 12, the costs 16% off).

        None where the integral over the window of `R` (the passive responses on the grid) stays, with the discounted
        response's level at L relative to its peak kept in self._mean_tails[agent] for the result's `window` check:
        level rows, instant observations and monitored deviations (their responses are not the maps' plain closed
        loop), a singular transform (a random walk driven by the agent that nothing in the passive world pulls back,
        at rho = 0: the response does not decay and its integral grows with the window -- two players pushing such a
        walk toward different targets with no information have no stationary means) and a passive world whose
        discounted response is largest at the window's edge (not decaying: the transform would be an analytic
        continuation, not the integral).  A random walk with no inputs at all is pinned (nothing moves it)."""
        c = self.c
        s = c.rho; nX = c.nX; nP = len(c.prim)
        ea = np.exp(-s * c.grid.nodes)
        tail = 0.0
        for nm, lag in atoms:
            if nm in agent.controls:
                continue
            f = np.abs(ea[:, None] * R[c.block(nm)])
            peak = f.max()
            if peak > 0:
                tail = max(tail, float(f[-1].max() / peak))
        if c.levels or c.instant_loads or len(self.model.privy(agent.name)) > 1:
            self._mean_tails[agent.name] = tail
            return None
        dm = c.grid.discounted_mass(s)
        M = np.zeros((nP, nP)); F = np.zeros((nP, len(agent.controls)))
        M[:nX, :nX] = s * np.eye(nX) - c.A
        for i, (nm, lag), coef in c.state_inputs:
            M[i, c.index[nm]] -= coef * np.exp(-s * lag)
        for i in range(nX):
            if not M[i].any():
                M[i, i] = 1.0                                # a random walk with no inputs: nothing moves it
        for ui, u in enumerate(agent.controls):
            M[c.index[u], c.index[u]] = 1.0; F[c.index[u], ui] = 1.0
        for b in self.model.agents:
            if b.name == agent.name:
                continue
            g = maps[b.name]
            for vi, v in enumerate(b.controls):
                k = c.index[v]; M[k, k] += 1.0
                for r, (_, drift, _, delay) in enumerate(c.rows[b.name]):
                    gh = float(dm @ g[vi, r])
                    for (n, l), coef in drift.items():
                        M[k, c.index[n]] -= gh * coef * np.exp(-s * (delay + l))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sla.LinAlgWarning)
            lu, piv = sla.lu_factor(M, check_finite=False)
        gecon, = sla.get_lapack_funcs(("gecon",), (lu,))
        if not float(gecon(lu, np.linalg.norm(M, 1))[0]) > self.PASSIVE_DC_RCOND or tail >= 1.0 - 1e-9:
            self._mean_tails[agent.name] = tail
            return None
        return sla.lu_solve((lu, piv), F, check_finite=False)

    def passive_tails(self, agent: Agent, maps: Dict[str, np.ndarray], ages: np.ndarray) -> Optional[np.ndarray]:
        """(nU, n_prim, len(ages)): int_A^inf e^{-rho a} R_p(a) da for every age A in `ages` (A >= 0), R the agent's
        passive response to a unit impulse of each of its controls under `maps` -- the part of a mean first-order
        condition's continuation beyond a transition's strip, which ends at T + L (spectral_means).  The whole-age
        transform (_passive_dc) less the integral up to A of the response on a grid extended to max(ages) (panels as
        wide as the last one, the maps zero beyond L: the response there is exact).  None where _passive_dc is."""
        c = self.c
        R0 = self._impulse_responses(agent, maps, self._spikes(c, maps, agent)[1])
        H = self._passive_dc(agent, maps, R0, c.loss[agent.name][0])
        if H is None:
            return None
        g = c.grid; top = float(np.max(ages))
        bp = [float(b) for b in g.breakpoints]
        width = bp[-1] - bp[-2]
        while bp[-1] < top - 1e-12:
            bp.append(bp[-1] + width)
        if len(bp) > len(g.breakpoints):
            S2 = type(self)(self.model._patch_horizon(window=bp[-1]).with_numerics(breakpoints=bp), **self.solver_kw)
            c2 = S2.c; g2 = c2.grid
            sides = g2.node_sides(); I = np.zeros((g2.N, g.N))
            for sd in (+1, -1):
                sel = (sides == sd) | ((sides == 0) & (sd == +1))
                I[sel] = g.interp(g2.nodes[sel], side=sd)                   # zero beyond the old window
            maps2 = {a: np.einsum("fn,urn->urf", I, m) for a, m in maps.items()}
            R = S2._impulse_responses(agent, maps2, S2._spikes(c2, maps2, agent)[1])
        else:
            c2, g2, R = c, g, R0
        xg, wg = np.polynomial.legendre.leggauss(g2.n + 2)
        out = np.zeros((len(agent.controls), len(c.prim), len(ages)))
        for k, A in enumerate(ages):
            pts, wts = [], []
            for lo, hi in zip(g2.breakpoints[:-1], g2.breakpoints[1:]):
                hi = min(hi, A)
                if hi <= lo:
                    break
                pts.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo)); wts.append(0.5 * (hi - lo) * wg)
            if not pts:
                inner = np.zeros((len(c.prim), len(agent.controls)))
            else:
                x = np.concatenate(pts); w = np.concatenate(wts) * np.exp(-c.rho * x)
                Iw = w @ g2.interp(x)
                inner = np.stack([Iw @ R[c2.block(nm)] for nm in c.prim])     # (n_prim, nU)
            out[:, :, k] = (H - inner).T
        return out

    def _assemble_means(self, maps, **variant):
        self._mean_tails = {}
        return super()._assemble_means(maps, **variant)
