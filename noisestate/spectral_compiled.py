"""The spectral finite engine's compiled model on the piecewise-spectral triangle (triangle.py): SpectralCompiled.

Every kernel K(t, s) lives on the triangle grid in (t, age) coordinates cut by
the delays.  Strategies are raw maps g[u][r](t, b): the control at t is the
sum over signal rows of int_0^t g(t, b) dY_r^seen(t - b), with b the age of the
observation increment.  This module holds what faces the grid: the breakpoint
sequence and its closure under the lags (or the unit panels within
horizon.unit_range), the grid, the past's wiring (the initial shocks' columns,
the old regime's loadings), the buffer's wiring (the frozen stationary maps), the
reads and node-to-node shifts (dense and as CSR matrices), the line paths (the
Volterra propagation, the convolutions, the past's and the old shocks' segments),
the discrete weights of the initial shocks and the masses.  Its neighbours: the
closed loop is assembled in closed_loop.py (ClosedLoopRows, called by
closed_loop()), the time-line operators of the means are the TimeLineOps mixin of
spectral_means.py, the best-response operators on the paths and sparse reads are
spectral_operators.py, the solver is finite_spectral.py.

With a known past (SpectralFiniteSolver(past=...), see past.py) the triangle becomes
the strip [0, T] x [0, L] of triangle.py: the nodes above the diagonal carry the
kernels on the shocks born before zero, whose state at time zero is the past's
kernel at age -s and whose pre-zero inputs and observations are read from the
past (a lagged atom before zero is the past's kernel of that quantity; a line
integral crossing time zero gets a known segment on the past's own age grid); the
map gains the same region, the weights on the increments observed before zero.
Initial shocks (point loadings at time 0-) are extra columns of the world, one
per shock, meaningful on the line s = 0, observed through discrete weights on the
time nodes stored after each row's map nodes (maps (nU, nR, N + Nt)).  Without a
past nothing of this runs and every array is the triangle's.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from functools import cached_property
import warnings

import numpy as np

from .closed_loop import ClosedLoopRows, ClosedLoopSources
from .compile import CompiledBase, close_under_delays, reject_leads
from .grid import bary_weights, cheb_lobatto
from .grid_cache import triangle_grid
from .spec import Atom, Model
from .spectral_means import TimeLineOps
from .triangle import TriangleGrid


def _common_unit_hint(lags, kmax: int = 100) -> str:
    """' (u works)' for the largest u = min(lags) / k, k <= kmax, of which every lag is a multiple; the lags
    are incommensurable at that resolution otherwise, which no panel grid carries."""
    lo = min(lags)
    for k in range(1, kmax + 1):
        u = lo / k
        if all(abs(l / u - round(l / u)) < 1e-9 for l in lags):
            return f" ({u:g} works)"
    return f" (none above {lo / kmax:g}: the finite engines need commensurable lags)"

class SpectralCompiled(TimeLineOps, ClosedLoopSources, CompiledBase):
    def __init__(self, model: Model, past=None, continuation=None):
        """past: a Past (the game starts at time zero from that regime); continuation: a converged
        StationaryResult of this model at the past's window, on which every agent's map is frozen on the
        buffer [T, T + L] after the horizon (the unknowns stay on [0, T]; the closed loop and the
        first-order conditions run to T + L, by when every shock born before T is forgotten), or None:
        the game ends at T.  Built in named steps, in this order: the regimes (the past and the
        continuation), the breakpoint sequence and its closure under the lags, the grid, the buffer's
        wiring, the past's wiring, the paths and the time nodes, the caches; each step's docstring names
        its invariant."""
        super().__init__(model)
        reject_leads(model, 'spectral finite engine')
        hz = model.horizon
        lags = model.all_lags()
        L = self._regimes(model, past, continuation, lags)
        bp, required, unit = self._breakpoints(hz, lags, L)
        self._grid(hz, lags, L, bp, required, unit)
        self._wire_buffer()
        self._wire_past(model, L)
        self._wire_time(L)
        self._caches()

    def _regimes(self, model: Model, past, continuation, lags) -> Optional[float]:
        """The past and the continuation as the game's regimes: T, self.past (validated against the model at the
        panel unit), row_delays and, with a past, the rows undelayed (their maps in raw age), self.cont (a
        converged StationaryResult of this model at the past's window L; a horizon shorter than L keeps old
        shocks alive on the buffer, whose pieces then carry the band above the line s = 0) and Tg, the
        grid's end.  Invariant: Tg = T + L with a continuation, T without; a continuation always has a past
        with a window.  Returns L, the past's window (None without one)."""
        hz = model.horizon
        self.T = float(hz.extent)
        self.past = past
        if past is not None:
            past.validate(model, (hz.unit or min(lags)) if lags else None)
        # with a past a row observed with delay d keeps its map in raw age (the weight the control at t puts on the
        # raw increment of age a, zero for a < d, whole pieces since d is a breakpoint) and is an undelayed row to
        # every operator, the band's included; without a past the map is stored at the shifted time (map_shift)
        self.row_delays: Dict[str, List[float]] = {a: [float(r[3]) for r in rr] for a, rr in self.rows.items()}
        if past is not None:
            self.rows = {a: [(n, drift, E, 0.0) for (n, drift, E, d) in rr] for a, rr in self.rows.items()}
        L = past.window if past is not None and past.window > 0 else None
        self.cont = continuation                      # the StationaryResult the buffer is frozen at (None: the game ends at T)
        self.continuation_info: Optional[dict] = None
        if continuation is not None:
            if L is None:
                raise ValueError("a stationary continuation needs a past with a window (the old regime's kernels): with initial "
                                 "shocks only the game ends at T")
            self._check_continuation(model, continuation, L)
        self.Tg = self.T + L if continuation is not None else self.T        # the grid's end: the buffer [T, T + L] follows T
        return L

    def _breakpoints(self, hz, lags, L: Optional[float]):
        """The breakpoint sequence of the triangle in time and age: within horizon.unit_range (self.coarse) the
        unit multiples up to it, T and, when the game ends at T, T - k unit, filled geometrically between
        (TriangleGrid.fill_geometric); else horizon.breakpoints or every multiple of the unit up to T
        (TriangleGrid.breakpoints), closed under the lags (close_under_delays, with a warning when it adds
        panels).  Invariant: every lag and delay of the model is a breakpoint, and unless coarse every panel
        shifted by a lag is again a panel, so a lagged read is a node-to-node shift (map_shift).  Returns
        (bp, required, unit): the required cuts and the unit of a coarse grid (None, None otherwise), which
        the strip's sequence is rebuilt from."""
        past, continuation = self.past, self.cont
        required = unit = None
        # unit_range below the window: the delay cuts stay at every multiple of the unit up to it, in time and
        # in age (the kink at the k-th delay line weakens with k), and the panels grow geometrically beyond;
        # the delay reads are then node-to-node within it and interpolated beyond (map_shift), and the panels
        # are not closed under the lags.  Default (None, or the window): today's grid, cut at every multiple.
        R = hz.unit_range
        self.coarse = bool(lags) and not hz.breakpoints and R is not None and R < self.T - 1e-12
        if self.coarse and max(lags) > R + 1e-12:
            raise ValueError(f"horizon.unit_range ({R:g}) is below the largest lag/delay {max(lags):g}: the unit panels must reach "
                             "every lag (a lagged read lands node to node only within unit_range)")
        if self.coarse:
            # the required cuts: the unit multiples up to unit_range and T (the strip adds L and the past's cuts within
            # unit_range).  Without a past a delayed row's map is stored at the shifted time and read node to node
            # by the action grid on every piece (an interpolated read from differently cut panels leaves map nodes
            # unidentified: a singular first-order condition), which closes the panels under the delay everywhere
            if past is None and any(d > 0 for rr in self.rows.values() for (_, _, _, d) in rr):
                raise ValueError(f"horizon.unit_range ({R:g}) below the window: a row observed with a delay keeps its map on the "
                                 "action grid shifted by the delay, so every time panel must shift onto a panel; without a "
                                 "past (where the map is in raw age) set unit_range to the window, or solve as a transition")
            unit = hz.unit or min(lags)
            required = set(np.arange(0.0, R + 1e-12, unit)) | {self.T}
            if continuation is None:       # the game ends at T: a control is idle within the last lag, the kernels kink at T - k unit
                required |= {round(self.T - b, 12) for b in np.arange(unit, R + 1e-12, unit) if self.T - b > 1e-12}
            bp = TriangleGrid.fill_geometric(required, unit)
        else:
            bp = list(hz.breakpoints) if hz.breakpoints else TriangleGrid.breakpoints(self.T, lags, hz.unit)
        missing = [l for l in lags if not any(abs(l - b) < 1e-12 for b in bp)]
        if missing:                      # every lag must be on the panels before the closure: closing under a lag off
            # the unit grid would shatter the panels down to the lags' common divisor, or never terminate
            if hz.breakpoints:
                raise ValueError(f"lag/delay {missing[0]} is not a breakpoint of horizon.breakpoints {bp}"
                                 + (f" (nor {missing[1:]})" if len(missing) > 1 else "") + "; list every lag and delay "
                                 "of the model among them, or drop horizon.breakpoints (the panels are then built from the lags)")
            unit = hz.unit or min(lags)
            raise ValueError(f"lag(s)/delay(s) {missing} are not multiples of the panel unit {unit} "
                             f"({'horizon.unit' if hz.unit else 'the smallest lag'}); set horizon.unit to a common divisor "
                             f"of the lags {lags}{_common_unit_hint(lags)}")
        # pieces closed under every lag (row delays, drift and loss lags): a lagged read and a delayed row's map are
        # then node-to-node shifts (map_shift) and the lag lines run along piece edges.  A window that is not a
        # multiple of a lag gets the breakpoints T - k lag as well (the kernels kink there: a control acting after
        # the lag is idle within the last lag), about twice the panels and four times the pieces.
        closed = close_under_delays(bp, lags) if lags and not self.coarse else bp
        if len(closed) > len(bp):
            added = [round(float(b), 6) for b in closed if not any(abs(b - x) < 1e-9 for x in bp)]
            P0, P1 = len(bp) - 1, len(closed) - 1
            warnings.warn(f"the time panels are closed under the lag(s) {lags}: breakpoints {added} added ({P1} panels, "
                          f"{P1 * (P1 + 1) // 2} pieces, instead of {P0} panels, {P0 * (P0 + 1) // 2} pieces); a window that "
                          "is a multiple of every lag, with breakpoints closed under them, avoids the extra panels")
        bp = closed
        return bp, required, unit

    def _grid(self, hz, lags, L: Optional[float], bp, required, unit) -> None:
        """The triangle grid self.g (shared through grid_cache.triangle_grid), N and rho: without a past the
        triangle on bp; with one the strip [0, T] x [0, L] on one sequence for time and age, bp with the past
        grid's panels below L and L itself (a coarse grid: the geometric fill between the required points and
        them), panels of width L when the model has no lags, with a continuation the buffer's time panels (the
        age panels shifted to T), closed under the lags again.  Invariant: the past's panels lie on piece edges
        (the old kernels kink there) and the buffer's pieces are the strip's on [0, L] with the origin at T."""
        past, continuation = self.past, self.cont
        R = hz.unit_range
        if L is None:
            self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes)   # shared
        else:
            # the strip: one breakpoint sequence for time and age, the past grid's panels and L among them
            # (the old kernels kink on the past's panels, which then lie on piece edges), closed under the lags
            old = [b for b in past.breakpoints if b < L - 1e-12 and (not self.coarse or b <= R + 1e-12)]
            if self.coarse:                                         # the stretches between the required points grow geometrically
                bp = TriangleGrid.fill_geometric(required | set(old) | {float(L)}, unit)
            else:
                bp = sorted(set(bp) | set(old) | {float(L)})
            if not lags and not hz.breakpoints:                     # no lags: time panels of width L (the shocks' lifetime), then T
                bp = sorted(set(bp) | {float(b) for b in np.arange(0.0, self.T - 1e-12, L)})
            if continuation is not None:                            # the buffer's time panels: the age panels shifted to T
                if self.T < L - 1e-12:
                    # T below the window: the buffer [T, T + L] overlaps [0, L], so the cuts below L must be closed under
                    # the shift by T (a cut b below L is a cut b + T and b - T within [0, L] as well) for the buffer's
                    # panels to be the age panels shifted; with the lags closed after, repeated to a fixed point
                    for _ in range(64):
                        below = sorted(b for b in bp if b <= L + 1e-12)
                        more = {round(b + k * self.T, 12) for b in below for k in (-1, 1) if -1e-12 <= b + k * self.T <= L + 1e-12}
                        new = sorted(set(below) | more)
                        new = close_under_delays(new, lags) if lags and not self.coarse else new
                        new = [b for b in new if b <= L + 1e-12]
                        if len(new) == len(below):
                            break
                        bp = new
                    else:
                        raise ValueError(f"the cuts below the window {L:g} do not close under the shift by T = {self.T:g} and the lags "
                                         f"{lags}: set horizon.unit to a common divisor of T, L and the lags")
                bp = sorted(set(bp) | {round(self.T + b, 12) for b in bp if b <= L + 1e-12})
            bp = close_under_delays(bp, lags) if lags and not self.coarse else bp
            if continuation is not None:
                self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes, self.Tg, float(L), self.T)
            else:
                self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes, self.Tg, float(L))
        self.N = self.g.N
        self.rho = float(hz.discount)

    def _wire_buffer(self) -> None:
        """The buffer: P_T, the time panels up to T (the unknowns' panels), the mask of the nodes of the pieces
        after T and, with a continuation, the frozen stationary maps at every node's age (_frozen_maps; None
        without).  Invariant: without a continuation no node is on the buffer; with one every buffer node reads
        the continuation's map at its age, and every unknown lies on a panel below P_T."""
        g = self.g; continuation = self.cont
        # the buffer: the nodes of the pieces after T, where every map is frozen at the continuation's stationary
        # map at the node's age; P_T counts the time panels up to T (the unknowns' panels)
        eps = 1e-12 * max(1.0, self.Tg)
        self.P_T = int(np.sum(g.bp < self.T - eps))
        self.buffer = np.concatenate([np.full(pc.n, bool(pc.t0 >= self.T - eps)) for pc in g.pieces]) if continuation is not None \
            else np.zeros(self.N, dtype=bool)
        self.frozen: Optional[Dict[str, np.ndarray]] = self._frozen_maps() if continuation is not None else None
        self.unfreeze()

    def unfreeze(self) -> None:
        """The fixed part of every map is the buffer alone (the default): fixed_nodes (N,) the buffer's mask,
        fixed_maps[agent] the frozen stationary maps there, fixed_time (Nt,) no time node, P_lo = 0 (the first
        time panel with unknowns), t_lo = 0."""
        self.P_lo = 0; self.t_lo = 0.0
        self.fixed_nodes = self.buffer
        self.fixed_maps: Optional[Dict[str, np.ndarray]] = self.frozen
        self.fixed_time: Optional[np.ndarray] = None

    def freeze_before(self, t_lo: float, maps: Dict[str, np.ndarray]) -> None:
        """Fix every agent's map on the time panels before t_lo (a breakpoint at or below T) at the given maps
        (agent -> (nU, nR, N[+ Nt]): the values on those panels, and the discrete weights on the time nodes of
        those panels), in addition to the buffer's frozen stationary maps: the closed loop and the right-hand
        sides read them as known (with_frozen), the unknowns (_identified, the FOC system, the projection, the
        preconditioner) are the identified nodes of the panels from t_lo on.  The march's local step: after the
        first step the strategies before the previous horizon less one window are converged (the end effect
        leaks back at the closed-loop rate only), so a step solves for the new stretch plus the last window of
        the old horizon.  t_lo <= 0 clears it (unfreeze)."""
        g = self.g; eps = 1e-9 * max(1.0, self.Tg)
        if not t_lo > eps:
            self.unfreeze(); return
        if self.cont is None:
            raise ValueError("freeze_before needs a stationary continuation (the buffer's frozen maps it extends)")
        if t_lo > self.T + eps or not np.any(np.abs(g.bp - t_lo) <= eps):
            raise ValueError(f"t_lo = {t_lo:g} must be a time breakpoint at or below T = {self.T:g}: {[float(b) for b in g.bp]}")
        self.P_lo = int(np.sum(g.bp < t_lo - eps)); self.t_lo = float(t_lo)
        early = np.concatenate([np.full(pc.n, bool(pc.t1 <= t_lo + eps)) for pc in g.pieces])
        self.fixed_nodes = early | self.buffer
        self.fixed_maps = {}
        for a in self.model.agents:
            gm = np.asarray(maps[a.name], dtype=float)
            self.fixed_maps[a.name] = np.where(self.buffer, self.frozen[a.name], gm[:, :, :self.N])
        self.fixed_time = np.repeat(np.arange(g.P), g.nt) < self.P_lo

    def _wire_past(self, model: Model, L: Optional[float]) -> None:
        """The columns of the world and the old regime's loadings: the channels, then one column per initial
        shock of the past (n_init, ncol, init_names; init_sigma (nX, n_init) the shocks' loads on the states,
        init_rows[agent] (nR, n_init) their point observations in the rows), and E_old, the past's direct noise
        loadings of every row (an increment observed before zero carries them; None without a window).
        Invariant: ncol = nW + n_init; without a past n_init = 0 and E_old is None."""
        past = self.past
        # the columns of the world: the channels, then one column per initial shock of the past
        self.n_init = past.n_initial if past is not None else 0
        self.ncol = self.nW + self.n_init
        self.init_names = list(past.initial_names) if past is not None else []
        if self.n_init:
            self.init_sigma = np.zeros((self.nX, self.n_init))
            self.init_rows: Dict[str, np.ndarray] = {a.name: np.zeros((len(a.signals), self.n_init)) for a in model.agents}
            for i, sh in enumerate(past.initial):
                for st, v in sh.loads.items():
                    self.init_sigma[model.state_names.index(st), i] = v
                for key, e in sh.rows.items():
                    an, rn = key.split(".", 1)
                    ag = next(a for a in model.agents if a.name == an)
                    self.init_rows[an][[r.name for r in ag.signals].index(rn), i] = e
        # the old regime's direct noise loadings of every row (the increments observed before zero carry them)
        self.E_old = {a.name: [past.row_noise(f"{a.name}.{r.name}") for r in a.signals] for a in model.agents} if L else None

    def _wire_time(self, L: Optional[float]) -> None:
        """The map-independent structures on the grid: the Volterra path (r from the shock time, or zero for an
        old shock, to t) with its e^{A(t - r)} weights when there are states; the time rows (trows, trow_by_pit,
        panel_of_node); and the time nodes tm (Nt of them, both one-sided values at a breakpoint, tm_side),
        the line s = 0 (diag, Nd) and mean_embed, which carries a path on the time nodes as the kernel constant
        in shock age on the new-shock region.  Invariant: a time row's nodes are contiguous within its panel,
        and diag is the nodes at age = t of the panels below L (every panel without a strip)."""
        g = self.g
        # state propagation operators (matrix exponentials of A)
        if self.nX:
            # state propagation e^{A(t-r)} along the Volterra path; entrywise weights from expm, which
            # is exact for defective A too (an eigen-decomposition would not be).  An old shock's
            # Volterra path runs from time zero (its earlier inputs are inside the past's state kernel).
            lp = g.path(g.t, g.a, r_lo=g.s if L is None else np.maximum(g.s, 0.0), r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]))
            self._vol_path = lp
            self._vol_E = self._expm_batch(g.t[lp.rows] - lp.r) if lp.rows is not None else None    # (nq, nX, nX)
        # time rows (for per-time projections)
        rows = {}
        for pc in g.pieces:
            for it in range(pc.nt):
                idx = pc.offset + it * pc.na + np.arange(pc.na)
                rows.setdefault((pc.p, it), []).append(idx)
        self.trows: List[Tuple[int, np.ndarray]] = [(k[0], np.concatenate(v)) for k, v in sorted(rows.items())]
        self.trow_by_pit: Dict[Tuple[int, int], np.ndarray] = {k: np.concatenate(v) for k, v in rows.items()}
        self.panel_of_node = np.concatenate([np.full(pc.n, pc.p) for pc in g.pieces])
        # the mean paths live on the time nodes, panel by panel (both one-sided values at a breakpoint): the
        # nodes of each panel's triangle piece on the line s = 0 (age = t), `diag`, where a kernel is the
        # response to a shock at time 0; mean_embed carries a path as the kernel constant in shock age.
        # On a strip cut at age L < T the line s = 0 exists on the panels below L only (diag covers those).
        tri = [g._piece_by_pq[(p, p)] for p in range(min(g.P, g.PL))]
        self.tm = np.concatenate([cheb_lobatto(g.nt, g.bp[p], g.bp[p + 1]) for p in range(g.P)])
        self.Nt = len(self.tm)
        self.tm_side = np.where(np.abs(self.tm - np.repeat(g.bp[1:g.P + 1], g.nt)) < 1e-13, -1, 1)   # a panel's last node reads from below
        self._bwt = bary_weights(g.nt)
        self.diag = np.concatenate([pc.offset + np.arange(pc.nt) * pc.na + pc.na - 1 for pc in tri])
        self.Nd = len(self.diag)                          # time nodes on which the line s = 0 exists
        self.mean_embed = np.zeros((self.N, self.Nt))
        for pc in g.pieces:
            if pc.band:
                continue                                  # a path is carried on the new-shock region only
            for it in range(pc.nt):
                self.mean_embed[pc.offset + it * pc.na + np.arange(pc.na), pc.p * g.nt + it] = 1.0

    def _caches(self) -> None:
        """The caches, empty: the time masses per discount, the mean reads per lag, the map shifts per delay, the
        row operators per (agent, row, excluded), the past's reads and the rows' pre-zero parts, the discrete-
        weight operators, the state columns per (excluded, impulses) and the sparse reads.  Invariant: every
        entry is map-independent, keyed by what it was built from, so it is built once per compiled model."""
        self._time_mass: Dict[float, np.ndarray] = {}
        self._mean_reads: Dict[float, np.ndarray] = {}
        self._map_shifts: Dict[float, np.ndarray] = {}
        self._row_ops: Dict[tuple, tuple] = {}            # (agent, row, excluded) -> (nonzero blocks, deltas) of the seen row (map-independent)
        self._past_reads: Dict[tuple, np.ndarray] = {}    # (name, lag) -> the past's kernel at the nodes before zero
        self._row_pasts: Dict[tuple, np.ndarray] = {}     # (agent, row) -> the seen row's pre-zero part at the nodes
        self._disc: Dict[tuple, np.ndarray] = {}          # ("embed"/"select", delay) -> discrete-weight operators
        self._state_cols: Dict[tuple, np.ndarray] = {}    # (excluded, impulses) -> shock and impulse columns of the state rows
        self._sparse: Dict[tuple, object] = {}            # sparse reads, shifts and row blocks of the closed loop

    @cached_property
    def Vol(self) -> np.ndarray:
        """(nX, nX, N, N) Volterra propagation of the state inputs: e^{A(t-r)} along the path from the shock
        time to t (dense; built on first use by the mean system's line s = 0, the closed loop works from the
        path panel by panel)."""
        Vol = np.zeros((self.nX, self.nX, self.N, self.N))
        if self._vol_E is not None:
            for i in range(self.nX):
                for j in range(self.nX):
                    Vol[i, j] = self._vol_path.apply(self._vol_E[:, i, j])
        return Vol

    def _vol_mat(self, i: int, j: int, X: np.ndarray) -> np.ndarray:
        """Vol[i, j] @ X for X (N, m): the same integrals with X read along the path (no N x N operator)."""
        lp = self._vol_path
        if self._vol_E is None:
            return np.zeros((self.N,) + X.shape[1:])
        G = lp.I @ X
        d = lp.w * self._vol_E[:, i, j]
        return lp.R @ (d[:, None] * G if G.ndim == 2 else d * G)

    # ------------------------------------------------------------ continuation
    @staticmethod
    def _check_continuation(model: Model, res, L: float) -> None:
        """The continuation is a converged StationaryResult of this model (channels, states, controls and
        signal rows by name) at the past's window."""
        from .results import StationaryResult
        if not isinstance(res, StationaryResult):
            raise TypeError(f"continuation must be a StationaryResult (or 'stationary' / 'end'), not {type(res).__name__}")
        if not res.converged:
            raise ValueError(f"the continuation {res.model.name!r} did not converge (residual {res.residual:.2e}, {res.message})")
        m = res.model
        if list(m.channels) != list(model.channels):
            raise ValueError(f"the continuation's channels {list(m.channels)} differ from the model's {list(model.channels)}")
        for what, a, b in (("states", m.state_names, model.state_names), ("controls", m.control_names, model.control_names)):
            if list(a) != list(b):
                raise ValueError(f"the continuation's {what} {list(a)} differ from the model's {list(b)}")
        rows = lambda mm: [(a.name, r.name, float(r.delay)) for a in mm.agents for r in a.signals]
        if rows(m) != rows(model):
            raise ValueError(f"the continuation's signal rows {rows(m)} differ from the model's {rows(model)} (agent, row, delay)")
        Lc = float(res.compiled.grid.L)
        if abs(Lc - L) > 1e-9 * max(1.0, L):
            raise ValueError(f"the continuation's window {Lc:g} differs from the past's {L:g}: the buffer after T is one window, on "
                             "which the stationary maps are read at the node's age; solve the continuation with window "
                             f"{L:g}")

    def _frozen_maps(self) -> Dict[str, np.ndarray]:
        """agent -> (nU, nR, N): the continuation's stationary map at every node's age (used on the buffer; on
        [T - L, T] it is what `settled` compares the solved maps with)."""
        res = self.cont; gs = res.compiled.grid; g = self.g
        top = np.abs(g.a - g.a1) < 1e-9 * max(1.0, self.Tg)      # a node on its piece's top age edge carries the left limit

        def at_ages(ages):
            I = gs.interp(ages, side=+1)
            I[top] = gs.interp(ages[top], side=-1)
            return I
        I0 = at_ages(g.a)
        m = res.model
        self.continuation_info = {"kind": "stationary", "name": m.name, "params": {k: float(v) for k, v in m.params.items()},
                                  "window": float(gs.L), "nodes": int(gs.n), "breakpoints": [float(b) for b in gs.breakpoints],
                                  "converged": bool(res.converged), "residual": float(res.residual),
                                  "window_tail": float(res.window_tail), "costs": {k: float(v) for k, v in res.costs.items()},
                                  "means": {k: float(v) for k, v in res.means.items() if v}}
        out = {}
        eps = 1e-9 * max(1.0, self.Tg)
        for a in self.model.agents:
            fm = np.zeros((len(a.controls), len(a.signals), self.N))
            for r in range(len(a.signals)):
                d = self.row_delays[a.name][r]                  # the stationary map is stored at the seen age a - d
                I = at_ages(g.a - d) if d > 0 else I0
                fm[:, r, :] = np.einsum("fn,un->uf", I, res.maps[a.name][:, r, :])
                if d > 0:
                    fm[:, r, g.a1 <= d + eps] = 0.0
            out[a.name] = fm
        return out

    def with_frozen(self, agent: str, ui: int, r: int, gker: np.ndarray, fixed: bool = True) -> np.ndarray:
        """The map kernel (N,) with the fixed nodes at their fixed values (the input elsewhere): the buffer's at the
        frozen stationary map and, with freeze_before, the panels before t_lo at the given maps; fixed=False takes
        the buffer alone (an excluded agent's kernel in the closed loop)."""
        if self.frozen is None:
            return gker
        if fixed and self.fixed_maps is not None:
            return np.where(self.fixed_nodes, self.fixed_maps[agent][ui, r], gker)
        return np.where(self.buffer, self.frozen[agent][ui, r], gker)

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
            M = g.interp(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
            if da > 0:
                M[g.a0 < da - 1e-12] = 0.0
            if dt > 0 and g.L is not None:
                M[self._before(dt)] = 0.0             # an old shock read before zero: the past's, not the strip's 0+ value
        cache[key] = M
        return M

    def _before(self, lag: float) -> np.ndarray:
        """Nodes of the band whose read `lag` earlier falls before time zero: t - lag < 0, or t - lag = 0 read
        from below (the last node of the panel ending at the lag, whose limit is the pre-zero value)."""
        g = self.g
        eps = 1e-12 * max(1.0, self.Tg)
        return g.upper & ((g.t - lag < -eps) | ((np.abs(g.t - lag) <= eps) & (g.side_t < 0)))

    def past_at(self, name: str, nodes: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """(len, nW): the past's kernel of `name` at the given ages for the given nodes, a node on its piece's top
        age edge reading the left limit (the old kernels jump at the delays: a control's kernel on its row's
        own noise starts at the delay), every other node the right one."""
        g = self.g
        top = np.abs(g.a[nodes] - g.a1[nodes]) < 1e-9 * max(1.0, self.Tg)
        out = self.past.read(name, ages, side=+1)
        if top.any():
            out[top] = self.past.read(name, ages[top], side=-1)
        return out

    def past_read(self, name: str, lag: float) -> np.ndarray:
        """(N, nW): the past's kernel of `name` at age a - lag on the band nodes whose read `lag` earlier is before
        zero (zero elsewhere, and on pieces whose ages start below the lag, where the shock had not arrived)."""
        key = (name, round(float(lag), 12))
        if key not in self._past_reads:
            g = self.g; out = np.zeros((self.N, self.nW))
            if g.L is not None and lag > 0 and name in self.past.kernels:
                sel = self._before(lag) & (g.a0 >= lag - 1e-12)
                if sel.any():
                    idx = np.where(sel)[0]
                    out[idx] = self.past_at(name, idx, g.a[idx] - lag)
            self._past_reads[key] = out
        return self._past_reads[key]

    def row_past(self, agent: str, r: int) -> np.ndarray:
        """(N, nW): the pre-zero part of the seen row r of `agent` on the band (its lagged atoms read before zero,
        every control included: the excluded agent's own pre-zero actions are history)."""
        key = (agent, r)
        if key not in self._row_pasts:
            name, drift, E, delay = self.rows[agent][r]
            out = np.zeros((self.N, self.nW))
            for (n, l), c in drift.items():
                if l + delay > 0:
                    out += c * self.past_read(n, l + delay)
            self._row_pasts[key] = out
        return self._row_pasts[key]

    def zeta_past(self, agent: str) -> np.ndarray:
        """(m, N, nW): the pre-zero part of the agent's loss atoms on the band (lagged atoms read before zero)."""
        atoms, Q, q = self.loss[agent]
        return np.stack([self.past_read(nm, lag) for (nm, lag) in atoms])

    def panel_shift(self, delay: float) -> int:
        """Number of time panels in a lag: the breakpoints are closed under the lags, so every panel
        shifted by a lag is again a panel."""
        g = self.g
        k = int(np.sum((g.bp > 1e-12) & (g.bp <= delay + 1e-12)))
        if abs(g.bp[k] - delay) > 1e-9 * max(1.0, self.Tg):
            raise ValueError(f"lag {delay} is not a breakpoint of the time panels {[float(b) for b in g.bp]}; the panels "
                             "are built from the model's lags and delays (horizon.unit, horizon.breakpoints) and closed "
                             "under them, so a lag read here must be one of the model's")
        return k

    def map_shift(self, delay: float) -> np.ndarray:
        """S (N x N) with (S g)(t, a) = g(t - delay, a - delay) for a map stored at the shifted time: the
        exact node-to-node shift on the delay-aligned pieces (piece (p, q) to (p - k, q - k), same local
        node), zero where the read leaves the domain.  Unlike read(delay, delay) it keeps the nodes of a
        triangle's degenerate bottom row distinct, so no map node is left unread."""
        key = round(float(delay), 12)
        if key not in self._map_shifts:
            g = self.g; S = np.zeros((self.N, self.N))
            for pc, tgt, idx in self._shift_ops(delay):
                if tgt is None:
                    S[idx] = g.interp(g.t[idx] - delay, g.a[idx] - delay, side_t=g.side_t[idx],
                                      side_a=g.side_a[idx], side_d=g.side_ds[idx])
                else:
                    S[idx, tgt.offset + np.arange(pc.n)] = 1.0
            self._map_shifts[key] = S
        return self._map_shifts[key]

    def _shift_ops(self, delay: float):
        """Where each piece's nodes read from under a shift by `delay`, decided once for both builders.

        Yields (piece, target, idx): `target` is the piece the nodes copy from ONE TO ONE, or None
        when the read has to be interpolated; `idx` is the piece's node indices.  The geometry is the
        subtle part -- a buffer piece shifted back into [0, T] lands node to node if it is a
        rectangle and inside that rectangle if it is a TRIANGLE cut by a diagonal of the buffer; a
        piece beyond unit_range is not aligned at all and is interpolated -- and it was written out
        twice, once to fill a dense array and once to append coo triples.  Only the filling differed.

        Raises when a panel shifted back by the lag is not a panel and the grid is not coarse.  The
        sparse builder used to reach that message by calling map_shift(), which built an entire dense
        N x N matrix for the side effect of raising.
        """
        g = self.g; k = self.panel_shift(delay)
        for pc in g.pieces:
            if pc.p - k < 0 or pc.q - k < 0:
                continue
            idx = pc.offset + np.arange(pc.n)
            if pc.p >= g.P_T and pc.p - k < g.P_T:
                # a buffer piece shifted back into [0, T]: a rectangle lands on the rectangle (p - k, q - k) node to
                # node; a triangle (cut by a diagonal of the buffer) lands inside that rectangle and is interpolated
                if pc.triangle:
                    yield pc, None, idx
                    continue
                tgt = g._piece_by_pq.get((pc.p - k, pc.q - k))
            else:
                tgt = (g._upper_by_pq if pc.upper else g._piece_by_pq).get((pc.p - k, pc.q - k))
            if not self._shift_aligned(pc, tgt, delay):
                if not self.coarse:
                    raise ValueError(f"the time panels {[float(b) for b in g.bp]} are not closed under the lag {delay}: the panel "
                                     f"[{pc.t0:g}, {pc.t1:g}] shifted back by the lag is not a panel; drop horizon.breakpoints, "
                                     f"or set horizon.unit to a common divisor of the lags and of the window {self.T}")
                yield pc, None, idx                            # beyond unit_range: the read is interpolated
                continue
            yield pc, tgt, idx

    def _shift_aligned(self, pc, tgt, delay: float) -> bool:
        """Whether the piece pc shifted back by the delay is the piece tgt node to node (the same panel widths
        in time and in age): true on every piece of a grid closed under the delay, within unit_range of a
        coarse one."""
        if tgt is None or tgt.nt != pc.nt or tgt.na != pc.na:
            return False
        tol = 1e-9 * max(1.0, self.Tg)
        return all(abs(x + delay - y) <= tol for x, y in ((tgt.t0, pc.t0), (tgt.t1, pc.t1), (tgt.a0, pc.a0), (tgt.a1, pc.a1)))

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_block(self, atom: Atom):
        """(primary index, the N x N block): where atom_op's one nonzero block sits and what it is.  The
        position is the atom's own primary, so nothing has to look for it."""
        name, lag = atom
        return self.index[name], (self.map_shift(lag) if lag > 0 else self.read(lag, lag))

    def atom_op(self, atom: Atom) -> np.ndarray:
        """N x (n_prim N) matrix giving the kernel of `name@lag` from the primary vector.  A lagged atom reads
        through map_shift (the exact node-to-node shift on the delay-aligned pieces, a triangle's degenerate
        corner nodes kept distinct), not read(lag, lag): that read copies one source node onto the whole
        degenerate row, which is not a contraction in the cost's mass norm (norm 4.8 at 4 and 9.0 at 6 nodes
        per side), so a cross term c D(t) D(t - tau) with |c| < r was not bounded by the own term r D(t)^2 on
        every nodal vector and the loss form of the second-order check turned indefinite (-2.8e-5 at 6 nodes,
        -4.3e-5 at 8 for c = 0.15, r = 0.5).  On the solution kernels, which are single-valued at those nodes,
        the two reads agree; costs and equilibria are unchanged."""
        name, lag = atom
        M = np.zeros((self.N, len(self.prim) * self.N))
        M[:, self.block(name)] = self.map_shift(lag) if lag > 0 else self.read(lag, lag)
        return M

    def expr_op(self, expr) -> np.ndarray:
        M = np.zeros((self.N, len(self.prim) * self.N))
        for (name, lag), c in expr.items():
            M[:, self.block(name)] += c * self.read(lag, lag)
        return M

    def row_blocks(self, agent: str, r: int, excluded: set):
        """Regular part of seen row r of `agent` (already shifted by the observation delay) as the N x N
        operators on the primary kernels it reads, {primary: operator}, and the instantaneous entries
        {source: [(age, weight)]}.  Map-independent, so cached per (agent, row, excluded controls) as the
        nonzero blocks only; the operators are shared and must not be written to."""
        key = (agent, r, frozenset(excluded))
        if key not in self._row_ops:
            self._row_ops[key] = self._row_blocks(agent, r, excluded)
        blocks, deltas = self._row_ops[key]
        return blocks, {k: list(v) for k, v in deltas.items()}

    def _nonzero_blocks(self, reg: np.ndarray):
        """The primaries whose N x N block of an operator on the primary vector is not identically zero."""
        return [p for p in range(len(self.prim)) if np.any(reg[:, p * self.N:(p + 1) * self.N])]

    def _row_blocks(self, agent: str, r: int, excluded: set, sparse: bool = False):
        """The blocks and deltas of row_blocks; sparse: the blocks as CSR matrices (the per-panel closed loop)."""
        name, drift, E, delay = self.rows[agent][r]
        read = self.read_sparse if sparse else self.read
        S = read(delay, delay)
        blocks: Dict[str, np.ndarray] = {}
        for (n, l), c in drift.items():
            if n not in excluded or self.cont is not None:      # an excluded control still acts on the buffer (its frozen map)
                op = c * (S @ read(l, l))
                blocks[n] = blocks[n] + op if n in blocks else op
        return blocks, self._row_deltas(agent, r, excluded)

    def _row_deltas(self, agent: str, r: int, excluded: set):
        """The instantaneous entries of seen row r: {source: [(age, weight)]}, the excluded controls' impulses
        at the observation delay plus their lag and the channels' noise at the delay."""
        name, drift, E, delay = self.rows[agent][r]
        deltas: Dict[str, List[Tuple[float, float]]] = {}
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
        for k, ch in enumerate(self.channels):
            # with a past the entry exists on the union of the two regimes' loadings: an increment observed before
            # zero carries the old E on the band (noise_weight), and a channel only the old row loaded is not dropped
            if E[k] != 0.0 or (self.E_old is not None and self.E_old[agent][r][k] != 0.0):
                deltas.setdefault(ch, []).append((delay, E[k]))
        return deltas

    # ------------------------------------------------------ sparse reads (the closed loop)
    # The reads, shifts and row blocks as CSR matrices: an interpolation touches one piece's nodes per point, so
    # the N x N dense forms (a gigabyte each at ten thousand nodes) are never built for the closed loop.
    def read_sparse(self, dt: float, da: float):
        """read(dt, da) as a CSR matrix."""
        from scipy.sparse import csr_matrix, diags, identity
        key = ("read", round(dt, 12), round(da, 12))
        if key not in self._sparse:
            g = self.g
            if dt == 0.0 and da == 0.0:
                M = identity(self.N, format="csr")
            else:
                M = g.interp_sparse(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
                zero = np.zeros(self.N, dtype=bool)
                if da > 0:
                    zero |= g.a0 < da - 1e-12
                if dt > 0 and g.L is not None:
                    zero |= self._before(dt)
                if zero.any():
                    M = csr_matrix(diags((~zero).astype(float)) @ M)
                    M.eliminate_zeros()
            self._sparse[key] = M
        return self._sparse[key]

    def map_shift_sparse(self, delay: float):
        """map_shift(delay) as a CSR matrix."""
        from scipy.sparse import csr_matrix
        key = ("shift", round(float(delay), 12))
        if key not in self._sparse:
            g = self.g
            rows, cols, vals = [], [], []
            for pc, tgt, idx in self._shift_ops(delay):
                if tgt is None:
                    I = g.interp_sparse(g.t[idx] - delay, g.a[idx] - delay, side_t=g.side_t[idx], side_a=g.side_a[idx],
                                        side_d=g.side_ds[idx]).tocoo()
                    rows.append(idx[I.row]); cols.append(I.col); vals.append(I.data)
                else:
                    rows.append(idx); cols.append(tgt.offset + np.arange(pc.n)); vals.append(np.ones(pc.n))
            if rows:
                M = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.N, self.N))
            else:
                M = csr_matrix((self.N, self.N))
            self._sparse[key] = M
        return self._sparse[key]

    def instant_sparse(self, age: float, delay: float = 0.0):
        """instant(age, delay) as a CSR matrix."""
        if delay > 0 and abs(age - delay) < 1e-12:
            return self.map_shift_sparse(delay)
        return self.read_sparse(delay, age)

    def atom_sparse(self, atom: Atom):
        """The N x N block of atom_op (the kernel of name@lag from the primary's kernel) as a CSR matrix."""
        name, lag = atom
        return self.map_shift_sparse(lag) if lag > 0 else self.read_sparse(lag, lag)

    def row_blocks_sparse(self, agent: str, r: int, excluded: set):
        """row_blocks with the blocks as CSR matrices (cached, shared: not to be written to)."""
        key = ("row", agent, r, frozenset(excluded))
        if key not in self._sparse:
            self._sparse[key] = self._row_blocks(agent, r, excluded, sparse=True)
        blocks, deltas = self._sparse[key]
        return blocks, {k: list(v) for k, v in deltas.items()}

    def state_inputs_sparse(self, excl: set):
        """The state inputs' operators of the closed loop as {state: {primary index: CSR N x N}} (the atoms the
        state's drift reads, an excluded control's kept only where it acts on a buffer)."""
        key = ("inputs", frozenset(excl))
        if key not in self._sparse:
            inp: Dict[int, Dict[int, object]] = {si: {} for si in range(self.nX)}
            for si, (nm, lag), c in self.state_inputs:
                if nm in excl and (self.past is None or self.cont is None):
                    continue
                pi = self.index[nm]
                op = c * self.atom_sparse((nm, lag))
                inp[si][pi] = inp[si][pi] + op if pi in inp[si] else op
            self._sparse[key] = inp
        return self._sparse[key]

    # ------------------------------------------------------ line operators
    # Each family of line integrals is a cached quadrature structure (triangle.LinePath);
    # an operator for a given known kernel is then two sparse products.
    # One quadrature structure per line geometry.  conv_right(d) is conv_left(d) with the roles of the
    # unknown and the known exchanged (the same points, cuts and weights; the read matrices I and J swap),
    # and the response path is conv_left at delay 0 (r from s to t, unknown at (r, r - s), known at
    # (t, t - r)); both are served from the conv_left path without building a second set of read matrices.
    @staticmethod
    def _path_alias(key):
        """(the key whose path is built, whether this key is its swap)."""
        if key == ("response",):
            return ("conv_left", 0.0), False
        if key[0] == "conv_right":
            return ("conv_left", key[1]), True
        return key, False

    def _path(self, key, **kw):
        paths = self.g.paths
        if key not in paths:
            base, swap = self._path_alias(key)
            if base not in paths:
                if swap:                    # build the base geometry: the unknown reads what this key's known reads
                    kw = dict(kw, point_fn=kw["known_fn"], known_fn=kw["point_fn"])
                paths[base] = self.g.path(self.g.t, self.g.a, side_t=self.g.side_t, side_d=self.g.side_ds, **kw)
            paths[key] = paths[base].swapped() if swap else paths[base]
        return paths[key]

    # The map on a row observed with delay d is stored at the shifted time: the nodal value at (t', b) is
    # g(t' + d, b), the weight the control at t' + d puts on the observation increment of age b.  The
    # map's domain b <= t - d is then the standard triangle b <= t', its instantaneous read from the
    # action grid is the exact node shift map_shift(d), and nothing is masked inside a piece.
    def past_conv_path(self):
        """For an old-shock action node (t, a): int_t^a g(t, b) kappa(a - b) db, the control's weight on the
        increments observed before zero (ages b in (t, a] before the control) against the past's raw row kernel
        at the increment's age after the shock."""
        key = ("past_conv", id(self.past.grid))
        if key not in self.g.paths:
            g = self.g; up = g.upper
            self.g.paths[key] = g.path(g.t, g.a, r_lo=np.where(up, g.t, 0.0), r_hi=np.where(up, g.a, 0.0),
                                       point_fn=lambda k, b: (np.full_like(b, g.t[k]), b), known_fn=lambda k, b: g.a[k] - b,
                                       extra_cuts=lambda k: [g.a[k] - c for c in self.past.breakpoints], side_t=g.side_t,
                                       side_d=g.side_ds, known_grid=self.past.grid)
        return self.g.paths[key]

    def past_proj_path(self):
        """For a map node (t, b) above the diagonal (the increment observed at u = t - b < 0):
        int_{t - L}^{u} phi(t, t - s) kappa(u - s) ds, the projection of the FOC kernel on that increment's
        regular part."""
        key = ("past_proj", id(self.past.grid))
        if key not in self.g.paths:
            g = self.g; up = g.upper; L = g.L
            self.g.paths[key] = g.path(g.t, g.a, r_lo=np.where(up, g.t - L, 0.0), r_hi=np.where(up, g.s, 0.0),
                                       point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r), known_fn=lambda k, r: g.s[k] - r,
                                       extra_cuts=lambda k: [g.s[k] - c for c in self.past.breakpoints], side_t=g.side_t,
                                       side_d=g.side_ds, known_grid=self.past.grid)
        return self.g.paths[key]

    def old_shock_proj_path(self):
        """For a map node (t, b) below the diagonal (the increment at u = t - b >= 0): int_{t - L}^{0} phi(t, t - s)
        y(u, u - s) ds, the projection of the FOC kernel on the old shocks the increment carries (both on the strip)."""
        key = ("old_proj",)
        if key not in self.g.paths:
            g = self.g; L = g.L
            lo = np.where(~g.upper & (g.t < L - 1e-12), g.t - L, 0.0)
            self.g.paths[key] = g.path(g.t, g.a, r_lo=lo, r_hi=np.zeros(self.N),
                                       point_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r),
                                       known_fn=lambda k, r: (np.full_like(r, g.s[k]), g.s[k] - r), side_t=g.side_t, side_d=g.side_ds)
        return self.g.paths[key]

    def past_row_kernel(self, agent: str, r: int) -> np.ndarray:
        """(N_past, nW): the past's raw row kernel of row r of `agent` on the past's age grid."""
        name = self.rows[agent][r][0]
        return self.past.rows[f"{agent}.{name}"][0]

    def disc_embed(self, delay: float = 0.0) -> np.ndarray:
        """(N, Nt): the discrete weight w(t') stored at the shifted time t' = t - delay carried to the nodes at
        time t of the new-shock region (constant in age, like a mean path)."""
        key = ("embed", round(float(delay), 12))
        if key not in self._disc:
            g = self.g; k = self.panel_shift(delay) if delay > 0 else 0
            if k and self.coarse:
                raise ValueError("a delayed row's discrete weights on the initial shocks need the time panels closed under the delay; "
                                 "horizon.unit_range below the window is not supported with initial shocks and a delayed row")
            E = np.zeros((self.N, self.Nt))
            for pc in g.pieces:
                if pc.band or pc.p - k < 0:
                    continue
                for it in range(pc.nt):
                    E[pc.offset + it * pc.na + np.arange(pc.na), (pc.p - k) * g.nt + it] = 1.0
            self._disc[key] = E
        return self._disc[key]

    def disc_select(self, delay: float = 0.0) -> np.ndarray:
        """(Nt, N): the kernel on the line s = 0 at time t' + delay, at the time node t' of a discrete weight."""
        key = ("select", round(float(delay), 12))
        if key not in self._disc:
            g = self.g; k = self.panel_shift(delay) if delay > 0 else 0
            if k and self.coarse:
                raise ValueError("a delayed row's discrete weights on the initial shocks need the time panels closed under the delay; "
                                 "horizon.unit_range below the window is not supported with initial shocks and a delayed row")
            S = np.zeros((self.Nt, self.N))
            for j, node in enumerate(self.diag):                       # time node p * nt + it of the line s = 0
                p, it = divmod(j, g.nt)
                if p - k >= 0:
                    S[(p - k) * g.nt + it, node] = 1.0
            self._disc[key] = S
        return self._disc[key]

    # ------------------------------------------------ the masses (expected_cost, finite_free)
    def cost_mass_sparse(self):
        """cost_mass as a CSR matrix (block diagonal by piece; the matrix-free best response)."""
        return self.g.mass_sparse(rho=self.rho, t_hi=None if self.cont is None else self.T)

    def buffer_mass_sparse(self):
        """buffer_mass as a CSR matrix."""
        return self.g.mass_sparse(rho=self.rho, t_lo=self.T)

    def diag_read_sparse(self, delay: float = 0.0):
        """diag_read as a CSR matrix."""
        key = ("diag_read_sparse", round(float(delay), 12))
        if key not in self._sparse:
            g = self.g
            self._sparse[key] = g.interp_sparse(g.t + delay, g.t + delay, side_t=g.side_t)
        return self._sparse[key]

    def shock_time_read_sparse(self):
        """shock_time_read as a CSR matrix."""
        from scipy.sparse import csr_matrix, diags
        key = ("shock_time_sparse",)
        if key not in self._sparse:
            g = self.g
            M = csr_matrix(diags((~g.upper).astype(float)) @ g.interp_sparse(g.s, g.s, side_t=g.side_t))
            M.eliminate_zeros()
            self._sparse[key] = M
        return self._sparse[key]

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None, impulse_controls=(), own_frozen: bool = True,
                    actions: Optional[Dict[str, np.ndarray]] = None):
        """maps[agent]: (n_ctrl, n_rows, N) nodal raw maps g(t, b).  Columns: Brownian channels,
        then one impulse column per control in impulse_controls (unit mass at the shock time).
        Returns Z (n_prim N, ncol).
        With a past: maps (n_ctrl, n_rows, N + Nt), the map on the strip (the band's nodes weigh the
        increments observed before zero) then the discrete weights on the initial shocks' point
        observations; the columns are the channels, the initial shocks, then the impulses.
        With a continuation every map is the frozen stationary one on the buffer; the excluded agent's
        too (its strategy off on [0, T] only: the buffer's is part of its environment) unless
        own_frozen=False switches it off on the buffer as well (the envelope response of its FOC).
        With actions (agent -> (nU, N, ncol)) every control's kernels are given and only the states are
        solved (world_from_actions).  The assembly, one time panel at a time: ClosedLoopRows."""
        return ClosedLoopRows(self, maps, excluded, impulse_controls, own_frozen, actions).solve()

    def noise_weight(self, agent: str, r: int, k: int, w: float) -> np.ndarray:
        """(N,): the weight of the row's own noise increment on channel k at every action node: the model's
        loading w on the new-shock region, the past's on the band (an increment observed before zero)."""
        g = self.g
        if g.L is None:
            return np.full(self.N, float(w))
        return np.where(g.upper, self.E_old[agent][r][k], float(w))

    @cached_property
    def _state_blocks(self) -> Dict[Tuple[int, int], np.ndarray]:
        """The state rows of the closed loop without the maps and with no agent excluded, dense: the nonzero
        N x N blocks {(state, primary): Vol @ input operator} of the Volterra propagation of the state inputs
        (the mean system's line s = 0 reads their diagonal; the closed loop itself works from the path)."""
        N = self.N; n = len(self.prim) * N
        inp = np.zeros((self.nX, N, n))
        for si, (nm, lag), c in self.state_inputs:
            inp[si] += c * self.atom_op((nm, lag))
        blocks: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(self.nX):
            for j in range(self.nX):
                for p in self._nonzero_blocks(inp[j]):
                    blk = self.Vol[i, j] @ inp[j][:, p * N:(p + 1) * N]
                    blocks[(i, p)] = blocks[(i, p)] + blk if (i, p) in blocks else blk
        return blocks

    @cached_property
    def _panel_ranges(self):
        """Node ranges [lo, hi) of every time panel: pieces are stored panel by panel, so each panel's
        nodes are contiguous within every primary's block."""
        panel = self.panel_of_node
        return [(int(np.searchsorted(panel, p)), int(np.searchsorted(panel, p, side="right"))) for p in range(self.g.P)]
