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
The means (targets, constant drifts, initial states) are deterministic paths on
the time nodes, solved at the end from every control's mean first-order condition
and the mean dynamics as one linear system (see SpectralFiniteSolver.mean_system).

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
from numpy.polynomial import legendre
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve

from .engine import EngineBase
from .compile import CompiledBase, close_under_delays, reject_leads
from .grid import bary_rows, bary_weights, cheb_lobatto
from .results import TriangleResult, TransitionResult
from .settings import tunable
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid
from .grid_cache import triangle_grid
from .past import Past


def _common_unit_hint(lags, kmax: int = 100) -> str:
    """' (u works)' for the largest u = min(lags) / k, k <= kmax, of which every lag is a multiple; the lags
    are incommensurable at that resolution otherwise, which no panel grid carries."""
    lo = min(lags)
    for k in range(1, kmax + 1):
        u = lo / k
        if all(abs(l / u - round(l / u)) < 1e-9 for l in lags):
            return f" ({u:g} works)"
    return f" (none above {lo / kmax:g}: the finite engines need commensurable lags)"


class SpectralCompiled(CompiledBase):
    def __init__(self, model: Model, past=None, continuation=None):
        """past: a Past (the game starts at time zero from that regime); continuation: a converged
        StationaryResult of this model at the past's window, on which every agent's map is frozen on the
        buffer [T, T + L] after the horizon (the unknowns stay on [0, T]; the closed loop and the
        first-order conditions run to T + L, by when every shock born before T is forgotten), or None:
        the game ends at T."""
        super().__init__(model)
        reject_leads(model, 'spectral finite engine')
        hz = model.horizon
        self.T = float(hz.window)
        lags = model.all_lags()
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
            if self.T < L - 1e-12:
                raise ValueError(f"the horizon T = {self.T:g} is shorter than the past's window L = {L:g}: with a stationary "
                                 "continuation the shocks born before zero must be forgotten by T; raise horizon.window to at least L")
        self.Tg = self.T + L if continuation is not None else self.T        # the grid's end: the buffer [T, T + L] follows T
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
        closed = close_under_delays(bp, lags) if lags else bp
        if len(closed) > len(bp):
            added = [round(float(b), 6) for b in closed if not any(abs(b - x) < 1e-9 for x in bp)]
            P0, P1 = len(bp) - 1, len(closed) - 1
            warnings.warn(f"the time panels are closed under the lag(s) {lags}: breakpoints {added} added ({P1} panels, "
                          f"{P1 * (P1 + 1) // 2} pieces, instead of {P0} panels, {P0 * (P0 + 1) // 2} pieces); a window that "
                          "is a multiple of every lag, with breakpoints closed under them, avoids the extra panels")
        bp = closed
        if L is None:
            self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes)   # shared
        else:
            # the strip: one breakpoint sequence for time and age, the past grid's panels and L among them
            # (the old kernels kink on the past's panels, which then lie on piece edges), closed under the lags
            bp = sorted(set(bp) | {b for b in past.breakpoints if b < L - 1e-12} | {float(L)})
            if not lags and not hz.breakpoints:                     # no lags: time panels of width L (the shocks' lifetime), then T
                bp = sorted(set(bp) | {float(b) for b in np.arange(0.0, self.T - 1e-12, L)})
            if continuation is not None:                            # the buffer's time panels: the age panels shifted to T
                bp = sorted(set(bp) | {round(self.T + b, 12) for b in bp if b <= L + 1e-12})
            bp = close_under_delays(bp, lags) if lags else bp
            if continuation is not None:
                self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes, self.Tg, float(L), self.T)
            else:
                self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes, self.Tg, float(L))
        g = self.g
        self.N = g.N
        self.rho = float(hz.discount)
        # the buffer: the nodes of the pieces after T, where every map is frozen at the continuation's stationary
        # map at the node's age; P_T counts the time panels up to T (the unknowns' panels)
        eps = 1e-12 * max(1.0, self.Tg)
        self.P_T = int(np.sum(g.bp < self.T - eps))
        self.buffer = np.concatenate([np.full(pc.n, bool(pc.t0 >= self.T - eps)) for pc in g.pieces]) if continuation is not None \
            else np.zeros(self.N, dtype=bool)
        self.frozen: Optional[Dict[str, np.ndarray]] = self._frozen_maps() if continuation is not None else None
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
        # state propagation operators (matrix exponentials of A)
        if self.nX:
            # state propagation e^{A(t-r)} along the Volterra path; entrywise weights from expm, which
            # is exact for defective A too (an eigen-decomposition would not be).  An old shock's
            # Volterra path runs from time zero (its earlier inputs are inside the past's state kernel).
            lp = g.path(g.t, g.a, r_lo=g.s if L is None else np.maximum(g.s, 0.0), r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]))
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
        self._time_mass: Dict[float, np.ndarray] = {}
        self._mean_reads: Dict[float, np.ndarray] = {}
        self._map_shifts: Dict[float, np.ndarray] = {}
        self._row_ops: Dict[tuple, tuple] = {}            # (agent, row, excluded) -> (nonzero blocks, deltas) of the seen row (map-independent)
        self._state_parts: Dict[tuple, tuple] = {}        # (excluded, impulses) -> state part of the closed loop
        self._past_reads: Dict[tuple, np.ndarray] = {}    # (name, lag) -> the past's kernel at the nodes before zero
        self._row_pasts: Dict[tuple, np.ndarray] = {}     # (agent, row) -> the seen row's pre-zero part at the nodes
        self._disc: Dict[tuple, np.ndarray] = {}          # ("embed"/"select", delay) -> discrete-weight operators

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

    def with_frozen(self, agent: str, ui: int, r: int, gker: np.ndarray) -> np.ndarray:
        """The map kernel (N,) with the buffer's nodes at the frozen stationary map (the input elsewhere)."""
        if self.frozen is None:
            return gker
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
            M = g.interp(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a, side_d=g.side_d)
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
            g = self.g; k = self.panel_shift(delay); S = np.zeros((self.N, self.N))
            for pc in g.pieces:
                if pc.p - k < 0 or pc.q - k < 0:
                    continue
                if pc.p >= g.P_T and pc.p - k < g.P_T:
                    # a buffer piece shifted back into [0, T]: a rectangle lands on the rectangle (p - k, q - k) node to
                    # node; a triangle (cut by a diagonal of the buffer) lands inside that rectangle and is interpolated
                    if pc.triangle:
                        idx = pc.offset + np.arange(pc.n)
                        S[idx] = g.interp(g.t[idx] - delay, g.a[idx] - delay, side_t=g.side_t[idx], side_a=g.side_a[idx], side_d=g.side_d[idx])
                        continue
                    tgt = g._piece_by_pq[(pc.p - k, pc.q - k)]
                else:
                    tgt = (g._upper_by_pq if pc.upper else g._piece_by_pq)[(pc.p - k, pc.q - k)]
                tol = 1e-9 * max(1.0, self.Tg)
                if abs(tgt.t0 + delay - pc.t0) > tol or abs(tgt.t1 + delay - pc.t1) > tol:
                    raise ValueError(f"the time panels {[float(b) for b in g.bp]} are not closed under the lag {delay}: the panel "
                                     f"[{pc.t0:g}, {pc.t1:g}] shifted back by the lag is not a panel; drop horizon.breakpoints, "
                                     f"or set horizon.unit to a common divisor of the lags and of the window {self.T}")
                S[pc.offset + np.arange(pc.n), tgt.offset + np.arange(pc.n)] = 1.0
            self._map_shifts[key] = S
        return self._map_shifts[key]

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

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

    def row_op(self, agent: str, r: int, excluded: set):
        """row_blocks assembled as one operator (N x n_prim N) on the primary vector (built on demand)."""
        blocks, deltas = self.row_blocks(agent, r, excluded)
        reg = np.zeros((self.N, len(self.prim) * self.N))
        for n, op in blocks.items():
            reg[:, self.block(n)] += op
        return reg, deltas

    def _nonzero_blocks(self, reg: np.ndarray):
        """The primaries whose N x N block of the row operator is not identically zero."""
        return [p for p in range(len(self.prim)) if np.any(reg[:, p * self.N:(p + 1) * self.N])]

    def _row_blocks(self, agent: str, r: int, excluded: set):
        name, drift, E, delay = self.rows[agent][r]
        S = self.read(delay, delay)
        blocks: Dict[str, np.ndarray] = {}
        deltas: Dict[str, List[Tuple[float, float]]] = {}
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
            if n not in excluded or self.cont is not None:      # an excluded control still acts on the buffer (its frozen map)
                op = c * (S @ self.read(l, l))
                blocks[n] = blocks[n] + op if n in blocks else op
        for k, ch in enumerate(self.channels):
            # with a past the entry exists on the union of the two regimes' loadings: an increment observed before
            # zero carries the old E on the band (noise_weight), and a channel only the old row loaded is not dropped
            if E[k] != 0.0 or (self.E_old is not None and self.E_old[agent][r][k] != 0.0):
                deltas.setdefault(ch, []).append((delay, E[k]))
        return blocks, deltas

    # ------------------------------------------------------ line operators
    # Each family of line integrals is a cached quadrature structure (triangle.LinePath);
    # an operator for a given known kernel is then two sparse products.
    @cached_property
    def _panel_idx(self):
        """Indices of every primary's unknowns on each time panel, in block order."""
        panel = np.concatenate([np.full(pc.n, pc.p) for pc in self.g.pieces])
        return [np.concatenate([q * self.N + np.where(panel == p)[0] for q in range(len(self.prim))]) for p in range(self.g.P)]

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
                paths[base] = self.g.path(self.g.t, self.g.a, side_t=self.g.side_t, side_d=self.g.side_d, **kw)
            paths[key] = paths[base].swapped() if swap else paths[base]
        return paths[key]

    # The map on a row observed with delay d is stored at the shifted time: the nodal value at (t', b) is
    # g(t' + d, b), the weight the control at t' + d puts on the observation increment of age b.  The
    # map's domain b <= t - d is then the standard triangle b <= t', its instantaneous read from the
    # action grid is the exact node shift map_shift(d), and nothing is masked inside a piece.
    def conv_left(self, gker: np.ndarray, delay: float) -> np.ndarray:
        """(C y)(t, s) = int_{s+delay}^{t} g(t, t - u) y(u, s) du  for a fixed map kernel g (stored at t - delay)."""
        g = self.g
        lp = self._path(("conv_left", delay), r_lo=g.s + delay, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k] - delay), g.t[k] - r))
        return lp.with_known(gker)

    def conv_right(self, yker: np.ndarray, delay: float) -> np.ndarray:
        """(C g)(t, s) = int_{s+delay}^{t} g(t, t - u) y_seen(u, s) du  for a fixed seen row y_seen (g stored at t - delay)."""
        g = self.g
        lp = self._path(("conv_right", delay), r_lo=g.s + delay, r_hi=g.t,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k] - delay), g.t[k] - r), known_fn=lambda k, r: (r, r - g.s[k]))
        return lp.with_known(yker)

    def response_op(self, rker: np.ndarray) -> np.ndarray:
        """(C c)(t, s) = int_s^t R(t, t - r) c(r, s) dr  for a fixed impulse-response kernel R."""
        g = self.g
        lp = self._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        return lp.with_known(rker)

    def continuation_op(self, rker: np.ndarray) -> np.ndarray:
        """(C z)(t, s) = int_t^T e^{-rho (tau - t)} R(tau, tau - t) z(tau, s) dtau (to T + L with a continuation)."""
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.Tg), point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-self.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        return lp.with_known(rker, disc)

    def projection_op(self, yker: np.ndarray, delay: float) -> np.ndarray:
        """(H phi)(t', b) = int_0^{u} phi(t' + delay, s) y_raw(u, s) ds with u = t' - b, at the map node (t', b)
        of a row observed with delay (the map stored at the shifted time t' = t - delay)."""
        g = self.g
        u = g.s
        lp = self._path(("projection", delay), r_lo=np.zeros(self.N), r_hi=u,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k] + delay), g.t[k] + delay - r),
                        known_fn=lambda k, r: (np.full_like(r, u[k]), u[k] - r))
        return lp.with_known(yker)

    # ------------------------------------------------------ known-past paths
    # Line integrals that cross time zero: the pre-zero segment reads the past's kernel on the past's own
    # age grid (LinePath with known_grid), cut where the age crosses the past grid's breakpoints.
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
                                       side_d=g.side_d, known_grid=self.past.grid)
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
                                       side_d=g.side_d, known_grid=self.past.grid)
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
                                       known_fn=lambda k, r: (np.full_like(r, g.s[k]), g.s[k] - r), side_t=g.side_t, side_d=g.side_d)
        return self.g.paths[key]

    def past_row_kernel(self, agent: str, r: int) -> np.ndarray:
        """(N_past, nW): the past's raw row kernel of row r of `agent` on the past's age grid."""
        name = self.rows[agent][r][0]
        return self.past.rows[f"{agent}.{name}"][0]

    def diag_read(self, delay: float = 0.0) -> np.ndarray:
        """(N, N): a kernel at (t + delay, s = 0), the line of a shock at time zero, from every node's time."""
        key = ("diag_read", round(float(delay), 12))
        if key not in self._disc:
            g = self.g
            self._disc[key] = g.interp(g.t + delay, g.t + delay, side_t=g.side_t)
        return self._disc[key]

    def shock_time_read(self) -> np.ndarray:
        """(N, N): a kernel at (s, s), its value at the node's own shock time on the line s = 0 (zero on the band)."""
        key = ("shock_time",)
        if key not in self._disc:
            g = self.g
            M = g.interp(g.s, g.s, side_t=g.side_t)
            M[g.upper] = 0.0
            self._disc[key] = M
        return self._disc[key]

    def disc_embed(self, delay: float = 0.0) -> np.ndarray:
        """(N, Nt): the discrete weight w(t') stored at the shifted time t' = t - delay carried to the nodes at
        time t of the new-shock region (constant in age, like a mean path)."""
        key = ("embed", round(float(delay), 12))
        if key not in self._disc:
            g = self.g; k = self.panel_shift(delay) if delay > 0 else 0
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
            S = np.zeros((self.Nt, self.N))
            for j, node in enumerate(self.diag):                       # time node p * nt + it of the line s = 0
                p, it = divmod(j, g.nt)
                if p - k >= 0:
                    S[(p - k) * g.nt + it, node] = 1.0
            self._disc[key] = S
        return self._disc[key]

    # ------------------------------------------------ kernel algebra (see EngineBase)
    def _many(self, lp, K: np.ndarray, extra=None) -> np.ndarray:
        """with_known over the nonzero columns of K (N, m) at once: (m, N, N), zero for a zero column."""
        out = np.zeros((K.shape[1], self.N, self.N))
        nz = [k for k in range(K.shape[1]) if np.abs(K[:, k]).max() > 0]
        if nz:
            out[nz] = lp.with_known_many(K[:, nz], extra)
        return out

    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(m, N, N) conv_right operators of the m seen row kernels Y (N, m) of a row observed with `delay`:
        the map stored at the shifted time -> the action kernel; zero for a zero column."""
        g = self.g
        lp = self._path(("conv_right", delay), r_lo=g.s + delay, r_hi=g.t,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k] - delay), g.t[k] - r), known_fn=lambda k, r: (r, r - g.s[k]))
        return self._many(lp, Y)

    def instant(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N) read of the map on a row observed with `delay` at the age of an instantaneous entry: the
        exact node shift map_shift(delay) for the row's own noise (age == delay), read(delay, age) otherwise."""
        if delay > 0 and abs(age - delay) < 1e-12:
            return self.map_shift(delay)
        return self.read(delay, age)

    def instant_adjoint(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N) adjoint of instant on the FOC kernel."""
        if delay > 0 and abs(age - delay) < 1e-12:
            return self.map_shift(delay).T
        return self.read(-delay, -age)

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        """(n_prim N, N) response operators of every primary to an action kernel, from the impulse responses
        Ru (n_prim, N); the own block (primary index `own`) is zeroed here and set to the identity by the base."""
        g = self.g; N = self.N
        lp = self._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        K = Ru.T.copy(); K[:, own] = 0.0
        return self._many(lp, K).reshape(len(self.prim) * N, N)

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        """(m, N, N) discounted continuation operators of the m atom responses Rj (N, m), int_t^T e^{-rho (tau - t)} ...
        (to T + L with a continuation: the buffer's frozen maps are in the responses)."""
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.Tg), point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-self.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        return self._many(lp, Rj, disc)

    def own_lag_read(self, lag: float) -> np.ndarray:
        """(N, N) read at (t + lag, a + lag): the FOC term of the control's own read `lag` later."""
        return self.read(-lag, -lag)

    def cost_mass(self) -> np.ndarray:
        """The discounted Gram matrix under which expected_cost integrates products of kernels: over [0, T] (the
        buffer of a continuation is buffer_mass)."""
        if self.cont is None:
            return self.g.mass_matrix(rho=self.rho)
        return self.g.mass_matrix(rho=self.rho, t_hi=self.T)

    def buffer_mass(self) -> np.ndarray:
        """The discounted Gram matrix over the buffer [T, T + L] of a continuation."""
        return self.g.mass_matrix(rho=self.rho, t_lo=self.T)

    # ------------------------------------------------------------ means
    def time_mass(self, rho: float) -> np.ndarray:
        """Weights w (Nt,) with int_0^T e^{-rho t} f(t) dt = w @ f for a path f on the time nodes: Gauss quadrature
        of the interpolant on every panel (exact for a polynomial of the panel's degree at rho = 0); zero on
        the buffer's panels."""
        key = round(float(rho), 12)
        if key not in self._time_mass:
            g = self.g; nt = g.nt
            xg, wg = legendre.leggauss(nt + 2)
            w = np.zeros(self.Nt)
            for p in range(self.P_T):
                t0, t1 = g.bp[p], g.bp[p + 1]
                tq = 0.5 * (t1 - t0) * xg + 0.5 * (t1 + t0); tw = 0.5 * (t1 - t0) * wg
                w[p * nt:(p + 1) * nt] = (tw * np.exp(-key * tq)) @ bary_rows(tq, self.tm[p * nt:(p + 1) * nt], self._bwt)
            self._time_mass[key] = w
        return self._time_mass[key]

    @cached_property
    def mean_line0(self) -> np.ndarray:
        """(Nt, N) reading a kernel at (t, age 0) on every time node, from the node's side of its panel: the value
        at the birth of a shock at time t, where the mean first-order condition of a strip lives (its
        continuation runs from there to t + L through the responses to an impulse at t)."""
        return self.g.interp(self.tm, np.zeros(self.Nt), side_t=self.tm_side)

    @cached_property
    def mean_volterra(self) -> np.ndarray:
        """(nX, nX, Nt, Nt): (V_ij f)(t) = int_0^t (e^{A(t - r)})_ij f(r) dr for a path f on the time nodes, by Gauss
        quadrature on every panel (nt + 2 points, the partial panel cut at t) of the panel's interpolant: the
        mean dynamics of a strip, on which the line s = 0 is cut at age L."""
        g = self.g; nt = g.nt; Nt, nX = self.Nt, self.nX
        V = np.zeros((nX, nX, Nt, Nt))
        if not nX:
            return V
        xg, wg = legendre.leggauss(nt + 2)
        for k, t in enumerate(self.tm):
            for p in range(g.P):
                t0, t1 = g.bp[p], min(g.bp[p + 1], t)
                if t1 - t0 < 1e-14:
                    break
                rq = 0.5 * (t1 - t0) * xg + 0.5 * (t1 + t0); wq = 0.5 * (t1 - t0) * wg
                Bq = bary_rows(rq, self.tm[p * nt:(p + 1) * nt], self._bwt)      # (nq, nt)
                EA = self.expA(t - rq)                                            # (nq, nX, nX)
                V[:, :, k, p * nt:(p + 1) * nt] += np.einsum("q,qij,qn->ijn", wq, EA, Bq)
        return V

    def mean_read(self, lag: float) -> np.ndarray:
        """(Nt x Nt) reading a path on the time nodes at t - lag, zero before 0, from each node's side of its panel:
        the kernels' read of a lagged atom on the line s = 0 (exact on the panels for the model's lags, which are
        breakpoints; the interpolant within a panel for any other lag, a definition's)."""
        key = round(float(lag), 12)
        if key not in self._mean_reads:
            if key == 0.0:
                self._mean_reads[key] = np.eye(self.Nt)
            else:
                t = self.tm - key
                a = t if self.g.L is None else np.zeros(self.Nt)          # the line s = 0, or (cut at L) the row's age-0 node
                self._mean_reads[key] = self.g.interp(t, a, side_t=self.tm_side) @ self.mean_embed
        return self._mean_reads[key]

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(N, m N), columns (channel, node): the projection of the FOC kernel on the m seen row kernels Y (N, m)
        of a row observed with `delay`, at every map node of the row (the raw row is Y read back by the delay)."""
        g = self.g; N = self.N; u = g.s
        lp = self._path(("projection", delay), r_lo=np.zeros(N), r_hi=u,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k] + delay), g.t[k] + delay - r),
                        known_fn=lambda k, r: (np.full_like(r, u[k]), u[k] - r))
        raw = Y
        if delay:
            raw = np.zeros_like(Y)
            for k in range(Y.shape[1]):
                if np.abs(Y[:, k]).max() > 0:
                    raw[:, k] = self.read(-delay, -delay) @ Y[:, k]
        return np.ascontiguousarray(self._many(lp, raw).transpose(1, 0, 2)).reshape(N, Y.shape[1] * N)

    # ------------------------------------------------------- closed loop
    row = row_op                                          # the engines' common name

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None, impulse_controls=(), own_frozen: bool = True):
        """maps[agent]: (n_ctrl, n_rows, N) nodal raw maps g(t, b).  Columns: Brownian channels,
        then one impulse column per control in impulse_controls (unit mass at the shock time).
        Returns Z (n_prim N, ncol).
        With a past: maps (n_ctrl, n_rows, N + Nt), the map on the strip (the band's nodes weigh the
        increments observed before zero) then the discrete weights on the initial shocks' point
        observations; the columns are the channels, the initial shocks, then the impulses.  The
        band's forcing is the past: the state at zero, the lagged atoms read before zero, the row's
        pre-zero increments under the map (past_conv_path) with the old noise loadings.
        With a continuation every map is the frozen stationary one on the buffer; the excluded agent's
        too (its strategy off on [0, T] only: the buffer's is part of its environment) unless
        own_frozen=False switches it off on the buffer as well (the envelope response of its FOC)."""
        if self.past is not None:
            return self._closed_loop_past(maps, excluded, impulse_controls, own_frozen)
        N = self.N; nW = self.nW
        imp = list(impulse_controls)
        n = len(self.prim) * N; ncol = nW + len(imp)
        M = np.zeros((n, n)); B = np.zeros((n, ncol))
        excl = set(next(a for a in self.model.agents if a.name == excluded).controls) if excluded else set()
        # states: map-independent, cached per (excluded, impulses) as the nonzero blocks
        if self.nX:
            blocks, B0 = self._state_part(excluded, excl, imp)
            for (i, p), blk in blocks.items():
                M[self.block(self.prim[i]), p * N:(p + 1) * N] = blk
            B[:] = B0
        # controls from maps
        for a in self.model.agents:
            if a.name == excluded:
                continue
            gm = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = self.block(u)
                for r, (rname, drift, E, delay) in enumerate(self.rows[a.name]):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gker = gm[ui, r]
                    C = self.conv_left(gker, delay)
                    for nm, op in blocks.items():                       # only the primaries the row reads
                        M[bl, self.block(nm)] += C @ op
                    for src, dl in deltas.items():
                        if src in self.channels:
                            col = self.channels.index(src)
                        elif src in imp:
                            col = nW + imp.index(src)
                        else:
                            continue
                        for (age, w) in dl:
                            B[bl, col] += w * (self.instant(age, delay) @ gker)
        return self._solve_causal(M, B)

    def noise_weight(self, agent: str, r: int, k: int, w: float) -> np.ndarray:
        """(N,): the weight of the row's own noise increment on channel k at every action node: the model's
        loading w on the new-shock region, the past's on the band (an increment observed before zero)."""
        g = self.g
        if g.L is None:
            return np.full(self.N, float(w))
        return np.where(g.upper, self.E_old[agent][r][k], float(w))

    def _closed_loop_past(self, maps, excluded, impulse_controls, own_frozen=True):
        N = self.N; nW = self.nW; ncol = self.ncol; g = self.g
        imp = list(impulse_controls)
        n = len(self.prim) * N; nc = ncol + len(imp)
        M = np.zeros((n, n)); B = np.zeros((n, nc))
        excl = set(next(a for a in self.model.agents if a.name == excluded).controls) if excluded else set()
        if self.nX:
            blocks, B0 = self._state_part(excluded, excl, imp)
            for (i, p), blk in blocks.items():
                M[self.block(self.prim[i]), p * N:(p + 1) * N] = blk
            B[:] = B0
        band = g.L is not None
        lower = ~g.upper
        for a in self.model.agents:
            off = a.name == excluded                        # the agent's own strategy off on [0, T]; on the buffer it is frozen
            if off and (self.cont is None or not own_frozen):
                continue
            gm = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = self.block(u)
                for r, (rname, drift, E, delay) in enumerate(self.rows[a.name]):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gker = self.with_frozen(a.name, ui, r, np.zeros(N) if off else gm[ui, r][:N])
                    C = self.conv_left(gker, delay)
                    for nm, op in blocks.items():
                        M[bl, self.block(nm)] += C @ op
                    if band:
                        B[bl, :nW] += C @ self.row_past(a.name, r)                                   # lagged atoms before zero
                        B[bl, :nW] += self.past_conv_path().bilinear(gker, self.past_row_kernel(a.name, r))   # increments before zero
                    for src, dl in deltas.items():
                        if src in self.channels:
                            col = self.channels.index(src)
                            for (age, w) in dl:
                                B[bl, col] += self.noise_weight(a.name, r, col, w) * (self.instant(age, delay) @ gker)
                        elif src in imp:
                            col = ncol + imp.index(src)
                            for (age, w) in dl:
                                B[bl, col] += w * lower * (self.instant(age, delay) @ gker)
                    if self.n_init and not off:
                        gd = gm[ui, r][N:]
                        for i in range(self.n_init):
                            e = self.init_rows[a.name][r, i]
                            if e:
                                B[bl, nW + i] += e * (self.disc_embed(delay) @ gd)
        return self._solve_causal(M, B)

    def _state_part(self, excluded, excl: set, imp: list):
        """The state rows of the closed-loop system without the maps: the nonzero N x N blocks of the
        Volterra propagation of the state inputs, {(state, primary): block}, and the shock and impulse
        columns B0 (n, ncol).  Map-independent, cached per (excluded agent, impulse controls)."""
        key = (excluded, tuple(imp))
        if key in self._state_parts:
            return self._state_parts[key]
        g = self.g; N = self.N; nW = self.nW; n = len(self.prim) * N
        if self.past is not None:
            return self._state_part_past(key, excl, imp)
        B0 = np.zeros((n, nW + len(imp)))
        EA = self.expA(g.a)                                             # (N, nX, nX)
        for k in range(nW):
            v = self.sigma[:, k]
            for i in range(self.nX):
                B0[self.block(self.prim[i]), k] += EA[:, i, :] @ v
        inp = np.zeros((self.nX, N, n))
        for si, (nm, lag), c in self.state_inputs:
            if nm in excl:
                continue
            inp[si] += c * self.atom_op((nm, lag))
        blocks: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(self.nX):
            for j in range(self.nX):
                for p in self._nonzero_blocks(inp[j]):
                    blk = self.Vol[i, j] @ inp[j][:, p * N:(p + 1) * N]
                    if (i, p) in blocks:
                        blocks[(i, p)] += blk
                    else:
                        blocks[(i, p)] = blk
        for col, u in enumerate(imp):
            for si, (nm, lag), c in self.state_inputs:
                if nm != u:
                    continue
                v = np.zeros(self.nX); v[si] = c
                EAd = self.expA(np.maximum(g.a - lag, 0.0)) if lag else EA
                on = (g.a0 >= lag - 1e-12) if lag else np.ones(N, dtype=bool)
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), nW + col] += on * (EAd[:, i, :] @ v)
        self._state_parts[key] = (blocks, B0)
        return blocks, B0

    def _state_part_past(self, key, excl: set, imp: list):
        """_state_part with a past: on the band the state at zero is the past's state kernel at age a - t
        propagated by e^{At}, and the lagged inputs read before zero (every control, the excluded one's
        pre-zero actions being history) are a known forcing through the Volterra operator; the initial
        shocks' columns start from their loads on the new-shock region; the impulse columns are zero on
        the band (a deviation before zero is sunk)."""
        g = self.g; N = self.N; nW = self.nW; n = len(self.prim) * N; ncol = self.ncol
        B0 = np.zeros((n, ncol + len(imp)))
        up = g.upper; lower = ~up
        EA = self.expA(g.a)                                             # (N, nX, nX)
        for k in range(nW):
            v = self.sigma[:, k]
            for i in range(self.nX):
                B0[self.block(self.prim[i]), k] += lower * (EA[:, i, :] @ v)
        if up.any():
            EAt = self.expA(g.t[up])                                     # (n_up, nX, nX)
            iu = np.where(up)[0]
            K0 = np.stack([self.past_at(self.prim[j], iu, g.a[iu] - g.t[iu]) for j in range(self.nX)], axis=1)   # (n_up, nX, nW)
            for i in range(self.nX):
                B0[self.block(self.prim[i]), :nW][up] += np.einsum("nj,njk->nk", EAt[:, i, :], K0)
        for col in range(self.n_init):
            v = self.init_sigma[:, col]
            for i in range(self.nX):
                B0[self.block(self.prim[i]), nW + col] += lower * (EA[:, i, :] @ v)
        inp = np.zeros((self.nX, N, n))
        for si, (nm, lag), c in self.state_inputs:
            if lag > 0 and g.L is not None:
                pr = self.past_read(nm, lag)
                if np.any(pr):
                    for i in range(self.nX):
                        B0[self.block(self.prim[i]), :nW] += c * (self.Vol[i, si] @ pr)
            if nm in excl and self.cont is None:
                continue                                    # with a continuation the excluded control acts on the buffer
            inp[si] += c * self.atom_op((nm, lag))
        blocks: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(self.nX):
            for j in range(self.nX):
                for p in self._nonzero_blocks(inp[j]):
                    blk = self.Vol[i, j] @ inp[j][:, p * N:(p + 1) * N]
                    if (i, p) in blocks:
                        blocks[(i, p)] += blk
                    else:
                        blocks[(i, p)] = blk
        for col, u in enumerate(imp):
            for si, (nm, lag), c in self.state_inputs:
                if nm != u:
                    continue
                v = np.zeros(self.nX); v[si] = c
                EAd = self.expA(np.maximum(g.a - lag, 0.0)) if lag else EA
                on = ((g.a0 >= lag - 1e-12) if lag else np.ones(N, dtype=bool)) & lower
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), ncol + col] += on * (EAd[:, i, :] @ v)
        self._state_parts[key] = (blocks, B0)
        return blocks, B0

    @cached_property
    def _panel_ranges(self):
        """Node ranges [lo, hi) of every time panel: pieces are stored panel by panel, so each panel's
        nodes are contiguous within every primary's block."""
        panel = self.panel_of_node
        return [(int(np.searchsorted(panel, p)), int(np.searchsorted(panel, p, side="right"))) for p in range(self.g.P)]

    def _solve_causal(self, M: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve (I - M) Z = B exploiting causality: a kernel value at time panel p depends only on
        values at panels <= p, so with nodes grouped by panel the system is block lower triangular
        and is solved by block forward substitution (one dense solve per panel).  The panel blocks are
        strided views of M reshaped by (primary, node), copied contiguously (the same blocks, in the
        same (primary, node) order, as gathering the panel's indices)."""
        n_prim = len(self.prim); N = self.N; ncol = B.shape[1]
        M4 = M.reshape(n_prim, N, n_prim, N); B3 = B.reshape(n_prim, N, ncol)
        Z = np.zeros_like(B); Z3 = Z.reshape(n_prim, N, ncol)
        ranges = self._panel_ranges
        for p, (lo, hi) in enumerate(ranges):
            np_ = n_prim * (hi - lo)
            rhs = B3[:, lo:hi].reshape(np_, ncol).copy()
            for q in range(p):
                lq, hq = ranges[q]; nq_ = n_prim * (hq - lq)
                blk = M4[:, lo:hi, :, lq:hq].reshape(np_, nq_)
                rhs -= (-blk) @ Z3[:, lq:hq].reshape(nq_, ncol)   # (I - M) has -M off the diagonal blocks
            diag = M4[:, lo:hi, :, lo:hi].reshape(np_, np_)
            Z3[:, lo:hi] = np.linalg.solve(np.eye(np_) - diag, rhs).reshape(n_prim, hi - lo, ncol)
        return Z


class SpectralFiniteSolver(EngineBase):
    RESULT = TriangleResult
    TOL, DAMPING, MAX_NEWTON = 1e-8, 0.5, 8
    MAP_RIDGE = tunable("map_ridge")      # ridge of the per-time-row map projection, relative to the row's own Gram (settings)

    def __init__(self, model: Model, verbose: bool = False, settings=None, past=None, continuation=None):
        """The triangle grid's compiled model and the map shapes (nU, nR, N).  settings: the tuning constants
        (noisestate.Settings, or a dict of its fields; the defaults when None).  past: the known past of a
        transition (a Past, a StationaryResult, a stationary Model/dict/path solved on the fly, or a list
        of initial shocks; see past.py): the game then starts at time zero from that regime.  continuation:
        how it goes on after T: None or "end" (the game ends at T), a converged StationaryResult of this
        model at the past's window, or "stationary" (that result solved here, at horizon.nodes): every
        agent's map is then frozen at the stationary map on a buffer [T, T + L] after the horizon, the
        closed loop and the first-order conditions run to T + L, and res.settled measures how far the maps
        on [T - L, T] are from the stationary ones.  Both are recorded in solver_kw, so refine() and
        stability() rebuild them.  With initial shocks the maps are (nU, nR, N + Nt): after each row's map
        nodes, the discrete weights on the row's point observation of the shocks, on the time nodes."""
        hz = model.horizon
        if hz.kind == "transition":                    # the file's blocks, each overridden by its keyword
            if past is None:
                past = Past.from_block(hz.past)
            if continuation is None:
                continuation = hz.continuation or "stationary"
        past = Past.of(past) if past is not None else None
        continuation = self._continuation_of(model, past, continuation, hz.stationary if hz.kind == "transition" else None)
        opts = {k: v for k, v in (("past", past), ("continuation", continuation)) if v is not None}
        super().__init__(model, verbose, settings=settings, **opts)
        self.c = SpectralCompiled(model, past=past, continuation=continuation)
        if past is not None:
            self.RESULT = TransitionResult
        self.Nm = self.c.N + (self.c.Nt if self.c.n_init else 0)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.Nm) for a in model.agents}
        self._rep_parts: Dict[str, Dict[str, float]] = {}      # agent -> where the representation error sits (with a past)

    @staticmethod
    def _continuation_of(model: Model, past, continuation, stationary: Optional[dict] = None):
        """None / "end" -> None; "stationary" -> this model's stationary equilibrium at the past's window, solved
        here (horizon.nodes per panel, the finite horizon's breakpoints, initial values and transition blocks
        dropped; `stationary` = {"window", "nodes"}, a transition file's sizing block, overrides the nodes and
        must agree with the past's window); a StationaryResult -> itself (checked by the compile)."""
        if continuation is None or continuation == "end":
            return None
        if isinstance(continuation, str):
            if continuation != "stationary":
                raise ValueError(f"continuation must be 'stationary', 'end' or a StationaryResult, not {continuation!r}")
            if past is None or not past.window > 0:
                raise ValueError("continuation='stationary' needs a past with a window (its window is the continuation's)")
            from . import solve
            d = model.to_dict()
            for s in d["states"].values():
                s.pop("initial", None)
            hz = d.setdefault("horizon", {}); hz.update(kind="stationary", window=float(past.window)); hz.pop("breakpoints", None)
            for k in ("past", "continuation", "stationary"):
                hz.pop(k, None)
            if stationary:
                if stationary.get("window") is not None and abs(float(stationary["window"]) - past.window) > 1e-9 * max(1.0, past.window):
                    raise ValueError(f"horizon.stationary.window ({stationary['window']:g}) must equal the past's window ({past.window:g}): "
                                     "the buffer after T is one window of the past, on which the stationary maps are read at the node's age")
                if stationary.get("nodes") is not None:
                    hz["nodes"] = int(stationary["nodes"])
            return solve(Model.from_dict(d)).check()
        return continuation

    def stationary_start(self) -> Dict[str, np.ndarray]:
        """The raw maps a solve with start="stationary" begins from: the continuation's stationary maps at every
        node's age (the frozen maps of the buffer, on the whole strip), zero weights on the initial shocks."""
        if self.c.cont is None:
            raise ValueError("start='stationary' needs a stationary continuation (continuation='stationary' or a StationaryResult): "
                             "the start is its maps read at every node's age")
        out = {}
        for a in self.model.agents:
            gm = np.zeros(self.shapes[a.name])
            gm[:, :, :self.c.N] = self.c.frozen[a.name]
            out[a.name] = gm
        return out

    @property
    def action_shapes(self) -> Dict[str, Tuple[int, int, int]]:
        """Action kernels: (n_controls, N, ncol) per agent, the columns the channels then the initial shocks."""
        return {a.name: (len(a.controls), self.c.N, self.c.ncol) for a in self.model.agents}

    def _finish(self, res) -> None:
        res.past = self.c.past
        res.continuation = self.c.cont
        super()._finish(res)
        if self.c.cont is not None:
            for a in self.model.agents:
                res.cost_parts[a.name]["continuation"] = self.continuation_cost(a, res.Z)
            res.settled = max(self.settled(res.maps), self.settled_means(res.means))
        if self.c.past is not None:
            res.times = self.c.tm.copy()
            for a in self.model.agents:
                res.loss_path[a.name] = self.loss_path(a, res.Z, res.means)
                if self.c.cont is not None:
                    res.excess_costs[a.name] = float(self.c.time_mass(self.c.rho) @ (res.loss_path[a.name] - self.c.cont.costs[a.name]))

    # ------------------------------------------------ the transition's paths
    def _zeta(self, agent: Agent, Z: np.ndarray) -> np.ndarray:
        """(m, N, ncol) the loss atoms' kernels in the world Z, the band's pre-zero part of a lagged atom included."""
        c = self.c; atoms = c.loss[agent.name][0]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])
        if c.past is not None and c.g.L is not None:
            zeta[:, :, :c.nW] += c.zeta_past(agent.name)
        return zeta

    def _row_variance(self, err: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """(Nt,) per time node the integral over shock age of quad(err(t, a)) summed over the channels, err (m, N, ncol):
        Gauss quadrature of the interpolated kernels on every piece crossed (row_quadrature), plus the initial shocks'
        columns on the line s = 0 (their own weight is the point).  quad(z) takes (m, nq, k) and returns (nq,)."""
        c = self.c; g = c.g
        out = np.zeros(c.Nt)
        for i, (t, side) in enumerate(zip(c.tm, c.tm_side)):
            I, w = g.row_quadrature(t, side)
            if len(w):
                out[i] = w @ quad(np.einsum("qn,mnk->mqk", I, err[:, :, :c.nW]))
            if c.n_init and i < c.Nd:
                out[i] += quad(err[:, None, c.diag[i], c.nW:])[0]
        return out

    def loss_path(self, agent: Agent, Z: np.ndarray, means=None) -> np.ndarray:
        """(Nt,) E[loss(t)] of the agent at every time node (res.times: [0, T] and the buffer): the variance part
        1/2 sum_ij Q_ij int zeta_i zeta_j da over every shock alive at t by row quadrature, plus the mean part
        1/2 zbar'Q zbar + q'zbar when the means are driven.  Its discounted integral over [0, T] (time_mass) is
        res.costs to quadrature accuracy."""
        c = self.c; atoms, Q, q = c.loss[agent.name]
        zeta = self._zeta(agent, Z)
        out = self._row_variance(zeta, lambda z: 0.5 * np.einsum("iqk,ij,jqk->q", z, Q, z))
        if means and any(np.any(means[n]) for n in c.prim):
            zbar = np.concatenate([np.asarray(means[n], dtype=float) for n in c.prim])
            zb = self._mean_atoms(zbar, atoms)
            out += 0.5 * np.einsum("it,ij,jt->t", zb, Q, zb) + q @ zb
        return out

    def belief_error(self, agent: Agent, name: str, Z: np.ndarray) -> np.ndarray:
        """(Nt,) the variance of the agent's estimation error of the quantity `name` at every time node: the
        quantity's kernel minus its projection on the agent's seen rows (the closed-loop rows of Z, its own
        controls on, the increments observed before zero and the initial shocks' point observations included:
        one weighted least-squares Gram per time row, maps_from_world), integrated over the shocks."""
        c = self.c
        K = Z[c.block(name)] if name in c.index else c.expr_op(self.model.expand({name: 1.0})) @ Z
        rows, inst = self._seen_rows(agent, Z, set())
        Bk = self._row_operator(agent, rows, inst)
        gm = self._maps_from_world_past(agent, Bk, K[None]) if c.past is not None else self.maps_from_world(agent, Z, K[None])
        recon = np.stack([Bk[k] @ gm[0].reshape(-1) for k in range(Bk.shape[0])], axis=1)
        return self._row_variance((K - recon)[None], lambda z: np.einsum("iqk,iqk->q", z, z))

    def warm_maps_from(self, prev) -> Dict[str, np.ndarray]:
        """Raw maps to start from, given a result of this engine on another grid of the same model (a sweep over the
        horizon T): the previous maps read at this grid's nodes where they exist, the continuation's frozen
        stationary maps beyond the previous domain (what a settled transition has there), zero without one."""
        c = self.c; g, gc = c.g, prev.compiled.g
        I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_d)
        outside = ~np.asarray(I != 0).any(axis=1)
        out = {}
        for a in self.model.agents:
            gm = np.zeros(self.shapes[a.name])
            gm[:, :, :c.N] = np.einsum("fn,urn->urf", I, prev.maps[a.name][:, :, :gc.N])
            if c.cont is not None:
                gm[:, :, :c.N][:, :, outside] = c.frozen[a.name][:, :, outside]
            out[a.name] = gm
        return out

    # ------------------------------------------------ best-response pieces
    def _identified(self, agent: Agent) -> np.ndarray:
        """The map on a row observed with delay d is stored at the shifted time t' = t - d, so its nodes on
        the time panels above T - d belong to controls after the horizon and are read by nothing: those
        entries are removed from every solve and left at zero.  (The nodes are whole panels, so nothing is
        masked inside a piece; with a mask cutting through a piece the interpolant of the map between the
        kept nodes and the zeroed ones is meaningless, and the delayed rows were not exact.)"""
        c = self.c; g = c.g; N = c.N
        if c.past is None:
            keep = np.ones(len(agent.signals) * N, dtype=bool)
            for r in range(len(agent.signals)):
                d = c.rows[agent.name][r][3]
                if d > 0:
                    keep[r * N:(r + 1) * N] = c.panel_of_node + c.panel_shift(d) < g.P
            return keep
        # with a past: the map is in raw age and the pieces below a row's delay read nothing (whole pieces: the
        # delay is a breakpoint), the band's map nodes of a row whose old regime carried nothing are masked
        # (nothing to read), the buffer's are frozen, and a row's discrete weights are kept where it sees an
        # initial shock, from the delay on
        Nm = self.Nm; nR = len(agent.signals)
        keep = np.zeros(nR * Nm, dtype=bool)
        eps = 1e-9 * max(1.0, c.Tg)
        for r in range(nR):
            d = c.row_delays[agent.name][r]
            flow = (g.a1 > d + eps) if d > 0 else np.ones(N, dtype=bool)
            if g.L is not None:
                empty = not np.any(c.past_row_kernel(agent.name, r)) and not np.any(c.E_old[agent.name][r])
                if empty:
                    flow = flow & ~g.upper
            keep[r * Nm:r * Nm + N] = flow & ~c.buffer                     # the buffer's map is frozen, not solved
            if c.n_init and np.any(c.init_rows[agent.name][r]):
                tpanel = np.repeat(np.arange(g.P), g.nt)
                keep[r * Nm + N:(r + 1) * Nm] = (tpanel < c.P_T) & (g.bp[tpanel] >= d - eps)
        return keep

    def _response_operators(self, agent: Agent, R: np.ndarray):
        """The base's, plus with a continuation the agent's own frozen reaction on the buffer: the own block of a
        control's response is the identity (the action itself) and the frozen map's response to it."""
        out = super()._response_operators(agent, R)
        c = self.c
        if c.cont is not None:
            for ui, u in enumerate(agent.controls):
                out[ui][c.block(u)] += c.response_op(R[c.block(u), ui])
        return out

    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        """The system on the kept unknowns, solved directly; a singular one raises (see _solve_regular).
        With a past the nodes of a Duffy triangle's degenerate corner row (one point, na nodes) are one
        unknown (see _maps_from_world_past): the system is restricted to that space (the group's columns
        summed, its equations summed), so the map's interpolant is single-valued at the corner."""
        keep = np.tile(self._identified(agent), len(agent.controls))
        gamma = np.zeros(Amat.shape[0])
        if self.c.past is None:
            gamma[keep] = self._solve_regular(agent, Amat[np.ix_(keep, keep)], -bvec[keep])
            return gamma
        Rm = self._corner_ties(agent)[keep]
        Rm = Rm[:, np.any(Rm, axis=0)]
        A = Rm.T @ Amat[np.ix_(keep, keep)] @ Rm
        gamma[keep] = Rm @ self._solve_regular(agent, A, -(Rm.T @ bvec[keep]))
        return gamma

    def _corner_ties(self, agent: Agent) -> np.ndarray:
        """(nG, nG') restriction of the FOC unknowns (control, row, node) to one value per degenerate corner row
        of every Duffy triangle (the lower one's row at t = t_p, the upper one's at t = t_{p+1}); every other
        unknown is its own column.  Cached per agent shape."""
        key = ("ties", len(agent.controls), len(agent.signals))
        if key not in self.c._disc:
            c = self.c; g = c.g; Nm = self.Nm
            nU, nR = len(agent.controls), len(agent.signals)
            group = -np.ones(Nm, dtype=int)                                     # node -> corner group id (per row block)
            ng = 0
            for pc in g.pieces:
                if pc.triangle:
                    nodes = pc.offset + (0 if not pc.upper else (pc.nt - 1) * pc.na) + np.arange(pc.na)
                    group[nodes] = ng; ng += 1
            nG = nU * nR * Nm
            cols = np.zeros(nG, dtype=int); col = 0
            for ui in range(nU):
                for r in range(nR):
                    base = (ui * nR + r) * Nm
                    seen = {}
                    for n in range(Nm):
                        if group[n] < 0:
                            cols[base + n] = col; col += 1
                        elif group[n] in seen:
                            cols[base + n] = seen[group[n]]
                        else:
                            seen[group[n]] = col; cols[base + n] = col; col += 1
            Rm = np.zeros((nG, col)); Rm[np.arange(nG), cols] = 1.0
            self.c._disc[key] = Rm
        return self.c._disc[key]

    def _project(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps (nU, nR, N) reproducing the action kernels cact (nU, N, nW) on the closed-loop rows of
        Zfull: maps_from_world, one weighted least-squares solve per time row."""
        return self.maps_from_world(agent, Zfull, cact)

    def maps_from_world(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps of `agent` reproducing its action kernels cact (nU, N, nW) given the closed-loop
        primary kernels Zfull: one weighted least-squares projection per time row."""
        c = self.c; g = c.g; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        if c.past is not None:
            return self._maps_from_world_past(agent, Bk, cact)
        gmap = np.zeros((nU, nR, N))
        shifts = [c.panel_shift(c.rows[agent.name][r][3]) if c.rows[agent.name][r][3] > 0 else 0 for r in range(nR)]
        systems: Dict[int, list] = {}                     # size -> [(cols, G, [rhs per control])]: solved in one call per size
        for (p, it), idx in sorted(c.trow_by_pit.items()):
            tv = g.t[idx[0]]
            w = g.row_weights(tv, side=(-1 if tv >= g.bp[p + 1] - 1e-12 else +1))[idx]
            # the action at this time row reads each row's map at the row shifted by the observation delay
            parts = [r * N + c.trow_by_pit[(p - shifts[r], it)] for r in range(nR) if p - shifts[r] >= 0]
            if not parts:
                continue
            cols = np.concatenate(parts)
            Bsub = Bk[:, idx][:, :, cols]
            G = sum((Bsub[k] * w[:, None]).T @ Bsub[k] for k in range(nW))
            if np.trace(G) <= 0:
                continue
            G = G + self.MAP_RIDGE * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            rhs = [sum((Bsub[k] * w[:, None]).T @ cact[ui, idx, k] for k in range(nW)) for ui in range(nU)]
            systems.setdefault(len(cols), []).append((cols, G, rhs))
        for size, items in systems.items():
            Gs = np.stack([G for _, G, _ in items])
            for ui in range(nU):
                sol = np.linalg.solve(Gs, np.stack([rhs[ui] for _, _, rhs in items])[:, :, None])   # one LAPACK solve per system, as before
                for (cols, _, _), x in zip(items, sol[:, :, 0]):
                    gmap[ui].reshape(-1)[cols] = x
        return gmap

    def _maps_from_world_past(self, agent: Agent, Bk: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """maps_from_world with a past: the Gram of a time row runs over the ages [0, L] of both shock families
        (the band's action nodes read the map's band through the past's row kernel) plus, for each initial
        shock, its point at the row's node on the line s = 0; the unknowns of the time row are the map nodes
        at the shifted time and, for a row seeing an initial shock, its discrete weight there.
        The degenerate corner of a Duffy triangle (the lower one's row at t = t_p, the upper one's at
        t = t_{p+1}, every theta node at one point) has no quadrature weight; the game starting at rest leaves it at zero, which is
        exact there (nothing has been seen at t = 0), but with a past the control reacts at once and the
        corner is the map's value at the oldest increment: the corner's nodes are tied to one unknown and
        the corner action node is a point condition (the instantaneous entry identifies it)."""
        c = self.c; g = c.g; N, nW = c.N, c.nW; Nm = self.Nm
        nR, nU = len(agent.signals), len(agent.controls)
        keep = self._identified(agent)
        gmap = np.zeros((nU, nR, Nm))
        shifts = [c.panel_shift(c.rows[agent.name][r][3]) if c.rows[agent.name][r][3] > 0 else 0 for r in range(nR)]
        diag_of = {int(node): j for j, node in enumerate(c.diag)}
        corner_of = {}                                                          # node -> its triangle, on the degenerate row
        for pc in g.pieces:
            if pc.triangle:                                                     # lower: the row at t_p; upper: the row at t_{p+1}
                for node in pc.offset + (0 if not pc.upper else (pc.nt - 1) * pc.na) + np.arange(pc.na):
                    corner_of[int(node)] = (pc.p, pc.q, pc.upper)
        for (p, it), idx in sorted(c.trow_by_pit.items()):
            if p >= c.P_T:
                continue                                                        # the buffer's rows are frozen
            tv = g.t[idx[0]]
            w = g.row_weights(tv, side=(-1 if tv >= g.bp[p + 1] - 1e-12 else +1))[idx]
            parts = []
            for r in range(nR):
                if p - shifts[r] < 0:
                    continue
                parts.append(r * Nm + c.trow_by_pit[(p - shifts[r], it)])
                if c.n_init:
                    parts.append(np.array([r * Nm + N + (p - shifts[r]) * g.nt + it]))
            if not parts:
                continue
            cols = np.concatenate(parts)
            cols = cols[keep[cols]]
            if cols.size == 0:
                continue
            # reduce the corner groups (row r, triangle p) to one unknown each
            groups = {}
            for j, col in enumerate(cols):
                r, node = divmod(int(col), Nm)
                if node < N and node in corner_of:
                    groups.setdefault((r, corner_of[node]), []).append(j)
            single = [j for j in range(len(cols)) if not any(j in js for js in groups.values())]
            Rm = np.zeros((len(cols), len(single) + len(groups)))
            for i, j in enumerate(single):
                Rm[j, i] = 1.0
            for i, js in enumerate(groups.values()):
                Rm[js, len(single) + i] = 1.0
            Bsub = Bk[:, idx][:, :, cols] @ Rm
            G = sum((Bsub[k] * w[:, None]).T @ Bsub[k] for k in range(nW))
            rhs = [sum((Bsub[k] * w[:, None]).T @ cact[ui, idx, k] for k in range(nW)) for ui in range(nU)]
            corners = {}                                                            # the time row's degenerate corners, one per triangle
            for j, node in enumerate(idx):
                if int(node) in corner_of:
                    corners.setdefault(corner_of[int(node)], j)
            if groups:
                for jc in corners.values():                                         # a point condition at each corner's action node
                    for k in range(nW):
                        G = G + np.outer(Bsub[k][jc], Bsub[k][jc])
                        for ui in range(nU):
                            rhs[ui] = rhs[ui] + Bsub[k][jc] * cact[ui, idx[jc], k]
            jd = [j for j, node in enumerate(idx) if int(node) in diag_of]          # the time row's node on s = 0, if any
            if c.n_init and jd:
                jd = jd[0]
                for i in range(c.n_init):
                    G = G + np.outer(Bsub[nW + i][jd], Bsub[nW + i][jd])
                    for ui in range(nU):
                        rhs[ui] = rhs[ui] + Bsub[nW + i][jd] * cact[ui, idx[jd], nW + i]
            if np.trace(G) <= 0:
                continue
            G = G + self.MAP_RIDGE * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            for ui in range(nU):
                gmap[ui].reshape(-1)[cols] = Rm @ np.linalg.solve(G, rhs[ui])
        if c.frozen is not None:
            gmap[:, :, :N][:, :, c.buffer] = c.frozen[agent.name][:, :, c.buffer]
        return gmap

    def world_from_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Closed-loop primary kernels when every agent's action kernels are given."""
        c = self.c; N, nW = c.N, c.nW
        n = len(c.prim) * N
        if c.past is not None:
            Z = np.zeros((n, c.ncol))
            for a in self.model.agents:
                for ui, u in enumerate(a.controls):
                    Z[c.block(u)] = actions[a.name][ui]
            if c.nX:
                blocks, B0 = c._state_part(None, set(), [])
                rhs = B0[:c.nX * N].copy(); Lx = np.zeros((c.nX * N, c.nX * N))
                for (i, p), blk in blocks.items():
                    if p < c.nX:
                        Lx[i * N:(i + 1) * N, p * N:(p + 1) * N] += blk
                    else:
                        rhs[i * N:(i + 1) * N] += blk @ Z[p * N:(p + 1) * N]
                Z[:c.nX * N] = np.linalg.solve(np.eye(c.nX * N) - Lx, rhs) if Lx.any() else rhs
            return Z
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
        """Every agent's raw maps (nU, nR, N) reproducing its action kernels (nU, N, nW) in the world those
        actions generate (the states from the Volterra propagation, then one projection per agent)."""
        Z = self.world_from_actions(actions)
        return {a.name: self.maps_from_world(a, Z, actions[a.name]) for a in self.model.agents}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """The variance part of the agent's discounted cost over [0, T] in the world Z (n_prim N, nW):
        1/2 sum Q_ij <zeta_i, zeta_j> under the discounted mass matrix of the triangle."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])                  # (m, N, nW)
        if c.past is not None:
            # the pre-zero part of the lagged atoms on the band, then the initial shocks along the line s = 0
            zeta[:, :, :c.nW] += c.zeta_past(agent.name)
            G = np.einsum("ink,nm,jmk->ij", zeta[:, :, :c.nW], c.cost_mass(), zeta[:, :, :c.nW])
            if c.n_init:
                w = c.time_mass(c.rho)[:c.Nd]
                zd = zeta[:, c.diag, c.nW:]                                   # (m, Nd, n_init)
                G = G + np.einsum("itk,t,jtk->ij", zd, w, zd)
            return float(0.5 * np.sum(Q * G))
        G = np.einsum("ink,nm,jmk->ij", zeta, c.cost_mass(), zeta)
        return float(0.5 * np.sum(Q * G))

    def continuation_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """The variance part of the agent's discounted cost over the buffer [T, T + L] under the frozen stationary
        maps (the shocks of the channels; the band and the initial shocks are gone by T >= L): reported in
        res.cost_parts[agent]["continuation"], not added to res.costs."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z[:, :c.nW] for at in atoms])
        G = np.einsum("ink,nm,jmk->ij", zeta, c.buffer_mass(), zeta)
        return float(0.5 * np.sum(Q * G))

    def settled(self, maps: Dict[str, np.ndarray]) -> float:
        """How far the maps on [T - L, T] are from the continuation's stationary maps: the largest difference on
        those nodes over agents, controls and rows, relative to the stationary map's peak."""
        c = self.c; g = c.g
        sel = ~g.upper & ~c.buffer & (g.t >= c.T - g.L - 1e-9)
        worst = 0.0
        for a in self.model.agents:
            fr = c.frozen[a.name]; gm = maps[a.name][:, :, :c.N]
            dev = np.abs(gm - fr).max(axis=(0, 1))
            worst = max(worst, float(dev[sel].max(initial=0.0) / max(1e-300, np.abs(fr).max())))
        return worst

    def settled_means(self, means) -> float:
        """How far the mean paths at T (the last node before the buffer) are from the continuation's stationary means,
        relative to the largest of those; zero when nothing drives the means (the paths are exactly zero)."""
        c = self.c
        if c.cont is None or not means or not any(np.any(means[n]) for n in c.prim):
            return 0.0
        iT = c.P_T * c.g.nt - 1
        scale = max(1e-300, max(abs(float(c.cont.means.get(n, 0.0))) for n in c.prim), max(float(np.abs(means[n]).max()) for n in c.prim))
        return max(abs(float(means[n][iT]) - float(c.cont.means.get(n, 0.0))) for n in c.prim) / scale

    def interpolate_maps(self, coarse) -> Dict[str, np.ndarray]:
        """The coarse result's raw maps read at this triangle's nodes from each node's side of its piece."""
        c = self.c; g, gc = c.g, coarse.compiled.g
        if c.past is not None:
            I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_d)
            out = {}
            for a in self.model.agents:
                gm = np.zeros(self.shapes[a.name])
                gm[:, :, :c.N] = np.einsum("fn,urn->urf", I, coarse.maps[a.name][:, :, :gc.N])
                if c.n_init:
                    It = gc.interp(c.tm, np.zeros(c.Nt), side_t=np.where(np.abs(c.tm - np.repeat(g.bp[1:g.P + 1], g.nt)) < 1e-12, -1, 1)) @ coarse.compiled.mean_embed
                    gm[:, :, c.N:] = np.einsum("fn,urn->urf", It, coarse.maps[a.name][:, :, gc.N:])
                out[a.name] = gm
            return out
        I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a)
        return {a.name: np.einsum("fn,urn->urf", I, coarse.maps[a.name]) for a in self.model.agents}

    # ------------------------------------------- best response with a past
    # Every piece below calls the base with no past and, with one, assembles the same objects over the
    # strip and the initial shocks: the seen rows gain their pre-zero part, the row operator the band's
    # read of the past's increments and the discrete weights, the projection the old-shock segments and
    # the point conditions, and the FOC system is assembled densely (H_k (Fu Resp) G_k summed over the
    # columns of the world) with the FOC kernel's affine pre-zero part.
    # With a continuation the world after T is the closed loop under the frozen stationary maps, the
    # agent's own included: its passive world has its strategy off on [0, T] and frozen on the buffer,
    # and the response operators carry the frozen reaction (own block: the action plus that reaction),
    # so Zfull = Zpass + Resp c is the world the buffer's maps produce.  The first-order condition is
    # the infinite problem's: its continuation runs through the envelope responses (the agent's own
    # reaction off everywhere, R_off), which vanish beyond t + L <= T + L, so that a settled transition
    # solves the infinite problem exactly (the buffer's actions are optimal for it, not for a problem
    # truncated at T + L: with the buffer's reaction in the FOC instead, the same-model identity fails
    # by 1e-4 on [T - L, T], the buffer's own first-order conditions being cut at T + L).
    def _seen_rows(self, agent: Agent, Z: np.ndarray, excluded: set):
        rows, inst = super()._seen_rows(agent, Z, excluded)
        c = self.c
        if c.past is not None and c.g.L is not None:
            for r in range(len(rows)):
                rows[r][:, :c.nW] += c.row_past(agent.name, r)
        return rows, inst

    def _row_operator(self, agent: Agent, rows, inst):
        c = self.c
        if c.past is None:
            return super()._row_operator(agent, rows, inst)
        N, nW, ncol, Nm = c.N, c.nW, c.ncol, self.Nm; nR = len(rows)
        Gk = np.zeros((ncol, N, nR * Nm))
        sup, groups = self._row_support(agent, rows)
        for d, rs in groups.items():
            pairs = [(r, k) for r in rs for k in np.where(sup[r])[0]]
            if pairs:
                ops = c.conv_rows(np.stack([rows[r][:, k] for r, k in pairs], axis=1), d)
                for i, (r, k) in enumerate(pairs):
                    Gk[k, :, r * Nm:r * Nm + N] = ops[i]
        for r in range(nR):
            d = c.rows[agent.name][r][3]
            if c.g.L is not None:
                Kp = c.past_row_kernel(agent.name, r)
                for k in range(nW):
                    if np.any(Kp[:, k]):
                        Gk[k, :, r * Nm:r * Nm + N] += c.past_conv_path().with_known(Kp[:, k])
            for (k, age, w) in inst[r]:
                Gk[k, :, r * Nm:r * Nm + N] += c.noise_weight(agent.name, r, k, w)[:, None] * c.instant(age, d)
            if c.n_init:
                for i in range(c.n_init):
                    e = c.init_rows[agent.name][r, i]
                    if e:
                        Gk[nW + i, :, r * Nm + N:(r + 1) * Nm] += e * c.disc_embed(d)
        return Gk

    def _projection_operator(self, agent: Agent, rows, inst):
        c = self.c
        if c.past is None:
            return super()._projection_operator(agent, rows, inst)
        N, nW, ncol, Nm = c.N, c.nW, c.ncol, self.Nm; nR = len(rows)
        H = np.zeros((nR * Nm, ncol * N))
        for r in range(nR):
            d = c.rows[agent.name][r][3]
            flow = slice(r * Nm, r * Nm + N)
            H[flow, :nW * N] += c.projection_rows(rows[r][:, :nW], d)
            if c.g.L is not None:
                raw = rows[r][:, :nW] if not d else c.read(-d, -d) @ rows[r][:, :nW]
                Kp = c.past_row_kernel(agent.name, r)
                for k in range(nW):
                    if np.any(raw[:, k]):
                        H[flow, k * N:(k + 1) * N] += c.old_shock_proj_path().with_known(raw[:, k])
                    if np.any(Kp[:, k]):
                        H[flow, k * N:(k + 1) * N] += c.past_proj_path().with_known(Kp[:, k])
            for (k, age, w) in inst[r]:
                H[flow, k * N:(k + 1) * N] += (c.noise_weight(agent.name, r, k, w)[:, None] * c.instant(age, d)).T
            if c.n_init:
                Id = c.diag_read(d); St = c.shock_time_read()
                for i in range(c.n_init):
                    col = slice((nW + i) * N, (nW + i + 1) * N)
                    yi = St @ rows[r][:, nW + i]                              # the row's kernel on the shock at the increment's time
                    if np.any(yi):
                        H[flow, col] += yi[:, None] * Id
                    e = c.init_rows[agent.name][r, i]
                    if e:
                        H[r * Nm + N:(r + 1) * Nm, col] += e * c.disc_select(d)
        return H

    def _foc_affine(self, agent: Agent, Ms) -> Optional[list]:
        """Per control, the pre-zero part of the FOC kernel on the band, (N, nW): the lagged loss atoms read
        before zero through the same operators as the kernels (None when there is none)."""
        c = self.c
        if c.g.L is None:
            return None
        zp = c.zeta_past(agent.name)
        if not np.any(zp):
            return None
        atoms, Q, q = c.loss[agent.name]
        out = []
        for ui in range(len(agent.controls)):
            MQ = np.tensordot(Q.T, Ms[ui], axes=1)                             # MQ[i] = sum_j Q[j, i] M_j
            out.append(sum(MQ[i] @ zp[i] for i in range(len(atoms))))
        return out

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        c = self.c
        if c.past is None:
            return super().best_response(agent, maps, want_decomp)
        N, nW, ncol, Nm = c.N, c.nW, c.ncol, self.Nm
        nR, nU = len(agent.signals), len(agent.controls)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :ncol], Zp[:, ncol:]
        R = self._impulse_responses(agent, maps, R)
        Roff = R
        if c.cont is not None:                       # the envelope responses: the agent's own reaction off on the buffer too
            Roff = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls, own_frozen=False)[:, ncol:]
            Roff = self._impulse_responses(agent, maps, Roff)
        Zpass = self._passive_world(agent, maps, Zpass, R)
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(agent, ytil, yinst)
        Resp = self._response_operators(agent, R)
        Fu, Ms = self._foc_operators(agent, Roff, atoms=True)
        phi_past = self._foc_affine(agent, Ms)
        H = self._projection_operator(agent, ytil, yinst)
        nG = nU * nR * Nm
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        for ui in range(nU):
            phi = Fu[ui] @ Zpass                                                # (N, ncol)
            if phi_past is not None:
                phi[:, :nW] += phi_past[ui]
            rows_u = slice(ui * nR * Nm, (ui + 1) * nR * Nm)
            bvec[rows_u] = sum(H[:, k * N:(k + 1) * N] @ phi[:, k] for k in range(ncol))
            for vi in range(nU):
                FR = Fu[ui] @ Resp[vi]
                Amat[rows_u, vi * nR * Nm:(vi + 1) * nR * Nm] = sum(H[:, k * N:(k + 1) * N] @ (FR @ Gk[k]) for k in range(ncol))
        gamma = self._solve_foc(agent, Amat, bvec).reshape(nU, nR, Nm)
        cact = np.stack([(Gk @ gamma[ui].reshape(-1)).T for ui in range(nU)])
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp[ui] @ cact[ui]
        if c.cont is not None:                       # the action on the buffer (the frozen map's) is in the world, not in gamma
            cact = np.stack([Zfull[c.block(u)] for u in agent.controls])
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            self._decompose(agent, out, Fu, Resp, Gk, maps)
            if phi_past is not None:
                for ui, u in enumerate(agent.controls):
                    for part in ("foc", "physical"):
                        out["decomp"][u][part] = out["decomp"][u][part] + phi_past[ui]
        return self._project(agent, Zfull, cact), out

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        if self.c.past is None:
            return super()._representation_error(agent, Zfull, actions, g)
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        c = self.c; gr = c.g; worst = 0.0
        # where the error sits: the band's tip (the upper triangle collapsing to the corner (L, L), where the
        # map's pieces degenerate), the last window [T - L, T] (the end), or the interior, so that a resolution
        # problem can be told from the two geometric floors
        tip = gr.upper & (gr.t >= gr.bp[gr.PL - 1] - 1e-9) if gr.L is not None else np.zeros(c.N, dtype=bool)
        last = ~gr.upper & ~c.buffer & (gr.t >= c.T - gr.L - 1e-9) if gr.L is not None else np.zeros(c.N, dtype=bool)
        parts = {"interior": 0.0, "band tip": 0.0, "last window": 0.0}
        regions = [("interior", ~tip & ~last & ~c.buffer), ("band tip", tip), ("last window", last)]
        if c.cont is not None:
            parts["buffer"] = 0.0; regions.append(("buffer", c.buffer))
        for ui in range(len(agent.controls)):
            recon = np.stack([Bk[k] @ g[ui].reshape(-1) for k in range(Bk.shape[0])], axis=1)
            err = np.abs(recon - actions[ui])
            err[:, c.nW:] = 0.0
            err[c.diag, c.nW:] = np.abs(recon - actions[ui])[c.diag, c.nW:]     # an initial shock's column: on the line s = 0 only
            rel = err.max(axis=1) / max(1e-300, np.abs(actions[ui][:, :c.nW]).max(), np.abs(actions[ui][c.diag, c.nW:]).max(initial=0.0))
            worst = max(worst, float(rel.max()))
            for key, sel in regions:
                parts[key] = max(parts[key], float(rel[sel].max(initial=0.0)))
        self._rep_parts[agent.name] = parts
        return worst

    # ------------------------------------------------------------ means
    MEAN_RCOND = tunable("mean_rcond")          # a mean system whose reciprocal condition estimate is below this is singular (settings)

    def mean_system(self, maps: Dict[str, np.ndarray]):
        """The linear system M zbar = b of the mean paths on the time nodes (states then controls, Nt values
        each) under the strategies `maps`.  A path is carried as a kernel constant in shock age (mean_embed),
        on which the kernels' own operators restricted to the line s = 0 (the nodes `diag`: a kernel's
        response to a shock at time 0) are the path's: a lagged read is the path at t - lag, zero before 0,
        the Volterra propagation integrates from 0 and the continuation reads the path along s = 0.  A
        state's rows are its mean dynamics, xbar(t) = e^{At} x0 + int_0^t e^{A(t-r)} (inputs at their mean
        paths + const) dr.  A control's rows are its owner's mean first-order condition at every time node:
        the kernels' first-order condition (_foc_operators: the instantaneous derivative of 1/2 z'Qz + q'z in
        the control, the discounted own lagged reads, the continuation int_t^T e^{-rho (t'-t)} R(t', t) g(t')
        dt' through the passive-world impulse responses, the other agents answering through their
        equilibrium kernels, the agent's own control passive) applied to the mean paths with no information
        constraint (a deterministic path is common knowledge) and the targets q as the driver in place of
        the shocks.  Linear in (q, x0, const): one direct solve, no iteration.
        On a strip cut at age L below the horizon (a past's window shorter than T, or a stationary continuation,
        whose buffer follows T) the line s = 0 does not reach T, and the system is built on the time line instead
        (_mean_system_line): the dynamics by a one-dimensional Volterra operator, each control's condition on
        the line age = 0 (the birth of a shock at t, its continuation running to t + L through the buffer's
        frozen maps), the paths on the buffer frozen at the continuation's stationary means."""
        c = self.c
        if c.Nd < c.Nt:
            return self._mean_system_line(maps)
        return self._mean_system_diag(maps)

    def _mean_system_diag(self, maps: Dict[str, np.ndarray]):
        """mean_system on the line s = 0 (every time panel below the window): the kernels' operators restricted to
        the nodes `diag` applied to the embedded paths."""
        c = self.c; N, Nt, nP, nX = c.N, c.Nt, len(c.prim), c.nX
        E, diag = c.mean_embed, c.diag
        blk = lambda i: slice(i * Nt, (i + 1) * Nt)
        M = np.zeros((nP * Nt, nP * Nt)); b = np.zeros(nP * Nt); ones = np.ones(N)
        x0 = self._mean_start()
        if nX:
            blocks, B0 = c._state_part(None, set(), [])
            for i in range(nX):
                M[blk(i), blk(i)] = np.eye(Nt)
            for (i, p), B in blocks.items():
                M[blk(i), blk(p)] -= B[diag] @ E
            EA = c.expA(c.tm)
            for i in range(nX):
                b[blk(i)] = EA[:, i, :] @ x0 + sum((c.const[j] * (c.Vol[i, j][diag] @ ones) for j in range(nX) if c.const[j]), 0.0)
                for si, (nm, lag), coef in c.state_inputs:                  # lagged inputs read before zero: the old means
                    pre = self._mean_before(nm, lag)
                    if pre is not None:
                        b[blk(i)] += coef * (c.Vol[i, si][diag] @ (E @ pre))
        for a in self.model.agents:
            atoms, Q, q = c.loss[a.name]
            R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls)[:, c.ncol:]
            R = self._impulse_responses(a, maps, R)
            Fu, Ms = self._foc_operators(a, R, atoms=True)
            for ui, u in enumerate(a.controls):
                row = blk(c.index[u])
                Fd = Fu[ui][diag].reshape(Nt, nP, N)
                for p in range(nP):
                    M[row, blk(p)] = Fd[:, p, :] @ E
                b[row] = -sum((q[j] * (Ms[ui][j][diag] @ ones) for j in range(len(atoms)) if q[j]), 0.0)
                MQ = None
                for i, (nm, lag) in enumerate(atoms):                       # lagged atoms read before zero: the old means
                    pre = self._mean_before(nm, lag)
                    if pre is not None:
                        MQ = np.tensordot(Q.T, Ms[ui], axes=1) if MQ is None else MQ
                        b[row] -= (MQ[i][diag] @ (E @ pre))
        return M, b

    def _mean_system_line(self, maps: Dict[str, np.ndarray]):
        """mean_system on the time line (a strip cut at age L < T).  The state rows are xbar(t) = e^{At} x0 +
        int_0^t e^{A(t-r)} (inputs at their mean paths + const) dr through mean_volterra, a lagged input read
        before zero at the past's constant.  A control's rows are its mean first-order condition at every time
        node: the per-atom operators Ms of _foc_operators (the instantaneous derivative, the discounted own
        lagged reads, the continuation through the passive-world impulse responses to T + L) applied to the
        embedded mean of Q zeta + q and read on the line age = 0 (mean_line0), the mean of a lagged atom being
        the path at t - lag (mean_read; the strip's own read of a lagged kernel is zero below age lag, which is
        right for a shock and wrong for a path).  With a continuation the paths on the buffer's time nodes are
        its stationary means (frozen, like the maps: the closure assumes the transition has settled by T)."""
        c = self.c; Nt, nP, nX = c.Nt, len(c.prim), c.nX
        E, S0 = c.mean_embed, c.mean_line0
        blk = lambda i: slice(i * Nt, (i + 1) * Nt)
        M = np.zeros((nP * Nt, nP * Nt)); b = np.zeros(nP * Nt); ones = np.ones(Nt)
        x0 = self._mean_start()
        reads = {}
        def read(lag):
            if lag not in reads:
                reads[lag] = c.mean_read(lag)
            return reads[lag]
        if nX:
            V = c.mean_volterra; EA = c.expA(c.tm)
            for i in range(nX):
                M[blk(i), blk(i)] = np.eye(Nt)
                b[blk(i)] = EA[:, i, :] @ x0 + sum((c.const[j] * (V[i, j] @ ones) for j in range(nX) if c.const[j]), 0.0)
                for si, (nm, lag), coef in c.state_inputs:
                    M[blk(i), blk(c.index[nm])] -= coef * (V[i, si] @ read(lag))
                    pre = self._mean_before(nm, lag)
                    if pre is not None:
                        b[blk(i)] += coef * (V[i, si] @ pre)
        for a in self.model.agents:
            atoms, Q, q = c.loss[a.name]
            # the envelope responses (best_response): the agent's own reaction off on the buffer as well
            R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls, own_frozen=False)[:, c.ncol:]
            R = self._impulse_responses(a, maps, R)
            Fu, Ms = self._foc_operators(a, R, atoms=True)
            for ui, u in enumerate(a.controls):
                row = blk(c.index[u])
                for j in range(len(atoms)):
                    Oj = S0 @ Ms[ui][j] @ E                                     # the condition's read of (Q zeta + q)_j's mean
                    if q[j]:
                        b[row] -= q[j] * (Oj @ ones)
                    for i, (nm, lag) in enumerate(atoms):
                        if Q[j, i]:
                            M[row, blk(c.index[nm])] += Q[j, i] * (Oj @ read(lag))
                            pre = self._mean_before(nm, lag)
                            if pre is not None:
                                b[row] -= Q[j, i] * (Oj @ pre)
        if c.cont is not None:                                                  # the buffer: the new stationary means
            frozen = np.arange(Nt) >= c.P_T * c.g.nt
            for p, name in enumerate(c.prim):
                idx = np.flatnonzero(frozen) + p * Nt
                M[idx, :] = 0.0; M[idx, idx] = 1.0; b[idx] = float(c.cont.means.get(name, 0.0))
        return M, b

    def _mean_start(self) -> np.ndarray:
        """The mean state at time zero: the model's per-state `initial` where given (a given 0 overrides the past),
        else the past's constant mean of the state (zero without a past)."""
        c = self.c
        x0 = np.array(c.x0, dtype=float)
        if c.past is not None:
            for i, s in enumerate(self.model.states):
                if s.initial is None:
                    x0[i] = c.past.mean(s.name)
        return x0

    def _mean_before(self, name: str, lag: float) -> Optional[np.ndarray]:
        """(Nt,): the pre-zero value of `name` read `lag` earlier at every time node (the past's constant mean
        on the time nodes before the lag, zero after); None when nothing is read before zero."""
        c = self.c; g = c.g
        if lag <= 0 or c.past is None or not c.past.mean(name):
            return None
        tpanel = np.repeat(np.arange(g.P), g.nt)
        return np.where(g.bp[tpanel + 1] <= lag + 1e-12, c.past.mean(name), 0.0)

    def solve_means(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """The mean paths of the primaries (states then controls, Nt values each) under `maps`: exactly zero,
        with no solve, when nothing drives them (every q zero, no constant drift, no initial state); else
        the direct solve of mean_system, refusing a singular system."""
        c = self.c
        driven = c.x0.any() or c.const.any() or any(q.any() for atoms, Q, q in c.loss.values())
        if c.past is not None and any(c.past.mean(n) for n in c.prim):
            driven = True
        if c.cont is not None and any(c.cont.means.get(n, 0.0) for n in c.prim):
            driven = True
        if not driven:
            return np.zeros(len(c.prim) * c.Nt)
        M, b = self.mean_system(maps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            lu, piv = lu_factor(M, check_finite=False)
        gecon, = get_lapack_funcs(("gecon",), (lu,))
        rcond = float(gecon(lu, np.linalg.norm(M, 1))[0])
        if not rcond > self.MEAN_RCOND:
            raise ValueError(f"the mean system is singular (reciprocal condition estimate {rcond:.1e}): a control's mean "
                             "first-order condition is empty (no quadratic term in the control's current value); the kernels "
                             "do not depend on the means, so the model solves without its targets, constant drifts and initial states")
        return lu_solve((lu, piv), b, check_finite=False)

    def _mean_atoms(self, zbar: np.ndarray, atoms) -> np.ndarray:
        """The loss atoms' mean paths (m, Nt) from the primaries' paths: the kernels' atom operators on the embedded
        paths, read on the line s = 0 (a lagged atom is the path at t - lag, zero before 0)."""
        c = self.c; Nt = c.Nt
        if c.Nd < Nt:                                     # the time line (see _mean_system_line): the path at t - lag
            out = np.stack([c.mean_read(lag) @ zbar[c.index[nm] * Nt:(c.index[nm] + 1) * Nt] for (nm, lag) in atoms])
        else:
            Zm = np.concatenate([c.mean_embed @ zbar[p * Nt:(p + 1) * Nt] for p in range(len(c.prim))])
            out = np.stack([(c.atom_op(at) @ Zm)[c.diag] for at in atoms])
        for i, (nm, lag) in enumerate(atoms):
            pre = self._mean_before(nm, lag)
            if pre is not None:
                out[i] += pre[:out.shape[1]]
        return out

    def mean_cost(self, agent: Agent, zbar: np.ndarray) -> float:
        """The mean part of the agent's discounted cost, int_0^T e^{-rho t} (1/2 zbar'Q zbar + q'zbar) dt over its
        loss atoms at the mean paths `zbar` (the constant of a target, theta^2, is not in the model): spectral
        quadrature on the time panels."""
        atoms, Q, q = self.c.loss[agent.name]
        zeta = self._mean_atoms(zbar, atoms); w = self.c.time_mass(self.c.rho)[:zeta.shape[1]]
        return float(0.5 * np.einsum("it,ij,jt,t->", zeta, Q, zeta, w) + q @ (zeta @ w))

    def _mean_part(self, res) -> None:
        """res.means: the path on the time nodes res.means_t of every state, control and definition and of the
        mean drift rate of every signal row ("agent.row", at the time of the observation, its delay not applied);
        res.cost_parts and the mean part added to res.costs."""
        c = self.c; m = self.model; Nt = c.Nt
        zbar = self.solve_means(res.maps)
        blk = lambda nm: slice(c.index[nm] * Nt, (c.index[nm] + 1) * Nt)
        before = lambda nm, lag: (lambda pre: np.zeros(Nt) if pre is None else pre)(self._mean_before(nm, lag))
        value = lambda expr: sum((coef * (c.mean_read(lag) @ zbar[blk(nm)] + before(nm, lag)) for (nm, lag), coef in expr.items()), np.zeros(Nt))
        res.means = {name: zbar[blk(name)].copy() for name in c.prim}
        res.means.update({d.name: value(m.expand({d.name: 1.0})) for d in m.definitions})
        res.means.update({f"{a.name}.{r.name}": value(m.expand(r.drift)) for a in m.agents for r in a.signals})
        res.means_t = c.tm.copy()
        for a in m.agents:
            mean = self.mean_cost(a, zbar)
            res.cost_parts[a.name] = {"variance": res.costs[a.name], "mean": mean}
            res.costs[a.name] += mean

    def _diagnostics(self, res) -> None:
        """The first-order-condition decomposition, the second-order check and the representation error of
        every agent's best response at the equilibrium (the stationary engine's, on this grid)."""
        self._second_order_cache.clear()
        order = [a for a in self.model.agents if self.c.rep[a.name] == a.name] + [a for a in self.model.agents if self.c.rep[a.name] != a.name]
        for a in order:
            g, out = self.best_response(a, res.maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            if out["second_order"] is not None:
                res.second_order[a.name] = out["second_order"]
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)
            if a.name in self._rep_parts:
                res.representation_parts[a.name] = dict(self._rep_parts[a.name])
        self._loss_forms.clear()                  # the second-order check is done: its (n_prim N)^2 form is not kept

