"""Finite-horizon equilibrium on the piecewise-spectral triangle: SpectralFiniteSolver, the engine.

The construction is the stationary engine's: closed loop as one linear system in the nodal kernels
(closed_loop.py, through the compiled model of spectral_compiled.py); best response in the agent's
passive world with the per-date first-order condition (instantaneous term, discounted continuation
through the impulse responses, delayed reads) affine in the map on the passive rows, solved on the
operators of spectral_operators.py by finite_free.py (factored within settings.foc_dense_max, GMRES
beyond); raw map by projection, one Gram per time row (maps_from_world, here); Anderson fixed point over
the action kernels (or the raw maps), Newton-Krylov polish (engine.py).  The means are the SpectralMeans
mixin of spectral_means.py.  This module holds the solver itself: the options (a past, a continuation),
the shapes, the transition's paths (loss_path, belief_error), the identified unknowns and the corner
ties, the projection, the costs, the settled measures, the warm starts and the diagnostics hooks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import finite_free
from .closed_loop import ClosedLoopRows
from .engine import EngineBase
from .past import Past
from .results import TriangleResult, TransitionResult
from ._settings import tunable
from .spec import Agent, Model
from .spectral_compiled import SpectralCompiled
from .spectral_means import SpectralMeans

__all__ = ["SpectralCompiled", "ClosedLoopRows", "SpectralFiniteSolver"]


class SpectralFiniteSolver(SpectralMeans, EngineBase):
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
        # beyond settings.foc_dense_max unknowns nU nR N (the largest agent's) the FOC system is solved by GMRES instead of
        # being assembled and factored (finite_free.FocSystem)
        self.foc_free = max(len(a.controls) * len(a.signals) for a in model.agents) * self.Nm > self.settings.foc_dense_max
        self._last_gamma: Dict[str, np.ndarray] = {}         # agent -> its last FOC solution (the warm start of the Krylov solve)
        self._krylov_log: List[Tuple[str, int, float]] = []  # (agent, GMRES iterations, relative residual) of every Krylov solve
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.Nm) for a in model.agents}
        self._rep_parts: Dict[str, Dict[str, float]] = {}      # agent -> where the representation error sits (with a past)
        self._fixed_disc: Dict[str, np.ndarray] = {}           # agent -> the fixed discrete weights (freeze_before)
        self._fixed_actions: Dict[str, np.ndarray] = {}        # agent -> its action kernels (nU, N, ncol) on the fixed panels, zero elsewhere

    @staticmethod
    def _continuation_of(model: Model, past, continuation, stationary: Optional[dict] = None):
        """None / "end" -> None; "stationary" -> this model's stationary equilibrium at the past's window, solved
        here (numerics.nodes per panel, or numerics.continuation_nodes when given; the finite horizon's
        breakpoints, initial values and transition blocks dropped; `stationary` = {"window", "nodes"}, the
        horizon's sizing block, must agree with the past's window); a StationaryResult -> itself (checked by
        the compile)."""
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
            #  the continuation's length is the PAST'S WINDOW -- its lag-truncation L, never the
            #  transition's extent, which is T.  The terminal time goes with the other transition
            #  blocks: a stationary horizon has none.
            hz = d.setdefault("horizon", {}); hz.update(kind="stationary", window=float(past.window))
            for k in ("past", "continuation", "stationary", "T"):
                hz.pop(k, None)
            nm = d.setdefault("numerics", {})
            for k in ("breakpoints", "engine", "continuation_nodes"):
                nm.pop(k, None)
            if stationary:
                if stationary.get("window") is not None and abs(float(stationary["window"]) - past.window) > 1e-9 * max(1.0, past.window):
                    raise ValueError(f"horizon.stationary.window ({stationary['window']:g}) must equal the past's window ({past.window:g}): "
                                     "the buffer after T is one window of the past, on which the stationary maps are read at the node's age")
                if stationary.get("nodes") is not None:
                    nm["nodes"] = int(stationary["nodes"])
            return solve(Model.from_dict(d)).require_converged()
        return continuation

    def stationary_start(self) -> Dict[str, np.ndarray]:
        """The raw maps a solve with start_policy="stationary" begins from: the continuation's stationary maps at every
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
                res.cost_parts[a.name]["continuation"] = self.continuation_cost(a, res.world)
            res.settled = max(self.settled(res.maps), self.settled_means(res.means))
        if self.c.past is not None:
            res.times = self.c.tm.copy()
            for a in self.model.agents:
                res.loss_path[a.name] = self.loss_path(a, res.world, res.means)
                if self.c.cont is not None:
                    res.excess_costs[a.name] = float(self.c.time_mass(self.c.rho) @ (res.loss_path[a.name] - self.c.cont.costs[a.name]))
            if self.c.cont is not None:
                self.excess_tail(res)

    # ------------------------------------------------ the excess cost's tail past T
    def window_masses(self):
        """[(lo, hi, w)] from the last window [T - L, T] back to time zero: the weights w (Nt,) of the discounted
        integral over each window of a path on the time nodes (time_mass restricted to the window's panels;
        a panel's shared end node counted once).  Stops early, with the windows it has, at a window edge that
        is not a panel edge."""
        c = self.c; g = c.g; L = g.L; T = c.T
        w = c.time_mass(c.rho); tm = c.tm; side = c.tm_side
        eps = 1e-9 * max(1.0, c.Tg); out = []; hi = T
        while hi > eps:
            lo = max(hi - L, 0.0)
            if not np.any(np.abs(g.bp - lo) <= eps):
                break
            sel = ((tm > lo + eps) | ((np.abs(tm - lo) <= eps) & (side > 0))) & ((tm < hi - eps) | ((np.abs(tm - hi) <= eps) & (side < 0)))
            out.append((lo, hi, w * sel)); hi = lo
        return out

    def excess_tail(self, res, factor: Optional[Dict[str, float]] = None, source: str = "loss path") -> None:
        """The excess cost's tail past T, extrapolated at the closed-loop rate: res.excess_windows[agent], the
        discounted integrals of E[loss(t)] minus the new stationary flow over the windows [T - L, T],
        [T - 2L, T - L], ... (the last window first); the factor r per window, per agent, from the decay of the
        loss path itself over the last two windows (E_last / E_prev, the default) or given (the march's gap
        ratio, `factor`); with 0 < r < 1 the tail is E_last r / (1 - r) (the geometric sum of the windows past
        T), res.excess_costs_tail[agent], and res.excess_costs_total = excess_costs + tail.  An agent whose
        factor is not in (0, 1) (a single window, a sign change, no decay yet) gets no tail; res.excess_tail
        records the factors and their source."""
        c = self.c
        wins = self.window_masses()
        factors: Dict[str, float] = {}
        res.excess_costs_tail = {}; res.excess_costs_total = {}
        for a in self.model.agents:
            E = [float(w @ (res.loss_path[a.name] - c.cont.costs[a.name])) for (_, _, w) in wins]
            res.excess_windows[a.name] = E
            r = None
            if factor is not None:
                r = factor.get(a.name)
            elif len(E) >= 2 and E[1] != 0.0:
                r = E[0] / E[1]
            if r is not None and 0.0 < r < 1.0 and E:
                factors[a.name] = float(r)
                res.excess_costs_tail[a.name] = float(E[0] * r / (1.0 - r))
                res.excess_costs_total[a.name] = float(res.excess_costs[a.name] + res.excess_costs_tail[a.name])
        res.excess_tail = {"source": source, "factor": factors, "windows": [(float(lo), float(hi)) for (lo, hi, _) in wins]}

    # ------------------------------------------------ the transition's paths
    def _atoms(self, agent: Agent, Z: np.ndarray) -> np.ndarray:
        """(m, N, ncol) the loss atoms' kernels in the world Z (n_prim N, ncol): the atoms' sparse reads applied."""
        c = self.c; atoms = c.loss[agent.name][0]
        return np.stack([c.atom_sparse(at) @ Z[c.block(at[0])] for at in atoms])
    def _gram(self, agent: Agent, zeta: np.ndarray, mass) -> np.ndarray:
        """(m, m) the Gram of the atoms' kernels (m, N, k) under the sparse mass."""
        return np.einsum("ink,jnk->ij", zeta, np.stack([mass @ z for z in zeta]))
    def _zeta(self, agent: Agent, Z: np.ndarray) -> np.ndarray:
        """(m, N, ncol) the loss atoms' kernels in the world Z, the band's pre-zero part of a lagged atom included."""
        c = self.c
        zeta = self._atoms(agent, Z)
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
        if name in c.index:
            K = Z[c.block(name)]
        else:
            expr = self.model.expand({name: 1.0})
            K = sum(coef * (c.read_sparse(lag, lag) @ Z[c.block(nm)]) for (nm, lag), coef in expr.items())
        gm = self.maps_from_world(agent, Z, K[None])
        recon = finite_free.reconstruction(self, agent, Z, gm)[0]
        return self._row_variance((K - recon)[None], lambda z: np.einsum("iqk,iqk->q", z, z))

    def warm_maps_from(self, prev) -> Dict[str, np.ndarray]:
        """Raw maps to start from, given a result of this engine on another grid of the same model (a sweep over the
        horizon T): the previous maps read at this grid's nodes where they exist, the continuation's frozen
        stationary maps beyond the previous domain (what a settled transition has there), zero without one."""
        c = self.c; g, gc = c.g, prev.compiled.g
        I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
        outside = ~np.asarray(I != 0).any(axis=1)
        out = {}
        for a in self.model.agents:
            gm = np.zeros(self.shapes[a.name])
            gm[:, :, :c.N] = np.einsum("fn,urn->urf", I, prev.maps[a.name][:, :, :gc.N])
            if c.cont is not None:
                gm[:, :, :c.N][:, :, outside] = c.frozen[a.name][:, :, outside]
            out[a.name] = gm
        return out

    def warm_actions_from(self, prev, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Action kernels to start from, given a result of this engine on another grid of the same model and the
        raw maps of warm_maps_from: the previous fixed point's own action kernels (prev.actions, the iterate the
        solve converged in; its maps are their projection) read at this grid's nodes where they exist, the closed
        loop of `maps` elsewhere (and everywhere when the previous solve iterated on maps)."""
        act = self.actions_from_maps(maps)
        if getattr(prev, "actions", None) is None:
            return act
        c = self.c; g, gc = c.g, prev.compiled.g
        I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
        inside = np.asarray(I != 0).any(axis=1)
        for a in self.model.agents:
            act[a.name][:, inside, :] = np.einsum("fn,unk->ufk", I[inside], prev.actions[a.name])
        return act

    def freeze_before(self, t_lo: float, maps: Optional[Dict[str, np.ndarray]] = None, actions: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Fix every agent's map on the time panels before t_lo at `maps` (raw maps of this engine's shapes; the
        march's warm start) and solve for the rest: SpectralCompiled.freeze_before.  The agents' own action kernels
        on those panels are fixed too, at `actions` (agent -> (nU, N, ncol); the previous fixed point's, warm_actions_from)
        or, without them, at the closed loop of the maps: an agent's best response adds their response to the passive
        world its first-order conditions see (finite_free.best_response), the closed loop keeping its kernel off
        there as everywhere on [0, T] (the impulse columns stay smooth across the fixed panels' shock times).  With
        the fixed point's own actions the reduced solve reproduces the whole strip's fixed point exactly (to 1e-15 in
        a best response); with the closed loop's it differs by the maps' representation error there (1e-5 on Chapter
        3's free maps at 12 nodes, the band's tip being where a map represents its actions worst).  The fixed point
        then runs on the free entries only (free_mask).  t_lo <= 0 clears it."""
        c = self.c
        if not t_lo > 0:
            c.unfreeze(); self._fixed_disc = {}; self._fixed_actions = {}; return
        if maps is None or self.init_kind(maps) != "maps":
            raise ValueError("freeze_before takes the raw maps to fix the early panels at (this engine's shapes)")
        c.freeze_before(t_lo, maps)
        self._fixed_disc = {a.name: np.asarray(maps[a.name], dtype=float)[:, :, c.N:] for a in self.model.agents}
        g = c.g; eps = 1e-9 * max(1.0, c.Tg)
        early = np.concatenate([np.full(pc.n, bool(pc.t1 <= c.t_lo + eps)) for pc in g.pieces])
        if actions is None:
            actions = self.actions_from_maps(maps)
        self._fixed_actions = {a.name: np.asarray(actions[a.name], dtype=float) * early[None, :, None] for a in self.model.agents}

    def free_mask(self, variable: str) -> Optional[np.ndarray]:
        """The free entries of the packed fixed-point vector under freeze_before (None when nothing is fixed
        beyond the buffer): for "actions" the action kernels on the nodes of the panels from t_lo on (the world
        before t_lo is the fixed strategies' and does not move), for "maps" the map nodes and time nodes there."""
        c = self.c
        if c.P_lo == 0:
            return None
        g = c.g; eps = 1e-9 * max(1.0, c.Tg)
        early = np.concatenate([np.full(pc.n, bool(pc.t1 <= c.t_lo + eps)) for pc in g.pieces])
        parts = []
        for a in self.model.agents:
            if variable == "actions":
                nU, N, ncol = self.action_shapes[a.name]
                parts.append(np.broadcast_to(~early[None, :, None], (nU, N, ncol)).ravel())
            else:
                nU, nR, Nm = self.shapes[a.name]
                free = np.concatenate([~early, ~c.fixed_time]) if Nm > c.N else ~early
                parts.append(np.broadcast_to(free[None, None, :], (nU, nR, Nm)).ravel())
        return np.concatenate(parts)

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
            keep[r * Nm:r * Nm + N] = flow & ~c.fixed_nodes                # the buffer's map is frozen, not solved (freeze_before: the early panels too)
            if c.n_init and np.any(c.init_rows[agent.name][r]):
                tpanel = np.repeat(np.arange(g.P), g.nt)
                keep[r * Nm + N:(r + 1) * Nm] = (tpanel < c.P_T) & (tpanel >= c.P_lo) & (g.bp[tpanel] >= d - eps)
        return keep

    def _corner_index(self, agent: Agent) -> np.ndarray:
        """(nG,) the column of _corner_ties of every FOC unknown: the unknowns of a degenerate corner row share
        one, every other unknown has its own (the matrix-free path ties them through this index)."""
        key = ("corner_index", len(agent.controls), len(agent.signals))
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
            self.c._disc[key] = cols
        return self.c._disc[key]

    def _project(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps (nU, nR, N) reproducing the action kernels cact (nU, N, nW) on the closed-loop rows of
        Zfull: maps_from_world, one weighted least-squares solve per time row."""
        return self.maps_from_world(agent, Zfull, cact)

    def maps_from_world(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
        """Raw maps of `agent` reproducing its action kernels cact (nU, N, ncol) given the closed-loop primary
        kernels Zfull: one weighted least-squares projection per time row on the row operator's rows (read one
        panel at a time, finite_free.PanelRows).  The unknowns of a time row are the map nodes of every seen
        row at the row shifted by its observation delay; the Gram runs over the row's action nodes under the
        row's quadrature weights, summed over the channels, with a ridge of MAP_RIDGE relative to its trace.
        With a past the unknowns are the identified ones (_identified: the buffer's map is frozen, a row
        seeing an initial shock adds its discrete weight at the time node), the Gram also runs over the band
        (the band's action nodes read the map through the past's row kernel) and, for each initial shock, the
        point at the row's node on the line s = 0 (its own weight is the point); the nodes of a Duffy
        triangle's degenerate corner row (the lower one's at t = t_p, the upper one's at t = t_{p+1}: one
        point, na nodes, no quadrature weight) are one unknown (_corner_index) with a point condition at the
        corner's action node, since with a past the control reacts at once and the corner is the map's value
        at the oldest increment.  The systems of one size are solved in one LAPACK call."""
        c = self.c; N, Nm = c.N, self.Nm
        nR, nU = len(agent.signals), len(agent.controls)
        past = c.past is not None
        sub = finite_free.panel_rows(self, agent, Zfull).sub
        shifts = [c.panel_shift(c.rows[agent.name][r][3]) if c.rows[agent.name][r][3] > 0 else 0 for r in range(nR)]
        gmap = np.zeros((nU, nR, Nm))
        ctx = self._projection_context(agent) if past else None
        systems: Dict[int, list] = {}                     # size -> [(cols, Rm, G, rhs (nU, n))]
        for (p, it), idx in sorted(c.trow_by_pit.items()):
            if past and (p >= c.P_T or p < c.P_lo):
                continue                                                            # the buffer's rows are frozen (and the panels before t_lo)
            item = self._time_row_system(agent, p, it, idx, cact, sub, shifts, ctx)
            if item is not None:
                systems.setdefault(item[2].shape[0], []).append(item)
        for size, items in systems.items():
            Gs = np.stack([G for _, _, G, _ in items])
            for ui in range(nU):
                sol = np.linalg.solve(Gs, np.stack([rhs[ui] for _, _, _, rhs in items])[:, :, None])[:, :, 0]
                for (cols, Rm, _, _), x in zip(items, sol):
                    gmap[ui].reshape(-1)[cols] = x if Rm is None else Rm @ x
        if c.fixed_maps is not None:
            gmap[:, :, :N][:, :, c.fixed_nodes] = c.fixed_maps[agent.name][:, :, c.fixed_nodes]
            if c.fixed_time is not None and c.n_init:
                gmap[:, :, N:][:, :, c.fixed_time] = self._fixed_disc[agent.name][:, :, c.fixed_time]
        return gmap

    def _projection_context(self, agent: Agent):
        """With a past, what every time row's system of maps_from_world shares: the identified map unknowns (keep),
        the corner group of every map unknown (group), the time index of every node on the line s = 0 (diag_of)
        and the triangle of every action node on a degenerate corner row (corner_of)."""
        c = self.c; g = c.g; Nm = self.Nm; nR = len(agent.signals)
        keep = self._identified(agent)
        group = self._corner_index(agent)[:nR * Nm]                             # map unknown -> its corner group (or itself)
        diag_of = {int(node): j for j, node in enumerate(c.diag)}
        corner_of = {}                                                          # action node -> its triangle, on the degenerate row
        for pc in g.pieces:
            if pc.triangle:
                for node in pc.offset + (0 if not pc.upper else (pc.nt - 1) * pc.na) + np.arange(pc.na):
                    corner_of[int(node)] = (pc.p, pc.q, pc.upper)
        return keep, group, diag_of, corner_of

    def _time_row_system(self, agent: Agent, p: int, it: int, idx: np.ndarray, cact: np.ndarray, sub, shifts, ctx):
        """The weighted least-squares system of one time row (p, it) with action nodes idx, (cols, Rm, G, rhs): the
        map unknowns cols of the seen rows at the row shifted by each delay (with a past the identified ones,
        the corner groups tied to one unknown each by Rm), the Gram G over the row's action nodes under its
        quadrature weights summed over the channels (with a past the point conditions at the corners and, per
        initial shock, at the row's node on s = 0 added, then the ridge) and the right-hand sides per control;
        None when the row has no unknown or no weight."""
        c = self.c; g = c.g; N, nW, Nm = c.N, c.nW, self.Nm
        nR, nU = len(agent.signals), len(agent.controls)
        past = c.past is not None
        if past:
            keep, group, diag_of, corner_of = ctx
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
            return None
        cols = np.concatenate(parts)
        Rm = None
        if past:
            cols = cols[keep[cols]]
            if cols.size == 0:
                return None
            uniq, inv = np.unique(group[cols], return_inverse=True)
            if uniq.size < cols.size:                                           # tie the corner groups to one unknown each
                Rm = np.zeros((cols.size, uniq.size)); Rm[np.arange(cols.size), inv] = 1.0
        Bsub = sub(idx, cols)
        if Rm is not None:
            Bsub = Bsub @ Rm
        G = sum((Bsub[k] * w[:, None]).T @ Bsub[k] for k in range(nW))
        rhs = np.stack([sum((Bsub[k] * w[:, None]).T @ cact[ui, idx, k] for k in range(nW)) for ui in range(nU)])
        if Rm is not None:
            corners = {}
            for j, node in enumerate(idx):
                corners.setdefault(corner_of[int(node)], j) if int(node) in corner_of else None
            for jc in corners.values():                                         # a point condition at each corner's action node
                for k in range(nW):
                    G = G + np.outer(Bsub[k][jc], Bsub[k][jc])
                    rhs = rhs + np.outer(cact[:, idx[jc], k], Bsub[k][jc])
        if past and c.n_init:
            jd = [j for j, node in enumerate(idx) if int(node) in diag_of]      # the time row's node on s = 0, if any
            if jd:
                jd = jd[0]
                for i in range(c.n_init):
                    G = G + np.outer(Bsub[nW + i][jd], Bsub[nW + i][jd])
                    rhs = rhs + np.outer(cact[:, idx[jd], nW + i], Bsub[nW + i][jd])
        if np.trace(G) <= 0:
            return None
        G = G + self.MAP_RIDGE * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
        return cols, Rm, G, rhs

    def world_from_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Closed-loop primary kernels when every agent's action kernels (nU, N, ncol) are given: the closed loop's
        assembly with the controls' rows replaced by the actions, the states solved time panel by time panel."""
        return self.c.closed_loop(None, actions=actions)

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
        zeta = self._atoms(agent, Z)                                          # (m, N, nW)
        mass = c.cost_mass_sparse()
        if c.past is not None:
            # the pre-zero part of the lagged atoms on the band, then the initial shocks along the line s = 0
            zeta[:, :, :c.nW] += c.zeta_past(agent.name)
            G = self._gram(agent, zeta[:, :, :c.nW], mass)
            if c.n_init:
                w = c.time_mass(c.rho)[:c.Nd]
                zd = zeta[:, c.diag, c.nW:]                                   # (m, Nd, n_init)
                G = G + np.einsum("itk,t,jtk->ij", zd, w, zd)
            return float(0.5 * np.sum(Q * G))
        G = self._gram(agent, zeta, mass)
        return float(0.5 * np.sum(Q * G))

    def continuation_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """The variance part of the agent's discounted cost over the buffer [T, T + L] under the frozen stationary
        maps (the shocks of the channels; the band and the initial shocks are gone by T >= L): reported in
        res.cost_parts[agent]["continuation"], not added to res.costs."""
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = self._atoms(agent, Z[:, :c.nW])
        G = self._gram(agent, zeta, c.buffer_mass_sparse())
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
            I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
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

    # ------------------------------------------------------- best response
    # The operators and the first-order-condition system are finite_free's (RowOps, RespOps, FocOps, ProjOps,
    # FocSystem: assembled and factored within settings.foc_dense_max unknowns, GMRES beyond).  With a past
    # they assemble the same objects over the strip and the initial shocks: the seen rows gain their pre-zero
    # part, the row operator the band's read of the past's increments and the discrete weights, the
    # projection the old-shock segments and the point conditions, and the FOC kernel its affine pre-zero part.
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
        """The agent's signal rows in the world Z, (N, ncol) per row through the sparse row blocks (the primaries
        the row reads only; a control in `excluded` is off), and the instantaneous entries [(channel, age,
        weight)] per row; with a band the rows' pre-zero part is added."""
        c = self.c
        rows, inst = [], []
        for r in range(len(agent.signals)):
            blocks, deltas = c.row_blocks_sparse(agent.name, r, excluded)
            y = np.zeros((c.N, Z.shape[1]))
            for nm, op in blocks.items():
                y += op @ Z[c.block(nm)]
            rows.append(y)
            inst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        if c.past is not None and c.g.L is not None:
            for r in range(len(rows)):
                rows[r][:, :c.nW] += c.row_past(agent.name, r)
        return rows, inst
    def _foc_affine(self, agent: Agent, foc) -> Optional[list]:
        """Per control, the pre-zero part of the FOC kernel on the band, (N, nW): the lagged loss atoms read
        before zero through the FOC operators (finite_free.FocOps), None when there is none."""
        c = self.c
        if c.g.L is None:
            return None
        zp = c.zeta_past(agent.name)
        if not np.any(zp):
            return None
        return [foc.foc(ui, zp) for ui in range(len(agent.controls))]

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        """The agent's best response to `maps` (EngineBase.best_response's contract): finite_free.best_response,
        the operators applied and the FOC system factored within settings.foc_dense_max, solved by GMRES beyond
        (self.foc_free); the dict also carries "krylov", the GMRES iterations (0 when factored)."""
        return finite_free.best_response(self, agent, maps, want_decomp)
    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        recon_all = finite_free.reconstruction(self, agent, Zfull, g)
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
            recon = recon_all[ui]
            err = np.abs(recon - actions[ui])
            err[:, c.nW:] = 0.0
            err[c.diag, c.nW:] = np.abs(recon - actions[ui])[c.diag, c.nW:]     # an initial shock's column: on the line s = 0 only
            rel = err.max(axis=1) / max(1e-300, np.abs(actions[ui][:, :c.nW]).max(), np.abs(actions[ui][c.diag, c.nW:]).max(initial=0.0))
            worst = max(worst, float(rel.max()))
            for key, sel in regions:
                parts[key] = max(parts[key], float(rel[sel].max(initial=0.0)))
        self._rep_parts[agent.name] = parts
        return worst

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
