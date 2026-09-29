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
from .monitoring import MonitoredDeviations, compose_spikes, spike_controls
from .past import Past
from .results import TriangleResult, TransitionResult
from ._settings import tunable
from .spec import Agent, Model
from .spectral_compiled import SpectralCompiled
from .spectral_means import SpectralMeans

__all__ = ["SpectralCompiled", "ClosedLoopRows", "SpectralFiniteSolver"]


class SpectralFiniteSolver(SpectralMeans, MonitoredDeviations, EngineBase):
    MONITORING = True                   # monitored deviations and instant observations, without a past (see __init__)
    RISK_SENSITIVE = True               # risk-averse agents (the entropic objective, risk.py): no past or initial shocks only (see __init__)
    RESULT = TriangleResult
    TOL, DAMPING, MAX_NEWTON = 1e-8, 0.5, 8
    MAP_RIDGE = tunable("map_ridge")      # ridge of the per-time-row map projection, relative to the row's own Gram (settings)

    def __init__(self, model: Model, verbose: bool = False, settings=None, past=None, continuation=None):
        """The triangle grid's compiled model and the map shapes (nU, nR, N).  settings: the tuning constants
        (noisestate.Settings, or a dict of its fields; the defaults when None).  past: the known past of a
        transition (a Past, a StationaryResult, a stationary Model/dict/path solved on the fly, or a list
        of initial shocks; see past.py): the game then starts at time zero from that regime.  continuation:
        how it goes on after T: None or "end" (the game ends at T), a converged StationaryResult of this
        model at the past's window, or "stationary" (that result solved here, at numerics.nodes): every
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
        if (past is not None or continuation not in (None, "end")) and any(a.monitors or a.instant for a in model.agents):
            raise NotImplementedError("monitored deviations and instant observations are solved on a finite horizon "
                                      "without a past or a continuation; a transition with them is not built yet")
        averse = [a.name for a in model.agents if a.risk_aversion]
        if averse and ((past is not None and past.window) or continuation not in (None, "end")):
            raise NotImplementedError(f"risk-averse agents ({', '.join(averse)}) are solved on a finite horizon, with no past or a past of "
                                      "initial shocks only (no window), and no continuation; a transition with them is not built yet")
        continuation = self._continuation_of(model, past, continuation,
                                             model.numerics.continuation_nodes if hz.kind == "transition" else None)
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
        self._rep_by_node: Dict[str, np.ndarray] = {}           # agent -> the relative representation error at every node
        self._fixed_disc: Dict[str, np.ndarray] = {}           # agent -> the fixed discrete weights (freeze_before)
        self._fixed_actions: Dict[str, np.ndarray] = {}        # agent -> its action kernels (nU, N, ncol) on the fixed panels, zero elsewhere
        self._profile = (None, None)                           # (maps key, their closed loop): the world a risk-averse agent's K is taken in
        self._risk_scale = 1.0                                 # the continuation's step in risk aversion (solve): theta times this
        self._zbar = None                                      # the primaries' means of the last solve_means (the entropic cost's linear part)
        self._consistent_info: Dict[str, dict] = {}            # agent -> the last consistent-planning shift's diagnostics
        if averse and self.settings.risk_planning not in ("precommitment", "consistent"):
            raise ValueError(f"settings.risk_planning must be 'precommitment' or 'consistent', not {self.settings.risk_planning!r}")
        if averse and self.settings.risk_planning == "consistent":
            why = []
            if past is not None:
                why.append("no past")
            if any(a.monitors or a.instant for a in model.agents):
                why.append("no monitoring or instant observations")
            if why:
                raise NotImplementedError(f"risk-averse agents ({', '.join(averse)}) under consistent planning on the finite engine need "
                                          + "; ".join(why))
        if averse and self.settings.risk_planning == "consistent" and self._mean_driven():
            raise NotImplementedError(f"risk-averse agents ({', '.join(averse)}) under consistent planning on the finite engine are "
                                      "solved without means (each date's linear part is its own)")
        if any(a.risk_aversion and a.integrals for a in model.agents) and self._mean_driven():
            raise NotImplementedError("risk-averse agents with stochastic-integral terms (integrals) are solved without means: the "
                                      "integrals' mean part (int xbar' L dW) joins the cost's linear part, which is not built yet")

    @staticmethod
    def _continuation_of(model: Model, past, continuation, nodes: Optional[int] = None):
        """None / "end" -> None; "stationary" -> this model's stationary equilibrium at the past's window, solved
        here (numerics.nodes per panel, or `nodes`, numerics.continuation_nodes, when given; the finite horizon's
        breakpoints, initial values and transition blocks dropped); a StationaryResult -> itself (checked by
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
            dropped = [hz.pop(k, None) for k in ("past", "continuation", "T", "settle")]
            #  a parameter that only the dropped lengths used (T: T) is unused in the continuation, where the
            #  unused-parameter check, a guard against typos in a written file, would refuse it: drop it too
            import json, re
            params = d.get("params") or {}
            for name in {n for v in dropped[2:] if isinstance(v, str) for n in re.findall(r"[A-Za-z_]\w*", v)} & set(params):
                elsewhere = json.dumps({k: v for k, v in d.items() if k != "params"}) + \
                    json.dumps({k: v for k, v in params.items() if k != name})
                if not re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", elsewhere):
                    params.pop(name)
            nm = d.setdefault("numerics", {})
            for k in ("breakpoints", "engine", "continuation_nodes"):
                nm.pop(k, None)
            if nodes is not None:
                nm["nodes"] = int(nodes)
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
        self._require_settled(res)
        super()._finish(res)
        from .risk import RiskBreakdown
        for a in self.model.agents:
            if a.risk_aversion:
                # the equilibrium's own spectrum (risk_report raises RiskBreakdown past it): an iterate beyond the
                # breakdown was answered at a smaller theta (risk.Tilt, clip), so a fixed point that still needs that is
                # no equilibrium of the entropic game
                try:
                    res.risk[a.name] = self.risk_report(a, res.world, res.costs[a.name], zbar=self._zbar, maps=res.maps)
                except RiskBreakdown as exc:
                    if res.converged:
                        raise
                    # a solve that did not converge ends at its best iterate, which can be anywhere: past the breakdown
                    # there is no evidence about the equilibrium, and raising made _risk_solve give up before its retry
                    # (Kyle-Back at eps 0.05, theta 1.7 from the 1.6 equilibrium: theta lambda_max 194 at a wild iterate)
                    res.risk[a.name] = {"risk_aversion": exc.theta, "entropic": float("nan"), "expected": float(res.costs[a.name]),
                                        "lambda_max": exc.lam_max, "theta_lambda_max": exc.theta * exc.lam_max}
                    res.message += (f"; the last iterate is past {a.name}'s risk-sensitive breakdown (theta lambda_max = "
                                    f"{exc.theta * exc.lam_max:.3g}): no entropic cost there")
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
        ctx = self._projection_context(agent)
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
        """What every time row's system of maps_from_world shares: with a past the identified map unknowns (keep),
        the corner group of every map unknown (group), the time index of every node on the line s = 0 (diag_of)
        and the triangle of every action node on a degenerate corner row (corner_of)."""
        c = self.c; g = c.g; Nm = self.Nm; nR = len(agent.signals)
        keep = self._identified(agent) if c.past is not None else None
        group = self._corner_index(agent)[:nR * Nm]                             # map unknown -> its corner group (or itself)
        diag_of = {int(node): j for j, node in enumerate(c.diag)}
        corner_of = {}                                                          # action node -> its triangle, on the degenerate row
        for pc in g.pieces:
            if pc.triangle:
                for node in pc.offset + (0 if not pc.upper else (pc.nt - 1) * pc.na) + np.arange(pc.na):
                    corner_of[int(node)] = (pc.p, pc.q, pc.upper)
        # without a past the corners are tied and point-conditioned only for an agent with a row whose noise loads a
        # shock that also drives a state: it answers that shock at once, so its map at age 0 on a corner row is not
        # zero and the projection, which gives the corner no quadrature weight, would leave it at the ridge's zero;
        # every other model keeps the projection it had, bit for bit
        if c.past is None:
            driving = {ch for st in self.model.states for ch, v in st.noise.items() if v}
            if not any(ch in driving and v for r in agent.signals for ch, v in r.noise.items()):
                group = np.arange(nR * Nm)
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
        if uniq.size < cols.size:                                               # tie the corner groups to one unknown each
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
        return {a.name: self.maps_from_world(a, Z, self._map_part(a, Z, actions[a.name])) for a in self.model.agents}

    # ------------------------------------------------------- instant observations and monitored deviations
    #  The iteration is monitoring.py's (MonitoredDeviations), as on the stationary engine; the pieces here: a spike
    #  of a control draws the instant reactions of the agents seeing its level (c.composite); for an origin i with
    #  privy players, the seed world W_i = Z0_i + sum_v C_v D^{v<-i}, the privy players' maps off in Z0_i, C_v the
    #  response operator of v's spike (here a two-time kernel: the response at t to a spike at the node's shock time,
    #  RespOps' path), the kernels D^{v<-i}(t, s) fixed by the privy players' first-order conditions on W_i at every
    #  node; the deviating player's own first-order condition sees its frozen spike answered by the privy players as
    #  the blip seeds it decomposes into (Lemma 6.6), and the path is built with the ordinary spike responses.

    def _spikes(self, c, maps, agent: Agent, excluded=None, own_frozen: bool = True):
        """(Zpass (n_prim N, ncol), R (n_prim N, nU)): the closed loop with `excluded` (default the agent) off, and
        the responses to a spike of each of the agent's controls with the instant reactions it draws."""
        comp = c.composite or {}
        ctrls = spike_controls(agent.controls, comp)
        Zp = c.closed_loop(maps, excluded=agent.name if excluded is None else excluded, impulse_controls=ctrls,
                           own_frozen=own_frozen)
        spike = compose_spikes(Zp[:, c.ncol:], ctrls, agent.controls, comp)
        R = np.stack([spike[u] for u in agent.controls], axis=1)
        return Zp[:, :c.ncol], R

    def _resp_dense(self, v: str, col: np.ndarray) -> np.ndarray:
        """(n_prim N, N): the world of an action kernel of control v through its spike's response column, v's own
        block the action, the instant reactions it draws moving with it (finite_free.RespOps)."""
        from types import SimpleNamespace
        return finite_free.RespOps(self, SimpleNamespace(controls=[v]), col[:, None]).dense(0)

    def _seed_setup(self, maps, origin: str):
        """(ctrls, Z0, C) as on the stationary engine: the privy controls (origin's first), their spike columns with
        the privy players' maps off, and per privy control its response operator (n_prim N x N)."""
        c = self.c
        comp = c.composite or {}
        owner = {a.name: a for a in self.model.agents}
        P = self.model.privy(origin)
        ctrls = [u for n in P for u in owner[n].controls]
        need = spike_controls(ctrls, comp)
        spike = compose_spikes(c.closed_loop(maps, excluded=tuple(P), impulse_controls=need)[:, c.ncol:], need, ctrls, comp)
        Z0 = np.stack([spike[v] for v in ctrls], axis=1)
        return ctrls, Z0, {v: self._resp_dense(v, spike[v]) for v in ctrls}

    def _monitor_focs(self, agent: Agent, maps, R: np.ndarray, origins, seed_consts) -> list:
        """The agent's first-order-condition operators on the monitored responses R (MonitoredDeviations._monitoring),
        dense (N x n_prim N) per control.  A risk-averse responder: f^xi_t(s) + theta <S f_t, K e_xi(s)> = 0, the
        second term linear in the seed world through its atoms (risk.Tilt.seed_operator) plus the seed spike's own
        point mass (seed_constant), which goes into seed_consts[(agent, control)] per origin it is privy to; the
        derivation is in docs/method.md."""
        c = self.c; N = c.N
        comp = c.composite or {}
        owner = {a.name: a for a in self.model.agents}
        ops = finite_free.FocOps(self, agent, R)
        out = [ops.dense(ui) for ui in range(len(agent.controls))]
        tilt = self._tilt(agent, maps, ops, R) if agent.risk_aversion else None
        if tilt is not None:
            Zp = self._profile_world(maps).reshape(len(c.prim), N, c.ncol)
            out = [out[ui] + tilt.seed_operator(ui, Zp) for ui in range(len(out))]
            for ui in range(len(out)):
                seed_consts[(agent.name, ui)] = {i: np.stack([tilt.seed_constant(ui, Zp, comp.get(u, {u: 1.0})) for u in owner[i].controls],
                                                             axis=1) for i in origins if agent.name in self.model.privy(i)}
        else:
            for ui in range(len(out)):
                seed_consts.pop((agent.name, ui), None)
        return out

    def _frozen_responses(self, j: str, setup, Dj: np.ndarray, own: int) -> np.ndarray:
        """R^mon_j: the responses to a frozen spike of each of j's controls, the privy players reading it as the blip
        seeds sigma, sigma + D^{j<-j} * sigma = delta (a Volterra equation in the seed's time, the kernels composed
        along the response path), and responding to them.  The kernels are composed through one path operator; only
        the own-control block the Volterra solve needs is formed as dense rows, the rest are applied."""
        from .spectral_operators import PathOp
        c = self.c; N = c.N; g = c.g
        ctrls, Z0, C = setup
        nC = len(ctrls)
        lp = c._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                     known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        op = PathOp(lp, np.stack([Dj[o2, kk] for o2 in range(own) for kk in range(nC)], axis=1))   # column o2 nC + kk
        col = lambda o2, kk: o2 * nC + kk
        if op.empty:
            M = np.eye(own * N)
        else:
            M = np.eye(own * N) + np.block([[op.rows(col(o2, u), 0, N) for o2 in range(own)] for u in range(own)])
        out = np.zeros((len(c.prim) * N, own))
        for o in range(own):
            rhs = -np.concatenate([Dj[o, u] for u in range(own)])
            try:
                s = np.linalg.solve(M, rhs).reshape(own, N)
            except np.linalg.LinAlgError:              # a trial point far from the equilibrium can make the discretised
                s = np.linalg.lstsq(M, rhs, rcond=None)[0].reshape(own, N)   # Volterra operator singular; the
                                                        # iteration keeps its best round (_monitoring) and moves on
            Is = [op.unknown(s[o2]) for o2 in range(own)] if not op.empty else None
            colv = Z0[:, o].copy()
            for kk, v in enumerate(ctrls):
                if kk < own:
                    continue
                x = Dj[o, kk].copy()
                if Is is not None:
                    for o2 in range(own):
                        x += op.apply(Is[o2], col(o2, kk))[:N]
                colv += C[v] @ x
            out[:, o] = colv
        return out

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
            return float(0.5 * np.sum(Q * G)) + self._terminal_variance(agent, Z)
        G = self._gram(agent, zeta, mass)
        return float(0.5 * np.sum(Q * G)) + self._terminal_variance(agent, Z)

    def _terminal_variance(self, agent: Agent, Z: np.ndarray) -> float:
        """The variance part of a terminal loss, e^{-rho T} 1/2 sum Q_T,ij E[zeta_i(T) zeta_j(T)]: over the channels'
        shocks alive at T (born in [0, T], and with a band the old ones born in [T - L, 0)) and, with initial shocks,
        their columns at T (a unit-variance draw each: the column's value at T squared).  Zero without one."""
        c = self.c
        terminal = (c.terminal or {}).get(agent.name)
        if not terminal:
            return 0.0
        atoms, QT, _ = terminal
        IT, w = c.terminal_quadrature
        za = [c.atom_sparse(at) @ Z[c.block(at[0])] for at in atoms]                         # (N, ncol) per atom
        zt = np.stack([IT @ z[:, :c.nW] for z in za])                                        # (m, nq, nW)
        G = np.einsum("iqk,q,jqk->ij", zt, w, zt)
        if Z.shape[1] > c.nW:                                                                # the initial shocks
            zi = np.stack([(c.terminal_point @ z[:, c.nW:])[0] for z in za])                  # (m, n_init)
            G = G + zi @ zi.T
        return float(np.exp(-c.rho * c.T) * 0.5 * np.sum(QT * G))

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
        if len(foc.atoms) > zp.shape[0]:            # a terminal loss's atoms (states at T) read nothing before zero
            zp = np.concatenate([zp, np.zeros((len(foc.atoms) - zp.shape[0],) + zp.shape[1:])])
        return [foc.foc(ui, zp) for ui in range(len(agent.controls))]

    def _profile_world(self, maps) -> np.ndarray:
        """The closed loop of every agent's current maps (n_prim N, ncol), kept for the evaluation's other agents."""
        key = self._maps_key(maps)
        if self._profile[0] != key:
            self._profile = (key, self.c.closed_loop(maps))
        return self._profile[1]

    def _theta(self, agent: Agent) -> float:
        """The agent's risk aversion as this solve uses it: theta scaled by the continuation's step (solve)."""
        return float(agent.risk_aversion) * self._risk_scale

    def _tilt(self, agent: Agent, maps, foc, R):
        """A risk-averse agent's correction (risk.Tilt) frozen at the profile `maps`: the cost kernel K and its spectrum in
        their closed loop, the spike responses R of the continuation (the envelope, as FocOps has them); None at a
        continuation step of zero risk aversion.  An iterate past the breakdown is answered at a smaller theta (clip)."""
        from .risk import Tilt, geometry
        th = self._theta(agent)
        if not th:
            return None
        return Tilt(geometry(self.c, self.settings), foc, agent, th, self._profile_world(maps), R, clip=True)

    def _responses_on(self, agent: Agent, maps, R0: np.ndarray) -> np.ndarray:
        """(n_prim N, nU): the world's response to a unit spike of each of the agent's controls with its own later reactions
        ON (consistent planning: the later selves play the map and do not share the date-t objective).  The later selves
        know their own orders, so they react to the spike's effects on their passive rows (own controls excluded): from R0
        (the others reacting, the own reaction off), the own reaction kernels c solve c = G Y (R0 + Resp c), G the map's
        convolution per row (the closed loop's conv_left rows) and Y the passive rows' reads (as the stationary engine's
        StationaryTilt._responses_on)."""
        c = self.c; N = c.N; nP = len(c.prim); nU = len(agent.controls)
        Resp = [finite_free.RespOps(self, agent, R0).dense(vi) for vi in range(nU)]            # (nP N, N) each
        gm = maps[agent.name]
        excl = frozenset(agent.controls)
        GY = [np.zeros((N, nP * N)) for _ in range(nU)]
        for r, (rname, drift, E, delay) in enumerate(c.rows[agent.name]):
            blocks, _ = c.row_blocks_sparse(agent.name, r, excl)
            for v in range(nU):
                Cr = c.conv_left_rows(gm[v, r][:N], delay, 0, N)                               # (N, N): row kernel -> action kernel
                for nm, S in blocks.items():
                    GY[v][:, c.block(nm)] += (S.T @ Cr.T).T
        A = np.eye(nU * N) - np.block([[GY[v] @ Resp[w] for w in range(nU)] for v in range(nU)])
        out = np.empty_like(R0)
        for ui in range(nU):
            cvec = np.linalg.solve(A, np.concatenate([GY[v] @ R0[:, ui] for v in range(nU)]))
            out[:, ui] = R0[:, ui] + sum(Resp[w] @ cvec[w * N:(w + 1) * N] for w in range(nU))
        return out

    def _consistent_shift(self, agent: Agent, maps) -> Optional[np.ndarray]:
        """Consistent planning (settings.risk_planning = "consistent"): the shift (nU, N, ncol) a risk-averse agent's best
        response adds to its risk-neutral FOC kernel, frozen at the profile `maps`, (phi^on - phi^off) + theta K_t S_t f_t^on on
        the seen nodes (risk.ConsistentTilt); None at a continuation step of zero risk aversion."""
        from .risk import ConsistentTilt, geometry
        th = self._theta(agent)
        if not th:
            return None
        c = self.c; N, nP, ncol = c.N, len(c.prim), c.ncol
        Z = self._profile_world(maps)
        R0 = self._spikes(c, maps, agent)[1]
        Ron = self._responses_on(agent, maps, R0)
        fon = finite_free.FocOps(self, agent, Ron, envelope=False)
        foff = finite_free.FocOps(self, agent, R0)
        Zr = Z.reshape(nP, N, ncol)
        tilt = ConsistentTilt(geometry(c, self.settings), fon, agent, th, Z, Ron, clip=True)
        aon, aoff = fon.atoms_of(Zr), foff.atoms_of(Zr)
        out = np.zeros((len(agent.controls), N, ncol))
        for ui in range(len(agent.controls)):
            out[ui] = fon.foc(ui, aon) - foff.foc(ui, aoff)
            out[ui, :, :c.nW] += tilt.consistent_delta(ui, Zr)
        self._consistent_info[agent.name] = tilt.info
        return out

    def _spectrum(self, agent: Agent, world: np.ndarray):
        """risk.Tilt with K's spectrum only, at theta = 0 (no breakdown test): its lam, lam_max and trace_K2."""
        from .risk import Tilt, geometry
        c = self.c
        foc = finite_free.FocOps(self, agent, np.zeros((len(c.prim) * c.N, len(agent.controls))))
        return Tilt(geometry(c, self.settings), foc, agent, 0.0, world, None, spectrum_only=True)

    def solve_means(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        zbar = super().solve_means(maps)
        self._zbar = zbar
        return zbar

    def _risk_mean_paths(self, agent: Agent, foc):
        """The affine map from the primaries' means zbar (nP Nt,) to a = Q mbar + q of the agent's flow atoms at the risk
        geometry's one-time nodes and a_T = Q_T mbar_T + q_T of its terminal atoms: (Ax (n1 P, m, nP Nt + 1), AT (mt, nP Nt + 1)),
        the last column the constant part (the targets q and the past's means read before zero)."""
        from .risk import geometry
        c = self.c; Nt, nP = c.Nt, len(c.prim)
        geo = geometry(c, self.settings)
        m = foc.m_flow; mt = len(foc.atoms) - m
        Qf = np.asarray(foc.Q, dtype=float)[:m, :m]; qf = np.asarray(foc.q, dtype=float)[:m]
        flow_atoms = list(foc.atoms[:m])
        cols = np.concatenate([np.eye(nP * Nt), np.zeros((nP * Nt, 1))], axis=1)       # unit means, then zbar = 0
        base = self._mean_atoms(np.zeros(nP * Nt), flow_atoms)                            # (m, Nt): the constant part
        M = np.stack([self._mean_atoms(cols[:, j], flow_atoms) - base for j in range(nP * Nt)], axis=2)   # (m, Nt, nP Nt)
        M = np.concatenate([M, base[:, :, None]], axis=2)
        mx = np.einsum("xt,jtb->xjb", geo.Btm, M)                                         # (n1 P, m, nP Nt + 1)
        Ax = np.einsum("jl,xlb->xjb", Qf, mx)
        Ax[:, :, -1] += qf[None, :]
        AT = None
        if mt:
            atoms, QT, qT = c.terminal[agent.name]
            read = c.g.interp_sparse(np.array([float(c.T)]), np.zeros(1), side_t=-1) @ c.mean_embed
            XT = np.zeros((mt, nP * Nt + 1))
            for i, (nm, lag) in enumerate(atoms):
                XT[i, c.index[nm] * Nt:(c.index[nm] + 1) * Nt] = np.asarray(read).ravel()
            AT = np.asarray(QT, dtype=float) @ XT
            AT[:, -1] += np.asarray(qT, dtype=float)
        return Ax, AT

    def _mean_conditions(self, agent: Agent, maps: Dict[str, np.ndarray], line: Optional[bool] = None):
        """SpectralMeans._mean_conditions; a risk-averse agent's mean condition gains the tilt of the cost's linear part,
        theta <f_t, S k> (derivation: the tilted law of the shocks given F_t has the mean (I - theta Sigma_t K)^-1 (W_hat +
        theta Sigma_t k), and on the kernels' own condition Sigma_t drops out of theta <(I - theta K Sigma_t)^-1 f_t, Sigma_t k>
        = theta <S f_t, k> = theta <f_t, S k>).  k is linear in the means (k = A' G (Q mbar + q)), so the system stays one
        linear solve: row t gains theta <f_t, S k(e_j)> in column j and theta <f_t, S k(0)> on the right."""
        Mu, bu = super()._mean_conditions(agent, maps, line=line)
        th = self._theta(agent)
        if not th:
            return Mu, bu
        if self._on_line(line):
            raise NotImplementedError("risk-averse agents with means are solved on the line s = 0 (no window)")
        from .risk import Tilt, geometry
        c = self.c; Nt, nP, N = c.Nt, len(c.prim), c.N
        R = self._spikes(c, maps, agent)[1]
        R = self._impulse_responses(agent, maps, R)
        foc = finite_free.FocOps(self, agent, R)
        world = self._profile_world(maps)
        tilt = Tilt(geometry(c, self.settings), foc, agent, th, world, R)
        if tilt.geo.n_rows != Nt:
            raise NotImplementedError("risk-averse agents with means: the time rows and the mean nodes differ")
        Ax, AT = self._risk_mean_paths(agent, foc)
        k, kxi, Sk, Skxi = tilt.linear_part(Ax, AT)
        Z = world.reshape(nP, N, c.ncol)
        for ui in range(len(agent.controls)):
            G, fxi = tilt.pairing(ui, Z)
            corr = G @ Sk.reshape(-1, Sk.shape[2])                                      # (Nt, nP Nt + 1)
            if fxi is not None:
                corr = corr + fxi @ Skxi
            row = slice(ui * Nt, (ui + 1) * Nt)
            Mu[row] += th * corr[:, :-1]
            bu[row] -= th * corr[:, -1]
        return Mu, bu

    def risk_report(self, agent: Agent, world: np.ndarray, expected: float, zbar: Optional[np.ndarray] = None, maps=None) -> dict:
        """A risk-averse agent's entropic cost in the closed loop `world` (n_prim N, ncol) whose expected cost is `expected`:
        {"risk_aversion", "entropic" (theta^-1 log E exp(theta C)), "expected", "lambda_max" (the largest eigenvalue of the
        cost kernel K), "theta_lambda_max"}; the discounted cost over [0, T] as res.costs has it.  Raises RiskBreakdown
        when theta lambda_max >= 1 (E exp(theta C) infinite)."""
        from .risk import RiskBreakdown
        th = self._theta(agent)
        t = self._spectrum(agent, world)
        if th * t.lam_max >= 1.0:
            raise RiskBreakdown(agent.name, th, t.lam_max)
        t.theta = th
        out = {"risk_aversion": th, "entropic": float(expected) + t.entropic_excess(), "expected": float(expected),
               "lambda_max": t.lam_max, "theta_lambda_max": th * t.lam_max}
        if zbar is not None and self._mean_driven():
            # the linear part: J gains theta / 2 <k, S k> (log E e^{theta (k'W + W'KW / 2)} = -1/2 log det(I - theta K) +
            # theta^2 / 2 k' S k)
            from .risk import Tilt, geometry
            c = self.c
            foc = finite_free.FocOps(self, agent, np.zeros((len(c.prim) * c.N, len(agent.controls))))
            tl = Tilt(geometry(c, self.settings), foc, agent, 0.0, world, None, spectrum_only=True, linear=True)
            Ax, AT = self._risk_mean_paths(agent, foc)
            x = np.concatenate([zbar, [1.0]])
            ax = (Ax @ x)[:, :, None]; aT = None if AT is None else (AT @ x)[:, None]
            parts = tl.linear_part(ax, aT, theta=th)
            lin = 0.5 * th * float(tl.quad_linear(*parts)[0])
            out["entropic"] += lin
            out["linear_part"] = lin
        return out

    RISK_STEP = 0.9          # a start at which every theta lambda_max is at most this is solved from directly (solve)
    RISK_GAIN = 0.6          # else a continuation step closes this fraction of the gap to the breakdown, 1 - theta lambda_max
    RISK_STEPS = 40          # continuation steps at most before the path is taken to end at the breakdown
    RISK_PATH_TOL = 1e-5     # the fixed-point tolerance of a continuation step short of the model's theta (solve)
    RISK_EDGE = 1e-3         # ... or once a step's equilibrium is within this of it (1 - theta lambda_max) short of the model's theta

    def solve(self, start_from=None, tol=None, damping=None, max_newton=None, variable: str = "actions", start_policy: str = "zero",
              max_evaluations=None, deadline=None, progress=None, diagnostics: bool = True):
        """EngineBase.solve; with risk-averse agents and no start given, by continuation in risk aversion when the start
        needs it.  The zero start is the uncontrolled world, whose entropic cost is infinite beyond a small theta (1/0.81
        on Chapter 1's game), and even the risk-neutral equilibrium can be past the breakdown (theta 2.5 there), where no
        best response is defined.  When every agent's theta lambda_max (lambda_max the largest eigenvalue of its cost
        kernel K, the breakdown at theta lambda_max = 1) is at most RISK_STEP in the uncontrolled world the solve starts
        from zero as any other; else from the risk-neutral equilibrium (every theta scaled by 0), then scaling theta up
        in steps warm-started from the last, each closing RISK_GAIN of the gap 1 - theta lambda_max at the last
        equilibrium (a step lands on scale 1 as soon as theta lambda_max <= RISK_STEP there).  A path whose step
        equilibrium comes within RISK_EDGE of the breakdown short of the model's theta, or that has not reached it within
        RISK_STEPS steps, ends at the breakdown: RiskBreakdown.  res.evaluations counts
        every step's; res.message names the steps."""
        kw = dict(tol=tol, damping=damping, max_newton=max_newton, variable=variable, progress=progress)
        averse = [a for a in self.model.agents if a.risk_aversion]
        self._risk_scale = 1.0
        if not averse or start_from is not None or start_policy != "zero":
            if not averse:
                return super().solve(start_from=start_from, start_policy=start_policy, max_evaluations=max_evaluations,
                                     deadline=deadline, diagnostics=diagnostics, **kw)
            return self._risk_solve(averse, dict(start_from=start_from, start_policy=start_policy, max_evaluations=max_evaluations,
                                                 deadline=deadline, diagnostics=diagnostics, **kw))
        import time
        from .risk import RiskBreakdown

        def use(world, scale):             # the largest theta lambda_max over the agents at `scale`, and whose it is
            lam = {a.name: self._spectrum(a, world).lam_max for a in averse}
            worst = max(averse, key=lambda a: a.risk_aversion * lam[a.name])
            return worst, lam[worst.name], worst.risk_aversion * lam[worst.name]
        zero = self.zero_maps()
        worst, lam, x = use(self.c.closed_loop(zero), 0.0)
        scale = 1.0 if x <= self.RISK_STEP else 0.0
        t0 = time.time(); evals = 0; scales = []; maps = None; prev = None
        try:
            while True:
                self._risk_scale = scale; scales.append(scale)
                last = scale >= 1.0
                left = None if max_evaluations is None else max(1, max_evaluations - evals)
                dl = None if deadline is None else max(0.0, deadline - (time.time() - t0))
                # a step short of the model's theta is only the next one's start and where its size is measured: solved to
                # RISK_PATH_TOL (the model's tol, if looser); from the second step on it starts from the secant through the
                # last two equilibria, extended to this scale (and from the last equilibrium itself if that does not converge)
                step_kw = kw if last else {**kw, "tol": max(self.TOL if tol is None else tol, self.RISK_PATH_TOL)}
                start = maps
                if prev is not None:
                    w = (scale - scales[-2]) / (scales[-2] - prev[0])
                    start = {n: maps[n] + w * (maps[n] - prev[1][n]) for n in maps}
                try:
                    if last:        # the model's theta: a stall or a failing best response is retried with tighter ones (_risk_solve)
                        res = self._risk_solve(averse, dict(start_from=start, max_evaluations=left, deadline=dl, diagnostics=diagnostics, **step_kw))
                    else:
                        res = super().solve(start_from=start, max_evaluations=left, deadline=dl, diagnostics=False, **step_kw)
                    if (not last and not res.converged and prev is not None and (max_evaluations is None or evals + res.evaluations < max_evaluations)
                            and (deadline is None or time.time() - t0 < deadline)):
                        evals += res.evaluations
                        left = None if max_evaluations is None else max(1, max_evaluations - evals)
                        dl = None if deadline is None else max(0.0, deadline - (time.time() - t0))
                        res = super().solve(start_from=maps, max_evaluations=left, deadline=dl, diagnostics=diagnostics if last else False, **step_kw)
                except RiskBreakdown as exc:
                    if last and len(scales) == 1:
                        raise
                    raise RiskBreakdown(exc.agent, max(a.risk_aversion for a in averse if a.name == exc.agent), exc.lam_max,
                                        reached=exc.theta) from None
                evals += res.evaluations
                if last or not res.converged:
                    break
                prev = None if maps is None else (scales[-2], maps)
                maps = res.maps
                worst, lam, x = use(res.world, scale)                   # x = theta lambda_max at the full theta
                done = scale * x                                        # ... and at this step's
                if 1.0 - done < self.RISK_EDGE or len(scales) > self.RISK_STEPS:   # at the breakdown short of the model's theta
                    raise RiskBreakdown(worst.name, worst.risk_aversion, lam, reached=scale * worst.risk_aversion)
                scale = 1.0 if x <= self.RISK_STEP else min(1.0, (done + self.RISK_GAIN * (1.0 - done)) / x)
        finally:
            self._risk_scale = 1.0
        steps = ", ".join(f"{s:.3g}" for s in scales)
        res.evaluations = evals
        res.seconds = time.time() - t0
        res.message = (f"continuation in risk aversion, theta scaled by {steps}: " + res.message) if len(scales) > 1 else res.message
        if not (scales[-1] >= 1.0):
            res.converged = False
            res.message += f"; stopped at risk aversion scaled by {scales[-1]:.3g} of the model's (that step did not converge)"
        return res

    RISK_KRYLOV_RETRY = 1e-6     # the risk-averse best responses' Krylov reduction on a retry after a stall or a failing best response (_risk_solve)
    RISK_PROX = 1.0              # ... and the weight of their proximal term then, relative to the system's mean diagonal (FocSystem._proximal)

    def _risk_solve(self, averse, args: dict):
        """EngineBase.solve(**args) for a model with risk-averse agents, retried once from the same start with the best
        responses' Krylov solves tightened from FocSystem.RISK_KRYLOV_REDUCTION (1e-4 of the warm start's residual) to
        RISK_KRYLOV_RETRY when it stalls or a best response fails: near a fixed point the looser stop leaves a noise floor in
        the fixed-point map that Anderson and the Newton polish read as a stall (Kyle-Back at eps 0.05, theta 1.5, 16 nodes:
        residual 2e-5; with the retry it converges).  The retry's best responses are also proximal (RISK_PROX,
        FocSystem._proximal): the frozen best response's system goes singular where the entropic cost is still convex, and
        the iteration then diverges (the same market from theta 1.58 on); the proximal one crosses it (theta 1.6, 1.7 and 2
        from the theta 1.5 equilibrium, 70 to 100 evaluations, against extras/kyle_reference.py).  If the retry fails too, the entropic cost's curvature is probed
        (_risk_curvature) at the best iterate, or at the start when a best response raised, and the message says whether
        the risk-averse best response has lost its minimum there."""
        from .risk import RiskBreakdown
        first, failed = None, None
        try:
            first = super().solve(**args)
            if first.converged:
                return first
        except RiskBreakdown:
            raise
        except ValueError as exc:
            failed = exc
        self._risk_krylov_reduction = self.RISK_KRYLOV_RETRY
        self._risk_prox = self.RISK_PROX
        # the best responses' GMRES warm starts are the failed attempt's last ones, wherever it wandered, and a warm-started
        # risk-averse solve stops at a fraction of its start's residual: from a wild one that is a loose answer
        self._last_gamma.clear()
        try:
            try:
                res = super().solve(**args)
            except RiskBreakdown:
                raise
            except ValueError as exc:
                start = args.get("start_from")
                if start is None:
                    start = self.zero_maps()
                elif self.init_kind(start) == "actions":
                    start = self.maps_from_actions(start)
                why = str(failed if failed is not None else exc)
                raise ValueError(f"{why}; retried with the best responses' Krylov reduction {self.RISK_KRYLOV_RETRY:g} and a proximal "
                                 f"term {self.RISK_PROX:g}: {str(exc)[:200]}; "
                                 f"at the start: {self._risk_curvature(start, averse)}") from None
        finally:
            self._risk_krylov_reduction = None
            self._risk_prox = None
        before = (f"the first attempt raised: {str(failed)[:200]}" if failed is not None else first.message)
        if first is not None:
            res.evaluations += first.evaluations
        res.message = (f"retried with the best responses' Krylov reduction {self.RISK_KRYLOV_RETRY:g} and a proximal term "
                       f"{self.RISK_PROX:g} after: {before}; " + res.message)
        if not res.converged:
            res.message += "; " + self._risk_curvature(res.maps, averse)
        return res

    def _risk_curvature(self, maps, averse) -> str:
        """The second-order probe of each risk-averse agent's entropic cost at `maps` (a best iterate that did not converge,
        or the start a best response failed from), the others' maps held: finite_free._entropic_probe along the lowest and
        highest directions of the expected cost's form (the risk-neutral second-order form of the agent's best response),
        in the closed loop of `maps`.  A negative curvature means the entropic cost has no minimum along that direction: the
        risk-averse best response is a saddle there and no fixed point settles.  Otherwise the probe finds no loss of the
        minimum (it checks one direction: it cannot prove there is none)."""
        c = self.c; out = []
        for a in averse:
            Zpass, R0 = self._spikes(c, maps, a)
            R = self._impulse_responses(a, maps, R0)
            Zpass = self._passive_world(a, maps, Zpass, R0)
            ytil, yinst = self._passive_rows(a, Zpass)
            system = finite_free.FocSystem(self, a, finite_free.RowOps(self, a, ytil, yinst), finite_free.ProjOps(self, a, ytil, yinst),
                                           finite_free.RespOps(self, a, R), finite_free.FocOps(self, a, R), Zpass, None)
            system.Zfull = self._profile_world(maps)
            idx = np.where(np.tile(self._identified(a), len(a.controls)))[0]
            if idx.size > self.SECOND_ORDER_DENSE:
                out.append(f"{a.name}: the strategy is too large for the second-order probe"); continue
            w, V = np.linalg.eigh(finite_free.symmetrize(finite_free._dense_form(self, a, system, idx)))
            pr = finite_free._entropic_probe(self, a, system, idx, V, float(w[0] / max(abs(w[0]), abs(w[-1]), 1e-300)))
            if pr.get("ok") is False:
                out.append(pr.get("message") or f"{a.name}'s entropic cost is concave along the expected cost's lowest direction "
                           f"(curvature {pr['min']:.1e} of the highest)")
                out[-1] += "; no risk-averse equilibrium settles near this iterate (lower risk_aversion)"
            else:
                out.append(f"{a.name}'s entropic cost is convex along the expected cost's lowest direction (curvature {pr['min']:.1e} of "
                           f"the highest; the expected cost's own {pr['expected_min']:.1e}): the probe finds no loss of the minimum; the "
                           "best-response iteration itself can be unstable there (res.stability() of the last equilibrium on the way: "
                           "a radius far above 1), so approach this theta in smaller steps, or raise numerics.nodes")
        return "; ".join(out)

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False, project: bool = True):
        """The agent's best response to `maps` (EngineBase.best_response's contract): finite_free.best_response,
        the operators applied and the FOC system factored within settings.foc_dense_max, solved by GMRES beyond
        (self.foc_free); the dict also carries "krylov", the GMRES iterations (0 when factored)."""
        return finite_free.best_response(self, agent, maps, want_decomp, project)
    def representation_probe(self, res) -> float:
        """The resolution row's value without the rest of the diagnostics: the largest representation error of the
        agents' best responses at res.maps (a tie group's representative stands for its group), one best response
        each (time_panels.solve_graded asks it of every grid it tries)."""
        worst = 0.0
        for a in self.model.agents:
            if self.c.rep[a.name] != a.name:
                continue
            g, out = self.best_response(a, res.maps)
            worst = max(worst, self._representation_error(a, out["Zfull"], out["action"], g))
        return worst

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        recon_all = finite_free.reconstruction(self, agent, Zfull, g)
        c = self.c; gr = c.g; worst = 0.0; by_node = None
        # where the error sits: the band's tip (the upper triangle collapsing to the corner (L, L), where the
        # map's pieces degenerate), the last window [T - L, T] (the end), or the interior, so that a resolution
        # problem can be told from the two geometric floors
        tip = gr.upper & (gr.t >= gr.bp[gr.PL - 1] - 1e-9) if gr.L is not None else np.zeros(c.N, dtype=bool)
        last = ~gr.upper & ~c.buffer & (gr.t >= c.T - gr.L - 1e-9) if gr.L is not None else np.zeros(c.N, dtype=bool)
        parts = {"interior": 0.0, "band tip": 0.0, "last window": 0.0}
        regions = [("interior", ~tip & ~last & ~c.buffer), ("band tip", tip), ("last window", last)]
        if c.cont is not None:
            parts["buffer"] = 0.0; regions.append(("buffer", c.buffer))
        actions = self._map_part(agent, Zfull, actions)
        for ui in range(len(agent.controls)):
            recon = recon_all[ui]
            err = np.abs(recon - actions[ui])
            err[:, c.nW:] = 0.0
            err[c.diag, c.nW:] = np.abs(recon - actions[ui])[c.diag, c.nW:]     # an initial shock's column: on the line s = 0 only
            rel = err.max(axis=1) / max(1e-300, np.abs(actions[ui][:, :c.nW]).max(), np.abs(actions[ui][c.diag, c.nW:]).max(initial=0.0))
            worst = max(worst, float(rel.max()))
            by_node = rel if by_node is None else np.maximum(by_node, rel)
            for key, sel in regions:
                parts[key] = max(parts[key], float(rel[sel].max(initial=0.0)))
        self._rep_parts[agent.name] = parts
        self._rep_by_node[agent.name] = by_node          # the error at every node (time_panels: which panels to split)
        return worst

    def _diagnostics(self, res) -> None:
        """The first-order-condition decomposition, the second-order check and the representation error of
        every agent's best response at the equilibrium (the stationary engine's, on this grid)."""
        self._fill_diagnostics(res)

    def _diagnostics_extra(self, res, agent) -> None:
        """With a past, where the representation error sits: interior, band tip, last window, buffer."""
        if agent.name in self._rep_parts:
            res.representation_parts[agent.name] = dict(self._rep_parts[agent.name])
