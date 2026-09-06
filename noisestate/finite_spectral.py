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
from .grid import bary_rows, cheb_lobatto
from .results import TriangleResult
from .settings import tunable
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid
from .grid_cache import triangle_grid


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
    def __init__(self, model: Model, past=None):
        super().__init__(model)
        reject_leads(model, 'spectral finite engine')
        hz = model.horizon
        self.T = float(hz.window)
        lags = model.all_lags()
        self.past = past
        if past is not None:
            past.validate(model, (hz.unit or min(lags)) if lags else None)
        L = past.window if past is not None and past.window > 0 else None
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
            if any(r[3] > 0 for rr in self.rows.values() for r in rr):
                raise NotImplementedError("a past with a window is supported for rows without observation delays: the map "
                                          "of a delayed row on the increments observed before zero has no place in the "
                                          "shifted-time storage; drift and loss lags are fine")
            bp = sorted(set(bp) | {b for b in past.breakpoints if b < L - 1e-12} | {float(L)})
            bp = close_under_delays(bp, lags) if lags else bp
            self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes, self.T, float(L))
        g = self.g
        self.N = g.N
        self.rho = float(hz.discount)
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
        self.diag = np.concatenate([pc.offset + np.arange(pc.nt) * pc.na + pc.na - 1 for pc in tri])
        self.Nd = len(self.diag)                          # time nodes on which the line s = 0 exists
        self.mean_embed = np.zeros((self.N, self.Nt))
        for pc in g.pieces:
            if pc.upper:
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
        eps = 1e-12 * max(1.0, self.T)
        return g.upper & ((g.t - lag < -eps) | ((np.abs(g.t - lag) <= eps) & (g.side_t < 0)))

    def past_read(self, name: str, lag: float) -> np.ndarray:
        """(N, nW): the past's kernel of `name` at age a - lag on the band nodes whose read `lag` earlier is before
        zero (zero elsewhere, and on pieces whose ages start below the lag, where the shock had not arrived)."""
        key = (name, round(float(lag), 12))
        if key not in self._past_reads:
            g = self.g; out = np.zeros((self.N, self.nW))
            if g.L is not None and lag > 0 and name in self.past.kernels:
                sel = self._before(lag) & (g.a0 >= lag - 1e-12)
                if sel.any():
                    out[sel] = self.past.read(name, g.a[sel] - lag)
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
        if abs(g.bp[k] - delay) > 1e-9 * max(1.0, self.T):
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
                tgt = (g._upper_by_pq if pc.upper else g._piece_by_pq)[(pc.p - k, pc.q - k)]
                tol = 1e-9 * max(1.0, self.T)
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
            else:
                op = c * (S @ self.read(l, l))
                blocks[n] = blocks[n] + op if n in blocks else op
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
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
                paths[base] = self.g.path(self.g.t, self.g.a, side_t=self.g.side_t, **kw)
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
        """(C z)(t, s) = int_t^T e^{-rho (tau - t)} R(tau, tau - t) z(tau, s) dtau."""
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.T), point_fn=lambda k, r: (r, r - g.s[k]),
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
                if pc.upper or pc.p - k < 0:
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
        """(m, N, N) discounted continuation operators of the m atom responses Rj (N, m), int_t^T e^{-rho (tau - t)} ..."""
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.T), point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-self.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        return self._many(lp, Rj, disc)

    def own_lag_read(self, lag: float) -> np.ndarray:
        """(N, N) read at (t + lag, a + lag): the FOC term of the control's own read `lag` later."""
        return self.read(-lag, -lag)

    def cost_mass(self) -> np.ndarray:
        """The discounted Gram matrix under which expected_cost integrates products of kernels."""
        return self.g.mass_matrix(rho=self.rho)

    # ------------------------------------------------------------ means
    def time_mass(self, rho: float) -> np.ndarray:
        """Weights w (Nt,) with int_0^T e^{-rho t} f(t) dt = w @ f for a path f on the time nodes: Gauss quadrature
        of the interpolant on every panel (exact for a polynomial of the panel's degree at rho = 0)."""
        key = round(float(rho), 12)
        if key not in self._time_mass:
            xg, wg = legendre.leggauss(self.g.nt + 2)
            w = np.zeros(self.Nt)
            for p in range(self.g.P):
                pc = self.g._piece_by_pq[(p, p)]
                tq = 0.5 * (pc.t1 - pc.t0) * xg + 0.5 * (pc.t1 + pc.t0); tw = 0.5 * (pc.t1 - pc.t0) * wg
                w[p * pc.nt:(p + 1) * pc.nt] = (tw * np.exp(-key * tq)) @ bary_rows(tq, pc.tn, pc.wt)
            self._time_mass[key] = w
        return self._time_mass[key]

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
                self._mean_reads[key] = self.g.interp(t, t, side_t=self.g.side_t[self.diag]) @ self.mean_embed
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

    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None, impulse_controls=()):
        """maps[agent]: (n_ctrl, n_rows, N) nodal raw maps g(t, b).  Columns: Brownian channels,
        then one impulse column per control in impulse_controls (unit mass at the shock time).
        Returns Z (n_prim N, ncol).
        With a past: maps (n_ctrl, n_rows, N + Nt), the map on the strip (the band's nodes weigh the
        increments observed before zero) then the discrete weights on the initial shocks' point
        observations; the columns are the channels, the initial shocks, then the impulses.  The
        band's forcing is the past: the state at zero, the lagged atoms read before zero, the row's
        pre-zero increments under the map (past_conv_path) with the old noise loadings."""
        if self.past is not None:
            return self._closed_loop_past(maps, excluded, impulse_controls)
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

    def _closed_loop_past(self, maps, excluded, impulse_controls):
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
            if a.name == excluded:
                continue
            gm = maps[a.name]
            for ui, u in enumerate(a.controls):
                bl = self.block(u)
                for r, (rname, drift, E, delay) in enumerate(self.rows[a.name]):
                    blocks, deltas = self.row_blocks(a.name, r, excl)
                    gker = gm[ui, r][:N]
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
                    if self.n_init:
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
            K0 = np.stack([self.past.read(self.prim[j], g.a[up] - g.t[up]) for j in range(self.nX)], axis=1)   # (n_up, nX, nW)
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

    def __init__(self, model: Model, verbose: bool = False, settings=None):
        """The triangle grid's compiled model and the map shapes (nU, nR, N).  settings: the tuning constants
        (noisestate.Settings, or a dict of its fields; the defaults when None)."""
        super().__init__(model, verbose, settings=settings)
        self.c = SpectralCompiled(model)
        self.shapes = {a.name: (len(a.controls), len(a.signals), self.c.N) for a in model.agents}

    # ------------------------------------------------ best-response pieces
    def _identified(self, agent: Agent) -> np.ndarray:
        """The map on a row observed with delay d is stored at the shifted time t' = t - d, so its nodes on
        the time panels above T - d belong to controls after the horizon and are read by nothing: those
        entries are removed from every solve and left at zero.  (The nodes are whole panels, so nothing is
        masked inside a piece; with a mask cutting through a piece the interpolant of the map between the
        kept nodes and the zeroed ones is meaningless, and the delayed rows were not exact.)"""
        c = self.c; g = c.g; N = c.N
        keep = np.ones(len(agent.signals) * N, dtype=bool)
        for r in range(len(agent.signals)):
            d = c.rows[agent.name][r][3]
            if d > 0:
                keep[r * N:(r + 1) * N] = c.panel_of_node + c.panel_shift(d) < g.P
        return keep

    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        """The system on the kept unknowns, solved directly; a singular one raises (see _solve_regular)."""
        keep = np.tile(self._identified(agent), len(agent.controls))
        gamma = np.zeros(Amat.shape[0])
        gamma[keep] = self._solve_regular(agent, Amat[np.ix_(keep, keep)], -bvec[keep])
        return gamma

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

    def world_from_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Closed-loop primary kernels when every agent's action kernels are given."""
        c = self.c; N, nW = c.N, c.nW
        n = len(c.prim) * N
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
        G = np.einsum("ink,nm,jmk->ij", zeta, c.cost_mass(), zeta)
        return float(0.5 * np.sum(Q * G))

    def interpolate_maps(self, coarse) -> Dict[str, np.ndarray]:
        """The coarse result's raw maps read at this triangle's nodes from each node's side of its piece."""
        g, gc = self.c.g, coarse.compiled.g
        I = gc.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a)
        return {a.name: np.einsum("fn,urn->urf", I, coarse.maps[a.name]) for a in self.model.agents}

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
        the shocks.  Linear in (q, x0, const): one direct solve, no iteration."""
        c = self.c; N, Nt, nP, nX, nW = c.N, c.Nt, len(c.prim), c.nX, c.nW
        E, diag = c.mean_embed, c.diag
        blk = lambda i: slice(i * Nt, (i + 1) * Nt)
        M = np.zeros((nP * Nt, nP * Nt)); b = np.zeros(nP * Nt); ones = np.ones(N)
        if nX:
            blocks, B0 = c._state_part(None, set(), [])
            for i in range(nX):
                M[blk(i), blk(i)] = np.eye(Nt)
            for (i, p), B in blocks.items():
                M[blk(i), blk(p)] -= B[diag] @ E
            EA = c.expA(c.tm)
            for i in range(nX):
                b[blk(i)] = EA[:, i, :] @ c.x0 + sum((c.const[j] * (c.Vol[i, j][diag] @ ones) for j in range(nX) if c.const[j]), 0.0)
        for a in self.model.agents:
            atoms, Q, q = c.loss[a.name]
            R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls)[:, nW:]
            R = self._impulse_responses(a, maps, R)
            Fu, Ms = self._foc_operators(a, R, atoms=True)
            for ui, u in enumerate(a.controls):
                row = blk(c.index[u])
                Fd = Fu[ui][diag].reshape(Nt, nP, N)
                for p in range(nP):
                    M[row, blk(p)] = Fd[:, p, :] @ E
                b[row] = -sum((q[j] * (Ms[ui][j][diag] @ ones) for j in range(len(atoms)) if q[j]), 0.0)
        return M, b

    def solve_means(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """The mean paths of the primaries (states then controls, Nt values each) under `maps`: exactly zero,
        with no solve, when nothing drives them (every q zero, no constant drift, no initial state); else
        the direct solve of mean_system, refusing a singular system."""
        c = self.c
        if not (c.x0.any() or c.const.any() or any(q.any() for atoms, Q, q in c.loss.values())):
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
        Zm = np.concatenate([c.mean_embed @ zbar[p * Nt:(p + 1) * Nt] for p in range(len(c.prim))])
        return np.stack([(c.atom_op(at) @ Zm)[c.diag] for at in atoms])

    def mean_cost(self, agent: Agent, zbar: np.ndarray) -> float:
        """The mean part of the agent's discounted cost, int_0^T e^{-rho t} (1/2 zbar'Q zbar + q'zbar) dt over its
        loss atoms at the mean paths `zbar` (the constant of a target, theta^2, is not in the model): spectral
        quadrature on the time panels."""
        atoms, Q, q = self.c.loss[agent.name]
        zeta = self._mean_atoms(zbar, atoms); w = self.c.time_mass(self.c.rho)
        return float(0.5 * np.einsum("it,ij,jt,t->", zeta, Q, zeta, w) + q @ (zeta @ w))

    def _mean_part(self, res) -> None:
        """res.means: the path on the time nodes res.means_t of every state, control and definition and of the
        mean drift rate of every signal row ("agent.row", at the time of the observation, its delay not applied);
        res.cost_parts and the mean part added to res.costs."""
        c = self.c; m = self.model; Nt = c.Nt
        zbar = self.solve_means(res.maps)
        blk = lambda nm: slice(c.index[nm] * Nt, (c.index[nm] + 1) * Nt)
        value = lambda expr: sum((coef * (c.mean_read(lag) @ zbar[blk(nm)]) for (nm, lag), coef in expr.items()), np.zeros(Nt))
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
        self._loss_forms.clear()                  # the second-order check is done: its (n_prim N)^2 form is not kept

