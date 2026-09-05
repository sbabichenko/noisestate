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
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from functools import cached_property

import numpy as np

from .engine import EngineBase
from .compile import CompiledBase, close_under_delays, reject_leads
from .results import TriangleResult
from .spec import Agent, Atom, Model
from .triangle import TriangleGrid
from .grid_cache import triangle_grid


class SpectralCompiled(CompiledBase):
    def __init__(self, model: Model):
        super().__init__(model)
        reject_leads(model, 'spectral finite engine')
        hz = model.horizon
        self.T = float(hz.window)
        lags = model.all_lags()
        bp = list(hz.breakpoints) if hz.breakpoints else TriangleGrid.breakpoints(self.T, lags, hz.unit)
        for l in lags:                   # explicit breakpoints must contain every lag (checked before the closure below)
            if not any(abs(l - b) < 1e-12 for b in bp):
                raise ValueError(f"lag {l} is not a breakpoint of the time/age partition {bp}; set horizon.unit "
                                 "so that every lag is a multiple of it")
        delays = sorted({float(r.delay) for a in model.agents for r in a.signals if r.delay > 0})
        if delays:                       # pieces closed under the row delays: the delay line must run along piece edges
            bp = close_under_delays(bp, delays)
        for l in lags:
            if not any(abs(l - b) < 1e-9 * max(1.0, self.T) for b in bp):
                raise ValueError(f"lag {l} is not a breakpoint of the time/age partition {bp}; set horizon.unit "
                                 "so that every lag is a multiple of it")
        self.g = triangle_grid(tuple(round(float(b), 12) for b in bp), hz.nodes, hz.nodes)   # shared
        g = self.g
        self.N = g.N
        self.rho = float(hz.discount)
        # state propagation operators (matrix exponentials of A)
        if self.nX:
            # state propagation e^{A(t-r)} along the Volterra path; entrywise weights from expm, which
            # is exact for defective A too (an eigen-decomposition would not be)
            lp = g.path(g.t, g.a, r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]))
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
        self._map_shifts: Dict[float, np.ndarray] = {}
        self._row_ops: Dict[tuple, tuple] = {}            # (agent, row, excluded) -> row_op result (map-independent)
        self._state_parts: Dict[tuple, tuple] = {}        # (excluded, impulses) -> state part of the closed loop

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
            M = g.interp(g.t - dt, g.a - da, side_t=g.side_t, side_a=g.side_a)
            if da > 0:
                M[g.a0 < da - 1e-12] = 0.0
        cache[key] = M
        return M

    def panel_shift(self, delay: float) -> int:
        """Number of time panels in a delay: the breakpoints are closed under the delays, so every panel
        shifted by a delay is again a panel."""
        g = self.g
        k = int(np.sum((g.bp > 1e-12) & (g.bp <= delay + 1e-12)))
        if abs(g.bp[k] - delay) > 1e-9 * max(1.0, self.T):
            raise ValueError(f"delay {delay} is not a breakpoint of {g.bp}")
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
                tgt = g._piece_by_pq[(pc.p - k, pc.q - k)]
                assert abs(tgt.t0 + delay - pc.t0) < 1e-9 and abs(tgt.t1 + delay - pc.t1) < 1e-9, "panels are not closed under the delay"
                S[pc.offset + np.arange(pc.n), tgt.offset + np.arange(pc.n)] = 1.0
            self._map_shifts[key] = S
        return self._map_shifts[key]

    def block(self, name: str) -> slice:
        i = self.index[name]
        return slice(i * self.N, (i + 1) * self.N)

    def atom_op(self, atom: Atom) -> np.ndarray:
        name, lag = atom
        M = np.zeros((self.N, len(self.prim) * self.N))
        M[:, self.block(name)] = self.read(lag, lag)
        return M

    def expr_op(self, expr) -> np.ndarray:
        M = np.zeros((self.N, len(self.prim) * self.N))
        for (name, lag), c in expr.items():
            M[:, self.block(name)] += c * self.read(lag, lag)
        return M

    def row_op(self, agent: str, r: int, excluded: set):
        """Seen row r of `agent` as (regular operator on Z, {source: [(age, weight)]}),
        the regular part already shifted by the observation delay.  Map-independent, so cached per
        (agent, row, excluded controls); the operator is shared and must not be written to."""
        key = (agent, r, frozenset(excluded))
        if key not in self._row_ops:
            reg, deltas = self._row_op(agent, r, excluded)
            self._row_ops[key] = (reg, deltas, self._nonzero_blocks(reg))
        reg, deltas, _ = self._row_ops[key]
        return reg, {k: list(v) for k, v in deltas.items()}

    def _nonzero_blocks(self, reg: np.ndarray):
        """The primaries whose N x N block of the row operator is not identically zero."""
        return [p for p in range(len(self.prim)) if np.any(reg[:, p * self.N:(p + 1) * self.N])]

    def _row_op(self, agent: str, r: int, excluded: set):
        name, drift, E, delay = self.rows[agent][r]
        S = self.read(delay, delay)
        reg = np.zeros((self.N, len(self.prim) * self.N))
        deltas: Dict[str, List[Tuple[float, float]]] = {}
        for (n, l), c in drift.items():
            if n in excluded:
                deltas.setdefault(n, []).append((delay + l, c))
            else:
                reg[:, self.block(n)] += c * (S @ self.read(l, l))
        for k, ch in enumerate(self.channels):
            if E[k] != 0.0:
                deltas.setdefault(ch, []).append((delay, E[k]))
        return reg, deltas

    # ------------------------------------------------------ line operators
    # Each family of line integrals is a cached quadrature structure (triangle.LinePath);
    # an operator for a given known kernel is then two sparse products.
    @cached_property
    def _panel_idx(self):
        """Indices of every primary's unknowns on each time panel, in block order."""
        panel = np.concatenate([np.full(pc.n, pc.p) for pc in self.g.pieces])
        return [np.concatenate([q * self.N + np.where(panel == p)[0] for q in range(len(self.prim))]) for p in range(self.g.P)]

    def _path(self, key, **kw):
        if key not in self.g.paths:
            self.g.paths[key] = self.g.path(self.g.t, self.g.a, side_t=self.g.side_t, **kw)
        return self.g.paths[key]

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

    # ------------------------------------------------ kernel algebra (see EngineBase)
    def _many(self, lp, K: np.ndarray, extra=None) -> np.ndarray:
        """with_known over the nonzero columns of K (N, m) at once: (m, N, N), zero for a zero column."""
        out = np.zeros((K.shape[1], self.N, self.N))
        nz = [k for k in range(K.shape[1]) if np.abs(K[:, k]).max() > 0]
        if nz:
            out[nz] = lp.with_known_many(K[:, nz], extra)
        return out

    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        g = self.g
        lp = self._path(("conv_right", delay), r_lo=g.s + delay, r_hi=g.t,
                        point_fn=lambda k, r: (np.full_like(r, g.t[k] - delay), g.t[k] - r), known_fn=lambda k, r: (r, r - g.s[k]))
        return self._many(lp, Y)

    def instant(self, age: float, delay: float = 0.0) -> np.ndarray:
        if delay > 0 and abs(age - delay) < 1e-12:
            return self.map_shift(delay)
        return self.read(delay, age)

    def instant_adjoint(self, age: float, delay: float = 0.0) -> np.ndarray:
        if delay > 0 and abs(age - delay) < 1e-12:
            return self.map_shift(delay).T
        return self.read(-delay, -age)

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        g = self.g; N = self.N
        lp = self._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        K = Ru.T.copy(); K[:, own] = 0.0
        return self._many(lp, K).reshape(len(self.prim) * N, N)

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        g = self.g
        lp = self._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, self.T), point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-self.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        return self._many(lp, Rj, disc)

    def own_lag_read(self, lag: float) -> np.ndarray:
        return self.read(-lag, -lag)

    def cost_mass(self) -> np.ndarray:
        """The discounted Gram matrix under which expected_cost integrates products of kernels."""
        return self.g.mass_matrix(rho=self.rho)

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
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
        Returns Z (n_prim N, ncol)."""
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
                    reg, deltas = self.row_op(a.name, r, excl)
                    nzb = self._row_ops[(a.name, r, frozenset(excl))][2]
                    gker = gm[ui, r]
                    C = self.conv_left(gker, delay)
                    for p in nzb:                                       # the zero blocks of reg contribute nothing
                        M[bl, p * N:(p + 1) * N] += C @ reg[:, p * N:(p + 1) * N]
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

    def _state_part(self, excluded, excl: set, imp: list):
        """The state rows of the closed-loop system without the maps: the nonzero N x N blocks of the
        Volterra propagation of the state inputs, {(state, primary): block}, and the shock and impulse
        columns B0 (n, ncol).  Map-independent, cached per (excluded agent, impulse controls)."""
        key = (excluded, tuple(imp))
        if key in self._state_parts:
            return self._state_parts[key]
        g = self.g; N = self.N; nW = self.nW; n = len(self.prim) * N
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
    MAP_RIDGE = 1e-13      # ridge of the per-time-row map projection, relative to the row's own Gram

    def __init__(self, model: Model, verbose: bool = False):
        super().__init__(model, verbose)
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
        keep = np.tile(self._identified(agent), len(agent.controls))
        gamma = np.zeros(Amat.shape[0])
        A = Amat[np.ix_(keep, keep)]
        try:
            gamma[keep] = np.linalg.solve(A, -bvec[keep])
        except np.linalg.LinAlgError:
            gamma[keep] = np.linalg.lstsq(A, -bvec[keep], rcond=1e-10)[0]
        return gamma

    def _project(self, agent: Agent, Zfull: np.ndarray, cact: np.ndarray) -> np.ndarray:
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
        Z = self.world_from_actions(actions)
        return {a.name: self.maps_from_world(a, Z, actions[a.name]) for a in self.model.agents}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        c = self.c
        atoms, Q, q = c.loss[agent.name]
        zeta = np.stack([c.atom_op(at) @ Z for at in atoms])                  # (m, N, nW)
        G = np.einsum("ink,nm,jmk->ij", zeta, c.cost_mass(), zeta)
        return float(0.5 * np.sum(Q * G))

    def _finish(self, res) -> None:
        self._second_order_cache.clear()
        order = [a for a in self.model.agents if self.c.rep[a.name] == a.name] + [a for a in self.model.agents if self.c.rep[a.name] != a.name]
        for a in order:
            res.costs[a.name] = self.expected_cost(a, res.Z)
            g, out = self.best_response(a, res.maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            if out["second_order"] is not None:
                res.second_order[a.name] = out["second_order"]
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)

