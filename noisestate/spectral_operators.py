"""The best-response operators of the spectral finite engine as applications of the line paths and the sparse reads.

Every operator is an application built from the line paths, PathOp: R diag(w J y) I on the path's
quadrature points with the known kernel y read once, applied to the columns of a world at a time, and
from the compiled model's sparse reads (spectral_compiled.py: read_sparse, map_shift_sparse,
row_blocks_sparse, mass_sparse): RowOps (the map on the seen rows -> the action kernels, G_k), RespOps
(an action kernel -> the world, Resp_u), FocOps (a world -> the FOC kernels, Fu_u) and ProjOps (a FOC
kernel -> its projection on the seen rows, H_k); with a past the band's read of the past's increments, the
old-shock and pre-zero segments and the initial shocks' discrete weights and point conditions are segments
of the same operators.  PanelRows serves the row operator's rows one time panel at a time (the per-time-row
Gram of maps_from_world in finite_spectral.py and the preconditioner).  Each operator has dense() (its rows,
the same sums as its application) for the factored first-order-condition system of finite_free.py, which
assembles and solves the system on them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .spec import Agent
from .triangle import _scratch, _scratch_owner


class PathOp:
    """The line-integral operators of one path (triangle.LinePath) for m known kernels K (N_known, m):
    op_j = R diag(w F_j) I with F = J K the knowns read at the quadrature points, as applications on
    vectors (columns of a world, a block of them) and as dense rows of a panel of output nodes; extra
    scales every point (the continuation's discount).  A path with no points (lp.rows None) is the zero
    operator (empty).  Where the path holds I as Factors, R diag(w F_j) is folded onto I's distinct points
    (a CSR matrix per j on LinePath.distinct_layout, data w F_j): unknown() then gives the unknown at the
    distinct points and the applications are those sparse products, the same products and sums as R (w F_j * I V)
    without taking every point's value."""

    def __init__(self, lp, K: np.ndarray, extra: Optional[np.ndarray] = None):
        self.lp = lp
        K = np.asarray(K, dtype=float)
        K = K if K.ndim == 2 else K[:, None]
        self.m = K.shape[1]
        self.n_out, self.N = lp.n_out, lp.N
        self._ops = None
        if lp.rows is None:
            self.F = None
        else:
            F = lp.read(K) if lp.Jf is not None else lp.J @ K
            if extra is not None:
                F = F * extra[:, None]
            self.FT = np.ascontiguousarray((lp.w[:, None] * F).T)        # (m, nq): each op's weights contiguous
            self.F = self.FT.T                                          # (nq, m): the weights included
            layout = lp.distinct_layout()
            if layout is not None:
                from scipy.sparse import csr_matrix
                idx, indptr, ncol = layout
                self._ops = [csr_matrix((self.FT[j], idx, indptr), shape=(self.n_out, ncol), copy=False) for j in range(self.m)]

    @property
    def empty(self) -> bool:
        return self.F is None

    def unknown(self, V: np.ndarray, scratch: bool = False) -> np.ndarray:
        """The unknown V (N, ...) read for the applications: at the distinct points (LinePath.read_distinct) where the
        operators are folded onto them, else I V at every quadrature point (nq, ...).  Through the interpolation
        factors (the same sums as the sparse I, 4 to 6 times faster).  scratch=True writes it into a work buffer
        (triangle._scratch), for a caller that applies it at once."""
        if self._ops is not None:
            return self.lp.read_distinct(V, scratch)
        return self.lp.read_unknown(V, scratch)

    def apply(self, IV: np.ndarray, j: int) -> np.ndarray:
        """op_j V given IV = unknown(V)."""
        if self._ops is not None:
            return self._ops[j] @ IV
        Fj = self.F[:, j]
        return self.lp.R @ np.multiply(Fj if IV.ndim == 1 else Fj[:, None], IV, out=_scratch("path_apply", IV.shape))

    def apply_all(self, IV: np.ndarray) -> np.ndarray:
        """Every op_j on the same vectors: (n_out, m, B) for IV = unknown(V), V (N, B)."""
        B = IV.shape[1]
        if self._ops is not None:
            out = np.empty((self.n_out, self.m, B))
            for j, op in enumerate(self._ops):
                out[:, j] = op @ IV
            return out
        FV = np.multiply(self.F[:, :, None], IV[:, None, :], out=_scratch("path_apply", (IV.shape[0], self.m, B)))
        return (self.lp.R @ FV.reshape(-1, self.m * B)).reshape(self.n_out, self.m, B)

    def apply_sum(self, IV: np.ndarray) -> np.ndarray:
        """sum_j op_j V[:, j] for IV = unknown(V), V (N, m, B): (n_out, B)."""
        if self._ops is not None:
            out = self._ops[0] @ IV[:, 0]
            for j in range(1, self.m):
                out += self._ops[j] @ IV[:, j]
            return out
        FV = np.multiply(self.F[:, :, None], IV, out=_scratch("path_apply", np.broadcast_shapes(self.F[:, :, None].shape, IV.shape)))
        return self.lp.R @ FV.sum(axis=1)

    def rows(self, j: int, lo: int, hi: int, cols=None) -> np.ndarray:
        """Rows [lo, hi) of op_j, dense (hi - lo, N) (or their columns [c0, c1) = cols): the same sums as
        LinePath.with_known(rows=)."""
        if self.F is None:
            return np.zeros((hi - lo, self.N if cols is None else cols[1] - cols[0]))
        return self.lp._weighted_sum(self.FT[j], rows=(lo, hi), cols=cols)

    def adjoint_sum(self, W: np.ndarray) -> np.ndarray:
        """sum_j op_j^T W[:, j] for W (n_out, m, B): I^T (sum_j F_j (R^T W)_j), (N, B)."""
        return self.lp.I.T @ (self.F[:, :, None] * W[self.lp.rows]).sum(axis=1)

    def adjoint(self, W: np.ndarray, j: int) -> np.ndarray:
        """op_j^T W for W (n_out, ...)."""
        Fj = self.F[:, j]
        RW = W[self.lp.rows]
        return self.lp.I.T @ (Fj * RW if RW.ndim == 1 else Fj[:, None] * RW)


def _entries(S, r0: int, r1: int, c0: int, c1: int):
    """The nonzero entries of the block S[r0:r1, c0:c1] of a sparse read or shift, (rows - r0, cols - c0, values), from
    S's canonical CSR (duplicates summed as toarray sums them; made once and kept on the matrix, which the compiled
    model caches): a block's dense copy is mostly zeros, and adding it adds nothing but its entries."""
    hit = getattr(S, "_ns_canonical", None)
    if hit is None:
        C = S.tocsr(copy=True); C.sum_duplicates()
        hit = S._ns_canonical = (None, C.indptr, C.indices, C.data)
    _, indptr, indices, data = hit
    a, b = int(indptr[r0]), int(indptr[r1])
    rows = np.repeat(np.arange(r0, r1), np.diff(indptr[r0:r1 + 1]))
    cols = indices[a:b]
    keep = (cols >= c0) & (cols < c1)
    return rows[keep] - r0, cols[keep] - c0, data[a:b][keep]


def _zeros(shape, scratch: Optional[str] = None) -> np.ndarray:
    """np.zeros(shape), or with a name the zeroed work buffer of that name (triangle._scratch): the dense operators the
    factored first-order-condition system is assembled from are made and dropped once per best response, at sizes
    glibc maps and faults in afresh on every call."""
    if scratch is None:
        return np.zeros(shape)
    out = _scratch(scratch, shape); out.fill(0.0)
    return out


def _nonzero_cols(Y: np.ndarray) -> List[int]:
    return [k for k in range(Y.shape[1]) if np.abs(Y[:, k]).max() > 0]


def _batched(X: np.ndarray, nd: int):
    """X with a trailing batch axis (added when X has nd axes), and whether it was added."""
    if X.ndim == nd:
        return X[..., None], True
    return X, False


class RowOps:
    """The row operator G of an agent as applications: the map on the seen rows -> the action kernel on every
    column of the world (the channels, then a past's initial shocks), c_k = sum_r (Conv[y_rk] + inst) gamma_r
    (EngineBase._row_operator, SpectralFiniteSolver._row_operator with a past: the band's read of the past's
    increments and the discrete weights on the initial shocks included), from the rows and instantaneous
    entries of _seen_rows."""

    def __init__(self, solver, agent: Agent, rows, inst):
        c = solver.c; g = c.g; self.c = c; self.solver = solver; self.agent = agent
        self.N, self.nW, self.ncol, self.Nm = c.N, c.nW, c.ncol, solver.Nm
        self.nR = len(rows); self.past = c.past is not None; self.band = self.past and g.L is not None
        self.conv: List[Optional[Tuple[List[int], PathOp]]] = []       # per row: (its nonzero channels, the convolution path)
        self.pconv: List[Optional[Tuple[List[int], PathOp]]] = []
        self.inst: List[List[Tuple[int, np.ndarray, object]]] = []      # per row: (channel, weight per node, sparse shift)
        self.embed: List[List[Tuple[int, float, np.ndarray]]] = []      # per row: (initial shock, loading, disc_embed)
        for r in range(self.nR):
            d = c.rows[agent.name][r][3]
            sup = _nonzero_cols(rows[r])                 # every column of the world the row carries, the initial shocks' too
            lp = c._path(("conv_right", d), r_lo=g.s + d, r_hi=g.t,
                         point_fn=lambda k, r_, d=d: (np.full_like(r_, g.t[k] - d), g.t[k] - r_), known_fn=lambda k, r_: (r_, r_ - g.s[k]))
            self.conv.append((sup, PathOp(lp, rows[r][:, sup])) if sup else None)
            pc = None
            if self.band:
                Kp = c.past_row_kernel(agent.name, r)
                ks = _nonzero_cols(Kp)
                if ks:
                    pc = (ks, PathOp(c.past_conv_path(), Kp[:, ks]))
            self.pconv.append(pc)
            ent = []
            for (k, age, w) in inst[r]:
                nw = c.noise_weight(agent.name, r, k, w) if self.past else np.full(self.N, float(w))
                ent.append((k, nw, c.instant_sparse(age, d)))
            self.inst.append(ent)
            em = []
            if c.n_init:
                for i in range(c.n_init):
                    e = c.init_rows[agent.name][r, i]
                    if e:
                        em.append((i, float(e), c.disc_embed(d)))
            self.embed.append(em)

    def _paths(self, r: int):
        return [ops for ops in (self.conv[r], self.pconv[r]) if ops is not None and not ops[1].empty]

    def apply(self, gamma: np.ndarray) -> np.ndarray:
        """The action kernels (N, ncol) of the map gamma (nR, Nm) (or (nR Nm,)); a block of maps (nR, Nm, B)
        gives (N, ncol, B)."""
        single = gamma.ndim < 3
        gamma = gamma.reshape(self.nR, self.Nm, -1)
        B = gamma.shape[2]
        out = np.zeros((self.N, self.ncol, B))
        for r in range(self.nR):
            gr = gamma[r, :self.N]
            for (ks, op) in self._paths(r):
                out[:, ks] += op.apply_all(op.unknown(gr, scratch=True))
            for (k, nw, S) in self.inst[r]:
                out[:, k] += nw[:, None] * (S @ gr)
            for (i, e, E) in self.embed[r]:
                out[:, self.nW + i] += e * (E @ gamma[r, self.N:])
        return out[..., 0] if single else out

    def adjoint(self, C: np.ndarray) -> np.ndarray:
        """sum_k G_k^T C[:, k]: (nR, Nm) for C (N, ncol), (nR, Nm, B) for a block (N, ncol, B)."""
        C, single = _batched(C, 2)
        out = np.zeros((self.nR, self.Nm, C.shape[2]))
        for r in range(self.nR):
            for (ks, op) in self._paths(r):
                out[r, :self.N] += op.adjoint_sum(C[:, ks])
            for (k, nw, S) in self.inst[r]:
                out[r, :self.N] += S.T @ (nw[:, None] * C[:, k])
            for (i, e, E) in self.embed[r]:
                out[r, self.N:] += e * (E.T @ C[:, self.nW + i])
        return out[..., 0] if single else out

    def rows(self, lo: int, hi: int, ranges: List[Tuple[int, int]], into=None):
        """The rows [lo, hi) (a panel's action nodes) of every G_k restricted to the map columns of each row r in
        ranges[r] = (lo_r, hi_r) plus its discrete weights: a list per row of (flow (ncol, hi - lo, hi_r - lo_r),
        disc (ncol, hi - lo, Nt) or None).  into: per row (flow, disc) arrays of those shapes, zero, to fill in
        place of new ones (PanelRows' block)."""
        out = []
        for r in range(self.nR):
            lo_r, hi_r = ranges[r]
            flow = np.zeros((self.ncol, hi - lo, hi_r - lo_r)) if into is None else into[r][0]
            for (ks, op) in self._paths(r):
                for j, k in enumerate(ks):
                    flow[k] += op.rows(j, lo, hi, (lo_r, hi_r))
            for (k, nw, S) in self.inst[r]:
                ri, ci, v = _entries(S, lo, hi, lo_r, hi_r)                 # nw[lo:hi, None] * S[lo:hi, lo_r:hi_r], its entries
                flow[k][ri, ci] += nw[lo + ri] * v
            disc = None
            if self.Nm > self.N:
                disc = np.zeros((self.ncol, hi - lo, self.Nm - self.N)) if into is None else into[r][1]
                for (i, e, E) in self.embed[r]:
                    disc[self.nW + i] += e * E[lo:hi]
            out.append((flow, disc))
        return out


    def sparse(self) -> list:
        """Every G_k as a CSR matrix (N, nR Nm), a list over the world's columns: dense() one time panel of rows at a
        time, kept by its nonzero entries (an action node reads the map on its own time row only: 0.4% of dense()'s
        entries on a seven-panel grid), so the dense array never exists."""
        from scipy.sparse import csr_matrix, vstack
        blocks = [[] for _ in range(self.ncol)]
        for lo, hi in self.c._panel_ranges:
            out = np.zeros((self.ncol, hi - lo, self.nR * self.Nm))
            into = [(out[:, :, r * self.Nm:r * self.Nm + self.N],
                     out[:, :, r * self.Nm + self.N:(r + 1) * self.Nm] if self.Nm > self.N else None) for r in range(self.nR)]
            self.rows(lo, hi, [(0, self.N)] * self.nR, into=into)
            for k in range(self.ncol):
                blocks[k].append(csr_matrix(out[k]))
        return [vstack(b, format="csr") for b in blocks]

    def dense(self, lo: int = 0, scratch: Optional[str] = None) -> np.ndarray:
        """Every G_k as a dense array, (ncol, N, nR Nm): the rows of every panel (the factored FOC system); with lo
        the rows of the action nodes from lo on only (the reduced system under freeze_before), zero before.  scratch:
        the name of a work buffer to build it in (_zeros), for a caller that drops it before the next build."""
        out = _zeros((self.ncol, self.N, self.nR * self.Nm), scratch)
        into = [(out[:, lo:, r * self.Nm + lo:r * self.Nm + self.N],                  # the rows written in place
                 out[:, lo:, r * self.Nm + self.N:(r + 1) * self.Nm] if self.Nm > self.N else None) for r in range(self.nR)]
        self.rows(lo, self.N, [(lo, self.N)] * self.nR, into=into)
        return out


class ProjOps:
    """The projection operator H of an agent as an application: the first-order-condition kernel phi (N, ncol)
    -> E[phi_t dY_r(t - b)] at every map node of every row, (nR, Nm) (EngineBase._projection_operator and
    SpectralFiniteSolver's with a past: the old-shock and pre-zero segments of the band, the point conditions
    of the initial shocks and their discrete weights)."""

    def __init__(self, solver, agent: Agent, rows, inst):
        c = solver.c; g = c.g; self.c = c
        self.N, self.nW, self.ncol, self.Nm = c.N, c.nW, c.ncol, solver.Nm
        self.nR = len(rows); self.past = c.past is not None; self.band = self.past and g.L is not None
        self.paths: List[List[Tuple[List[int], PathOp]]] = []           # per row: the regular, old-shock and pre-zero paths
        self.inst: List[List[Tuple[int, np.ndarray, object]]] = []
        self.init: List[List[Tuple[int, np.ndarray, object]]] = []       # per row: (initial shock, y_i on the shock, diag read)
        self.select: List[List[Tuple[int, float, np.ndarray]]] = []      # per row: (initial shock, loading, disc_select)
        u = g.s
        for r in range(self.nR):
            d = c.rows[agent.name][r][3]
            Y = rows[r][:, :self.nW]
            sup = _nonzero_cols(Y)
            raw = Y[:, sup]
            if d and sup:
                raw = c.read_sparse(-d, -d) @ raw
            paths = []
            if sup:
                lp = c._path(("projection", d), r_lo=np.zeros(self.N), r_hi=u,
                             point_fn=lambda k, r_, d=d: (np.full_like(r_, g.t[k] + d), g.t[k] + d - r_),
                             known_fn=lambda k, r_: (np.full_like(r_, u[k]), u[k] - r_))
                paths.append((sup, PathOp(lp, raw)))
                if self.band:
                    paths.append((sup, PathOp(c.old_shock_proj_path(), raw)))
            if self.band:
                Kp = c.past_row_kernel(agent.name, r)
                ks = _nonzero_cols(Kp)
                if ks:
                    paths.append((ks, PathOp(c.past_proj_path(), Kp[:, ks])))
            self.paths.append([ops for ops in paths if not ops[1].empty])
            ent = []
            for (k, age, w) in inst[r]:
                nw = c.noise_weight(agent.name, r, k, w) if self.past else np.full(self.N, float(w))
                ent.append((k, nw, c.instant_sparse(age, d)))
            self.inst.append(ent)
            ini, sel = [], []
            if c.n_init:
                Id = c.diag_read_sparse(d); St = c.shock_time_read_sparse()
                for i in range(c.n_init):
                    yi = St @ rows[r][:, self.nW + i]
                    if np.any(yi):
                        ini.append((i, yi, Id))
                    e = c.init_rows[agent.name][r, i]
                    if e:
                        sel.append((i, float(e), c.disc_select(d)))
            self.init.append(ini); self.select.append(sel)

    def apply(self, phi: np.ndarray) -> np.ndarray:
        """sum_k H_k phi[:, k]: (nR, Nm) for phi (N, ncol); (nR, Nm, B) for a block (N, ncol, B)."""
        phi, single = _batched(phi, 2)
        out = np.zeros((self.nR, self.Nm, phi.shape[2]))
        for r in range(self.nR):
            for (ks, op) in self.paths[r]:
                out[r, :self.N] += op.apply_sum(op.unknown(phi[:, ks].reshape(self.N, -1), scratch=True).reshape(-1, len(ks), phi.shape[2]))
            for (k, nw, S) in self.inst[r]:
                out[r, :self.N] += S.T @ (nw[:, None] * phi[:, k])
            for (i, yi, Id) in self.init[r]:
                out[r, :self.N] += yi[:, None] * (Id @ phi[:, self.nW + i])
            for (i, e, Sel) in self.select[r]:
                out[r, self.N:] += e * (Sel @ phi[:, self.nW + i])
        return out[..., 0] if single else out

    def rows(self, r: int, lo_r: int, hi_r: int, lo: int, hi: int):
        """Rows [lo_r, hi_r) (map nodes of row r) of every H_k restricted to the columns [lo, hi) (a panel's
        action nodes): flow (ncol, hi_r - lo_r, hi - lo) and the discrete weights' rows disc (ncol, Nt, hi - lo)
        or None."""
        flow = np.zeros((self.ncol, hi_r - lo_r, hi - lo))
        for (ks, op) in self.paths[r]:
            for j, k in enumerate(ks):
                flow[k] += op.rows(j, lo_r, hi_r, (lo, hi))
        for (k, nw, S) in self.inst[r]:
            ri, ci, v = _entries(S, lo, hi, lo_r, hi_r)                     # S[lo:hi, lo_r:hi_r]' * nw[lo:hi], its entries
            flow[k][ci, ri] += v * nw[lo + ri]
        for (i, yi, Id) in self.init[r]:
            ri, ci, v = _entries(Id, lo_r, hi_r, lo, hi)                     # yi[lo_r:hi_r, None] * Id[lo_r:hi_r, lo:hi]
            flow[self.nW + i][ri, ci] += yi[lo_r + ri] * v
        disc = None
        if self.Nm > self.N:
            disc = np.zeros((self.ncol, self.Nm - self.N, hi - lo))
            for (i, e, Sel) in self.select[r]:
                disc[self.nW + i] += e * Sel[:, lo:hi]
        return flow, disc


    def dense(self, lo: int = 0, scratch: Optional[str] = None) -> np.ndarray:
        """H as a dense array (nR Nm, ncol N), columns (channel, node): every row's map nodes against every
        action node (the factored FOC system); with lo the map nodes and action nodes from lo on only (the
        reduced system under freeze_before), zero before.  scratch: a work buffer's name (_zeros)."""
        N, Nm, ncol = self.N, self.Nm, self.ncol
        out = _zeros((self.nR * Nm, ncol * N), scratch)
        for r in range(self.nR):
            flow, disc = self.rows(r, lo, N, lo, N)
            for k in range(ncol):
                out[r * Nm + lo:r * Nm + N, k * N + lo:(k + 1) * N] = flow[k]
                if disc is not None:
                    out[r * Nm + N:(r + 1) * Nm, k * N + lo:(k + 1) * N] = disc[k]
        return out


class RespOps:
    """The response operators of an agent's controls as applications: the action kernel of control u (N, ...)
    -> the world (nP, N, ...) it produces through the impulse responses R (nP N, nU), the own block the
    action itself (plus, with a continuation, the frozen reaction on the buffer)."""

    def __init__(self, solver, agent: Agent, R: np.ndarray, keep_own: bool = False):
        """keep_own: R's own block is kept (the action plus it: an instant observer's own reactions in a monitored R)."""
        c = solver.c; g = c.g; self.c = c
        self.N, self.nP = c.N, len(c.prim)
        lp = c._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                     known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        self.ops: List[Tuple[int, List[int], Optional[PathOp]]] = []
        # the instant reactions the action draws move with it at the same node (c.composite: a trader on the quote)
        self.inst = [[(c.prim.index(v), coef) for v, coef in ((c.composite or {}).get(u) or {}).items() if v != u]
                     for u in agent.controls]
        for ui, u in enumerate(agent.controls):
            own = c.prim.index(u)
            K = R[:, ui].reshape(self.nP, self.N).T.copy()
            if c.cont is None and not keep_own:
                K[:, own] = 0.0
            ps = _nonzero_cols(K)
            op = PathOp(lp, K[:, ps]) if ps else None
            self.ops.append((own, ps, op if op is not None and not op.empty else None))

    def apply(self, ui: int, C: np.ndarray) -> np.ndarray:
        """The world (nP, N, ...) of the action kernel C (N, ...) of control ui."""
        own, ps, op = self.ops[ui]
        Z = np.zeros((self.nP,) + C.shape)
        if op is not None:
            IC = op.unknown(C.reshape(self.N, -1), scratch=True)
            for j, p in enumerate(ps):
                Z[p] += op.apply(IC, j).reshape(C.shape)
        Z[own] += C
        for p, coef in self.inst[ui]:
            Z[p] += coef * C
        return Z

    def dense(self, ui: int, lo: int = 0, scratch: Optional[str] = None) -> np.ndarray:
        """Resp_u as a dense array (nP N, N): the rows of the response path per responding primary, the own
        block the identity (the factored FOC system); with lo the world's nodes from lo on only (the reduced
        system under freeze_before), zero before.  scratch: a work buffer's name (_zeros)."""
        own, ps, op = self.ops[ui]
        Z = _zeros((self.nP, self.N, self.N), scratch)
        if op is not None:
            for j, p in enumerate(ps):
                Z[p, lo:] += op.rows(j, lo, self.N)
        Z[own] += np.eye(self.N)
        for p, coef in self.inst[ui]:
            Z[p] += coef * np.eye(self.N)
        return Z.reshape(self.nP * self.N, self.N)

    def adjoint(self, ui: int, Z: np.ndarray) -> np.ndarray:
        """Resp_u^T Z: (N, ...) for Z (nP, N, ...)."""
        own, ps, op = self.ops[ui]
        out = Z[own].copy()
        for p, coef in self.inst[ui]:
            out += coef * Z[p]
        if op is not None:
            for j, p in enumerate(ps):
                out += op.adjoint(Z[p].reshape(self.N, -1), j).reshape(out.shape)
        return out


class InstantResp:
    """RespOps closed under the agent's own instant reactions (an agent with `instant` observations, compile.instant_loads):
    its control v moves by sum_u h_vu times the level u it sees, a reaction its loss fixes and its map does not carry, so
    the world of an action kernel C is Z = W + A x with W = base(C), A = [base_w] and x = (I - B A)^-1 B W, B the
    loadings on the seen levels' blocks.  close(Z) closes a world computed with the agent's controls off (the passive
    world); apply, dense and adjoint are RespOps' with the closure."""

    def __init__(self, base: RespOps, pairs):
        from scipy.linalg import lu_factor
        self.base, self.pairs = base, pairs                 # pairs: (control index v, primary index of u, h)
        self.N, self.nP = base.N, base.nP
        self.nU = len(base.ops)
        N, nU = self.N, self.nU
        self._dense = [base.dense(wi) for wi in range(nU)]  # (nP N, N) each
        BA = np.zeros((nU * N, nU * N))
        for vi, p, h in pairs:
            for wi in range(nU):
                BA[vi * N:(vi + 1) * N, wi * N:(wi + 1) * N] += h * self._dense[wi][p * N:(p + 1) * N]
        self.lu = lu_factor(np.eye(nU * N) - BA)

    def _B(self, Z: np.ndarray) -> np.ndarray:
        out = np.zeros((self.nU * self.N,) + Z.shape[2:])
        for vi, p, h in self.pairs:
            out[vi * self.N:(vi + 1) * self.N] += h * Z[p]
        return out

    def close(self, Z: np.ndarray) -> np.ndarray:
        """The world Z (nP, N, ...) closed under the agent's instant reactions."""
        from scipy.linalg import lu_solve
        x = lu_solve(self.lu, self._B(Z).reshape(self.nU * self.N, -1))
        out = Z.copy()
        for wi in range(self.nU):
            out += self.base.apply(wi, x[wi * self.N:(wi + 1) * self.N].reshape((self.N,) + Z.shape[2:]))
        return out

    def apply(self, ui: int, C: np.ndarray) -> np.ndarray:
        return self.close(self.base.apply(ui, C))

    def dense(self, ui: int, lo: int = 0, scratch: Optional[str] = None) -> np.ndarray:
        from scipy.linalg import lu_solve
        N, nP = self.N, self.nP
        D = self._dense[ui]
        x = lu_solve(self.lu, self._B(D.reshape(nP, N, N)))
        out = D + sum(self._dense[wi] @ x[wi * N:(wi + 1) * N] for wi in range(self.nU))
        if lo:
            out = out.reshape(nP, N, N).copy(); out[:, :lo] = 0.0; out = out.reshape(nP * N, N)
        return out

    def adjoint(self, ui: int, Z: np.ndarray) -> np.ndarray:
        """Resp_u^T Z for the closed response: base_u^T (Z + B^T (I - B A)^-T A^T Z)."""
        from scipy.linalg import lu_solve
        N = self.N
        AtZ = np.concatenate([self.base.adjoint(wi, Z).reshape(N, -1) for wi in range(self.nU)])
        y = lu_solve(self.lu, AtZ, trans=1)
        Zt = Z.copy()
        for vi, p, h in self.pairs:
            Zt[p] += h * y[vi * N:(vi + 1) * N].reshape(Z.shape[1:])
        return self.base.adjoint(ui, Zt)


def instant_resp(solver, agent: Agent, resp: RespOps):
    """RespOps closed under the agent's own instant reactions (InstantResp), or resp itself for an agent without any."""
    c = solver.c
    loads = c.instant_loads or {}
    pairs = [(vi, c.prim.index(u), float(h)) for vi, v in enumerate(agent.controls) for u, h in loads.get(v, {}).items() if h]
    return InstantResp(resp, pairs) if pairs else resp


class FocOps:
    """The first-order-condition operators of an agent's controls as applications (EngineBase._foc_operators):
    a world (nP, N, ...) -> the loss atoms' kernels (the sparse atom reads), (Q zeta) per atom, and per
    control u the sum over the atoms of the instantaneous derivative, the discounted continuation through
    the atom's impulse response R_off (its own reactions off: the envelope) and the delayed read of the
    control itself.  A terminal loss (1/2 z'Q_T z + q_T'z on the states at T) appends its atoms after the flow
    loss's (m_flow of them), Q block-diagonal, each with the adjoint's terminal condition as its operator: at a
    node (t, s), e^{-rho (T - t)} times the atom's response at T to an impulse at t, times (Q_T zeta)(T, s)."""

    def __init__(self, solver, agent: Agent, R: np.ndarray, envelope: bool = True):
        """envelope=False: the continuation also runs through the agent's own controls' atoms (R then holds the agent's
        own later reactions: consistent planning, whose later selves do not share the date-t objective)."""
        c = solver.c; g = c.g; self.c = c; self.agent = agent
        self.N, self.nP = c.N, len(c.prim)
        self.atoms, self.Q, _ = c.loss[agent.name]
        self.AO = [(c.index[nm], c.atom_sparse((nm, lag))) for (nm, lag) in self.atoms]
        lp = c._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, c.Tg), point_fn=lambda k, r: (r, r - g.s[k]),
                     known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-c.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        self.per_control = []                       # per control: (its instantaneous atom, continuation (atoms, op), own lag reads)
        self.inst = []                              # per control: [(atom, coef)], its instantaneous term with the reactions it draws
        for ui, u in enumerate(agent.controls):
            j0 = self.atoms.index((u, 0.0)) if (u, 0.0) in self.atoms else None
            self.inst.append([(self.atoms.index((v, 0.0)), coef) for v, coef in ((c.composite or {}).get(u) or {}).items()
                              if v != u and (v, 0.0) in self.atoms])
            cont, lags = None, []
            if not agent.myopic:
                # own atoms: with envelope=False (R holds the own later reactions), or where R carries the agent's own
                # instant reactions (a player privy to it answers its spike and the agent reacts at once to the level it
                # sees): a fixed reaction, not a choice the envelope drops
                inst_own = bool(c.instant_loads) and any(np.any(R[c.block(v), ui]) for v in agent.controls)
                js = [j for j, (nm, lag) in enumerate(self.atoms)
                      if not envelope or nm not in agent.controls or (inst_own and np.any(R[c.block(nm), ui]))]
                if js:
                    Rj = np.stack([self.AO[j][1] @ R[c.block(self.atoms[j][0]), ui] for j in js], axis=1)
                    keep = _nonzero_cols(Rj)
                    if keep:
                        op = PathOp(lp, Rj[:, keep], disc)
                        if not op.empty:
                            cont = ([js[i] for i in keep], op)
                for j, (nm, lag) in enumerate(self.atoms):
                    if nm == u and lag > 0:
                        lags.append((j, float(np.exp(-c.rho * lag)), c.read_sparse(-lag, -lag)))
            self.per_control.append((j0, cont, lags))
        self.q = c.loss[agent.name][2]
        self.m_flow = len(self.atoms)
        self.per_terminal = [[] for _ in agent.controls]          # per control: [(atom index, N x N operator)]
        terminal = (c.terminal or {}).get(agent.name)
        if terminal and terminal[0]:
            tatoms, QT, qT = terminal
            m = self.m_flow; mt = len(tatoms)
            Q = np.zeros((m + mt, m + mt)); Q[:m, :m] = self.Q; Q[m:, m:] = QT
            self.atoms = list(self.atoms) + list(tatoms); self.Q = Q; self.q = np.concatenate([self.q, qT])
            self.AO = self.AO + [(c.index[nm], c.atom_sparse((nm, lag))) for (nm, lag) in tatoms]
            if not agent.myopic:                                  # myopic: no continuation effect, the terminal one included
                from scipy.sparse import diags
                IZ, IR, disc = c.terminal_reads
                for ui in range(len(agent.controls)):
                    for jt, (nm, lag) in enumerate(tatoms):
                        w = disc * (IR @ (self.AO[m + jt][1] @ R[c.block(nm), ui]))
                        if np.any(w):
                            self.per_terminal[ui].append((m + jt, (diags(w) @ IZ).tocsr()))

    def atoms_of(self, Z: np.ndarray) -> np.ndarray:
        """The loss atoms' kernels (n_atoms, N, ...) in the world Z (nP, N, ...)."""
        return np.stack([(A @ Z[p].reshape(self.N, -1)).reshape(Z.shape[1:]) for (p, A) in self.AO])

    def foc(self, ui: int, a: np.ndarray) -> np.ndarray:
        """The FOC kernel (N, ...) of control ui from the atoms' kernels a (n_atoms, N, ...)."""
        return self.on_qzeta(ui, np.tensordot(self.Q, a, axes=1))       # b_j = sum_i Q[j, i] a_i

    def on_qzeta(self, ui: int, b: np.ndarray) -> np.ndarray:
        """sum_j M_j b_j, (N, ...) for b (n_atoms, N, ...): the per-atom operators of control ui (the instantaneous
        derivative, the discounted continuation, the own lagged reads) on the kernels of Q zeta (+ q, for the
        means)."""
        j0, cont, lags = self.per_control[ui]
        out = np.zeros(b.shape[1:])
        if j0 is not None:
            out += b[j0]
        for j, coef in self.inst[ui]:
            out += coef * b[j]
        if cont is not None:
            js, op = cont
            for i, j in enumerate(js):
                out += op.apply(op.unknown(b[j].reshape(self.N, -1), scratch=True), i).reshape(out.shape)
        for (j, w, S) in lags:
            out += w * (S @ b[j].reshape(self.N, -1)).reshape(out.shape)
        for (j, W) in self.per_terminal[ui]:
            out += (W @ b[j].reshape(self.N, -1)).reshape(out.shape)
        return out

    def apply(self, ui: int, Z: np.ndarray) -> np.ndarray:
        return self.foc(ui, self.atoms_of(Z))

    def dense(self, ui: int, lo: int = 0, scratch: Optional[str] = None) -> np.ndarray:
        """Fu_u as a dense array (N, nP N): the per-atom operators M_j as dense rows (the identity, the rows of the
        continuation path, the lag reads), contracted with Q and the atoms' reads, sum_i (sum_j Q[j, i] M_j) A_i
        (the factored FOC system); with lo the rows of the action nodes from lo on only (the reduced system under
        freeze_before), zero before.  scratch: a work buffer's name (_zeros)."""
        N = self.N
        j0, cont, lags = self.per_control[ui]
        M = np.zeros((len(self.atoms), N - lo, N))
        if j0 is not None:
            M[j0] += np.eye(N)[lo:]
        for j, coef in self.inst[ui]:
            M[j] += coef * np.eye(N)[lo:]
        if cont is not None:
            js, op = cont
            for i, j in enumerate(js):
                M[j] += op.rows(i, lo, N)
        for (j, w, S) in lags:
            M[j] += w * S.toarray()[lo:]
        for (j, W) in self.per_terminal[ui]:
            M[j] += W.toarray()[lo:]
        MQ = np.tensordot(self.Q.T, M, axes=1)                              # MQ[i] = sum_j Q[j, i] M_j
        out = _zeros((N, self.nP * N), scratch)
        for i, (p, A) in enumerate(self.AO):
            if np.any(MQ[i]):
                out[lo:, p * N:(p + 1) * N] += (A.T @ MQ[i].T).T
        return out

    def own_lags(self) -> Dict[float, float]:
        """The controls' own lags with their discount, {lag: e^{-rho lag}}, the instantaneous read included."""
        out = {0.0: 1.0}
        for (j0, cont, lags) in self.per_control:
            for (j, w, S) in lags:
                out[float(self.atoms[j][1])] = w
        return out


class PanelRows:
    """The rows of the row operator G_k (RowOps.rows) one time panel at a time, for the per-time-row Gram of
    maps_from_world and the preconditioner: sub(idx, cols) is Bk[:, idx][:, :, cols] (ncol, len(idx), len(cols))
    for action nodes idx of one time row and map unknowns cols (row, node) of it, built from the panel's block
    (kept while the rows of that panel are asked for)."""

    def __init__(self, solver, rowops: RowOps, shifts: List[int]):
        self.solver = solver; self.c = solver.c; self.ops = rowops; self.shifts = shifts
        self.N, self.Nm, self.nR = self.c.N, solver.Nm, rowops.nR
        self.ranges = self.c._panel_ranges
        self.panel = None; self.block = None; self.colidx = None; self.lo = 0

    def _build(self, p: int):
        """The panel's block (ncol, hi - lo, its columns), the rows written straight into it (RowOps.rows(into=)), in a
        work buffer shared by every PanelRows of the thread (triangle._scratch) and reused from panel to panel: sub()
        hands out copies, and a PanelRows whose block another has since overwritten builds it again."""
        lo, hi = self.ranges[p]
        rng = [self.ranges[p - s] if p - s >= 0 else (0, 0) for s in self.shifts]
        nd = self.Nm - self.N
        cols = []
        for r in range(self.nR):
            lo_r, hi_r = rng[r]
            cols.append(r * self.Nm + np.arange(lo_r, hi_r))
            if nd > 0:
                cols.append(r * self.Nm + self.N + np.arange(nd))
        width = sum(cc.size for cc in cols); ncol = self.c.ncol
        block = _scratch("panel_rows", (ncol, hi - lo, width)); block.fill(0.0)
        _scratch_owner()["panel_rows"] = id(self)
        into, pos = [], 0
        for r in range(self.nR):
            lo_r, hi_r = rng[r]
            flow = block[:, :, pos:pos + hi_r - lo_r]; pos += hi_r - lo_r
            disc = None
            if nd > 0:
                disc = block[:, :, pos:pos + nd]; pos += nd
            into.append((flow, disc))
        self.ops.rows(lo, hi, rng, into=into)
        self.colidx = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
        self.block = block
        self.panel = p; self.lo = lo

    def sub(self, idx: np.ndarray, cols: np.ndarray) -> np.ndarray:
        p = int(self.c.panel_of_node[idx[0]])
        if self.panel != p or _scratch_owner().get("panel_rows") != id(self):
            self._build(p)
        pos = np.searchsorted(self.colidx, cols)
        if pos.size and not ((pos < self.colidx.size).all() and (self.colidx[pos] == cols).all()):
            raise AssertionError("a map column outside the panel's block")
        return self.block[:, idx - self.lo][:, :, pos]
