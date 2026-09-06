"""The matrix-free best response of the spectral finite engine (settings.foc_dense_max).

The best response of EngineBase, and its version with a past in SpectralFiniteSolver, builds the row
operator G_k, the response operators Resp_u, the first-order-condition operators Fu_u and the projection
operators H_k as dense arrays (about thirty N x N operators) and the (nU nR N)^2 system Amat, tens of
gigabytes at ten thousand nodes.  Beyond settings.foc_dense_max unknowns nU nR N the solver comes here:

  * every operator is an application built from the same line paths, PathOp: R diag(w J y) I on the path's
    quadrature points with the known kernel y read once, applied to the columns of a world at a time, and
    from the compiled model's sparse reads (read_sparse, map_shift_sparse, row_blocks_sparse, mass_sparse);
  * the first-order conditions are solved by GMRES on the operator
        gamma -> sum_k H_k Fu_u (sum_v Resp_v G_k gamma_v)
    with the right-hand side assembled the same way (FocSystem), to settings.foc_krylov_tol relative to the
    right-hand side, warm-started from the agent's last solution, preconditioned by the part of the
    operator that is block diagonal by time row: the own cost read instantaneously through the row
    operator and projected back, kron(Q_l, sum_k H_k D_l G_k) summed over the control's own lags l (the
    lag's read where it exists, D_l), assembled one time panel at a time from the rows of G_k and H_k
    (PanelRows) and LU-factored per time row;
  * the projection of the action kernels on the seen rows (maps_from_world) and the representation error
    read the row operator's rows one panel at a time, the second-order check's form is applied by the same
    operators (a block of strategies at a time), the decomposition too, and the costs use the sparse mass.

The same integrals and sums as the dense path: the results agree to BLAS rounding and the Krylov
tolerance, not to the bit, which is why a system within foc_dense_max keeps the dense path.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres

from .engine import singular_system_message
from .spec import Agent


class PathOp:
    """The line-integral operators of one path (triangle.LinePath) for m known kernels K (N_known, m):
    op_j = R diag(w F_j) I with F = J K the knowns read at the quadrature points, as applications on
    vectors (columns of a world, a block of them) and as dense rows of a panel of output nodes; extra
    scales every point (the continuation's discount).  A path with no points (lp.rows None) is the zero
    operator (empty)."""

    def __init__(self, lp, K: np.ndarray, extra: Optional[np.ndarray] = None):
        self.lp = lp
        K = np.asarray(K, dtype=float)
        K = K if K.ndim == 2 else K[:, None]
        self.m = K.shape[1]
        self.n_out, self.N = lp.n_out, lp.N
        if lp.rows is None:
            self.F = None
        else:
            F = lp.read(K) if lp.Jf is not None else lp.J @ K
            if extra is not None:
                F = F * extra[:, None]
            self.F = lp.w[:, None] * F                          # (nq, m): the weights included

    @property
    def empty(self) -> bool:
        return self.F is None

    def unknown(self, V: np.ndarray) -> np.ndarray:
        """I V: the unknown read at the quadrature points, (nq, ...) for V (N, ...)."""
        return self.lp.I @ V

    def apply(self, IV: np.ndarray, j: int) -> np.ndarray:
        """op_j V given IV = I V (nq, ...)."""
        Fj = self.F[:, j]
        return self.lp.R @ (Fj * IV if IV.ndim == 1 else Fj[:, None] * IV)

    def apply_all(self, IV: np.ndarray) -> np.ndarray:
        """Every op_j on the same vectors: (n_out, m, B) for IV = I V (nq, B)."""
        B = IV.shape[1]
        return (self.lp.R @ (self.F[:, :, None] * IV[:, None, :]).reshape(-1, self.m * B)).reshape(self.n_out, self.m, B)

    def apply_sum(self, IV: np.ndarray) -> np.ndarray:
        """sum_j op_j V[:, j] for IV = I V (nq, m, B): (n_out, B), one product."""
        return self.lp.R @ (self.F[:, :, None] * IV).sum(axis=1)

    def rows(self, j: int, lo: int, hi: int) -> np.ndarray:
        """Rows [lo, hi) of op_j, dense (hi - lo, N): the same sums as LinePath.with_known(rows=)."""
        if self.F is None:
            return np.zeros((hi - lo, self.N))
        return self.lp._weighted_sum(self.F[:, j], rows=(lo, hi))

    def adjoint_sum(self, W: np.ndarray) -> np.ndarray:
        """sum_j op_j^T W[:, j] for W (n_out, m, B): I^T (sum_j F_j (R^T W)_j), (N, B)."""
        return self.lp.I.T @ (self.F[:, :, None] * W[self.lp.rows]).sum(axis=1)

    def adjoint(self, W: np.ndarray, j: int) -> np.ndarray:
        """op_j^T W for W (n_out, ...)."""
        Fj = self.F[:, j]
        RW = W[self.lp.rows]
        return self.lp.I.T @ (Fj * RW if RW.ndim == 1 else Fj[:, None] * RW)


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
                out[:, ks] += op.apply_all(op.unknown(gr))
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

    def rows(self, lo: int, hi: int, ranges: List[Tuple[int, int]]):
        """The rows [lo, hi) (a panel's action nodes) of every G_k restricted to the map columns of each row r in
        ranges[r] = (lo_r, hi_r) plus its discrete weights: a list per row of (flow (ncol, hi - lo, hi_r - lo_r),
        disc (ncol, hi - lo, Nt) or None)."""
        out = []
        for r in range(self.nR):
            lo_r, hi_r = ranges[r]
            flow = np.zeros((self.ncol, hi - lo, hi_r - lo_r))
            for (ks, op) in self._paths(r):
                for j, k in enumerate(ks):
                    flow[k] += op.rows(j, lo, hi)[:, lo_r:hi_r]
            for (k, nw, S) in self.inst[r]:
                flow[k] += nw[lo:hi, None] * S[lo:hi, lo_r:hi_r].toarray()
            disc = None
            if self.Nm > self.N:
                disc = np.zeros((self.ncol, hi - lo, self.Nm - self.N))
                for (i, e, E) in self.embed[r]:
                    disc[self.nW + i] += e * E[lo:hi]
            out.append((flow, disc))
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
                out[r, :self.N] += op.apply_sum(op.unknown(phi[:, ks].reshape(self.N, -1)).reshape(-1, len(ks), phi.shape[2]))
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
                flow[k] += op.rows(j, lo_r, hi_r)[:, lo:hi]
        for (k, nw, S) in self.inst[r]:
            flow[k] += S[lo:hi, lo_r:hi_r].toarray().T * nw[lo:hi][None, :]
        for (i, yi, Id) in self.init[r]:
            flow[self.nW + i] += yi[lo_r:hi_r, None] * Id[lo_r:hi_r, lo:hi].toarray()
        disc = None
        if self.Nm > self.N:
            disc = np.zeros((self.ncol, self.Nm - self.N, hi - lo))
            for (i, e, Sel) in self.select[r]:
                disc[self.nW + i] += e * Sel[:, lo:hi]
        return flow, disc


class RespOps:
    """The response operators of an agent's controls as applications: the action kernel of control u (N, ...)
    -> the world (nP, N, ...) it produces through the impulse responses R (nP N, nU), the own block the
    action itself (plus, with a continuation, the frozen reaction on the buffer)."""

    def __init__(self, solver, agent: Agent, R: np.ndarray):
        c = solver.c; g = c.g; self.c = c
        self.N, self.nP = c.N, len(c.prim)
        lp = c._path(("response",), r_lo=g.s, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                     known_fn=lambda k, r: (np.full_like(r, g.t[k]), g.t[k] - r))
        self.ops: List[Tuple[int, List[int], Optional[PathOp]]] = []
        for ui, u in enumerate(agent.controls):
            own = c.prim.index(u)
            K = R[:, ui].reshape(self.nP, self.N).T.copy()
            if c.cont is None:
                K[:, own] = 0.0
            ps = _nonzero_cols(K)
            op = PathOp(lp, K[:, ps]) if ps else None
            self.ops.append((own, ps, op if op is not None and not op.empty else None))

    def apply(self, ui: int, C: np.ndarray) -> np.ndarray:
        """The world (nP, N, ...) of the action kernel C (N, ...) of control ui."""
        own, ps, op = self.ops[ui]
        Z = np.zeros((self.nP,) + C.shape)
        if op is not None:
            IC = op.unknown(C.reshape(self.N, -1))
            for j, p in enumerate(ps):
                Z[p] += op.apply(IC, j).reshape(C.shape)
        Z[own] += C
        return Z

    def adjoint(self, ui: int, Z: np.ndarray) -> np.ndarray:
        """Resp_u^T Z: (N, ...) for Z (nP, N, ...)."""
        own, ps, op = self.ops[ui]
        out = Z[own].copy()
        if op is not None:
            for j, p in enumerate(ps):
                out += op.adjoint(Z[p].reshape(self.N, -1), j).reshape(out.shape)
        return out


class FocOps:
    """The first-order-condition operators of an agent's controls as applications (EngineBase._foc_operators):
    a world (nP, N, ...) -> the loss atoms' kernels (the sparse atom reads), (Q zeta) per atom, and per
    control u the sum over the atoms of the instantaneous derivative, the discounted continuation through
    the atom's impulse response R_off (its own reactions off: the envelope) and the delayed read of the
    control itself."""

    def __init__(self, solver, agent: Agent, R: np.ndarray):
        c = solver.c; g = c.g; self.c = c; self.agent = agent
        self.N, self.nP = c.N, len(c.prim)
        self.atoms, self.Q, _ = c.loss[agent.name]
        self.AO = [(c.index[nm], c.atom_sparse((nm, lag))) for (nm, lag) in self.atoms]
        lp = c._path(("continuation",), r_lo=g.t, r_hi=np.full(self.N, c.Tg), point_fn=lambda k, r: (r, r - g.s[k]),
                     known_fn=lambda k, r: (r, r - g.t[k]))
        disc = np.exp(-c.rho * (lp.r - g.t[lp.rows])) if lp.rows is not None else None
        self.per_control = []                       # per control: (its instantaneous atom, continuation (atoms, op), own lag reads)
        for ui, u in enumerate(agent.controls):
            j0 = self.atoms.index((u, 0.0)) if (u, 0.0) in self.atoms else None
            cont, lags = None, []
            if not agent.myopic:
                js = [j for j, (nm, lag) in enumerate(self.atoms) if nm not in agent.controls]
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

    def atoms_of(self, Z: np.ndarray) -> np.ndarray:
        """The loss atoms' kernels (n_atoms, N, ...) in the world Z (nP, N, ...)."""
        return np.stack([(A @ Z[p].reshape(self.N, -1)).reshape(Z.shape[1:]) for (p, A) in self.AO])

    def foc(self, ui: int, a: np.ndarray) -> np.ndarray:
        """The FOC kernel (N, ...) of control ui from the atoms' kernels a (n_atoms, N, ...)."""
        b = np.tensordot(self.Q, a, axes=1)                    # b_j = sum_i Q[j, i] a_i
        j0, cont, lags = self.per_control[ui]
        out = np.zeros(a.shape[1:])
        if j0 is not None:
            out += b[j0]
        if cont is not None:
            js, op = cont
            for i, j in enumerate(js):
                out += op.apply(op.unknown(b[j].reshape(self.N, -1)), i).reshape(out.shape)
        for (j, w, S) in lags:
            out += w * (S @ b[j].reshape(self.N, -1)).reshape(out.shape)
        return out

    def apply(self, ui: int, Z: np.ndarray) -> np.ndarray:
        return self.foc(ui, self.atoms_of(Z))

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
        lo, hi = self.ranges[p]
        rng = [self.ranges[p - s] if p - s >= 0 else (0, 0) for s in self.shifts]
        parts = self.ops.rows(lo, hi, rng)
        cols, blocks = [], []
        for r, (flow, disc) in enumerate(parts):
            lo_r, hi_r = rng[r]
            cols.append(r * self.Nm + np.arange(lo_r, hi_r)); blocks.append(flow)
            if disc is not None:
                cols.append(r * self.Nm + self.N + np.arange(self.Nm - self.N)); blocks.append(disc)
        self.colidx = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
        self.block = np.concatenate(blocks, axis=2) if blocks else np.zeros((self.c.ncol, hi - lo, 0))
        self.panel = p; self.lo = lo

    def sub(self, idx: np.ndarray, cols: np.ndarray) -> np.ndarray:
        p = int(self.c.panel_of_node[idx[0]])
        if self.panel != p:
            self._build(p)
        pos = np.searchsorted(self.colidx, cols)
        if pos.size and not ((pos < self.colidx.size).all() and (self.colidx[pos] == cols).all()):
            raise AssertionError("a map column outside the panel's block")
        return self.block[:, idx - self.lo][:, :, pos]


class FocSystem:
    """The first-order-condition system of an agent's best response as an operator on its kept map unknowns,
    Amat gamma = -bvec with Amat = sum_k H_k Fu Resp G_k, its right-hand side, the time-row preconditioner
    and the GMRES solve."""

    def __init__(self, solver, agent: Agent, rowops: RowOps, projops: ProjOps, resp: RespOps, foc: FocOps,
                 Zpass: np.ndarray, phi_past):
        c = solver.c; self.solver = solver; self.c = c; self.agent = agent
        self.rowops, self.projops, self.resp, self.foc = rowops, projops, resp, foc
        self.N, self.nP, self.ncol, self.Nm = c.N, len(c.prim), c.ncol, solver.Nm
        self.nU, self.nR = len(agent.controls), len(agent.signals)
        self.nG = self.nU * self.nR * self.Nm
        # the kept unknowns, tied at a Duffy triangle's degenerate corner rows with a past (_solve_foc)
        keep = np.tile(solver._identified(agent), self.nU)
        self.kept = np.where(keep)[0]
        if c.past is not None:
            grp = solver._corner_index(agent)[self.kept]
            uniq, self.inv = np.unique(grp, return_inverse=True)
            self.n = uniq.size
        else:
            self.inv = np.arange(self.kept.size); self.n = self.kept.size
        self.matvecs = 0; self.iterations = 0; self.residual = 0.0; self.block_rcond = np.inf
        # the right-hand side: the FOC of the passive world projected on the rows
        a = foc.atoms_of(Zpass.reshape(self.nP, self.N, self.ncol))
        b = np.zeros((self.nU, self.nR, self.Nm))
        for ui in range(self.nU):
            phi = foc.foc(ui, a)
            if phi_past is not None:
                phi[:, :c.nW] += phi_past[ui]
            b[ui] = projops.apply(phi)
        self.bvec = self.reduce(b.reshape(-1))

    # ---- the reduced space
    def expand(self, x: np.ndarray) -> np.ndarray:
        gamma = np.zeros(self.nG); gamma[self.kept] = x[self.inv]
        return gamma

    def reduce(self, r: np.ndarray) -> np.ndarray:
        return np.bincount(self.inv, weights=r[self.kept], minlength=self.n)

    # ---- the operator
    def world_of(self, gamma: np.ndarray) -> np.ndarray:
        """sum_v Resp_v G gamma_v: the world (nP, N, ncol[, B]) of the strategy gamma (nU, nR, Nm[, B]) (Zpass excluded)."""
        Zd = None
        for vi in range(self.nU):
            W = self.resp.apply(vi, self.rowops.apply(gamma[vi]))
            Zd = W if Zd is None else Zd + W
        return Zd

    def apply(self, gamma: np.ndarray) -> np.ndarray:
        """Amat gamma on the full unknowns (nU, nR, Nm)."""
        a = self.foc.atoms_of(self.world_of(gamma.reshape(self.nU, self.nR, self.Nm)))
        out = np.zeros((self.nU, self.nR, self.Nm))
        for ui in range(self.nU):
            out[ui] = self.projops.apply(self.foc.foc(ui, a))
        return out.reshape(-1)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        self.matvecs += 1
        return self.reduce(self.apply(self.expand(np.asarray(x, dtype=float).ravel())))

    # ---- the preconditioner: the time-row-diagonal part of the operator
    def preconditioner(self):
        """Per time row (p, it) the block kron(Q_l, sum_k H_k D_l G_k) summed over the controls' own lags l on the
        row's kept unknowns (tied at the corners), LU-factored: [(reduced indices, lu)]."""
        c = self.c; g = c.g; N, Nm, nR, nU = self.N, self.Nm, self.nR, self.nU
        atoms, Q = self.foc.atoms, self.foc.Q
        ctrl = list(self.agent.controls)
        Ql = {}
        for l, w in self.foc.own_lags().items():
            M = np.zeros((nU, nU))
            for ui, u in enumerate(ctrl):
                for vi, v in enumerate(ctrl):
                    if (u, l) in atoms and (v, l) in atoms:
                        M[ui, vi] = Q[atoms.index((u, l)), atoms.index((v, l))]
            if np.any(M):
                Ql[l] = (w * M, self._lag_read(l))
        shifts = [c.panel_shift(c.rows[self.agent.name][r][3]) if c.rows[self.agent.name][r][3] > 0 else 0 for r in range(nR)]
        panel = PanelRows(self.solver, self.rowops, shifts)
        gecon, = get_lapack_funcs(("gecon",), (np.zeros((1, 1)),))
        hrows: Dict[tuple, tuple] = {}                          # (panel, row) -> the projection's rows of the row's map panel
        kept_pos = np.full(self.nG, -1, dtype=int); kept_pos[self.kept] = np.arange(self.kept.size)
        blocks = []
        ranges = c._panel_ranges
        for (p, it), idx in sorted(c.trow_by_pit.items()):
            if c.past is not None and p >= c.P_T:
                continue
            lo, hi = ranges[p]
            parts = []                                          # (row, map unknowns of the row at this time row)
            for r in range(nR):
                if p - shifts[r] < 0:
                    continue
                cr = [r * Nm + c.trow_by_pit[(p - shifts[r], it)]]
                if c.n_init:
                    cr.append(np.array([r * Nm + N + (p - shifts[r]) * g.nt + it]))
                parts.append((r, np.concatenate(cr)))
            if not parts:
                continue
            cols = np.concatenate([cr for _, cr in parts])
            Gsub = panel.sub(idx, cols)                                                  # (ncol, nidx, ncols)
            Hsub = np.zeros((self.ncol, cols.size, idx.size))                            # the projection's rows at the map nodes
            pos = 0
            for r, cr in parts:
                if (p, r) not in hrows:
                    if any(k[0] != p for k in hrows):
                        hrows.clear()
                    lo_r, hi_r = ranges[p - shifts[r]]
                    hrows[(p, r)] = (lo_r, self.projops.rows(r, lo_r, hi_r, lo, hi))
                lo_r, (flow, disc) = hrows[(p, r)]
                nf = int(np.sum(cr < r * Nm + N))
                Hsub[:, pos:pos + nf] = flow[:, cr[:nf] - r * Nm - lo_r][:, :, idx - lo]
                if disc is not None and nf < cr.size:
                    Hsub[:, pos + nf:pos + cr.size] = disc[:, cr[nf:] - r * Nm - N][:, :, idx - lo]
                pos += cr.size
            P = np.zeros((nU * cols.size, nU * cols.size))
            for l, (M, D) in Ql.items():
                S = np.zeros((cols.size, cols.size))
                Dl = D[idx]
                for k in range(self.ncol):
                    S += Hsub[k] @ (Dl[:, None] * Gsub[k])
                P += np.kron(M, S)
            # the kept unknowns of the block, tied at the corners
            full = np.concatenate([ui * nR * Nm + cols for ui in range(nU)])
            kp = kept_pos[full]
            sel = np.where(kp >= 0)[0]
            if sel.size == 0:
                continue
            uniq, loc = np.unique(self.inv[kp[sel]], return_inverse=True)
            Rm = np.zeros((sel.size, uniq.size)); Rm[np.arange(sel.size), loc] = 1.0
            B = Rm.T @ P[np.ix_(sel, sel)] @ Rm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                lu = lu_factor(B, check_finite=False)
            rcond = float(gecon(lu[0], np.linalg.norm(B, 1))[0])
            if not rcond > self.solver.FOC_RCOND:
                # the dense path's condition test on the whole system, here on its time-row blocks (the causes named are the
                # same: a control with no quadratic term in its current value, two rows carrying the same information)
                raise ValueError(singular_system_message(self.agent.name) +
                                 f" (reciprocal condition estimate {rcond:.1e} of the time-row block at t = {g.t[idx[0]]:g})")
            self.block_rcond = min(self.block_rcond, rcond)
            blocks.append((uniq, lu))
        return blocks

    def _lag_read(self, l: float) -> np.ndarray:
        """D_l (N,): where the control's read l later of its own map shifted back by l exists (1 inside, 0
        past the horizon or the window): the row sums of read(-l, -l) map_shift(l)."""
        if l == 0.0:
            return np.ones(self.N)
        return np.asarray((self.c.read_sparse(-l, -l) @ self.c.map_shift_sparse(l)).sum(axis=1)).ravel()

    def solve(self, x0: Optional[np.ndarray] = None):
        """gamma (nU, nR, Nm) solving the system to settings.foc_krylov_tol (relative to the right-hand side),
        from the warm start x0 (a full gamma), and the GMRES iteration count.  A system that does not converge
        within foc_krylov_maxiter iterations raises the ValueError of a singular system."""
        st = self.solver.settings
        tol, maxiter = st.foc_krylov_tol, st.foc_krylov_maxiter
        blocks = self.preconditioner()
        n = self.n

        def prec(r):
            r = np.asarray(r, dtype=float).ravel(); x = r.copy()
            for (ix, lu) in blocks:
                x[ix] = lu_solve(lu, r[ix], check_finite=False)
            return x
        A = LinearOperator((n, n), matvec=self.matvec, dtype=float)
        M = LinearOperator((n, n), matvec=prec, dtype=float)
        b = -self.bvec
        bn = float(np.linalg.norm(b))
        if bn == 0.0:
            return np.zeros((self.nU, self.nR, self.Nm)), 0
        xs = None
        if x0 is not None:
            xs = np.asarray(x0, dtype=float).ravel()[self.kept]
            xs = np.bincount(self.inv, weights=xs, minlength=n) / np.maximum(np.bincount(self.inv, minlength=n), 1)
        self.matvecs = 0
        restart = min(maxiter, 300)                       # scipy's maxiter counts the restarts
        x, info = gmres(A, b, x0=xs, M=M, rtol=tol, atol=0.0, restart=restart, maxiter=-(-maxiter // restart))
        resid = float(np.linalg.norm(self.matvec(x) - b)) / bn
        self.iterations = self.matvecs - 1
        self.residual = resid
        if not resid <= 10 * tol:
            raise ValueError(singular_system_message(self.agent.name) +
                             f" (GMRES did not converge: relative residual {resid:.1e} after {self.iterations} iterations)")
        return self.expand(x).reshape(self.nU, self.nR, self.Nm), self.iterations


def best_response(solver, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
    """SpectralFiniteSolver.best_response on the matrix-free path (with or without a past)."""
    c = solver.c; N, ncol = c.N, c.ncol
    nU = len(agent.controls); nP = len(c.prim)
    Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
    Zpass, R = Zp[:, :ncol], Zp[:, ncol:]
    R = solver._impulse_responses(agent, maps, R)
    Roff = R
    if c.cont is not None:                       # the envelope responses: the agent's own reaction off on the buffer too
        Roff = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls, own_frozen=False)[:, ncol:]
        Roff = solver._impulse_responses(agent, maps, Roff)
    Zpass = solver._passive_world(agent, maps, Zpass, R)
    ytil, yinst = solver._passive_rows(agent, Zpass)
    rowops = RowOps(solver, agent, ytil, yinst)
    projops = ProjOps(solver, agent, ytil, yinst)
    resp = RespOps(solver, agent, R)
    foc = FocOps(solver, agent, Roff)
    phi_past = solver._foc_affine_free(agent, foc)
    system = FocSystem(solver, agent, rowops, projops, resp, foc, Zpass, phi_past)
    gamma, iters = system.solve(solver._last_gamma.get(agent.name))
    solver._last_gamma[agent.name] = gamma
    solver._krylov_log.append((agent.name, iters, system.residual))
    cact = np.stack([rowops.apply(gamma[ui]) for ui in range(nU)])          # (nU, N, ncol)
    Zfull = Zpass.reshape(nP, N, ncol).copy()
    for ui in range(nU):
        Zfull += resp.apply(ui, cact[ui])
    Zfull = Zfull.reshape(nP * N, ncol)
    if c.cont is not None:                       # the action on the buffer (the frozen map's) is in the world, not in gamma
        cact = np.stack([Zfull[c.block(u)] for u in agent.controls])
    out = {"gamma": gamma, "action": cact, "Zfull": Zfull, "krylov": iters}
    if want_decomp:
        _decompose(solver, agent, out, system, maps)
        if phi_past is not None:
            for ui, u in enumerate(agent.controls):
                for part in ("foc", "physical"):
                    out["decomp"][u][part] = out["decomp"][u][part] + phi_past[ui]
    return solver._project(agent, Zfull, cact), out


def _decompose(solver, agent: Agent, out: dict, system: FocSystem, maps) -> None:
    """EngineBase._decompose with the operators applied: the second-order check on the matrix-free form and
    the FOC decomposition (instantaneous / physical / wedge)."""
    c = solver.c; ncol = c.ncol; N = c.N; nP = len(c.prim)
    Zfull = out["Zfull"].reshape(nP, N, ncol)
    rep = c.rep[agent.name]
    if rep != agent.name and rep in solver._second_order_cache:
        out["second_order"] = solver._second_order_cache[rep]
    else:
        out["second_order"] = _second_order(solver, agent, system)
        solver._second_order_cache[agent.name] = out["second_order"]
    if agent.name not in solver._rphys:
        solver._rphys[agent.name] = c.closed_loop(solver.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, ncol:]
    fphys = FocOps(solver, agent, solver._rphys[agent.name])
    a = system.foc.atoms_of(Zfull); ap = fphys.atoms_of(Zfull)
    dec = {}
    for ui, u in enumerate(agent.controls):
        phi = system.foc.foc(ui, a); phi_phys = fphys.foc(ui, ap)
        dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
    out["decomp"] = dec


def _second_order(solver, agent: Agent, system: FocSystem) -> Optional[dict]:
    """EngineBase._second_order on the matrix-free operators: the form M = T' G T on the kept strategy, T the
    strategy -> world map (RowOps then RespOps), G the loss form (the atoms' kernels under Q and the sparse
    mass; a past's initial columns under the point form of the line s = 0).  Within second_order_dense the
    form is assembled by matvecs on blocks of unit strategies and diagonalised; beyond, its extreme
    eigenvalues come from Lanczos on the matvec."""
    c = solver.c
    if not (solver.SECOND_ORDER_QUADRATIC or c.rho == 0):
        return None
    nW, ncol, N, nP = c.nW, c.ncol, c.N, len(c.prim)
    nU, nR, Nm = system.nU, system.nR, system.Nm
    idx = np.where(np.tile(solver._identified(agent), nU))[0]
    atoms, Q, _ = c.loss[agent.name]
    mass = c.cost_mass_sparse()
    imass = c.time_mass(c.rho)[:c.Nd] if ncol > nW else None
    AO = system.foc.AO

    def GT(Zd):                                     # the loss form on the world (nP, N, ncol, B), column by column
        b = np.tensordot(Q, system.foc.atoms_of(Zd), axes=1)      # (m, N, ncol, B)
        B = b.shape[3]
        Mb = np.zeros_like(b)
        for j in range(len(atoms)):
            Mb[j, :, :nW] = (mass @ b[j, :, :nW].reshape(N, -1)).reshape(N, nW, B)
        if imass is not None:
            Mb[:, c.diag, nW:] = imass[None, :, None, None] * b[:, c.diag, nW:]
        out = np.zeros((nP, N, ncol, B))
        for j, (p, A) in enumerate(AO):
            out[p] += (A.T @ Mb[j].reshape(N, -1)).reshape(N, ncol, B)
        return out

    def matvec(V):                                  # the form on a block of kept strategies (n, B)
        V = np.asarray(V, dtype=float)
        V = V.reshape(-1, 1) if V.ndim == 1 else V
        full = np.zeros((nU * nR * Nm, V.shape[1])); full[idx] = V
        GZ = GT(system.world_of(full.reshape(nU, nR, Nm, -1)))
        out = np.zeros((nU, nR, Nm, V.shape[1]))
        for ui in range(nU):
            out[ui] = system.rowops.adjoint(system.resp.adjoint(ui, GZ))
        return out.reshape(nU * nR * Nm, -1)[idx]
    n = idx.size
    if n <= solver.SECOND_ORDER_DENSE:
        Mfull = np.zeros((n, n)); step = 64
        for i in range(0, n, step):
            E = np.zeros((n, min(step, n - i))); E[i + np.arange(E.shape[1]), np.arange(E.shape[1])] = 1.0
            Mfull[:, i:i + E.shape[1]] = matvec(E)
        w = np.linalg.eigvalsh((Mfull + Mfull.T) / 2)
        lo, hi = float(w[0]), float(w[-1])
    else:
        res = solver._lanczos_extremes(lambda v: matvec(v)[:, 0], n)
        if "message" in res:
            return res
        lo, hi = res["lo"], res["hi"]
    scale = max(abs(lo), abs(hi), 1e-300)
    return {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -solver.SECOND_ORDER_TOL * scale), "converged": True}


def reconstruction(solver, agent: Agent, Zfull: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The action kernels the raw maps g (nU, nR, Nm) reconstruct on the closed-loop rows of Zfull, (nU, N, ncol),
    with the row operator applied (the representation error compares them with the actions)."""
    rows, inst = solver._seen_rows(agent, Zfull, set())
    ops = RowOps(solver, agent, rows, inst)
    return np.stack([ops.apply(g[ui]) for ui in range(len(agent.controls))])


def panel_rows(solver, agent: Agent, Zfull: np.ndarray) -> PanelRows:
    """The row operator's rows one panel at a time on the closed-loop rows of Zfull (maps_from_world)."""
    c = solver.c
    rows, inst = solver._seen_rows(agent, Zfull, set())
    ops = RowOps(solver, agent, rows, inst)
    shifts = [c.panel_shift(c.rows[agent.name][r][3]) if c.rows[agent.name][r][3] > 0 else 0 for r in range(len(rows))]
    return PanelRows(solver, ops, shifts)
