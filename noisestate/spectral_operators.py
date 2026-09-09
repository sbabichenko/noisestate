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
        """I V: the unknown read at the quadrature points, (nq, ...) for V (N, ...).  Through the
        interpolation factors where the path kept them (the same sums, 4 to 6 times faster)."""
        return self.lp.read_unknown(V)

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


    def dense(self, lo: int = 0) -> np.ndarray:
        """Every G_k as a dense array, (ncol, N, nR Nm): the rows of every panel (the factored FOC system); with lo
        the rows of the action nodes from lo on only (the reduced system under freeze_before), zero before."""
        out = np.zeros((self.ncol, self.N, self.nR * self.Nm))
        for r, (flow, disc) in enumerate(self.rows(lo, self.N, [(lo, self.N)] * self.nR)):
            out[:, lo:, r * self.Nm + lo:r * self.Nm + self.N] = flow
            if disc is not None:
                out[:, lo:, r * self.Nm + self.N:(r + 1) * self.Nm] = disc
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


    def dense(self, lo: int = 0) -> np.ndarray:
        """H as a dense array (nR Nm, ncol N), columns (channel, node): every row's map nodes against every
        action node (the factored FOC system); with lo the map nodes and action nodes from lo on only (the
        reduced system under freeze_before), zero before."""
        N, Nm, ncol = self.N, self.Nm, self.ncol
        out = np.zeros((self.nR * Nm, ncol * N))
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

    def dense(self, ui: int, lo: int = 0) -> np.ndarray:
        """Resp_u as a dense array (nP N, N): the rows of the response path per responding primary, the own
        block the identity (the factored FOC system); with lo the world's nodes from lo on only (the reduced
        system under freeze_before), zero before."""
        own, ps, op = self.ops[ui]
        Z = np.zeros((self.nP, self.N, self.N))
        if op is not None:
            for j, p in enumerate(ps):
                Z[p, lo:] += op.rows(j, lo, self.N)
        Z[own] += np.eye(self.N)
        return Z.reshape(self.nP * self.N, self.N)

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
        return self.on_qzeta(ui, np.tensordot(self.Q, a, axes=1))       # b_j = sum_i Q[j, i] a_i

    def on_qzeta(self, ui: int, b: np.ndarray) -> np.ndarray:
        """sum_j M_j b_j, (N, ...) for b (n_atoms, N, ...): the per-atom operators of control ui (the instantaneous
        derivative, the discounted continuation, the own lagged reads) on the kernels of Q zeta (+ q, for the
        means)."""
        j0, cont, lags = self.per_control[ui]
        out = np.zeros(b.shape[1:])
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

    def dense(self, ui: int, lo: int = 0) -> np.ndarray:
        """Fu_u as a dense array (N, nP N): the per-atom operators M_j as dense rows (the identity, the rows of the
        continuation path, the lag reads), contracted with Q and the atoms' reads, sum_i (sum_j Q[j, i] M_j) A_i
        (the factored FOC system); with lo the rows of the action nodes from lo on only (the reduced system under
        freeze_before), zero before."""
        N = self.N
        j0, cont, lags = self.per_control[ui]
        M = np.zeros((len(self.atoms), N - lo, N))
        if j0 is not None:
            M[j0] += np.eye(N)[lo:]
        if cont is not None:
            js, op = cont
            for i, j in enumerate(js):
                M[j] += op.rows(i, lo, N)
        for (j, w, S) in lags:
            M[j] += w * S.toarray()[lo:]
        MQ = np.tensordot(self.Q.T, M, axes=1)                              # MQ[i] = sum_j Q[j, i] M_j
        out = np.zeros((N, self.nP * N))
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
