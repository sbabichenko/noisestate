"""The best response of the spectral finite engine: the first-order-condition system on the operators of
spectral_operators.py, its solve (factored within settings.foc_dense_max, matrix-free by GMRES beyond), the
world of the response and the checks at the solution (the decomposition, the second-order form).

  * the first-order conditions Amat gamma = -bvec with Amat = sum_k H_k Fu_u (sum_v Resp_v G_k gamma_v) are
    assembled and solved by FocSystem on the kept unknowns (a delayed row's unread nodes out, a Duffy
    triangle's degenerate corner rows tied to one unknown each with a past).  Up to settings.foc_dense_max
    unknowns nU nR N the system is assembled by applying the operator to blocks of the identity and
    LU-factored (_solve_regular's condition estimate refuses a singular one): at a few thousand unknowns a
    factorisation beats Krylov.  Beyond, it is solved by GMRES on the matvec to settings.foc_krylov_tol
    relative to the right-hand side, warm-started from the agent's last solution, preconditioned by the part
    of the operator that is block diagonal by time row: the own cost read instantaneously through the row
    operator and projected back, kron(Q_l, sum_k H_k D_l G_k) summed over the control's own lags l (the
    lag's read where it exists, D_l), assembled one time panel at a time from the rows of G_k and H_k
    (PanelRows) and LU-factored per time row (a singular block raises as the dense factorisation does);
  * best_response runs the whole thing for SpectralFiniteSolver (finite_spectral.py): the passive world and
    the impulse responses from the closed loop (closed_loop.py), the operators on the passive rows, the FOC
    system solved, the world of the response, the projection of the action kernels on the seen rows
    (maps_from_world, which reads the row operator's rows one panel at a time through panel_rows), and
    with want_decomp the decomposition and the second-order check (the representation error applies the
    row operator, reconstruction); the second-order check's form is applied by the same operators (a block
    of strategies at a time), assembled from their dense rows within second_order_dense.

The two solves are the same integrals and sums: they agree to BLAS rounding and the Krylov tolerance.
"""
from __future__ import annotations

from typing import Dict, Optional

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres

from .engine import dense_curvature_form, singular_system_message, symmetrize
from .spec import Agent
from .spectral_operators import FocOps, PanelRows, PathOp, ProjOps, RespOps, RowOps

__all__ = ["FocSystem", "best_response", "reconstruction", "panel_rows",
           "PathOp", "RowOps", "ProjOps", "RespOps", "FocOps", "PanelRows"]


class FocSystem:
    """The first-order-condition system of an agent's best response as an operator on its kept map unknowns,
    Amat gamma = -bvec with Amat = sum_k H_k Fu Resp G_k, its right-hand side, the time-row preconditioner
    and the GMRES solve."""

    RISK_KRYLOV_REDUCTION = 1e-4    # a warm-started risk-averse solve stops at this fraction of its initial residual (solve)

    def __init__(self, solver, agent: Agent, rowops: RowOps, projops: ProjOps, resp: RespOps, foc: FocOps,
                 Zpass: np.ndarray, phi_past, tilt=None):
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
        self.tilt = tilt                        # a risk-averse agent's correction (risk.Tilt), linear in the world
        # the right-hand side: the FOC of the passive world projected on the rows
        Zp = Zpass.reshape(self.nP, self.N, self.ncol)
        a = foc.atoms_of(Zp)
        b = np.zeros((self.nU, self.nR, self.Nm))
        for ui in range(self.nU):
            phi = foc.foc(ui, a)
            d = None if tilt is None else tilt.delta(ui, Zp, a, phi)       # before the past's part joins phi (in place)
            if phi_past is not None:
                phi[:, :c.nW] += phi_past[ui]
            if tilt is not None:
                phi = phi + d
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
        """Amat gamma on the full unknowns (nG,), or on a block of them (nG, B)."""
        single = gamma.ndim == 1
        G = gamma.reshape(self.nU, self.nR, self.Nm, -1)
        Zd = self.world_of(G)
        a = self.foc.atoms_of(Zd)
        out = np.zeros((self.nU, self.nR, self.Nm, G.shape[3]))
        for ui in range(self.nU):
            phi = self.foc.foc(ui, a)
            if self.tilt is not None:
                phi = phi + self.tilt.delta(ui, Zd, a, phi)
            out[ui] = self.projops.apply(phi)
        out = out.reshape(self.nG, -1)
        return out[:, 0] if single else out

    def matvec(self, x: np.ndarray) -> np.ndarray:
        self.matvecs += 1
        return self.reduce(self.apply(self.expand(np.asarray(x, dtype=float).ravel())))

    def matrix(self, with_tilt: bool = False) -> np.ndarray:
        """The system on the kept unknowns (n, n), assembled from the operators' dense rows: Amat[u, v] =
        sum_k H_k (Fu_u Resp_v) G_k over the columns of the world, restricted to the kept unknowns with a
        corner group's columns summed and its equations summed.  A risk-averse agent's correction is not in it
        (solve() handles it by preconditioned GMRES) unless with_tilt, which adds it column by column (the
        tests' check of that solve)."""
        from scipy.sparse import csr_matrix
        N, ncol, nU, nR, Nm, nG = self.N, self.ncol, self.nU, self.nR, self.Nm, self.nG
        Rm = csr_matrix((np.ones(self.kept.size), (self.inv, np.arange(self.kept.size))), shape=(self.n, self.kept.size))
        c = self.c
        if c.past is not None and c.P_lo > 0:
            # freeze_before: the kept unknowns lie on the panels from t_lo on, where the free maps act, the world
            # responds and the first-order conditions are projected (a lagged atom's read before t_lo meets a zero
            # response); every operator is built on those nodes only and the kept system is assembled directly:
            # the whole strip's products never exist
            lo = c._panel_ranges[c.P_lo][0]
            act = slice(lo, N); world = np.concatenate([p * N + np.arange(lo, N) for p in range(self.nP)])
            Gk = self.rowops.dense(lo); H = self.projops.dense(lo)
            Resp = [self.resp.dense(vi, lo)[world][:, act] for vi in range(nU)]
            blk = nR * Nm
            kp = [self.kept[(self.kept >= ui * blk) & (self.kept < (ui + 1) * blk)] - ui * blk for ui in range(nU)]
            off = np.cumsum([0] + [k.size for k in kp])
            A = np.zeros((self.kept.size, self.kept.size))
            Gr = [[Gk[k][act][:, kp[vi]] for vi in range(nU)] for k in range(ncol)]
            for ui in range(nU):
                Fu = self.foc.dense(ui, lo)[act][:, world]
                Hr = [H[kp[ui]][:, k * N + lo:(k + 1) * N] for k in range(ncol)]
                for vi in range(nU):
                    FR = Fu @ Resp[vi]
                    A[off[ui]:off[ui + 1], off[vi]:off[vi + 1]] = sum(Hr[k] @ (FR @ Gr[k][vi]) for k in range(ncol))
            return np.asarray(Rm @ A @ Rm.T)
        Gk = self.rowops.dense(); H = self.projops.dense()
        Resp = [self.resp.dense(vi) for vi in range(nU)]
        Amat = self._tilt_matrix() if (with_tilt and self.tilt is not None) else np.zeros((nG, nG))
        for ui in range(nU):
            Fu = self.foc.dense(ui)
            rows_u = slice(ui * nR * Nm, (ui + 1) * nR * Nm)
            for vi in range(nU):
                FR = Fu @ Resp[vi]
                Amat[rows_u, vi * nR * Nm:(vi + 1) * nR * Nm] += sum(H[:, k * N:(k + 1) * N] @ (FR @ Gk[k]) for k in range(ncol))
        return np.asarray(Rm @ Amat[np.ix_(self.kept, self.kept)] @ Rm.T)

    def _tilt_matrix(self) -> np.ndarray:
        """(nG, nG): the risk-averse correction's part of Amat, sum_u H (Delta_u (sum_v Resp_v G gamma_v)), applied to
        the identity a block of unknowns at a time (the correction is linear in the world, not diagonal in its columns)."""
        nG = self.nG; out = np.zeros((nG, nG)); step = self.tilt.chunk
        for j0 in range(0, nG, step):
            j1 = min(nG, j0 + step)
            E = np.zeros((nG, j1 - j0)); E[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
            Zd = self.world_of(E.reshape(self.nU, self.nR, self.Nm, -1))
            for ui in range(self.nU):
                out[ui * self.nR * self.Nm:(ui + 1) * self.nR * self.Nm, j0:j1] = self.projops.apply(self.tilt.delta(ui, Zd)).reshape(-1, j1 - j0)
        return out

    # ---- the preconditioner: the time-row-diagonal part of the operator
    def preconditioner(self):
        """Per time row (p, it) the block kron(Q_l, sum_k H_k D_l G_k) summed over the controls' own lags l on the
        row's kept unknowns (tied at the corners), LU-factored: [(reduced indices, lu)]."""
        c = self.c; g = c.g; N, Nm, nR, nU = self.N, self.Nm, self.nR, self.nU
        Ql = self._lag_forms()
        shifts = [c.panel_shift(c.rows[self.agent.name][r][3]) if c.rows[self.agent.name][r][3] > 0 else 0 for r in range(nR)]
        panel = PanelRows(self.solver, self.rowops, shifts)
        gecon, = get_lapack_funcs(("gecon",), (np.zeros((1, 1)),))
        hrows: Dict[tuple, tuple] = {}                          # (panel, row) -> the projection's rows of the row's map panel
        kept_pos = np.full(self.nG, -1, dtype=int); kept_pos[self.kept] = np.arange(self.kept.size)
        blocks = []
        ranges = c._panel_ranges
        for (p, it), idx in sorted(c.trow_by_pit.items()):
            if c.past is not None and (p >= c.P_T or p < c.P_lo):
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
            P = self._time_row_block(p, idx, lo, hi, parts, cols, shifts, panel, hrows, Ql)
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

    def _lag_forms(self) -> Dict[float, tuple]:
        """{lag l: (e^{-rho l} Q_l, D_l)} over the controls' own lags (FocOps.own_lags): Q_l the block of Q on the
        controls' atoms (u, l), (v, l), D_l where the lag's read exists (_lag_read); a lag whose block is zero is
        left out."""
        atoms, Q = self.foc.atoms, self.foc.Q
        ctrl = list(self.agent.controls); nU = self.nU
        Ql = {}
        for l, w in self.foc.own_lags().items():
            M = np.zeros((nU, nU))
            for ui, u in enumerate(ctrl):
                for vi, v in enumerate(ctrl):
                    if (u, l) in atoms and (v, l) in atoms:
                        M[ui, vi] = Q[atoms.index((u, l)), atoms.index((v, l))]
            if np.any(M):
                Ql[l] = (w * M, self._lag_read(l))
        return Ql

    def _time_row_block(self, p: int, idx: np.ndarray, lo: int, hi: int, parts, cols: np.ndarray, shifts, panel: PanelRows,
                        hrows: Dict[tuple, tuple], Ql) -> np.ndarray:
        """P (nU ncols, nU ncols): the block kron(Q_l, sum_k H_k D_l G_k) of one time row on its map unknowns cols
        (parts: per seen row r its unknowns), summed over the lags of Ql, from the rows of G_k at the action
        nodes idx (panel.sub) and the rows of H_k at the map nodes (projops.rows per (panel, row), kept in hrows
        while the panel lasts)."""
        c = self.c; N, Nm, nU = self.N, self.Nm, self.nU
        ranges = c._panel_ranges
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
        return P

    def _lag_read(self, l: float) -> np.ndarray:
        """D_l (N,): where the control's read l later of its own map shifted back by l exists (1 inside, 0
        past the horizon or the window): the row sums of read(-l, -l) map_shift(l)."""
        if l == 0.0:
            return np.ones(self.N)
        return np.asarray((self.c.read_sparse(-l, -l) @ self.c.map_shift_sparse(l)).sum(axis=1)).ravel()

    def solve(self, x0: Optional[np.ndarray] = None):
        """gamma (nU, nR, Nm) solving the system, and the GMRES iteration count (0 when factored).  Within
        settings.foc_dense_max the system is assembled (matrix) and factored, a singular one refused by the
        condition estimate of _solve_regular; beyond, GMRES to settings.foc_krylov_tol (relative to the
        right-hand side) from the warm start x0 (a full gamma), a system that does not converge within
        foc_krylov_maxiter iterations raising the ValueError of a singular system."""
        n = self.n
        st = self.solver.settings
        tol, maxiter = st.foc_krylov_tol, st.foc_krylov_maxiter
        if not self.solver.foc_free:
            A0 = self.matrix()
            if self.tilt is None:
                x = self.solver._solve_regular(self.agent, A0, -self.bvec)
                self.iterations = 0; self.residual = 0.0
                return self.expand(x).reshape(self.nU, self.nR, self.Nm), 0
            # a risk-averse agent: the risk-neutral system factored (its condition refused as always) preconditions GMRES
            # on the whole one, whose correction is applied (one application of Delta per iteration)
            lu = self.solver._factor_regular(self.agent, A0)
            prec = lambda r: lu_solve(lu, np.asarray(r, dtype=float).ravel(), check_finite=False)       # noqa: E731
        else:
            blocks = self.preconditioner()

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
        if self.tilt is not None:
            # a risk-averse agent: every product applies the correction, so none is spent twice (_gmres_left).  Warm-
            # started, the residual is reduced by RISK_KRYLOV_REDUCTION from the warm start's (or to foc_krylov_tol of
            # the right-hand side, whichever is larger): far from the equilibrium the warm start is far from this
            # response and 1e-12 of b is precision the next best response discards; near it the warm start's residual
            # is small and the floor is foc_krylov_tol as before
            x = np.zeros(n) if xs is None else xs.copy()
            r = b.copy() if xs is None else b - self.matvec(x)
            atol = tol * bn if xs is None else max(tol * bn, self.RISK_KRYLOV_REDUCTION * float(np.linalg.norm(r)))
            x, rnorm, self.iterations = _gmres_left(self.matvec, prec, b, x, r, atol, restart, -(-maxiter // restart))
            resid = rnorm / bn
            atol /= 10.0 * bn                           # the check below: 10 times the tolerance, relative to b
        else:
            atol = 0.0
            x, info = gmres(A, b, x0=xs, M=M, rtol=tol, atol=0.0, restart=restart, maxiter=-(-maxiter // restart))
            resid = float(np.linalg.norm(self.matvec(x) - b)) / bn
            self.iterations = self.matvecs - 1
        self.residual = resid
        if not resid <= 10 * max(tol, atol):
            raise ValueError(singular_system_message(self.agent.name) +
                             f" (GMRES did not converge: relative residual {resid:.1e} after {self.iterations} iterations)")
        return self.expand(x).reshape(self.nU, self.nR, self.Nm), self.iterations


def _gmres_left(matvec, psolve, b: np.ndarray, x: np.ndarray, r: np.ndarray, atol: float, restart: int, maxiter: int):
    """scipy.sparse.linalg.gmres (1.18) with the preconditioner psolve on the left: the same Arnoldi steps (modified
    Gram-Schmidt, Givens rotations), the same inner tolerance and its control, so the same iterates, from x with its
    residual r = b - A x given.  One difference: scipy checks each cycle's solution with a product of its own, b - A x;
    here that residual is r - (A V) y from the products the cycle made (A V, kept), the same vector up to round-off,
    so a solve costs one product fewer.  Returns (x, the norm of its residual, the Arnoldi steps taken)."""
    n = b.size
    eps = np.finfo(float).eps
    bnrm2 = float(np.linalg.norm(b))
    ptol_max_factor = 1.0
    ptol = float(np.linalg.norm(psolve(b))) * min(ptol_max_factor, atol / bnrm2)
    rnorm = float(np.linalg.norm(r))
    if rnorm < atol:
        return x, rnorm, 0
    lartg = get_lapack_funcs("lartg", dtype=x.dtype)
    v = np.empty([restart + 1, n]); av = np.empty([restart, n])
    h = np.zeros([restart, restart + 1]); givens = np.zeros([restart, 2])
    steps = 0; presid = 0.0
    for _ in range(maxiter):
        v[0, :] = psolve(r)
        tmp = np.linalg.norm(v[0, :])
        v[0, :] *= (1 / tmp)
        S = np.zeros(restart + 1); S[0] = tmp
        breakdown = False
        for col in range(restart):
            av[col, :] = matvec(v[col, :])
            w = psolve(av[col, :])
            h0 = np.linalg.norm(w)
            for k in range(col + 1):
                tmp = np.dot(v[k, :], w)
                h[col, k] = tmp
                w -= tmp * v[k, :]
            h1 = np.linalg.norm(w)
            h[col, col + 1] = h1
            v[col + 1, :] = w[:]
            if h1 <= eps * h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1, :] *= (1 / h1)
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k + 1]]
                h[col, [k, k + 1]] = [c * n0 + s * n1, -s.conj() * n0 + c * n1]
            c, s, mag = lartg(h[col, col], h[col, col + 1])
            givens[col, :] = [c, s]
            h[col, [col, col + 1]] = mag, 0
            tmp = -np.conjugate(s) * S[col]
            S[[col, col + 1]] = [c * S[col], tmp]
            presid = np.abs(tmp)
            steps += 1
            if presid <= ptol or breakdown:
                break
        if h[col, col] == 0:
            S[col] = 0
        y = np.zeros([col + 1])
        y[:] = S[:col + 1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp * h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]
        x += y @ v[:col + 1, :]
        r = r - y @ av[:col + 1, :]
        rnorm = float(np.linalg.norm(r))
        if rnorm <= atol or breakdown:
            break
        elif presid <= ptol:
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        ptol = presid * min(ptol_max_factor, atol / rnorm)
    return x, rnorm, steps


def best_response(solver, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False, project: bool = True):
    """SpectralFiniteSolver.best_response (with or without a past): the passive world and the impulse responses
    from the closed loop, the operators on the passive rows, the FOC system solved (factored or by GMRES), the
    world of the response and the projection of the action kernels on the seen rows."""
    c = solver.c; N, ncol = c.N, c.ncol
    nU = len(agent.controls); nP = len(c.prim)
    # the on-path world is built with the ordinary spike responses R0 (the others' filters); the first-order
    # condition and the second-order form see the agent's deviations answered by the players privy to them (R)
    Zpass, R0 = solver._spikes(c, maps, agent)
    R = solver._impulse_responses(agent, maps, R0)
    Roff = R
    if c.cont is not None:                       # the envelope responses: the agent's own reaction off on the buffer too
        Roff = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls, own_frozen=False)[:, ncol:]
        Roff = solver._impulse_responses(agent, maps, Roff)
    Zpass = solver._passive_world(agent, maps, Zpass, R0)
    ytil, yinst = solver._passive_rows(agent, Zpass)
    rowops = RowOps(solver, agent, ytil, yinst)
    projops = ProjOps(solver, agent, ytil, yinst)
    resp = RespOps(solver, agent, R0)
    fixed = getattr(solver, "_fixed_actions", {}).get(agent.name)
    if fixed is not None:
        # freeze_before: the agent's own actions on the fixed panels are known; their response joins the passive
        # world the first-order conditions see (the right-hand side, the loss atoms), while the rows the free map
        # unknowns read stay the ones with the agent's strategy off everywhere on [0, T]: the same representation
        # as the whole strip's solve, whose fixed point this reproduces with the early unknowns moved to the right
        Zp = Zpass.reshape(nP, N, ncol).copy()
        for ui in range(nU):
            Zp += resp.apply(ui, fixed[ui])
        Zpass = Zp.reshape(nP * N, ncol)
    foc = FocOps(solver, agent, Roff)
    phi_past = solver._foc_affine(agent, foc)
    tilt = solver._tilt(agent, maps, foc, Roff) if agent.risk_aversion else None
    system = FocSystem(solver, agent, rowops, projops, resp, foc, Zpass, phi_past, tilt)
    gamma, iters = system.solve(solver._last_gamma.get(agent.name))
    if solver.foc_free or tilt is not None:
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
        if R is not R0:
            system.resp = RespOps(solver, agent, R)          # the second-order form is about deviations
        _decompose(solver, agent, out, system, maps)
        if phi_past is not None:
            for ui, u in enumerate(agent.controls):
                for part in ("foc", "physical"):
                    out["decomp"][u][part] = out["decomp"][u][part] + phi_past[ui]
    return (solver._project(agent, Zfull, solver._map_part(agent, Zfull, cact)) if project else None), out


def _decompose(solver, agent: Agent, out: dict, system: FocSystem, maps) -> None:
    """The second-order check on the operators' form and the FOC decomposition (foc / physical / wedge, the
    physical part through the impulse responses with every reaction off)."""
    c = solver.c; ncol = c.ncol; N = c.N; nP = len(c.prim)
    Zfull = out["Zfull"].reshape(nP, N, ncol)
    out["second_order"] = solver._shared_second_order(agent, lambda: _second_order(solver, agent, system))
    fphys = FocOps(solver, agent, solver._physical_responses(agent, ncol))
    a = system.foc.atoms_of(Zfull); ap = fphys.atoms_of(Zfull)
    dec = {}
    for ui, u in enumerate(agent.controls):
        phi = system.foc.foc(ui, a); phi_phys = fphys.foc(ui, ap)
        dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
        if system.tilt is not None:
            # a risk-averse agent: the kernel whose projection vanishes is Sigma f, the risk-neutral one plus the
            # correction (risk.py); physical + wedge + risk = foc
            risk = system.tilt.delta(ui, Zfull)
            dec[u]["risk"] = risk
            dec[u]["foc"] = phi + risk
    out["decomp"] = dec


def _second_order(solver, agent: Agent, system: FocSystem) -> Optional[dict]:
    """The second-order check (EngineBase._second_order's contract) on the operators: the form M = T' G T on
    the kept strategy, T the strategy -> world map (RowOps then RespOps), G the loss form (the atoms' kernels
    under Q and the sparse mass; a past's initial columns under the point form of the line s = 0, the time
    weights on the diagonal, as expected_cost integrates them).  Within second_order_dense the form is
    assembled from the operators' dense rows (_dense_form) and diagonalised; beyond, its extreme eigenvalues
    come from Lanczos on the matvec of the applied operators."""
    c = solver.c
    nW, ncol, N, nP = c.nW, c.ncol, c.N, len(c.prim)
    nU, nR, Nm = system.nU, system.nR, system.Nm
    idx = np.where(np.tile(solver._identified(agent), nU))[0]
    atoms, Q, _ = c.loss[agent.name]
    mass = c.cost_mass_sparse()
    imass = c.time_mass(c.rho)[:c.Nd] if ncol > nW else None
    AO = system.foc.AO

    mf = system.foc.m_flow
    terminal = (c.terminal or {}).get(agent.name)
    if terminal:
        IT, wT = c.terminal_quadrature
        tform = np.exp(-c.rho * c.T) * (IT.T @ (wT[:, None] * IT.toarray()))     # N x N: e^{-rho T} int f(T, T - s) g(T, T - s) ds
        tAO = system.foc.AO[mf:]
        if ncol > nW:
            r = c.terminal_point.toarray()
            tpoint = np.exp(-c.rho * c.T) * (r.T @ r)                           # N x N: e^{-rho T} f(T) g(T) on the line s = 0

    def GT(Zd):                                     # the loss form on the world (nP, N, ncol, B), column by column
        b = np.tensordot(Q, system.foc.atoms_of(Zd)[:mf], axes=1)      # (m, N, ncol, B)
        B = b.shape[3]
        Mb = np.zeros_like(b)
        for j in range(len(atoms)):
            Mb[j, :, :nW] = (mass @ b[j, :, :nW].reshape(N, -1)).reshape(N, nW, B)
        if imass is not None:
            Mb[:, c.diag, nW:] = imass[None, :, None, None] * b[:, c.diag, nW:]
        out = np.zeros((nP, N, ncol, B))
        for j, (p, A) in enumerate(AO[:mf]):
            out[p] += (A.T @ Mb[j].reshape(N, -1)).reshape(N, ncol, B)
        if terminal:                                # the terminal loss's form on the shocks' columns
            zt = system.foc.atoms_of(Zd)[mf:]
            bt = np.tensordot(terminal[1], zt, axes=1)
            for j, (p, A) in enumerate(tAO):
                Mt = (tform @ bt[j, :, :nW].reshape(N, -1)).reshape(N, nW, B)
                out[p, :, :nW] += (A.T @ Mt.reshape(N, -1)).reshape(N, nW, B)
                if ncol > nW:                       # the initial shocks' columns at T: the point form of the corner
                    Mi = (tpoint @ bt[j, :, nW:].reshape(N, -1)).reshape(N, ncol - nW, B)
                    out[p, :, nW:] += (A.T @ Mi.reshape(N, -1)).reshape(N, ncol - nW, B)
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
        Mfull = _dense_form(solver, agent, system, idx)
        w = np.linalg.eigvalsh(symmetrize(Mfull))
        lo, hi = float(w[0]), float(w[-1])
    else:
        res = solver._lanczos_extremes(lambda v: matvec(v)[:, 0], n)
        if "message" in res:
            return res
        lo, hi = res["lo"], res["hi"]
    scale = max(abs(lo), abs(hi), 1e-300)
    out = {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -solver.SECOND_ORDER_TOL * scale), "converged": True}
    if system.tilt is not None:
        # the entropic cost's curvature along a change d of the strategy is E^Q[C''] + theta Var^Q(C') >= E^Q[C''] =
        # tr(Sigma B_d) >= tr(B_d) = E[C''], the form above, when the loss Hessian is positive semidefinite (B_d >= 0 and
        # Sigma = (I - theta K)^-1 >= I): the expected cost's curvature is then a lower bound of the objective's
        QT = ((c.terminal or {}).get(agent.name) or (None, None, None))[1]
        psd = all(M is None or np.asarray(M).size == 0 or np.linalg.eigvalsh(0.5 * (M + M.T))[0] >= -1e-12 * max(1.0, np.abs(M).max())
                  for M in (Q, QT))
        if not psd:
            return {"min": None, "max": None, "ok": None, "converged": False,
                    "message": f"{agent.name} is risk averse and its loss Hessian is not positive semidefinite: the expected cost's "
                               "curvature does not bound the entropic cost's, whose second-order condition is not checked"}
        out["bound"] = "entropic"          # the form is the expected cost's, a lower bound of the entropic cost's curvature
    return out


def _dense_form(solver, agent: Agent, system: FocSystem, idx: np.ndarray) -> np.ndarray:
    """The second-order form on the kept strategies idx, (n, n), from the operators' dense rows: M[u, v] =
    sum_k G_k' (Resp_u' G_(k) Resp_v) G_k with G_(k) the column's loss form (the channels': the atoms' reads
    under kron(Q, mass); an initial shock's: under the point mass of the line s = 0), the inner form H_uv
    N x N through the responding primaries' nodes only, and the sum over the columns of one loss form one
    product of the stacked row operators restricted per column to the rows whose block of G_k is not zero."""
    from scipy.sparse import diags
    c = solver.c; N, nW, ncol, nP = c.N, c.nW, c.ncol, len(c.prim)
    nU, nR, Nm = system.nU, system.nR, system.Nm
    atoms, Q, _ = c.loss[agent.name]
    AO = system.foc.AO
    Gk = system.rowops.dense()                                                                  # (ncol, N, nR Nm)
    Resp = [system.resp.dense(ui) for ui in range(nU)]                                          # (nP N, N)

    def loss_form(mass, Q=Q, AO=AO[:system.foc.m_flow]):                                       # AO' kron(Q, mass) AO, dense (nP N, nP N)
        G = np.zeros((nP * N, nP * N))
        for i, (p, Ai) in enumerate(AO):
            for j, (p2, Aj) in enumerate(AO):
                if Q[i, j] != 0.0:
                    G[p * N:(p + 1) * N, p2 * N:(p2 + 1) * N] += Q[i, j] * (Ai.T @ (mass @ Aj)).toarray()
        return G
    G0 = loss_form(c.cost_mass_sparse())
    terminal = (c.terminal or {}).get(agent.name)
    if terminal:                                    # the terminal loss's form on the shocks' columns, e^{-rho T} at T
        from scipy.sparse import csr_matrix
        IT, wT = c.terminal_quadrature
        tmass = csr_matrix(np.exp(-c.rho * c.T) * (IT.T @ (wT[:, None] * IT.toarray())))
        G0 = G0 + loss_form(tmass, terminal[1], AO[system.foc.m_flow:])
    forms = [(G0, slice(0, nW))]
    if ncol > nW:
        w = np.zeros(N); w[c.diag] = c.time_mass(c.rho)[:c.Nd]
        Gi = loss_form(diags(w, format="csr"))
        if terminal:                                # an initial shock's column at T: the point form of the corner
            r = c.terminal_point.toarray()
            Gi = Gi + loss_form(csr_matrix(np.exp(-c.rho * c.T) * (r.T @ r)), terminal[1], AO[system.foc.m_flow:])
        forms.append((Gi, slice(nW, ncol)))
    return dense_curvature_form(Resp, Gk, forms, nU, nR, Nm, idx)


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
