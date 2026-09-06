"""The means of the spectral finite engine on the time line: deterministic paths (targets, constant drifts,
initial states) on the time nodes.

TimeLineOps is the compiled model's part (a mixin of SpectralCompiled, spectral_compiled.py): the discounted
quadrature weights on the time nodes, the read of a kernel at age 0 on every time node, the one-dimensional
Volterra operator of the mean dynamics and the read of a path at t - lag.  SpectralMeans is the solver's
part (a mixin of SpectralFiniteSolver, finite_spectral.py): the mean system on the line s = 0 or on the time
line, its solve, the loss atoms' mean paths, the mean part of the costs and the result's mean fields; its
first-order conditions go through the FOC operators of spectral_operators.py (FocOps).  The stationary and
cell engines carry their own copies (stationary.py, finite.py) until the mean layer is shared (step C5)."""
from __future__ import annotations

from typing import Dict, Optional

from functools import cached_property
import warnings

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve

from . import finite_free
from .grid import bary_rows
from .settings import tunable
from .spec import Agent


class TimeLineOps:
    """The time-line operators of SpectralCompiled (mixed in): weights and reads on the time nodes tm (Nt), the
    mean paths' carrier mean_embed and the line s = 0 (diag) being the compiled model's."""

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



class SpectralMeans:
    """The mean system of SpectralFiniteSolver (mixed in), over the compiled model self.c (SpectralCompiled) and the
    model self.model; EngineBase's _mean_part hook is filled in here."""
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
            for i in range(nX):
                M[blk(i), blk(i)] = np.eye(Nt)
            for (i, p), B in c._state_blocks.items():
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
            foc = finite_free.FocOps(self, a, R)
            pre = np.zeros((len(atoms), N))                                 # lagged atoms read before zero: the old means
            bq = np.zeros((len(atoms), N))                                  # the targets' kernels (constant)
            for i, (nm, lag) in enumerate(atoms):
                before = self._mean_before(nm, lag)
                if before is not None:
                    pre[i] = E @ before
                bq[i] = q[i] * ones
            for ui, u in enumerate(a.controls):
                row = blk(c.index[u])
                for p in range(nP):                                         # the condition on primary p's embedded path
                    Zp = np.zeros((nP, N, Nt)); Zp[p] = E
                    M[row, blk(p)] = foc.apply(ui, Zp)[diag]
                b[row] = -foc.on_qzeta(ui, bq)[diag] - foc.foc(ui, pre)[diag]
        return M, b

    def _mean_system_line(self, maps: Dict[str, np.ndarray]):
        """mean_system on the time line (a strip cut at age L < T).  The state rows are xbar(t) = e^{At} x0 +
        int_0^t e^{A(t-r)} (inputs at their mean paths + const) dr through mean_volterra, a lagged input read
        before zero at the past's constant.  A control's rows are its mean first-order condition at every time
        node: the per-atom operators of finite_free.FocOps (the instantaneous derivative, the discounted own
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
            foc = finite_free.FocOps(self, a, R)
            for ui, u in enumerate(a.controls):
                row = blk(c.index[u])
                for j in range(len(atoms)):
                    bj = np.zeros((len(atoms), c.N, Nt)); bj[j] = E
                    Oj = S0 @ foc.on_qzeta(ui, bj)                              # the condition's read of (Q zeta + q)_j's mean
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
