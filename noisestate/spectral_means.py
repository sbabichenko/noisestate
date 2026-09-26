"""The means of the spectral finite engine on the time line: deterministic paths (targets, constant drifts,
initial states) on the time nodes.

TimeLineOps is the compiled model's part (a mixin of SpectralCompiled, spectral_compiled.py): the discounted
quadrature weights on the time nodes, the read of a kernel at age 0 on every time node, the one-dimensional
Volterra operator of the mean dynamics and the read of a path at t - lag.  SpectralMeans is the solver's
part (a mixin of SpectralFiniteSolver, finite_spectral.py): the mean system on the line s = 0 or on the time
line and the loss atoms' mean paths, the hooks of EngineBase's mean layer (engine.py: the assembly, the solve,
the mean costs and the result's mean fields, shared by the three engines); its first-order conditions go
through the FOC operators of spectral_operators.py (FocOps)."""
from __future__ import annotations

from typing import Dict, Optional

from functools import cached_property

import numpy as np
from numpy.polynomial import legendre

from . import finite_free
from .grid import bary_rows
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
    def terminal_reads(self):
        """The reads at the end of the game T, for a terminal loss: (IZ, IR, disc) with IZ (N x N, CSR) reading a
        kernel at (T, age T - s_k) for every node k (the node's shock at T), IR at (T, age T - t_k) (an impulse at
        the node's time t_k, seen at T) and disc = e^{-rho (T - t_k)}."""
        g = self.g; T = float(self.T); ones = np.full(self.N, T)
        IZ = g.interp_sparse(ones, T - g.s, side_t=-1)
        IR = g.interp_sparse(ones, T - g.t, side_t=-1)
        return IZ, IR, np.exp(-self.rho * (T - g.t))

    @cached_property
    def terminal_quadrature(self):
        """(I, w): I (nq x N, CSR) reads a kernel at (T, T - s) on Gauss points s of every time panel of [0, T] and
        w (nq,) their weights, so that sum_q w_q f(T, T - s_q) is int_0^T f(T, T - s) ds: the variance of a
        quantity at T over the shocks born in [0, T]."""
        g = self.g; T = float(self.T)
        xg, wg = legendre.leggauss(g.nt + 2)
        s, w = [], []
        for p in range(self.P_T):
            t0, t1 = g.bp[p], min(g.bp[p + 1], T)
            if t1 - t0 < 1e-14:
                continue
            s.append(0.5 * (t1 - t0) * xg + 0.5 * (t1 + t0)); w.append(0.5 * (t1 - t0) * wg)
        s = np.concatenate(s); w = np.concatenate(w)
        return g.interp_sparse(np.full(s.size, T), T - s, side_t=-1), w

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
                M = self.g.interp(t, a, side_t=self.tm_side) @ self.mean_embed
                # a node whose t - lag is 0 read from the left is the path just before zero, the history (the old
                # means, _mean_before's), never the path's value at 0+: the grid has no piece below zero to read, and
                # took the one above, putting a jump into the node's panel (a 3e-3 error on every later mean)
                M[(np.abs(t) <= 1e-12) & (self.tm_side < 0)] = 0.0
                self._mean_reads[key] = M
        return self._mean_reads[key]



class SpectralMeans:
    """The mean hooks of SpectralFiniteSolver (mixed in) for EngineBase's mean layer, over the compiled model
    self.c (SpectralCompiled) and the model self.model.  A path is carried as a kernel constant in shock age
    (mean_embed), on which the kernels' own operators restricted to the line s = 0 (the nodes `diag`: a kernel's
    response to a shock at time 0) are the path's: a lagged read is the path at t - lag, zero before 0, the
    Volterra propagation integrates from 0 and the continuation reads the path along s = 0.  On a strip cut at
    age L below the horizon (a past's window shorter than T, or a stationary continuation, whose buffer follows
    T) the line s = 0 does not reach T, and the system is built on the time line instead (line=True): the
    dynamics by a one-dimensional Volterra operator, each control's condition on the line age = 0 (the birth of
    a shock at t, its continuation running to t + L through the buffer's frozen maps), the paths on the buffer
    frozen at the continuation's stationary means."""

    def _mean_times(self) -> np.ndarray:
        return self.c.tm

    def _mean_weights(self) -> np.ndarray:
        return self.c.time_mass(self.c.rho)

    def _mean_driven(self) -> bool:
        c = self.c
        return (super()._mean_driven() or (c.past is not None and any(c.past.mean(n) for n in c.prim))
                or (c.cont is not None and any(c.cont.means.get(n, 0.0) for n in c.prim)))

    def _on_line(self, line: Optional[bool]) -> bool:
        #  The two systems agree to rounding wherever both exist (5e-15 on the delayed example with targets, the
        #  check tests/test_transition_means.py makes); the diagonal one, on grid nodes, is used where it exists.
        return self.c.Nd < self.c.Nt if line is None else line

    def _mean_system_diag(self, maps: Dict[str, np.ndarray]):
        """mean_system on the line s = 0 (every time panel below the window)."""
        return self.mean_system(maps, line=False)

    def _mean_system_line(self, maps: Dict[str, np.ndarray]):
        """mean_system on the time line (a strip cut at age L < T)."""
        return self.mean_system(maps, line=True)

    def _mean_dynamics(self, line: Optional[bool] = None):
        """The states' rows, xbar(t) = e^{At} x0 + int_0^t e^{A(t-r)} (inputs at their mean paths + const) dr: on
        the line s = 0 the kernels' state blocks restricted to the nodes `diag` applied to the embedded paths;
        on the time line mean_volterra, a lagged input read before zero at the past's constant."""
        c = self.c; N, Nt, nP, nX = c.N, c.Nt, len(c.prim), c.nX
        E = c.mean_embed
        blk = lambda i: slice(i * Nt, (i + 1) * Nt)
        Mx = np.zeros((nX * Nt, nP * Nt)); bx = np.zeros(nX * Nt)
        if not nX:
            return Mx, bx, []
        x0 = self._mean_start(); EA = c.expA(c.tm)
        if not self._on_line(line):
            diag = c.diag; ones = np.ones(N)
            for i in range(nX):
                Mx[blk(i), blk(i)] = np.eye(Nt)
            for (i, p), B in c._state_blocks.items():
                Mx[blk(i), blk(p)] -= B[diag] @ E
            for i in range(nX):
                bx[blk(i)] = EA[:, i, :] @ x0 + sum((c.const[j] * (c.Vol[i, j][diag] @ ones) for j in range(nX) if c.const[j]), 0.0)
                for si, (nm, lag), coef in c.state_inputs:                  # lagged inputs read before zero: the old means
                    pre = self._mean_before(nm, lag)
                    if pre is not None:
                        bx[blk(i)] += coef * (c.Vol[i, si][diag] @ (E @ pre))
        else:
            V = c.mean_volterra; ones = np.ones(Nt)
            for i in range(nX):
                Mx[blk(i), blk(i)] = np.eye(Nt)
                bx[blk(i)] = EA[:, i, :] @ x0 + sum((c.const[j] * (V[i, j] @ ones) for j in range(nX) if c.const[j]), 0.0)
                for si, (nm, lag), coef in c.state_inputs:
                    Mx[blk(i), blk(c.index[nm])] -= coef * (V[i, si] @ c.mean_read(lag))
                    pre = self._mean_before(nm, lag)
                    if pre is not None:
                        bx[blk(i)] += coef * (V[i, si] @ pre)
            self._freeze_buffer(Mx, bx, range(nX))
        return Mx, bx, []

    def _mean_conditions(self, agent: Agent, maps: Dict[str, np.ndarray], line: Optional[bool] = None):
        """The agent's mean first-order condition at every time node, one block per control: the per-atom
        operators of finite_free.FocOps (the instantaneous derivative, the discounted own lagged reads, the
        continuation through the passive-world impulse responses) applied to the embedded mean paths and read
        on the line s = 0 (diag), or on the time line read on the line age = 0 (mean_line0) with the mean of a
        lagged atom the path at t - lag (mean_read; the strip's own read of a lagged kernel is zero below age
        lag, which is right for a shock and wrong for a path) and the continuation to T + L (the envelope
        responses: the agent's own reaction off on the buffer as well)."""
        c = self.c; N, Nt, nP = c.N, c.Nt, len(c.prim); a = agent
        E = c.mean_embed
        blk = lambda i: slice(i * Nt, (i + 1) * Nt)
        Mu = np.zeros((len(a.controls) * Nt, nP * Nt)); bu = np.zeros(len(a.controls) * Nt)
        if not self._on_line(line):
            diag = c.diag; ones = np.ones(N)
            R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls)[:, c.ncol:]
            R = self._impulse_responses(a, maps, R)
            foc = finite_free.FocOps(self, a, R)
            atoms, Q, q = foc.atoms, foc.Q, foc.q                          # the flow loss's atoms, then a terminal loss's
            pre = np.zeros((len(atoms), N))                                 # lagged atoms read before zero: the old means
            bq = np.zeros((len(atoms), N))                                  # the targets' kernels (constant)
            for i, (nm, lag) in enumerate(atoms):
                before = self._mean_before(nm, lag)
                if before is not None:
                    pre[i] = E @ before
                bq[i] = q[i] * ones
            for ui in range(len(a.controls)):
                for p in range(nP):                                         # the condition on primary p's embedded path
                    Zp = np.zeros((nP, N, Nt)); Zp[p] = E
                    Mu[blk(ui), blk(p)] = foc.apply(ui, Zp)[diag]
                bu[blk(ui)] = -foc.on_qzeta(ui, bq)[diag] - foc.foc(ui, pre)[diag]
            return Mu, bu
        S0 = c.mean_line0; ones = np.ones(Nt)
        reads = {}
        def read(lag):
            if lag not in reads:
                reads[lag] = c.mean_read(lag)
            return reads[lag]
        R = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls, own_frozen=False)[:, c.ncol:]
        R = self._impulse_responses(a, maps, R)
        foc = finite_free.FocOps(self, a, R)
        atoms, Q, q = foc.atoms, foc.Q, foc.q
        for ui in range(len(a.controls)):
            row = blk(ui)
            for j in range(len(atoms)):
                bj = np.zeros((len(atoms), N, Nt)); bj[j] = E
                Oj = S0 @ foc.on_qzeta(ui, bj)                              # the condition's read of (Q zeta + q)_j's mean
                if q[j]:
                    bu[row] -= q[j] * (Oj @ ones)
                for i, (nm, lag) in enumerate(atoms):
                    if Q[j, i]:
                        Mu[row, blk(c.index[nm])] += Q[j, i] * (Oj @ read(lag))
                        pre = self._mean_before(nm, lag)
                        if pre is not None:
                            bu[row] -= Q[j, i] * (Oj @ pre)
        self._freeze_buffer(Mu, bu, [c.index[u] for u in a.controls])
        return Mu, bu

    def _freeze_buffer(self, M: np.ndarray, b: np.ndarray, prims) -> None:
        """On the time line with a continuation: the rows of the given primaries (in the order of M's row blocks)
        on the buffer's time nodes are frozen at the continuation's stationary means (like the maps: the closure
        assumes the transition has settled by T)."""
        c = self.c; Nt = c.Nt
        if c.cont is None:
            return
        frozen = np.flatnonzero(np.arange(Nt) >= c.P_T * c.g.nt)
        for k, p in enumerate(prims):
            rows = frozen + k * Nt; cols = frozen + p * Nt
            M[rows, :] = 0.0; M[rows, cols] = 1.0; b[rows] = float(c.cont.means.get(c.prim[p], 0.0))

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

    def _terminal_mean_cost(self, agent: Agent, zbar: np.ndarray) -> float:
        """e^{-rho T} (1/2 xbar_T' Q_T xbar_T + q_T' xbar_T) over the terminal loss's atoms (states at T)."""
        c = self.c; Nt = c.Nt
        terminal = (c.terminal or {}).get(agent.name)
        if not terminal:
            return 0.0
        atoms, QT, qT = terminal
        read = c.g.interp_sparse(np.array([float(c.T)]), np.zeros(1), side_t=-1) @ c.mean_embed     # a path's value at T
        xT = np.array([float((read @ zbar[c.index[nm] * Nt:(c.index[nm] + 1) * Nt])[0]) for (nm, lag) in atoms])
        return float(np.exp(-c.rho * c.T) * (0.5 * xT @ QT @ xT + qT @ xT))

    def _terminal_constant(self, agent: Agent) -> float:
        c = self.c
        return float(np.exp(-c.rho * c.T) * (c.terminal_constant or {}).get(agent.name, 0.0))

    def _mean_atoms(self, zbar: np.ndarray, atoms) -> np.ndarray:
        """The loss atoms' mean paths (m, Nt) from the primaries' paths: the kernels' atom operators on the embedded
        paths, read on the line s = 0 (a lagged atom is the path at t - lag, zero before 0)."""
        c = self.c; Nt = c.Nt
        if not atoms:
            return np.zeros((0, Nt))
        if c.Nd < Nt:                                     # the time line (see _mean_conditions): the path at t - lag
            out = np.stack([c.mean_read(lag) @ zbar[c.index[nm] * Nt:(c.index[nm] + 1) * Nt] for (nm, lag) in atoms])
        else:
            Zm = [c.mean_embed @ zbar[p * Nt:(p + 1) * Nt] for p in range(len(c.prim))]
            # an atom operator is one N x N block (the shift into its own primary) in a row of zeros, and only
            # the line s = 0 is read: take the block against that primary's path on those rows alone, never the
            # full-width dense operator (144 MB per solve of the delayed example at 10 nodes, two thirds zeros)
            out = np.stack([c.atom_sparse(at)[c.diag] @ Zm[c.index[at[0]]] for at in atoms])
        for i, (nm, lag) in enumerate(atoms):
            pre = self._mean_before(nm, lag)
            if pre is not None:
                out[i] += pre[:out.shape[1]]
        return out
