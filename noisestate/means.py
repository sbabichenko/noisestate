"""The mean layer shared by the three engines: the means (targets, constant drifts, initial states) and the mean
part of the costs, solved once at the end of a solve.

The kernels never depend on the means (the model is linear-quadratic-Gaussian), so the means are one linear
system on the primaries' means (states then controls, Nt values each: one on the stationary engine, the time
nodes on the finite ones) under the equilibrium strategies: the states' rows from the mean dynamics, each
control's rows from its owner's mean first-order condition, the kernels' condition applied to a deterministic
path (common knowledge: no information constraint) with the targets q as the driver in place of the shocks.
MeanLayer assembles the joint (xbar, ubar) system from the mean hooks each engine fills in (its class docstring
lists them; EngineBase's hooks table in engine.py says which engine overrides which), solves it with the
rcond guard, integrates the mean cost by the engine's quadrature and fills the result's mean fields."""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import scipy.linalg as sla

from .settings import tunable
from .spec import Agent


class MeanLayer:
    """The mean hooks (the engine's part, Nt = 1 on the stationary engine, the time nodes on the finite ones;
    nP primaries, states first):

        _mean_times()             the time nodes (Nt,) of the mean paths, None for stationary constants (floats)
        _mean_start()             the mean state at time zero (nX,)
        _mean_driven()            whether anything moves the means; else solve_means returns zeros with no solve
        _mean_dynamics()          (Mx (nX Nt, nP Nt), bx (nX Nt,), pinned rows): the states' rows
        _mean_conditions(a, maps) (Mu (nU Nt, nP Nt), bu (nU Nt,)): the agent's rows, one block per control
        _mean_atoms(zbar, atoms)  the loss atoms' mean paths (m, Nt) from the primaries' means (nP Nt,)
        _mean_weights()           the discounted quadrature weights (Nt,) of the mean cost

    and the layer on them: mean_system, solve_means, mean_cost, _mean_part (the _finish hook of EngineBase).
    Needs self.c (the compiled model: prim, index, nX, const, x0, loss) and self.model."""
    MEAN_RCOND = tunable("mean_rcond")          # a mean system whose reciprocal condition estimate is below this is singular (settings)

    def _mean_times(self) -> Optional[np.ndarray]:
        """Mean hook (finite engines): the time nodes (Nt,) the mean paths live on, None when the means are
        stationary constants (Nt = 1, res.means holds floats)."""
        return None

    def _mean_nodes(self) -> int:
        t = self._mean_times()
        return 1 if t is None else len(t)

    def _mean_start(self) -> np.ndarray:
        """Mean hook: the mean state at time zero, (nX,): the model's per-state `initial` [c.x0]; the spectral
        engine takes the past's means where no `initial` is given."""
        return np.array(self.c.x0, dtype=float)

    def _mean_driven(self) -> bool:
        """Mean hook: whether anything moves the means (a constant drift, a target q, a nonzero initial state);
        when nothing does solve_means returns zeros with no solve."""
        c = self.c
        return bool(c.const.any() or any(q.any() for atoms, Q, q in c.loss.values()) or self._mean_start().any())

    def _mean_dynamics(self, **variant):
        """Mean hook (abstract): the states' rows of the mean system, (Mx (nX Nt, nP Nt), bx (nX Nt,), pinned):
        how the states' means respond to the controls' means and the constants (A xbar + inputs + const = 0 on
        the stationary engine; the Volterra operator xbar(t) = e^{At} x0 + int_0^t e^{A(t - r)} (inputs + const) dr
        on the finite engines), the columns over the primaries (states then controls, Nt values each); `pinned`
        the rows of the states whose level the system does not determine (the stationary random walk with no
        inputs, set to 0 after the solve)."""
        raise NotImplementedError

    def _mean_conditions(self, agent: Agent, maps: Dict[str, np.ndarray], **variant):
        """Mean hook (abstract): the agent's mean first-order conditions, (Mu (nU Nt, nP Nt), bu (nU Nt,)), one
        block of Nt rows per control: the kernels' first-order condition applied to the mean paths with no
        information constraint (a deterministic path is common knowledge) and the targets q as the driver in
        place of the shocks: the instantaneous derivative of 1/2 z'Qz + q'z in the control, the discounted own
        lagged reads, and the discounted continuation of every other loss atom through its passive-world
        impulse response under `maps` (the DC gain int e^{-rho a} R_j(a) da on the stationary engine, the
        time-line operator int_t^T e^{-rho (t' - t)} R_j(t', t) dt' on the finite ones); a myopic agent has
        the instantaneous term only."""
        raise NotImplementedError

    def _mean_atoms(self, zbar: np.ndarray, atoms) -> np.ndarray:
        """Mean hook: the loss atoms' mean paths (m, Nt) from the primaries' means `zbar` (nP Nt,): a lagged atom
        is the path at t - lag (the past's constant before zero) [the primary's mean: stationary constants]."""
        c = self.c; Nt = self._mean_nodes()
        return np.array([zbar[c.index[nm] * Nt:(c.index[nm] + 1) * Nt] for (nm, lag) in atoms]).reshape(len(atoms), Nt)

    def _mean_weights(self) -> np.ndarray:
        """Mean hook: the discounted quadrature weights (Nt,) of the mean cost, int_0^T e^{-rho t} f(t) dt = w @ f
        [1: the stationary flow loss per unit time]."""
        return np.ones(1)

    def _assemble_means(self, maps: Dict[str, np.ndarray], **variant):
        c = self.c; Nt = self._mean_nodes(); nP, nX = len(c.prim), c.nX
        M = np.zeros((nP * Nt, nP * Nt)); b = np.zeros(nP * Nt)
        Mx, bx, pinned = self._mean_dynamics(**variant)
        M[:nX * Nt] = Mx; b[:nX * Nt] = bx
        for a in self.model.agents:
            Mu, bu = self._mean_conditions(a, maps, **variant)
            for ui, u in enumerate(a.controls):
                row = slice(c.index[u] * Nt, (c.index[u] + 1) * Nt)
                M[row] = Mu[ui * Nt:(ui + 1) * Nt]; b[row] = bu[ui * Nt:(ui + 1) * Nt]
        return M, b, pinned

    def mean_system(self, maps: Dict[str, np.ndarray], **variant):
        """The linear system M zbar = b of the primaries' means (states then controls, Nt values each: one on
        the stationary engine, the time nodes on the finite ones) under the strategies `maps`: the states'
        rows from _mean_dynamics, each control's from its owner's _mean_conditions.  Linear in (q, x0, const):
        one direct solve, no iteration.  `variant` is passed to both hooks (the spectral engine's line=)."""
        return self._assemble_means(maps, **variant)[:2]

    def solve_means(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """The primaries' means (nP Nt,) under `maps`: exactly zero, with no solve, when nothing drives them
        (_mean_driven); else the direct solve of mean_system, refusing a singular system (reciprocal condition
        estimate below MEAN_RCOND), the pinned states exactly zero."""
        c = self.c
        if not self._mean_driven():
            return np.zeros(len(c.prim) * self._mean_nodes())
        M, b, pinned = self._assemble_means(maps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sla.LinAlgWarning)
            lu, piv = sla.lu_factor(M, check_finite=False)
        gecon, = sla.get_lapack_funcs(("gecon",), (lu,))
        rcond = float(gecon(lu, np.linalg.norm(M, 1))[0])
        if not rcond > self.MEAN_RCOND:
            raise ValueError(f"the mean system is singular (reciprocal condition estimate {rcond:.1e}): a state's mean is "
                             "undetermined (a random walk driven by controls whose first-order conditions do not read it), or a "
                             "control's mean first-order condition is empty (no quadratic term in the control's current value); "
                             "the kernels do not depend on the means, so the model solves without its targets, constant drifts "
                             "and initial states")
        zbar = sla.lu_solve((lu, piv), b, check_finite=False)
        zbar[pinned] = 0.0
        return zbar

    def mean_cost(self, agent: Agent, zbar: np.ndarray) -> float:
        """The mean part of the agent's cost, 1/2 zbar'Q zbar + q'zbar over its loss atoms at the primaries'
        means `zbar` (the constant of a target, theta^2, is not in the model): the stationary flow loss per
        unit time, or its discounted integral over [0, T] by the engine's quadrature (_mean_weights)."""
        atoms, Q, q = self.c.loss[agent.name]
        zeta = self._mean_atoms(zbar, atoms); w = self._mean_weights()[:zeta.shape[1]]
        return float(sum(wt * (0.5 * z @ Q @ z + q @ z) for wt, z in zip(w, np.ascontiguousarray(zeta.T))))

    def _mean_part(self, res) -> None:
        """The means (targets, constant drifts, initial states) and the mean part of every cost, on the result
        with res.maps, res.Z and res.costs already holding the variance part of every agent's cost: res.means
        (name -> a float on the stationary engine, a path over res.means_t on the finite engines) for every
        primary, definition and "agent.row" drift rate (at the time of the observation, its delay not applied),
        res.cost_parts[agent] = {"variance", "mean"}, and the mean part added to res.costs[agent].  Part of the
        answer, not a check: it runs with diagnostics=False too."""
        c = self.c; m = self.model; Nt = self._mean_nodes(); times = self._mean_times()
        zbar = self.solve_means(res.maps)
        path = (lambda v: float(v[0])) if times is None else (lambda v: v)
        def value(expr):
            paths = self._mean_atoms(zbar, list(expr))
            return path(sum((coef * paths[i] for i, coef in enumerate(expr.values())), np.zeros(Nt)))
        res.means = {name: path(zbar[c.index[name] * Nt:(c.index[name] + 1) * Nt].copy()) for name in c.prim}
        res.means.update({d.name: value(m.expand({d.name: 1.0})) for d in m.definitions})
        res.means.update({f"{a.name}.{r.name}": value(m.expand(r.drift)) for a in m.agents for r in a.signals})
        if times is not None:
            res.means_t = times.copy()
        for a in m.agents:
            mean = self.mean_cost(a, zbar)
            res.cost_parts[a.name] = {"variance": res.costs[a.name], "mean": mean}
            res.costs[a.name] += mean
