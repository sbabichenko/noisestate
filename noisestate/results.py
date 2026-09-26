"""Results returned by the engines: one Result, three grids.

    res.converged, res.residual, res.message     outcome of the outer solve
    res.evaluations                              best-response evaluations made
    res.diagnostics                              the checks: .statuses, .rows, .flags, .assess(policy), .summary()
    res.require_converged()                      raise ConvergenceError unless converged (convergence ALONE)
    res.require_ok(policy)                       raise unless the policy's assessment accepts it (the full verdict)
    res.axes                                     the coordinate arrays of kernel(): {"age": ages} (stationary), {"time", "age",
                                                 "shock_time"} node-wise (spectral finite; shock_time < 0 on a transition's band),
                                                 {"time", "shock_time"} (cells, the two axes of the (N, N) matrix); and "maps":
                                                 {agent: {row: {axis: values}}}, where each row's map values belong (map_axes)
    res.times                                    the time nodes of the paths (None on the stationary engine)
    res.paths                                    {"means": {name: path}} over res.times on a finite horizon; a transition adds
                                                 "loss" ({agent: E[loss(t)]}) and "belief_error" (a callable (agent, name) -> path)
    res.world                                    the closed-loop kernels of every primary on the shocks
    res.extra                                    engine-specific extras: window_tail (stationary); past, continuation, settled
                                                 (a transition), old_flows, new_flows, excess_costs, representation_parts
    res.numerics                                 the resolved Numerics the result was solved with
    res.costs[agent]                             stationary flow loss per unit time, or the discounted
                                                 integral over [0, T] (res.cost_kind says which): the variance
                                                 part (the shocks) plus the mean part (targets, constant drifts)
    res.cost_parts[agent]                        {"variance", "mean"}, the two parts
    res.means[name]                              mean of every state, control and definition, and the mean drift rate of
                                                 every signal row as "agent.row": a constant (stationary) or the path on the
                                                 time nodes res.mean_times (finite; the spectral res.mean(name, t) interpolates)
    res.kernel(name, shock=None)                 closed-loop kernel of a state or control
    res.maps[agent]                              raw strategies on the agent's signal rows, indexed by the age of the
                                                 increment as the agent sees it (a delayed row's raw increment is older
                                                 by the delay; the finite engine stores a delayed row's map at the
                                                 shifted time t - delay): res.MAP_CONVENTION, to_dict()["agents"]
    res.to_dict()                                JSON-ready payload (version, params, model spec, options, grid, kernels,
                                                 maps with their axes, costs, FOC parts)
    res.summary()                                one paragraph

Kernel layout by engine:
  StationaryResult    kernel(name) -> (N, nW) values at the shock ages res.ages; kernel(name, ch) -> (N,)
  TriangleResult      kernel(name) -> (N, nW) at the triangle nodes (res.grid.t, res.grid.s);
                      res.evaluate(name, ch, t, s) interpolates; kernel(name, ch) -> (N,)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


import numpy as np

from .accel import ConvergenceError, DiagnosticsError
from ._settings import DEFAULT, Settings, tunable
from .schema import PAYLOAD_VERSION      # the format's version lives with the format
from .diagnostics import (CHECKS, DIAGNOSTIC_ONLY, RESIDUAL_NORM, RESIDUAL_TOLERANCE,
                          Assessment, Policy, Refinement, Stability, Status, applicable, assess, classify,
                          verification)
from .kernel import Kernel
from .spec import Model


class _StabilityBudget(Exception):
    """Raised by stability()'s matvec at its evaluation budget, so ARPACK unwinds to the fallback."""


class Diagnostics:
    """The checks a result carries, and what a policy makes of them.

    Replaces diagnostic_rows(), diagnostic_records(), diagnostic_summary(), diagnostic_verdict(),
    status and resolution_ok, which overlapped and disagreed about what a missing check meant.
    """

    def __init__(self, result: "Result"):
        self._res = result

    @property
    def rows(self) -> tuple:
        """Every emitted check, with its presentation and automation fields."""
        return tuple(self._res._diagnostic_record(d) for d in self._res._check_rows())

    @property
    def flags(self) -> tuple:
        """The flag text of every check that FAILED -- what a reader is shown, and what a test that
        cares about the wording matches on.  Acceptance is assess().accepted, not this."""
        return tuple(r["flag"] for r in self.rows if r["ok"] is False and r["flag"])

    def by_category(self, category: str) -> tuple:
        return tuple(r for r in self.rows if r.get("category") == category)

    @property
    def statuses(self) -> dict:
        """Every APPLICABLE check by root name, with the status of each.

        Computed from the model and the engine's capability, not from the rows that were emitted:
        a check that applies and produced nothing still appears, under the reason it produced
        nothing.
        """
        res, model = self._res, self._res.model
        emitted: dict = {}
        for r in self._res._check_rows():
            root = r["name"].split(":", 1)[0]
            ok = r["ok"]
            if root in emitted and emitted[root] is False:
                continue                                  # one failing row fails the root
            emitted[root] = ok if root not in emitted or ok is False else emitted[root]
        supported = res.supported_checks()
        skipped_off = res.solve_kw.get("diagnostics") is False
        out = {}
        applies = applicable(CHECKS, model)
        for check in sorted(CHECKS):
            if check not in applies:
                out[check] = Status.NOT_APPLICABLE      # considered, and meaningless for this model
            elif check not in supported:
                out[check] = Status.UNSUPPORTED
            elif skipped_off and check in DIAGNOSTIC_ONLY:
                out[check] = Status.SKIPPED
            elif check not in emitted or emitted[check] is None:
                out[check] = Status.MISSING
            else:
                out[check] = Status.PASSED if emitted[check] else Status.FAILED
        return out

    def assess(self, policy: Policy = Policy.PUBLICATION) -> Assessment:
        detail = {}
        for r in self._res._check_rows():
            if r["ok"] is False and r["flag"]:
                detail.setdefault(r["name"].split(":", 1)[0], r["flag"])
        return assess(self.statuses, policy, self._res.model, detail)

    def summary(self, detailed: bool = False) -> str:
        return self._res._diagnostic_summary(detailed)

    def __repr__(self) -> str:
        return f"<Diagnostics {self.assess()}>"


def _nm(x):
    """A name from a name or an object of the equations form (a State, Control, Agent, Shock or Param)."""
    return x if x is None or isinstance(x, str) else getattr(x, "name", x)


class Response:
    """res.response(...): one shock followed through time.  .over(t) evaluates it."""

    def __init__(self, res, quantity: str, shock: str, at: float, agent: Optional[str]):
        self.res, self.quantity, self.shock, self.at, self.agent = res, quantity, shock, at, agent

    def over(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        K = (self.res.estimate(self.agent, self.quantity, self.shock) if self.agent is not None
             else self.res.kernel(self.quantity, self.shock))
        if self.res.kind == "stationary":
            return np.asarray(K.at(t), dtype=float).reshape(t.shape)
        flat = t.ravel(); live = flat >= self.at
        out = np.zeros_like(flat)
        if live.any():
            out[live] = np.asarray(K.at(flat[live], np.full(int(live.sum()), self.at)), dtype=float).ravel()
        return out.reshape(t.shape)

    def __repr__(self) -> str:
        who = f"{self.agent}'s estimate of " if self.agent else ""
        return f"Response({who}{self.quantity} to a unit {self.shock} at t = {self.at:g})"


class SeedResponse:
    """res.deviation_response(...): kernels in the age of a deviation seed; .over(ages) evaluates them, (len(ages),
    number of quantities)."""

    def __init__(self, grid, K: np.ndarray, names: List[str]):
        self.grid, self.K, self.names = grid, K, names

    def over(self, ages) -> np.ndarray:
        return np.asarray(self.grid.interp(np.asarray(ages, dtype=float)) @ self.K, dtype=float)

    def __repr__(self) -> str:
        return f"SeedResponse({', '.join(self.names)})"


class SeedResponse2:
    """A finite result's deviation_response: kernels in (time, seed time); .over(t, s) evaluates them at the
    points (t_k, s_k), (number of points, number of quantities)."""

    def __init__(self, grid, K: np.ndarray, names: List[str]):
        self.grid, self.K, self.names = grid, K, names

    def over(self, t, s) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float)); s = np.broadcast_to(np.asarray(s, dtype=float), t.shape)
        return np.asarray(self.grid.interp(t, t - s) @ self.K, dtype=float)

    def __repr__(self) -> str:
        return f"SeedResponse2({', '.join(self.names)})"


class Responses:
    """res.response(X, ...) for a vector X: one Response per component; .over(t) stacks them on a last axis."""

    def __init__(self, parts: List[Response]):
        self.parts = parts

    def __getitem__(self, i) -> Response:
        return self.parts[i]

    def __len__(self) -> int:
        return len(self.parts)

    def over(self, t) -> np.ndarray:
        return np.stack([p.over(t) for p in self.parts], axis=-1)

    def __repr__(self) -> str:
        return f"Responses({', '.join(p.quantity for p in self.parts)})"


@dataclass
class Result:
    """One result type; the engines' subclasses (StationaryResult, TriangleResult, TransitionResult)
    fill in the grid-specific hooks and are internal: isinstance(res, noisestate.Result) holds for every
    result, and the fields and methods below read the same on every engine."""
    model: Model
    compiled: object
    maps: Dict[str, np.ndarray]
    world: np.ndarray                          # the closed-loop kernels of every primary on the shocks, stacked
    converged: bool
    residual: float
    evaluations: int
    seconds: float
    costs: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    representation_error: Dict[str, float] = field(default_factory=dict)   # agent -> relative residual; the resolution check
    representation_parts: Dict[str, Dict[str, float]] = field(default_factory=dict)   # agent -> where it sits (transition: interior / band tip / last window)
    refinement: Optional[object] = None        # filled by refine(): a Refinement, with the finer Result
    solver_class: object = None                # the engine that produced this result, with its options, so that
    solver_kw: dict = field(default_factory=dict)      # refine()/stability() rebuild the same solver
    solve_kw: dict = field(default_factory=dict)
    second_order: Dict[str, dict] = field(default_factory=dict)   # agent -> {"min", "max", "ok"}: is the best response a minimum
    foc: Dict[str, dict] = field(default_factory=dict)            # agent -> control -> {"foc", "physical", "wedge"} kernels
    means: Dict[str, object] = field(default_factory=dict)        # quantity -> its mean, a float (stationary) or a path (finite); zero when nothing drives it
    mean_times: Optional[np.ndarray] = None                       # the time nodes of the mean paths (finite engines)
    cost_parts: Dict[str, Dict[str, float]] = field(default_factory=dict)   # agent -> {"variance", "mean", "constant"} parts of its cost
    settings: Settings = DEFAULT                                  # the tuning constants of the engine that produced this result
    kind: str = "base"

    # the thresholds of the checks: aliases of the fields of self.settings (settings.py)
    RESOLUTION_TOL = tunable("resolution_tol")
    WINDOW_TAIL_TOL = tunable("window_tail_tol")
    MAP_CONVENTION = ""             # how maps[agent][u][row] is indexed, in words, for a consumer of to_dict()

    def map_axes(self, delay: float) -> dict:
        """Hook (every result class overrides it): where the values of a map on a row observed with `delay`
        belong, as {axis name: list over the map's nodes} (the time of the control, the age or time of the
        raw increment; the keys MAP_CONVENTION names): to_dict() puts them beside the row's delay so a
        consumer of the payload can place a delayed row's map without this code.
        The other per-result hooks are kernel(name, shock) (a state's, control's or definition's
        closed-loop kernel in the engine's layout, see the module docstring), grid_info() (the JSON-ready
        grid description of to_dict()), _kernel_change(fine) (the relative change of the kernels against
        a finer result of the same engine, for refine()), summary() and plot(path); window_tail is optional
        (the stationary result's; diagnose() reads it when present)."""
        raise NotImplementedError

    MEAN_ZERO = tunable("mean_zero")          # below this a mean is round-off (the mean system is solved only when something drives it)

    # ----------------------------------------------------------- the common surface
    @property
    def times(self) -> Optional[np.ndarray]:
        """The time nodes of the paths; None on the stationary engine (its means are constants)."""
        return None

    def _node_axes(self) -> dict:
        """Hook: the coordinate arrays of kernel()'s node axis (or axes), by name."""
        raise NotImplementedError

    @property
    def axes(self) -> dict:
        """The coordinate arrays of kernel(name, shock) by name (the module docstring lists them per engine),
        and under "maps" where every agent's row's map values belong ({agent: {row: {axis: values}}})."""
        out = dict(self._node_axes())
        out["maps"] = {a.name: {r.name: {k: np.asarray(v) for k, v in self.map_axes(r.delay).items()} for r in a.signals}
                       for a in self.model.agents}
        return out

    @property
    def paths(self) -> dict:
        """The paths over res.times: "means" ({name: path}) on a finite horizon; a transition adds "loss"
        ({agent: E[loss(t)]}) and "belief_error" (a callable (agent, name) -> path).  Empty on the stationary engine."""
        if self.times is None:
            return {}
        return {"means": {k: np.asarray(v, dtype=float) for k, v in self.means.items()}}

    #  Which checks this engine CAN compute.  A check that applies and is not here is UNSUPPORTED,
    #  never merely absent -- the distinction the cell engine used to fall through.
    SUPPORTED_CHECKS = frozenset(CHECKS)

    def supported_checks(self) -> frozenset:
        """Which checks THIS engine can compute for THIS model.

        A property of the ENGINE: the cell engine computes neither a representation error nor a
        second-order form, and says so in its SUPPORTED_CHECKS.

        This used to subtract the second-order check on a discounted stationary model, on the
        ground that the discounted objective is not a quadratic form in the stationary kernel.  That
        was wrong, and the dissertation says so: the discounted objective IS written as a quadratic
        form, and its joint running Hessian carries no discount -- rho enters only as the strictly
        positive weight e^{-rho t}, which cannot change the sign of a form that is semidefinite
        pointwise in t.  The check is made on the average-cost system and holds at every rho, which
        is what the Kyle-Back chapter does.
        """
        return frozenset(self.SUPPORTED_CHECKS)

    @property
    def diagnostics(self) -> "Diagnostics":
        """The checks this result carries, and what a policy makes of them (see Diagnostics)."""
        return Diagnostics(self)

    @property
    def extra(self) -> dict:
        """Engine-specific extras (the module docstring lists them); {} when the engine has none."""
        return {}

    @property
    def has_means(self) -> bool:
        """Whether any mean is nonzero beyond round-off."""
        return any(np.max(np.abs(np.asarray(v, dtype=float))) > self.MEAN_ZERO for v in self.means.values() if np.size(v))

    def _mz(self, v: float) -> float:
        """A mean for printing: round-off shown as an unsigned zero."""
        return float(v) if abs(float(v)) > self.MEAN_ZERO else 0.0

    @property
    def _resolution_ok(self) -> Optional[bool]:
        """Internal: the resolution row's raw verdict, None when nothing was computed.  Not public --
        a three-valued answer under an *_ok name is what res.diagnostics.statuses replaced."""
        if not self.representation_error:
            return None
        return all(v <= self.RESOLUTION_TOL for v in self.representation_error.values())

    def _make_solver(self, model: Model):
        """The engine that produced this result, with the same constructor options, on `model`."""
        if self.solver_class is None:
            from .engines import solver
            return solver(model)
        return self.solver_class(model, **self.solver_kw)

    @property
    def numerics(self):
        """The resolved Numerics this result was solved with: the model's grid and engine, the solve's
        tolerance, damping, Newton steps and iterate, the engine's settings."""
        from dataclasses import replace
        num = self.model.numerics.resolved(self.model.horizon.kind)
        return replace(num, settings=self.settings, **{k: v for k, v in self.solve_kw.items()
                                                       if k in ("tol", "damping", "max_newton", "variable") and v is not None})

    # ----------------------------------------------------------- common
    def require_converged(self):
        """Return self, or raise ConvergenceError if the solve did not reach its tolerance.

        Convergence ONLY.  A converged solve is a numerical solution, within tolerance, of the *discretised,
        truncated* model, so this
        passes on a result whose window is too short or whose grid is too coarse: those are the checks,
        and res.diagnostics.assess(policy) is what weighs them.  require_ok() is this plus the checks,
        and is the call to put in front of a number that will be used rather than looked at."""
        if not self.converged:
            raise ConvergenceError(f"{self.model.name}: residual {self.residual:.2e} ({self.message})")
        return self

    def require_ok(self, policy: Policy = Policy.PUBLICATION):
        """Return self, or raise when `policy`'s assessment is not accepted.

        Accepted means every check the POLICY requires and the MODEL makes applicable has status
        PASSED.  Nothing else grants it: not an absent record, not an empty category, not an
        engine that cannot run the check.  Use this wherever a number is going to be used rather
        than looked at.

        WHICH exception follows the reason, so each one's contract stays true:
            the `converged` check blocked   -> ConvergenceError
            anything else blocked           -> DiagnosticsError
        Both carry the Assessment as .assessment; both are ResultValidationError.
        """
        verdict = self.diagnostics.assess(policy)
        if verdict.accepted:
            return self
        why = "; ".join(str(b) for b in verdict.blocking)
        if any(b.check == "converged" for b in verdict.blocking):
            err = ConvergenceError(f"{self.model.name}: {why}")
        else:
            err = DiagnosticsError(f"{self.model.name}: converged, but {why}", verdict)
        err.assessment = verdict
        raise err

    def kernel(self, name, shock=None) -> "Kernel":
        """The closed-loop kernel of a state, control or definition (on every shock, or one), as a Kernel: an
        ndarray in the engine's layout (the subclass's _kernel documents it) carrying .axes (res.axes' node
        coordinates), .at(*coords) (the engine's own interpolant) and .plot(path=None).  name and shock are
        names or the objects of the equations form (X, dW0)."""
        name, shock = _nm(name), _nm(shock)
        return Kernel.of(self._kernel(name, shock), self, name, shock)

    def _kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def shocks(self) -> List[str]:
        """The columns of the kernels: the model's shocks (a transition adds the initial shocks of its past)."""
        return list(self.compiled.channels)

    def _project(self, agent, K: np.ndarray) -> np.ndarray:
        """Hook: the kernel of E[L_t | agent's information at t] for the process L with kernel K at this result's
        nodes (the projection on the agent's seen rows); the stationary and spectral finite engines supply it."""
        raise NotImplementedError(f"estimates and strategies are not available on the {self.kind} engine; "
                                  "solve with the spectral finite engine or the stationary engine")

    def estimate(self, agent: str, name: str, shock: Optional[str] = None) -> Kernel:
        """`agent`'s estimate of the quantity `name` (a state, control or definition) as a kernel on the primitive
        shocks: at the nodes of res.kernel(name), the response of E[name_t | agent's information at t] at time t to a
        unit shock at time s.  Evaluate it with .at(t, s), like res.kernel(name).  Its difference from
        res.kernel(name) is the agent's estimation error, whose variance res.belief_error integrates; an agent's
        estimate of its own control is the control.  shock picks one, as in res.kernel(name, shock)."""
        agent, name, shock = _nm(agent), _nm(name), _nm(shock)
        K = self._project(agent, self._kernel(name))
        return Kernel.of(K if shock is None else K[:, self.shocks.index(shock)], self, f"E[{name} | {agent}]", shock)

    def strategy(self, control: str, shock: Optional[str] = None) -> Kernel:
        """The strategy kernel of `control` on its agent's noise-state (Chapter 1, Definition 1.4): the weight the
        action at time t puts on the agent's estimate of the shock at time u.  From the first-order condition
        (Remark 1.13) the action is the agent's estimate of -(G^DD)^-1 (G^DX X + B' H), so this kernel is
        D_W - (G^DD)^-1 phi, with D_W the controls' kernels, phi the kernels of their first-order conditions
        (res.foc) and G^DD the Hessian of the agent's loss in its controls (a matrix for an agent with several, a
        vector control); the agent's estimate of it is the control's kernel again.  Defined for a game without
        delayed or lead terms, solved with diagnostics.  shock picks one, as in res.kernel(name, shock); a vector
        control (Control("D", 2)) gives a list, one kernel per component."""
        if not isinstance(control, str) and hasattr(control, "items") and isinstance(control.items, list):
            return [self.strategy(c, shock) for c in control.items]
        control, shock = _nm(control), _nm(shock)
        agent = next((a for a in self.model.agents if control in a.controls), None)
        if agent is None:
            raise KeyError(f"{control!r} is not a control")
        atoms, Q, _ = self.compiled.loss[agent.name]
        if any(lag for (nm, lag) in atoms if nm in agent.controls) or self.model.all_lags():
            raise NotImplementedError("strategy() is defined for a game without delayed or lead terms in the control")
        missing = [u for u in agent.controls if (u, 0.0) not in atoms]
        if missing:
            raise NotImplementedError(f"the loss of {agent.name} has no quadratic term in {missing}")
        ix = [atoms.index((u, 0.0)) for u in agent.controls]
        G = Q[np.ix_(ix, ix)]                                       # G^DD, the Hessian in the agent's controls
        focs = [(self.foc.get(agent.name) or {}).get(u) for u in agent.controls]
        if any(f is None for f in focs) or np.linalg.eigvalsh(0.5 * (G + G.T))[0] <= 0:
            raise NotImplementedError("strategy() needs the first-order-condition kernels (solve with diagnostics) "
                                      "and a positive definite curvature in the agent's controls")
        row = np.linalg.solve(G, np.eye(len(ix)))[agent.controls.index(control)]      # the control's row of (G^DD)^-1
        K = self._kernel(control) - sum(w * np.asarray(f["foc"], dtype=float) for w, f in zip(row, focs))
        return Kernel.of(K if shock is None else K[:, self.shocks.index(shock)], self, f"strategy of {control}", shock)

    def response(self, quantity, to, at: float = 0.0, seen_by=None) -> "Response":
        """One shock followed through time: the response of `quantity` (or, with seen_by, that agent's estimate of
        it) to a unit shock `to` that struck at time `at`.  `.over(t)` gives the values at the times t (zero before
        the shock).  On a stationary game only the shock's age matters: `.over(ages)`.
            res.response(X, to=dW0).over(t)                       # the state
            res.response(X, to=dW0, seen_by=player1).over(t)       # player 1's estimate of it
        A vector quantity (State("X", 3)) gives every component: .over(t) has a last axis of its length."""
        if not isinstance(quantity, str) and hasattr(quantity, "items") and isinstance(quantity.items, list):
            return Responses([Response(self, _nm(q), _nm(to), float(at), _nm(seen_by)) for q in quantity.items])
        return Response(self, _nm(quantity), _nm(to), float(at), _nm(seen_by))

    def grid_summary(self) -> dict:
        raise NotImplementedError

    #  summary() and plot() are DECLARED here, though every engine overrides them, because a reader
    #  looking at noisestate.Result -- help(), dir(), an editor's completion -- has only this class
    #  to look at: the subclasses are internal.  A method that exists on every result but appears on
    #  none of the public surface is one a user finds by accident or not at all.
    #  What every engine's summary says, and where each one differs.  The three implementations
    #  repeated the header and the cost loop verbatim and diverged only in the grid line, the cost
    #  label and precision, and how the means are shown -- so a change to the header had to be made
    #  three times, and the formats drifted apart (one prints costs to 8 figures, two to 6).
    COST_LABEL = "cost"                 # what the number is called, per engine
    COST_FIGURES = 6                    # decimal places; the triangle carries more

    def _grid_line(self) -> str:
        """Hook: the engine's grid, ending the header line (`grid 1 panels x 24 nodes on [0, 3.0], rho=0`)."""
        raise NotImplementedError

    def _means_line(self) -> str:
        """Hook: how this engine shows the means; reached only when has_means."""
        raise NotImplementedError

    def _agent_lines(self, agent) -> list:
        """Hook: whatever an engine adds under an agent's cost ({} for most)."""
        return []

    def summary(self, diagnostics: bool = True) -> str:
        """One block of text: convergence, residual, evaluations, runtime, the grid, each agent's cost,
        and (with `diagnostics`) the checks that failed.  print(res.summary()) is the usual first look
        at a result; res.diagnostics.summary() is the checks alone."""
        f = self.COST_FIGURES
        lines = [f"{self.model.name}: {self._status(compact=not diagnostics)} residual {self.residual:.2e} "
                 f"in {self.evaluations} evaluations, {self.seconds:.1f}s; {self._grid_line()}"]
        for a in self.model.agents:
            parts = self.cost_parts.get(a.name)
            lines.append(f"  {a.name}: {self.COST_LABEL} = {self.costs.get(a.name, float('nan')):+.{f}f}"
                         + (f" (variance {parts['variance']:+.{f}f}, mean {parts['mean']:+.{f}f})"
                            if parts and parts["mean"] != 0.0 else "")
                         + (f" + continuation {parts['continuation']:+.{f}f} on the buffer"
                            if parts and "continuation" in parts else ""))
            lines.extend(self._agent_lines(a))
        if self.has_means:
            lines.append(self._means_line())
        return "\n".join(lines)

    def plot(self, path: str) -> None:
        """Write a figure of the result's kernels to `path` (.pdf or .png; matplotlib required).
        res.kernel(name, shock).plot(path) draws one kernel instead of all of them."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """One line: what was solved, whether it converged, and whether it is accepted.

        The generated dataclass repr ran to about 79,000 characters -- the compiled engine, every
        kernel and every map -- which is what a notebook prints when the last line of a cell is a
        result.  The detail is still one explicit call away: summary(), to_dict(), or the field.
        """
        state = "converged" if self.converged else "NOT converged"
        try:
            costs = ", ".join(f"{a}={v:.5g}" for a, v in list(self.costs.items())[:3])
            if len(self.costs) > 3:
                costs += ", ..."
        except Exception:
            costs = ""
        try:
            verdict = "accepted" if self.diagnostics.assess().accepted else "not accepted"
        except Exception:                       # a repr must not raise, whatever the result holds
            verdict = "?"
        return (f"<{type(self).__name__} {self.model.name!r} {self.kind}: {state} "
                f"(residual {self.residual:.2e}, {self.evaluations} evaluations, {self.seconds:.1f}s); "
                f"costs {costs or 'none'}; publication: {verdict}>")

    @property
    def cost_kind(self) -> str:
        return "stationary flow loss per unit time" if self.kind == "stationary" else "discounted integral over [0, T]"

    REFINES_EXPONENTIALLY = True        # a small change on refinement means resolved
    KERNEL_NOTE = ""                    # what Kernel.note says about this engine's kernels

    def _refined_nodes(self, n0: int, factor: float) -> int:
        import math
        return max(n0 + 2, int(math.ceil(n0 * factor)))

    def refine(self, factor: float = 1.5, **solve_kw) -> "Refinement":
        """Re-solve on a finer grid (nodes x factor) and report the relative change of every agent's
        cost and of the kernels, the honest test of resolution (window, corner and product errors alike).
        Stored in self.refinement and shown by summary().  Returns a Refinement, which carries the
        finer Result itself rather than only numbers taken from it."""
        import math
        n0 = int(self.model.numerics.nodes)
        n1 = self._refined_nodes(n0, factor)
        # this solve's bounds and a skipped diagnostics pass are not the refinement's
        kw = {k: v for k, v in self.solve_kw.items()
              if k not in ("start_from", "start_policy", "max_evaluations", "deadline", "diagnostics")}
        kw.update(solve_kw)
        solver = self._make_solver(self.model.with_numerics(nodes=n1))
        try:                                                  # start the fine solve from this equilibrium, interpolated
            kw.setdefault("start_from", solver.interpolate_maps(self))
        except NotImplementedError:
            pass
        fine = solver.solve(**kw)
        scale = max(1e-300, max(abs(v) for v in self.costs.values()))          # one scale: a zero-profit agent is not "unresolved"
        cost_change = max(abs(fine.costs[k] - self.costs[k]) / scale for k in self.costs)
        kernel_change = self._kernel_change(fine)
        rep = {"nodes": n1, "converged": bool(fine.converged), "cost_change": float(cost_change),
               "kernel_change": float(kernel_change)}
        # the spectral engines converge exponentially, so a small change means resolved; a first-order engine's
        # change only halves per doubling: no verdict, the numbers are the report
        rep["resolved"] = None if not self.REFINES_EXPONENTIALLY else bool(fine.converged and cost_change < self.REFINE_COST_TOL
                                                                          and kernel_change < self.REFINE_KERNEL_TOL)
        # A saddle the second-order check reports is a claim about the model; a negative curvature that
        # shrinks towards zero as the grid refines is a claim about the grid (a quadrature direction on the
        # diagonal, alternating in sign between neighbouring age nodes, is the shape it takes).  The two read
        # the same on one grid, so the refinement says which: `shrinking` is the finer grid's curvature being
        # less negative.  The stationary engine settles the analogous question on the window with its own
        # embedded-curvature hook; this is the finite engines' answer, and it costs the refinement, not a solve.
        curvature = {}
        for name, here in (self.second_order or {}).items():
            there = (fine.second_order or {}).get(name)
            if not (here and there) or here.get("min") is None or there.get("min") is None:
                continue
            if here["min"] < 0:
                curvature[name] = {"min": float(here["min"]), "fine_min": float(there["min"]),
                                   "nodes": n1, "shrinking": bool(there["min"] > here["min"])}
        out = Refinement(coarse=self, fine=fine, nodes=n1, cost_change=float(cost_change),
                         kernel_change=float(kernel_change), converged=bool(fine.converged),
                         resolved=rep["resolved"], curvature=curvature)
        self.refinement = out
        return out

    def _kernel_change(self, fine) -> float:
        raise NotImplementedError

    REFINE_COST_TOL, REFINE_KERNEL_TOL = tunable("refine_cost_tol"), tunable("refine_kernel_tol")

    def _check_rows(self) -> List[dict]:
        """Every check this result carries, as rows {name, value, threshold, ok, flag, advice}: ok is
        True/False, or None when the check gives no verdict (not computed, or not applicable).  The
        summary prints the rows that fail; to_dict() carries them all.  The thresholds are the class
        constants named in each row."""
        rows = []

        def row(name, value, threshold, ok, flag, advice=""):
            rows.append({"name": name, "value": value, "threshold": threshold, "ok": ok, "flag": flag, "advice": advice})
        row("converged", float(self.residual), self.solve_kw.get("tol"), bool(self.converged),
            "NOT converged", self.message)
        if self.solve_kw.get("diagnostics") is False:
            row("diagnostics", None, None, None, "diagnostics skipped (solve(diagnostics=False): no second-order check, "
                "first-order-condition decomposition or representation error)", "solve again with diagnostics=True")
        rep = max(self.representation_error.values()) if self.representation_error else None
        where = {}                                   # a transition's error by region, the worst agent per region
        for parts in self.representation_parts.values():
            for k, v in parts.items():
                where[k] = max(where.get(k, 0.0), float(v))
        at = " (" + ", ".join(f"{k} {v:.1e}" for k, v in where.items()) + ")" if where else ""
        row("resolution", rep, self.RESOLUTION_TOL, self._resolution_ok,
            f"UNDER-RESOLVED (representation error {rep:.1e}{at}: raise numerics.nodes" + ("; an error only on the band tip or the last "
            "window is the geometry there, not the interior's resolution)" if where else ")") if rep is not None else "", "raise numerics.nodes")
        tail = getattr(self, "window_tail", None)
        if tail is not None:
            row("window", float(tail), self.WINDOW_TAIL_TOL, bool(tail <= self.WINDOW_TAIL_TOL),
                f"WINDOW TOO SHORT (a kernel still moves by {tail:.1%} of its peak over the last tenth of the window: raise horizon.window"
                + ("; the means' continuation integrals are truncated there as well)" if self.has_means else ")"),
                "raise horizon.window")
        for a, so in self.second_order.items():
            if so.get("converged") is False:
                row(f"second_order:{a}", None, None, None, f"second-order check did not converge for {a!r}", so.get("message", ""))
            else:
                # a refinement that shows the curvature shrinking towards zero overturns the verdict, as the
                # window's embedded curvature does: the direction is the quadrature's, not a strategy
                grid = (self.refinement.curvature if self.refinement else {}).get(a)
                shrinks = bool(grid and grid["shrinking"])
                row(f"second_order:{a}", so["min"], -self.settings.second_order_tol if self.solver_class is not None else None,
                    so["ok"] or shrinks,
                    f"NOT A MINIMUM (the best response of {a!r} is a saddle: its loss is not convex in its own strategy, "
                    f"smallest curvature {so['min']:.1e} of the largest)", "the loss is not convex in the agent's own strategy")
                if so.get("edge"):
                    row(f"second_order_edge:{a}", so["min"], -self.settings.second_order_tol if self.solver_class is not None else None, None,
                        f"window edge: the curvature of {a!r} is negative ({so['min']:.1e}) on this window but positive "
                        f"({so['embedded']:.1e}) on a window longer by two lags: a truncation of the lagged loss terms at the edge, not a saddle",
                        "a wider window moves it, a quadratic term in the control's current value removes it")
        if self.refinement:
            f = self.refinement
            row("refinement", {"cost_change": f.cost_change, "kernel_change": f.kernel_change, "nodes": f.nodes},
                {"cost_change": self.REFINE_COST_TOL, "kernel_change": self.REFINE_KERNEL_TOL}, f.resolved,
                f"refinement to {f.nodes} nodes moves costs by {f.cost_change:.1e} and kernels by {f.kernel_change:.1e}"
                + ("" if f.resolved in (True, None) else " (NOT RESOLVED)"), "raise numerics.nodes")
            for a, cv in (f.curvature or {}).items():
                row(f"second_order_grid:{a}", cv["fine_min"], None, bool(cv["shrinking"]),
                    (f"the negative curvature of {a!r} shrinks with the grid ({cv['min']:.1e} here, "
                     f"{cv['fine_min']:.1e} at {cv['nodes']} nodes): the quadrature, not a saddle"
                     if cv["shrinking"] else
                     f"the negative curvature of {a!r} holds under refinement ({cv['min']:.1e} here, "
                     f"{cv['fine_min']:.1e} at {cv['nodes']} nodes): a saddle, not the grid"),
                    "refine again to see the trend; a direction on the diagonal a = t that alternates in "
                    "sign between neighbouring age nodes is the quadrature's, not a strategy")
        st = getattr(self, "stability_report", None)
        if st:
            row("stability", float(st.radius), 1.0, bool(st.stable),
                f"best-response dynamics {'stable' if st.stable else 'UNSTABLE'} (spectral radius {st.radius:.3f}"
                + (", untied game" if st.untied else "") + (f", by {st.method}" if st.method != "arnoldi" else "") + ")",
                "naive best-response adjustment would not find this equilibrium")
        return rows

    def _diagnostic_record(self, d: dict) -> dict:
        """Add stable presentation and automation fields to an existing numerical check."""
        name = d["name"]
        root = name.split(":", 1)[0]
        category = ("solve" if root == "converged" else "equilibrium" if root.startswith("second_order") or root == "stability"
                    else "numerics")
        meanings = {
            "converged": "Whether the fixed-point solve reached its requested tolerance.",
            "resolution": "Whether the computed strategy is represented accurately on this grid.",
            "window": "Whether stationary kernels have stopped changing near the window boundary.",
            "past window": "Whether the inherited stationary kernels have stopped changing near their boundary.",
            "continuation window": "Whether the stationary continuation kernels have stopped changing near their boundary.",
            "settled": "Whether transition strategies have reached the stationary continuation by the end of the horizon.",
            "settle floor": "Whether the requested settle tolerance is distinguishable on this grid.",
            "stability": "Whether small strategy deviations shrink under naive best-response iteration.",
        }
        options = {}
        if name == "window":
            options["window"] = 2 * float(self.model.horizon.window)
        elif name == "settled":
            #  the settled guard wants a longer TERMINAL TIME, not a longer lag window: under the
            #  old shared name it printed "--window", which the transition command no longer takes.
            options["T"] = 2 * float(self.compiled.T)
        elif name == "past window":
            options["past_window"] = 2 * float(self.past.window)
        elif name == "continuation window":
            options["past_window"] = 2 * float((self.compiled.continuation_info or {})["window"])   # the continuation's is the past's
        elif name in ("resolution", "settle floor", "refinement"):
            options["nodes"] = max(int(self.numerics.nodes) + 2, int(np.ceil(1.5 * self.numerics.nodes)))
        out = dict(d)
        out.update(code=name.lower().replace(":", "_").replace(" ", "_"), category=category,
                   severity="error" if d["ok"] is False else "info" if d["ok"] is None else "ok",
                   meaning=meanings.get(name, "A numerical or equilibrium check reported by the solver."),
                   action=d.get("advice", ""), suggested_options=options)
        trend = None; trend_source = None
        if name == "window" and hasattr(self, "window_tail_extrapolation"):
            trend_source = self
        elif name == "past window" and getattr(getattr(self, "past", None), "source", None) is not None:
            trend_source = self.past.source
        elif name == "continuation window" and getattr(self, "continuation", None) is not None:
            trend_source = self.continuation
        if trend_source is not None:
            trend = trend_source.window_tail_extrapolation()
            if not trend_source.converged or trend_source._resolution_ok is False:
                trend["assessment"] = "inconclusive"
                trend["reason"] = "the underlying stationary solve is unconverged or under-resolved"
        if trend is not None:
            out["trend"] = trend
            if d["ok"] is False and trend["assessment"] == "not_decaying":
                out["action"] = "the tail is not decaying; check whether the stationary problem exists before extending the window"
                out["suggested_options"] = {}
        return out

    def _diagnostic_summary(self, detailed: bool = False) -> str:
        """Compact grouped verdict, with full explanations when ``detailed`` is true."""
        rows = [self._diagnostic_record(d) for d in self._check_rows()]
        judged = [d for d in rows if d["ok"] is not None]
        failed = [d for d in judged if d["ok"] is False]
        lines = [f"Diagnostics: {len(failed)} failed, {len(judged) - len(failed)} passed"]
        for category, label in (("solve", "Solve"), ("numerics", "Numerics"), ("equilibrium", "Equilibrium")):
            group = [d for d in judged if d["category"] == category]
            bad = [d["name"] for d in group if d["ok"] is False]
            lines.append(f"  {label:<11} " + ("FAIL — " + ", ".join(bad) if bad else "PASS" if group else "NOT CHECKED"))
        for d in failed:
            value = d["value"]
            threshold = d["threshold"]
            measure = (f"{value:.2e}" if isinstance(value, (int, float)) else str(value))
            limit = (f"{threshold:.2e}" if isinstance(threshold, (int, float)) else str(threshold))
            lines.append(f"  FAIL  {d['name']:<20} {measure}" + (f" > {limit}" if threshold is not None else ""))
            if detailed:
                lines.extend((f"        meaning: {d['meaning']}", f"        action: {d['action'] or 'inspect the diagnostic details'}",
                              f"        detail: {d['flag']}"))
                if d["suggested_options"]:
                    suggested = " ".join(f"--{key.replace('_', '-')} {value:g}" for key, value in d["suggested_options"].items())
                    lines.append(f"        suggested: {suggested}")
                if d.get("trend"):
                    trend = d["trend"]
                    lo, hi = trend["projection_range"]
                    lines.append(f"        tail trend: {trend['assessment']} (per-segment ratio {trend['ratio']:.2f}, "
                                 f"rough tail at 2x window {trend['predicted_at_double_window']:.2e}, "
                                 f"benchmark range {lo:.2e}–{hi:.2e})"
                                 + (f"; {trend['reason']}" if trend.get("reason") else ""))
        suggestions = {key: value for d in failed for key, value in d["suggested_options"].items()}
        shared = " (also the continuation's window, which is the past's)" if "past_window" in suggestions else ""
        if suggestions:
            opts = " ".join(f"--{key.replace('_', '-')} {value:g}" for key, value in suggestions.items())
            lines.append(f"  Next: retry with {opts}{shared}")
        elif any(d.get("trend", {}).get("assessment") == "not_decaying" for d in failed):
            lines.append("  Next: inspect whether the stationary problem exists; the measured tail is not decaying")
        unjudged = [d for d in rows if d["ok"] is None and d["flag"]]
        if detailed:
            for d in unjudged:
                lines.extend((f"  INFO  {d['name']}", f"        meaning: {d['meaning']}", f"        detail: {d['flag']}"))
        return "\n".join(lines)

    def _status(self, compact: bool = False) -> str:
        """One line: the outcome, then every check that failed or has no verdict, and the informational
        rows (refinement, stability, a skipped diagnostics pass) whenever they were computed."""
        rows = self._check_rows()
        if compact:
            converged = next(d for d in rows if d["name"] == "converged")
            return "converged" if converged["ok"] else f"NOT converged ({converged['advice']})"
        parts = []
        for d in rows:
            if d["name"] == "converged":
                parts.append("converged" if d["ok"] else f"NOT converged ({d['advice']})")
            elif d["name"] in ("refinement", "stability", "diagnostics") or d["ok"] is False or (d["ok"] is None and d["name"].startswith("second_order")):
                if d["flag"]:
                    parts.append(d["flag"])
        return "; ".join(parts)

    STABILITY_K, STABILITY_EPS, STABILITY_TOL = tunable("stability_k"), tunable("stability_eps"), tunable("stability_tol")
    STABILITY_MAX_EVALUATIONS = tunable("stability_max_evaluations")
    STABILITY_FALLBACK = tunable("stability_fallback")     # of those evaluations, the rounds kept for the power iteration when Arnoldi does not settle

    def stability(self, untied: bool = True, policy: Policy = Policy.PUBLICATION,
                  adjustment: float = 0.5) -> "Stability":
        """Stability of this equilibrium under best-response dynamics: the eigenvalues of largest
        modulus of the Jacobian of the best-response map at the equilibrium, by Arnoldi iteration on
        finite differences.  A spectral radius below one means small deviations by any agent die out
        under naive best-response adjustment (tatonnement-stable); above one, they grow, and the
        equilibrium is one that adjustment dynamics would not find.  With untied=True (default) a
        tied model is assessed on the untied game, so asymmetric deviations are allowed.
        Returns {"radius", "eigenvalues", "stable", "evaluations", "untied", "method",
        "fixed_point_residual"} (method is "arnoldi", "power iteration ..." when ARPACK did not settle or
        was stopped, or "zero" for a single agent whose best response does not depend on itself); also
        stored in self.stability_report.  At most STABILITY_MAX_EVALUATIONS rounds of best responses are
        made (each matvec is one): the Arnoldi iteration is stopped STABILITY_FALLBACK short of that and
        the power iteration gets the rest, with "method" saying so."""
        from scipy.sparse.linalg import LinearOperator, eigs
        k, eps, tol, max_evaluations = self.STABILITY_K, self.STABILITY_EPS, self.STABILITY_TOL, self.STABILITY_MAX_EVALUATIONS
        model = self.model
        if untied and model.ties:
            d = model.to_dict(); d["ties"] = []
            model = Model.from_dict(d)
        S = self._make_solver(model)
        maps = {a.name: np.array(self.maps[a.name], copy=True) for a in model.agents}
        z0 = S.pack(maps)
        F0 = S.pack(S.response_map(S.unpack(z0)))
        scale = max(1.0, float(np.linalg.norm(z0)))
        count = [1]; cap = [None]                            # rounds made (F0 is one); the count at which matvec stops

        def matvec(v):
            v = np.asarray(v, dtype=float).ravel()
            nv = np.linalg.norm(v)
            if nv == 0:
                return np.zeros_like(v)
            if cap[0] is not None and count[0] >= cap[0]:
                raise _StabilityBudget()
            h = eps * scale / nv
            count[0] += 1
            return (S.pack(S.response_map(S.unpack(z0 + h * v))) - F0) / h
        n = z0.size
        kk = max(1, min(k, n - 2))
        op = LinearOperator((n, n), matvec=matvec, dtype=float)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n); method = "arnoldi"
        if np.linalg.norm(matvec(v)) <= 1e-12 * np.linalg.norm(v):
            vals = np.array([0.0]); method = "zero"          # a single agent: its best response does not depend on itself
        else:
            cap[0] = max(count[0] + 1, max_evaluations - self.STABILITY_FALLBACK)      # ARPACK's maxiter counts restarts, not rounds
            try:
                vals = eigs(op, k=kk, which="LM", tol=tol, maxiter=max_evaluations, v0=v, return_eigenvectors=False)
            except _StabilityBudget:
                method = f"power iteration (arnoldi stopped at the evaluation budget of {max_evaluations})"
            except Exception:                               # ARPACK did not settle: power iteration, and say so
                method = "power iteration (arnoldi did not converge)"
            if method != "arnoldi":
                steps = min(self.STABILITY_FALLBACK, max(1, max_evaluations - count[0])); cap[0] = count[0] + steps; lam = 0.0
                for _ in range(steps):
                    w = matvec(v); nw = np.linalg.norm(w)
                    lam = nw / np.linalg.norm(v)
                    if nw == 0:
                        break
                    v = w / nw
                vals = np.array([lam])
        radius = float(np.max(np.abs(vals)))
        residual = float(np.linalg.norm(F0 - z0) / scale)
        #  The classification is gated on VERIFICATION, not on convergence: a converged solve is
        #  not by itself an equilibrium, and the spectrum of a Jacobian at a non-fixed point is a
        #  legitimate object that simply must not be called equilibrium stability.
        ok, reasons = verification(self.diagnostics.statuses, policy, self.model, residual)
        rep = Stability(
            radius=radius, eigenvalues=tuple(complex(x) for x in np.asarray(vals)), method=method,
            fixed_point_residual=residual, residual_norm=RESIDUAL_NORM,
            residual_tolerance=RESIDUAL_TOLERANCE, verified=ok, unverified_reasons=reasons,
            **({k: v for k, v in classify(vals, radius, method, adjustment).items()
                if k != "dominant_eigenvalue"} if ok else
               {"full_response": None, "adjusted_response": None,
                "adjusted_radius_bound": None, "adjustment": None}),
            untied=bool(untied and self.model.ties), evaluations=int(count[0]))
        self.stability_report = rep
        return rep

    def to_dict(self) -> dict:
        """JSON-serialisable view.  Provenance: the package version, the parameter values, the model spec
        (`model`, which Model.from_dict rebuilds), the horizon and the engine and solve options.  Then the
        grid, the kernels per quantity and shock, the raw maps with each agent's controls and signal rows
        (delay and map axes, see MAP_CONVENTION), costs with their variance and mean parts, the means, and the
        first-order-condition decomposition where the engine provides it."""
        from . import __version__
        c = self.compiled; m = self.model
        num = self.numerics
        out = {"payload_version": PAYLOAD_VERSION, "version": __version__, "name": m.name, "engine": num.engine, "kind": self.kind,
               "converged": bool(self.converged),
               "residual": float(self.residual), "evaluations": int(self.evaluations), "seconds": float(self.seconds),
               "message": self.message, "params": {k: float(v) for k, v in m.params.items()}, "model": m.to_dict(),
               "horizon": m.to_dict()["horizon"], "numerics": num.to_dict(),
               "axes": {k: np.asarray(v).tolist() for k, v in self._node_axes().items()},
               "times": None if self.times is None else np.asarray(self.times).tolist(),
               "options": {"numerics": num.to_dict(),
                           "solver": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in self.solver_kw.items()},
                           "solve": dict(self.solve_kw)},
               "grid": self.grid_summary(), "discount": float(c.rho), "shocks": self.shocks,
               "agents": {a.name: {"controls": list(a.controls),
                                   "signals": {r.name: {"delay": float(r.delay), **self.map_axes(r.delay)} for r in a.signals}}
                          for a in m.agents},
               "map_convention": self.MAP_CONVENTION, "kernels": {}, "maps": {}, "foc": {},
               "costs": {k: float(v) for k, v in self.costs.items()},
               "cost_parts": {a: {k: float(v) for k, v in p.items()} for a, p in self.cost_parts.items()},
               "means": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in self.means.items()},
               "mean_times": None if self.mean_times is None else self.mean_times.tolist()}
        out["representation_error"] = {k: float(v) for k, v in self.representation_error.items()}
        if self.representation_parts:
            out["representation_parts"] = {a: {k: float(v) for k, v in p.items()} for a, p in self.representation_parts.items()}
        out["diagnostics"] = [{k: (None if v is None else v) for k, v in self._diagnostic_record(d).items()} for d in self._check_rows()]
        #  The payload carries the FULL status of every check and the policy that judged them, not
        #  a single ok: "passed" and "could not be run" are the distinction the assessment exists
        #  to make, and a boolean cannot express it.  (api_spec PART 7.)
        verdict = self.diagnostics.assess()
        out["assessment"] = verdict.to_dict()
        for row in out["diagnostics"]:
            row["status"] = str(verdict.statuses.get(row["name"].split(":", 1)[0], ""))
        out["cost_kind"] = self.cost_kind
        out["second_order"] = {a: dict(so) for a, so in self.second_order.items()}
        out["notes"] = self.model.notes
        if self.refinement:
            out["refinement"] = self.refinement.to_dict()      # the payload is data, not the object
        tail = getattr(self, "window_tail", None)
        if tail is not None:
            out["window_tail"] = float(tail)
        rep = getattr(self, "stability_report", None)
        if rep:
            #  the whole Stability, so the payload carries the EVIDENCE as well as the verdict:
            #  the residual, its norm and tolerance, and the reasons when the point is unverified
            out["stability"] = rep.to_dict()
        for name in c.prim:
            out["kernels"][name] = {ch: self.kernel(name, ch).tolist() for ch in self.shocks}
        for a in self.model.agents:
            g = self.maps[a.name]
            out["maps"][a.name] = {u: {r.name: g[ui, ri].tolist() for ri, r in enumerate(a.signals)}
                                   for ui, u in enumerate(a.controls)}
        if self.foc:
            for aname, dec in self.foc.items():
                out["foc"][aname] = {u: {part: {ch: arr[:, k].tolist() for k, ch in enumerate(self.compiled.channels)}
                                         for part, arr in parts.items()} for u, parts in dec.items()}
        return out




@dataclass(repr=False)
class StationaryResult(Result):
    kind: str = "stationary"
    MAP_CONVENTION = ("maps[agent][u][row][n] is the weight the control puts on the increment of the row as the agent "
                      "sees it (delayed) at age ages[n] before the control; that increment entered the raw row at age "
                      "agents[agent].signals[row].map_age[n] = ages[n] + delay, and the map is zero where map_age is "
                      "beyond the window")

    def _project(self, agent, K: np.ndarray) -> np.ndarray:
        return _stationary_projection(self, agent, K)

    @property
    def ages(self) -> np.ndarray:
        return self.compiled.grid.nodes

    def _node_axes(self) -> dict:
        return {"age": self.ages}

    @property
    def extra(self) -> dict:
        return {"window_tail": self.window_tail}

    def map_axes(self, delay: float) -> dict:
        return {"map_age": (self.ages + delay).tolist()}

    @property
    def window_tail(self) -> float:
        """How far the kernels are from having settled at the window edge L: the largest change of a
        kernel over the last tenth of the window, |K(L) - K(0.9 L)|, relative to the kernel's peak,
        over states and controls.  A kernel that has decayed, or that has reached a constant limit
        (a random-walk state, a price that tracks it), scores near zero; one still moving at L is
        truncated by the window, and above WINDOW_TAIL_TOL the summary says so.  (The value at L
        alone would flag every random-walk state.)"""
        g = self.compiled.grid; L = float(g.L)
        I1, I0 = g.interp(np.array([L])), g.interp(np.array([0.9 * L]))
        worst = 0.0
        for name in self.compiled.prim:
            K = self.kernel(name)
            peak = np.abs(K).max()
            if peak > 0:
                worst = max(worst, float(np.abs(I1 @ K - I0 @ K).max() / peak))
        return worst

    def window_tail_extrapolation(self) -> dict:
        """Estimate tail decay from four consecutive tenths near the current window boundary.

        The median ratio of adjacent kernel increments is robust to one irregular segment.  Raising that
        ratio through the ten further tenths in a doubled window gives a directional projection, not a new
        solve.  A ratio at or above one says extension is not currently reducing the truncation error.
        """
        g = self.compiled.grid; L = float(g.L)
        points = np.arange(0.6, 1.01, 0.1) * L
        I = g.interp(points)
        worst = None
        for name in self.compiled.prim:
            K = np.asarray(self.kernel(name), dtype=float); peak = float(np.max(np.abs(K)))
            if peak <= 0:
                continue
            steps = np.max(np.abs(np.diff(I @ K, axis=0)), axis=1) / peak
            if worst is None or steps[-1] > worst[0][-1]:
                worst = (steps, name)
        if worst is None:
            steps, name = np.zeros(4), ""
        else:
            steps, name = worst
        ratios = [float(b / a) for a, b in zip(steps[:-1], steps[1:]) if a > 1e-15]
        ratio = float(np.median(ratios)) if ratios else 0.0
        assessment = "decaying" if ratio < 0.9 else "slow_decay" if ratio < 1.0 else "not_decaying"
        predicted = float(steps[-1] * ratio ** 10) if np.isfinite(ratio) else float("inf")
        return {"assessment": assessment, "ratio": ratio, "predicted_at_double_window": predicted,
                "projection_range": [0.5 * predicted, 2.5 * predicted],
                "quantity": name, "segment_changes": [float(x) for x in steps],
                "method": "median ratio of kernel changes over [0.6L, 0.7L], ..., [0.9L, L]",
                "range_basis": "0.5x to 2.5x the projection covered seven resolved doubled-window benchmarks"}

    def _kernel_change(self, fine) -> float:
        I = fine.compiled.grid.interp(self.ages)
        worst = 0.0
        for name in self.compiled.prim:
            K0 = self.kernel(name); K1 = I @ fine.kernel(name)
            worst = max(worst, float(np.abs(K0 - K1).max() / max(1e-12, np.abs(K0).max())))
        return worst

    def _kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """Closed-loop kernel of a quantity at the shock ages: (N, nW), or (N,) for one channel."""
        c = self.compiled
        K = self.world[c.block(name)] if name in c.index else c.expr_op(c.model.expand({name: 1.0})) @ self.world
        return K if channel is None else K[:, self.shocks.index(channel)]

    def deviation_response(self, origin, quantities, control=None) -> "SeedResponse":
        """How quantities respond to a unit deviation seed of `origin` (a unit impulse of its control, the first
        unless `control` names one), as functions of the seed's age: `.over(ages)`, one column per quantity.  The
        players privy to the origin (Chapter 6's monitoring relation) respond through their response kernels, the
        origin itself resuming play after the blip; naive players filter it.  Without privy players it is the
        impulse response with the origin's own reaction frozen, the all-naive corner's."""
        origin = _nm(origin)
        a = next((x for x in self.model.agents if x.name == origin), None)
        if a is None:
            raise KeyError(f"no agent {origin!r}; the agents are {[x.name for x in self.model.agents]}")
        o = 0 if control is None else a.controls.index(_nm(control))
        S = self._make_solver(self.model); c = self.compiled
        if len(self.model.privy(origin)) > 1:
            W = S._monitoring(self.maps)[1][origin][:, o]
        else:
            W = c.closed_loop(self.maps, excluded=origin, impulse_controls=a.controls)[:, c.nW + o]
        names = [quantities] if isinstance(quantities, str) or not hasattr(quantities, "__iter__") else list(quantities)
        names = [_nm(q) for q in names]
        K = np.stack([W[c.block(n)] if n in c.index else c.expr_op(c.model.expand({n: 1.0})) @ W for n in names], axis=1)
        return SeedResponse(c.grid, K, names)

    def plot(self, path: str) -> None:
        """Kernels by shock for every state and control, one panel per quantity (needs matplotlib)."""
        from .plotting import plot_stationary
        plot_stationary(self, path)

    def grid_summary(self) -> dict:
        g = self.compiled.grid
        return {"kind": "stationary", "breakpoints": [float(b) for b in g.breakpoints], "nodes_per_panel": g.n,
                "ages": g.nodes.tolist()}

    COST_LABEL = "flow loss"

    def _grid_line(self) -> str:
        c = self.compiled
        return f"grid {c.grid.P} panels x {c.grid.n} nodes on [0, {c.grid.L}], rho={c.rho}"

    def _agent_lines(self, agent) -> list:
        """The stationary engine shows each control's weight on the newest shock, which is the number
        a reader compares across channels."""
        out = []
        for u in agent.controls:
            k = self.kernel(u)
            out.append(f"    {u}(0+) on channels: " + ", ".join(f"{ch}={k[0, j]:+.4f}" for j, ch in enumerate(self.compiled.channels)))
        return out

    def _means_line(self) -> str:
        return "  means: " + ", ".join(f"{n}={self._mz(self.means[n]):+.6f}" for n in self.compiled.prim)


def _stationary_projection(res, agent, K: np.ndarray) -> np.ndarray:
    """E[L | agent's information] on the stationary engine: the least-squares map on the agent's seen rows that
    reproduces the kernel K (the engine's own _project, as for an action), read back through the row operator."""
    agent = _nm(agent)
    a = next((x for x in res.model.agents if x.name == agent), None)
    if a is None:
        raise KeyError(f"no agent {agent!r}; the agents are {[x.name for x in res.model.agents]}")
    S = res._make_solver(res.model); Z = res.world; nW = S.c.nW
    K = np.asarray(K, dtype=float)
    g = S._project(a, Z, np.repeat(K[None], len(a.controls), axis=0))
    rows, inst = S._seen_rows(a, Z, set())
    Bk = S._row_operator(a, rows, inst)
    return np.stack([Bk[k] @ g[0].reshape(-1) for k in range(nW)], axis=1)


@dataclass(repr=False)
class TriangleResult(Result):
    kind: str = "finite"
    past: object = None                     # the Past a transition started from (None: the game starts at rest)
    continuation: object = None             # the StationaryResult the maps are frozen at after T (None: the game ends at T)
    settled: Optional[float] = None         # with a continuation: how far the maps on [T - L, T] are from its stationary maps
    actions: Optional[Dict[str, np.ndarray]] = None   # agent -> the fixed point's action kernels (nU, N, ncol) when it iterated on them (the march's warm start)
    SETTLED_TOL = tunable("settled_tol")
    MAP_CONVENTION = ("the map on a row observed with delay d is stored at the shifted time t - d: maps[agent][u][row][n] "
                      "is the weight the control at time agents[agent].signals[row].map_time[n] = grid.t[n] + delay puts "
                      "on the increment of the row as the agent sees it at age grid.age[n]; that increment entered the raw "
                      "row at time grid.s[n] (age map_age[n] = grid.age[n] + delay before the control), and the map is zero "
                      "where map_time is beyond the horizon; with a past the map is stored in raw age instead: "
                      "maps[agent][u][row][n] is the weight the control at map_time[n] = grid.t[n] puts on the raw increment "
                      "of age map_age[n] = grid.age[n] (zero below the delay), nodes with grid.s[n] < 0 weigh the increments "
                      "observed before zero, and the entries after the grid's nodes (map_init_time) are the weights on the "
                      "row's point observation of the initial shocks")

    @property
    def grid(self):
        return self.compiled.g

    @property
    def stationary(self):
        """The new regime's stationary result the transition is continued by (res.continuation; None when the
        game ends at T)."""
        return self.continuation

    @property
    def shocks(self) -> List[str]:
        """The columns of the kernels: the channels, then the initial shocks of the past."""
        return list(self.compiled.channels) + list(getattr(self.compiled, "init_names", []))

    def _node_axes(self) -> dict:
        g = self.grid
        return {"time": g.t, "age": g.a, "shock_time": g.s}

    @property
    def times(self) -> Optional[np.ndarray]:
        return self.mean_times

    @property
    def extra(self) -> dict:
        out = {}
        if self.past is not None:
            out.update(past=self.past, continuation=self.continuation, settled=self.settled)
        if self.representation_parts:
            out["representation_parts"] = self.representation_parts
        return out

    def map_axes(self, delay: float) -> dict:
        g = self.grid; c = self.compiled
        if self.past is not None:
            delay = 0.0                                       # raw-age storage (MAP_CONVENTION)
        out = {"map_time": (g.t + delay).tolist(), "map_age": (g.a + delay).tolist()}
        if getattr(c, "n_init", 0):
            out["map_init_time"] = (c.tm + delay).tolist()
        return out

    def _kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """Closed-loop kernel at the triangle nodes (res.grid.t, res.grid.s): (N, nW) or (N,); the band's
        nodes (s < 0) are the old shocks.  channel may also name an initial shock of the past (its kernel
        is meaningful on the nodes with s = 0)."""
        c = self.compiled
        K = self.world[c.block(name)] if name in c.index else c.expr_op(c.model.expand({name: 1.0})) @ self.world
        return K if channel is None else K[:, self.shocks.index(channel)]

    def plot(self, path: str) -> None:
        """Each kernel as a function of the shock time s at five dates t (needs matplotlib)."""
        from .plotting import plot_triangle
        plot_triangle(self, path)

    def _kernel_change(self, fine) -> float:
        # read the fine kernel at the coarse nodes from the same side of each piece boundary as the coarse
        # node (kernels jump across the delay line; a one-sided read from the other side is not an error)
        g = self.grid; worst = 0.0
        I = fine.grid.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
        for name in self.compiled.prim:
            for ch in self.compiled.channels:
                K0 = self.kernel(name, ch); K1 = I @ fine.kernel(name, ch)
                worst = max(worst, float(np.abs(K0 - K1).max() / max(1e-12, np.abs(K0).max())))
        return worst

    def _project(self, agent, K: np.ndarray) -> np.ndarray:
        """The kernel (N, ncol) of E[L_t | agent's information at t] for the linear process L with kernel K at the
        triangle nodes: the projection on the agent's seen rows that belief_error and the representation error use."""
        agent = _nm(agent)
        a = next((x for x in self.model.agents if x.name == agent), None)
        if a is None:
            raise KeyError(f"no agent {agent!r}; the agents are {[x.name for x in self.model.agents]}")
        solver = self._make_solver(self.model)
        if not hasattr(solver, "maps_from_world"):
            raise NotImplementedError(f"estimates need the spectral finite engine, not {type(solver).__name__}")
        from .finite_free import reconstruction
        K = np.repeat(np.asarray(K)[None], len(a.controls), axis=0)       # one copy per control: the projection's shape
        return reconstruction(solver, a, self.world, solver.maps_from_world(a, self.world, K))[0]

    def deviation_response(self, origin, quantities, control=None) -> "SeedResponse2":
        """How quantities respond to a unit deviation seed of `origin` (a spike of its control, the first unless
        `control` names one) at time s, seen at time t >= s: `.over(t, s)`, one column per quantity.  As on the
        stationary engine: the players privy to the origin respond through their response kernels, the origin
        resuming play after the blip; naive players filter the seed."""
        origin = _nm(origin)
        a = next((x for x in self.model.agents if x.name == origin), None)
        if a is None:
            raise KeyError(f"no agent {origin!r}; the agents are {[x.name for x in self.model.agents]}")
        o = 0 if control is None else a.controls.index(_nm(control))
        S = self._make_solver(self.model); c = self.compiled
        if len(self.model.privy(origin)) > 1:
            W = S._monitoring(self.maps)[1][origin][:, o]
        else:
            W = S._spikes(c, self.maps, a)[1][:, o]
        names = [quantities] if isinstance(quantities, str) or not hasattr(quantities, "__iter__") else list(quantities)
        names = [_nm(q) for q in names]
        K = np.stack([W[c.block(n)] if n in c.index else c.expr_op(c.model.expand({n: 1.0})) @ W for n in names], axis=1)
        return SeedResponse2(self.grid, K, names)

    def evaluate(self, name: str, shock: str, t, s) -> np.ndarray:
        """Kernel value at (t, s) points: response at time t to a unit `shock` at time s."""
        name, shock = _nm(name), _nm(shock)
        t = np.asarray(t, dtype=float); s = np.asarray(s, dtype=float)
        return self.grid.interp(t, t - s) @ self.kernel(name, shock)

    def mean(self, name: str, t) -> np.ndarray:
        """The mean path of `name` (any key of res.means) interpolated at the times t (from above at a breakpoint)."""
        t = np.atleast_1d(np.asarray(t, dtype=float)); g = self.grid
        a = t if g.L is None else np.zeros_like(t)              # the line s = 0, or (a strip, cut at age L) the age-0 line
        return g.interp(t, a) @ (self.compiled.mean_embed @ np.asarray(self.means[name], dtype=float))

    def grid_summary(self) -> dict:
        g = self.grid; c = self.compiled
        out = {"kind": "finite", "breakpoints": [float(b) for b in g.bp], "nodes_per_side": g.nt,
               "t": g.t.tolist(), "age": g.a.tolist(), "s": g.s.tolist()}
        if g.L is not None:
            out["window"] = float(g.L); out["horizon"] = float(c.T)
            if self.continuation is not None:
                out["buffer"] = [float(c.T), float(g.T)]
        return out

    def _check_rows(self) -> List[dict]:
        """The common rows, then a transition's: the past's own window tail, and with a continuation the `settled`
        check (the maps on [T - L, T] against the stationary maps the buffer is frozen at, threshold
        settings.settled_tol) and the continuation's window tail."""
        rows = super()._check_rows()

        def row(name, value, threshold, ok, flag, advice=""):
            rows.append({"name": name, "value": value, "threshold": threshold, "ok": ok, "flag": flag, "advice": advice})
        tol = self.WINDOW_TAIL_TOL
        if self.past is not None and self.past.provenance.get("window_tail") is not None:
            tail = float(self.past.provenance["window_tail"])
            row("past window", tail, tol, bool(tail <= tol),
                f"PAST WINDOW TOO SHORT (a kernel of the past still moves by {tail:.1%} of its peak over the last tenth of its "
                f"window {self.past.window:g}: solve the past with a longer window)", "solve the past with a longer window")
        if self.settled is not None:
            info = self.compiled.continuation_info or {}
            stopped = getattr(self, "march_stop", None) == "max_window"
            row("settled", float(self.settled), self.SETTLED_TOL, bool(self.settled <= self.SETTLED_TOL),
                f"TRANSITION NOT SETTLED by T - L: raise horizon.window (a map on [T - L, T] is {self.settled:.1e} of its peak "
                f"from the stationary map the buffer is frozen at, against settled_tol {self.SETTLED_TOL:g}: the closed-loop "
                "decay over a unit of t, not the grid's floor)"
                + (f"; the settle march stopped at max_window, T = {self.compiled.T:g}, with the gap at "
                   f"{max(self.march[-1].gap.values()):.1e} against settle {self.march_settle:g}: raise max_window" if stopped else ""),
                "raise max_window" if stopped else "raise horizon.window")
            if getattr(self, "march_stop", None) == "floor" and self.march:
                fl = max(self.march_floor.values()) if self.march_floor else float("nan")
                g1 = max(self.march[-1].gap.values())
                where = (f"the settle march stopped at T = {self.compiled.T:g} with the gap at {g1:.1e}, within a factor {2:g} of the floor"
                         if len(self.march) > 1 else "nothing was marched")
                row("settle floor", float(fl), float(self.march_settle), False,
                    f"SETTLE BELOW THE GRID'S FLOOR ({where}: the grid's floor, the one-shot deviation of the stationary rules from "
                    f"themselves at {self.compiled.g.nt} nodes, is {fl:.1e}, above settle {self.march_settle:g}; the maps are within "
                    f"{self.settled:.1e} of the stationary ones: raise numerics.nodes)", "raise numerics.nodes")
            if info.get("window_tail") is not None:
                tail = float(info["window_tail"])
                row("continuation window", tail, tol, bool(tail <= tol),
                    f"CONTINUATION WINDOW TOO SHORT (a kernel of the continuation still moves by {tail:.1%} of its peak over the "
                    f"last tenth of its window {info.get('window', 0):g}: solve it with a longer window)",
                    "solve the continuation with a longer window")
        return rows

    def to_dict(self) -> dict:
        out = super().to_dict()
        if self.past is not None:
            out["past"] = self.past.to_dict()
            out["settled"] = self.settled
            if self.continuation is not None:                # its provenance, in the result and in the solver options
                out["continuation"] = dict(self.compiled.continuation_info)
                out["options"]["solver"]["continuation"] = dict(self.compiled.continuation_info)
        return out

    COST_LABEL = "discounted cost"
    COST_FIGURES = 8

    def _grid_line(self) -> str:
        c = self.compiled
        return (f"triangle grid {c.g.P} panels, {len(c.g.pieces)} pieces x {c.g.nt}x{c.g.na} nodes "
                f"= {c.N} nodes on [0, {c.T}]"
                + (f" and the buffer [{c.T}, {c.g.T}] (maps frozen at the stationary ones)"
                   if self.continuation is not None else "") + f", rho={c.rho}")

    def _means_line(self) -> str:
        c = self.compiled
        at = np.array([0.0, 0.5 * c.T, c.T])
        return "  means at t = 0, T/2, T: " + ", ".join(
            f"{n}=" + "/".join(f"{self._mz(v):+.4f}" for v in self.mean(n, at)) for n in c.prim)


@dataclass(repr=False)
class TransitionResult(TriangleResult):
    """A transition from a known past: the kernels on the strip (s < 0 on the band), the loss paths E[loss(t)] on
    res.times ([0, T] and the buffer), the excess costs over the new stationary flow, the belief errors on
    demand, and the settled guard."""
    kind: str = "transition"
    times: Optional[np.ndarray] = None                              # the time nodes of the paths
    loss_path: Dict[str, np.ndarray] = field(default_factory=dict)   # agent -> E[loss(t)] on times
    excess_costs: Dict[str, float] = field(default_factory=dict)     # agent -> int_0^T e^{-rho t} (E loss(t) - the new stationary flow) dt
    excess_windows: Dict[str, list] = field(default_factory=dict)    # agent -> the excess's discounted integral per window, the last window [T - L, T] first
    excess_costs_tail: Dict[str, float] = field(default_factory=dict)   # agent -> the tail past T: the last window's excess times r / (1 - r)
    excess_costs_total: Dict[str, float] = field(default_factory=dict)  # agent -> excess_costs + excess_costs_tail
    excess_tail: Optional[dict] = None                               # {"source": "loss path" | "march gaps", "factor": {agent: r per window}, "windows": [...]}
    march: Optional[list] = None                                     # the settle march's rows {T, gap, evaluations, seconds, monitor} (transition(settle=))
    march_stop: Optional[str] = None                                 # why it stopped: "settled", "settled at T = 0" or "max_window"
    march_settle: Optional[float] = None                             # its tolerance
    march_floor: Optional[Dict[str, float]] = None                   # the grid's floor per agent (transition.settle_floor)
    march_window: Optional[float] = None                             # the window the march found when it is not the strip's T (0: nothing solved)

    @property
    def paths(self) -> dict:
        out = super().paths
        out["loss"] = {k: np.asarray(v, dtype=float) for k, v in self.loss_path.items()}
        out["belief_error"] = self.belief_error
        return out

    @property
    def extra(self) -> dict:
        out = super().extra
        out.update(old_flows=self.old_flows, new_flows=self.new_flows, excess_costs=dict(self.excess_costs))
        if self.excess_tail is not None:
            out.update(excess_costs_tail=dict(self.excess_costs_tail), excess_costs_total=dict(self.excess_costs_total), excess_tail=self.excess_tail)
        out["T"] = float(self.compiled.T) if self.march_window is None else float(self.march_window)   # the T the game ran to (a march: the T it found)
        if self.march is not None:
            out.update(march=list(self.march), march_stop=self.march_stop, settle_floor=self.march_floor)
        return out

    @property
    def old_flows(self) -> Dict[str, float]:
        """The old regime's stationary flow losses per agent (the past's provenance; empty for a past of shocks)."""
        return {k: float(v) for k, v in (self.past.provenance.get("costs") or {}).items()} if self.past is not None else {}

    @property
    def new_flows(self) -> Dict[str, float]:
        """The new regime's stationary flow losses per agent (the continuation's; empty when the game ends at T)."""
        return {k: float(v) for k, v in self.continuation.costs.items()} if self.continuation is not None else {}

    def belief_error(self, agent: str, name: str) -> np.ndarray:
        """The variance of `agent`'s estimation error of the quantity `name` (a state, control or definition)
        at every time node res.times: the quantity's kernel minus its projection on the agent's seen rows (one
        weighted least-squares Gram per date), integrated over the shocks alive."""
        a = next((x for x in self.model.agents if x.name == agent), None)
        if a is None:
            raise KeyError(f"no agent {agent!r}; the agents are {[x.name for x in self.model.agents]}")
        return self._make_solver(self.model).belief_error(a, name, self.world)

    def grid_summary(self) -> dict:
        out = super().grid_summary(); out["kind"] = "transition"
        return out

    def to_dict(self) -> dict:
        out = super().to_dict()
        out["times"] = None if self.times is None else self.times.tolist()
        out["loss_path"] = {k: v.tolist() for k, v in self.loss_path.items()}
        out["belief_error"] = {a.name: {name: self.belief_error(a.name, name).tolist() for name in self.model.state_names}
                               for a in self.model.agents}
        out["excess_costs"] = {k: float(v) for k, v in self.excess_costs.items()}
        out["old_flows"] = self.old_flows; out["new_flows"] = self.new_flows
        if self.excess_tail is not None:
            out["excess_windows"] = {k: [float(x) for x in v] for k, v in self.excess_windows.items()}
            out["excess_costs_tail"] = {k: float(v) for k, v in self.excess_costs_tail.items()}
            out["excess_costs_total"] = {k: float(v) for k, v in self.excess_costs_total.items()}
            out["excess_tail"] = {"source": self.excess_tail["source"], "factor": {k: float(v) for k, v in self.excess_tail["factor"].items()},
                                  "windows": [list(w) for w in self.excess_tail["windows"]]}
        out["T"] = float(self.compiled.T) if self.march_window is None else float(self.march_window)   # the T the game ran to (a march: the T it found)
        if self.march is not None:
            out["march"] = [r.to_dict() for r in self.march]      # MarchPoint serialises itself
            out["march_stop"] = self.march_stop; out["march_settle"] = self.march_settle
            out["settle_floor"] = None if self.march_floor is None else {k: float(v) for k, v in self.march_floor.items()}
        return out

    def plot(self, path: str) -> None:
        """The kernels against the shock time s from -L at five dates (the band s < 0 shaded), a row with
        E[loss(t)] per agent and the old and new stationary flows as horizontal lines, a row with each agent's
        belief-error variance of every state, and the mean paths when driven (needs matplotlib)."""
        from .plotting import plot_transition
        plot_transition(self, path)

    def summary(self, diagnostics: bool = True) -> str:
        lines = [super().summary(diagnostics=diagnostics)]
        if self.excess_costs:
            lines.append("  excess cost over the new stationary flow on [0, T]: " + ", ".join(f"{k}={v:+.6f}" for k, v in self.excess_costs.items()))
        if self.excess_costs_total:
            f = self.excess_tail["factor"]
            lines.append(f"  with the tail past T at the closed-loop rate (factor per window from the {self.excess_tail['source']}: "
                         + ", ".join(f"{k}={v:.2e}" for k, v in f.items()) + "): " + ", ".join(f"{k}={v:+.6f}" for k, v in self.excess_costs_total.items()))
        return "\n".join(lines)


