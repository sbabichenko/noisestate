"""Results returned by the engines: one Result, three grids.

    res.converged, res.residual, res.message     outcome of the outer solve
    res.evaluations                              best-response evaluations made (res.iterations, deprecated, warns)
    res.status                                   {"ok", "flags", "rows"}: the verdict, the failing checks' flags, diagnose()'s rows
    res.check()                                  raise ConvergenceError unless converged
    res.axes                                     the coordinate arrays of kernel(): {"age": ages} (stationary), {"time", "age",
                                                 "shock_time"} node-wise (spectral finite; shock_time < 0 on a transition's band),
                                                 {"time", "shock_time"} (cells, the two axes of the (N, N) matrix); and "maps":
                                                 {agent: {row: {axis: values}}}, where each row's map values belong (map_axes)
    res.times                                    the time nodes of the paths (None on the stationary engine)
    res.paths                                    {"means": {name: path}} over res.times on a finite horizon; a transition adds
                                                 "loss" ({agent: E[loss(t)]}) and "belief_error" (a callable (agent, name) -> path)
    res.world                                    the closed-loop kernels of every primary on the shocks (res.Z, deprecated, warns)
    res.extra                                    engine-specific extras: window_tail (stationary); past, continuation, settled
                                                 (a transition), old_flows, new_flows, excess_costs, representation_parts
    res.numerics                                 the resolved Numerics the result was solved with
    res.costs[agent]                             stationary flow loss per unit time, or the discounted
                                                 integral over [0, T] (res.cost_kind says which): the variance
                                                 part (the shocks) plus the mean part (targets, constant drifts)
    res.cost_parts[agent]                        {"variance", "mean"}, the two parts
    res.means[name]                              mean of every state, control and definition, and the mean drift rate of
                                                 every signal row as "agent.row": a constant (stationary) or the path on the
                                                 time nodes res.means_t (finite; the spectral res.mean(name, t) interpolates)
    res.kernel(name, channel=None)               closed-loop kernel of a state or control
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
  CellResult          kernel(name, ch) -> (N, N) matrix K[i, j]: response at cell i to a unit
                      increment of the channel in cell j (channel required)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .accel import ConvergenceError
from .settings import DEFAULT, Settings, tunable
from .kernel import Kernel, AttrDict
from .spec import Model


class _StabilityBudget(Exception):
    """Raised by stability()'s matvec at its evaluation budget, so ARPACK unwinds to the fallback."""


@dataclass
class Result:
    """One result type; the engines' subclasses (StationaryResult, TriangleResult, TransitionResult, CellResult)
    fill in the grid-specific hooks and are internal: isinstance(res, noisestate.Result) holds for every
    result, and the attributes above read the same on every engine."""
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
    representation_error: Dict[str, float] = field(default_factory=dict)   # agent -> relative residual, see resolution_ok
    representation_parts: Dict[str, Dict[str, float]] = field(default_factory=dict)   # agent -> where it sits (transition: interior / band tip / last window)
    refinement: Optional[dict] = None          # filled by refine(): change of costs/kernels under a finer grid
    solver_class: object = None                # the engine that produced this result, with its options, so that
    solver_kw: dict = field(default_factory=dict)      # refine()/stability() rebuild the same solver
    solve_kw: dict = field(default_factory=dict)
    second_order: Dict[str, dict] = field(default_factory=dict)   # agent -> {"min", "max", "ok"}: is the best response a minimum
    foc: Dict[str, dict] = field(default_factory=dict)            # agent -> control -> {"foc", "physical", "wedge"} kernels
    means: Dict[str, object] = field(default_factory=dict)        # quantity -> its mean, a float (stationary) or a path (finite); zero when nothing drives it
    means_t: Optional[np.ndarray] = None                          # the time nodes of the mean paths (finite engines)
    cost_parts: Dict[str, Dict[str, float]] = field(default_factory=dict)   # agent -> {"variance", "mean"} parts of its cost
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
        The other per-result hooks are kernel(name, channel) (a state's, control's or definition's
        closed-loop kernel in the engine's layout, see the module docstring), grid_info() (the JSON-ready
        grid description of to_dict()), _kernel_change(fine) (the relative change of the kernels against
        a finer result of the same engine, for refine()), summary() and plot(path); window_tail is optional
        (the stationary result's; diagnose() reads it when present)."""
        raise NotImplementedError

    MEAN_ZERO = tunable("mean_zero")          # below this a mean is round-off (the mean system is solved only when something drives it)

    # ----------------------------------------------------------- the common surface (C and E of the API design)
    @property
    def iterations(self) -> int:
        """Deprecated name of evaluations (removed in 0.6): a DeprecationWarning on every read."""
        warnings.warn("res.iterations is deprecated, read res.evaluations (res.iterations goes in 0.6)", DeprecationWarning, stacklevel=2)
        return self.evaluations

    @iterations.setter
    def iterations(self, value: int) -> None:
        warnings.warn("res.iterations is deprecated, set res.evaluations (res.iterations goes in 0.6)", DeprecationWarning, stacklevel=2)
        self.evaluations = int(value)

    @property
    def Z(self) -> np.ndarray:
        """Deprecated name of world (removed in 0.6): a DeprecationWarning on every read."""
        warnings.warn("res.Z is deprecated, read res.world (res.Z goes in 0.6)", DeprecationWarning, stacklevel=2)
        return self.world

    @Z.setter
    def Z(self, value: np.ndarray) -> None:
        warnings.warn("res.Z is deprecated, set res.world (res.Z goes in 0.6)", DeprecationWarning, stacklevel=2)
        self.world = value

    @property
    def times(self) -> Optional[np.ndarray]:
        """The time nodes of the paths; None on the stationary engine (its means are constants)."""
        return None

    def _node_axes(self) -> dict:
        """Hook: the coordinate arrays of kernel()'s node axis (or axes), by name."""
        raise NotImplementedError

    @property
    def axes(self) -> dict:
        """The coordinate arrays of kernel(name, channel) by name (the module docstring lists them per engine),
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

    @property
    def status(self) -> dict:
        """{"ok": converged and no check failed, "flags": the flags of the failing checks (and of the checks
        without a verdict, when they carry one), "rows": diagnose()'s rows}; a dict whose keys are also
        attributes (res.status.ok)."""
        rows = self.diagnose()
        failed = [d["flag"] for d in rows if d["ok"] is False and d["flag"]]
        return AttrDict({"ok": bool(self.converged) and not failed, "flags": failed, "rows": rows})

    @property
    def extra(self) -> dict:
        """Engine-specific extras (the module docstring lists them); {} when the engine has none."""
        return {}

    @property
    def means_driven(self) -> bool:
        """Whether any mean is nonzero beyond round-off."""
        return any(np.max(np.abs(np.asarray(v, dtype=float))) > self.MEAN_ZERO for v in self.means.values() if np.size(v))

    def _mz(self, v: float) -> float:
        """A mean for printing: round-off shown as an unsigned zero."""
        return float(v) if abs(float(v)) > self.MEAN_ZERO else 0.0

    @property
    def resolution_ok(self) -> Optional[bool]:
        """False when the grid is too coarse for the equilibrium it reports (representation error of a
        best response on the raw rows above RESOLUTION_TOL); raise horizon.nodes and re-solve.  None
        when the engine does not compute the representation error (the cell engine)."""
        if not self.representation_error:
            return None
        return all(v <= self.RESOLUTION_TOL for v in self.representation_error.values())

    def _make_solver(self, model: Model):
        """The engine that produced this result, with the same constructor options, on `model`."""
        if self.solver_class is None:
            from .sweep import make_solver
            return make_solver(model)
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
    def check(self):
        """Return self, or raise ConvergenceError if the solve did not reach its tolerance."""
        if not self.converged:
            raise ConvergenceError(f"{self.model.name}: residual {self.residual:.2e} ({self.message})")
        return self

    @property
    def channels(self) -> List[str]:
        return list(self.compiled.channels)

    def kernel(self, name: str, channel: Optional[str] = None) -> "Kernel":
        """The closed-loop kernel of a state, control or definition (every channel, or one), as a Kernel: an
        ndarray in the engine's layout (the subclass's _kernel documents it) carrying .axes (res.axes' node
        coordinates), .at(*coords) (the engine's own interpolant) and .plot(path=None)."""
        return Kernel.of(self._kernel(name, channel), self, name, channel)

    def _kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError

    def action_kernel(self, control: str, channel: Optional[str] = None) -> np.ndarray:
        if control not in self.model.control_names:
            raise KeyError(f"{control!r} is not a control")
        return self.kernel(control, channel)

    def grid_info(self) -> dict:
        raise NotImplementedError

    @property
    def cost_kind(self) -> str:
        return "stationary flow loss per unit time" if self.kind == "stationary" else "discounted integral over [0, T]"

    def refine(self, factor: float = 1.5, **solve_kw) -> dict:
        """Re-solve on a finer grid (nodes x factor) and report the relative change of every agent's
        cost and of the kernels, the honest test of resolution (window, corner and product errors alike).
        Stored in self.refinement and shown by summary()."""
        import math
        n0 = int(self.model.horizon.nodes)
        n1 = 2 * n0 if self.kind == "finite_cells" else max(n0 + 2, int(math.ceil(n0 * factor)))   # cells: keep lags aligned
        # this solve's bounds and a skipped diagnostics pass are not the refinement's
        kw = {k: v for k, v in self.solve_kw.items() if k not in ("init", "start", "max_evaluations", "deadline", "diagnostics")}
        kw.update(solve_kw)
        solver = self._make_solver(self.model.with_numerics(nodes=n1))
        try:                                                  # start the fine solve from this equilibrium, interpolated
            kw.setdefault("init", solver.interpolate_maps(self))
        except NotImplementedError:
            pass
        fine = solver.solve(**kw)
        scale = max(1e-300, max(abs(v) for v in self.costs.values()))          # one scale: a zero-profit agent is not "unresolved"
        cost_change = max(abs(fine.costs[k] - self.costs[k]) / scale for k in self.costs)
        kernel_change = self._kernel_change(fine)
        rep = {"nodes": n1, "converged": bool(fine.converged), "cost_change": float(cost_change),
               "kernel_change": float(kernel_change)}
        # the spectral engines converge exponentially, so a small change means resolved; the cell engine is
        # first order and its change halves per doubling: no verdict, the numbers are the report
        rep["resolved"] = None if self.kind == "finite_cells" else bool(fine.converged and cost_change < self.REFINE_COST_TOL
                                                                        and kernel_change < self.REFINE_KERNEL_TOL)
        self.refinement = rep
        return rep

    def _kernel_change(self, fine) -> float:
        raise NotImplementedError

    REFINE_COST_TOL, REFINE_KERNEL_TOL = tunable("refine_cost_tol"), tunable("refine_kernel_tol")

    def diagnose(self) -> List[dict]:
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
        row("resolution", rep, self.RESOLUTION_TOL, self.resolution_ok,
            f"UNDER-RESOLVED (representation error {rep:.1e}{at}: raise horizon.nodes" + ("; an error only on the band tip or the last "
            "window is the geometry there, not the interior's resolution)" if where else ")") if rep is not None else "", "raise horizon.nodes")
        tail = getattr(self, "window_tail", None)
        if tail is not None:
            row("window", float(tail), self.WINDOW_TAIL_TOL, bool(tail <= self.WINDOW_TAIL_TOL),
                f"WINDOW TOO SHORT (a kernel still moves by {tail:.1%} of its peak over the last tenth of the window: raise horizon.window"
                + ("; the means' continuation integrals are truncated there as well)" if self.means_driven else ")"),
                "raise horizon.window")
        for a, so in self.second_order.items():
            if so.get("converged") is False:
                row(f"second_order:{a}", None, None, None, f"second-order check did not converge for {a!r}", so.get("message", ""))
            else:
                row(f"second_order:{a}", so["min"], -self.settings.second_order_tol if self.solver_class is not None else None, so["ok"],
                    f"NOT A MINIMUM (the best response of {a!r} is a saddle: its loss is not convex in its own strategy, "
                    f"smallest curvature {so['min']:.1e} of the largest)", "the loss is not convex in the agent's own strategy")
                if so.get("edge"):
                    row(f"second_order_edge:{a}", so["min"], -self.settings.second_order_tol if self.solver_class is not None else None, None,
                        f"window edge: the curvature of {a!r} is negative ({so['min']:.1e}) on this window but positive "
                        f"({so['embedded']:.1e}) on a window longer by two lags: a truncation of the lagged loss terms at the edge, not a saddle",
                        "a wider window moves it, a quadratic term in the control's current value removes it")
        if self.refinement:
            f = self.refinement
            row("refinement", {"cost_change": f["cost_change"], "kernel_change": f["kernel_change"], "nodes": f["nodes"]},
                {"cost_change": self.REFINE_COST_TOL, "kernel_change": self.REFINE_KERNEL_TOL}, f["resolved"],
                f"refinement to {f['nodes']} nodes moves costs by {f['cost_change']:.1e} and kernels by {f['kernel_change']:.1e}"
                + ("" if f["resolved"] in (True, None) else " (NOT RESOLVED)"), "raise horizon.nodes")
        st = getattr(self, "stability_report", None)
        if st:
            row("stability", float(st["radius"]), 1.0, bool(st["stable"]),
                f"best-response dynamics {'stable' if st['stable'] else 'UNSTABLE'} (spectral radius {st['radius']:.3f}"
                + (", untied game" if st["untied"] else "") + (f", by {st['method']}" if st["method"] != "arnoldi" else "") + ")",
                "naive best-response adjustment would not find this equilibrium")
        return rows

    def _status(self) -> str:
        """One line: the outcome, then every check that failed or has no verdict, and the informational
        rows (refinement, stability, a skipped diagnostics pass) whenever they were computed."""
        rows = self.diagnose()
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

    def stability(self, untied: bool = True) -> dict:
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
        rep = {"radius": radius, "eigenvalues": [complex(x) for x in np.asarray(vals)], "stable": bool(radius < 1.0),
               "evaluations": int(count[0]), "untied": bool(untied and self.model.ties), "method": method,
               "fixed_point_residual": float(np.linalg.norm(F0 - z0) / scale)}
        self.stability_report = rep
        return rep

    def to_dict(self) -> dict:
        """JSON-serialisable view.  Provenance: the package version, the parameter values, the model spec
        (`model`, which Model.from_dict rebuilds), the horizon and the engine and solve options.  Then the
        grid, the kernels per quantity and channel, the raw maps with each agent's controls and signal rows
        (delay and map axes, see MAP_CONVENTION), costs with their variance and mean parts, the means, and the
        first-order-condition decomposition where the engine provides it."""
        from . import __version__
        c = self.compiled; m = self.model
        num = self.numerics
        out = {"payload_version": 1, "version": __version__, "name": m.name, "engine": num.engine, "kind": self.kind,
               "converged": bool(self.converged),
               "residual": float(self.residual), "evaluations": int(self.evaluations), "seconds": float(self.seconds),
               "message": self.message, "params": {k: float(v) for k, v in m.params.items()}, "model": m.to_dict(),
               "horizon": m.to_dict()["horizon"], "numerics": num.to_dict(),
               "axes": {k: np.asarray(v).tolist() for k, v in self._node_axes().items()},
               "times": None if self.times is None else np.asarray(self.times).tolist(),
               "options": {"numerics": num.to_dict(),
                           "solver": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in self.solver_kw.items()},
                           "solve": dict(self.solve_kw)},
               "grid": self.grid_info(), "discount": float(c.rho), "channels": self.channels,
               "agents": {a.name: {"controls": list(a.controls),
                                   "signals": {r.name: {"delay": float(r.delay), **self.map_axes(r.delay)} for r in a.signals}}
                          for a in m.agents},
               "map_convention": self.MAP_CONVENTION, "kernels": {}, "maps": {}, "foc": {},
               "costs": {k: float(v) for k, v in self.costs.items()},
               "cost_parts": {a: {k: float(v) for k, v in p.items()} for a, p in self.cost_parts.items()},
               "means": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in self.means.items()},
               "means_t": None if self.means_t is None else self.means_t.tolist()}
        out["representation_error"] = {k: float(v) for k, v in self.representation_error.items()}
        if self.representation_parts:
            out["representation_parts"] = {a: {k: float(v) for k, v in p.items()} for a, p in self.representation_parts.items()}
        out["diagnostics"] = [{k: (None if v is None else v) for k, v in d.items()} for d in self.diagnose()]
        out["resolution_ok"] = None if self.resolution_ok is None else bool(self.resolution_ok)
        st = self.status
        out["status"] = {"ok": st["ok"], "flags": st["flags"]}
        out["cost_kind"] = self.cost_kind
        out["second_order"] = {a: dict(so) for a, so in self.second_order.items()}
        out["notes"] = self.model.notes
        if self.refinement:
            out["refinement"] = self.refinement
        tail = getattr(self, "window_tail", None)
        if tail is not None:
            out["window_tail"] = float(tail)
        rep = getattr(self, "stability_report", None)
        if rep:
            out["stability"] = {"radius": rep["radius"], "stable": rep["stable"], "untied": rep["untied"], "method": rep["method"],
                                "eigenvalues": [[x.real, x.imag] for x in rep["eigenvalues"]]}
        for name in c.prim:
            out["kernels"][name] = {ch: self.kernel(name, ch).tolist() for ch in self.channels}
        for a in self.model.agents:
            g = self.maps[a.name]
            out["maps"][a.name] = {u: {r.name: g[ui, ri].tolist() for ri, r in enumerate(a.signals)}
                                   for ui, u in enumerate(a.controls)}
        if self.foc:
            for aname, dec in self.foc.items():
                out["foc"][aname] = {u: {part: {ch: arr[:, k].tolist() for k, ch in enumerate(self.channels)}
                                         for part, arr in parts.items()} for u, parts in dec.items()}
        return out


BaseResult = Result                      # the old name (until 0.6)


@dataclass
class StationaryResult(Result):
    kind: str = "stationary"
    MAP_CONVENTION = ("maps[agent][u][row][n] is the weight the control puts on the increment of the row as the agent "
                      "sees it (delayed) at age ages[n] before the control; that increment entered the raw row at age "
                      "agents[agent].signals[row].map_age[n] = ages[n] + delay, and the map is zero where map_age is "
                      "beyond the window")

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
        return K if channel is None else K[:, self.channels.index(channel)]

    def plot(self, path: str) -> None:
        """Kernels by channel for every state and control, one panel per quantity (needs matplotlib)."""
        plt = _pyplot(); c = self.compiled
        names = self.model.state_names + self.model.control_names
        nrow = (len(names) + 1) // 2
        fig, axes = plt.subplots(nrow, 2, figsize=(10.4, 2.6 * nrow), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            K = self.kernel(name)
            for k, ch in enumerate(self.channels):
                if np.abs(K[:, k]).max() > 1e-12:
                    ax.plot(c.grid.nodes, K[:, k], lw=1.1, label=ch)
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name}: kernel by channel", fontsize=10); ax.set_xlabel("shock age")
            ax.legend(fontsize=7, frameon=False, ncol=2)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.suptitle(f"{self.model.name}  (residual {self.residual:.1e})", fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150)

    def grid_info(self) -> dict:
        g = self.compiled.grid
        return {"kind": "stationary", "breakpoints": [float(b) for b in g.breakpoints], "nodes_per_panel": g.n,
                "ages": g.nodes.tolist()}

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.evaluations} evaluations, "
                 f"{self.seconds:.1f}s; grid {c.grid.P} panels x {c.grid.n} nodes on [0, {c.grid.L}], rho={c.rho}"]
        for a in self.model.agents:
            parts = self.cost_parts.get(a.name)
            lines.append(f"  {a.name}: flow loss = {self.costs.get(a.name, float('nan')):+.6f}"
                         + (f" (variance {parts['variance']:+.6f}, mean {parts['mean']:+.6f})" if parts and parts["mean"] != 0.0 else ""))
            for u in a.controls:
                k = self.kernel(u)
                lines.append(f"    {u}(0+) on channels: " + ", ".join(f"{ch}={k[0, j]:+.4f}" for j, ch in enumerate(self.channels)))
        if self.means_driven:
            lines.append("  means: " + ", ".join(f"{n}={self._mz(self.means[n]):+.6f}" for n in c.prim))
        return "\n".join(lines)


@dataclass
class TriangleResult(Result):
    kind: str = "finite"
    past: object = None                     # the Past a transition started from (None: the game starts at rest)
    continuation: object = None             # the StationaryResult the maps are frozen at after T (None: the game ends at T)
    settled: Optional[float] = None         # with a continuation: how far the maps on [T - L, T] are from its stationary maps
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
        return self.means_t

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
        T = self.compiled.T
        def curves(name, ch):
            for t in np.linspace(0.2, 1.0, 5) * T:
                s = np.linspace(0, t, 200)
                yield t, s, self.evaluate(name, ch, np.full_like(s, t), s)
        _plot_by_shock_time(self, curves, path)

    def _kernel_change(self, fine) -> float:
        # read the fine kernel at the coarse nodes from the same side of each piece boundary as the coarse
        # node (kernels jump across the delay line; a one-sided read from the other side is not an error)
        g = self.grid; worst = 0.0
        I = fine.grid.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_ds)
        for name in self.compiled.prim:
            for ch in self.channels:
                K0 = self.kernel(name, ch); K1 = I @ fine.kernel(name, ch)
                worst = max(worst, float(np.abs(K0 - K1).max() / max(1e-12, np.abs(K0).max())))
        return worst

    def evaluate(self, name: str, channel: str, t, s) -> np.ndarray:
        """Kernel value at (t, s) points: response at time t to a unit shock of `channel` at time s."""
        t = np.asarray(t, dtype=float); s = np.asarray(s, dtype=float)
        return self.grid.interp(t, t - s) @ self.kernel(name, channel)

    def mean(self, name: str, t) -> np.ndarray:
        """The mean path of `name` (any key of res.means) interpolated at the times t (from above at a breakpoint)."""
        t = np.atleast_1d(np.asarray(t, dtype=float)); g = self.grid
        a = t if g.L is None else np.zeros_like(t)              # the line s = 0, or (a strip, cut at age L) the age-0 line
        return g.interp(t, a) @ (self.compiled.mean_embed @ np.asarray(self.means[name], dtype=float))

    def grid_info(self) -> dict:
        g = self.grid; c = self.compiled
        out = {"kind": "finite", "breakpoints": [float(b) for b in g.bp], "nodes_per_side": g.nt,
               "t": g.t.tolist(), "age": g.a.tolist(), "s": g.s.tolist()}
        if g.L is not None:
            out["window"] = float(g.L); out["horizon"] = float(c.T)
            if self.continuation is not None:
                out["buffer"] = [float(c.T), float(g.T)]
        return out

    def diagnose(self) -> List[dict]:
        """The common rows, then a transition's: the past's own window tail, and with a continuation the `settled`
        check (the maps on [T - L, T] against the stationary maps the buffer is frozen at, threshold
        settings.settled_tol) and the continuation's window tail."""
        rows = super().diagnose()

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
                   f"{max(self.march[-1]['gap'].values()):.1e} against settle {self.march_settle:g}: raise max_window" if stopped else ""),
                "raise max_window" if stopped else "raise horizon.window")
            if getattr(self, "march_stop", None) == "floor" and self.march:
                g1, g0 = max(self.march[-1]["gap"].values()), max(self.march[-2]["gap"].values())
                row("settle floor", float(g1), float(self.march_settle), False,
                    f"SETTLE BELOW THE GRID'S FLOOR (the settle march stopped at T = {self.compiled.T:g}: the gap fell from {g0:.1e} to "
                    f"{g1:.1e} over the last window, where a transient falls by hundreds, so settle {self.march_settle:g} is below what "
                    f"{self.compiled.g.nt} nodes resolve; the maps are within {self.settled:.1e} of the stationary ones: raise numerics.nodes)",
                    "raise numerics.nodes")
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
            for name in self.compiled.prim:
                for ch in self.shocks[len(self.channels):]:
                    out["kernels"][name][ch] = self.kernel(name, ch).tolist()
        return out

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.evaluations} evaluations, "
                 f"{self.seconds:.1f}s; triangle grid {c.g.P} panels, {len(c.g.pieces)} pieces x {c.g.nt}x{c.g.na} nodes "
                 f"= {c.N} nodes on [0, {c.T}]" + (f" and the buffer [{c.T}, {c.g.T}] (maps frozen at the stationary ones)"
                                                  if self.continuation is not None else "") + f", rho={c.rho}"]
        for a in self.model.agents:
            parts = self.cost_parts.get(a.name)
            lines.append(f"  {a.name}: discounted cost = {self.costs.get(a.name, float('nan')):+.8f}"
                         + (f" (variance {parts['variance']:+.8f}, mean {parts['mean']:+.8f})" if parts and parts["mean"] != 0.0 else "")
                         + (f" + continuation {parts['continuation']:+.8f} on the buffer" if parts and "continuation" in parts else ""))
        if self.means_driven:
            at = np.array([0.0, 0.5 * c.T, c.T])
            lines.append("  means at t = 0, T/2, T: " + ", ".join(f"{n}=" + "/".join(f"{self._mz(v):+.4f}" for v in self.mean(n, at)) for n in c.prim))
        return "\n".join(lines)


@dataclass
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
        out["window"] = float(self.compiled.T)
        if self.march is not None:
            out.update(march=list(self.march), march_stop=self.march_stop)
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

    def grid_info(self) -> dict:
        out = super().grid_info(); out["kind"] = "transition"
        return out

    def to_dict(self) -> dict:
        out = super().to_dict()
        out["times"] = None if self.times is None else self.times.tolist()
        out["loss_path"] = {k: v.tolist() for k, v in self.loss_path.items()}
        out["excess_costs"] = {k: float(v) for k, v in self.excess_costs.items()}
        out["old_flows"] = self.old_flows; out["new_flows"] = self.new_flows
        if self.excess_tail is not None:
            out["excess_windows"] = {k: [float(x) for x in v] for k, v in self.excess_windows.items()}
            out["excess_costs_tail"] = {k: float(v) for k, v in self.excess_costs_tail.items()}
            out["excess_costs_total"] = {k: float(v) for k, v in self.excess_costs_total.items()}
            out["excess_tail"] = {"source": self.excess_tail["source"], "factor": {k: float(v) for k, v in self.excess_tail["factor"].items()},
                                  "windows": [list(w) for w in self.excess_tail["windows"]]}
        out["window"] = float(self.compiled.T)
        if self.march is not None:
            out["march"] = [{"T": r["T"], "gap": {k: float(v) for k, v in r["gap"].items()}, "evaluations": int(r["evaluations"]),
                             "seconds": float(r["seconds"]), "monitor": r["monitor"]} for r in self.march]
            out["march_stop"] = self.march_stop; out["march_settle"] = self.march_settle
        return out

    def plot(self, path: str) -> None:
        """The kernels against the shock time s from -L at five dates (the band s < 0 shaded), a row with
        E[loss(t)] per agent and the old and new stationary flows as horizontal lines, a row with each agent's
        belief-error variance of every state, and the mean paths when driven (needs matplotlib)."""
        _plot_transition(self, path)

    def summary(self) -> str:
        lines = [super().summary()]
        if self.excess_costs:
            lines.append("  excess cost over the new stationary flow on [0, T]: " + ", ".join(f"{k}={v:+.6f}" for k, v in self.excess_costs.items()))
        if self.excess_costs_total:
            f = self.excess_tail["factor"]
            lines.append(f"  with the tail past T at the closed-loop rate (factor per window from the {self.excess_tail['source']}: "
                         + ", ".join(f"{k}={v:.2e}" for k, v in f.items()) + "): " + ", ".join(f"{k}={v:+.6f}" for k, v in self.excess_costs_total.items()))
        return "\n".join(lines)


@dataclass
class CellResult(Result):
    kind: str = "finite_cells"
    MAP_CONVENTION = ("maps[agent][u][row][i][v] is the weight the control in cell i (time grid.t[i]) puts on the increment "
                      "of the row as the agent sees it in cell v; that increment entered the raw row in cell v - delay / h "
                      "(time agents[agent].signals[row].map_shock_time[v] = grid.t[v] - delay), and cells v below the delay "
                      "are zero")

    @property
    def times(self) -> np.ndarray:
        return self.compiled.times

    def _node_axes(self) -> dict:
        return {"time": self.times, "shock_time": self.times}

    def map_axes(self, delay: float) -> dict:
        return {"map_time": self.times.tolist(), "map_shock_time": (self.times - delay).tolist()}

    def _kernel_change(self, fine) -> float:
        # compare at the coarse cell times: fine cell index = round(t / h_fine)
        c0, c1 = self.compiled, fine.compiled; worst = 0.0
        idx = np.clip(np.round(c0.times / c1.h).astype(int), 0, c1.N - 1)
        for name in c0.prim:
            for ch in self.channels:
                K0 = self.kernel(name, ch); K1 = fine.kernel(name, ch)[np.ix_(idx, idx)]
                worst = max(worst, float(np.abs(K0 - K1).max() / max(1e-12, np.abs(K0).max())))
        return worst

    def _kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """K[i, j]: response of `name` at cell i to a unit increment of `channel` in cell j, (N, N); without a
        channel the stack over the channels, (N, N, nW): kernel(name)[..., k] is kernel(name, channels[k]), the
        last axis one column per channel as on the other engines."""
        c = self.compiled
        K = c.expr_kernel(self.world, c.model.expand({name: 1.0}))
        if channel is None:
            return np.stack([K[:, k * c.N:(k + 1) * c.N] for k in range(len(c.channels))], axis=-1)
        k = c.channels.index(channel)
        return K[:, k * c.N:(k + 1) * c.N]

    def grid_info(self) -> dict:
        c = self.compiled
        return {"kind": "finite_cells", "cells": int(c.N), "h": float(c.h), "t": c.times.tolist()}

    def plot(self, path: str) -> None:
        """Each kernel as a function of the shock time s at five dates t (needs matplotlib)."""
        N, h = self.compiled.N, self.compiled.h
        def curves(name, ch):
            K = self.kernel(name, ch)
            for t_i in np.linspace(N // 5, N - 1, 5).astype(int):
                yield t_i * h, np.arange(t_i) * h, K[t_i, :t_i]
        _plot_by_shock_time(self, curves, path)

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.evaluations} evaluations, "
                 f"{self.seconds:.1f}s; {c.N} cells on [0, {c.T}], rho={c.rho}"]
        for a in self.model.agents:
            parts = self.cost_parts.get(a.name)
            lines.append(f"  {a.name}: discounted cost = {self.costs.get(a.name, float('nan')):+.6f}"
                         + (f" (variance {parts['variance']:+.6f}, mean {parts['mean']:+.6f})" if parts and parts["mean"] != 0.0 else ""))
        if self.means_driven:
            lines.append("  means at t = 0, T/2: " + ", ".join(f"{n}={self._mz(self.means[n][0]):+.4f}/{self._mz(self.means[n][c.N // 2]):+.4f}" for n in c.prim))
        return "\n".join(lines)


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("plotting needs matplotlib: pip install 'noisestate[plot]'") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_transition(res, path: str) -> None:
    plt = _pyplot()
    names = res.model.state_names + res.model.control_names; chans = res.channels
    agents = [a.name for a in res.model.agents]; states = res.model.state_names
    c = res.compiled; g = res.grid; T = c.T; L = g.L or 0.0
    means = res.means_driven and res.means_t is not None
    ncol = max(len(chans), len(agents), 1)
    fig, axes = plt.subplots(len(names) + 2 + means, ncol, figsize=(3.6 * ncol, 2.5 * (len(names) + 2 + means)), squeeze=False)
    for i, name in enumerate(names):
        for k, ch in enumerate(chans):
            ax = axes[i, k]
            for t in np.linspace(0.2, 1.0, 5) * T:
                s = np.linspace(max(-L, t - L) if L else 0.0, t, 300)
                ax.plot(s, res.evaluate(name, ch, np.full_like(s, t), s), lw=1, label=f"t={t:.2f}")
            if L:
                ax.axvspan(-L, 0.0, color="0.85", alpha=0.6, lw=0)
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9); ax.set_xlabel("shock time s")
            if i == 0 and k == 0:
                ax.legend(fontsize=6, frameon=False)
        for ax in axes[i, len(chans):]:
            ax.axis("off")
    old, new = res.old_flows, res.new_flows
    for k, a in enumerate(agents):
        ax = axes[len(names), k]
        if a in res.loss_path:
            ax.plot(res.times, res.loss_path[a], lw=1, label="E[loss(t)]")
        if a in old:
            ax.axhline(old[a], color="C1", lw=0.8, ls="--", label="old flow")
        if a in new:
            ax.axhline(new[a], color="C2", lw=0.8, ls=":", label="new flow")
        ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{a}: expected loss", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
        ax = axes[len(names) + 1, k]
        for name in states:
            ax.plot(res.times, res.belief_error(a, name), lw=1, label=name)
        ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{a}: belief error variance", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
    for r in (len(names), len(names) + 1):
        for ax in axes[r, len(agents):]:
            ax.axis("off")
    if means:
        ax = axes[-1, 0]
        for name in names:
            ax.plot(res.means_t, res.means[name], lw=1, label=name)
        ax.axhline(0, color="k", lw=0.4); ax.set_title("mean paths", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
        for ax in axes[-1, 1:]:
            ax.axis("off")
    fig.suptitle(f"{res.model.name}  (residual {res.residual:.1e}" + (f", settled {res.settled:.1e}" if res.settled is not None else "") + ")", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_by_shock_time(res, curves, path: str) -> None:
    """Finite horizon: one panel per (quantity, channel), the kernel against the shock time s at a few dates t;
    a last row with the mean paths against t when they are nonzero."""
    plt = _pyplot()
    names = res.model.state_names + res.model.control_names; chans = res.channels
    means = res.means_driven and res.means_t is not None
    fig, axes = plt.subplots(len(names) + means, len(chans), figsize=(3.6 * len(chans), 2.5 * (len(names) + means)), squeeze=False)
    for i, name in enumerate(names):
        for k, ch in enumerate(chans):
            ax = axes[i, k]
            for t, s, y in curves(name, ch):
                ax.plot(s, y, lw=1, label=f"t={t:.2f}")
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9); ax.set_xlabel("shock time s")
            if i == 0 and k == 0:
                ax.legend(fontsize=6, frameon=False)
    if means:
        ax = axes[-1, 0]
        for name in names:
            ax.plot(res.means_t, res.means[name], lw=1, label=name)
        ax.axhline(0, color="k", lw=0.4); ax.set_title("mean paths", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
        for ax in axes[-1, 1:]:
            ax.axis("off")
    fig.suptitle(f"{res.model.name}  (residual {res.residual:.1e})", fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150)
