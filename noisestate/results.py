"""Results returned by the engines: one interface, three grids.

    res.converged, res.residual, res.message     outcome of the outer solve
    res.check()                                  raise ConvergenceError unless converged
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

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .accel import ConvergenceError
from .settings import DEFAULT, Settings, tunable
from .spec import Model


class _StabilityBudget(Exception):
    """Raised by stability()'s matvec at its evaluation budget, so ARPACK unwinds to the fallback."""


@dataclass
class BaseResult:
    model: Model
    compiled: object
    maps: Dict[str, np.ndarray]
    Z: np.ndarray
    converged: bool
    residual: float
    iterations: int
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

    # ----------------------------------------------------------- common
    def check(self):
        """Return self, or raise ConvergenceError if the solve did not reach its tolerance."""
        if not self.converged:
            raise ConvergenceError(f"{self.model.name}: residual {self.residual:.2e} ({self.message})")
        return self

    @property
    def channels(self) -> List[str]:
        return list(self.compiled.channels)

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
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
        d = self.model.to_dict(); hz = d.setdefault("horizon", {})
        n0 = int(hz.get("nodes", 16))
        n1 = 2 * n0 if self.kind == "finite_cells" else max(n0 + 2, int(math.ceil(n0 * factor)))   # cells: keep lags aligned
        hz["nodes"] = n1
        # this solve's bounds and a skipped diagnostics pass are not the refinement's
        kw = {k: v for k, v in self.solve_kw.items() if k not in ("init", "start", "max_evaluations", "deadline", "diagnostics")}
        kw.update(solve_kw)
        solver = self._make_solver(Model.from_dict(d))
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
        out = {"version": __version__, "name": m.name, "engine": self.kind, "converged": bool(self.converged),
               "residual": float(self.residual), "evaluations": int(self.iterations), "seconds": float(self.seconds),
               "message": self.message, "params": {k: float(v) for k, v in m.params.items()}, "model": m.to_dict(),
               "horizon": {k: v for k, v in asdict(m.horizon).items() if v is not None},
               "options": {"solver": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in self.solver_kw.items()},
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


@dataclass
class StationaryResult(BaseResult):
    kind: str = "stationary"
    MAP_CONVENTION = ("maps[agent][u][row][n] is the weight the control puts on the increment of the row as the agent "
                      "sees it (delayed) at age ages[n] before the control; that increment entered the raw row at age "
                      "agents[agent].signals[row].map_age[n] = ages[n] + delay, and the map is zero where map_age is "
                      "beyond the window")

    @property
    def ages(self) -> np.ndarray:
        return self.compiled.grid.nodes

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

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """Closed-loop kernel of a quantity at the shock ages: (N, nW), or (N,) for one channel."""
        c = self.compiled
        K = self.Z[c.block(name)] if name in c.index else c.expr_op(c.model.expand({name: 1.0})) @ self.Z
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
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
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
class TriangleResult(BaseResult):
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

    def map_axes(self, delay: float) -> dict:
        g = self.grid; c = self.compiled
        if self.past is not None:
            delay = 0.0                                       # raw-age storage (MAP_CONVENTION)
        out = {"map_time": (g.t + delay).tolist(), "map_age": (g.a + delay).tolist()}
        if getattr(c, "n_init", 0):
            out["map_init_time"] = (c.tm + delay).tolist()
        return out

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """Closed-loop kernel at the triangle nodes (res.grid.t, res.grid.s): (N, nW) or (N,); the band's
        nodes (s < 0) are the old shocks.  channel may also name an initial shock of the past (its kernel
        is meaningful on the nodes with s = 0)."""
        c = self.compiled
        K = self.Z[c.block(name)] if name in c.index else c.expr_op(c.model.expand({name: 1.0})) @ self.Z
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
        I = fine.grid.interp(g.t, g.a, side_t=g.side_t, side_a=g.side_a, side_d=g.side_d)
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
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.grid.interp(t, t) @ (self.compiled.mean_embed @ np.asarray(self.means[name], dtype=float))

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
            row("settled", float(self.settled), self.SETTLED_TOL, bool(self.settled <= self.SETTLED_TOL),
                f"TRANSITION NOT SETTLED by T - L: raise horizon.window (a map on [T - L, T] is {self.settled:.1e} of its peak "
                f"from the stationary map the buffer is frozen at, against settled_tol {self.SETTLED_TOL:g}: the closed-loop "
                "decay over a unit of t, not the grid's floor)", "raise horizon.window")
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
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
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
class CellResult(BaseResult):
    kind: str = "finite_cells"
    MAP_CONVENTION = ("maps[agent][u][row][i][v] is the weight the control in cell i (time grid.t[i]) puts on the increment "
                      "of the row as the agent sees it in cell v; that increment entered the raw row in cell v - delay / h "
                      "(time agents[agent].signals[row].map_shock_time[v] = grid.t[v] - delay), and cells v below the delay "
                      "are zero")

    @property
    def times(self) -> np.ndarray:
        return self.compiled.times

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

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """K[i, j]: response of `name` at cell i to a unit increment of `channel` in cell j (channel required)."""
        c = self.compiled
        if channel is None:
            raise ValueError("the cell engine stores kernels per channel: kernel(name, channel)")
        k = c.channels.index(channel)
        K = c.expr_kernel(self.Z, c.model.expand({name: 1.0}))
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
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
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
