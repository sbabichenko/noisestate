"""Results returned by the engines: one interface, three grids.

    res.converged, res.residual, res.message     outcome of the outer solve
    res.check()                                  raise ConvergenceError unless converged
    res.costs[agent]                             expected discounted (or average) loss
    res.kernel(name, channel=None)               closed-loop kernel of a state or control
    res.maps[agent]                              raw strategies on the agent's signal rows
    res.to_dict()                                JSON-ready payload (grid, kernels, maps, costs, FOC parts)
    res.summary()                                one paragraph

Kernel layout by engine:
  StationaryResult    kernel(name) -> (N, nW) values at the shock ages res.ages; kernel(name, ch) -> (N,)
  TriangleResult      kernel(name) -> (N, nW) at the triangle nodes (res.grid.t, res.grid.s);
                      res.evaluate(name, ch, t, s) interpolates; kernel(name, ch) -> (N,)
  CellResult          kernel(name, ch) -> (N, N) matrix K[i, j]: response at cell i to a unit
                      increment of the channel in cell j (channel required)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .accel import ConvergenceError
from .spec import Model


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
    history: List[float] = field(default_factory=list)
    message: str = ""
    representation_error: Dict[str, float] = field(default_factory=dict)   # agent -> relative residual, see resolution_ok
    refinement: Optional[dict] = None          # filled by refine(): change of costs/kernels under a finer grid
    kind: str = "base"

    RESOLUTION_TOL = 1e-6
    WINDOW_TAIL_TOL = 0.02

    @property
    def resolution_ok(self) -> bool:
        """False when the grid is too coarse for the equilibrium it reports (representation error of a
        best response on the raw rows above RESOLUTION_TOL); raise horizon.nodes and re-solve."""
        return all(v <= self.RESOLUTION_TOL for v in self.representation_error.values())

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
        from .sweep import make_solver
        d = self.model.to_dict(); hz = d.setdefault("horizon", {})
        n0 = int(hz.get("nodes", 16)); n1 = max(n0 + 2, int(math.ceil(n0 * factor)))
        hz["nodes"] = n1
        fine = make_solver(Model.from_dict(d)).solve(**solve_kw)
        cost_change = max(abs(fine.costs[k] - self.costs[k]) / max(1e-12, abs(self.costs[k])) for k in self.costs)
        kernel_change = self._kernel_change(fine)
        rep = {"nodes": n1, "converged": bool(fine.converged), "cost_change": float(cost_change),
               "kernel_change": float(kernel_change), "resolved": bool(fine.converged and cost_change < 1e-6 and kernel_change < 1e-5)}
        self.refinement = rep
        return rep

    def _kernel_change(self, fine) -> float:
        raise NotImplementedError

    def _status(self) -> str:
        s = "converged" if self.converged else f"NOT converged ({self.message})"
        if self.representation_error and not self.resolution_ok:
            s += f"; UNDER-RESOLVED (representation error {max(self.representation_error.values()):.1e}: raise horizon.nodes)"
        if self.refinement:
            s += (f"; refinement to {self.refinement['nodes']} nodes moves costs by {self.refinement['cost_change']:.1e} and kernels by "
                  f"{self.refinement['kernel_change']:.1e}" + ("" if self.refinement["resolved"] else " (NOT RESOLVED)"))
        tail = getattr(self, "window_tail", None)
        if tail is not None and tail > self.WINDOW_TAIL_TOL:
            s += f"; WINDOW TOO SHORT (a kernel still has {tail:.1%} of its peak at the window edge: raise horizon.window)"
        rep = getattr(self, "stability_report", None)
        if rep:
            s += f"; best-response dynamics {'stable' if rep['stable'] else 'UNSTABLE'} (spectral radius {rep['radius']:.3f}"
            s += ", untied game)" if rep["untied"] else ")"
        return s

    def stability(self, untied: bool = True, k: int = 2, eps: float = 1e-6, tol: float = 1e-3,
                  max_evaluations: int = 200) -> dict:
        """Stability of this equilibrium under best-response dynamics: the eigenvalues of largest
        modulus of the Jacobian of the best-response map at the equilibrium, by Arnoldi iteration on
        finite differences.  A spectral radius below one means small deviations by any agent die out
        under naive best-response adjustment (tatonnement-stable); above one, they grow, and the
        equilibrium is one that adjustment dynamics would not find.  With untied=True (default) a
        tied model is assessed on the untied game, so asymmetric deviations are allowed.
        Returns {"radius", "eigenvalues", "stable", "evaluations", "untied"}; also stored in
        self.stability_report."""
        import numpy as np
        from scipy.sparse.linalg import LinearOperator, eigs
        from .sweep import make_solver
        model = self.model
        if untied and model.ties:
            d = model.to_dict(); d["ties"] = []
            model = Model.from_dict(d)
        S = make_solver(model)
        maps = {a.name: np.array(self.maps[a.name], copy=True) for a in model.agents}
        z0 = S.pack(maps)
        F0 = S.pack(S.response_map(S.unpack(z0)))
        scale = max(1.0, float(np.linalg.norm(z0)))
        count = [1]

        def matvec(v):
            v = np.asarray(v, dtype=float).ravel()
            nv = np.linalg.norm(v)
            if nv == 0:
                return np.zeros_like(v)
            h = eps * scale / nv
            count[0] += 1
            return (S.pack(S.response_map(S.unpack(z0 + h * v))) - F0) / h
        n = z0.size
        kk = max(1, min(k, n - 2))
        op = LinearOperator((n, n), matvec=matvec, dtype=float)
        rng = np.random.default_rng(0)
        try:
            vals = eigs(op, k=kk, which="LM", tol=tol, maxiter=max_evaluations, v0=rng.standard_normal(n),
                        return_eigenvectors=False)
        except Exception:                                   # ARPACK did not settle: fall back to power iteration
            v = rng.standard_normal(n); lam = 0.0
            for _ in range(30):
                w = matvec(v); lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / max(np.linalg.norm(w), 1e-300)
            vals = np.array([lam])
        radius = float(np.max(np.abs(vals)))
        rep = {"radius": radius, "eigenvalues": [complex(x) for x in np.asarray(vals)], "stable": bool(radius < 1.0),
               "evaluations": int(count[0]), "untied": bool(untied and self.model.ties),
               "fixed_point_residual": float(np.linalg.norm(F0 - z0) / scale)}
        self.stability_report = rep
        return rep

    def to_dict(self) -> dict:
        """JSON-serialisable view: grid, kernels per quantity and channel, raw maps, costs, and the
        first-order-condition decomposition where the engine provides it."""
        c = self.compiled
        out = {"model": self.model.name, "engine": self.kind, "converged": bool(self.converged),
               "residual": float(self.residual), "evaluations": int(self.iterations), "seconds": float(self.seconds),
               "message": self.message, "grid": self.grid_info(), "discount": float(c.rho),
               "channels": self.channels, "kernels": {}, "maps": {}, "foc": {},
               "costs": {k: float(v) for k, v in self.costs.items()}}
        out["representation_error"] = {k: float(v) for k, v in self.representation_error.items()}
        out["resolution_ok"] = bool(self.resolution_ok)
        out["cost_kind"] = self.cost_kind
        out["notes"] = self.model.notes
        if self.refinement:
            out["refinement"] = self.refinement
        tail = getattr(self, "window_tail", None)
        if tail is not None:
            out["window_tail"] = float(tail)
        rep = getattr(self, "stability_report", None)
        if rep:
            out["stability"] = {"radius": rep["radius"], "stable": rep["stable"], "untied": rep["untied"],
                                "eigenvalues": [[x.real, x.imag] for x in rep["eigenvalues"]]}
        for name in c.prim:
            out["kernels"][name] = {ch: self.kernel(name, ch).tolist() for ch in self.channels}
        for a in self.model.agents:
            g = self.maps[a.name]
            out["maps"][a.name] = {u: {r.name: g[ui, ri].tolist() for ri, r in enumerate(a.signals)}
                                   for ui, u in enumerate(a.controls)}
        foc = getattr(self, "foc", None)
        if foc:
            for aname, dec in foc.items():
                out["foc"][aname] = {u: {part: {ch: arr[:, k].tolist() for k, ch in enumerate(self.channels)}
                                         for part, arr in parts.items()} for u, parts in dec.items()}
        return out


@dataclass
class StationaryResult(BaseResult):
    foc: Dict[str, dict] = field(default_factory=dict)      # agent -> control -> {"foc","physical","wedge"} (N, nW)
    kind: str = "stationary"

    @property
    def ages(self) -> np.ndarray:
        return self.compiled.grid.nodes

    @property
    def window_tail(self) -> float:
        """Largest kernel value at the window edge L relative to that kernel's peak, over states and
        controls: above WINDOW_TAIL_TOL the processes have not decayed within the window and the model
        solved is the truncated one."""
        worst = 0.0
        for name in self.compiled.prim:
            K = self.kernel(name)
            peak = np.abs(K).max()
            if peak > 0:
                worst = max(worst, float(np.abs(K[-1]).max() / peak))
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
        K = c.expr_op(c.model.expand({name: 1.0})) @ self.Z
        return K if channel is None else K[:, self.channels.index(channel)]

    def grid_info(self) -> dict:
        g = self.compiled.grid
        return {"kind": "stationary", "breakpoints": [float(b) for b in g.breakpoints], "nodes_per_panel": g.n,
                "ages": g.nodes.tolist()}

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
                 f"{self.seconds:.1f}s; grid {c.grid.P} panels x {c.grid.n} nodes on [0, {c.grid.L}], rho={c.rho}"]
        for a in self.model.agents:
            lines.append(f"  {a.name}: flow loss = {self.costs.get(a.name, float('nan')):+.6f}")
            for u in a.controls:
                k = self.kernel(u)
                lines.append(f"    {u}(0+) on channels: " + ", ".join(f"{ch}={k[0, j]:+.4f}" for j, ch in enumerate(self.channels)))
        return "\n".join(lines)


@dataclass
class TriangleResult(BaseResult):
    kind: str = "finite_triangle"

    @property
    def grid(self):
        return self.compiled.g

    def kernel(self, name: str, channel: Optional[str] = None) -> np.ndarray:
        """Closed-loop kernel at the triangle nodes (res.grid.t, res.grid.s): (N, nW) or (N,)."""
        c = self.compiled
        K = c.expr_op(c.model.expand({name: 1.0})) @ self.Z
        return K if channel is None else K[:, self.channels.index(channel)]

    def _kernel_change(self, fine) -> float:
        g = self.grid; worst = 0.0
        for name in self.compiled.prim:
            for ch in self.channels:
                K0 = self.kernel(name, ch); K1 = fine.evaluate(name, ch, g.t, g.s)
                worst = max(worst, float(np.abs(K0 - K1).max() / max(1e-12, np.abs(K0).max())))
        return worst

    def evaluate(self, name: str, channel: str, t, s) -> np.ndarray:
        """Kernel value at (t, s) points: response at time t to a unit shock of `channel` at time s."""
        t = np.asarray(t, dtype=float); s = np.asarray(s, dtype=float)
        return self.grid.interp(t, t - s) @ self.kernel(name, channel)

    def grid_info(self) -> dict:
        g = self.grid
        return {"kind": "finite_triangle", "breakpoints": [float(b) for b in g.bp], "nodes_per_side": g.nt,
                "t": g.t.tolist(), "age": g.a.tolist(), "s": g.s.tolist()}

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
                 f"{self.seconds:.1f}s; triangle grid {c.g.P} panels, {len(c.g.pieces)} pieces x {c.g.nt}x{c.g.na} nodes "
                 f"= {c.N} nodes on [0, {c.T}], rho={c.rho}"]
        for a in self.model.agents:
            lines.append(f"  {a.name}: discounted cost = {self.costs.get(a.name, float('nan')):+.8f}")
        return "\n".join(lines)


@dataclass
class CellResult(BaseResult):
    kind: str = "finite_cells"

    @property
    def times(self) -> np.ndarray:
        return self.compiled.times

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

    def summary(self) -> str:
        c = self.compiled
        lines = [f"{self.model.name}: {self._status()} residual {self.residual:.2e} in {self.iterations} evaluations, "
                 f"{self.seconds:.1f}s; {c.N} cells on [0, {c.T}], rho={c.rho}"]
        for a in self.model.agents:
            lines.append(f"  {a.name}: discounted cost = {self.costs.get(a.name, float('nan')):+.6f}")
        return "\n".join(lines)
