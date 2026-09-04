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
    kind: str = "base"

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

    def _status(self) -> str:
        return "converged" if self.converged else f"NOT converged ({self.message})"

    def to_dict(self) -> dict:
        """JSON-serialisable view: grid, kernels per quantity and channel, raw maps, costs, and the
        first-order-condition decomposition where the engine provides it."""
        c = self.compiled
        out = {"model": self.model.name, "engine": self.kind, "converged": bool(self.converged),
               "residual": float(self.residual), "evaluations": int(self.iterations), "seconds": float(self.seconds),
               "message": self.message, "grid": self.grid_info(), "discount": float(c.rho),
               "channels": self.channels, "kernels": {}, "maps": {}, "foc": {},
               "costs": {k: float(v) for k, v in self.costs.items()}}
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
            lines.append(f"  {a.name}: E[loss] = {self.costs.get(a.name, float('nan')):+.6f}")
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
            lines.append(f"  {a.name}: E[cost] = {self.costs.get(a.name, float('nan')):+.8f}")
        return "\n".join(lines)


@dataclass
class CellResult(BaseResult):
    kind: str = "finite_cells"

    @property
    def times(self) -> np.ndarray:
        return self.compiled.times

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
            lines.append(f"  {a.name}: E[cost] = {self.costs.get(a.name, float('nan')):+.6f}")
        return "\n".join(lines)
