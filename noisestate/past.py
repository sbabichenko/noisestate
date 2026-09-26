"""The known past of a transition: what the time before zero leaves behind.

The new problem needs, from the time before zero, the dependence of the physical state and of
each agent's information on the shocks that arrived before zero: for every state and control of
the old regime and every signal row, its kernel as a function of shock age on the past's window
[0, L], per channel, the rows' direct noise loadings E, and the old constant means.  A
StationaryResult already is this (Past.from_result); a hand-built prior is the same object with
point loadings at age zero and an empty window (Past.from_shocks); a stationary Model, dict or
file is solved on the fly (Past.from_model).  The old model's controls, losses, discount and ties
never enter: Past.validate(model) matches channels, states and rows by name and checks that the
past's breakpoints and delays lie on the new model's panel unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spec import Model


@dataclass
class InitialShock:
    """One unit-variance shock at time 0-: `loads` displaces the states, `rows` is seen at once as a point
    observation e xi on the named rows ("agent.row")."""
    name: str
    loads: Dict[str, float] = field(default_factory=dict)
    rows: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "loads": {k: float(v) for k, v in self.loads.items()},
                "rows": {k: float(v) for k, v in self.rows.items()}}


class Past:
    """Loadings of the pre-zero shocks on the old regime's quantities, on the past's own age grid.

    kernels[name]: (N_past, nW) per old state and control; rows["agent.row"]: (kernel (N_past, nW) of the
    raw row's drift, E (nW,), delay); means[name]: the old constant means (primaries and "agent.row"
    drift rates); initial: the point shocks; window L (0 for a past of point shocks only); grid: the
    AgeGrid of the kernels (None when L = 0); provenance: what to_dict() reports."""

    def __init__(self, channels: List[str], window: float, grid, kernels: Dict[str, np.ndarray],
                 rows: Dict[str, Tuple[np.ndarray, np.ndarray, float]], means: Dict[str, float],
                 initial: Optional[List[InitialShock]] = None, provenance: Optional[dict] = None, source=None):
        self.channels = list(channels)
        self.window = float(window)
        self.grid = grid
        self.kernels = kernels
        self.rows = rows
        self.means = dict(means)
        self.initial = list(initial or [])
        self.provenance = dict(provenance or {})
        self.source = source                 # the StationaryResult it came from, if any (not serialised)
        names = [sh.name for sh in self.initial]
        if len(set(names)) != len(names):
            raise ValueError(f"initial shocks must have distinct names, not {names}")
        for sh in self.initial:
            if sh.name in self.channels:
                raise ValueError(f"initial shock {sh.name!r} is named like one of the model's shocks")

    # ------------------------------------------------------------ constructors
    @classmethod
    def of(cls, obj, initial=None) -> "Past":
        """A Past from a Past, a StationaryResult, a stationary Model / dict / path (solved on
        the fly, which must converge) or a list of initial-shock dicts [{"name", "loads", "rows"}]."""
        from .results import StationaryResult, Result
        if isinstance(obj, Past):
            return obj
        if isinstance(obj, StationaryResult):
            return cls.from_result(obj, initial)
        if isinstance(obj, Result):
            raise TypeError(f"past must be a stationary result, not a {obj.kind!r} one ({type(obj).__name__}): the time "
                            "before zero is a stationary regime")
        if isinstance(obj, (list, tuple)):
            return cls.from_shocks(obj)
        if isinstance(obj, (Model, dict, str)):
            return cls.from_model(obj, initial)
        raise TypeError(f"past must be a Past, a StationaryResult, a Model, a dict, a path or a list of initial shocks, "
                        f"not {type(obj).__name__}")

    @classmethod
    def from_block(cls, block: dict) -> "Past":
        """The past of a transition model's horizon.past block: {"model": a path or an inline stationary model
        dict (solved on the fly), "initial": [shocks]} (either, or both)."""
        if block.get("model") is not None:
            return cls.of(block["model"], block.get("initial"))
        return cls.from_shocks(block["initial"])

    @classmethod
    def from_model(cls, model, initial=None, **solve_kw) -> "Past":
        """Solve the stationary model (a Model, dict or path) and take its result."""
        from . import solve
        from .spec import as_model
        model = as_model(model)
        if model.horizon.kind != "stationary":
            raise TypeError(f"past model {model.name!r} has horizon.kind {model.horizon.kind!r}; the time before zero is a "
                            "stationary regime")
        return cls.from_result(solve(model, **solve_kw), initial)

    @classmethod
    def from_result(cls, res, initial=None) -> "Past":
        """The kernels, rows, noise loadings and means of a converged StationaryResult."""
        from .results import StationaryResult
        if not isinstance(res, StationaryResult):
            raise TypeError(f"past must be a stationary result, not {type(res).__name__}")
        if not res.converged:
            raise ValueError(f"the past {res.model.name!r} did not converge (residual {res.residual:.2e}, {res.message}); "
                             "a transition starts from an equilibrium")
        c = res.compiled; m = res.model
        kernels = {name: np.array(res.world[c.block(name)]) for name in c.prim}
        rows = {}
        for a in m.agents:
            for r, (rname, drift, E, delay) in zip(a.signals, c.rows[a.name]):
                rows[f"{a.name}.{rname}"] = (c.expr_op(drift) @ res.world, np.array(E, dtype=float), float(delay))
        means = {k: float(v) for k, v in res.means.items()}
        prov = {"kind": "stationary", "name": m.name, "params": {k: float(v) for k, v in m.params.items()},
                "window": float(c.grid.L), "nodes": int(c.grid.n), "breakpoints": [float(b) for b in c.grid.breakpoints],
                "converged": bool(res.converged), "residual": float(res.residual), "window_tail": float(res.window_tail),
                "costs": {k: float(v) for k, v in res.costs.items()}}
        shocks = [sh if isinstance(sh, InitialShock) else InitialShock(**sh) for sh in (initial or [])]
        return cls(c.channels, c.grid.L, c.grid, kernels, rows, means, shocks, prov, source=res)

    @classmethod
    def from_shocks(cls, shocks, channels=None) -> "Past":
        """Point shocks at time 0- only: an empty window, kernels and rows; the channels are the new model's
        (filled in by validate)."""
        out = []
        for i, sh in enumerate(shocks):
            if isinstance(sh, InitialShock):
                out.append(sh); continue
            if not isinstance(sh, dict):
                raise TypeError(f"an initial shock is a dict {{'name', 'loads', 'rows'}}, not {type(sh).__name__}")
            unknown = set(sh) - {"name", "loads", "rows"}
            if unknown:
                raise ValueError(f"initial shock {i}: unknown field(s) {sorted(unknown)}; the fields are name, loads, rows")
            out.append(InitialShock(str(sh.get("name", f"xi{i}")), {k: float(v) for k, v in (sh.get("loads") or {}).items()},
                                    {k: float(v) for k, v in (sh.get("rows") or {}).items()}))
        prov = {"kind": "initial", "window": 0.0, "initial": [sh.to_dict() for sh in out]}
        return cls(list(channels or []), 0.0, None, {}, {}, {}, out, prov)

    # ------------------------------------------------------------ views
    @property
    def n_initial(self) -> int:
        return len(self.initial)

    @property
    def initial_names(self) -> List[str]:
        return [sh.name for sh in self.initial]

    @property
    def nW(self) -> int:
        return len(self.channels)

    def kernel(self, name: str) -> np.ndarray:
        """(N_past, nW) kernel of an old state or control; zero when the past has no window."""
        if name in self.kernels:
            return self.kernels[name]
        raise KeyError(f"the past has no kernel for {name!r}")

    def read(self, name: str, ages, side: int = +1) -> np.ndarray:
        """The kernel of `name` at the ages (zero at negative ages and beyond the window): (len, nW)."""
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        if self.grid is None:
            return np.zeros((len(ages), self.nW))
        return self.grid.interp(ages, side=side) @ self.kernel(name)

    def row_noise(self, key: str) -> np.ndarray:
        """E (nW,) of the row "agent.row" in the old regime (zero without a window)."""
        return self.rows[key][1] if key in self.rows else np.zeros(self.nW)

    def mean(self, name: str) -> float:
        return float(self.means.get(name, 0.0))

    @property
    def breakpoints(self) -> List[float]:
        return [float(b) for b in self.grid.breakpoints] if self.grid is not None else [0.0]

    def to_dict(self) -> dict:
        """Provenance only (kind, window, nodes, costs, the initial shocks): no kernels."""
        out = dict(self.provenance)
        out.setdefault("kind", "stationary" if self.grid is not None else "initial")
        out["window"] = float(self.window)
        out["shocks"] = list(self.channels)
        out["initial"] = [sh.to_dict() for sh in self.initial]
        out["means"] = {k: float(v) for k, v in self.means.items() if v}
        return out

    # ------------------------------------------------------------ validation
    def validate(self, model: Model, unit: Optional[float] = None) -> None:
        """Check the past fits the new model: channels identical (names and order), every state and every
        "agent.row" of the new model present, every control a lagged drift or loss atom reads across zero
        present, the initial shocks' loads on states and rows on rows of the model; with `unit` (the new
        panel unit) the past grid's breakpoints and the old rows' delays on it."""
        chans = list(model.shocks)
        if self.grid is not None:
            if self.channels != chans:
                raise ValueError(f"the past's shocks {self.channels} differ from the model's {chans}: a transition keeps "
                                 "the same shocks (names and order)")
            for s in model.state_names:
                if s not in self.kernels:
                    raise ValueError(f"the past has no kernel for the state {s!r}; every state of the new model must exist "
                                     "in the past")
            for a in model.agents:
                for r in a.signals:
                    if f"{a.name}.{r.name}" not in self.rows:
                        raise ValueError(f"the past has no row {a.name}.{r.name}; every signal row of the new model must "
                                         "exist in the past (agent and row by name)")
            read_across = set()
            for s in model.states:
                for (n, l) in model.expand(s.drift):
                    if l > 0 and n in model.control_names:
                        read_across.add(n)
            for a in model.agents:
                for term in a.loss:
                    for atom in term[1:]:
                        for (n, l) in model.expand({atom: 1.0}):
                            if l > 0 and n in model.control_names:
                                read_across.add(n)
            for n in sorted(read_across):
                if n not in self.kernels:
                    raise ValueError(f"the past has no kernel for the control {n!r}, which a lagged drift or loss atom reads "
                                     "across time zero")
            if unit:
                for b in self.breakpoints:
                    if b <= self.window + 1e-12 and abs(b / unit - round(b / unit)) > 1e-9:
                        raise ValueError(f"the past grid's breakpoint {b:g} is not a multiple of the panel unit {unit:g}; set "
                                         "numerics.unit to a common divisor of the new model's lags and the past's panels")
                for key, (_, _, d) in self.rows.items():
                    if d > 0 and abs(d / unit - round(d / unit)) > 1e-9:
                        raise ValueError(f"the past row {key}'s delay {d:g} is not a multiple of the panel unit {unit:g}; set "
                                         "numerics.unit to a common divisor of the lags and delays of both regimes")
        elif not self.channels:
            self.channels = chans
        elif self.channels != chans:
            raise ValueError(f"the past's shocks {self.channels} differ from the model's {chans}")
        rows = {f"{a.name}.{r.name}" for a in model.agents for r in a.signals}
        for sh in self.initial:
            for s in sh.loads:
                if s not in model.state_names:
                    raise ValueError(f"initial shock {sh.name!r} loads {s!r}, which is not a state of the model {model.state_names}")
            for r in sh.rows:
                if r not in rows:
                    raise ValueError(f"initial shock {sh.name!r} is seen on {r!r}, which is not a row of the model {sorted(rows)}")
