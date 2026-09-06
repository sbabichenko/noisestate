"""Numerics: how a model is solved, apart from what the model is.

    from noisestate import Numerics, solve
    res = solve("examples/ch4_kyle_back.yaml", Numerics(nodes=32))
    res = solve(model, {"engine": "cells", "nodes": 48})
    res = solve(model, Numerics(tol=1e-12, settings={"anderson_m": 10}))

A model file keeps the economics under `horizon:` (kind, window, discount, past, continuation) and may carry
a `numerics:` block with the same fields as this class; `solve(model, numerics)` lays the given fields over
the model's block (a field left None keeps the model's), so a Numerics is a change of resolution, engine or
tolerance and never a model edit.  `res.numerics` is the resolved object a result was solved with, and the
payload carries it under options.numerics.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import List, Optional, Union

from .settings import DEFAULT, Settings

ENGINES_OF_KIND = {"stationary": "stationary", "finite": "spectral", "transition": "spectral"}
ENGINE_NAMES = ("stationary", "spectral", "cells")
GRID_FIELDS = ("nodes", "unit", "unit_range", "breakpoints", "continuation_nodes")
SOLVE_FIELDS = ("tol", "damping", "max_newton", "variable")


@dataclass
class Numerics:
    """The numerical choices of a solve.  Every field is optional: None means the model's own value, and
    beyond that the default named here.

    engine              "stationary" (the age grid), "spectral" (the piecewise-spectral triangle; every finite
                        horizon and every transition) or "cells" (the first-order uniform-cell cross-check on a
                        finite horizon); default from the horizon kind (stationary -> stationary, else spectral)
    nodes               Chebyshev nodes per panel (stationary) or per side of each piece (spectral); cells on
                        the cell engine; default 16
    unit                the panel unit: every lag and delay must be a multiple of it (default the smallest lag)
    unit_range          the age (stationary) or time (spectral) up to which the panels are unit panels; beyond
                        it they grow geometrically (default the whole window)
    breakpoints         an explicit panel sequence from 0 to the window (default from the lags and the unit)
    continuation_nodes  a transition's stationary continuation solved at this many nodes (default `nodes`)
    tol, damping, max_newton, variable
                        the fixed point's tolerance, Anderson mixing weight, Newton polish steps and iterate
                        ("actions" or "maps"); defaults per engine (tol 1e-10 stationary, 1e-8 finite)
    settings            the tuning constants (noisestate.Settings, or a dict of its fields)"""
    engine: Optional[str] = None
    nodes: Optional[int] = None
    unit: Optional[float] = None
    unit_range: Optional[float] = None
    breakpoints: Optional[List[float]] = None
    continuation_nodes: Optional[int] = None
    tol: Optional[float] = None
    damping: Optional[float] = None
    max_newton: Optional[int] = None
    variable: Optional[str] = None
    settings: Settings = DEFAULT

    def __post_init__(self):
        self.settings = Settings.of(self.settings)
        if self.engine is not None and self.engine not in ENGINE_NAMES:
            raise ValueError(f"numerics.engine must be one of {list(ENGINE_NAMES)}, not {self.engine!r}")
        if self.nodes is not None:
            if self.nodes != int(self.nodes) or int(self.nodes) < 2:
                raise ValueError(f"numerics.nodes must be an integer of at least 2, got {self.nodes!r}")
            self.nodes = int(self.nodes)
        if self.continuation_nodes is not None:
            if self.continuation_nodes != int(self.continuation_nodes) or int(self.continuation_nodes) < 2:
                raise ValueError(f"numerics.continuation_nodes must be an integer of at least 2, got {self.continuation_nodes!r}")
            self.continuation_nodes = int(self.continuation_nodes)
        if self.breakpoints is not None:
            self.breakpoints = [float(b) for b in self.breakpoints]
        if self.variable is not None and self.variable not in ("actions", "maps"):
            raise ValueError(f"numerics.variable must be 'actions' or 'maps', not {self.variable!r}")

    def __repr__(self) -> str:
        parts = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.name != "settings" and getattr(self, f.name) is not None]
        changed = self.settings.changed()
        if changed:
            parts.append(f"settings={changed!r}")
        return f"Numerics({', '.join(parts)})"

    @classmethod
    def of(cls, value: Union[None, "Numerics", dict]) -> "Numerics":
        """None is an empty Numerics (the model's own), a Numerics is itself, a dict names fields."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            bad = sorted(set(value) - {f.name for f in fields(cls)})
            if bad:
                raise TypeError(f"unknown numerics field(s) {bad}; the fields are {cls.field_names()}")
            return cls(**value)
        raise TypeError(f"numerics must be a Numerics, a dict of its fields or None, not {type(value).__name__}")

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def merged(self, other: "Numerics") -> "Numerics":
        """This object with `other`'s given fields laid over it (None keeps this one's; settings when other's
        differ from the defaults)."""
        changes = {f.name: getattr(other, f.name) for f in fields(self) if f.name != "settings" and getattr(other, f.name) is not None}
        if other.settings != DEFAULT:
            changes["settings"] = other.settings
        return replace(self, **changes)

    def resolved(self, kind: str) -> "Numerics":
        """The same with the engine and the nodes filled in from their defaults (the horizon kind; 16)."""
        return replace(self, engine=self.engine or ENGINES_OF_KIND[kind], nodes=16 if self.nodes is None else self.nodes)

    def solve_kw(self) -> dict:
        """The fixed point's options for an engine's solve(), the given ones only."""
        return {k: getattr(self, k) for k in SOLVE_FIELDS if getattr(self, k) is not None}

    def to_dict(self) -> dict:
        """The file block: the given fields, settings as the fields that differ from the defaults."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "settings" and getattr(self, f.name) is not None}
        if self.breakpoints is not None:
            out["breakpoints"] = list(self.breakpoints)
        changed = self.settings.changed()
        if changed:
            out["settings"] = changed
        return out
