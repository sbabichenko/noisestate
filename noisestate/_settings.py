"""The tuning constants of the solver in one place.

`Settings` holds every numerical threshold and budget the engines and the results read, with its
default and a one-line meaning; `DEFAULT` is the instance with the defaults.  An engine takes
`settings=` (a Settings, or a dict of the fields to change) in its constructor, as does
`noisestate.solve()`, and records the fields that differ from the defaults in `res.solver_kw`, so
a result rebuilds an engine with the same settings for refine() and stability(), and the JSON
payload carries them under options.solver.  The older class-attribute names (`EngineBase.FOC_RCOND`,
`Result.STABILITY_MAX_EVALUATIONS`, `SpectralFiniteSolver.MAP_RIDGE`, ...) remain as aliases
(`tunable`) that read the same field of the instance's settings, and the code reads those names
through the alias, so assigning a class attribute (a monkeypatch in the tests) still takes effect.

The per-engine defaults of solve()'s own arguments (`TOL`, `DAMPING`, `MAX_NEWTON`) stay on the
engines: they are public arguments of solve(), recorded in res.solve_kw, and differ by engine.
`ACTIONS` is a property of an engine, not a tunable.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, fields, replace
from typing import Optional, Union


@dataclass(frozen=True)
class Settings:
    """Every tunable of the solver; see the module docstring.  Frozen: make a changed copy with
    dataclasses.replace(DEFAULT, field=value), or Settings(field=value)."""
    # ---- the outer fixed point (engine.solve, accel.solve_fixed_point)
    anderson_m: int = 15                # Anderson memory: 15 with damping 0.6 takes Ch5 from 58 to 37 evaluations, Kyle-Back from 81 to 47
    anderson_iters: int = 150           # Anderson iterations before the Newton-Krylov polish takes over
    anderson_reg: float = 1e-8          # Tikhonov regularisation of the Anderson secant system, relative to its trace
    newton_inner_m: int = 15            # fresh Krylov vectors (evaluations of the map) per Newton step of the polish
    anderson_m_instant: int = 25        # Anderson memory (at least) for a model with instant observations (a level seen at once, Chapter 6's quote): its best-response map has a ring of complex eigenvalues beyond the unit circle (0.24 +- 1.64i on the transparent market), and 15 secant pairs stalled there (310 evaluations with the polish, 101 at 25; Kyle-Back and Ch5 are slower at 25)
    best_responses: str = "auto"        # the fixed-point map on action kernels: "sequential" (Gauss-Seidel: the agents answer in the model's order, each against the profile the agents before it have answered; the origins of monitored deviations answer together; ties answer simultaneously), "simultaneous" (Jacobi, every agent against the same profile), or "auto": sequential for a model with instant observations (Chapter 6's markets), simultaneous otherwise (monitoring alone gains nothing: Kyle-Back with a privy market maker 59 -> 63 evaluations).  The fixed points are the same equilibria; results move within the solve's tolerance.  Sequential took Chapter 6's two-trader market from 507 evaluations to 54 from zero (its Jacobian's radius 1.12 -> 0.75); elsewhere it gains less (Ch1 delayed 13 -> 9 evaluations, Ch3's transition 20 -> 15, Kyle-Back 47 -> 50), moves every result within the tolerance and leaves a symmetric pair of untied agents symmetric only to it, so it is not the default there
    # ---- the best response
    foc_rcond: float = 1e-10            # a best-response system whose reciprocal condition estimate is below this is singular
    stationary_map_ridge: float = 1e-14  # ridge of the stationary map projection's Gram, relative to its mean diagonal
    map_ridge: float = 1e-13            # ridge of the finite engines' per-time-row (per-cell) map projection, relative to the row's own Gram
    foc_dense_max: int = 500            # spectral finite engine: unknowns nU nR N (the largest agent's) up to which the first-order-condition system is assembled (the operators applied to the identity) and LU-factored; above it it is solved by GMRES on the operators, preconditioned by time row (finite_free.FocSystem).  Measured 2026-09-06: dense is never faster (0.26 s vs 0.12 s at 482 unknowns, 90 s vs 0.76 s at 4926, four times the memory); 500 keeps the tiny systems on the direct solve
    foc_krylov_tol: float = 1e-12       # spectral finite engine, matrix-free path: relative tolerance of the LGMRES solve of the first-order conditions
    foc_krylov_maxiter: int = 400       # spectral finite engine, matrix-free path: LGMRES iterations at most (beyond them the system is reported singular)
    # ---- risk-averse (CARA) agents, the spectral finite engine
    risk_basis: int = 0                 # Legendre functions per time panel (and channel) of the Galerkin part of the entropic correction (risk.py); 0: the grid's nodes per side plus 4
    risk_planning: str = "precommitment"  # the spectral finite engine's criterion for risk-averse agents: "precommitment" (J_0 over the whole strategy) or "consistent" (every date's self minimises the entropic cost of its own continuation, the later selves playing the equilibrium map; risk.ConsistentTilt)
    # ---- the second-order check
    second_order_tol: float = 1e-4      # curvature (relative to the largest) below which a negative value is window truncation
    second_order_dense: int = 8000      # strategy dimension up to which the form is built densely (always settles; 2.7 s at 1600, on the finite engine 7 s at 4032 and 45 s at 7920 where Lanczos took 83 s and 409 s); Lanczos above
    second_order_lanczos_tol: float = 1e-6   # tolerance of the Lanczos extreme eigenvalues above that dimension
    second_order_lanczos_maxiter: int = 300  # Lanczos iterations per extreme eigenvalue
    # ---- the means
    mean_rcond: float = 1e-12           # a mean system whose reciprocal condition estimate is below this is singular
    lead_weight_warn: float = 100.0     # warn when a lead's past flows outweigh the current one by more than this (exp(rho tau))
    # ---- the result's checks
    resolution_tol: float = 1e-6        # representation error above which a result is under-resolved (raise numerics.nodes)
    auto_panels_max: int = 8000         # spectral finite engine: unknowns nU nR N up to which a one-panel horizon that fails the resolution check (or is singular) is re-solved on panels graded from both ends (time_panels.py); 0: never
    window_tail_tol: float = 0.02       # a kernel still moving by more of its peak over the last tenth of the window: window too short
    window_cost_tol: float = 1e-6       # publication: an agent's flow loss beyond the window (its last tenths' geometric decay extrapolated), relative to the whole, above this: window too short for the costs (the kernel tail at 2% leaves costs up to 5e-4 off); refine_cost_tol's level
    settled_tol: float = 1e-4           # a transition is settled when its maps on [T - L, T] are within this (relative to the map's peak) of the stationary continuation: the closed-loop decay per unit of t (1e-2 on Chapter 3), not the grid's floor
    mean_zero: float = 1e-12            # below this a mean is round-off (printed as an unsigned zero, not counted as driven)
    refine_cost_tol: float = 1e-6       # refine(): relative cost change below which the grid is resolved
    refine_kernel_tol: float = 1e-5     # refine(): relative kernel change below which the grid is resolved
    stability_k: int = 2                # stability(): eigenvalues of largest modulus asked of Arnoldi
    stability_eps: float = 1e-6         # stability(): finite-difference step of the best-response Jacobian, relative to the strategy
    stability_tol: float = 1e-3         # stability(): ARPACK tolerance
    stability_max_evaluations: int = 200    # stability(): rounds of best responses at most, Arnoldi and power iteration together
    stability_fallback: int = 30        # stability(): of those, the rounds kept for the power iteration when Arnoldi does not settle

    def __post_init__(self):
        """The type of every field, checked where it is set: a count given as "15" reached the Anderson loop as a
        string (a TypeError from a comparison deep in the solve), a NaN threshold made every comparison with it False,
        and a misspelt risk_planning passed silently unless an agent was risk averse.  The signs are not checked:
        the tests move thresholds past their meaningful range on purpose (a negative second_order_tol, e.g.)."""
        for f in fields(self):
            v = getattr(self, f.name)
            default = f.default
            if isinstance(f.default, str):
                if not isinstance(v, str):
                    raise TypeError(f"settings.{f.name} must be a string, not {v!r}")
            elif isinstance(f.default, int):
                if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                    raise TypeError(f"settings.{f.name} must be an integer (default {default!r}), not {v!r}")
                if v < 0:
                    raise ValueError(f"settings.{f.name} must be a non-negative integer (default {default!r}), not {v!r}")
            elif isinstance(f.default, float):
                if isinstance(v, bool) or not isinstance(v, numbers.Real) or math.isnan(v):
                    raise TypeError(f"settings.{f.name} must be a number (default {default!r}), not {v!r}")
        if self.best_responses not in ("auto", "sequential", "simultaneous"):
            raise ValueError(f"settings.best_responses must be 'auto', 'sequential' or 'simultaneous', not {self.best_responses!r}")
        if self.risk_planning not in ("precommitment", "consistent"):
            raise ValueError(f"settings.risk_planning must be 'precommitment' or 'consistent', not {self.risk_planning!r}")

    @classmethod
    def of(cls, value: Union[None, "Settings", dict]) -> "Settings":
        """The Settings an engine was given: None is DEFAULT, a Settings is itself, a dict names the
        fields to change from the defaults (unknown fields are an error)."""
        if value is None:
            return DEFAULT
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            bad = sorted(set(value) - {f.name for f in fields(cls)})
            if bad:
                raise TypeError(f"unknown settings {bad}; the fields are {[f.name for f in fields(cls)]}")
            return replace(DEFAULT, **value)
        raise TypeError(f"settings must be a Settings, a dict of its fields or None, not {type(value).__name__}")

    def changed(self) -> dict:
        """The fields that differ from the defaults (the class's own, not a `noisestate.settings()` block's), as a
        dict (what solver_kw records)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(PRISTINE, f.name)}


DEFAULT = Settings()        # what an engine given no settings reads; `noisestate.settings(...)` replaces it for a block
PRISTINE = Settings()       # the class's defaults, the reference of changed()


class tunable:
    """A class-attribute alias of one Settings field: read on an instance it gives
    instance.settings.<field> (DEFAULT's value when the instance has no settings, and on the class
    itself), so the older names keep their meaning and the code keeps reading them.  Assigning the
    class attribute (a monkeypatch) replaces the alias with a plain value, which every instance of
    the class then sees."""

    def __init__(self, field: str):
        self.field = field

    def __get__(self, obj, owner=None):
        s: Optional[Settings] = getattr(obj, "settings", None) if obj is not None else None
        return getattr(DEFAULT if s is None else s, self.field)

    def __repr__(self) -> str:
        return f"tunable({self.field!r})"
