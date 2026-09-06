"""The tuning constants of the solver in one place.

`Settings` holds every numerical threshold and budget the engines and the results read, with its
default and a one-line meaning; `DEFAULT` is the instance with the defaults.  An engine takes
`settings=` (a Settings, or a dict of the fields to change) in its constructor, as does
`noisestate.solve()`, and records the fields that differ from the defaults in `res.solver_kw`, so
a result rebuilds an engine with the same settings for refine() and stability(), and the JSON
payload carries them under options.solver.  The older class-attribute names (`EngineBase.FOC_RCOND`,
`BaseResult.STABILITY_MAX_EVALUATIONS`, `SpectralFiniteSolver.MAP_RIDGE`, ...) remain as aliases
(`tunable`) that read the same field of the instance's settings, and the code reads those names
through the alias, so assigning a class attribute (a monkeypatch in the tests) still takes effect.

The per-engine defaults of solve()'s own arguments (`TOL`, `DAMPING`, `MAX_NEWTON`) stay on the
engines: they are public arguments of solve(), recorded in res.solve_kw, and differ by engine.
`SECOND_ORDER_QUADRATIC` and `ACTIONS` are properties of an engine, not tunables.
"""
from __future__ import annotations

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
    # ---- the best response
    foc_rcond: float = 1e-10            # a best-response system whose reciprocal condition estimate is below this is singular
    stationary_map_ridge: float = 1e-14  # ridge of the stationary map projection's Gram, relative to its mean diagonal
    map_ridge: float = 1e-13            # ridge of the finite engines' per-time-row (per-cell) map projection, relative to the row's own Gram
    cell_dense_max: int = 200           # cell engine: unknowns up to which the best-response system is assembled densely; LGMRES above
    cell_krylov_rtol: float = 1e-12     # cell engine: relative tolerance of the LGMRES best-response solve
    foc_dense_max: int = 8192           # spectral finite engine: unknowns nU nR N (the largest agent's) up to which the first-order-condition system is assembled (the operators applied to the identity) and LU-factored; above it it is solved by GMRES on the operators, preconditioned by time row (finite_free.FocSystem)
    foc_krylov_tol: float = 1e-12       # spectral finite engine, matrix-free path: relative tolerance of the LGMRES solve of the first-order conditions
    foc_krylov_maxiter: int = 400       # spectral finite engine, matrix-free path: LGMRES iterations at most (beyond them the system is reported singular)
    cell_krylov_maxiter: int = 400      # cell engine: LGMRES iterations of the first attempt
    cell_krylov_retry: int = 1000       # cell engine: LGMRES iterations of the second attempt, warm-started from the first
    # ---- the second-order check
    second_order_tol: float = 1e-4      # curvature (relative to the largest) below which a negative value is window truncation
    second_order_dense: int = 4000      # strategy dimension up to which the form is built densely (always settles, 2.7 s at 1600); Lanczos above
    second_order_lanczos_tol: float = 1e-6   # tolerance of the Lanczos extreme eigenvalues above that dimension
    second_order_lanczos_maxiter: int = 300  # Lanczos iterations per extreme eigenvalue
    # ---- the means
    mean_rcond: float = 1e-12           # a mean system whose reciprocal condition estimate is below this is singular
    lead_weight_warn: float = 100.0     # warn when a lead's past flows outweigh the current one by more than this (exp(rho tau))
    # ---- the result's checks
    resolution_tol: float = 1e-6        # representation error above which a result is under-resolved (raise horizon.nodes)
    window_tail_tol: float = 0.02       # a kernel still moving by more of its peak over the last tenth of the window: window too short
    settled_tol: float = 1e-4           # a transition is settled when its maps on [T - L, T] are within this (relative to the map's peak) of the stationary continuation: the closed-loop decay per unit of t (1e-2 on Chapter 3), not the grid's floor
    mean_zero: float = 1e-12            # below this a mean is round-off (printed as an unsigned zero, not counted as driven)
    refine_cost_tol: float = 1e-6       # refine(): relative cost change below which the grid is resolved
    refine_kernel_tol: float = 1e-5     # refine(): relative kernel change below which the grid is resolved
    stability_k: int = 2                # stability(): eigenvalues of largest modulus asked of Arnoldi
    stability_eps: float = 1e-6         # stability(): finite-difference step of the best-response Jacobian, relative to the strategy
    stability_tol: float = 1e-3         # stability(): ARPACK tolerance
    stability_max_evaluations: int = 200    # stability(): rounds of best responses at most, Arnoldi and power iteration together
    stability_fallback: int = 30        # stability(): of those, the rounds kept for the power iteration when Arnoldi does not settle

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
        """The fields that differ from the defaults, as a dict (what solver_kw records)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(DEFAULT, f.name)}


DEFAULT = Settings()


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
