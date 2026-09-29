# Settings

The tuning constants live in one frozen dataclass, `noisestate.Settings` (`noisestate/settings.py`), each
with its default and a one-line meaning.  They are the `settings` field of `Numerics`: pass
`ns.Numerics(settings=ns.Settings(second_order_tol=1e-3))` (or `{"settings": {...}}`, a dict of the fields to
change) to `solve()` or `sweep()`, or `settings=` to an engine's constructor; the fields that differ from the
defaults are recorded in `res.numerics`, `res.solver_kw` and the payload's `options`, so `refine()`,
`stability()` and a re-solve from the payload keep them.

| field | default | meaning |
|---|---|---|
| `anderson_m` | 15 | Anderson memory: 15 with damping 0.6 takes Ch5 from 58 to 37 evaluations, Kyle-Back from 81 to 47 |
| `anderson_iters` | 150 | Anderson iterations before the Newton-Krylov polish takes over |
| `anderson_reg` | 1e-8 | Tikhonov regularisation of the Anderson secant system, relative to its trace |
| `newton_inner_m` | 15 | fresh Krylov vectors (evaluations of the map) per Newton step of the polish |
| `anderson_m_instant` | 25 | Anderson memory (at least) for a model with instant observations: its best-response map has a ring of complex eigenvalues beyond the unit circle (0.24 +- 1.64i on the transparent market), where 15 secant pairs stalled (310 evaluations with the polish, 101 at 25; Kyle-Back and Ch5 are slower at 25) |
| `best_responses` | auto | the fixed-point map on action kernels: "sequential" (Gauss-Seidel: the agents answer in the model's order, each against the profile the agents before it have answered; the origins of monitored deviations answer together; ties answer simultaneously), "simultaneous" (Jacobi, every agent against the same profile), or "auto": sequential for a model with instant observations (Chapter 6's markets), simultaneous otherwise (monitoring alone gains nothing: Kyle-Back with a privy market maker 59 -> 63 evaluations). The fixed points are the same equilibria; results move within the solve's tolerance. Sequential took Chapter 6's two-trader market from 507 evaluations to 54 from zero (its Jacobian's radius 1.12 -> 0.75); elsewhere it gains less (Ch1 delayed 13 -> 9 evaluations, Ch3's transition 20 -> 15, Kyle-Back 47 -> 50), moves every result within the tolerance and leaves a symmetric pair of untied agents symmetric only to it, so it is not the default there |
| `foc_rcond` | 1e-10 | a best-response system whose reciprocal condition estimate is below this is singular |
| `stationary_map_ridge` | 1e-14 | ridge of the stationary map projection's Gram, relative to its mean diagonal |
| `map_ridge` | 1e-13 | ridge of the finite engines' per-time-row (per-cell) map projection, relative to the row's own Gram |
| `foc_dense_max` | 500 | spectral finite engine: unknowns nU nR N (the largest agent's) up to which the first-order-condition system is assembled (the operators applied to the identity) and LU-factored; above it it is solved by GMRES on the operators, preconditioned by time row (finite_free.FocSystem) |
| `foc_krylov_tol` | 1e-12 | spectral finite engine, matrix-free path: relative tolerance of the LGMRES solve of the first-order conditions |
| `foc_krylov_maxiter` | 400 | spectral finite engine, matrix-free path: LGMRES iterations at most (beyond them the system is reported singular) |
| `risk_basis` | 0 | Legendre functions per time panel (and channel) of the Galerkin part of the entropic correction (risk.py); 0: the grid's nodes per side plus 4 |
| `risk_planning` | precommitment | the spectral finite engine's criterion for risk-averse agents: "precommitment" (J_0 over the whole strategy) or "consistent" (every date's self minimises the entropic cost of its own continuation, the later selves playing the equilibrium map; risk.ConsistentTilt) |
| `second_order_tol` | 1e-4 | curvature (relative to the largest) below which a negative value is window truncation |
| `second_order_dense` | 8000 | strategy dimension up to which the form is built densely (always settles; 2.7 s at 1600, on the finite engine 7 s at 4032 and 45 s at 7920 where Lanczos took 83 s and 409 s); Lanczos above |
| `second_order_lanczos_tol` | 1e-6 | tolerance of the Lanczos extreme eigenvalues above that dimension |
| `second_order_lanczos_maxiter` | 300 | Lanczos iterations per extreme eigenvalue |
| `mean_rcond` | 1e-12 | a mean system whose reciprocal condition estimate is below this is singular |
| `lead_weight_warn` | 100.0 | warn when a lead's past flows outweigh the current one by more than this (exp(rho tau)) |
| `resolution_tol` | 1e-6 | representation error above which a result is under-resolved (raise numerics.nodes) |
| `auto_panels_max` | 4096 | spectral finite engine: the largest automatic time-panel grid (`time_panels.py`), in unknowns nU nR N of the largest agent: the grid placed from the model's time scales when one panel is predicted not to resolve them, and the local refinements of a grid that fails the resolution check; 0: never (one panel, as given; `res.panels["suggested"]` still names the grid, `res.sharpen()` solves there).  4096 holds the regulator at T = 30 on 12 nodes (4032: 18 s and 1.35 GB with the checks) |
| `auto_panels_growth` | 2.0 | a refinement round may take the unknowns to at most this multiple of the first automatic grid's; past it (or past `auto_panels_max` / `auto_panels_memory`) the best result is returned with a warning naming the breakpoints of the resolved answer and their cost (`res.panels["suggested"]`, `res.sharpen()`).  Raise it to spend more |
| `auto_panels_memory` | 1536 | ... and no automatic grid whose estimated peak memory in MB (the solve and its checks, the best earlier grid kept alive: 100 + 8e-5 sum over the answering agents of (nU nR N)^2, measured to 15%) is above this |
| `window_tail_tol` | 0.02 | a kernel still moving by more of its peak over the last tenth of the window: window too short |
| `window_cost_tol` | 1e-6 | publication: a longer window (`res.check_window()`, 1.5 L) moving an agent's cost by more than this, relative to the largest cost: window too short for the costs |
| `settled_tol` | 1e-4 | a transition is settled when its maps on [T - L, T] are within this (relative to the map's peak) of the stationary continuation: the closed-loop decay per unit of t (1e-2 on Chapter 3), not the grid's floor |
| `mean_zero` | 1e-12 | below this a mean is round-off (printed as an unsigned zero, not counted as driven) |
| `refine_cost_tol` | 1e-6 | refine(): relative cost change below which the grid is resolved |
| `refine_kernel_tol` | 1e-5 | refine(): relative kernel change below which the grid is resolved |
| `stability_k` | 2 | stability(): eigenvalues of largest modulus asked of Arnoldi |
| `stability_eps` | 1e-6 | stability(): finite-difference step of the best-response Jacobian, relative to the strategy |
| `stability_tol` | 1e-3 | stability(): Arnoldi stops when the dominant Ritz pairs' relative residual is below 1e-3 of it |
| `stability_max_evaluations` | 200 | stability(): rounds of best responses at most, Arnoldi and power iteration together |
| `stability_fallback` | 30 | stability(): of those, the rounds kept for the power iteration when Arnoldi does not settle |

## Groups

The outer fixed point (`anderson_m`, `anderson_iters`, `anderson_reg`, `newton_inner_m`, `anderson_m_instant`,
`best_responses`); the best response
(`foc_rcond`, `stationary_map_ridge`, `map_ridge`, and the spectral finite engine's `foc_dense_max`, the unknowns
nU nR N up to which the first-order-condition system is assembled from the operators' rows and LU-factored,
beyond which it is solved by GMRES on the applied operators to `foc_krylov_tol` within `foc_krylov_maxiter`,
preconditioned by its time-row blocks); risk-averse agents (`risk_basis`, `risk_planning`); the second-order check (`second_order_tol`, `second_order_dense`,
`second_order_lanczos_tol`, `second_order_lanczos_maxiter`); the means (`mean_rcond`, `lead_weight_warn`);
and the result's checks (`resolution_tol`, `window_tail_tol`, `settled_tol` for a transition's maps and means
on [T - L, T] against the stationary ones, `mean_zero`, `refine_cost_tol`, `refine_kernel_tol`, `stability_k`,
`stability_eps`, `stability_tol`, `stability_max_evaluations`, `stability_fallback`).

The older class-attribute names (`EngineBase.FOC_RCOND`, `BaseResult.STABILITY_MAX_EVALUATIONS`,
`SpectralFiniteSolver.MAP_RIDGE`, ...) remain as aliases of the same fields.  The defaults of the fixed
point's own options (`tol`, `damping`, `max_newton`) stay per engine (`TOL`, `DAMPING`, `MAX_NEWTON`), since
they differ by engine; a `Numerics` overrides them and `res.numerics` reports the resolved values.
