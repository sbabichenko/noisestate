# Architecture

The package solves linear-quadratic-Gaussian games with private information for noise-state linear
strategies: every process is a kernel on the shocks, a strategy a map from an agent's observed increments to
its control, and an equilibrium a fixed point of the best-response map.  Two engines share one base; this
page says which module holds what, and follows one best response and one transition through them.

## Modules

| module | lines | holds |
|---|---|---|
| `spec.py` | 1356 | the model as data: shocks, states, definitions, agents with signal rows and losses, the horizon (the economics) and `Model.numerics`; `Model.from_dict` (either form), `to_dict`, `to_equations`, `save`, the `with_*` transforms, validation; `load` and `as_model`, the one way in from a path, dict or Model |
| `equations.py` | 372 | the equations form of a file: the evaluator (`to_grammar`, and `signal_block` for `with_signal`), the writer (`from_grammar`), which is also what `describe()` prints |
| `expr.py` | 1176 | the Python form: `Param`, `State` (`X.d = ... * dt + ...`), `Control`, `define`, `Signal`, `Agent`, `shocks`, `dt`, `Game`; `compile_model` to the grammar |
| `description.py` | 234 | `Model.describe()`, text and HTML, from the equations writer |
| `numerics.py` | 121 | `Numerics`: engine, nodes, unit, unit_range, breakpoints, continuation_nodes, tol, damping, max_newton, variable, settings |
| `_settings.py` | 100 | `Settings`, the tuning constants; `tunable` binds a class attribute to one of them |
| `algebra.py` | 153 | `KernelAlgebra`, the interface of a compiled model the engines call |
| `compile.py` | 145 | `CompiledBase`, what every engine reads off a model: primaries, state inputs, rows, losses (atoms, Q, q), ties |
| `grid.py`, `grid_cache.py` | 511, 96 | the stationary engine's piecewise-Chebyshev age grid; grids shared between compiles |
| `triangle.py` | 804 | the spectral engine's triangle in (time, age), its strip and buffer for a transition, the interpolation, and `LinePath`, the quadrature of a family of line integrals |
| `past.py` | 245 | `Past`: what the time before zero leaves behind (a stationary result's kernels, or initial shocks) |
| `accel.py` | 178 | the Anderson fixed point and the Newton-Krylov polish |
| `engine.py` | 597 | `EngineBase`, what every engine shares: tie packing, the passive world and rows, the fixed point (`solve`), `_result` and `_finish`, the diagnostics loop, the second-order Lanczos, the hooks |
| `means.py` | 167 | `MeanLayer`, a base of `EngineBase`: the mean system from the engines' mean hooks, its solve and the mean cost |
| `stationary.py` | 1094 | the stationary engine: `Compiled` (the kernel algebra on the age grid, the closed loop and its cyclic-symmetric reduction) and `StationarySolver` with its best response (the FOC system on the passive rows, the projection, the decomposition, the second-order form) |
| `spectral_compiled.py`, `closed_loop.py` | 937, 299 | the spectral engine's compiled model: breakpoints, grid, the past's and the buffer's wiring, reads and shifts, line paths; the closed loop (I - M) Z = B one time panel at a time, each panel by block elimination |
| `spectral_operators.py`, `finite_free.py` | 484, 458 | the spectral best response as operators (`RowOps`, `ProjOps`, `RespOps`, `FocOps`) and its solve (`FocSystem`: factored, or GMRES), decomposition and second-order form |
| `spectral_means.py`, `finite_spectral.py` | 265, 659 | the means on the time line; `SpectralFiniteSolver`, the engine (a past, a continuation, the projection, the costs, the transition's paths) |
| `results.py` | 1285 | `Result` and its engine subclasses; `kernel`, `response`, `estimate`, `strategy`, `check`, `refine`, `stability`, `to_dict`, the summaries |
| `kernel.py`, `diagnostics.py` | 132, 348 | `Kernel` (a kernel with its axes and interpolant); the checks and the `Assessment` |
| `engines.py` | 74 | `stationary`, `spectral` and `solver`: the power user's namespace, and where `solve()` builds the engine |
| `sweep.py`, `comparison.py`, `transition.py` | 153, 202, 373 | parameter sweeps with warm starts; scenario comparisons; transitions and the march in T |
| `symmetry.py` | 154 | the cyclic symmetry of a tied model |
| `schema.py` | 323 | JSON Schema for the model file and the payload, and `validate` |
| `plotting.py`, `cli.py`, `__init__.py` | 319, 337, 184 | the plots; the command line; `solve` and the exports |

The engines never import each other: `finite_spectral` reaches the stationary engine through `noisestate.solve`
for a `continuation="stationary"` only.  The first-order uniform-cell engine that used to be the third
engine lives outside the package, in `extras/cells.py`, as a cross-check.

## One best response (spectral finite engine)

`EngineBase.solve` iterates on the action kernels (or the raw maps) of the tie representatives; each
evaluation calls `SpectralFiniteSolver.best_response`, which is `finite_free.best_response`:

1. **Compiled model** (`spectral_compiled.py`).  Everything map-independent was built at compile time: the
   grid, the sparse reads and shifts, the line paths (`_path`, cached on the grid), the state columns.
2. **Closed loop** (`closed_loop.py`).  `c.closed_loop(maps, excluded=agent, impulse_controls=agent.controls)`
   assembles the rows of (I - M) Z = B time panel by time panel from the Volterra rows times the state inputs'
   sparse atom operators and the convolution rows of every other agent's map times its seen rows' blocks, with
   the shocks, the initial shocks and one unit impulse per control of the agent as the forcing, and solves it
   by forward substitution over the panels (each panel by block elimination: the primaries with no coupling
   inside the panel are eliminated, the rest solve a Schur complement): Z_pass (the passive world, the agent's strategy off) and R (the impulse responses).
   With a continuation a second solve with the agent's frozen buffer reaction off gives the envelope responses.
3. **Passive rows** (`engine.py`, `finite_spectral._seen_rows`).  The agent's signal rows in Z_pass through the
   sparse row blocks, their instantaneous entries, and with a band their pre-zero part.
4. **Operators** (`spectral_operators.py`).  On those rows: `RowOps` (map -> action kernels), `ProjOps` (FOC
   kernel -> its projection on the rows), `RespOps` (action kernel -> world, from R), `FocOps` (world -> FOC
   kernels, from the envelope responses); `_foc_affine` adds the FOC's pre-zero part on the band.
5. **FOC solve** (`finite_free.FocSystem`).  Amat gamma = -bvec on the kept unknowns (`_identified`, the corner
   ties of `_corner_index`): the operators' dense rows assembled and LU-factored within `foc_dense_max`, GMRES
   on the matvec with the time-row preconditioner beyond.  The action kernels are `RowOps.apply(gamma)`, the
   world of the response Z_full = Z_pass + sum_u Resp_u c_u.
6. **Projection** (`finite_spectral.maps_from_world`).  The raw maps reproducing the action kernels on the
   closed-loop rows of Z_full: one weighted least-squares system per time row (`_time_row_system`, the rows of
   G_k read one panel at a time by `PanelRows`), the systems of one size solved in one LAPACK call.
7. **Checks** (with `want_decomp`, at the equilibrium: `_diagnostics`).  The FOC decomposition (foc, physical,
   wedge), the second-order form on the operators (`_second_order`, dense within `second_order_dense`, Lanczos
   beyond) and the representation error (`reconstruction` against the actions, located by region).

The means (`spectral_means.py`, the hooks of `EngineBase`'s mean layer) are solved once at the end, in
`_mean_part`: one linear system on the time nodes from every control's mean first-order condition
(`FocOps.on_qzeta` on the embedded paths) and the mean dynamics.

## One transition

`SpectralFiniteSolver(model, past=..., continuation=...)`, or `transition(old, new, T)`:

1. **Past** (`past.py`).  The old regime's kernels on its age grid, its row noise loadings and means, or a
   list of initial shocks; validated against the model at the panel unit.
2. **Band** (`spectral_compiled.py`).  The grid becomes the strip [0, T] x [0, L]: the pieces above the diagonal
   carry the shocks born before zero.  Their state at time zero is the past's kernel (`past_at`, in the state
   columns), their pre-zero inputs and observations are the past's (`past_read`, `row_past`, `zeta_past`), the
   increments observed before zero enter the row operator through `past_conv_path` and the projection through
   `past_proj_path` and `old_shock_proj_path`; the initial shocks are extra columns of the world with discrete
   weights on the time nodes (`disc_embed`, `disc_select`).  The maps keep the same shapes on the strip.
3. **Buffer** (`_wire_buffer`, `_frozen_maps`).  With a continuation the grid runs to T + L and every map is frozen
   at the continuation's stationary map on the buffer's nodes; the unknowns stay on [0, T] (`_identified`),
   the closed loop and the first-order conditions run to T + L, and `settled` measures the maps on [T - L, T]
   against the frozen ones.
4. **Continuation** (`_continuation_of`).  `"stationary"` solves this model's stationary equilibrium at the
   past's window through `noisestate.solve`; a `StationaryResult` is checked for the same channels, states,
   controls, rows and window.  `continuation_cost` reports the buffer's cost apart; `loss_path`, `excess_costs`
   and `belief_error` are the transition's paths on the time nodes.

## Where the engines share the base

`EngineBase` (engine.py) holds what does not depend on the grid: the packing of the tie representatives, the
passive world and rows, the fixed point and its polish, `_result` and `_finish`, the diagnostics loop
(`_fill_diagnostics`: one best response per agent at the equilibrium, representatives first), the second-order
check's Lanczos, the singular-system message and the hooks table.  The best response is each engine's own: the
stationary engine's on its compiled model's kernel algebra (`StationarySolver.best_response` and the pieces
under it, `_foc_system`, `_projection_operator`, `_decompose`, `_second_order`), the spectral engine's on its
operators (`finite_free.best_response`).  The mean layer (`mean_system`, `solve_means`, `mean_cost`,
`_mean_part`; `means.MeanLayer`, a base of `EngineBase`) is the base's as well, over mean hooks the engines fill
in: the time nodes of the mean paths, the mean state at time zero, whether anything drives the means, the mean
dynamics operator, the mean first-order-condition operator, the loss atoms' mean paths and the discounted
quadrature weights of the mean cost.  The base assembles the joint (xbar, ubar) system, solves it with the
rcond guard and fills `res.means`, `res.mean_times` and `res.cost_parts`.
