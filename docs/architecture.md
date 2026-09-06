# Architecture

The package solves linear-quadratic-Gaussian games with private information for noise-state linear
strategies: every process is a kernel on the shocks, a strategy a map from an agent's observed increments to
its control, and an equilibrium a fixed point of the best-response map.  Three engines share one base; this
page says which module holds what, and follows one best response and one transition through them.

## Modules

| module | lines | holds |
|---|---|---|
| `spec.py` | 989 | the model as data: states, controls, definitions, agents with signal rows and losses, the horizon block (the economics) and the numerics block; `Model.from_dict`, `expand`, `all_lags`, `with_numerics` |
| `algebra.py` | 147 | `KernelAlgebra`, the interface of a compiled model: the operators the base engine calls (the closed loop, the blocks and atom operators, the seen rows, the best-response pieces, the cost mass) with their shapes, and the attributes every engine has; a member an engine lacks raises `NotImplementedError` naming it |
| `compile.py` | 145 | `CompiledBase` (a `KernelAlgebra`), what every engine reads off a model: the primaries and their index, the state inputs, the rows (name, drift, noise loading, delay), the losses (atoms, Q, q), the ties; `close_under_delays`, `reject_leads` |
| `grid.py` | 522 | the piecewise-Chebyshev age grid of the stationary engine (`Grid`), barycentric interpolation, Gauss quadrature |
| `triangle.py` | 717 | the piecewise-spectral triangle `TriangleGrid` in (time, age), its strip and buffer for a transition, the interpolation (dense and sparse), the masses, and `LinePath`: the quadrature structure of a family of line integrals, cut at every piece edge, applied to a known kernel |
| `grid_cache.py` | 100 | grids shared between compiles (a sweep, a slider) |
| `past.py` | 256 | `Past`: what the time before zero leaves behind (a stationary result's kernels on the past's age grid, or initial shocks) |
| `settings.py` | 102 | `Settings`, the tuning constants; `tunable` binds a class attribute to one of them |
| `accel.py` | 156 | the Anderson-accelerated fixed point and the Newton-Krylov polish on the best-response map |
| `engine.py` | 858 | `EngineBase`: the packing of the tie representatives, the passive world and passive rows, the base best response on the kernel algebra (`algebra.py`; the stationary engine's), the fixed point (`solve`), `_finish`, the second-order check's Lanczos, the hooks; its class docstring lists which engine overrides which hook |
| `means.py` | 162 | `MeanLayer`, a base of `EngineBase`: the mean system assembled from the engines' mean hooks, its solve with the rcond guard, the mean cost quadrature and the result's mean fields |
| `stationary.py` | 774 | the stationary engine: `Compiled` (the kernel algebra on the age grid, the closed loop, its cyclic-symmetric reduction) and `StationarySolver` |
| `finite.py` | 406 | the uniform-cell finite-horizon engine `FiniteSolver`, first order in the cell, kept as a cross-check |
| `spectral_compiled.py` | 863 | `SpectralCompiled`: the breakpoint sequence and its closure under the lags (or the unit panels within `unit_range`), the grid, the past's and the buffer's wiring, the reads and node-to-node shifts (dense and CSR), the line paths, the initial shocks' discrete weights, the masses |
| `closed_loop.py` | 212 | `ClosedLoopSources` (the Volterra rows, the states' forcing columns, a control's convolution rows; mixed into `SpectralCompiled`) and `ClosedLoopRows`, the assembly of (I - M) Z = B one time panel at a time and its forward substitution |
| `spectral_operators.py` | 479 | the best-response operators as applications of the line paths and the sparse reads: `PathOp`, `RowOps` (G_k), `ProjOps` (H_k), `RespOps` (Resp_u), `FocOps` (Fu_u), `PanelRows` (the rows of G_k one time panel at a time); each with `dense()` |
| `finite_free.py` | 463 | `FocSystem` (the first-order conditions on the kept unknowns: assembled and factored within `foc_dense_max`, GMRES with the time-row preconditioner beyond), `best_response`, the decomposition and the second-order form, `reconstruction`, `panel_rows` |
| `spectral_means.py` | 262 | the means on the time line: `TimeLineOps` (the compiled model's time-node operators) and `SpectralMeans` (the spectral engine's mean hooks: the mean dynamics and conditions on the line s = 0 or on the time line, the atoms' mean paths), both mixins |
| `finite_spectral.py` | 556 | `SpectralFiniteSolver`: the options (a past, a continuation), the shapes, the identified unknowns and corner ties, `maps_from_world` (the projection, one system per time row), the costs, `loss_path`, `belief_error`, `settled`, the warm starts, the diagnostics hooks |
| `results.py` | 933 | `Result`, the one type every engine returns (axes, times, paths, status, extra, numerics), and its internal subclasses `StationaryResult`, `TriangleResult`, `TransitionResult`, `CellResult` over the three grids; `check`, `refine`, `stability`, `diagnose`, `to_dict`, the plots |
| `symmetry.py` | 154 | the cyclic symmetry of a tied model, found from the ties and verified on the expanded model |
| `sweep.py`, `transition.py` | 126, 100 | parameter sweeps with warm starts; `transition(old, new, T)` |
| `numerics.py` | 124 | `Numerics`: how a model is solved (engine, nodes, unit, unit_range, breakpoints, continuation_nodes, tol, damping, max_newton, variable, settings), laid over the model's own block |
| `engines.py` | 64 | `stationary`, `spectral`, `cells`, `ENGINES` by engine name and `build`: the power user's namespace, and where `solve()` builds the engine and the default start |
| `schema.py` | 263 | JSON Schema (draft 2020-12) for the model file and the payload; `validate` lists the violations with their paths, through `jsonschema` when importable, else the module's own validator |
| `cli.py`, `__init__.py` | 192, 141 | the command line (solve, validate, schema, transition, plot); `solve`, `load`, the exports, the deprecated names |

The engines never import each other: `finite_spectral` reaches the stationary engine through `noisestate.solve`
for a `continuation="stationary"` only.

## Metrics

The consolidation plan's limits are a function under 80 lines and a file under 900 (`wc -l`).  The spectral
modules meet both: no function of `spectral_compiled.py`, `closed_loop.py`, `spectral_operators.py`,
`finite_free.py`, `spectral_means.py`, `finite_spectral.py` or `triangle.py`'s paths is over 80 lines, and none
of those files is over 900.  Two files exceed 900 lines, `spec.py` (989) and `results.py` (933), and eight
functions elsewhere exceed 80: `finite.best_response` (127), `symmetry.find_cyclic_symmetry` (122),
`engine._second_order` (109), `spec.from_dict` (99), `stationary.closed_loop_symmetric` (97),
`triangle.TriangleGrid.__init__` (87), `engine.solve` (83) and `cli._run` (81).  Those ten are the next
pass's targets.

## One best response (spectral finite engine)

`EngineBase.solve` iterates on the action kernels (or the raw maps) of the tie representatives; each
evaluation calls `SpectralFiniteSolver.best_response`, which is `finite_free.best_response`:

1. **Compiled model** (`spectral_compiled.py`).  Everything map-independent was built at compile time: the
   grid, the sparse reads and shifts, the line paths (`_path`, cached on the grid), the state columns.
2. **Closed loop** (`closed_loop.py`).  `c.closed_loop(maps, excluded=agent, impulse_controls=agent.controls)`
   assembles the rows of (I - M) Z = B time panel by time panel from the Volterra rows times the state inputs'
   sparse atom operators and the convolution rows of every other agent's map times its seen rows' blocks, with
   the shocks, the initial shocks and one unit impulse per control of the agent as the forcing, and solves it
   by forward substitution: Z_pass (the passive world, the agent's strategy off) and R (the impulse responses).
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

`EngineBase` (engine.py) holds the packing of the tie representatives, the fixed point and its polish, `_finish`,
the passive world and passive rows, the base best response written on the kernel algebra `algebra.KernelAlgebra`
declares (every compiled model derives from it through `CompiledBase`), the second-order check's Lanczos, the singular-system message and the hooks table.  The stationary
engine uses the base best response on its `Compiled`'s kernel algebra; the spectral finite engine overrides
`best_response`, `_seen_rows`, `_representation_error` and `expected_cost` with the operator form and supplies
`closed_loop`, `block` and `atom_op` only; the cell engine overrides `best_response` wholesale and uses the
packing, the fixed point, `_finish` and the mean layer.  The mean layer (`mean_system`, `solve_means`,
`mean_cost`, `_mean_part`; `means.MeanLayer`, a base of `EngineBase`) is the base's as well, over seven mean hooks the engines fill in: the time nodes of
the mean paths, the mean state at time zero, whether anything drives the means, the mean dynamics operator
(the states' rows: a matrix on the stationary engine, a Volterra operator on the time line for the finite
ones), the mean first-order-condition operator (each agent's rows: the instantaneous derivative plus the
discounted continuation through the passive-world impulse responses, the DC gain on the stationary engine),
the loss atoms' mean paths and the discounted quadrature weights of the mean cost.  The base assembles the
joint (xbar, ubar) system, solves it with the rcond guard and fills `res.means`, `res.means_t` and
`res.cost_parts`.
