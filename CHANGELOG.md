# Changelog

## Unreleased

- **The fast suite runs at four BLAS threads: 743 s to 148 s.**  Nothing capped the thread count, so on a
  16-core machine every solve ran a 16-way GEMM on matrices of a few hundred to a few thousand unknowns,
  where the synchronisation costs more than the arithmetic; the curve is flat from 1 to 8 threads and falls
  off a cliff at 16 (`test_means_finite.py` 75.9 s uncapped against 16.9 s at four, `test_stability.py`
  55.2 s against 8.3 s).  `tests/conftest.py` now sets OMP/OPENBLAS/MKL/NUMEXPR/VECLIB to 4 before numpy
  loads, by `setdefault`, so an explicit setting still wins.  No test was gated, shortened or dropped: the
  same 233 pass.  tests/SLOW.md said "under two minutes" and its per-test seconds were already recorded "at
  four BLAS threads" -- the measurements were right and the setting under them had simply never been in the
  repo.
- `expr.compile_model` split into the pieces it already had as comment-delimited phases: `_check_kinds`,
  `_no_repeats`, a `_Walk` collector for the definitions reached, the shocks loaded and the Params used,
  `_check_quantities` and `_param_values`, leaving a 40-line assembler.  The check that an atom names a
  quantity of this model was written out four times and is now one closure; the duplicate-name check three
  times and is now one call.  No behaviour changes (the error wordings are pinned by
  test_expr.py::test_errors_name_the_object).

- **`Model.describe()`: the model as it now stands, without a solve.**  It reads the resolved fields, not the
  source dictionary, so a coefficient set on the object after loading shows up and no compile or solve is
  needed: the horizon and its window, the parameters at their current values, each state's and signal row's
  differential (a part that is identically zero is dropped, so a static state prints `dV = 0`), the delays,
  the definitions, each agent's controls and quadratic flow loss, the ties, the naive-observer assumptions,
  a transition's past and continuation, and `model.notes`, the conventions that apply.  The value is a
  string that prints as aligned plain text in a terminal (equals signs lined up within a block, everything
  wrapped to 88 columns, nothing escaped) and returns as HTML from a notebook cell, with the agents in a
  table; both views are rendered from one extraction, so they cannot drift apart.

- `foc_dense_max` 8192 -> 500: the dense first-order-condition solve of the spectral finite engine is never
  faster than the matrix-free one (0.26 s vs 0.12 s at 482 unknowns, 90 s vs 0.76 s at 4926, four times
  the memory), so only the smallest systems stay on the direct solve.  Shipped cases above 500 unknowns
  move in their last bits (costs and evaluation counts unchanged); the baseline is re-recorded.
- **The march's steps are local.**  After the first step, a solve at the next T fixes every strategy on the panels before
  T_prev - L at the previous solve's values (`SpectralFiniteSolver.freeze_before(t_lo, maps, actions=)`: the fixed
  maps enter the closed loop as known, the agents' own fixed actions join the passive world their first-order
  conditions see, the kept unknowns are the identified nodes from t_lo on, and the first-order-condition system,
  the projection and the preconditioner are built on those nodes only) and the fixed point runs on the free entries
  of its vector (`free_mask`); the warm start carries the previous fixed point's own action kernels (`res.actions`,
  `warm_actions_from`), so a reduced best response reproduces the whole strip's to 1e-15.  A local step leaves the
  previous handover frozen into the early part (2e-8 on Chapter 3 at 12 nodes), so after the last one a polishing pass
  with every unknown free, warm-started from it, runs to the solve tolerance (one evaluation at tol 1e-8, five at
  1e-11: the maps then equal the explicit solve's to 4e-11; the last row's `polish`).  The diagnostics run once, on
  the final strip.  Chapter 3, 3 -> 10 at 12 nodes: T = 3, 6, 9 in 20, 7, 5 evaluations on 576, 864, 576 unknowns
  plus the polish, 12 s (12.7 s before), the same window and gaps.  Unit steps below the past's window stay dear (the strip below L is cut at every unit: 1728 and 3168
  unknowns at T = 1 and 2, 135 and 197 s per step); above it a unit step costs 2.4 to 3.4 s.  `res.march` rows carry
  `unknowns`.
- **Nothing to solve, and the floor up front.**  When the T = 0 pass is under `settle`, the transition is the stationary
  equilibrium and the march returns it without solving: the continuation's maps on the first strip built, window 0
  (`res.extra["window"]`, `res.march_window`), one march row, no evaluation.  Before marching, the grid's floor is
  measured (`transition.settle_floor`: the same-model gap on the first strip, one best response per agent;
  `res.march_floor`, `res.extra["settle_floor"]`, in the payload): a `settle` below it stops at once with
  `march_stop = "floor"`, the stationary result and the `settle floor` row (raise numerics.nodes), and a gap within a
  factor FLOOR_FACTOR = 2 of it during the walk is the floor (the fall-rate rule is gone).  Chapter 3's first strip:
  3.3e-2 at 6 nodes, 2.5e-3 at 8, 7.8e-6 at 12 (the first window holds the band's tip, where the one-shot floor is
  worst: the last window's floor at a later T is lower, 3.7e-3 at 6 nodes, 1.8e-4 at 8).

- **The excess cost's tail past T, and the march's floor stop.**  `res.excess_windows[agent]` (the excess's discounted
  integral per window, the last window first), `res.excess_costs_tail` (the last window's excess times r / (1 - r),
  the factor r per window from the loss path's decay over the last two windows or from the march's last two gaps),
  `res.excess_costs_total` and `res.excess_tail` (source, factors, windows), in the payload and the summary.  On
  Chapter 3's 3 -> 10 at 12 nodes the excess per window falls by about 4000 per window (2.40e-2, 5.85e-6, the floor),
  so the untailed value (0.0240215, 0.0240273, 0.0240273 at T = 3, 6, 9) is converged by T = 6 to 1e-9 and the tail is
  below the floor (1.4e-9 at T = 6, -1e-12 at T = 9).  The march stops at the grid's one-shot floor (two windows past
  the first, a gap that fell by less than a factor of 4 over a window: a transient falls by hundreds) with
  `march_stop = "floor"` and a `settle floor` row flagging SETTLE BELOW THE GRID'S FLOOR, advice raise numerics.nodes
  (settle 1e-6 at 8 nodes: stopped at T = 12 after 1.8e-4, 1.8e-4); the default step is one window (see below).
- **A transition horizon shorter than the past's window.**  The compile's refusal of T < L was an implementation
  artefact: for any T > 0 (a positive multiple of the unit) the strip is the rectangle [0, T] x [0, L], the new shocks
  below the diagonal, the old ones above it, the buffer [T, T + L] closing it as before.  `TriangleGrid` marks the
  buffer's pieces above the line s = 0 as the band (old shocks alive past T; a square straddling s = 0 is cut along its
  diagonal, a rectangle straddling it is refused), the compile closes the cuts below L under the shift by T (the
  buffer's panels are the age panels shifted), and every node-based read carries its piece's s-range (`side_ds`): on a
  diagonal that is not its own piece's a node reads the side its piece lies on (the corner (T, T) of the buffer's first
  upper triangle lies on s = 0 and read the band).  The same-model identity holds below the window to the T = 6 floor on
  every node, the buffer and its band included (Chapter 3 at 12 nodes, T = 1 and 1.5: one-shot 3.8e-6 / 1.3e-5, kernels
  within 1.3e-5); the per-diagonal side also lowers the T = 3 one-shot floor from 4.2e-5 to 3.8e-6.  Baseline at 0.
  `transition_gap` now runs on the smallest strip, one unit (`numerics.unit`, else the smallest lag, else L).  The
  march's first solve stays at T = L with one window per step; unit steps are available by an explicit `step=` only,
  since a first window cannot certify anything (its monitor window [0, T] holds the initial transient: on Chapter 3
  with `unit: 1` the gaps at T = 1, 2, 3 are 0.575, 0.576, 0.577) and without panel reuse they do not pay (3024 nodes
  at T = 1 against 576 at T = 3: 155 s against 4 s per step at 12 nodes).
- **The horizon as an output: `transition(old, new, settle=tol)`, the march in T.**  Exactly one of `T` and `settle`
  (`T` keeps today's behaviour bit for bit); the file form takes `settle:` in place of `window:` under kind transition
  (exactly one), the CLI `transition --settle TOL [--step DT] [--max-window K]`.  The march starts at the T = 0 pass
  (`transition_gap`; under the tolerance the smallest-window solve is the transition), grows T by `step` (default one
  window; the engine's floor is T = L) with each solve warm-started from the previous maps (`warm_maps_from`, the
  continuation solved once), and after each solve runs the monitor, the best-response pass (`transition.gap_passes`)
  on the last window [T - L, T], the range of `settled`: the design note's caution that the gap at T is the handover and
  never small was wrong (on Chapter 3 the explicit T = 9 solve settles at 2.7e-6 under a 1e-4 tolerance; a monitor on
  the window before the last stopped one window late, at T = 12), so the march stops at the smallest T whose explicit
  solve settles.  Stops under `settle` or at `max_window` windows (default 8: the settled flag stays and names the
  stop).  `res.extra["window"]`, `res.march` (rows `{T, gap, evaluations, seconds, monitor}`), `res.march_stop`, in
  the payload too.  Chapter 3, 3 -> 10 at 12 nodes, settle 1e-4: T = 3, 6, 9 in 20, 8, 4 evaluations, gaps 0.577,
  9.2e-4, 9.3e-7 (a factor of 625 then 990 per window), 32 evaluations against 20 for the explicit solve at T = 9,
  the maps agreeing to 5e-8; the same-model past stops at T = 0.  No number of an explicit solve moves (baseline at 0).

- **The T = 0 pass of the settle march: `transition_gap(old, new)`.**  With nothing solved, every agent at the
  new model's stationary rules from date zero and the old regime's shocks attached, one best response per agent;
  returned per agent is the relative distance of the best-response rule from the stationary rule (max over the
  identified nodes, relative to the rule's peak).  The engine cannot build a strip shorter than the past's window,
  so the pass runs on [0, L].  Chapter 3, 3 -> 10 at 12 nodes: 0.585 and 0.061; 3 -> 6: 0.287; 3 -> 3.03: 0.0033
  (linear in the mismatch); the same model as its own past sits at the one-shot floor, 4e-5 at 12 nodes, 1e-5 at
  16, 2.4e-6 at 24.  `transition.gap_pass(S, maps, lo, hi)` is the pass on a time range, the march's monitor.
  `transition()` is unchanged (its model construction moved to `_transition_model`); no number moves.

- **Models as equations.**  `noisestate.expr`: `Param` (arithmetic renders to the file's coefficient expressions,
  `p1**0.5` to `"sqrt(p1)"`), `shocks()`, `State` (`X.drift = D + sigma * w.w0`), `Control`, `define()`, `.lag()` /
  `.lead()`, `Signal(name, expr, delay)`, `Agent(name, controls, signals, loss, myopic, naive_observers)` with a
  quadratic loss compiled to the term list (`(X - theta)**2` is `[1, X, X], [-2*theta, X]`, the constant noted in
  `model.notes`), and `ns.Stationary`, `ns.Finite`, `ns.Transition`.  `ns.Model(name, states=, agents=, ...)` accepts
  the expression form beside the field form (one class; the expression form is compiled through `from_dict`, so
  `to_dict()` is the file) and gains `solve`, `sweep(param=values)`, `finite`, `stationary`, `save`, `load`.
  `res.kernel()` returns a `Kernel` (an ndarray with `.axes`, `.values`, `.at()` on the engine's interpolant,
  `.plot()`); `res.status` and sweep rows are dicts with attribute access; `ns.settings(**overrides)` is a context
  manager over the default `Settings`.  `examples/expr_examples.py` writes the seven shipped models as equations
  and `tests/test_expr.py` checks each against its YAML file's dict and the baseline's costs; README gains
  "Models as equations".  No number moves (baseline at 0).

## 0.5.0 (2026-09-06) — consolidation and the explicit API

The consolidation pass (the spectral finite engine's operators, assembly, modules and mean layer, one kernel
algebra) and the API stage (Numerics apart from the model, an explicit solve(), one Result, the schema, the
CLI), with the review's fixes; every shipped case at distance 0 from the record taken at ec1b533, the newer
spectral steps within 4e-15 of the record before it.  The bullets are grouped by theme, newest first within
each.

### Numbers: the best response, the closed loop and the grid

- `TriangleGrid.path` cuts and quadratures every output node at once (the consolidation pass, step C3): the
  read point's breakpoint crossings in t and in a, the triangles' diagonals, the known's age grid and the
  extra cuts are computed as arrays over the nodes (`_crossings`, `_crossings_1d`, `_cut_values`), the edges
  sorted and deduplicated per node with one lexsort, the Gauss points and weights of every interval in one
  step, and `point_fn`, `known_fn` and `extra_cuts` are called once with the array of nodes instead of once
  per node (the callbacks are written for it; the docstring says so).  The result is identical: every path
  family of ch1_delayed at 8 nodes, Chapter 3 as its own transition and the Kyle-Back prior agrees with the
  per-node loop in rows, points, weights, I, J and R to the array (tests/test_triangle.py holds the per-node
  oracle), and the seven families of the Chapter 1 transition at 5 nodes (N = 14700) too: their construction
  104 s -> 14 s (the largest, 3.9 M points, 58 s -> 8.7 s); at 7 nodes (N = 28812) the first best response,
  paths included, takes 149 s (49 s of it the paths' construction, 29 s of that `interp_sparse`).  The profile
  of one best response on the GMRES path (cProfile, the BLAS/LAPACK entry points timed by an LD_PRELOAD shim,
  the scipy.sparse kernels from the profile): the Chapter 1 transition at 5 nodes 34 s = 20 s Python and
  numpy elementwise + 13 s scipy.sparse + 1 s BLAS; at 7 nodes 149 = 69 + 74 + 6; the two-firm market at
  tau 0.5, L = T = 6, 5 nodes (N = 11100, 9 primaries) 163 = 67 + 80 + 16 (11 s of the BLAS the closed loop's
  panel solves).  The Python-level own time is numpy work on the quadrature points (`interp_factors`' per-piece
  masks, `panel_of`, `PathOp.apply`'s products), not interpretation: a compiled point location would save
  about 10 s at 5 nodes and 35 s at 7 nodes of the first best response and nothing of the later ones, whose
  time is the sparse products of the operators (csr_matvecs, 60 to 70 % of a best response).
- One best response (the consolidation pass, step C3).  The spectral finite engine's row, response,
  first-order-condition and projection operators are defined once, as `finite_free`'s applications of the
  line paths and the sparse reads (`RowOps`, `RespOps`, `FocOps`, `ProjOps`; with a past the band's read of
  the past's increments, the old-shock and pre-zero segments and the initial shocks' discrete weights and
  point conditions are segments of the same operators).  `FocSystem` assembles and solves the first-order
  conditions on the kept unknowns: within `settings.foc_dense_max` (unchanged, 8192) from the operators'
  dense rows (`dense()` on each operator: the same sums as their applications, `LinePath._weighted_sum`),
  LU-factored with the condition estimate of `_solve_regular`; beyond, by GMRES on the applied operators as
  before.  The second-order check's dense form, the FOC decomposition, `maps_from_world` (the with-a-past
  copy `_maps_from_world_past` folded in: the identified unknowns, the corner ties and the point conditions
  are the past's segment of one loop; the systems of a size solved in one LAPACK call), the representation
  error, `belief_error`, the costs (the sparse mass) and the mean systems (`FocOps.on_qzeta` in place of the
  dense per-atom operators) all go through the operators.  Gone: the dense `_row_operator`,
  `_projection_operator`, `_response_operators`, `_foc_affine`, `_solve_foc`, `_corner_ties`, `_init_mass`
  and the dense `best_response` body of the spectral engine; `SpectralCompiled.conv_left`, `conv_right`,
  `response_op`, `continuation_op`, `projection_op`, `conv_rows`, `instant`, `instant_adjoint`, `response`,
  `continuation`, `own_lag_read`, `cost_mass`, `buffer_mass`, `diag_read`, `shock_time_read`,
  `projection_rows`, `row_op`, `_many`; `TriangleGrid.mass_matrix` and `LinePath.with_known_many`.  The
  engine base's dense pieces stay for the stationary engine.  Numbers move in the last bits only: every
  shipped case's costs to 1e-12 with the same evaluation counts and Z within 4e-15 of its peak (the
  baseline record is not re-written); the fast suite 183 in 107 s, the slow set 18 in 240 s (was 285 s), the
  fast suite with GMRES forced 183 in 167 s; `finite_spectral.py` 2117 -> 1750 lines.
- One closed-loop assembly (the consolidation pass, step C2).  `SpectralCompiled.closed_loop` builds the rows
  of the system (I - M) Z = B one time panel at a time from the line paths and the sparse reads and solves them
  by block forward substitution at every size (`finite_spectral.ClosedLoopRows`, whose docstring states the
  assembly: the state rows from the Volterra path times the state inputs' atom operators, a control's rows
  from the map convolutions times its seen rows' blocks, the band's old shocks, the buffer's frozen rows, the
  initial-shock and impulse columns as the forcing); `world_from_actions` is the same assembly with the
  controls' rows given.  The dense (n_prim N)^2 system, `_closed_loop_past`, `_closed_loop_panels`,
  `_state_part`, `_solve_causal` and the two dense `world_from_actions` bodies are gone, and with them
  `settings.closed_loop_dense_max` (the per-panel forward substitution is the block solve the dense path did;
  `foc_dense_max` keeps its meaning).  Closed loop numbers change in the last bits; costs and evaluation counts
  unchanged: every shipped case's costs agree with master 222eb96 to 1e-12 with the same evaluation counts and
  the fixed points' Z to 2e-15 of their peak, and tests/refs/baseline_0.4.json is re-recorded (at 12
  significant digits a few dozen entries of thousands sit on a rounding boundary, so the 12-digit SHA moved on
  four cases and the raw bytes on five; ch3_two_player, ch4_kyle_back and ch5_cycle_market are bit for bit).
  One convention settled: the dense `world_from_actions` read a lagged control input through the interpolation
  `read(lag, lag)`, the closed loop and the per-panel path through the node shift `map_shift(lag)`; the two
  agreed to 1e-15 on continuous action kernels (every equilibrium) and differed by 3e-3 at the piece
  boundaries of a random one, and the shift is now the one reading.  Per-panel Volterra rows are computed,
  not cached: caching them saved nothing measurable (2.0 against 2.2 s per repeated closed loop at N = 14700)
  and cost 0.9 GB.  One closed loop of ch1_delayed at 16 nodes (N = 2560): 3.2 s and 4.60 GB peak to 2.9 s and
  4.14 GB; ch1_delayed as its own transition (window 3, T = 6, 5 nodes, N = 14700): 19.7 s and 3.72 GB to
  20.1 s and 3.72 GB, the first best response 103.6 s and 7.77 GB to 106.5 s and 7.77 GB (timing noise).
  tests/test_causal_solve.py checks the forward substitution against the dense solve of the stacked rows
  (plain, an agent excluded with its impulse column, the actions given; with a past and a continuation).
- The spectral finite engine's best response matrix-free.  Beyond `settings.foc_dense_max` unknowns nU nR N
  (the largest agent's; default 8192, above every shipped example and test, whose numbers are unchanged to
  the bit) `SpectralFiniteSolver.best_response` no longer builds the row, response, first-order-condition and
  projection operators as N x N arrays nor the (nU nR N)^2 system: `noisestate/finite_free.py` applies each
  of them from the line paths (`PathOp`: R diag(w J y) I on the path's quadrature points, the known kernel
  read once; `RowOps`, `RespOps`, `FocOps`, `ProjOps`) and the compiled model's sparse reads (`atom_sparse`,
  `instant_sparse`, the new `mass_sparse`, `diag_read_sparse`, `shock_time_read_sparse`), and solves the
  first-order conditions by GMRES on gamma -> sum_k H_k Fu (sum_v Resp_v G_k gamma_v) (`FocSystem`) to
  `settings.foc_krylov_tol` (1e-12, relative to the right-hand side) within `foc_krylov_maxiter` (400),
  warm-started from the agent's last solution, preconditioned by the part of the operator that is block
  diagonal by time row (the own cost read instantaneously through the row operator and projected back,
  kron(Q_l, sum_k H_k D_l G_k) over the control's own lags l, assembled one panel at a time from the rows of
  G_k and H_k and LU-factored per time row); a block whose reciprocal condition estimate is below
  `foc_rcond` raises the singular-system error the dense path raises.  The projection on the seen rows
  (`maps_from_world`, now through `sub(idx, cols)`), the representation error, the belief error, the costs
  (sparse mass), `world_from_actions` (the per-panel closed loop with the actions given) and the
  second-order check (the form applied by the same operators, a block of strategies at a time; Lanczos
  beyond `second_order_dense`, `EngineBase._lanczos_extremes`) go the same way, so nothing N x N is built.
  Measured against the dense path with the threshold at 0: ch1_delayed at 8 nodes (random maps, and the
  fixed point on the action kernels: the same 13 evaluations, costs to 1e-15), Chapter 3 as its own
  transition and the Kyle-Back prior agree to 1e-12 of the peak on gamma, actions, world, maps and the
  checks; the whole suite passes with the path forced (`NOISESTATE_FOC_FREE=1`, 194 passed in 470 s
  against 239 s dense).  GMRES takes 11 to 24 iterations from zero on the examples and 48 to 62 on
  ch1_delayed as its own transition (window 3, T = 6), 1 warm-started at the same maps; at 4 nodes
  (N = 9408) the best response goes from 106 s and 21 GB (dense, both operators and system) to 8 s and
  3.0 GB, agreeing to 8e-12, at 5 nodes (N = 14700) it takes 17 s and 7.8 GB (the first one 121 s, the
  paths being built), and at 7 nodes (N = 28812) the closed loop (124 s, 18.6 GB), the projection path
  (140 s, 24 GB) and the continuation path fit but the first best response exceeds ten minutes of path
  construction (`TriangleGrid.path` is a Python loop over the nodes), so that case and the two-firm
  Chapter 5 market at 5 nodes are not yet measured to convergence.

- `horizon.unit_range` on the triangle grid.  The finite engine cut the time and age panels at every multiple
  of the unit up to the window; within unit_range it still does, and beyond it the panels grow geometrically
  (`TriangleGrid.fill_geometric`, the stationary grid's rule: the kink at the k-th delay line weakens with k),
  the required cuts kept: T, T - k unit within unit_range when the game ends at T (a control is idle within the
  last lag), the strip's L and the past's cuts within unit_range, the buffer's panels.  Beyond unit_range a
  lagged read is interpolated (`map_shift`, `map_shift_sparse`) and the panels are not closed under the lags;
  the default (None, or the window) is today's grid bit for bit.  Measured on ch1_delayed with player2's row
  undelayed (window 3, 6 nodes): unit_range 1.0 keeps the costs to 1.2e-8 and the kernels to 6.9e-6 of their
  peak with N 2808 -> 1980 and 104 s -> 44 s; 0.5 to 3.2e-6 and 1.8e-4 with N 1008 and 8.9 s.  On the
  transition of ch1_delayed as its own past (window 3, T = 6, 4 nodes) unit_range 1.5 takes N from 9408 to
  3168 (588 to 198 pieces) and the one-shot identity from 5.8e-4 / 4.0e-4 to 2.5e-3 / 6.0e-3 (the costs of the stationary maps on the strip within 1e-6 relative), 1.0 to N 2352
  and 1.2e-2 / 2.2e-2 (the delayed row's map suffers first).  A row observed with a delay and no past keeps
  its map on the action grid shifted by the delay and must be read node to node on every piece (an
  interpolated read leaves map nodes unidentified: a singular first-order condition), so unit_range below the
  window is refused there (solve as a transition, where the map is in raw age); initial shocks with a delayed
  row are refused as well.  `Model.with_horizon` drops `breakpoints` and `unit_range` on a change of kind
  unless given, as `transition()` does: a stationary grid's unit_range is not a finite grid's.
- The spectral finite engine's closed loop assembled one time panel at a time.  Beyond
  `settings.closed_loop_dense_max` unknowns (n_prim N; default 16384) `SpectralCompiled.closed_loop` no longer
  builds the (n_prim N)^2 dense system: for each time panel the rows of the state part (the Volterra path
  restricted to the panel's output nodes, `LinePath.apply(rows=)`) and of the map convolutions
  (`conv_left_rows`, `LinePath.with_known(rows=)`) are built against the sparse reads of the primaries
  (`read_sparse`, `map_shift_sparse`, `row_blocks_sparse`, `state_inputs_sparse`: an interpolation touches one
  piece per point) and solved by forward substitution from the earlier panels, then discarded; the band's and
  the buffer's forcing goes the same way, and the state columns are shared with the dense path
  (`_state_columns`, the Volterra products on the path without the N x N operator).  Peak memory is
  n_prim^2 N N_panel: ch1_delayed as its own transition at window 3, T = 6 (36 panels) goes from 13.0 GB and
  44 s to 1.3 GB and 7 s per closed loop at 4 nodes (N = 9408), from 31.9 GB and 157 s to 3.6 GB and 20 s
  at 5 nodes (N = 14700), and runs at 7 nodes (N = 28812, 86k unknowns, a 56 GB dense system) in 18 GB and
  123 s.  Within the limit the dense path was kept at first and every shipped example and transition case was
  the 0.4.0 result bit for bit; the per-panel world agrees with it to BLAS rounding (2e-16 to 3e-13 relative on
  the examples), not to the bit, because OpenBLAS rounds the product of a row block differently from the
  rows of the full product (one entry in a thousand differs in the last bit on this machine); the per-panel
  assembly is now the only one (the entry above).  The Volterra
  operator `Vol` is built on first use (the mean system's line s = 0); the best response still
  holds about thirty N x N dense operators (`conv_rows`, `response`, `continuation`, `projection_rows`, the
  row and projection operators), 20 GB at N = 9408, which is now the size ceiling.

### Structure: the modules, the mean layer and the kernel algebra

- **The kernel algebra as an explicit interface.**  `noisestate.algebra.KernelAlgebra` declares every operator
  the base engine calls on a compiled model (the closed loop, `block`, `atom_op`, `expr_op`, `expr_kernel`,
  `row_blocks` / `row`, `conv_rows`, `instant`, `instant_adjoint`, `response`, `continuation`, `own_lag_read`,
  `projection_rows`, `cost_mass`, `causal_chunks`) with its shapes and the attributes every compiled model
  has; `CompiledBase` derives from it, so the stationary `Compiled`, `SpectralCompiled` and `FiniteCompiled`
  implement it, a member an engine lacks raising `NotImplementedError` naming it.  `results.py` and `past.py`
  read compiled models through the interface's members and documented attributes only.  Bit identity: every
  shipped case at distance 0, function bodies unchanged.
- **One mean layer.**  `mean_system`, `solve_means`, `mean_cost` and `_mean_part` are written once, in
  `noisestate.means.MeanLayer` (a base of `EngineBase`), over seven mean hooks each engine fills in: the time
  nodes of the mean paths, the mean state at time zero, whether anything drives the means, the mean dynamics
  operator (the states' rows), the mean first-order-condition operator (each agent's rows), the loss atoms'
  mean paths and the discounted quadrature weights.  The three engine copies are gone; the cell engine's mean
  solve gains the rcond guard the other two had.  Means bit-identical on every shipped case; the finite mean
  costs move by a summation order (4e-16 relative); the kernels untouched (baseline distance 0).
- The three long functions of the spectral finite engine in named parts, the same statements in the same order
  (step C4, bit for bit): `SpectralCompiled.__init__` (168 lines) calls `_regimes` (the past and the
  continuation, T, Tg), `_breakpoints` (the sequence and its closure under the lags, or the unit panels
  within unit_range), `_grid` (the strip's sequence and the triangle grid), `_wire_buffer`, `_wire_past`,
  `_wire_time` (the Volterra path, the time rows, the time nodes and mean_embed) and `_caches`, each
  docstring naming its invariant; `maps_from_world` (86) builds each time row's weighted least-squares
  system in `_time_row_system` (the shared masks and corner maps of a past in `_projection_context`) and
  keeps the batched solve; `FocSystem.preconditioner` (80) takes the lag forms from `_lag_forms` and each
  time row's block kron(Q_l, sum_k H_k D_l G_k) from `_time_row_block`, and keeps the reduction to the kept
  unknowns and the factorisation.  `_vol_rows`, `_state_columns` and `conv_left_rows`, the map-independent
  blocks and forcing only the assembly reads, are the ClosedLoopSources mixin in `closed_loop.py`.  No
  function of the spectral modules is over 80 lines; no spectral module over 900.
- The spectral finite engine in six modules, one responsibility each (the consolidation pass, step C4; pure
  moves, every shipped case's Z bit for bit): `spectral_compiled.py` (SpectralCompiled: the breakpoints and
  their closure, the grid, the past's and the buffer's wiring, the reads and sparse reads, the line paths,
  the masses), `closed_loop.py` (ClosedLoopRows), `spectral_operators.py` (PathOp, RowOps, ProjOps,
  RespOps, FocOps, PanelRows), `finite_free.py` (FocSystem and its preconditioner, best_response, the
  decomposition and the second-order form; it re-exports the operators), `spectral_means.py` (the means on
  the time line: TimeLineOps, the compiled model's time-node operators, and SpectralMeans, the solver's
  mean system, both mixed in) and `finite_spectral.py` (SpectralFiniteSolver with maps_from_world and the
  diagnostics hooks; `from noisestate.finite_spectral import SpectralCompiled, ClosedLoopRows` still
  works).  No function body changed; the module docstrings and EngineBase's say where each neighbour is.

### API: Numerics, solve(), Result, the schema, the CLI and the deprecations

- **Review fixes (the consolidation branch's review).**  The names kept from before one `Result` warn: a
  `DeprecationWarning` on `res.iterations` and `res.Z` (read `res.evaluations`, `res.world`), on
  `solve(nodes=)`, `solve(settings=)`, `transition(nodes=)` and `transition(stationary=)` (pass a `Numerics`),
  and on `noisestate.StationaryResult`, `TriangleResult`, `TransitionResult`, `CellResult` by name (every
  engine returns `noisestate.Result`; the classes stay in `noisestate.results`); all go in 0.6.  The cell
  engine's `res.kernel(name)` without a channel returns the stack over the channels, (N, N, nW), the last axis
  one column per channel as on the other engines, instead of raising.  A `Settings` field passed to `solve()`
  is a `TypeError` naming the `Numerics(settings={...})` route.  README's error contract: a past on the cell
  engine is a `ValueError` from the numerics, not a `NotImplementedError`.  `engine.py` drops the dead
  fallbacks to `c.row` and to a compiled model without `causal_chunks` (`KernelAlgebra` declares
  `row_blocks` and `causal_chunks`; `row` leaves the interface).  `docs/architecture.md`'s module table
  refreshed (`numerics.py`, `engines.py`, `schema.py`, `means.py`, `algebra.py` listed) with a "Metrics"
  paragraph naming the files and functions over the plan's limits.  Tests: each deprecated name warns once
  and the suite reads the new names; the cell engine's kernel stack; its mean solve's rcond guard.
- **API stage, D, F and G: the schema, one transition default, the CLI.**  `noisestate.schema("model" |
  "payload")` returns JSON Schema (draft 2020-12) for the model file (the `numerics:` block; the deprecated
  nested keys and `kind: finite_cells` marked `deprecated`) and for the payload (`payload_version` 1);
  `noisestate.schema.validate(doc, which)` lists the violations with their paths, through the `jsonschema`
  package when it is importable (not a dependency) and otherwise through `schema.py`'s validator of the
  keywords the schemas use.  The CLI gains `schema {model|payload}`, `transition old.yaml new.yaml --window T
  [-o] [--plot] [--nodes]` and `plot result.json out.pdf` (re-solves the payload's model under its recorded
  options), and `validate` checks the file against the schema first, reporting each error with its path.
  `start` defaults to `"stationary"` wherever a continuation is given (`solve(past=, continuation=)`, the
  file form, `sweep()`'s first point, `transition()`), else `"zero"`; `solve(start="zero")` asks for the zero
  start explicitly (`solve()`'s `start` default is None).  The file form of a transition therefore starts from
  the stationary maps too: `examples/ch3_precision_change.yaml` goes from 23 to 20 evaluations and its costs
  and excess costs move by 4e-8 (both fixed points within tol 1e-8).  The example is now a case of the baseline
  record, re-taken with it (the eight earlier cases bit-identical in Z; ch1_mean_sweep_p10's mean costs carry the
  mean layer's 4e-16 summation-order move).

- **API stage, C and E: one `Result`.**  `noisestate.Result` is the type every engine returns
  (`isinstance(res, ns.Result)`; `StationaryResult`, `TriangleResult`, `TransitionResult` and `CellResult` are
  its internal subclasses, importable until 0.6, `BaseResult` an alias of `Result`).  A consumer reads a kernel
  without knowing the engine: `res.axes` gives the coordinates of `kernel(name, channel)` ({"age"} on the
  stationary engine; {"time", "age", "shock_time"} node-wise on the spectral triangle, shock_time < 0 on a
  transition's band; {"time", "shock_time"} for the cell engine's matrices) and, under "maps", where every
  row's map values belong (the former `map_axes`); `res.times` and `res.paths` ({"means"}; a transition adds
  "loss" and "belief_error") hold the paths; `res.status` = {"ok", "flags", "rows"}; `res.extra` the
  engine's extras (window_tail; past, continuation, settled, old_flows, new_flows, excess_costs,
  representation_parts).  Names: `res.evaluations` (`iterations` kept as an alias until 0.6), `res.world`
  (`Z` kept).  The payload carries `payload_version` 1, `engine` (the Numerics engine) beside `kind`, the
  `horizon` as the file's economics and `numerics` beside it, `axes`, `times` and `status`.

- **API stage, A and B: the numerics apart from the model, and an explicit `solve()`.**  `noisestate.Numerics`
  (`numerics.py`) holds how a model is solved: `engine` ("stationary" | "spectral" | "cells", default from
  the horizon kind), `nodes`, `unit`, `unit_range`, `breakpoints`, `continuation_nodes`, `tol`, `damping`,
  `max_newton`, `variable` and `settings`.  The model file keeps the economics under `horizon:` (kind,
  discount, window, past, continuation, `stationary: {window}`) and gains an optional top-level `numerics:`
  block; the keys once nested under `horizon:` (`nodes`, `unit`, `unit_range`, `breakpoints`,
  `stationary.nodes`, `kind: finite_cells`) are still read and mapped, with one deprecation note each in
  `model.notes`, until 0.6 (a nested key that disagrees with the block is an error).  `solve(model,
  numerics=None, *, init, start, tol, max_evaluations, deadline, progress, diagnostics, refine, stability,
  verbose, naive_observers, past, continuation)`: no introspection; `nodes=` and `settings=` are accepted
  as aliases of the Numerics fields until 0.6, every other former engine keyword is a TypeError naming the
  field.  `noisestate.engines` (`stationary`, `spectral`, `cells`, `ENGINES` by engine name, `build`) is
  the power user's namespace; the engine classes stay importable.  `sweep(..., numerics=)`,
  `transition(old, new, T, numerics=)` (`nodes=` and `stationary={"nodes"}` aliases until 0.6),
  `make_solver(model, numerics=)`, `Model.numerics`, `Model.with_numerics()`, `ModelBuilder.numerics()`;
  `res.numerics` is the resolved object and the payload's `options.numerics` carries it.  The shipped
  examples and the README use the block.  No number moved: every shipped case at distance 0 from ec1b533.


### Docs, tests and tooling

- **README as the user's document; the reference material in `docs/`.**  README (259 lines) is what it is,
  install, a first solve on Model / Numerics / Result, the model file in brief, transitions and sweeps in
  brief, the guards with the flag text each prints, settings, the error contract, the command line and where
  things are, every snippet executed.  `docs/` holds `model_file.md` (every key of the schema with type,
  default and the deprecated keys marked), `payload.md` (every `to_dict` key), `guards.md`, `settings.md`,
  `validation.md`, `method.md`, `limits.md`, `transitions.md` and `design/` (the transition, size and
  consolidation records); `docs/README.md` indexes them.
- The baseline record re-taken at ec1b533 (`tests/refs/baseline_0.4.json` and `.npz`): the operator step (C3) had
  moved the spectral cases by up to 4e-15 in the last bits, and the API stage that follows must move no number,
  so the record is the state at its starting point; every case at distance exactly 0 from there on.
- `docs/architecture.md` (step C4): the modules and what each holds, the data flow of one best response
  (compiled model, closed loop, passive rows, operators, FOC solve, projection, checks) and of one transition
  (past, band, buffer, continuation), and where the three engines share the base; linked from README's
  "How it works".
- The baseline record compares Z itself.  `extras/compare_baseline.py write` stores every case's Z as float64
  in tests/refs/baseline_0.4.npz next to the JSON (the costs, evaluation counts, residuals and means stay
  there), and `check` compares Z by max |dZ| / max |Z| against 1e-12, the value reported per case, instead
  of the SHA at 12 significant digits (which a rounding boundary could move without any change of the
  numbers); the raw-bytes SHA stays as information.  tests/test_baseline.py follows; the record is
  re-written at this commit (every case at distance 0).
- Tests and tooling only (the package is untouched).  `tests/helpers.py` holds the builders the transition
  and means tests repeated: the example loaders (`example`, `example_dict`, `example_path`), a model solved
  stationary at a number of nodes (`stationary`, `delayed_stationary` for the delayed Chapter 1 game), the
  same-model past-and-continuation setup on a strip (`same_model_solver`, `same_model_setup`), the two-firm
  Chapter 5 market with its ties and linear terms dropped (`two_firm_market`), the one-shot best-response
  identity (`one_shot_deviation`, `one_shot_from_the_stationary_maps`), the stationary maps and kernels carried
  onto a strip, the discounted one-agent prior model, and `solve_record` (costs, evaluations, residual, scalar
  means, SHA-256 of Z at 12 digits and of its bytes).  The fast suite runs in 111 s (183 tests) against 256 s
  (197): fourteen solves above five seconds whose pin is repeated at a smaller size or by another test are
  gated behind NOISESTATE_SLOW=1 (`helpers.slow`, `helpers.slow_param`, which also mark them `slow`);
  `tests/SLOW.md` lists every gated test with what it pins, and the weekly CI job selects them with `-m slow`
  (its `-k` expression would have missed the moved tests).  No assertion or tolerance changed.
- The re-baselining instrument.  `extras/compare_baseline.py` solves the five shipped examples, the Chapter 1
  target sweep's p = 10 point, Chapter 3 as its own past and continuation at 6 nodes and the Kyle-Back prior,
  and writes a record per case (costs and their parts to full repr, the evaluation count, the residual,
  `settled`, the scalar means, Z's shape, Z's SHA-256 at 12 significant digits and the SHA-256 of its raw
  bytes; `write` and `check` modes).  `tests/refs/baseline_0.4.json` is the record of master 222eb96 and
  `tests/test_baseline.py` (under NOISESTATE_SLOW=1, 11 s) holds the package to it: costs to 1e-12, the
  evaluation counts equal, Z equal at 12 digits; a change of Z's bytes with the 12-digit SHA intact (BLAS
  rounding) is reported as a warning, not asserted.

## 0.4.0 (2026-09-06) — transitions from a stationary past

- Transitions.  The spectral finite engine solves the equilibrium path of a regime change: the game runs
  in the old stationary equilibrium until time zero, its coefficients change, and the new path is solved
  on [0, T] from a known *past* (`noisestate/past.py`: `Past`, the loadings of the pre-zero shocks on the
  old regime's states, controls and signal rows as functions of shock age on the past's window L, the
  rows' noise loadings and the old constant means), given as a converged `StationaryResult`, a stationary
  model / dict / path solved on the fly, or a list of initial shocks `{"name", "loads": {state: coef},
  "rows": {"agent.row": coef}}` (point loadings at time 0-, a prior on a state seen at once by the rows
  named; the loadings may be parameter expressions).  `Past.validate` matches channels, states, controls
  and rows by name and checks the past grid's breakpoints and delays against the new panel unit; an
  unconverged past is a `ValueError`, a finite result a `TypeError`.
- The strip.  `TriangleGrid(breakpoints, nt, na, T=, window=, buffer=)`: with a window the domain is
  [0, T] x [0, L], today's pieces below the diagonal (same nodes and order, cut off at age L) and above it
  the shocks born before zero (the band: mirrored Duffy triangles and rectangles, every square above a
  diagonal split along its own diagonal, since a switch of the maps at t0 reaches a state through a lag
  tau only at t0 + tau and the kernels kink along s = t0 - k tau), each degenerate corner row one unknown
  with a point condition; `interp` takes `side_d` for a diagonal, a lagged read at a corner takes the side
  the reader's own sides imply, `path` a `known_grid` (a known kernel on an `AgeGrid`, cut at its
  breakpoints), `row_quadrature` the Gauss points of a time row.  `SpectralCompiled(model, past=,
  continuation=)`: the band's state at zero is the past's state kernel propagated by e^{At}, lagged
  atoms read before zero are the past's kernels, a row's increments observed before zero enter the
  closed loop through the past's row kernel on the past's own grid with the old noise loadings (on the
  union of the two regimes' loadings), initial shocks are columns of the world with discrete observation
  weights, a row observed with a delay keeps its map in raw age (masked below the delay, whole pieces;
  `MAP_CONVENTION`, `map_axes`).  Without a past every array, the grid's cache key and the shipped
  examples are unchanged to the digit: a solve with no past is the finite engine bit for bit (Z, maps,
  costs, evaluations, the same cached grid).
- The engine.  `SpectralFiniteSolver(model, past=, continuation=)` and `solve(model, past=,
  continuation=)`: the best response runs on the strip (the passive rows carry their pre-zero part, the
  excluded agent's own pre-zero actions being history; the map gains the band; the projection the old-
  shock segments and the point conditions; the cost is additive over the shock families; the FOC system is
  assembled densely over the columns of the world).  `continuation` is a converged `StationaryResult` of
  the new model at the past's window, `"stationary"` (solved on the fly) or `"end"` / None (the game ends
  at T): with one, every map is frozen at the stationary map on a buffer [T, T + L], the closed loop runs
  over the whole domain, the first-order condition of every date integrates its continuation to T + L
  through the envelope responses (the agent's own reaction off everywhere, the frozen buffer's included:
  with the buffer's own reactions in the FOC the same-model identity fails by 1e-4 on [T - L, T]), and
  the agent's world is the buffer's closed loop.  A settled transition solves the infinite problem
  exactly: Chapter 3 as its own past and continuation returns its stationary maps in one best response
  and keeps every kernel K(t - s) on every node to 2e-9 at 16 nodes (the band, the last window and the
  buffer included), `examples/ch1_delayed_finite.yaml` (a control lag and a delayed row, 56 pieces) to
  2e-9 at 8 nodes, Chapter 3 with its own control lagged 0.5 (in a loss cross term, or as the only
  coercive term) to the grid's closed-loop floor, and the two-firm Chapter 5 market (tau = 1, L = T = 2,
  5 nodes) to 2.3e-4 against its own floor of 1.4e-4 to 2.5e-4.  Both options are recorded in
  `res.solver_kw`, so `refine()` and `stability()` rebuild the engine with the same past and
  continuation.
- The guards.  `res.settled` is the largest relative distance of any map on [T - L, T] from the frozen
  stationary map (with driven means, of the mean paths at T- from the continuation's means as well);
  the `settled` row of `diagnose()` flags "TRANSITION NOT SETTLED by T - L: raise horizon.window" above
  `settings.settled_tol` = 1e-4, the closed loop's decay over a unit of t (1e-2 on Chapter 3), not the
  grid's floor (the same-model identity sits at 6.3e-7 at 12 nodes; the Chapter 3 precision change at
  4.8e-4 at T = 6, 2.7e-6 at T = 9, 0.58 at T = L).  The `past window` and `continuation window` rows echo
  their window tails.  `res.representation_parts[agent]` = {"interior", "band tip", "last window",
  "buffer"} says where the resolution guard's error sits (the band's collapsing tip and the last window
  are geometry, not resolution) and the resolution row's flag names them.  The second-order check's form
  includes a past's initial-shock columns, each under the point form of the line s = 0 that `expected_cost`
  integrates them with (`_loss_form(agent, init=True)`, the engine hook `_init_mass`), so a saddle on the
  point weights is seen (they were a zero direction of the form before); without a past the form is
  unchanged, and the Kyle-Back trader's flagged direction, with no weight on the point weights, keeps its
  raw curvature to 1e-7.
- The model file and the API.  `horizon: {kind: transition, window: T, nodes: n, discount: rho, past:
  {model: old.yaml | an inline stationary model, initial: [...]}, continuation: stationary | end,
  stationary: {window: L, nodes: m}}` compiles to the spectral finite engine with the past solved on the
  fly (a relative `past.model` path is taken from the model file's directory by `load()` and the CLI) and
  the continuation solved at `stationary.nodes` (default horizon.nodes; its window must be the past's);
  the keywords `solve(model, past=, continuation=)` override the file's blocks; `Model.validate` gains
  `_check_transition` (kind transition needs a past block; the other kinds refuse the blocks; continuation
  is 'stationary' or 'end'); `ModelBuilder.transition(T, nodes, past=, continuation=, discount=,
  stationary=, unit=)` mirrors the file; `noisestate validate` prints the transition's structure and
  `noisestate solve` handles the kind.  `noisestate.transition(old, new, T, nodes=12, **solve_kw)`
  (`noisestate/transition.py`) solves the old model (or takes its result), the new model's stationary
  equilibrium and the transition, starting from the new stationary maps read at every node's age
  (`solve(start="stationary")`, new; `solve()` keeps `start="zero"`), and returns the result with
  `res.past` and `res.stationary` attached; the file form and the helper equal the keyword form bit for
  bit.
- The result.  `TransitionResult(TriangleResult)`, kind "transition", from every solve with a past:
  `res.past`, `res.continuation` (`res.stationary`), `res.shocks` (the channels then the initial shocks,
  `res.kernel(name, "v0")`), the kernels on the band through `kernel()` / `evaluate()` (s < 0),
  `res.times` and `res.loss_path[agent]` (E[loss(t)] at every time node of [0, T] and the buffer: the
  variance part by row quadrature over every shock alive, the mean part when driven; its discounted
  integral over [0, T] is `res.costs`, which keeps its meaning, to 3e-14 on a constant path and 1.6e-6 at
  12 nodes on the regime change's transient), `res.excess_costs[agent]` = int_0^T e^{-rho t} (E loss(t)
  - the new stationary flow) dt (finite at rho = 0; empty when the game ends at T), `res.old_flows`,
  `res.new_flows`, `res.belief_error(agent, name)` (the variance of the agent's estimation error of a
  quantity at every time node: its kernel minus the projection on the agent's seen rows, one Gram per
  date, the pre-zero increments and the initial shocks' point observations included; the Kalman variance
  of the one-agent prior start to 3.4e-6), `res.cost_parts[agent]["continuation"]` (the buffer's cost,
  reported apart).  `plot()` draws the kernels against the shock time from -L with the band shaded, a
  row of E[loss(t)] with the old and new flows as lines, a row of belief-error variances and the mean
  paths.  `to_dict()` carries the past's and the continuation's provenance (kind, model, parameters,
  window, nodes, breakpoints, convergence, window tail, costs, means; no kernels), `times`, `loss_path`,
  `excess_costs`, `old_flows`, `new_flows`, `settled`, `representation_parts`, the initial shocks'
  kernels and `map_init_time`; `engine` and `grid.kind` are "transition".
- The means.  Mean paths start from the past's constant means (`State.initial` is None when not given, so
  `initial: 0` overrides a nonzero past mean; without a past an explicit `initial: 0` now round-trips
  through `to_dict()`), lagged reads before zero return the old constants, and with a stationary
  continuation (a `NotImplementedError` before, as with a past shorter than T) the system is built on the
  time line (`SpectralFiniteSolver._mean_system_line`; the construction on the line s = 0,
  `_mean_system_diag`, is kept where that line reaches T and the two agree to 1e-15): the dynamics
  through a one-dimensional Volterra operator on the time nodes (`SpectralCompiled.mean_volterra`), each
  control's condition as the per-atom operators of `_foc_operators` (the envelope responses) applied to
  the embedded mean of Q zeta + q and read on the age-0 line (`mean_line0`; the birth of a shock at t,
  its continuation running to t + L through the buffer's frozen maps), the mean of a lagged atom the path
  at t - lag (`mean_read`), the paths on the buffer frozen at the continuation's stationary means.
  `TriangleResult.mean()` reads a strip's path on the age-0 line.  Chapter 3 with a target -2 X as its
  own past and continuation keeps the stationary means on every time node to 4.7e-10 at 16 nodes, the
  mean cost T times the stationary mean flow to 1e-11; a target moving from 1 to 2 has X's mean rising
  monotonically from the old 0.8662 to the new 1.7324 while D1 jumps at 0+ from 1.7989 to 4.1115 and
  falls to 3.5979, the means settling more slowly than the maps (1.9e-4 against 6.3e-7 in `settled`).
- Sweeps.  `sweep` takes "horizon.window" as the parameter (the window of a stationary model, the horizon
  T of a finite one or of a transition, each point warm-started from the previous maps read on the new
  grid and the stationary maps beyond it, `SpectralFiniteSolver.warm_maps_from`: 8 evaluations against
  21 at T = 9 on the precision change) and, on a transition model, solves the past once for every point.
- Examples and tests.  `examples/ch3_precision_change.yaml` (player1's precision 3 to 10, T = 6, 12
  nodes, 9 s: excess costs 0.024027 and 0.028619, E[loss(0+)] of player1 0.43506 against the old flow
  0.42895 with the state's variance continuous at zero, settled 4.8e-4 with the guard firing) and
  `examples/kyle_back_prior.yaml` (the Kyle-Back market started from a prior V ~ N(0, Sigma0) seen at
  once by the insider, the market maker from the prior variance, the game ending at T = 2, eps = 0.1:
  lambda(0+) = 0.658872 at 8, 12 and 16 nodes, the market maker's belief error falling monotonically to
  0.132 Sigma0; the price is a martingale with a constant impact, so lambda^2 sigma_Z^2 T = Sigma0 -
  Sigma_T is the validation: 0.614143 against 0.614142 at eps 0.2 and 0.658872 against 0.658865 at 0.1 at
  12 nodes, pinned to 1e-4, 3e-5 and 6e-5 at 0.05 and 0.03; Back's eps = 0 limit is sqrt(Sigma0/T)/sigma_Z
  = 0.7071 at T = 2 and the eps sweep converges to it, 0.614, 0.659, 0.683, 0.692, 0.697 at 0.2 down to
  0.02, eps 0.05 and below warm-started only; the trader's second-order check reports NOT A MINIMUM,
  -0.0021 at 8 nodes and -0.0018 at 12 at eps 0.2, -0.0102 at 12 at eps 0.1 relative to the form's largest
  curvature, the prior column's (-0.0098, -0.0128 and -0.078 relative to the flow map's own), passing at
  eps 1: a correct
  report on the discrete objective, the strip quadrature's error on the D P cross term along the line
  s = 0 at a small trading cost, present with or without the prior, not vanishing with nodes, a known
  limitation of the quadrature and not a saddle of the market; README, Transitions and Limits).  The
  tests: `tests/test_transition.py` (the identities,
  the prior start against the discounted Riccati and Kalman closed form, the regime change with its
  12-node costs 2.55448876 and 2.55908247 pinned at 1e-6 and past=Model equal to past=StationaryResult
  bit for bit, the old noise loading on a channel the new row drops, the validation errors),
  `tests/test_transition_api.py`, `tests/test_transition_result.py`, `tests/test_transition_means.py`,
  `tests/test_transition_examples.py`; the same-model fixed points from zero and the 8-node delayed
  identity run under NOISESTATE_SLOW=1.
- `extras/test_cpp_cascade.py` sets `rho = 0` on the two-trader model it builds from
  `examples/ch4_kyle_back.yaml`: its reference (`ch4_N24_L8_e0.2_r0_q1_g1_1.json`) is the undiscounted
  case and the example ships with `rho: 0.5` since 0.2.2, which made the script fail by 0.17 on the
  kernels.
- `noisestate.Settings`: the tuning constants (Anderson memory and iterations, the singular-system
  and mean-system condition thresholds, the projection ridges, the second-order tolerance and the
  dense/Lanczos switch, the result's resolution, window-tail, refinement and stability thresholds
  and the stability budget) are one frozen dataclass in `noisestate/settings.py`, each with its
  default and meaning (README, Settings).  `settings=` on `solve()`, `sweep(solver_kw=...)` and the
  engine constructors takes a `Settings` or a dict of the fields to change; the changed fields are
  recorded in `res.solver_kw` and the payload's `options.solver`, so `refine()`, `stability()` and a
  re-solve from the payload keep them.  Every default is unchanged, and the older class-attribute
  names (`EngineBase.FOC_RCOND`, `BaseResult.STABILITY_MAX_EVALUATIONS`, `SpectralFiniteSolver.MAP_RIDGE`,
  ...) remain as aliases of the same fields.  Two literals now read from it under their own names:
  the cell engine's dense/Krylov switch (200 unknowns) and LGMRES tolerances, and the stationary
  projection's ridge (1e-14).
- Mean paths on the finite engines.  Targets (linear loss terms), constant drifts and the new
  per-state `initial` value (`X: {drift: ..., noise: ..., initial: 1.0}`; zero by default, a number
  or a parameter expression, refused in a stationary model, which has no initial time) move the
  means of the states and controls on a finite horizon, deterministic paths on [0, T] that the
  finite engines now solve at the end of every solve, `diagnostics=False` included.  The spectral
  engine carries a path on the time nodes of its triangle as a kernel constant in shock age, on
  which the kernels' own operators restricted to the line s = 0 (the nodes `SpectralCompiled.diag`,
  a kernel's response to a shock at time 0) are the path's: a control's mean first-order condition
  at every time node is `_foc_operators` (the instantaneous derivative of `1/2 z'Qz + q'z` in the
  control, the discounted own lagged reads, the continuation `int_t^T e^{-rho (t'-t)} R(t', t) g(t')
  dt'` through the passive-world impulse responses, the other agents answering through their
  kernels) applied to the paths with no information constraint and the targets `q` as the driver in
  place of the shocks, the state's rows are `xbar(t) = e^{At} x0 + int_0^t e^{A(t-r)} (inputs +
  const) dr` through the same Volterra operator, and the whole is one direct linear solve
  (`SpectralFiniteSolver.mean_system`, `solve_means`, `mean_cost`), linear in the targets, x0 and
  the constants.  The cell engine solves the same system on its cells, first order in the cell
  length like its kernels (`FiniteSolver.mean_system`), so no engine rejects a constant drift any
  more (`reject_constants` is gone).  `res.means[name]` is the path on the time nodes `res.means_t`
  (every state, control and definition and each signal row's mean drift rate), `res.mean(name, t)`
  interpolates it on the spectral engine, `res.cost_parts[agent]` is `{"variance", "mean"}` with
  the mean part the discounted integral of `1/2 zbar'Q zbar + q'zbar` over [0, T] (spectral
  quadrature on the time panels; the constant theta^2 of a target is not in the model) and
  `res.costs` their sum; the summary prints the parts and the means at t = 0, T/2 and T,
  `to_dict()` and the CLI JSON carry `means`, `means_t` and `cost_parts`, and `plot()` adds the
  paths as a last row.  Exactly zero, with no solve, without a driver, so the shipped finite
  examples are identical to the release before the means to every digit.  Validation
  (`tests/test_means_finite.py`, the README's table): the Chapter 1 game with targets b1 = 1, b2 =
  -1 (`examples/ch1_mean_sweep.py`, which sweeps the signal precision) against the dissertation's
  spectral solver: Dbar1(0) and Dbar1(T/2) within 0.1% up to precision 100 (1.9e-5 to 5.3e-4) and
  Jbar within 0.1% up to precision 1 (once the target's constant b^2 T, which the reference
  includes, is set aside), 1.1e-3 and 1.3e-3 at 10 and 100; the p = 10 path within 1.4e-2 at every
  t (1.9e-3 of Dbar1(0), the largest at t = 0.075); at precision 1000 (sharp kernels: 20 nodes per
  side) 2e-3 on the paths (1.9e-3 and 1.8e-3) and 3.1e-3 on the cost.  Those gaps are the
  reference's own error: the cell engine's Richardson limits close on the spectral paths as h^2
  (4.8e-3 then 1.2e-3 at T/2), and the reference's variance cost is off by the same order (9e-5 at
  10, 2e-4 at 100, 1.1e-3 at 1000).
  One agent alone reproduces the deterministic finite-horizon LQ optimum (Riccati, solve_ivp at
  rtol 1e-12) within 1e-11 on the paths and 1e-10 on the cost, with a target, an initial state and a
  discount, for one and two states; the means are linear in the targets and the mean cost
  quadratic, with the kernels identical to the bit; with signals that carry nothing the paths are
  the open-loop Nash 10 (1 - t) to 6.5e-9; on the delayed example with targets and x0 the mean
  system's residual is 6e-16, the mean dynamics agree with solve_ivp to 2.5e-12 and the
  first-order condition is the derivative of the mean cost to 1e-11 (central differences through
  the response operator).  Existing tests: `tests/test_means.py` no longer expects the finite
  engines to leave `res.means` empty or to refuse a constant drift.  New: `State.initial`,
  `Structure.x0`, `SpectralCompiled.tm`, `Nt`, `diag`, `mean_embed`, `time_mass`, `mean_read`,
  `TriangleResult.mean`, `BaseResult.means_t`, `means_driven`; `EngineBase._foc_operators(atoms=True)`
  returns the per-atom operators as well.
- Means on the stationary engine.  Linear loss terms (a target theta on X is `[1, X, X], [-2*theta, X]`)
  and a constant in a state's drift (`drift: {X: -a, D: 1.0, const: 0.3}`; the key `const` is new,
  allowed in a state's drift only, without a lag, and reserved as a name) move the means of the states
  and controls, which were not solved.  They are now, as constants: each control's mean first-order
  condition is the kernels' condition applied to a constant path (the instantaneous derivative of
  `1/2 z'Qz + q'z`, the discounted own lagged reads, and the continuation through the DC gains
  `int_0^L e^{-rho a} R(a) da` of the passive-world impulse responses, the other agents answering
  through their equilibrium kernels), with no information constraint, and the mean dynamics close
  the system: one direct linear solve at the end of every solve, `diagnostics=False` included, no
  iteration.  `res.means` has every state, control and definition and each signal row's mean drift
  rate (`agent.row`); `res.cost_parts[agent]` is `{"variance", "mean"}` and `res.costs` their sum
  (the constant theta^2 of a target is not in the model); the summary, `to_dict()` and the CLI JSON
  carry them.  Exactly zero, with no solve, when nothing drives them, so the shipped examples'
  kernels and costs are unchanged except the Chapter 5 market, whose `-2 kappa` terms on deliveries
  and sales give it nonzero means and a mean part in its costs (the two-firm variant: prices -0.2746,
  orders 1.1400, a mean part of -0.5117 on a variance part of 4.6290).  A random walk with no inputs
  (Kyle-Back's V, the market's q) has no stationary mean and is pinned at 0, which `model.notes` says;
  a random walk with a constant drift and no feedback, and a singular mean system, are refused with a
  `ValueError`.  (The finite engines did not solve the means in this entry's commit: a constant
  drift was rejected there and a linear term noted as not solved; the entry above lifts that.)  Checked against closed
  forms (`tests/test_means.py`): one agent with a target reproduces the windowed closed form to 3e-15
  and the exact `ubar = theta a / (1 + r a (a + rho))` to the window's truncation e^{-(a+rho)L}; a
  constant drift is the shifted target (the same ubar, xbar shifted by kappa/a, the mean cost larger
  by theta^2); two agents with opposite targets sit at the open-loop Nash of the deterministic game
  to 4e-8 when their signals carry nothing (precision 1e-6) and fall monotonically toward the
  closed-loop Nash of the coupled algebraic Riccati equations as the precision grows (0.6667 to
  0.5622 at precision 1000, against 0.5570), the separation failure between them.  The window's
  truncation of the continuation integrals is what `window_tail` already reports; its flag says so
  when the means are nonzero.  New: `AgeGrid.discounted_mass(rho)`, `Structure.const`,
  `Model.constant`, `Model.means_driven`, `compile.reject_constants`, `StationarySolver.mean_system`,
  `solve_means`, `mean_cost`; `EngineBase._finish` calls a `_mean_part` hook.

## 0.3.0 (2026-09-05) — readiness fixes

- Bounding and watching a solve.  `solve(max_evaluations=N)` caps the best-response evaluations
  (Anderson mixing and the Newton-Krylov polish together, the count `res.iterations` reports) and
  `deadline=S` the wall time in seconds; at least one evaluation is made, and past either bound the
  best iterate so far comes back with `converged=False` and `res.message` naming the bound (nothing
  raises; `check()` does).  A coarse start is bounded the same way and skips its own diagnostics.
  `progress=callback` is called after every evaluation with `{"evaluation", "residual", "phase",
  "seconds"}` (phase `anderson` or `newton`, `coarse ` prefixed during a coarse start); an exception
  it raises propagates, which is how a solve is cancelled.  `diagnostics=False` skips the checks at
  the end (first-order-condition decomposition, second-order check, representation error: `res.foc`
  and `res.second_order` stay empty, `res.resolution_ok` is None, the summary says `diagnostics
  skipped`) and fills the costs only; a warm-started re-solve of the Chapter 3 game at 96 nodes takes
  half the time.  The bounds and `diagnostics=False` are recorded in `res.solve_kw` and not inherited
  by `refine()`; `sweep(solve_kw=...)` forwards all four; the CLI has `--max-evaluations` and
  `--deadline` on `solve` and `sweep`.
- The Newton-Krylov polish's inner budget holds: scipy's `newton_krylov` replaces LGMRES's outer loop
  by the Newton steps, so the `inner_maxiter=15` it was given bounded nothing and a step could take
  30 fresh Krylov vectors (LGMRES's `inner_m`), each an evaluation of the best-response map; the
  call passes `inner_m=15`.  A step also re-multiplies the up to 10 directions LGMRES carries from
  earlier steps and makes the line search's evaluation, so it costs 16 to 26 evaluations (28 seen
  when the line search backtracks) against 31 to 49 before (linear test systems: 12 steps take 259
  against 439; no shipped example reaches the polish, so their numbers are unchanged).
  `stability()` makes at most `STABILITY_MAX_EVALUATIONS` (200) rounds of best responses, a number
  it passed to ARPACK as a count of restarts (up to 18 rounds each): the Arnoldi iteration is
  stopped at 170 and the power-iteration fallback gets the remaining 30, `method` saying so.  A
  best-response map that returns a non-finite value (an overflow: a lead term at a discount rate
  times the window above 709) stops the iteration with a `RuntimeError` naming the evaluation;
  every test of the Anderson loop is False on NaN, so it iterated on NaN and the next evaluation's
  closed loop died with a bare `LinAlgError: Singular matrix`.
- Finite engines: a singular best-response system raises the stationary engine's `ValueError` (`the
  best-response system of market_maker is singular: ...`) instead of falling through to a
  least-squares solve that returned a zero strategy as a converged equilibrium with cost 0, a clean
  `check()` and `second_order ok` (Kyle-Back with the market maker's `[1, P, P]` term dropped, on
  both finite engines).  The kept unknowns are LU-factored and a reciprocal condition estimate below
  `EngineBase.FOC_RCOND` (1e-10) is singular: singular systems sit at 1e-16 or 0, the worst regular
  finite-engine system in the tests at 2.6e-3, and the results are unchanged.  The cell engine's
  Krylov branch (above 200 unknowns) probes each control's block instead, and its non-converged
  message names the singular cause.  A quadratic only in a lagged read of the control (`[r, D@0.5,
  D@0.5]` without `[r, D, D]`) is singular on the finite horizon as well: the control is free over
  the last lag.  Validation warns (`UserWarning`) when a control has no strictly positive quadratic
  term in its own current value, or in a lagged read of it for a non-myopic agent, saying why the
  best response is then usually singular; the Chapter 5 firms' prices, penalised through a lagged
  read, do not warn, and no shipped example does.  The message lives once
  (`engine.singular_system_message`) and says "within tau of the window's edge".
- Finite spectral engine: the triangle's panels are closed under every lag (drift and loss lags, row
  delays), not the row delays only, so a lagged atom on a window that is not a multiple of its lag
  (a loss term `D1@0.3` at window 1.0; `ch1_delayed_finite` at window 1.1 without the delayed row)
  solves instead of dying on a bare assertion in `map_shift`.  The compile warns with the panel and
  piece counts when the closure adds panels: the kernels kink at `T - k tau`, so such a window costs
  about twice the panels and four times the pieces (the delayed example at window 1.1 takes 120 s
  against 3 s at 1.0, 8 nodes).  `TriangleGrid.breakpoints` no longer merges a short trailing panel
  (only a round-off remainder is dropped): a delay of 0.9 at window 1 was reported as 'not a
  breakpoint', and a window off the lag by less than a quarter unit gets one more panel.  A lag off the panel unit
  is rejected with the common divisor to set as `horizon.unit` (0.25 and 0.3: 0.05), and the
  remaining checks in `map_shift` and `panel_shift` are `ValueError`s naming the lag and the panels.
- Result payload provenance: `to_dict()` (the CLI's `-o` and `sweep -o` JSON) carries the package
  `version`, the `params`, the `model` spec (`Model.from_dict` rebuilds it; the name is now `name`),
  the numeric `horizon`, the engine and solve `options`, and per agent its `controls` and `signals`
  with their `delay` and the axes of the row's map (`map_age` on the stationary engine, `map_time`
  and `map_age` on the finite engine, `map_time` and `map_shock_time` on the cell engine), with
  `map_convention` saying in words how `maps[agent][u][row]` is indexed: the finite engine stores a
  delayed row's map at the shifted time t - delay, which a plot over `grid.t` drew early.  Sweep
  rows name their `param`.  `res.solve_kw["start"]` is the option string (it held the starting
  arrays, so the recorded options were neither JSON nor a repeat of the solve);
  `solve(**res.solve_kw)` repeats a coarse start.  `res.seconds` is stamped after the diagnostics
  (12% under on Chapter 5).  `noisestate --version`.
- Tests: both finite engines against the closed form of a discounted one-agent finite-horizon problem
  (`tests/test_finite_discount.py`: discounted Riccati equation, Kalman filter, closed-loop impulse
  responses) where the only discounted finite test asserted `cost > 0`: the spectral engine at 12
  nodes per side matches the cost to 2e-8 and the kernels to 6e-5 at rho = 0 and 0.5, the cell
  engine's error halves from 48 to 96 cells and its Richardson pair is within 5e-4.  The Chapter 1
  references (`tests/refs/ch1_spec_p3_p3.txt`, `ch1_grid_N160.json`), read by no test, are read by
  `tests/test_ch1_refs.py` at lags >= 0.1 with tolerances at their own error (2e-3 to 4.9e-2 against
  `spec_ch1`, 3e-3 to 3e-2 against the grid solver, which the package is closer to than `spec_ch1`
  is on every control kernel); the README's Chapter 1 row claimed kernel agreement with `spec_ch1`
  to 1e-3, which was false, and the pinned cost 0.39690577 is now stated as the package's own
  converged value (the cell engine's Richardson pairs (80, 160) and (160, 320) give 0.396956 and
  0.396918, closing on it as h^2; the dissertation's solvers report 0.39665 and 0.39657).
- Documentation.  The README states the exception contract (`ValueError` a model problem,
  `TypeError` a wrong argument, `NotImplementedError` a feature an engine lacks, `RuntimeError` and
  `ConvergenceError` a solver problem; a solve that does not reach `tol` returns `converged=False`
  and does not raise) and the CLI's exit status (0 converged, 1 not converged, 2 a usage error or an
  error the package raises).  Limits says the finite engines have no initial state distribution and
  no terminal cost (the finite-horizon Kyle-Back model is outside the grammar; a state with empty
  `drift` and `noise` is carried as zero) and the Kyle-Back validation row is marked as the
  stationary variant.  Stale claims fixed: the Chapter 5 example solves in 6 s at 4 threads (the
  table said 19 s), Install states Python >= 3.10, the model-file listing shows the shipped
  `rho: 0.5`, the sweep `change` is the action kernels on the finite spectral engine, `ridge` is gone
  from the refine/stability note, `ModelBuilder.finite()`'s fields are stated.
  `extras/patches/README.md` points at `extras/patches/` and `examples/make_ch5_cycle_market.py`
  cites the package's Chapter 5 reference instead of dissertation-tree files; `solve()`'s docstring
  names `start` (not the removed `method`) and `stability()`'s lists `method` and
  `fixed_point_residual`.
- CLI: a missing or unreadable model file, bad YAML and an output path that cannot be written exit 2
  with `error: ...`, as the exit-status contract says; they raised through `main` (a traceback and
  exit 1, the code of a solve that did not converge).  The `diagnostics=False` test counts the best
  responses a solve makes (none after the fixed point) instead of timing it on a shared CPU.
- Version 0.3.0: `solve()` takes `max_evaluations`, `deadline`, `progress` and `diagnostics`, the
  payload's model name key is `name` and `res.solve_kw["start"]` is a string, so a minor bump rather
  than a patch.  `__version__` (the payload's `version`, `noisestate --version`) is
  `pyproject.toml`'s when the package is imported from a source tree (a checkout on the path, an
  editable install bumped since it was installed), else the installed distribution's: this branch
  reported the installed 0.2.4 on every payload it wrote.  `tests/test_payload.py` checks it against
  `pyproject.toml`.
- Stationary lead term: the exponential of the discount is taken within the lead only (it overflowed
  at a discount rate times the window above 709 and made every first-order condition NaN); a lead
  whose past flows outweigh the current one by more than 100 (exp(rho tau)) is announced at compile.
  The Newton polish reports scipy's "Jacobian inversion yielded zero vector" as non-convergence with
  the Anderson iterate instead of raising a ValueError, while a ValueError from the map itself (a
  singular best-response system) still propagates.

## 0.2.4 (2026-09-04) — second review round

- Lead atoms are accepted only in a loss cross term with the agent's own current control; a led
  quantity squared or a lead on a control gave a wrong equilibrium (cost off by 2x) and is now
  rejected with the lag rewrite suggested.
- A built `Model` is single-sourced: `params` is read-only, `with_params()` / `with_horizon()` make
  new models; the plain solve, `sweep`, `refine` and `stability` cannot disagree on the model.
- Ties compare the dynamics of private states and whether a row's noise channel drives a state;
  agents with different private-state parameters no longer pass as tied.
- Validation: lags, delays and leads must be below the window on every engine (the cell engine
  solved a lag beyond the horizon silently, the spectral engine crashed on a delay equal to it);
  `unit_range` at most the window; integer `nodes`; a control absent from its owner's loss is
  rejected again; empty agent blocks; `naive_observers` names; warm starts of the wrong shape on
  every engine (one check on `EngineBase`).
- Guards: `refine` measures cost changes against the largest cost and gives no verdict on the
  first-order cell engine; the sweep `jump` flag compares the change per point with the sweep's
  median (geometric sweeps are not flagged); sweep warm starts require the same grid object;
  `second_order` and `stability` report whether their eigensolver converged and which method
  produced the number.  The singular-system message names lagged-only own quadratic terms.
- CLI catches every model-level error class.
- Simplification: `solve()` has one outer method (Anderson mixing, then a Newton-Krylov polish);
  `method`, `pre_iterations`, `pre_tol`, the sweep `predictor` switch, the `L`/`T` aliases of
  `window`, the `Result`/`SpectralResult`/`FiniteResult` aliases, `sweep.result_to_dict`, the
  `.npz` output, the `ridge` constructor option (now a class constant) and the tuning arguments of
  `stability()` are gone.  The two iteration variables stay: raw maps stall on the delayed
  Chapter 1 finite model where action kernels converge.
- Second-order check on the stationary engine: a negative direction is re-evaluated, zero-extended, on
  a window longer by two lags under the same maps (one operator build); positive there means the
  windowed objective's truncation at the edge, reported as `embedded` with a `window edge` note.
  The Chapter 5 firm's negative curvature was that truncation (the direction sits on the last panel,
  is window-invariant in the mass norm, turns positive when embedded in a wider window, and the
  untruncated Hessian is positive definite); the example and the two-firm variant pass with an edge
  note, the non-convex models stay flagged.
- Finite engine: a lagged atom in a loss (`D@tau`) is read node to node through the map shift
  instead of the interpolating read, which copied one node onto a triangle piece's degenerate corner
  row (mass norm 5 to 9 instead of 1) and made the loss form indefinite on directions of almost no
  mass.  The negative second-order curvature reported for own-lag cross terms was this: it is gone
  (6 nodes, c = 0.15: -2.8e-5 to +1.9e-5) with costs and kernels unchanged to every printed digit.
  On the stationary engine the discrete objective was shown convex by three independent
  constructions of its Hessian (the delay-aligned shift is an isometry of the cost's Gram norm, so
  r > |c| suffices); the earlier note misattributed the item.
- Triangle grid: a read point within round-off of a breakpoint is snapped to it before its panel is
  chosen by the requested side; before, a read time t - a with t and a on the same panel edge chose
  its panel by the sign of an ulp (found by an agent sharing quadrature paths across pieces).
- Measured and not adopted: sharing the finite engine's quadrature paths across translation-
  equivalent pieces.  The convolution family is invariant along time at a fixed age panel, the
  continuation along age at a fixed time, and only the projection along the diagonal, so the class
  count is about 3P for P panels, not P; the saving is 15% of path memory at 4 panels and half at 8
  to 14, with the per-member application slightly slower at 105 pieces.  Patch and analysis kept
  outside the tree.
- Coarse-to-fine warm starts: `solve(start="coarse")` solves at half the nodes and interpolates
  the raw maps onto the grid (each node read from its own panel's side); fine-grid evaluations fall
  from 37 to 25 (Ch5), 47 to 20 (Kyle-Back), 21 to 9 (Ch3), 13 to 6 (delayed Ch1 finite), 25 to 3
  (delayed Ch3), with the same equilibria to 1e-10 (stationary) and the tolerance (finite).
  `refine()` now starts the finer solve from the result being refined.
- Age grid operators built from their block-Toeplitz core.  On uniform panels of width w the
  convolution tensor satisfies T[a, i, j] = K[p - q - r, alpha, beta, gamma] with p, q, r the panels
  and alpha, beta, gamma the local nodes, the offset p - q - r in {0, 1}; the correlation tensor the
  same with a factor exp(-rho q w).  The operators are now assembled from the two-piece core on one
  panel plus dense slabs for the non-uniform tail (agent work; reconstruction exact to 2e-15).  The
  Chapter 5 grid holds 20 MB instead of 131 (6.6 MB per extra discount rate instead of 66), grids
  with only uniform panels hold no dense tensor at all (the delayed Chapter 3 solve drops from 2.9 s
  to 0.33 s), operator construction is 4 to 5 times faster, and the Chapter 5 example solves in 5.6 s
  at 126 ms per evaluation.  An FFT application of the same structure to the downstream products was
  implemented, verified and rejected: 5 to 9 times slower than the dense products at 16 panels and
  3.4 times at 64, with a crossover extrapolated near 300 uniform panels, because OpenBLAS runs the
  dense products at 270 GFlop/s and the Toeplitz saving is only a factor P/8n in flops.
- Symmetric closed loop: only the modes k <= m/2 are factorised and solved; a conjugate pair's
  contribution is twice the real part of one member's.  Agrees with the general solve to 7e-16 (the
  real part is taken directly, so nothing drifts); the Chapter 5 example solves in 6.0 s.
- Memory at no speed cost: the raw correlation tensor is not stored (only its flat layout, which
  `corr_tensor` views); loss forms are released after the finishing step; the finite engine's
  conv_left, conv_right and response paths share one quadrature path with the interpolants swapped;
  the grid cache is bounded by bytes (1.5 GB) as well as count; the finite row operators are cached
  as their nonzero blocks.  Ch5 age grid 131 MB to 66 MB per discount rate, finite paths 300 MB to
  150 MB.  All bitwise.
- Third speed round, round-off-level reassociations allowed (costs within 1e-11, maps within 1e-10
  of the previous results, evaluation counts unchanged): the age-grid tensor products run per panel
  over the nonzero index ranges (the tensors are 93 to 95 percent exact zeros), the stationary map
  projection's Gram is a symmetric rank update per causal chunk, and the finite engine reads known
  kernels through the stored barycentric factors as dense products instead of sparse matvecs.  Ch5
  example 7.0 s to 6.4 s (161 to 147 ms per evaluation), delayed Chapter 1 finite 3.1 s.
- Second speed round (two agents, exact to round-off, bitwise where stated).  Stationary: the cost
  integrals as two matrix products instead of a three-index einsum (89 ms to 1 ms); the second-order
  form associated as M[u, v] = sum_k G_k' (Resp_u' G Resp_v) G_k over the responding primaries with
  channels grouped by row support and only u <= v computed (900 ms to 130 ms); the loss form
  assembled block by block and cached; no index copies before the FOC and projection solves; the
  symmetric closed loop skips the control-to-state products when no control drives a state and
  uses slices for its orbit blocks.  Finite: the line-integral operators build their output by a
  scaled CSR densification instead of two sparse products, all known kernels of an operator are read
  in one sparse product, the regular row operators and the map-independent state part of the closed
  loop are cached per (agent, row, exclusion), zero blocks are skipped exactly, and the causal block
  solve extracts its blocks without copies.  Chapter 5 example 8.5 s to 7.0 s (finishing step 1.8 s
  to 0.7 s); the delayed Chapter 1 finite example 4.3 s to 3.1 s cold, one evaluation 258 ms to
  170 ms.  Rejected with numbers: a conjugate-mode DFT restructure of the symmetric closed loop (39%
  faster but not exact to 1e-12), dsyrk Grams, dense per-piece reads of known kernels.
- First-order-condition assembly restructured (agent work, exact to round-off: Amat 1e-15, maps
  3e-12, identical evaluation counts): the regular parts of the row and projection operators are
  assembled on the nonzero (row, channel) support only and batched per delay; the projection's
  columns are ordered (node, channel) so the products need no transposes; the strict block
  triangularity of convolution and correlation in age is used to skip the zero blocks of the big
  product; the FOC operators contract through Q first (2.6 GFlop to 0.1); the map projection's Gram
  is one product and one multi-right-hand-side solve; the dense second-order form is restricted to
  the responding primaries.  Block-Toeplitz/FFT products on the uniform panels were measured and
  rejected (the structure is real to 2e-15, but the products would run far below dgemm speed and the
  non-uniform tail needs dense corrections).  The Chapter 5 example solves in 8.5 s (was 15.9 s at
  37 evaluations, 20 s at 58, and about 40 s at the start of the day).
- Anderson memory 15 with damping 0.6 on the stationary engine (was 6 and 0.3): the Chapter 5 example
  takes 37 evaluations instead of 58 (15.9 s), Kyle-Back 47 instead of 81, the delayed Chapter 3
  game 25 instead of 33, with the same maps to 2e-9; the Kyle-Back sweep still reaches a trading
  cost of 0.01.
- Finite engine: delayed rows discretised exactly.  Two defects found by an agent against the
  closed-form one-agent delayed problem: quadrature reads at a node on the top edge of a time panel
  took the next panel's bottom row (the top row of every panel was fitted without its convolution
  term), and a delayed row's map was interpolated through forced zeros inside the piece the delay
  line cuts.  The map of a row observed with delay d is now stored at the shifted time t - d, so
  every read is node to node, and one-sided reads follow the output node's side.  Representation
  error on the one-agent model 5e-2 (flat) to 7e-5, 3e-6, 1e-7, 6e-10 at 4 to 8 nodes; costs
  converge to 1e-7; the delayed Chapter 1 example's undelayed player drops from 6e-3 to 5e-10.
  `res.maps` for a delayed row is indexed at the shifted time.  The top-edge read affected every
  multi-panel grid, so undelayed finite models with lags also move, toward a fine reference (the
  delayed Chapter 1 example with its delay removed: kernels move by 7e-3 at 6 nodes per side, to
  within 3e-5 of a 10-node solve, representation error 5.5e-3 to 8e-8); single-panel models are
  bit-for-bit unchanged.  The dense second-order threshold is 4000 unknowns.
- Cyclic symmetry (`noisestate/symmetry.py`): a single tie group listed in cycle order is checked to be
  a relabelling that leaves the model unchanged (semantically, on expanded atoms); the stationary
  closed loop then solves one block per Fourier mode over the cycle, built from the representative
  agent's rows, and the passive world (one agent switched off) is a Woodbury correction on top.
  Identical to the dense solve to 1e-15 for the full world and every agent's passive world; the
  closed loop on a 6-firm market drops from 412 ms to 92 ms.  Since the states are already
  eliminated, the best-response assembly (linear in the number of firms) now dominates an
  evaluation at 3 to 6 firms, so the gain shows from about ten firms up.
- Two cases a skeptic broke after the exact-delay fix are repaired: the breakpoints are closed under
  both b - d and b + d (a finer map panel read by a coarser action panel left unseen modes and a
  singular system), and a shifted node that lands within round-off of a breakpoint is snapped to it
  before its side is chosen (delay 0.3 on a 0.3 grid).  Tests for both.
- The FOC system is assembled with one product per control over all responding controls, the
  second-order form is built densely up to 2500 unknowns (2.7 s at 1600 where Lanczos took 7.5 s),
  and tied agents share one second-order check; the Chapter 5 example solves in 20 s.
- Stationary closed loop: the states are eliminated through a cached factorisation of the
  propagator part (map-independent) and the control rows are assembled block-wise over the
  primaries a row reads; the solve is over the controls only.  Identical to the dense solve to
  round-off; on the Chapter 5 market the closed loop drops from 390 ms to 48 ms.
- Stationary engine: delayed rows and lagged reads are discretised exactly.  A kernel read at a
  lag jumps there; the duplicated breakpoint nodes now carry the two one-sided limits (the lower
  copy reads the left limit, zero at the lag; the upper copy the right limit) and a lead is the
  exact transpose of the lag, so the projection stays the adjoint of the row operator; the
  breakpoints are closed under subtraction of every row delay; the map keeps the lower copy at the
  window edge and the reduced FOC system is solved directly (no least-squares cutoff).  Against a
  closed-form one-agent delayed problem (`tests/test_exact_delay.py`) the cost agrees to 7e-11 and
  the kernels to 6e-6 (was 8e-4 and 1e-3); representation errors on delayed models drop from 2e-3
  to 3e-13; the Chapter 5 example's representation error drops from 2.3e-3 to 1.3e-5 and its firm
  cost moves by 1% (5.0702 to 5.0218 at 8 nodes per panel), which was that example's
  discretisation error.  Undelayed results are bit-for-bit unchanged.  Found by an agent working
  from a first-principles discrete gradient of the one-agent problem.
- Finite engine: the map entries a delayed row cannot identify are removed from the FOC system and
  the map projection (the stationary engine's keep mask) instead of being regularised by a ridge;
  the fixed-point map is smooth again and the delayed Chapter 1 example converges to 1e-12 in 13
  evaluations where it stalled at 1e-8.  Costs unchanged to 1e-12.
- `refine()` on the finite engine reads the fine kernel from the coarse node's side of each piece
  boundary (a one-sided read across the delay line reported a kernel change of 1.0).
- Anderson mixing detects a stall (no 30% improvement over 20 iterations) and stops with a message;
  within two decades of the tolerance the Newton polish is skipped.
- Triangle path quadrature builds its interpolation sparsely: a 12-node delay-cut grid compiles in
  seconds instead of minutes.
- The two spectral engines' best response is one function on `EngineBase`, written against a
  seven-operation kernel algebra each compiled model supplies; the FOC decomposition and the
  second-order check are now on the finite engine too (its discounted objective is a quadratic
  form at every discount).  The second-order form is built densely up to dimension 1000, which
  settles where Lanczos did not.  All recorded results unchanged to round-off.
- One outer solve loop on `EngineBase` (the three engines' copies removed; each contributes its
  result class, defaults and a `_finish`); `CompiledBase` adopts the structure once; FOC operators
  built per impulse-response set, the physical set only for the decomposition; one-pass parameter
  tracking; `res.plot(path)` on every result; caches declared where their objects are built.
- `res.diagnose()`: every check as one row; `summary()` and `to_dict()["diagnostics"]` are built
  from it, and the thresholds are class constants.
- Shared plumbing: `_seen_rows`, `_passive_rows`, `_representation_error` and the lead rejection
  live in one place; `_row_operator`/`_projection_operator` take the agent on every engine;
  `expected_cost` on every engine (`expected_loss` kept as an alias on the stationary one).
- The C++ replica, its test and reference output, the patches and the comparison scripts moved to
  `extras/` (outside the wheel); the slow tests run in a scheduled CI job.
- The map projection of the spectral finite engine uses one ridge for every time row, relative to
  the best-identified row (a ridge relative to a row's own tiny Gram regularised nothing).
- The raw-map stall on models with delayed rows is diagnosed (README, known open items): the
  jump-interpolation artefact makes the best-response action slightly non-causal, which no map of
  the rows can reproduce; the action-kernel iteration is unaffected.
- The delayed-row least-squares cutoff is documented as immaterial (costs move by 1e-7 across
  cutoffs 1e-9 to 1e-6), and the jump-interpolation floor on the representation error for
  delayed rows and lagged control reads is recorded as a known open item.

## 0.2.3 (2026-09-04) — release review

Fixes from an adversarial test pass and a code review before release.

- Stationary engine: a signal row with a positive delay crashed with a singular matrix; the map on
  a delayed row is now zero where it reads nothing within the window, and the reduced system is
  solved (equilibrium independent of the window to 1e-6).  Same fix in the cell engine.
- `window_tail` measures the change of a kernel over the last tenth of the window, so random-walk
  states and prices that track them are no longer flagged; the undiscounted Kyle-Back example is
  flagged, correctly.
- `res.second_order[agent]` (stationary, undiscounted): exact second-order condition of the best
  response on the feasible strategies; the summary says `NOT A MINIMUM` for non-convex losses.
- `refine()` and `stability()` rebuild the same engine with the same options; `Model.to_dict()`
  reflects changes made on the object; `to_dict(numeric=True)` is loadable.
- Validation: causal drifts, zero noise loadings, breakpoints ending at the window, boolean
  `myopic`, list-typed `controls`/`signals`/`loss`, parameters used before their definition,
  `ModelBuilder` accepted by `solve`/`sweep`.  The unused-parameter check runs after the
  structural checks, so it no longer masks them.
- One-agent `stability()` reports radius 0 instead of NaN; `resolution_ok` is `None` when the
  engine does not compute it (cells); `refine()` on the cell engine doubles the cells so lags
  stay aligned; a warm start of the wrong kind is an error, not a reshape failure.
- Finite engines: dead `pre_iterations`/`pre_tol` options removed; result `engine` string is
  `"finite"` like `horizon.kind`; one engine registry (`noisestate.ENGINES`).
- CLI: model errors print a message and exit 2; `--nodes 0`, `--window 0` and `--param p=abc`
  are errors.
- One `EngineBase` (`noisestate/engine.py`) holds the packing, tie fill-in and best-response fan-out
  that the three engines each carried; `maps_from_actions` on every engine that iterates on actions.
- Packaging: `LICENSE` file; `scipy >= 1.12` (the cell engine uses `lgmres(rtol=...)`); CI also
  runs at the declared floor.

## 0.2.2 (2026-09-04) — guards against misleading results

- `refine=True` / `--refine` / `res.refine()`: re-solve at 1.5x the nodes and report the change of
  costs and kernels (`res.refinement`, summary flag `NOT RESOLVED`).
- `StationaryResult.window_tail` and the summary flag `WINDOW TOO SHORT`.
- Sweep rows carry `change` and `jump` (branch-jump detection).
- Unreferenced parameters are an error; `Model.notes` and `res.cost_kind` state the conventions
  that apply (predictable part of observed controls, myopia, flow loss vs discounted cost);
  summaries print `flow loss` / `discounted cost` rather than `E[cost]`.

## 0.2.1 (2026-09-04) — review fixes

- Sweeping a `Model` object re-parametrises its source; ties require structural identity, not
  matching shapes; parameters may be expressions in earlier parameters.
- Coefficient expressions are parsed by an AST whitelist; `eval` is gone.
- One result interface (`BaseResult`: `check`, `kernel`, `to_dict`, `grid_info`, `summary`);
  `noisestate.solve` validates its options; `ConvergenceError` and `sweep` are exported; the CLI
  works on every engine.
- Lead atoms in stationary losses get their past-date term; the finite engines reject leads;
  lagged state feedback works in the action-kernel path.
- Grids and their operator caches are shared across compiles (sweeps, sliders).
- Spectral engine: delays must be breakpoints; `expm`-based propagation for defective state
  matrices.  Epsilons scale with the window.  matplotlib is optional (`noisestate[plot]`).

## 0.2.0 (2026-09-04) — stability

- Model files are validated strictly: unknown keys at any level, channels that nothing loads, and
  controls that do not enter their owner's loss are errors, not silent no-ops.
- `converged` means the relative residual is at or below `tol` in every engine; every result
  carries a `message` describing the outer-solver path, `summary()` shows it when not converged,
  and `result.check()` raises `ConvergenceError`.  No exception is swallowed on the way.
- Reported costs use exact Gram matrices of the nodal bases; the best response is now provably
  optimal against random feasible perturbations in both engines (tests/test_properties.py).
- Tied agents iterate on raw maps (action kernels need the symmetry's channel permutation).
- Property tests: optimality, equivalence of the two iteration variables, invariance to channel
  relabelling and agent order, rejection of typos.
- Performance: batched tensor contractions, cached loss operators, causal block solve in the
  finite engine, warm-started sweeps with a secant predictor.

## 0.1.0 (2026-09-03)

- Stationary engine, spectral finite-horizon engine, cell cross-check engine, CLI, examples and
  regression tests against the dissertation's chapter solvers.
