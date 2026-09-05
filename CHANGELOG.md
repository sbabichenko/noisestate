# Changelog

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
