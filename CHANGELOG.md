# Changelog

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
