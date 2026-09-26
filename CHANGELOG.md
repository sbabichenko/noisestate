# Changelog

## 1.1.0 (unreleased)

A model reads like its equations, in a file and in Python.

- **Equations in the model file.** `states: {X: "(D1 + D2) dt + sigma dW0"}`, `observes: "sqrt(p1) X dt + dW1"`,
  `loss: "(X - b1)^2 + r1 D1^2"`, `shocks: [W0, W1]`, `horizon: {T: T}` (noisestate.equations; docs/model_file.md).
  `load()` reads this form and the grammar alike, `model.save()` now writes the equations (`form="grammar"` for
  the old layout), `model.to_equations()` returns them, and `noisestate validate` checks them.
- **Equations in Python.** `ns.dt` and `X.d = (D1 + D2) * dt + sigma * dW0`; `ns.Agent(..., observes=...)`;
  `dW0, dW1 = ns.shocks(2)`; `ns.params(...)`; `ns.Game(states, agents, T=...)` and `game.solve(nodes=24)`.
  A drift term without its `dt` is an error.
- **Reading results.** `res.response(X, to=dW0, at=0, seen_by=player1).over(t)` follows one shock through time;
  `res.estimate(agent, name)` is an agent's estimate of a quantity as a kernel; `res.strategy(control)` is the
  action as a rule on the agent's noise-state (Remark 1.13 of the dissertation).  Every reader takes names or the
  objects of the Python form.
- **Costs include a loss's constant.** `(X - b)^2` has the constant `b^2`, which was dropped; it is now kept on
  the agent (`constant` in the file) and reported as `res.cost_parts[agent]["constant"]`, so `res.costs` is the
  expected loss.  Costs of models with targets rise by `b^2 T` (finite) or `b^2` per unit time (stationary).
- **`describe()`** writes coefficients in the parameters (`sqrt(p1) * X dt`, `sigma * dW[W0]`), not their values.
- **`solve(model, nodes=24)`**: a Numerics field given directly is laid over the numerics, in solve() and
  transition() (refused since 0.6).
- **Removed: `naive_observers`.** It used Chapter 6's naive and privy the other way round, computed neither of the
  chapter's corners, and its solves failed their own first-order conditions (it zeroed the observers' reactions in
  the impulse responses that also build the on-path world).  Monitored deviations will come back as a model's
  monitoring relation.
- Fixed: estimates, strategies and `response(..., seen_by=)` on the stationary engine (they existed on the finite
  engine only); `sweep()` of a file now loads it through `load()`, so a transition's relative past resolves from the
  file's directory; a transition's `res.extra` and payload key for the T solved on is `T` (was `window`).
- Removed: `ModelBuilder` (the Python equations replace it; examples/make_ch5_cycle_market.py is rewritten and
  compiles to the same model), and `res.strategy_kernel()`, which returned the control's kernel, not its
  strategy (use `res.kernel(control)`, or `res.strategy(control)` for the strategy).
- **"Shock" everywhere.** `model.shocks`, `res.shocks`, the file key `shocks:` in both forms, and the keyword of
  `res.kernel(name, shock=)`, `res.estimate(..., shock=)` and `res.strategy(..., shock=)` (was `channel`); the
  payload's `shocks` lists every kernel column, a transition's initial shocks included.  Files with `channels:`
  are refused as an unknown key.
- **A transition's window is its past's.**  `Transition(window=)`, `with_transition(window=)`, a transition's
  `horizon.window`, `horizon.stationary` and the CLI's `--continuation-window` are gone (the engine always used
  the past's window, and the options only checked that they agreed); `--past-window` sets it.
- The shipped examples are written as equations, `model.save()` writes short lists and numeric maps on one line,
  and definitions in a file may come in any order.
- **One way in.**  `ns.load(path)` and `ns.as_model(x)` (a Model, a dict in either form, or a path) are what every
  function taking a model uses; `Model.load` and `ns.read_yaml` are gone.  `Model(...)` builds from equations only
  (a file structure is `Model.from_dict` or `ns.load`).
- **`describe()` prints the file's equations** (`dX = (D1 + D2) dt + sigma dw0`), from the same writer as `save()`.
  A state, definition or agent changed in place on a model is now written as it is: `save()` wrote the stale source
  while `solve()` used the new value.
- **Numerics live on `model.numerics` only**; the horizon holds the economics.  Error messages name
  `numerics.unit`, `numerics.breakpoints`, ... (they said `horizon.unit`, a key that does not exist).
- **`with_signal(name, equation)`** takes the row as a file writes it, `"(D1 + D2) dt + 0.5 dw_flow"`, or a
  `Signal`; a shock it names that the model lacks is added.  The `drift={...}, noise={...}` form is gone.
- **`sweep()` and `compare()` take `solve()`'s options as keywords** (`sweep(m, "p1", values, nodes=12,
  max_evaluations=40)`); `solver_kw=` and `solve_kw=` are gone.  A sweep now applies the `tol` and `damping` of
  its numerics, which it ignored.
- **Vectors in the Python form.**  `ns.State("X", 3)`, `ns.Control("D", 2)` and `ns.shocks(3)` are vectors;
  matrices act with `@` (`X.d = (A @ X + B @ D) * dt + Sigma @ dW`), `x @ Q @ x` is a quadratic form, a vector
  observation gives one row per component, and `res.response(X, ...)` returns every component.  The model is its
  components (a saved one writes one equation per component); checked against the matrix LQG closed form
  (9e-7 at window 10).
- Fixed: `res.estimate()` and `response(..., seen_by=)` on the spectral engine failed for an agent with more than
  one control.
- Fixed: the mean paths of a transition with a lagged input and T beyond the window (the time-line mean system)
  read the lagged input at t - lag = 0 as its value just after zero instead of the history before it, leaving an
  error of order 1e-3 on every later mean (3e-3 on the delayed example's scale of 8) that shrank only slowly with
  the nodes.  The two mean systems now agree to rounding.
- `res.strategy()` works for an agent with several controls (a vector control), with the Hessian block G^DD.
- **Terminal losses are solved.**  `terminal: "q (X - b)^2"` on an agent (Python: `Agent(..., terminal=...)`) is a loss
  paid at T on the states, entering the first-order conditions as the adjoint's terminal condition
  `H^X_T = G^XX(T) X_T + G^X_T`, the mean system, the cost (its variance, mean and constant parts, discounted from T)
  and the second-order check.  Checked against the discounted Riccati closed form with `S(T) = q_T` (cost to 3.5e-8
  and kernels to 1.2e-4 at 16 nodes, exponential in the nodes) and its terminal-target mean path (1e-12).  A
  transition with a past refuses one for now.
- **`ns.Game(...)` is the Python form's constructor** (`params=` fixes the order the file writes them);
  `ns.Model(...)` is the type and no longer builds.
- **Transitions in the equations form:** `horizon: {T: 6, past: ch3_two_player.yaml}` (or `past:` a list of initial
  shocks, `settle:` for `T`, `continuation: end`); `save()` writes it that way.
- **A shock named so that `d` + its name is another symbol is refused** (a definition `dev0` beside a shock `ev0`):
  the equations could not tell the increment `dev0` from the quantity.
- The README leads with `res.response(...).over(t)`, `res.estimate` and `res.strategy`, which read the same on every
  engine; `res.kernel()` and `res.maps` are the engine-layout arrays underneath.
- **The old Python spelling is gone:** `X.drift = ...`, `Agent(signals=...)` and signal rows without `dt`.
- **The cell engine left the package** for `extras/cells.py`, where it stays as the first-order cross-check;
  `numerics.engine` is `stationary` or `spectral`, and the `cell_*` settings went with it.
- Removed: `ns.using_settings` (process-wide and not thread-safe; `solve(..., settings={...})` does the same for
  one solve), the stationary result's `expected_loss` alias (`expected_cost`), and the shims of 0.x names.
- Faster: a closed-loop time panel is solved by eliminating the primaries with no coupling inside the panel and
  solving the rest (a Schur complement), the state rows of each panel are cached, the best responses inside the
  fixed point skip the projection they discard, and scipy's Newton solver is imported only when the polish runs.
  Times fall by 30-45% on the transition and spectral examples (ch3_precision_change 5.9 s to 4.1 s,
  kyle_back_prior 0.91 s to 0.50 s) and small stationary solves halve; results change by rounding only
  (6e-14 relative on the world at most, costs to 2e-16, the same evaluation counts).  A singular panel is now
  a ValueError that names the panel.
- Leaner: the stationary engine no longer caches each loss atom as a dense N x (n_prim N) matrix, which held
  one N x N block (Chapter 5's market: peak 342 MB to 268 MB, bit-identical).

## 1.0.1 (2026-09-23)

- Lower peak memory on the stationary engine: the second-order check computes eigenvalues only
  (the lowest eigenvector only when a failed check needs it), symmetrises its form in place, and
  the first-order system is released before the checks run. The Chapter 5 example's peak falls
  from about 430 MB to 350 MB, with no change in time.
- The finite spectral engine reads multi-column kernels along quadrature paths with a batched
  product, about 5% faster on the Chapter 3 transition. Results change only in the last bits.
- The baseline regression test compares costs and kernels to 1e-11 relative (was 1e-12, costs
  absolute). The Chapter 5 example differs by up to 2e-12 between BLAS thread counts, inside its
  own fixed-point residual, which failed the weekly slow test job.

## 1.0.0 (2026-09-10) — Initial public release

Equilibrium solver for linear-quadratic-Gaussian games with private information,
following the dissertation *Noise-State Calculus for Dynamic Games with Strategic
Information* (Babichenko, 2026).

### Capabilities

- Define models in YAML, with Python expressions, or with `ModelBuilder`.
- Solve for causal linear strategies on stationary and finite horizons, including
  regime transitions and supported observation and action delays.
- Inspect equilibrium costs, shock-response kernels, strategies on signal histories,
  mean paths, and first-order-condition decompositions.
- Assess convergence, resolution, window truncation, and second-order conditions;
  refine solutions and analyse best-response stability.
- Run parameter sweeps and compare observation scenarios.
- Use the Python API or CLI, export results as JSON, and plot results with the
  optional matplotlib dependency.
- Start from seven bundled example models and a worked tutorial.

### Limitations

Models require linear dynamics and observations and quadratic running losses.
Hard control constraints and terminal penalties are not supported. Solutions are
sought within the causal linear strategy class.

Numerical convergence alone does not establish an adequate grid or lag window.
Some bundled benchmarks intentionally trigger diagnostics, and the cell engine
does not support every check. Undiscounted stationary models use the formal
average-cost equations; the stationary verification assumes a positive discount.

See [limits](docs/limits.md) for details and [validation](docs/validation.md)
for comparisons with closed forms and the dissertation's solvers.

The [pre-release development history](docs/design/pre-release-history.md) preserves
the internal 0.x notes and the work leading to 1.0.0.
