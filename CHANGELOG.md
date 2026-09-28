# Changelog

## Unreleased

### Added

- Risk-averse agents with monitored deviations and instant observations on the finite engine (Chapter 6). A risk-averse
  privy player's response to a deviation it knows solves `f^xi_t(s) + theta <S f_t, K e_xi(s)> = 0`: the seed enters its
  cost as a known linear part and the conditional covariance drops out, so no prior on the seed is needed. The term needs
  `S f_t` over the whole horizon; it is linear in the seed world's kernels and enters the monitoring solve's matrix
  (`risk.Tilt.seed_operator`), and the seed spike's own point mass adds a constant (`seed_constant`). The deviator's own
  blip continuation gets it too, so a risk-averse market maker against a risk-neutral trader is covered. Checked against
  `extras/leqg_reference.py` `MonitorGame`, a brute-force discrete game whose privy players minimise their exact
  seed-world entropic objective over their response columns (tests/refs/leqg_monitor.json): responses to 5e-5 and
  entropic costs to 1e-6 at 14 nodes; without the term the responses are off by 0.1 to 1 (tests/test_cara_monitoring.py).
- An integral's quantity may be the agent's own current control on the finite engine (a market maker's noise-trade profit
  and loss, `int (P - V) sigma_Z dW_Z`): the spike's point mass on the shock of its own instant is carried by the
  correction (`risk.Tilt.Ldelta`). Checked against `extras/leqg_reference.py` `Game(udw)` (tests/refs/leqg_ch1_udw.json):
  J to 4e-6 and D1 to 7e-6 at 16 nodes (algebraic convergence, N^-2.4). The stationary engine refuses it.

- Risk-averse agents on the stationary engine, under consistent planning: every date's self minimises the entropic
  cost of its own discounted continuation, the later selves playing the stationary strategy
  (`noisestate/stationary_risk.py`). The condition is `P_0 S f_0 = 0` on the shocks born in (-L, inf), with f_0 taken
  with the agent's own later reactions on (no envelope for this criterion). The best response adds a shift frozen at the
  current profile (a lattice Nystrom solve with FFT convolutions, GMRES on `(I - theta K_0 Sigma)`, Sigma the projector
  off what the agent has seen, and Richardson over two lattices). `res.risk[agent]` carries the date-0 self's
  conditional entropic cost averaged over the past and `theta_mu_max`; a solution whose conditional entropic cost is
  infinite is reported not converged. Checked against `extras/stationary_cara_reference.py`, a discrete stationary
  one-agent game solved by brute force under consistent planning (tests/test_cara_ext.py), and the lattice operators
  against direct quadrature on Chapter 4's wealth model.
- `integrals: [[coef, quantity, shock]]`: stochastic-integral terms `coef int e^{-rho t} quantity dW` in an agent's
  realised cost (mean zero: a risk-neutral solve is unchanged to the bit; a risk-averse agent prices them). Chapter 4's
  insider on wealth without a terminal time: `-D V + D P + eps D^2` plus `[[-sigma_V, Q, wV]]`. Checked against
  `extras/leqg_reference.py` (`Game(xdw)`, tests/refs/leqg_ch1_xdw.json): costs to 1e-7, D1 to 3e-5; and the
  discounted finite Kyle insider on wealth against `extras/kyle_reference.py` (cost "dw", rho 0.5,
  tests/refs/leqg_kyle_dw_rho.json): J to 2e-6.
- Risk-averse agents with means (targets, constant drifts, initial states) on the finite engine: the mean condition
  gains `theta <f_t, S k>` (k the cost's linear part), the entropic cost `theta / 2 <k, S k>`; the maps are unchanged.
  Checked against `extras/leqg_reference.py` (`MeanGame`, tests/refs/leqg_ch1_means.json): costs to 1e-7, mean paths
  to 1e-6.

- Risk-averse agents with a past of initial shocks (a value drawn at 0-, `horizon.past` without a window): the shocks
  join the correction's Galerkin basis as unit vectors (exact on their block). Kyle-Back with a CARA insider now solves
  (tests/test_cara_kyle.py): against a brute-force discrete reference (extras/kyle_reference.py, Richardson over
  n = 40 to 120) the entropic and expected costs agree to 3.4e-8 at 12 nodes for theta 0 to 4, the kernels to 1e-5;
  two identical insiders to 1.5e-7 (costs) at 20 nodes. The entropic cost is stationary in the insider's own map (slope
  1e-9 against 1e-3 at the risk-neutral equilibrium). Cross terms (D V, D P) needed nothing new: the risk-adjusted
  condition holds for any quadratic loss under a linear profile. docs/method.md says why the loss must be the
  realised cost (wealth), not the fundamental-valued flow.

- Risk-averse (CARA) agents are solved on the spectral finite engine. An agent with `risk_aversion: theta` minimises
  the entropic cost theta^-1 log E exp(theta C) of its realised cost. The first-order condition is the risk-neutral
  one evaluated at the risk-adjusted noise-state (Ch1 appendix, thm:risk_sensitive_appendix): the point at which the
  derivative is taken, not a belief. The engine keeps the future-shock part of the FOC kernel and adds the
  correction theta K (I - theta K)^-1 f, with K the cost kernel over [0, T]^2 in the risk-averse closed loop
  (`noisestate/risk.py`). The line integrals are exact; a Legendre Galerkin solve per time panel, with Sloan's
  iterate, handles the smooth remainder.
  - Checked against `extras/leqg_reference.py`, a brute-force discrete-time solver (log-det entropic cost, L-BFGS best
    responses), Richardson-extrapolated in 1/n from n = 50 to 400 steps (`tests/refs/leqg_ch1.json`). The limit's own
    error is about 5e-8: its value at theta = 0 against the risk-neutral engine. On Chapter 1's game at 12 nodes both
    players' entropic and expected costs agree to 2e-7 for theta = 0.5 to 1.5, and to 6e-7 at theta = 2. D1's
    response to a state shock agrees to 1.2e-5 (5e-5 at theta = 2; 2.5e-6 at 16 nodes). Near the breakdown, at
    theta = 2.5 (theta lambda_max = 0.94), 20 nodes reach 4e-7. A risk-averse player against a risk-neutral one and
    an asymmetric game (p2 = 1, r2 = 0.2, theta = 1 and 0.5) agree to 1e-7 on the costs and 6e-6 on D1.
  - The entropic cost of the solution, computed from the closed loop and K's spectrum alone, is stationary in the
    agent's own strategy. Its slope is 5e-10 at 16 nodes, against 4e-3 to 5e-2 at the risk-neutral equilibrium, and
    5e-8 with a discount and a terminal loss.
  - `res.risk[agent]` gives theta, the entropic and the expected cost, lambda_max and theta lambda_max, and
    `res.entropic_costs` gives every agent's objective. The payload has `risk` and the summary prints the entropic
    cost. `res.costs` stays the expected cost.
  - The breakdown (theta lambda_max(K) >= 1, where E exp(theta C) is infinite) raises `ns.RiskBreakdown`, a
    ValueError.
  - When the uncontrolled start is past the breakdown, `solve()` starts from the risk-neutral equilibrium and raises
    theta in warm-started steps. On Chapter 1's game this happens beyond theta = 1.1; at theta = 2.5 even the
    risk-neutral equilibrium is past the breakdown.
  - `settings.risk_basis` sets the Galerkin basis. Past, continuation, monitoring and means are refused with a
    NotImplementedError, and so is the stationary engine. See docs/method.md, "Risk-averse agents".

- Level rows: `observes: {quote: {level: P, filter: true}}` (or `ns.level(P, filter=True)`) makes the exact path of a
  state or of another agent's control information the agent filters, not only a quantity it reacts to. The row is the
  quantity's increments: drift its kernel's rate of change, noise loading its jump at age 0, both set by the
  equilibrium. This solves Chapter 6's opaque market, whose trader sees the quote but not the order flow and reads a
  quote off the rule as noise flow. Stationary engine only, without lags, delays or ties. Checked exactly: a state's
  level gives the equilibrium of the row of its increments written out, and a competitive price level that of the order
  flow it filters (`tests/test_level_rows.py`).
- A player privy to a control's owner who sees the control's level on a level row: on the path its map on the level
  reacts at once (the instant reactions include its map at age 0), while a spike of the control is no news to it, so
  in the owner's seed worlds and first-order condition its instant reaction is its loss's alone (`seed_composite`).
  With that, Chapter 6's transparent market is the same whether the trader sees the quote and the order flow or only
  the quote, being privy (checked to 1e-8), and a market with one privy and one naive trader, both seeing only the
  quote, solves (the same trader on the path; after a quote spike the privy one sells 2.5 at once, the naive one 0.84).

### Changed

- `deviation_response` follows Chapter 6's blip convention by default: after its spike the deviating player, which
  knows its seed, continues through its own response to it (D^{i<-i}), whether or not anyone else is privy. Before,
  a deviator nobody else monitored held its control after the spike (the frozen continuation), so the paths showed
  its opponents reacting while it did nothing; that path is still `continuation="frozen"`. With players privy to the deviator, `"frozen"` now gives the frozen
  spike they answer seed by seed (before, it silently gave the blip): the privy players' responses depend on what they
  expect the deviator to do next. The first-order conditions,
  and so every equilibrium, are the same under both (the chapter's lemma on blip and frozen continuations). Checked
  on a lone regulator, whose blip continuation is the full-information feedback on its own displacement.

- `solve(model, continue_from={parameter: value})`: continuation from values where the model solves cold to its own,
  each point starting from the last, the step halved where one does not converge; the path is `res.continued`.
- A warning when a player sees another's control level without monitoring it while also seeing what that control's
  owner observes (a trader seeing the quote and the order flow): such a player is privy, and built naive its reading
  of a quote off the rule is not pinned down.

### Fixed

- The monitoring iteration's Volterra solve for a frozen spike's seeds falls back to least squares when a trial point
  far from the equilibrium makes the discretised operator singular (the iteration keeps its best round), instead of
  raising.

- `deviation_response` on the stationary engine, all-naive corner: a spike of a control whose level others observe
  (`{level: P}`) now carries the instant reactions it draws, as the solve's own spike responses always did. The market
  maker's quote spike in Chapter 6's market left out the trader's same-instant order, so the inventory started at 0
  instead of 2.5. The finite engine already had it right.

## 2.0.0 (2026-09-26)

A model reads like its equations, in a file and in Python.

### Breaking changes from 1.0.1

Code and files written for 1.0.1 need these changes; nothing old is kept as an alias.

| 1.0.1 | 2.0.0 |
|---|---|
| file key `channels:`; `res.channels`; `kernel(..., channel=)` | `shocks:`; `res.shocks`; `kernel(name, shock)` |
| `ModelBuilder`; `Model(...)`; `Model.load`; `read_yaml` | the equations form (`ns.Game`, `ns.State`, `ns.Agent`, ...); `ns.load` / `ns.as_model` |
| `using_settings`; `solve(..., solver_kw=, solve_kw=)` | `Numerics(settings=...)`; `solve()`'s own keywords |
| `naive_observers` | `monitors:` (Chapter 6's monitoring relation, the other way round) |
| `res.strategy_kernel()`; `expected_loss` | `res.kernel(control)` or `res.strategy(control)`; `res.costs` |
| `with_signal(drift=, noise=)` | `with_signal(name, "equation")` |
| engine `cells`; CLI `--window` / `--continuation-window` on transitions | `extras/cells.py`; a transition's window is its past's |
| `save()` writes the grammar | `save()` writes the equations (`form="grammar"` for the old layout) |
| costs without a loss's constant | costs include it (`cost_parts[agent]["constant"]`) |
| result payload version 2 | version 3 (`channels` -> `shocks`, a transition's `extra["window"]` -> `extra["T"]`) |

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
- **Solver quality of life.** A misspelled quantity, shock, agent, control or mean fails at the call (res.response
  no longer defers it to `.over()`) and names the nearest ones; `solve(max_iter=...)` suggests `max_evaluations`.
  The repr names the failed checks ("failed: under-resolved, window too short, trader not a minimum") instead of
  "publication: not accepted", and `summary()` gives each failed check its own line.  `res.status` tells a near miss
  ("near tolerance": the residual within 10x of the tolerance) from a failure; `res.converged` is unchanged.
  `res.save(path)` / `ns.load_result(path)` bring a result back from its payload by re-solving from the saved maps
  (1 to 3 evaluations).  `verbose=True` prints every evaluation with its elapsed time and phase, the switch to the
  Newton polish, and the outcome.  The upper-case tuning constants are out of `dir(res)`.
  `res.foc_residual(agent, seed=origin)` checks Chapter 6's response kernels independently of the solver.
  Second round: every block's keys are checked before any equation is read, so `control:` for `controls:` is named
  where it is (it surfaced as an unknown name in a state's equation); `with_params`, unknown parameters in equations
  and `ns.example` name the nearest (one-edit typos included: r3 for r1); `res.response([X, D1], to=[w0, w1])` gives
  (..., quantities, shocks); `sweep()` returns `ns.Sweep`, still a list, with `.values`, `.costs[agent]`,
  `.converged`, `.results` and `.table()`; `res.kernel("X + 2 D1")`, `res.response(X - D1, ...)` and
  `res.estimate(agent, expression)` read weighted sums of unlagged quantities.
- **Chapter 6 kernel check.** A solve with monitored deviations is failed for unsettled response kernels only above
  its own tolerance (at least 1e-10), not above a fixed 1e-10 that a fine grid's rounding floor can exceed.
- **Fixes for transitions.** A parameter used only by a transition's `horizon.T` (`T: T`) no longer makes its
  stationary continuation fail the unused-parameter check (1.0.1 has the same bug).  The "settled" check now says
  to raise `horizon.T` (it said `horizon.window`, a pre-0.8 name) and, when the transition's grid and its
  continuation's differ, that the mismatch leaves a floor no T removes.
- **Risk aversion, declared (not yet solved).** `risk_aversion: theta` on an agent (`ns.Agent(..., risk_aversion=gamma)`,
  a number or a parameter) is the entropic objective theta^-1 log E exp(theta C) of the realised cost C, CARA with
  coefficient theta when the loss is minus wealth (the Ch1 appendix, thm:risk_sensitive_appendix).  It is validated
  (finite, >= 0), saved and loaded in both file forms, part of the tie signature, and listed in `model.notes`; every
  engine refuses theta > 0 with a NotImplementedError.  theta = 0, the default, is the risk-neutral model and writes
  no key.
- **Monitored deviations and instant reactions on a finite horizon** (the spectral engine, without a past or a
  continuation, which it refuses): the same algebra as the stationary engine's, the response kernels two-time kernels
  on the triangle.  All-privy tracking on [0, 2] matches the finite feedback Nash Riccati to 3e-6 at 12 nodes; the
  strategic market maker with no inventory cost is the competitive one to 1e-11.  `res.deviation_response(origin,
  quantities).over(t, s)` reads the responses.
- **Instant reactions.**  An agent can see the current level of another agent's control and react within the same
  instant: `observes: {quote: {level: P}}` (Python: `observes={"quote": ns.level(P)}`), a trader trading on the posted
  quote.  A signal is reacted to predictably, after it is observed; a level is reacted to at once, with the loading
  its loss gives (`-G^DD^-1 G^DP`).  On the path the two coincide; they differ for deviations, where a spike of the
  quote draws an order spike at once, which moves the inventory and gives a quote with no square in its owner's loss
  its curvature (Chapter 6's G^{MM,0}).  The instant reactions may not form a cycle.  Both engines (the finite one without a past or a
  continuation); with `gamma = 0` the strategic market maker of Chapter 6 reproduces Chapter 4's competitive market
  exactly.
- **Monitored deviations (Chapter 6) on the stationary engine.**  `monitors: [market_maker]` on an agent makes it
  privy to that agent's deviations (`Agent(..., monitors=...)` in Python), checked for transitivity; `model.privy(i)`
  lists the privy set.  The players privy to a deviation respond to it through response kernels fixed by their own
  first-order conditions (sequential rationality), the deviating player resuming play after the blip; everyone else
  filters it; every agent's first-order condition sees its deviations answered that way, while the path is built as
  always.  `res.deviation_response(origin, quantities).over(ages)` reads the responses.  The all-privy tracking game
  reproduces the feedback Nash gain 1/sqrt(3) (4e-9), and a privy market maker leaves the Kyle-Back trader a positive
  profit (0.5).  Without `monitors` a model is the all-naive corner, as before.
- **Removed: `naive_observers`.** It used Chapter 6's naive and privy the other way round, computed neither of the
  chapter's corners, and its solves failed their own first-order conditions (it zeroed the observers' reactions in
  the impulse responses that also build the on-path world).  Monitored deviations return in this release as a model's
  monitoring relation (`monitors:`, above).
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
  transition that ends at T takes one too: on a prior's column (read on the line s = 0 at T) against the Kalman
  filter from P(0) = P0 with S(T) = q_T (cost 5e-8 at 16 nodes, 9e-10 at 20), and on a stationary past's band
  against the closed form started at the stationary filter (within the transition's own floor, 6e-7 at 12 nodes).
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
