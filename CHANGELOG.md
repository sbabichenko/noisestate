# Changelog

## Unreleased

### Changed

- Exact shifts on the stationary engine, and its age panels chosen automatically.  A kernel read at a lag was resampled
  onto the panels, exact only where they are aligned with the lag; on a geometric tail it interpolated a function that
  breaks inside a panel, and the error polluted the whole solution (Chapter 5's market on its shipped grid: 1.9e-6 of a
  kernel's peak, 2e-6 in its unit panels, against a 576-node uniform reference; at 12 and 24 firms a first-order system
  of condition number 5e10 whose map noise stalled the fixed point).  Where a lag is not aligned, every lag is now an
  argument of the operators (the convolutions, correlations, masses and the states' propagators split their quadratures
  at the shifted breakpoints; the shifted tensors are held as their n x n blocks per node and panel pair), the spike
  responses keep a map read at a lag as a kernel of that shift, and the first-order condition is a sum over shift
  classes.  On aligned panels nothing changes (both agree to 1e-14 there).  12 and 24 firms converge (50 evaluations,
  condition 2.9e8), and the second-order form on the strategies read within the window is positive definite with the
  bespoke solver's condition number.  Instant observations, level rows, monitoring and risk aversion keep the
  resampling (no test model has them with lags) and warn where their panels are not aligned.
- A stationary model whose numerics give none of `nodes`, `unit_range`, `breakpoints` has its panels chosen
  (`age_panels.py`): unit panels through its lags, then growing by 1.5 on the unit lattice, cut at L - tau and L - 2 tau,
  12 nodes, checked by the kernels' Chebyshev tails per panel against `settings.auto_grid_tol` (1e-7, new) and cut
  where they fail; `res.panels` records it, `res.numerics` holds the grid, `sweep()` keeps its first point's grid.
  `Model.numerics.nodes` is None when not given (a solve resolves it; an engine built directly takes 16).  Chapter 5's
  example ships without a grid: 132 nodes in one round, 3.9 s against 4.5 s for the shipped 168 nodes (2 threads, the
  checks included), its kernels 2.6e-8 and its cost 3e-10 from the uniform reference against 1.9e-6 and 2.6e-8; the
  former default (no numerics) was 224 nodes and 10.7 s at 8.4e-8.  At the shipped grid's accuracy the exact shifts
  need 88 nodes, 1.8 s.  On the suite's 84 lagged stationary models the automatic grid meets its target on fewer nodes
  than the former defaults; on an expert's long unit range the exact path costs more than the resampling did (Chapter
  5 on 16 unit panels at 8 nodes: 5.9 s against 4.5 s).
- Faster and leaner, same results.  Stationary risk-averse agents: the date-0 entropic cost (`res.risk`) no longer builds
  or eigen-decomposes the n x n form of K_0 on the lattice (n = 7365 on Chapter 3's game): K_0 is block tridiagonal in
  time-ordered coordinates and Sigma K Sigma differs from it on the first two blocks only, so the cost is read off a
  block Cholesky factor (the sum over the eigenvalues row by row, without cancellation), a block forward substitution
  and Lanczos for the largest eigenvalue; the correction's convolutions reuse the lag kernels' transforms (the same
  sums as scipy's fftconvolve, to the bit); the information basis is a staircase QR where its matrix is not tall (the
  C++ port's).  Chapter 3's two-player game with theta 0.3 and rho 0.5: 209 -> 3.9 s and 3.9 GB -> 167 MB peak;
  Kyle-Back's insider on wealth (the C++ port's stat_cara_kyle) 62 -> 17.4 s and 945 -> 218 MB (2 threads).  The equilibria are
  bit-identical except where the staircase QR is used (Kyle-Back's maps move by 9e-15 relative); the entropic costs
  agree to 5e-14.  The spectral finite engine's sparse second-order form no longer copies an N x N matrix per row
  operator (the Chapter 1 regulator at T = 10 with graded panels 2.25 -> 2.20 s, 549 -> 531 MB); the stationary loss
  form is held as its nonzero blocks and the state elimination is factored in place (bit-identical).
- Faster again, results moving within each solve's tolerance (no longer to the bit).  The Anderson iteration drops the
  secant pairs from far above the current residual, which dominated its regulariser and reduced every finite solve to
  plain mixing: Chapter 1's game 15 -> 10 evaluations, the delayed game 13 -> 9, Chapter 3's transition 20 -> 16, the
  finite CARA games 16-18 -> 11-13.  The stationary best response solves its first-order system and its map projection
  by GMRES preconditioned with the factors of the last system built (refactored when that takes more than 5
  iterations), from the last solution.  A cyclic tie's closed loop is linear in the number of firms (FFT mode transforms,
  the passive world by a Woodbury correction where that is cheaper, the state forcing cached, no state elimination to
  factor when no state is driven), its followers' diagnostics and mean conditions are the representative's relabelled,
  and validating a tie no longer grows quadratically with it.  Stationary risk-averse agents go straight from the
  risk-neutral equilibrium to their theta when the breakdown measure estimated there is at most 0.5, and warm-start the
  correction's GMRES.  The Newton polish's stopping rule now implies the residual tolerance (it could stop at up to
  sqrt(n) times it, reported not converged: Kyle-Back with eight tied traders).  Interleaved medians of 3 at 2 threads
  (a loaded machine): the benchmark 53.6 -> 36.6 s, Chapter 5's market 10.4 -> 6.2 s (nodes 10: 19.1 -> 10.7), Chapter 3
  with theta 0.3 4.8 -> 2.9 s, the finite cases 20-29% faster; the gap study's Chapter 5 market at 24 firms (unit_range 2)
  64 -> 21 s and 1.4 GB -> 655 MB.  Maps move by at most 1.1e-8 (norm, relative) on the finite engines (tol 1e-8) and
  4.3e-10 on the stationary (tol 1e-10), costs by at most 5e-9; every check's verdict is unchanged.  The test record
  (tests/refs/baseline_0.4) and four cost pins were re-recorded.

### Fixed

- The stationary lead term (a loss's cross term with a quantity read ahead, `D X@-2`): its past-date convolution set both
  copies of the node at the lead's end, the next panel's first node carrying the response's value at 0, and every such
  model converged as n^-2 just past the lead (a one-agent target model's control 2.3e-3 of its peak at 16 nodes; the
  C++ port's stat_misc cost 6e-5 off).  The upper copy reads the right limit, zero: 16 and 24 nodes agree to 2e-13.

## 2.1.0 (2026-09-29)

### Changed

- `examples/ch4_kyle_back.yaml` ships on a window of 12 (was 8), where it passes every publication check: on 8 the
  `window cost` check (a solve on 1.5 L) moved the trader's profit by 6.5e-5 (0.820818 against 0.820871) and
  `require_ok()` refused the example the README prints in full; on 12 it moves it by 8e-8.  The trader's cost is
  now -0.820871150 (the market maker's, which omits V^2 and grows with the window, -9.8835).  The Python form in
  `examples/expr_examples.py` follows; tests that compare with records made on 8 (the C++ references, Chapter 6's
  competitive corner, the undiscounted artefact) pin the window of 8 themselves.

- A model with instant observations (Chapter 6's markets) iterates on sequential best responses
  (Gauss-Seidel, `settings.best_responses`, default "auto"; "sequential" and "simultaneous" force either): the agents
  answer in the model's order, each against the profile the agents before it have answered, the origins of monitored
  deviations together (one coupled monitoring solve). The fixed points are the same equilibria; results move within the
  solve's tolerance. Chapter 6's markets, whose simultaneous map has complex eigenvalues beyond the unit circle, gain:
  the two-trader market at gamma = 0 from zero 507 -> 54 evaluations (its Jacobian's radius 1.12 -> 0.75), and the
  opaque market now converges from a cold start at gamma = 0.1 (59 evaluations, the continuation's point to 2e-10).
  Other models keep the simultaneous map (bit-identical).
- A model with instant observations keeps at least `settings.anderson_m_instant` (25) Anderson secant pairs: the
  transparent market (a ring of eigenvalues about 1.5-1.7 in modulus, some 55 Krylov directions deep) stalled at 15
  (310 evaluations with the Newton polish, 101 now).
- `solve(continue_from=...)` solves the points before the model's own to 1e-6 (`CONTINUATION_TOL`, or the solve's tol
  if looser): they only start the next point.
- Together, on Chapter 6's benchmark markets (median of 3): transparent 0.756 -> 0.301 s (24 nodes) and 0.707 -> 0.413 s
  (32), opaque by continuation 0.657 -> 0.435 s, two traders by continuation 2.446 -> 0.631 s, and from zero at gamma = 0
  1.583 -> 0.236 s; kernels, maps and costs move by at most 7e-9 relative on the stationary engine (tol 1e-10) and 2e-7
  on the finite (tol 1e-8). Models without instant observations are bit-identical unless solved by continuation
  (whose last point now starts from a 1e-6 path: it moves within its tolerance).

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
- Internal: Chapter 6's response-kernel iteration and the pieces around it are written once for both engines
  (`noisestate/monitoring.py`, `MonitoredDeviations`), and `deviation_response` once for both results; unused code
  removed (StationaryTilt's unconditional excess, unread attributes).  Results bit-identical.

### Fixed

- Stationary engine, second-order check at a positive discount: the form is now the discounted objective's own, every
  response to the agent's deviation weighted by e^{-rho tau / 2} at its age tau (an atom read at lag l older by l), which
  at rho = 0 is the average-cost form it was.  The unweighted form is the flow loss of a deviation made at every date,
  the past's included; the argument that a discount cannot change its sign holds only for a loss positive semidefinite
  pointwise, and a trader's is not.  Chapter 6's transparent market (gamma 0.1, window 8): the trader read NOT A
  MINIMUM at -0.062 (the equilibrium is not even a critical point of that form's objective: slope 5.8e-2 along its
  lowest direction, 2e-4 for the discounted objective); it is +0.0059, and the same market's naive trader (not privy)
  +0.0058 (was -0.054).  Checked against the discounted Riccati solution of one agent trading against its own transient impact:
  convex exactly when the weighted form is, where the unweighted form called the Riccati optimum a saddle.  Every
  stationary second-order value at rho > 0 moves (Kyle-Back's trader stays positive at every rho).

- Stationary engine: the continuation in a mean first-order condition (targets, constant drifts) is the passive world's
  discounted DC gain over all ages, its Laplace transform at the discount rate, not its integral over the window.  A
  constant is felt at every age, and the passive world (one reaction fewer than the equilibrium) can decay far more
  slowly than the equilibrium's kernels, which the `window` check measured: in a two-player tug of war on a random walk
  (conflicting targets, one row delayed, window 3; the script tests/test_expr.py drives) player 1's passive response
  was still 42% of its peak at the window's edge while the kernels were at 0.5%, the mean push 7.24 against 8.40 at
  window 12, and the costs 16% off.  Now 8.453 at window 3 and 8.4205 from window 6 on, the costs within 0.7% at 3.
  One agent's means are its closed form to round-off at any window (they were off by the window's e^{-(a + rho) L}).
  Level rows, instant observations and monitored deviations keep the window's integral, as does a passive world that
  does not decay; `res.mean_tail` then keeps the passive response's level at the window's edge and the `window` check
  reads it (`WINDOW TOO SHORT FOR THE MEANS`).  Changed numbers: every stationary model with means, by its passive
  worlds' tail -- the Chapter 5 market's costs by 7e-7 relative (mean part 6e-6); Chapter 3 with a target for player
  1 at window 3, mean X 0.866 -> 0.767 (0.7479 from window 6 on; the window's integral reached 0.753 only at 12).  The
  dissertation's numbers do not use this path (its Chapter 1 means are finite-horizon).  A transition with a
  stationary continuation adds, to the mean condition on its strip (cut at age L), the continuation's passive response
  past L (`StationarySolver.passive_tails`), so a model as its own past and continuation keeps its constant means.

- Finite engine: a signal row whose noise loads a shock that also drives a state (correlated observation noise; Chapter
  6's finite market, whose flow row loads the noise trades that move the inventory) is answered at once, so its map at
  age 0 on a triangle's degenerate corner row is not zero; the projection on the rows gives that corner no quadrature
  weight and left it at zero, which bent the raw maps' interpolant near every corner.  The reported equilibrium (built
  from the raw maps) was 3.5e-4 off the closed form at 12 nodes and converged only algebraically, with a representation
  error of 0.2 to 0.9 at every node count; the corners of such an agent's projection now carry the point condition they
  had with a past, and the one-agent model is exact to 1e-12 (the finite Chapter 6 market at gamma 0.1 moves by 8e-5
  to its converged value).  Every other model is bit-identical.  Found by the randomized campaign (extras/fuzz).
- An instant observer best-responds in the passive world closed under its own instant reaction (the map off, the
  reaction on), and its action is reported as map part plus reaction.  Before, the reaction was switched off with the
  map, so the best response was the best function of its rows alone and left a remainder no map represents when the
  seen level carries information beyond the rows (representation error 0.37; the two iteration variables settled on two
  different fixed points, neither the equilibrium).  Stationary and finite now agree with an independent discrete
  reference; the FOC operator keeps the agent's own atoms where a monitored response carries its own instant reaction
  (foc_residual 3.9e-2 -> 2.5e-12).  Chapter 6's markets, whose quote is in the trader's rows, move by at most 1e-10.
- Ties: a single tie group whose agents differ by the names of public states (a ring where each player sees its
  neighbour's state) is tied by a verified cyclic relabelling (`symmetry.find_cyclic_symmetry`) instead of refused.

- The Chapter 5 market's representation-error floor (about 1.2e-6 at 12 to 16 nodes, failing the resolution check at every
  grid): the kernels jump at L - d (a lagged read past the window is cut off), which lay inside the last geometric panel
  beyond unit_range. The geometric panels are cut at L - d for every lag; the error falls with the nodes (3.6e-7 at 8,
  7.0e-8 at 12, 6.9e-9 at 14). Every stationary model with lags and unit_range below the window gets the extra cuts.
- Stationary risk-averse agents: the continuation in theta jumped from 0.5 to 1 and could land on a spurious fixed point
  past the breakdown (the one-agent signal model at theta 1, reported not converged); past the 0.5 step it now takes the
  finite engine's steps, halves a step that lands past the breakdown, raises RiskBreakdown with `reached` where the path
  ends short of theta, and raises RiskBreakdown for a converged solution past the breakdown (was "not converged").
- The stationary date-0 entropic cost's lattices were L / 40 .. L / 160, so their error grew with the window (the gap
  between the finest two 9.3e-3 at L = 16); the step is capped at 0.2 (`ENTROPIC_STEP`), unchanged up to L = 8.

- The monitoring iteration's Volterra solve for a frozen spike's seeds falls back to least squares when a trial point
  far from the equilibrium makes the discretised operator singular (the iteration keeps its best round), instead of
  raising.

- `deviation_response` on the stationary engine, all-naive corner: a spike of a control whose level others observe
  (`{level: P}`) now carries the instant reactions it draws, as the solve's own spike responses always did. The market
  maker's quote spike in Chapter 6's market left out the trader's same-instant order, so the inventory started at 0
  instead of 2.5. The finite engine already had it right.

- A non-finite number anywhere in a model (a NaN or infinite parameter, coefficient, delay, constant, length or numerics
  value) is refused with its field. A NaN parameter used to solve to "converged, residual 0" with every check passed and a
  NaN cost; an infinite one, or a NaN discount (NaN < 0 is False), failed as a "singular" system. A coefficient whose
  arithmetic fails (1/0, sqrt(-1), log(0), an overflow, a negative number to a fractional power) is a ValueError naming it.
- An equations-form file takes expression parameters (`r: "0.2 / 2"`, `s2: "2 * sigma"`), as the grammar and
  docs/model_file.md always did; it refused them ("the value must be a number").
- A key given twice in one YAML mapping (a second `horizon:`, two agents of one name) is refused with its line; YAML
  kept the last silently.
- numerics.tol and damping must be positive (tol 0 was accepted, a negative tol reported "not converged" at the exact
  answer), max_newton a non-negative integer; the Settings fields are type-checked where they are set ("15" for a count
  failed deep in the solve; a misspelt risk_planning passed on a risk-neutral model).
- Kernel.at, res.evaluate and res.mean refuse NaN and a date past the grid, and the stationary Kernel.at an age past the
  window; the interpolant's zero row outside its domain read as a response of exactly 0 (a random walk's too). A time
  before 0, or a shock after t, still reads 0.
- An initial shock may not be named like a Brownian shock (it was accepted, and res.kernel(X, "w0") read one of the two).
  An agent in two tie groups, or twice in one, is refused.
- The finite engine's matrix-free best response (from foc_dense_max, 16 nodes on Chapter 6's finite market) refused a
  quote with no square of its own as "singular at t = 0": its time-row preconditioner took the own square alone, where the
  curvature is the orders the quote draws at once (FocOps.inst's term). The block is now C Q C' over the spikes with their
  instant reactions; the dense and matrix-free systems agree (1e-14 on the costs at 16 and 20 nodes).
- The Lanczos second-order check (above second_order_dense) sought the lowest eigenvalue with ARPACK's tolerance relative
  to itself, near 0 on a nearly singular form, and ran its budget without an answer (3042 products on a 2160-unknown
  strategy); it is now hi less the largest eigenvalue of hi I - A, settled at the tolerance of hi.
- A model's warnings are shown once per model; validate() runs three times on the way to a solve, and each warned from its
  own line (three copies finite, five stationary).
- sweep(): a point that raises names its value in the exception's notes and carries the points solved before it on
  `exc.partial`.

### Added

- `extras/fuzz`: randomized differential testing -- seeded random models across the feature families, an independent
  discrete brute-force reference, the one-agent closed form, cross-engine checks, invariances and self-consistency under
  one rule (correct within the stated accuracy, or say so); `tests/test_fuzz.py` (a smoke set in the fast suite, the
  campaign under NOISESTATE_SLOW) and `extras/fuzz/campaign.py` for larger campaigns with repros.
- `Result.check_window(factor=1.5)`: re-solves a stationary model on a 1.5x longer window and reports the relative cost
  change as the `window cost` check (`settings.window_cost_tol`, 1e-6).  It is part of `Policy.PUBLICATION`, SKIPPED
  until measured, and measured by `require_ok()` and `--require-ok`; the kernel-tail `window` check alone let a result
  be 5e-4 off in cost.
- A finite horizon long against the model's time scales is re-cut automatically (`noisestate/time_panels.py`), for a model
  with no `numerics.breakpoints`, no lags and no past window or continuation. Before the first solve its rates are read
  off its linear algebra (the agents' Kalman filters, the Nash feedback Riccati of their controls, the closed and open
  loops, the discount; `time_panels.time_scales`), and when a tanh layer of those rates is predicted to leave a Chebyshev
  tail above 1e-2 on the one panel (at 8 nodes or more) the first solve is on panels graded from both ends by them
  (`time_panels.suggest`): the one-agent regulator at T = 30 (190% off the closed form on one panel) on
  `[0, 1, 3, 7, 23, 27, 29, 30]`, the cost to 5e-10 in one solve. Every other model is solved on its one panel first,
  bit-identical where it resolves; one that fails badly (above 5e-4, or singular, and a trial finds grading helps) is
  refined. A result that fails the resolution check is refined locally: the intervals whose nodes carry a representation
  error above the tolerance are bisected, the worst first, each round warm-started from the last. Within a budget:
  `settings.auto_panels_max` (4096 unknowns, was 8000), `settings.auto_panels_memory` (1536 MB, estimated) and
  `settings.auto_panels_growth` (4: a refinement round's estimated memory against the first grid's); past it the best
  result is returned with a warning naming the breakpoints of the resolved answer, `res.panels["suggested"]`, and
  `res.sharpen()` re-solves there from the result. `res.panels` records the route, rates, history and verdict. The
  website's wedge at p = 1000, which the first version of this graded by halving in 56-77 s and 3.7 GB, takes 8-10 s and
  0.7 GB for the same kernels (5e-5 of the converged reference, the cost to 2e-9) and warns that the check's 1e-6 needs
  4032 unknowns (`auto_panels_growth = 1`: its three time-scale panels alone, 4.4 s and 0.35 GB, kernels to 1.7e-4).
  The long-horizon regulators of the limits pass (T = 5 .. 30 at p = 30, r = 0.01, noise x5; T = 100 at 8 nodes) take
  at most 34 s and 1.4 GB where they took up to 153 s and 4.4 GB; their costs are within 1e-6 of the closed forms (1e-8 ..
  1e-11 before); the six the budget stops short of the check (noise x5 at T = 5, 10, 30; T = 30 at p = 30 and at
  r = 0.01; T = 100) warn and name the grid (three of them warned before as well, at 3.7-4.4 GB).
- `RowOps.sparse()`; the finite engine's second-order form is assembled from sparse row operators, the responses' identity
  blocks skipped and M built a column block at a time (`dense_curvature_form(sparse=True)`), and `second_order_dense`
  is 8000 (was 4000): T = 30 on 10 panels, 12 nodes (7920 unknowns), solve and checks 425 s / 5.8 GB -> 48 s / 2.9 GB,
  and the check's lowest eigenvalue is exact where Lanczos had it to 1e-6 of the highest only.

- The stationary engine reports a `cost window` row (never required by a policy) when an agent's flow loss still accrues
  more than window_tail_tol of itself over the last tenth of the window (`res.cost_tail`): a loss term in a quantity that
  has not decayed by L. The window check passes a random walk (its kernel does not move), so a myopic agent leaving one
  uncontrolled reported a cost equal to L with every check passed; Chapter 4's market maker (P^2 - 2 P V) reports about
  2.1 - L and now says so.

- Risk-averse solves at small trading costs: a stalled solve, or one whose best response fails, is retried once from the
  same start with the best responses' Krylov solves tightened to 1e-6 (`RISK_KRYLOV_RETRY`); a warm-started risk-averse
  GMRES runs its restart cycles up to `foc_krylov_maxiter` steps instead of two cycles (which reported a regular system
  singular). Kyle-Back with a prior at eps 0.05, theta 1.5 now converges (CE 0.53980, as with 20 nodes). If the retry
  fails too, the message carries a second-order probe of the entropic cost, and the second-order check of a risk-averse
  agent with an indefinite loss Hessian is that probe (`"bound": "probe"`) instead of "not checked". The retry's best
  responses are proximal (`RISK_PROX`): the frozen best response's system, the expected cost's Hessian under the tilted
  measure, goes singular at theta 1.58 on that market while the entropic cost stays convex (freezing K drops its
  positive `theta Var^Q(C')`), and the plain iteration could not pass it; the proximal term, anchored at the action
  iterate so that it vanishes at a fixed point, carries the path to theta 3 (0.4953722 at theta 2 against the brute-force
  reference's 0.4953707). A solve that stops short no longer raises RiskBreakdown at its last iterate.

- Consistent planning for risk-averse agents on the finite engine, `settings.risk_planning = "consistent"` (the default
  stays precommitment, J_0 over the whole strategy): every date's self minimises the entropic cost of its own
  continuation, discounted from its date, the later selves playing the equilibrium map. The condition is
  `P_t S_t f_t^on = 0`, `S_t = (I - theta K_t)^-1` with `K_t` the continuation's kernel and `f_t^on` taken with the agent's
  own later reactions on (`SpectralFiniteSolver._responses_on`, `FocOps(envelope=False)`); `risk.ConsistentTilt` computes
  it with one Galerkin matrix per time row, and the best response adds it as a shift frozen at the profile. Checked
  against `extras/leqg_reference.py` `ConsistentGame`, a brute-force discrete game solved by backward sweeps over the
  date selves with exact conditional entropic objectives (tests/refs/leqg_ch1_consistent.json,
  tests/refs/leqg_ch1_consistent_rho1.json; tests/test_cara_consistent.py). Refused with a past, monitoring, instant
  observations or means.

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
