<!-- Design record, copied verbatim from the transition stage's working directory (transition/plan.md), 2026-09-05. -->

# Transition engine: solving from a given stationary past — plan

Date 2026-09-05.  Inputs: the three design documents in this directory (grid, equations, api), Sam's
reduction of the past object, and the rank check below.

## What is settled

**The past is a set of loadings, not a model.**  What the new problem needs from the time before zero is
the dependence of the physical state and of each agent's information on the past shocks: for every
state and every agent row (and every control that a lagged drift or loss reads across zero), its
kernel as a function of shock age on [0, L], per channel, plus the old constant means.  A
StationaryResult already is this.  A hand-built prior is the same object with point loadings.  The old
model's controls, losses, discount, ties and myopic flags never enter; validation reduces to matching
channels, states and rows by name, and old delays lying on the new unit.

**The unknowns do not reduce.**  The world's post-zero loading on old shocks, Z(t; s) for s < 0, is a
genuinely two-dimensional object.  In the same-model case it equals K(t - s), and the family of
translates of a stationary kernel has no small sufficient statistic: on the shipped Chapter 3 model the
translate matrix (216 x 96) has numerical rank 26 / 61 / 75 / 87 at relative tolerances 1e-3 / 1e-6 /
1e-8 / 1e-10; Kyle-Back 24 / 57 / 74 / 86.  This is the forecasting-the-forecasts-of-others regress:
an agent's optimal weight on its pre-zero observations is a free function of their age, determined by
the same first-order condition and projection as its post-zero map.  So the band of nodes above the
diagonal stays, at today's node density.

**Domain.**  Coordinates (t, age) with one shared breakpoint sequence for both axes (the grid and the
equations designers agree; the shock-time rectangle of the api designer is rejected because its delay
lines are diagonals, which loses node-to-node lag shifts, the defect fixed in 2f4aaf3).  The region is
the rectangle [0, T] x [0, L]: below the diagonal today's triangle (unchanged nodes, unchanged order,
truncated at age L for t > L exactly as the stationary engine truncates), above it the old shocks
(t < age <= L, present only for t < L).  Every shock is forgotten past age L on both families; this is
what makes the same-model identity exact.  The extra pieces are one upper triangle of [0, L]^2,
independent of T.  Priors and a known initial state are discrete columns on the diagonal at t = 0
(the impulse-column machinery), never a narrow band.

**Continuation.**  For dates near T the continuation integral runs past T through the new model's
stationary equilibrium: a known affine tail built once from the new StationaryResult (kernels and
impulse responses at age t' - s).  Old shocks are gone by then (T >= L).  Under `continuation: end`
the game ends at T as today.  A `settled` guard compares the kernels on [T - L, T] with the new
stationary kernels.

**Means.**  The kernel solve stays mean-free.  The mean layer starts from the past's constant means
(a per-state `initial` overrides), pre-zero lagged reads return the old constants, and beyond T the new
stationary means close the tail.

**Best response on the band.**  The excluded agent's own pre-zero actions are history and stay in its
passive world; only its post-zero actions are switched off.  Its pre-zero passive rows are the past's
rows.  Its map gains the upper region g(t, b), b in (t, L]; the first-order condition, the projection
(the Gram over seen rows with the pre-zero part read from the past loadings) and the cost (additive
over shock families) extend to it with the same operators plus a known-past segment on every line path
that crosses t = 0, read from the past's one-dimensional age grid.

## Stages

**Stage 0, preparation (about half a day).  Recommended before any of the rest.**
The transition lands on finite_spectral.py, the most intricate file, which just absorbed the mean
layer.  (a) Document the EngineBase hook contract: one docstring per hook an engine overrides
(_identified, _solve_foc, _project, _impulse_responses, _lead_term, _mean_part, _diagnostics,
interpolate_maps, maps_from_actions, expected_cost, map_axes): inputs, shapes, what the base assumes.
(b) Gather the fifteen tuning constants (SECOND_ORDER_TOL, FOC_RCOND, RESOLUTION_TOL, WINDOW_TAIL_TOL,
ANDERSON_M, SECOND_ORDER_DENSE, MEAN_RCOND, MEAN_ZERO, LEAD_WEIGHT_WARN, STABILITY_*, ...) into one
documented table with a supported override (a `settings` argument or module-level object), leaving
every default unchanged.  (c) Split Model.validate (156 lines) into named checks.  Suite unchanged.

**Stage 1, the band with the game ending at T (2 to 3 days of agent work, the hard stage).**
- `Past` object: from a StationaryResult, from a stationary model/dict/path solved on the fly, or from
  a hand-built list of initial shocks `{"loads": {state: coef}, "rows": {"agent.row": coef}}`; carries
  loadings per state/row/control on its age grid, the direct noise loadings E, constant means, L, and
  provenance.  Validation as above.
- Grid: TriangleGrid gains the upper pieces (mirrored Duffy triangles on the diagonal, rectangles
  above) with today's nodes and order untouched; truncation at age L.
- Compiled: known-past reads on every line path crossing t = 0 (a 1-D LinePath on the past's grid,
  cached forcing); read/map_shift/atom_op for reads before zero (old control and state kernels);
  _state_part for old shocks from the initial condition e^{At} K_old(-s) plus the Volterra blocks;
  initial-shock columns as impulse columns with instantaneous row entries.
- Engine: passive rows with the pre-zero part; the map's upper region in _identified; FOC, projection
  and cost over both families; _mean_part with the old means as the initial condition.
- Tests, in this order: (1) zero past, bit identity of Z, maps, costs and evaluation count with the
  finite engine on ch1_two_player_finite and ch1_delayed_finite; (2) same-model stationary past on
  Chapter 3: kernels equal K(t - s) on every node in the interior (t < T - L) to spectral accuracy, the
  end effect confined to the last L; (3) one-agent prior start against the closed form (the discounted
  Riccati benchmark of tests/test_finite_discount.py extended with a prior variance P0 and a Kalman
  filter from P0; the informed variant with row loading 1 gives P(0) = 0); (4) a prior on a state with
  no row loading equals sigma0 times the finite engine's s = 0 column; (5) a regime change (precision
  step on Chapter 3): state kernels at 0+ equal the old kernels at age -s, and `past=Model` equals
  `past=StationaryResult` bit for bit.

**Stage 2, the stationary continuation (about 1 day).**
- Solve the new model's stationary equilibrium first (or take it); build the tail once (known forcing
  on the right-hand side, a Gram-like operator on the agent's own action for dates within L of T).
- `continuation: stationary` becomes the default of kind `transition`; `end` keeps stage 1.
- Tests: the same-model identity exact on the whole region including the last panel; kernels on
  [T - L, T] equal the new stationary kernels; the `settled` guard fires when T is too short.

**Stage 3, API, results, documentation (about 1 day).**
- Model file `horizon: {kind: transition, window: T, past: {...}, continuation: stationary, stationary:
  {window, nodes}}`; `solve(model, past=...)`; `transition(old, new, T, **kw)` starting from the new
  stationary maps; `ModelBuilder.transition`.
- `TransitionResult`: kernels with s < 0, `loss_path` (E[loss(t)] by row quadrature), `excess_costs`
  (the discounted excess over the new stationary flow), `belief_error(agent, name)`, `settled`,
  diagnose rows; `res.costs` keeps its meaning; plots with the band shaded and the old/new flows as
  lines; to_dict provenance without the past kernels; sweep over T or over the change size with a warm
  start.
- Examples: a precision change on Chapter 3; Kyle-Back with a prior on the value (insider informed,
  market maker with the prior variance), pinned by node convergence and a documented comparison with
  Back's initial price impact; README section, validation rows, Limits, CHANGELOG; release 0.4.0.

## Process

One implementing agent per stage in a worktree branch, full suite before each commit, a skeptic over
`git diff master..HEAD` with the shipped examples compared against master, a repair commit, then merge.
Agents run one at a time (shared CPU).  Bound every structured output (the design panel lost three
agents to the output ceiling and to a usage limit).  Stage 1 is large enough to split into two commits
(grid and operators; then the engine and tests), each reviewed.

## Risks

- Delay-cut pieces above the diagonal: the old shocks' kink lines are age lines and lie on the shared
  breakpoints, but the truncation edge age = L must be a breakpoint of the past's grid too (geometric
  panels beyond unit_range must match or be resampled once, with the resampling error reported).
- The known-past reads are a new path type; their quadrature must split at the past grid's
  breakpoints or the spectral accuracy of the same-model test is lost.
- The tail operator for dates within L of T couples the agent's own action to the stationary
  continuation; its correctness is checked only by the same-model identity, so that test must be exact
  to 1e-9 before anything else is built on it.
- Cost: the band adds one L-triangle of pieces; for a delayed model at L = 3, T = 4 the node count
  roughly doubles.  Acceptable, but the transition should not be the default engine for anything.
