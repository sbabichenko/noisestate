<!-- Design record, copied verbatim from the transition stage's working directory (transition/design_api.md). -->

# Transition solver: stationary past, finite transition, stationary future

## Domain and grid
- Shock time s in [-L, T], date t in [0, T]. The s >= 0 part is today's TriangleGrid, untouched (same nodes, same cached paths, same grid-cache key when there is no past).
- The past band {t in [0, T], s in [-L, 0)} is appended as extra pieces: unit cells in (t, s) cut by the three kink families t = ku, s = -ku, t - s = ku (delay lines are diagonals there), so each cell is two Duffy triangles with Chebyshev nodes. interp, path, row_weights and mass extend over the band; node order is triangle first, band after, so every existing index is unchanged.
- An "initial" shock list is the degenerate past L = 0: one extra column per listed shock, entering as a state value at 0 and as an instantaneous entry (delta) on the named rows at u = 0-, through the existing impulse-column and `instant` machinery; no band is built.

## Unknowns
- Per agent the raw map g(t, b) now has nodes for b in (t, t + L] (weights on pre-zero increments) on the band, and the action kernels gain the band columns. With a zero past those extra unknowns have exactly zero rows and columns and are dropped by `_identified`, so the solved system is today's.

## Operators that change
- State part: for s < 0 the Volterra path starts at t = 0 with the old state kernel at age -s as initial condition (e^{At} K_old(-s)); the impulse column of an initial shock starts at 0 too.
- Lagged reads and delayed rows crossing t = 0 read the old control or state kernel at age (t - lag) - s, a one-dimensional interpolation from the StationaryResult's age grid (its breakpoints must be band breakpoints so kinks stay on edges).
- Passive rows before 0: the old closed-loop row kernels with the agent's own control terms removed (row_blocks with own controls excluded, applied to the old Z); the projection at date t integrates s from -L to t, its pre-zero segment read from those known kernels.
- Continuation: `horizon.continuation: stationary` (default) closes the game with the new model's stationary equilibrium: beyond T every kernel is its stationary value at age tau - s and the impulse response is R_new(tau - t), so the tail int_T^inf splits into a known term (added to bvec) and a Gram-like operator on the agent's own action (added to Amat), both built once from the new StationaryResult; `continuation: none` ends the game at T (today's closure).
- Costs stay additive over shocks: the band's row weights join the Gram.

## Mean layer (branch "means")
- The transition mean path starts from the past's stationary means (x0 = past.means of the states; a model `initial` overrides per state), pre-zero lagged reads return the old constants, and the tail beyond T uses the new stationary means; the mean solver itself is the finite one on the extended domain.

## Past object and API
- `solve(model, past=...)` with `past` a StationaryResult, a stationary Model/dict/path solved on the fly at its own horizon, or `{"initial": [{"loads": {"V": sigma0}, "rows": {"trader1.y": 1.0}}, ...]}` (a second entry with rows only is signal noise: the Kyle-Back prior). `past` is an engine constructor option, recorded in `solver_kw` so `refine()` and `stability()` rebuild it.
- Model file: `horizon: {kind: transition, window: T, nodes: n, discount: rho, continuation: stationary, past: {model: old.yaml | inline dict, initial: [...]}, stationary: {window: L, nodes: m}}`; the `stationary` block sizes the new model's own stationary solve for the closure (default: the past's window and nodes). A keyword `past` overrides the file block. `ModelBuilder.transition(T, nodes, ...)` mirrors it.
- `transition(old, new, T, nodes=12, **solve_kw)` helper: solves the old stationary model (or takes a result), solves the new one, builds the transition horizon on `new`, returns the result with `res.past` and `res.stationary` attached; it starts from the new stationary maps (`start="stationary"`), while `solve()` keeps `start="zero"` as its default.
- A new kind rather than a finite-with-past block: the closure, the cost kind and the result differ; the finite engine stays as is.

## Validation
- Old and new must share channels (names, order), states, controls per agent, agent names and each agent's row names in order; any mismatch is a ValueError naming the first item. May differ: every coefficient, noise loadings, losses, discount, delays, myopic flags, ties, definitions.
- Old delays and the old grid's breakpoints must be multiples of the new unit (else the existing "set horizon.unit" ValueError); a finite result as past is a TypeError; an unconverged past is a ValueError ("solve the past to tolerance first"); an initial entry naming an unknown state or row, or loading nothing, is a ValueError; `past` on a stationary or finite model is the existing unknown-option TypeError.

## Result
- `TransitionResult(TriangleResult)`, kind "transition": `kernel(name, ch)` over triangle and band nodes, `grid.s` negative on the band, `evaluate(name, ch, t, s)` for s >= -L; `res.past`, `res.stationary`; `res.times` and `res.loss_path[agent]` (E[loss(t)] by row quadrature at every time node); `res.costs` stays the discounted integral over [0, T] (bit identity), `res.excess_costs[agent]` = int_0^T e^{-rho t} (E loss(t) - flow_new) dt, the cost of the transition, finite at rho = 0; `res.belief_error(agent, name)` on demand (one per-date Gram, the state's kernel minus its projection on the agent's rows).
- Guard: `res.settled`, the largest relative distance of any kernel on [T - L, T] from the new stationary kernel at the same age; a `diagnose()` row "transition" with threshold 1e-6 and the flag "TRANSITION NOT SETTLED by T - L: raise horizon.window", since the closure assumes it. The past's own `window_tail` is echoed as a row.
- `plot()`: kernels against s from -L with the band shaded, a row with E[loss(t)] and the old and new flows as horizontal lines, a row of belief errors. `to_dict()` adds `past` (provenance: kind, old model, params, window, nodes, costs; no kernels), `stationary` (costs and provenance), `loss_path`, `times`, `excess_costs`, `settled`, `past_window` in `grid`.
- `sweep` accepts `"horizon.window"` (T) with a warm start by interpolating the previous action kernels onto the new grid, zero beyond the old T; a sweep over a change size reuses one past through `solver_kw={"past": res_old}`.

## Tests, in the order of what they pin
1. Zero past, `continuation: none`: Z, maps, costs and evaluation count identical (array_equal) to the finite engine on the Chapter 1 and the delayed examples.
2. Same-model stationary past with stationary continuation (Chapter 3, L = 10): every kernel equals K_stat(t - s) to 1e-9 on every node, loss_path constant, settled below 1e-9, error on the last panel no larger than in the middle.
3. One-agent prior start (test_finite_discount's model, X(0) ~ N(0, P0) as an initial shock): kernels, loss path and cost against the closed form (constant infinite-horizon Riccati gain, Kalman filter from P0) at 1e-6; the informed variant (row loading 1) gives P(0) = 0.
4. Prior shock on a state with no row loading, `continuation: none`: equals sigma0 times the finite engine's s = 0 column to round-off.
5. Kyle-Back prior start (insider knows V(0), MM with prior variance): no finite-horizon reference exists in tests/refs or numerics/kyleback (all stationary; the canonical model has a terminal payoff outside the grammar), so it pins node convergence, the monotone fall of the MM's belief error, and a documented comparison of lambda(0+) with Back's sigma_V/sigma_Z at small eps.
6. Regime change (parameter step): state kernels at t = 0+ equal the old kernels at age -s; kernels near T equal the new stationary ones; `past=Model` equals `past=StationaryResult` bit for bit; the deviation from stationarity scales linearly in the step size.
7. Validation: each rejection above, with the CLI exiting 2.
8. Payload: to_dict round trip, `solve(**res.solve_kw)` with the same past repeats, refine() and stability() rebuild with the past.
9. Guard: T < L is flagged, T = 3L passes, settled falls monotonically over three T values.
10. Sweeps: over T the warm start cuts evaluations; over the change size the past is solved once.
11. Means: with old targets the mean path starts at the old constant and reaches the new one; one-agent deterministic LQ closed form from x0 = xbar_old.
12. Performance: Chapter 3 transition, T = L = 10 at 12 nodes per side, under 60 s and 2 GB, at most three times the finite solve at that T; the same-model test under 30 s.

## Staged plan
- Stage 0 (1 day): spec kind, past normalisation, validation, `transition()` sugar; test 7.
- Stage 1 (3 days): band geometry and extended grid; zero-past bit identity; test 1.
- Stage 2 (4 days): old-shock columns, crossing reads, pre-zero projection, band map unknowns; tests 3, 4, 6.
- Stage 3 (3 days): stationary continuation operators; tests 2, 6, 9.
- Stage 4 (2 days): result fields, guard, plots, payload, CLI; tests 5, 8.
- Stage 5 (2 days): sweeps, warm starts, means hookup, performance, README and CHANGELOG; tests 10-12.

## Key decisions
- A new horizon kind `transition` with `past`, `stationary` and `continuation` blocks, rather than a `past` block on `finite`: the closure and the cost kind differ.
- The past is one object, a StationaryResult (or a stationary model solved on the fly); an `initial` shock list is the L = 0 past carried as extra impulse columns with instantaneous row entries, not a band.
- The s >= 0 triangle and its node order are untouched and the band is appended, so a zero past reproduces the finite engine bit for bit; rejected: one new grid for both regions.
- Beyond T the new stationary equilibrium closes the game as a known tail term plus a Gram-like operator built once from the new StationaryResult; `continuation: none` keeps today's ending; rejected: ending the game at T by default.
- `res.costs` stays the discounted integral over [0, T]; the transition is reported separately as `loss_path` and `excess_costs`; rejected: redefining costs to include the stationary tail.
- The guard `settled` compares the kernels on [T - L, T] with the new stationary kernels and is a diagnose row with a summary flag; `check()` keeps its error contract (convergence only).
- Old and new must share channels, states, controls, agents and row names; everything numeric, plus delays on the same unit, may change; rejected: allowing structural differences with zero-padding.
- `past` is an engine constructor option recorded in solver_kw (rebuilt by refine and stability) with provenance in to_dict, not the kernels; rejected: serialising the past kernels into every payload.
- `solve()` keeps `start="zero"` for reproducible counts and the bit-identity test; `transition()` starts from the new stationary maps.
- Kyle-Back with a prior is pinned by convergence and a documented Back comparison, since no finite-horizon reference exists; the exact prior-start pin is the one-agent Kalman/Riccati closed form.

## Risks
- The band's diagonal kinks t - s = k tau double the pieces per cell; a mis-cut shows as lost spectral accuracy first in the same-model stationary test.
- The stationary closure is exact only once the transition has settled by T - L; a short T gives a biased answer unless the guard is loud in summary and to_dict.
- Bit identity for a zero past needs the past=None path to skip every extra term and reuse the same cached grid, not add zeros to a different object.
- Dense operators grow as (N + N_band)^2: for L = T the node count doubles and memory quadruples; the performance target may force fewer nodes on the band.
- The past's own window truncation becomes an initial-condition error of the transition; echoing the past's window_tail is the only warning.
- The means branch defines x0 and the transition redefines it from the past; the two must agree on precedence before merging.
