# Guards against misleading results

Diagnostics check convergence, numerical resolution, and the conditions needed to interpret a
result.  Convergence alone does not establish that the discretisation is adequate.

`res.diagnostics.rows` lists every check as a row `{name, value, threshold, ok, flag, advice}` (`ok` is
None where a check gives no verdict); `res.diagnostics.statuses` gives each check's status, and
`res.diagnostics.assess(policy)` is the verdict a policy makes of them --- see
[Did the required checks pass](../README.md#did-the-required-checks-pass).  `summary()` prints the rows that fail,
`to_dict()["diagnostics"]` carries them all and `to_dict()["assessment"]` the verdict.  The thresholds
are fields of `noisestate.Settings` ([settings.md](settings.md)).

## Converged, and accepted

A converged solve satisfies the discretised equations within the solver's tolerance.
`require_converged()` checks this condition.  `require_ok(policy)` also checks the diagnostics
required by the policy.

Each check reports one of six statuses:

| status | meaning |
|---|---|
| `passed` | ran; the model met it |
| `failed` | ran; the model did not meet it |
| `skipped` | this engine could have run it here; `solve(diagnostics=False)` meant it did not |
| `unsupported` | this **engine** cannot compute it (the cross-check cell engine in `extras/cells.py` builds no second-order form) |
| `not_applicable` | the check has no meaning for this **model** (a lag window on a plain finite horizon) |
| `missing` | applicable, supported, was to run, produced no record |

An `unsupported` check leaves a requirement unmet.  A `not_applicable` check does not block
acceptance because it is irrelevant to the model.  Neither status emits a failing row.

A stationary model has a `window` check.  For a transition, the lag-window checks apply to the
inherited past and stationary continuation separately: `past window` and `continuation window`.
The `settled` check measures agreement with the continuation near the end of the solved interval.
It is `not_applicable` when the game ends at T.  A prior on the initial state has no `past window`
check.  Each applicable window or settling check can block acceptance under a policy that requires it.

### Policies

A policy specifies the required checks.  `Policy.PUBLICATION` is the default;
`Policy.EXPLORATORY` requires only convergence and must be requested explicitly:

```python
res.diagnostics.statuses                                   # every applicable check and its status
res.diagnostics.assess()                                   # an Assessment: .accepted, .blocking, .uncomputed
res.diagnostics.assess(ns.Policy.EXPLORATORY).accepted     # True while you are still exploring
res.require_ok(ns.Policy.EXPLORATORY)
print(res.diagnostics.summary())                           # the grouped verdict the CLI prints
```

On the command line the same split is `--require-ok`, with `--policy exploratory` for the weaker
standard.  A check the engine cannot compute is reported as `cannot be checked here` rather than as a
failure, and says whether any setting could change it.

## The rows and their flags

| row | value | threshold | flag the user sees | advice |
|---|---|---|---|---|
| `converged` | the fixed point's residual | `tol` | `NOT converged` | `res.message` |
| `diagnostics` | none | none | `diagnostics skipped (solve(diagnostics=False): no second-order check, first-order-condition decomposition or representation error)` | solve again with diagnostics=True |
| `resolution` | the largest representation error | `resolution_tol` 1e-6 | `UNDER-RESOLVED (representation error 1.3e-05: raise numerics.nodes)`; a transition adds the error by region and `an error only on the band tip or the last window is the geometry there, not the interior's resolution` | raise numerics.nodes |
| `window` (stationary) | `res.window_tail` | `window_tail_tol` 0.02 | `WINDOW TOO SHORT (a kernel still moves by 4.1% of its peak over the last tenth of the window: raise horizon.window)`; `WINDOW TOO SHORT FOR THE MEANS (...)` when a mean condition integrates a passive response over the window (`res.mean_tail`, below) | raise horizon.window |
| `window cost` (stationary; required by the publication policy) | the largest change of an agent's cost, relative to the largest cost, on a window 1.5 times as long (`res.check_window()`; `require_ok()` measures it when the policy needs it, `skipped` until then) | `window_cost_tol` 1e-6 | `WINDOW TOO SHORT FOR THE COSTS (a window of 12 moves trader1's cost by 6.5e-05 of the largest; market_maker left out, whose cost is the window's (cost window)): raise horizon.window` | raise horizon.window |
| `cost window` (stationary; reported, required by no policy) | the largest share of an agent's flow loss accrued over the last tenth of the window (`res.cost_tail`) | `window_tail_tol` 0.02 | `COST GROWS WITH THE WINDOW (market_maker: 12% of the flow loss accrues over the last tenth of the window): a loss term reads a quantity whose response has not decayed by L ...` | raise horizon.window; near 10% the term never decays: drop it from the loss if it moves no choice |
| `second_order:<agent>` | the smallest curvature relative to the largest | `-second_order_tol` (-1e-4) | `NOT A MINIMUM (the best response of 'trader1' is a saddle: its loss is not convex in its own strategy, smallest curvature -1.8e-03 of the largest)`; `second-order check did not converge for 'trader1'` when Lanczos did not settle, or for a risk-averse agent whose loss Hessian is not positive semidefinite (the form, the expected cost's, bounds the entropic cost's curvature from below only when it is; the record then carries `"bound": "entropic"`) | the loss is not convex in the agent's own strategy |
| `second_order_edge:<agent>` | the same curvature | the same | `window edge: the curvature of 'firm0' is negative (-3.0e-05) on this window but positive (1.2e-04) on a window longer by two lags: a truncation of the lagged loss terms at the edge, not a saddle` (no verdict) | a wider window moves it, a quadratic term in the control's current value removes it |
| `refinement` | the cost and kernel changes and the finer node count | `refine_cost_tol` 1e-6, `refine_kernel_tol` 1e-5 | `refinement to 36 nodes moves costs by 2.1e-07 and kernels by 3.4e-06`, with ` (NOT RESOLVED)` when either is above its tolerance | raise numerics.nodes |
| `stability` | the spectral radius of the best-response map | 1.0 | `best-response dynamics stable (spectral radius 0.412)` or `best-response dynamics UNSTABLE (spectral radius 1.400, by power)` | naive best-response adjustment would not find this equilibrium |
| `past window` (transition) | the past's window tail | `window_tail_tol` | `PAST WINDOW TOO SHORT (a kernel of the past still moves by 2.5% of its peak over the last tenth of its window 8: solve the past with a longer window)` | solve the past with a longer window |
| `settled` (transition) | `res.settled` | `settled_tol` 1e-4 | `TRANSITION NOT SETTLED by T - L: raise horizon.T (a map on [T - L, T] is 4.8e-04 of its peak from the stationary map the buffer is frozen at, against settled_tol 0.0001: the closed-loop decay over a unit of t, not the grid's floor)` | raise horizon.T |
| `continuation window` (transition) | the continuation's window tail | `window_tail_tol` | `CONTINUATION WINDOW TOO SHORT (a kernel of the continuation still moves by 2.5% of its peak over the last tenth of its window 8: solve it with a longer window)` | solve the continuation with a longer window |

The numbers in the flags are examples; each flag prints the value it measured.  `summary()` prints one line:
the outcome (`converged`, or `NOT converged` with `res.message`), then every row that failed or has no
verdict, then the informational rows (refinement, stability, a skipped diagnostics pass) whenever they were
computed.

## The checks

* `solve(..., start="coarse")` solves first at half the nodes and starts the fine iteration from
  that equilibrium interpolated onto the grid: fine-grid evaluations fall by a factor of 1.5 to 8
  across the examples (the delayed Chapter 3 game: 25 to 3), the equilibrium is the same to the
  tolerance, and the coarse solve itself costs a few fine evaluations.  The default start is zero
  so recorded evaluation counts stay reproducible.  `refine()` always starts the finer solve from the
  result it is refining.
* `noisestate solve model.yaml --refine` (or `solve(..., refine=True)`, `res.refine()`) re-solves
  on a grid with 1.5 times the nodes and reports the change
  of every cost, relative to the largest cost, and of the kernels; `res.refinement["resolved"]`
  is the verdict for the spectral engines, and the summary says `NOT RESOLVED`.  A first-order
  engine (the cross-check in `extras/cells.py`) reports the changes without a verdict.  On the finite engine with delays
  the refinement rebuilds the delay-cut triangle's quadrature and can take far longer than the
  solve; the delayed Chapter 1 example refines from 8 to 12 nodes per side in about a minute.
* Stationary results carry `res.window_tail`, the largest change of a kernel over the last tenth
  of the window relative to that kernel's peak; above 2% the summary says `WINDOW TOO SHORT`,
  because the equilibrium solved is that of the model truncated at `horizon.window`.  A kernel that
  has decayed or reached a constant limit (a random-walk state, a price that tracks it) is not
  flagged.  The means' continuation (the DC gains of the passive-world impulse responses) is not
  truncated: it is the passive world's transform at the discount rate, over all ages, because a
  passive world (one reaction fewer than the equilibrium) can decay far more slowly than the
  equilibrium's kernels.  Where that transform does not apply (level rows, instant observations,
  monitored deviations; a passive response not decaying at L) the condition integrates over the
  window, `res.mean_tail` keeps the passive response's level at L relative to its peak, and the
  `window` check reads the larger of the two: `WINDOW TOO SHORT FOR THE MEANS (...)`.  The Kyle-Back example with `rho: 0` is flagged: with no discounting the trader's
  stationary problem has no solution and the kernels are window artefacts, so its profit settles
  neither in the window nor in the resolution -- 0.35, 0.93, 0.93 on a window of 8 and 0.63, 0.76,
  0.66 on 16, at 12, 24 and 48 nodes.  There is no number to quote, which is the point; an earlier
  version of this line pinned two of them.  The example ships with `rho: 0.5`, where the profit is
  0.820818, 0.820871 and 0.820871 on windows of 8, 16 and 32: a discount makes the problem
  well posed and the answer stops depending on the truncation.
* Every stationary result, at any discount, and every finite-horizon result carry
  `res.second_order[agent]`: the agent's objective is a quadratic form in its strategy, computed
  exactly on the feasible strategies from the cost's own Gram matrix, and its smallest eigenvalue
  relative to the largest says whether the first-order condition is a minimum.  The
  summary says `NOT A MINIMUM` when it is not (a loss that is not convex in the agent's own
  strategy, for instance a negative weight on its own control, or a cross term with no own
  quadratic term).  Curvatures within 1e-4 of zero are not flagged: the objective is truncated
  at the window, and on coarse panels the discrete strategies find a little curvature of either
  sign there (the Chapter 5 example sits at -3e-5 with 6 nodes per panel); the value is reported
  in `res.second_order` and `to_dict()` either way.
  A windowed stationary objective omits the flows past the edge that read the strategy within the
  last lag, so a cross term between a control and lagged quantities can look indefinite there.  When
  the check finds a negative direction it re-evaluates that direction, zero-extended, on a window
  longer by two lags with the same maps (one operator build, no new fixed point): positive there
  means truncation, reported as `embedded` and a `window edge` note rather than a saddle.
  A DISCOUNT DOES NOT TAKE THE CHECK AWAY.  The discounted stationary objective is a quadratic form,
  and its joint running Hessian carries no discount: rho enters only as the strictly positive weight
  `e^{-rho t}`, which cannot change the sign of a form that is semidefinite pointwise in t.  The
  check is therefore made on the average-cost system and its verdict holds at every rho, as the
  dissertation's Kyle-Back chapter does ("the second-order checks are made on the average-cost
  system and do not rely on the rho > 0 hypothesis").  Up to a strategy dimension of 1000 the form is built densely and always
  settles; above that a Lanczos iteration is used, and when it does not settle the report says so
  (`converged: False`) instead of staying silent.  The undiscounted Kyle-Back model at a
  trading cost of 0.01 crosses the threshold (-1.6e-4) on the window of 8: the truncation
  effect grows as the trading cost shrinks, and a positive discount removes it.
* `stability()` reports the spectral radius of the best-response map (tatonnement stability) and
  the method that produced it.  This is a different question from whether the fixed-point
  solver converged: the Kyle-Back example converges under Anderson mixing while its radius is
  1.4, so naive best-response adjustment would not find that equilibrium.  It makes at most
  `STABILITY_MAX_EVALUATIONS` (200) rounds of best responses: the Arnoldi iteration is stopped at 170
  and a power iteration gets the remaining 30, with `method` saying so.
* Every sweep row has `change` (relative change of the strategy from the previous point on the
  same grid: the raw maps, or the action kernels on the finite spectral engine) and `jump` (that
  change is more than five times the sweep's median), so a branch jump between neighbouring points
  is visible instead of silently plotted as a curve.  The change is per point, not per unit of the
  parameter, so a geometric sweep is not flagged.
* `res.cost_kind` and `model.notes` name what the numbers are: stationary costs are flow losses
  per unit time, finite-horizon costs are discounted integrals; a row that observes a control
  directly sees only its predictable part; a myopic agent ignores its effect on future flows; a
  linear loss term, a constant drift or an initial state moves only the means, which every engine
  solves, and has no effect on the kernels; a random walk with no inputs has no stationary mean
  and is pinned at 0.  `noisestate validate` prints the notes.  A parameter that nothing references is an
  error, the usual sign of a misspelled name elsewhere in the file; so are a drift that depends on
  a future value, a zero noise loading, `breakpoints` that do not end at the window, a
  `myopic` that is not a boolean, a `risk_aversion` that is negative or not finite, a lag, delay or lead that is not below the window, a
  `unit_range` above the window, and an unknown agent in `monitors` (or a cycle of instant observations).
* `refine()` and `stability()` rebuild the engine that produced the result, with the same options
  (tolerances, iteration variable, tuning settings).  A built model is single-sourced: its
  coefficients are numbers, so `model.params` is read-only and `model.with_params(p=4.0)` returns a
  new model.  Use `model.with_numerics(nodes=32)` to change resolution or, for a stationary model,
  `model.with_stationary(9.0)` to change its lag window.  Both return new models; pass the changed
  model to `solve` or `sweep`.  A result's `refine()` and `stability()` use the model it was solved from.
* `ties` are checked structurally: rows, losses, delays and coefficients up to relabelling, the
  dynamics of each agent's private states, and whether the shock in a row's noise also drives a state.
