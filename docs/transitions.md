# Transitions

A regime change solved on the spectral finite engine from a stationary past.  The README has the short form;
this page has the construction, the result's fields, the identities that pin it and the two shipped examples.
The design record is in [design/](design/README.md).

A regime change: the game runs in one stationary equilibrium until time zero, its coefficients
change, and the new equilibrium path is solved on [0, T].  What the new problem needs from the
time before zero is a *past*: the dependence of the physical state and of every agent's
information on the shocks that arrived before zero (each state's, control's and signal row's
kernel as a function of shock age on the past's window L, the rows' noise loadings, the old
constant means).  A past is given in three ways: a converged `StationaryResult` of the old
model, the old model itself (a `Model`, dict or path, solved on the fly), or a list of initial
shocks `[{"name": "v0", "loads": {V: sigma}, "rows": {"trader1.flow": 1.0}}]` (a value drawn
once at time 0- from a given covariance, seen at once by the rows named: a prior on a state).
Old and new must share the channels, the states, each agent's controls and signal rows by
name; every coefficient, delay, loss and discount may change.

```yaml
horizon:
  kind: transition
  window: 6.0                          # the horizon T (at least the past's window L)
  past: {model: ch3_two_player.yaml}   # or an inline model, or initial: [{name, loads, rows}, ...]
  continuation: stationary             # the new model's stationary equilibrium closes the game (or: end)
numerics: {nodes: 12, continuation_nodes: 12}   # per side of each piece; the continuation's solve (its window is the past's)
```

```python
res = ns.solve("examples/ch3_precision_change.yaml")        # the file form
res = ns.solve(new_model, past=old_result, continuation="stationary")   # the keywords override the file's blocks
res = ns.transition(old, new_model, T=6.0, numerics={"nodes": 12})   # solves old, new's stationary equilibrium and the transition
res.past, res.stationary, res.settled, res.loss_path, res.excess_costs, res.belief_error("player2", "X")
```

Every transition with a continuation starts from the continuation's stationary maps (`start="stationary"`,
the default of `solve()`, `transition()`, `sweep()` and the file form alike, where a settled transition ends;
`start="zero"` asks for the zero start).  The old shocks stay alive on a band of nodes with shock time s < 0 until age L, where every
shock is forgotten (on both families: this is what makes the same-model identity exact); the
new shocks live on today's triangle, whose nodes and order are untouched, so a solve with no
past is the finite engine bit for bit.  With `continuation: stationary` (the default of kind
`transition`) the new model's stationary equilibrium is solved first and every map is frozen at
its stationary value on a buffer [T, T + L]; the first-order conditions of [0, T] integrate their
continuation to T + L through it, so a transition that has settled by T - L solves the infinite
problem exactly.  Two exact identities pin the construction: a model as its own past and
continuation returns its stationary kernels on every node (Chapter 3 to 2e-9 at 16 nodes, its
mean paths to 4.7e-10, the delayed Chapter 1 model to 2e-9 at 8 nodes, the two-firm market to
2.3e-4 at 5 nodes, each at its grid's closed-loop floor), and a prior on the state of the
one-agent discounted model reproduces the Kalman filter from P0 (cost to 1e-7, the belief
error to 3.4e-6).  `continuation: end` ends the game at T instead (the maps within L of T, hence
the kernels within about 3L of T, carry the end).  Two guards: `res.settled`, the largest relative
distance of any map on [T - L, T] (and, with driven means, of the mean paths at T-) from the
stationary ones, with the `settled` row flagging "TRANSITION NOT SETTLED by T - L: raise
horizon.window" above `settled_tol` (1e-4, the closed loop's decay over a unit of t; on the
Chapter 3 precision change 4.8e-4 at T = 6, 2.7e-6 at T = 9); and `res.representation_parts`,
which says whether the resolution guard's error sits in the interior, on the band's collapsing
tip, on the last window or on the buffer (the tip and the last window are geometry, not
resolution).  The past's and the continuation's own window tails are echoed as rows.

`ns.transition_gap(old, new, numerics=)` is the settle march's T = 0 pass (design/transition_settle_march.md): everyone at the new model's stationary rules from date zero with the old regime's shocks attached, one best response per agent, and per agent the relative distance of that rule from the stationary one (max over the identified nodes, relative to the rule's peak), on the smallest strip the engine builds, [0, L] (the compile refuses T < L).  Chapter 3's precision change 3 -> 10 gives 0.585 and 0.061 at 12 nodes, 3 -> 3.03 gives 0.0033 (linear in the mismatch); the same model as its own past sits at the one-shot floor, 4e-5 at 12 nodes, 1e-5 at 16, 2.4e-6 at 24, so a tolerance below that floor is never met.

`ns.transition(old, new, settle=1e-4)` (the file form: `settle:` in place of `window:` under kind transition, exactly one
of the two; the CLI: `transition old.yaml new.yaml --settle 1e-4 [--step DT] [--max-window K]`) makes the horizon an
output: the march in T of design/transition_settle_march.md.  It starts at the T = 0 pass above (under the tolerance the
smallest-window solve at T = L, one or two evaluations from the stationary start, is the transition: the same-model past
stops here), then grows T by `step` (default one window L; the engine's floor is T = L, so there are no unit steps below
the first window) with each solve warm-started from the previous maps read on the new grid and the stationary rules on the
new stretch (`warm_maps_from`, the sweep's warm start; the continuation solved once), and after each solve runs the
monitor: the best-response pass on the converged maps over the window before the last, [T - 2L, T - L] once T >= 2L
(the gap on the last window is the handover and is never zero; below 2L the strip under T - L, the whole strip at T = L;
each row says which), stopping when every agent's gap is under `settle` or when T would pass `max_window` windows
(default 8: the settled flag then stays and names the stop).  The result carries `res.extra["window"]` (the T found),
`res.march` (rows `{T, gap, gap_last, evaluations, seconds, monitor}`, T = 0 first; `gap_last` is the same pass on
[T - L, T], what `settled` measures), `res.march_stop` and `res.settled` as before.  Chapter 3's precision change 3 -> 10
at 12 nodes, settle 1e-4: T = 0 gap 0.585; T = 3 in 20 evaluations (5.3 s), gap 0.577 on [0, 3]; T = 6 in 8 (3.7 s),
0.577 on [0, 3] again; T = 9 in 4 (3.2 s), 9.2e-4 on [3, 6]; T = 12 in 4 (4.2 s), 9.3e-7 on [6, 9]: a factor of 625 then
990 per window, the closed-loop rate, the last at the 12-node floor.  36 evaluations and 16.8 s in all against 20
evaluations and 12.3 s for the one explicit solve at T = 12 from the stationary start (23 and 13.9 s from zero), the
maps agreeing to 5e-8 at the default tol (3e-11 at tol 1e-11, where each solve takes 19 to 28 evaluations and the
warm start no longer saves).  Because the monitor certifies the solve one window back, the march's T is one window past
the smallest T whose explicit solve settles on its last window (T = 9: settled 2.7e-6; T = 6: 4.8e-4).  Not built yet
(the next stage): steps by units with reuse of the panels, path factors and preconditioner blocks across steps.

`ns.transition(old, new, T, nodes=12, **solve_kw)` solves the old regime (or takes its result),
the new model's stationary equilibrium and the transition, starting from the new stationary
maps (`start="stationary"`; `solve()` keeps `start="zero"`), and returns the result with
`res.past` and `res.stationary` attached.  The result (`TransitionResult`, kind `transition`)
carries the kernels on the band (`res.kernel(name)` with `res.grid.s < 0`, `res.evaluate(name, ch,
t, s)` for s >= -L, an initial shock as a column named after it: `res.shocks`), `res.times` and
`res.loss_path[agent]` (E[loss(t)] at every time node of [0, T] and the buffer, by row quadrature;
its discounted integral over [0, T] is `res.costs`, which keeps its meaning), `res.excess_costs
[agent]` = the discounted integral over [0, T] of E[loss(t)] minus the new stationary flow (the
cost of the transition, finite at rho = 0), `res.old_flows` and `res.new_flows`, and
`res.belief_error(agent, name)` (the variance of the agent's estimation error of a quantity at
every time node: its kernel minus the projection on the agent's seen rows, one Gram per date).
`res.cost_parts[agent]["continuation"]` is the buffer's cost, reported apart.  The means run on
[0, T] from the past's constants (a per-state `initial` overrides) and are frozen at the new
stationary means on the buffer.  `plot()` draws the kernels against the shock time from -L with
the band shaded, a row of E[loss(t)] with the old and new flows as horizontal lines, a row of
belief-error variances and the mean paths.  `to_dict()` adds the past's and the continuation's
provenance (kind, model, parameters, window, nodes, costs, window tail; no kernels), `times`,
`loss_path`, `excess_costs`, `old_flows`, `new_flows` and `settled`; `refine()` and `stability()`
rebuild the engine with the same past and continuation.  `sweep(model, "horizon.window", [6, 9,
12])` sweeps T with each point warm-started from the previous maps read on the new grid (the
stationary maps beyond it: 8 evaluations against 21 at T = 9), and a sweep over a change size
on a transition model solves the past once.

Two examples: `examples/ch3_precision_change.yaml` (player1's precision 3 to 10, T = 6, 12
nodes, 9 s: excess costs 0.0240 and 0.0286 over the new flows, E[loss(0+)] of player1 0.4351
against the old flow 0.4290, the state's variance continuous at zero while the controls jump)
and `examples/kyle_back_prior.yaml` (the Kyle-Back market started from a prior V ~ N(0, Sigma0),
the insider seeing V at once, the market maker from the prior variance, the game ending at
T = 2, trading cost eps = 0.1: the initial price impact lambda(0+) = 0.658872 at 8, 12 and 16
nodes, the market maker's belief error falling monotonically from Sigma0 to 0.132 Sigma0 at T).
No finite-horizon reference exists for the latter (`tests/refs` and the dissertation's numerics
are stationary; Back's model has a terminal payoff outside the grammar), but the market's own
structure pins it: the price is a martingale with a constant impact (the market maker's map is
the constant lambda on every node of the strip, spread 7e-7 at eps 0.2), so lambda^2 sigma_Z^2 T
= Sigma0 - Sigma_T, with Sigma_T = `res.belief_error("market_maker", "V")[-1]`, must hold at
every eps, and it does: at 12 nodes lambda = 0.614143 against sqrt((Sigma0 - Sigma_T)/T) =
0.614142 at eps 0.2, 0.658872 against 0.658865 at 0.1, 0.682548 against 0.682514 at 0.05,
0.692248 against 0.692185 at 0.03 (`tests/test_transition_examples.py` pins the first two to
1e-4).  Back's eps = 0 limit, where V is revealed by T, is lambda = sqrt(Sigma0/T)/sigma_Z =
0.7071 here (sqrt(Sigma0)/sigma_Z only at T = 1), and a sweep over eps converges to it: 0.614,
0.659, 0.683, 0.692, 0.697 at 0.2, 0.1, 0.05, 0.03, 0.02, the revealed share of Sigma0 by T
rising from 75% to 99%; eps 0.05 and below converge only warm-started from the previous point
(`sweep`), not from a zero start, and at 0.02 the fixed point no longer converges from the warm
start (0.697066 against the identity's 0.697067 all the same).  The trader's second-order check
reports NOT A MINIMUM: -0.0021 at 8 nodes and -0.0018 at 12 at eps 0.2, -0.0145, -0.0102, -0.0083
at 8, 12, 16 at eps 0.1 relative to the form's largest curvature, which sits on the prior column
(relative to the flow map's own largest, -0.0098, -0.0128 and -0.072, -0.078, -0.086), passing at
eps 1.  The report is correct about the discrete objective
(finite differences of the trader's cost along the flagged direction equal the form to 7 digits)
and has nothing to do with the prior: the direction is the trader's response to the first flow
increments, alternating in t along the line s = 0, with no weight on the prior column, and the
same finite-horizon market without a prior (V a random walk seen by the trader) flags the trader
as well (-0.0042 at 8 nodes).  It is a quadrature artefact of the strip on the D P cross term:
with a constant lambda the continuous cross term is (lambda/2)(int delta dt)^2 >= 0 and the
continuous problem is strongly convex (modulus 2 eps), but the strip does not integrate the
product of an interpolant with its own Volterra integral exactly on the s = 0 mode, where the eps
term's convexity is only the edge weights' (along the flagged direction at 8 nodes the eps term
gives +5.8e-3 and the cross term -7.0e-3); the error does not vanish with nodes (the form's
generalised eigenvalue against the eps term -1.83 at 8, -2.21 at 12) and vanishes when eps
dominates.  A known limitation of the strip quadrature ([limits.md](limits.md)), not a saddle of the market.
