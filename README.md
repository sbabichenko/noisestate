# noisestate

Equilibrium solver for linear-quadratic-Gaussian games with private information.
You describe the model as data (a YAML file or a few lines of Python); the solver
returns the equilibrium in noise-state linear strategies: every agent's action as
a kernel over the primitive shocks, its raw strategy on its own signal history, and
the decomposition of each first-order condition into the instantaneous part, the
physical continuation, and the information wedge (the part that works through
the other agents' reactions).

The framework is the decentralized LQG game of *Forecasting and Manipulating the
Forecasts of Others* (Babichenko, 2026): a linear state driven by the agents'
controls and Brownian channels; each agent observes noisy linear rows of the
states and of other agents' controls, possibly with delay; each agent minimises a
discounted (or average) quadratic flow loss; strategies are causal linear maps of
the agent's own observation history.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests        # regression tests against the chapter solvers
```

Dependencies: numpy >= 1.24, scipy >= 1.12, pyyaml; matplotlib only for `--plot`
(`pip install -e ".[plot]"`).  The examples and reference data live in the repository
(`examples/`, `tests/refs/`), not in the wheel.  MIT licence.

## Use

```bash
noisestate validate examples/ch4_kyle_back.yaml
noisestate solve examples/ch4_kyle_back.yaml -o kb.json --plot kb.pdf --param rho=0.5
noisestate --version
```

```python
import noisestate as ns
res = ns.solve("examples/ch3_two_player.yaml")
print(res.summary())
res.ages                     # shock ages (Chebyshev nodes on the panels)
res.kernel("X")              # closed-loop kernel of X, one column per channel
res.action_kernel("D1")      # closed-loop kernel of a control
res.maps["player1"]          # raw strategy g[u][r](b) on the agent's signal rows, b the age of the increment as the
                             # agent sees it (a delayed row's raw increment is older by the delay: res.MAP_CONVENTION)
res.foc["player1"]["D1"]     # {"foc", "physical", "wedge"} kernels of the first-order condition
```

## The model file

```yaml
name: ch4_kyle_back
params: {eps: 0.2, rho: 0.0, gamma1: 1.0, sigma_V: 1.0, sigma_Z: 1.0}
channels: [wV, wZ, w1]                          # Brownian channels
states:
  V: {drift: {}, noise: {wV: sigma_V}}          # dV = sigma_V dW_V (random walk on the window)
agents:
  market_maker:
    controls: [P]
    myopic: true                                # competitive: no continuation effects of own action
    signals:
      flow: {drift: {D1: 1.0}, noise: {wZ: sigma_Z}}
    loss: [[1.0, P, P], [-2.0, P, V]]           # (P - V)^2  ->  P = E[V | flow history]
  trader1:
    controls: [D1]
    signals:
      y1: {drift: {V: gamma1, P: "-gamma1"}, noise: {w1: 1.0}}
      flow: {drift: {}, noise: {wZ: sigma_Z}}   # sees the flow net of its own orders
    loss: [[-1.0, D1, V], [1.0, D1, P], [eps, D1, D1]]
horizon: {kind: stationary, discount: rho, window: 8.0, nodes: 24}
```

* **Atoms.** `name` is a state, a control, or a definition; `name@tau` is its value
  `tau` earlier (a lag), `name@-tau` its value `tau` later (a lead; allowed only in a loss
  cross term with the agent's own current control, see Limits).
* **States.** `drift` is linear in atoms (other states, controls, lagged controls,
  definitions); `noise` gives the loading on each channel.
* **Definitions.** Named linear combinations of atoms, usable anywhere:
  `Pidx: {P0@tau: 0.333, P1@tau: 0.333, P2@tau: 0.333}`.
* **Signals.** Each row has a linear `drift` (states, other agents' controls,
  definitions), a `noise` loading, and an optional observation `delay`.  Rows
  with a pure noise loading and no drift make a channel directly observed.
* **Loss.** A list of terms `[coef, a, b]` (quadratic) and `[coef, a]` (linear);
  the flow loss is their sum and the agent minimises `E int e^{-rho t} loss dt`.
  Linear terms affect only the means.
* **Ties.** `ties: [[firm0, firm1, firm2]]` makes the listed agents share one
  strategy (a symmetric equilibrium): only the first is solved for.
* **Horizon.** `stationary` with `discount`, `window` (lag window L), `nodes` per
  panel and optional `unit`/`unit_range`/`breakpoints` (panels are aligned to the
  delays automatically); or `finite` with `window` = T and `nodes` per side of
  each piece of the triangle (12 is usually converged; 6-8 when delays cut the
  domain into small pieces; the panels are the lags' multiples closed under every
  lag, see Limits for a window that is not a multiple of them); `finite_cells`
  selects the first-order cell scheme.
* Coefficients may be numbers or expressions in the parameters (`"sqrt(p1)"`).

The same structure is available from Python through `ModelBuilder` (see
`examples/make_ch5_cycle_market.py`, which builds an N-firm cycle in a loop).

## Sweeps and interactive use

```python
from noisestate import sweep
rows = sweep("examples/ch4_kyle_back.yaml", "eps", [0.2, 0.1, 0.05, 0.02])   # each point warm-started
rows[-1]["result"].summary(); rows[-1]["result"].to_dict()                   # JSON-ready
```

`noisestate sweep model.yaml eps 0.2,0.1,0.05 -o sweep.json` does the same from the shell.  Each
point starts from a secant extrapolation of the previous two equilibria in the parameter, which is
what carries the Kyle-Back sweep down to a trading cost of 0.01 where a plain restart fails.  A
warm-started point costs a handful of best responses, which is what a slider in a front end needs;
`to_dict()` is the payload such a front end would render, and it carries its own provenance: the
package `version`, the `params`, the `model` spec (`Model.from_dict` rebuilds it), the `horizon`,
the engine and solve `options`, then the grid, kernels per quantity and channel, raw maps, costs
and the first-order-condition decomposition.  `agents` lists each agent's controls and signal rows
with their `delay` and the axes of the row's map (`map_time`, `map_age` or `map_shock_time`, per
`map_convention`), since a delayed row's map is not indexed like an undelayed one: the finite engine
stores it at the shifted time t - delay.  Every sweep row names its `param`.

On panels of equal width the convolution and correlation tensors are block-Toeplitz in the panel
index, so the age grid stores a two-piece core on one panel and dense slabs for any non-uniform tail
(20 MB instead of 131 on the Chapter 5 grid) and assembles its operators from them; applying that
structure to the downstream products by FFT was measured and is slower than dense products below
about 300 uniform panels.  Grids and their operator caches are shared across solves in a process (`noisestate.clear_grid_cache()`
releases them; a large stationary grid holds a few hundred MB of convolution tensors).  The cache keeps at
most 32 grids and at most 1.5 GB of their tensors, paths and read matrices (`noisestate.grid_cache.BUDGET_BYTES`),
dropping the least recently used grids beyond that.

## How it works

A model whose single tie group is a cycle (the Chapter 5 market) is solved with that symmetry: the
closed loop is block diagonal in the Fourier basis over the cycle, so the world solve is linear in
the number of tied agents, and switching one agent off for its passive world is a low-rank
correction.  The symmetry is found from the ties and verified on the expanded model, and the
result is identical to the general solve.

The two spectral engines share one best response, written against a kernel algebra of seven
operations that each compiled model supplies (convolution with a row, the instantaneous entry
and its adjoint, the response to an action, the discounted continuation, the read of a lagged own
control, the projection onto a row).  The engines keep what differs: the closed-loop solve (dense
on the age grid, causal block substitution on the triangle), the regularisation of the
first-order-condition system, and the projection back to raw maps.

Stationary form: every process is a kernel in shock age on `[0, L]`, stored at
Chebyshev nodes on panels whose breakpoints include every delay, so delays are
exact shifts and kernels may jump there.  Given all strategies, the closed loop is
one linear system in the nodal kernels.  An agent's best response is computed in
its *passive world*, the closed loop with its own strategy switched off: its
information is the history of its passive signal rows, which does not depend on
its own strategy, so writing its control as kernels on those rows makes the
per-date first-order condition (instantaneous derivative plus the discounted
continuation through the physical state and through the other agents' reactions)
affine in the unknown, and the best response is a single linear solve.  The raw
strategy is recovered by projecting the resulting action kernel on the agent's
closed-loop rows, and the equilibrium is the fixed point of the best-response map
(all engines iterate on the action kernels with Tikhonov-regularised Anderson
acceleration, the outer solver of the Chapter 5 market solver, and derive the raw
maps by projection; a Newton-Krylov polish runs if Anderson stalls.  `solve(variable=
"maps")` iterates on the raw maps instead, which is what happens with ties in any case).
Both variables are kept because each fails somewhere the other does not: on the delayed
Chapter 1 finite model the raw maps stall at a residual of 9e-7 after 313 evaluations
where the action kernels converge in 16; with ties only the raw maps carry over between
tied agents.

Finite horizon: the same construction on a piecewise-spectral triangle.  Kernels
K(t, s) live in (time, shock-age) coordinates on the domain cut by the delays:
rectangles where the age panel lies below the time panel, Duffy-mapped triangles
where they coincide, Chebyshev nodes on each piece.  Kernels are analytic on each
piece, so 12 nodes per side already give the Chapter 1 equilibrium to eight digits,
and delays and delayed observations are exact.  Every operator (state propagation,
action from a map, response to an action, discounted continuation, projection on
the observation history) is a line integral built by Gauss quadrature split at
the piece boundaries; their quadrature structure is cached once per model, so a
best response is a few sparse products and one dense solve.  A first-order
uniform-cell scheme (`horizon.kind: finite_cells`) is kept as a cross-check.

## Validation

`tests/` reproduce the dissertation's chapter solvers from the model files:

| chapter | model | reference | agreement |
|---|---|---|---|
| 3 | two-player stationary tracking game | `solve_spectral` | 1e-11 at L = 10 (1e-5 at L = 3, window truncation) |
| 4 | Kyle-Back, one trader, rho = 0 and 0.5 | `kb_spectral_q` | 1e-4 at 24 nodes, 1e-5 at 48 |
| 5 | purchase-order market on a 3-cycle with delay | `spectral_market` sweep (16 nodes/panel) | 8 nodes/panel: 0.05-0.2% (quotes), 0.4-1.5% (orders), 19 s |
| 1 + delays | control lag and a delayed observation, finite horizon | cell scheme, Richardson-extrapolated | cost within 1e-4, kernels within 1e-3 at smooth ages; exact zero response before the observation delay |
| 1 | finite-horizon two-player game | `spec_ch1` (16 x 16 nodes, Tikhonov 1e-7) and the Chapter 1 grid solver (160 nodes, first order); `tests/test_ch1_refs.py` | converged at 12 nodes per side: cost 0.39690577, stable to 1e-11 to 20 nodes, and the cell scheme's Richardson pairs close on it as h^2 (0.396956 at 80/160 cells, 0.396918 at 160/320); the kernels at lags >= 0.1 agree with the references to their own error, 2e-3 to 4.9e-2 (the largest on a player's response to its own signal noise) for `spec_ch1` and 3e-3 to 3e-2 for the grid solver, which the package is closer to than `spec_ch1` is on every control kernel; near the diagonal `spec_ch1`'s own README reports weakly determined modes (0.35 apart below lag 0.1) and the cell scheme's Richardson limit agrees with the spectral engine to 2e-3; the dissertation's solvers put the cost 3e-4 lower (`spec_ch1` 0.39665; its README estimates the penalty's bias at +6e-4 on 0.39689-0.39690) |
| finite, discounted | one agent, rho = 0.5 (and 0 as the control) on [0, 3] | closed form: discounted Riccati equation, Kalman filter, closed-loop impulse responses (`tests/test_finite_discount.py`) | cost within 2e-8 and kernels within 6e-5 at 12 nodes per side; the cell scheme's error halves from 48 to 96 cells and its Richardson pair is within 5e-4 |

Kyle-Back with two traders: the reference grid solver (`kb_multi.py`), Richardson-
extrapolated, agrees with noisestate to 3-4 decimals; the C++ spectral port
`kb_spectral_q` differs by 2-6% and its solution is not a best response to itself
(the replica of the defect and its reference output are in `extras/`).

## Stability guarantees

* A model file with a misspelled key, an unused channel, or a control that does not enter its
  owner's loss is rejected with a message naming the offending item.
* A singular best-response system is refused on every engine with a `ValueError` naming the agent
  and the usual causes: a control with no quadratic term in its current value whose effect on the
  loss goes through nothing else (Kyle-Back with the market maker's `[1, P, P]` dropped), a
  quadratic only in a lagged read of the control (free within the lag of the window's edge), two
  rows carrying the same information, a zero noise loading.  The finite engines used to fall
  through to a least-squares solve and report a zero strategy as a converged equilibrium with a
  clean `check()`; they now LU-factor the system on its kept unknowns and refuse a reciprocal
  condition estimate below `EngineBase.FOC_RCOND` (1e-10; the worst regular system in the tests
  is at 2.6e-3).  On the cell engine's Krylov branch (above 200 unknowns) the test is one probe per
  control, which catches a control its first-order condition does not respond to; a partial
  deficiency there ends as a non-converged linear solve.  Validation warns (`UserWarning`) when a
  control has no strictly positive quadratic term in its own current value, or in a lagged read of
  it for a non-myopic agent, which is the usual cause.
* `converged` means the residual of the fixed point is at or below `tol`, where the residual is
  the norm of the update divided by the larger of one and the norm of the iterate (so for a
  solution of norm below one it is an absolute residual).  `res.message` says what the outer
  solver did, `res.summary()` shows it when the solve did not converge, and `res.check()` raises
  `ConvergenceError` so a pipeline cannot use a failed solve by accident.
* A signal row with a positive `delay` is uninformative about shocks younger than the delay; the
  map on that row is set to zero at ages above `window - delay`, where it reads nothing within the
  window.  A kernel read at a lag (a delayed row, `P@tau`) jumps at the lag, and the panels'
  duplicated breakpoint nodes carry the two one-sided limits: the lower copy reads the left limit
  (zero at the lag), the upper copy the right limit, and a lead is the exact transpose of the lag.
  With that, the breakpoints closed under adding and subtracting every row delay (so the map's
  panels and the action's panels are unions of each other shifted by the delay; beyond `unit_range`
  a delayed model's panels become uniform; the finite triangle is closed under every lag as well,
  so a lagged read is a node-to-node shift), and the map removed where the row reads nothing, the
  delayed problem is discretised exactly: on a one-agent problem with a delayed observation whose
  solution is known in closed form (certainty equivalence and a delay-differential system,
  `tests/test_exact_delay.py`) the cost agrees to 7e-11 and the kernels to 6e-6 at 16 nodes per
  panel, the representation error is 3e-13, the action is exactly zero below the delay, and the
  costs are identical to eight digits between 12 and 24 nodes at a fixed window.
* Costs are integrated with exact Gram matrices, so a converged best response is optimal against
  every feasible perturbation to round-off; `tests/test_properties.py` checks this on both engines
  without any reference solution, together with the equivalence of the two iteration variables
  and invariance to channel relabelling and agent order.

### Guards against misleading results

A converged solve is a solution of the discretised, truncated model.  `res.diagnose()` lists every
check as a row `{name, value, threshold, ok, flag, advice}` (ok is None where a check gives no
verdict); `summary()` prints the rows that fail and `to_dict()["diagnostics"]` carries them all.
The checks:

* `solve(..., start="coarse")` solves first at half the nodes and starts the fine iteration from
  that equilibrium interpolated onto the grid: fine-grid evaluations fall by a factor of 1.5 to 8
  across the examples (the delayed Chapter 3 game: 25 to 3), the equilibrium is the same to the
  tolerance, and the coarse solve itself costs a few fine evaluations.  The default start is zero
  so recorded evaluation counts stay reproducible.  `refine()` always starts the finer solve from the
  result it is refining.
* `noisestate solve model.yaml --refine` (or `solve(..., refine=True)`, `res.refine()`) re-solves
  on a grid with 1.5 times the nodes (twice the cells for the cell engine) and reports the change
  of every cost, relative to the largest cost, and of the kernels; `res.refinement["resolved"]`
  is the verdict for the spectral engines, and the summary says `NOT RESOLVED`.  The cell engine
  is first order, so it reports the changes without a verdict.  On the finite engine with delays
  the refinement rebuilds the delay-cut triangle's quadrature and can take far longer than the
  solve; the delayed Chapter 1 example refines from 8 to 12 nodes per side in about a minute.
* Stationary results carry `res.window_tail`, the largest change of a kernel over the last tenth
  of the window relative to that kernel's peak; above 2% the summary says `WINDOW TOO SHORT`,
  because the equilibrium solved is that of the model truncated at `horizon.window`.  A kernel that
  has decayed or reached a constant limit (a random-walk state, a price that tracks it) is not
  flagged.  The Kyle-Back example with `rho: 0` is flagged: with no discounting the trader's
  stationary problem has no solution and the kernels are window artefacts (profit 0.93 on a window
  of 8, 0.38 on 16); the example ships with `rho: 0.5`, where the profit is 0.8208 on both.
* Stationary results with `discount: 0`, and finite-horizon results at any discount, carry
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
  Discounted stationary models are not checked (their objective is not a quadratic form in the
  stationary kernel).  Up to a strategy dimension of 1000 the form is built densely and always
  settles; above that a Lanczos iteration is used, and when it does not settle the report says so
  (`converged: False`) instead of staying silent.  The undiscounted Kyle-Back model at a
  trading cost of 0.01 crosses the threshold (-1.6e-4) on the window of 8: the truncation
  effect grows as the trading cost shrinks, and a positive discount removes it.
* `stability()` reports the spectral radius of the best-response map (tatonnement stability) and
  the method that produced it.  This is a different question from whether the fixed-point
  solver converged: the Kyle-Back example converges under Anderson mixing while its radius is
  1.4, so naive best-response adjustment would not find that equilibrium.
* Every sweep row has `change` (relative change of the raw maps from the previous point, on the
  same grid) and `jump` (that change is more than five times the sweep's median), so a branch
  jump between neighbouring points is visible instead of silently plotted as a curve.  The
  change is per point, not per unit of the parameter, so a geometric sweep is not flagged.
* `res.cost_kind` and `model.notes` name what the numbers are: stationary costs are flow losses
  per unit time, finite-horizon costs are discounted integrals; a row that observes a control
  directly sees only its predictable part; a myopic agent ignores its effect on future flows; a
  linear loss term moves only the means, which are not solved, and has no effect on kernels or
  costs.  `noisestate validate` prints the notes.  A parameter that nothing references is an
  error, the usual sign of a misspelled name elsewhere in the file; so are a drift that depends on
  a future value, a zero noise loading, `breakpoints` that do not end at the window, a
  `myopic` that is not a boolean, a lag, delay or lead that is not below the window, a
  `unit_range` above the window, and a misspelled agent in `naive_observers`.
* `refine()` and `stability()` rebuild the engine that produced the result, with the same options
  (naive observers, ridge, tolerances).  A built model is single-sourced: its coefficients are
  numbers, so `model.params` is read-only and `model.with_params(p=4.0)` returns a new model, while
  the horizon fields (`nodes`, `window`, ...) may be changed on the object or through
  `model.with_horizon(nodes=32)`; `solve`, `sweep`, `refine` and `stability` all see the same model.
* `ties` are checked structurally: rows, losses, delays and coefficients up to relabelling, the
  dynamics of each agent's private states, and whether a row's noise channel also drives a state.

## Known open items

* (Resolved.)  A loss cross term between a control and its own lagged read (`[c, D@tau, D]`) gave the
  finite engine a second-order curvature of -3e-5 to -6e-5 relative to the largest at 6 to 8 nodes per
  side for `c = 0.15`, `r = 0.5` (positive at 4 nodes and at `c = 0.05`; the delayed and undelayed rows
  agreed).  The stationary engine never showed it: on delay-aligned panels its shift is an exact
  contraction of the cost's Gram norm, its loss form is positive definite, and the reported curvature is
  the Hessian of the discrete flow loss in the map on the passive rows (rebuilt independently to every
  printed digit; +2.4e-3 at 6 nodes, +1.8e-3 at `c = 0.15`).  On the triangle the lagged atom read
  through `read(lag, lag)`, which copies one source node onto a triangle's whole degenerate corner row:
  a 0/1 matrix of mass norm 4.8 (4 nodes) to 9.0 (6 nodes), so the discrete cross term was not bounded
  by `|c|` times the own term on every nodal vector and the loss form itself was indefinite.  Lagged
  atoms now read through the node-to-node `map_shift` (mass norm exactly 1): the form is positive
  definite, the curvature positive (+1.9e-5 at 6 nodes, +7e-6 at 8), and costs and kernels unchanged.

* (Resolved.)  The Chapter 5 cycle-market example reported a second-order curvature of -3e-5 to
  -4.6e-5, the two-firm variant on a short window -7e-4 to -2e-3.  It is window truncation: a
  firm's own price enters its loss quadratically only through lagged reads, which within the
  largest lag of the window edge fall past it, so on that slab only the current-time cross term
  between orders and prices acts, and it is indefinite.  The direction lives on the last panel,
  its mass-weighted value is identical at windows 24, 36 and 48, it is positive when embedded in a
  wider window, and the untruncated Hessian is positive definite on every grid tried; a quadratic
  in the firm's current price above c^2/r removes it (0.25 for two firms, 0.5 for three).  The
  check now re-evaluates a negative direction on a window longer by two lags and reports the
  truncation as such.

## Limits

Scalar states and controls (write vector models as several scalars); no exact
(noise-free) observation of a state that is not itself a channel; means (targets,
linear loss terms) are not yet solved.  Lead atoms (`X@-0.5`) are accepted only in a
stationary loss cross term with the agent's own current control (`[c, D, X@-0.5]`),
where the covariance is computed exactly (its first-order condition carries the extra
term from flows before *t* that read the quantity after *t*); a led quantity squared,
or a lead on a control, would need the part of the kernel on shocks arriving after *t*,
which the age grid does not carry, and is rejected (write the flow with lags: at
discount 0 the time average of *X(t+tau)^2* equals that of *X(t)^2*).  The finite
engines reject leads.  With lagged *state* feedback in a drift
(`X@0.5` in the drift of `X`), the map and action-kernel iterations agree only to
first order in the node count (1.6e-5 at 24 nodes per panel on the Chapter 3 game);
the action-kernel path is the default and the more accurate one.  A game can have
several equilibria: `ties` selects the symmetric one, an untied solve from a zero
start may land on another.

On the finite spectral engine the time and age panels are the multiples of the
lags closed under every lag and delay, so a lagged read and a delayed row are exact
node-to-node shifts.  A window that is not a multiple of a lag doubles the panels
(the kernels kink at `T - k tau` as well as at `k tau`: a control acting after the
lag is idle within the last lag), which quadruples the pieces and multiplies the
dense operators and the solve time by far more: `examples/ch1_delayed_finite.yaml`
solves in 3 s at `window: 1.0` (5 panels, 10 pieces) and in about 120 s at
`window: 1.1` (9 panels, 45 pieces) at 8 nodes per side.  The compile warns with the
counts when the closure adds panels; a window that is a multiple of every lag, or
fewer nodes per side, keeps the cost down.  Lags that are not multiples of one unit
(0.25 and 0.3 without `horizon.unit: 0.05`) are rejected with the unit to set.
