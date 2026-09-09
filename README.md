# noisestate

Equilibrium solver for linear-quadratic-Gaussian games with private information.  You describe the model
as data (a YAML file or a few lines of Python) and the solver returns the equilibrium in noise-state
linear strategies: every agent's action as a kernel over the primitive shocks, its raw strategy on its
own signal history, and the decomposition of each first-order condition into the instantaneous part,
the physical continuation and the information wedge.  The framework is the decentralized LQG game of
*Forecasting and Manipulating the Forecasts of Others* (Babichenko, 2026): a linear state driven by the
agents' controls and Brownian channels, each agent observing noisy linear rows of the states and of the
other agents' controls, possibly with delay, and minimising a discounted (or average) quadratic flow
loss over causal linear strategies on its own observation history.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests        # regression tests against the chapter solvers
```

Python >= 3.10.  Dependencies: numpy >= 1.24, scipy >= 1.12, pyyaml; matplotlib only for plots
(`pip install -e ".[plot]"`).  The examples and reference data live in the repository (`examples/`,
`tests/refs/`), not in the wheel.  MIT licence.

## A first solve

```python
import noisestate as ns

print(ns.load("examples/ch3_two_player.yaml").describe())         # the model itself, no solve: equations, delays, losses, conventions (HTML in a notebook)
res = ns.solve("examples/ch3_two_player.yaml")                    # the model file's own numerics
res = ns.solve("examples/ch3_two_player.yaml", ns.Numerics(nodes=32, tol=1e-12))   # a change of resolution, not of model
print(res.summary())
res.check()                  # raises ConvergenceError unless converged -- convergence only, not the guards
res.status["ok"]             # False while any guard fails; res.status["flags"] says which
res.costs["player1"]         # the cost of each agent (res.cost_kind says what it is)
res.kernel("X")              # closed-loop kernel of a state, one column per channel (the last axis); res.axes gives its coordinates
res.action_kernel("D1")      # closed-loop kernel of a control
res.maps["player1"]          # raw strategy g[u][r](b) on the agent's own signal rows
res.foc["player1"]["D1"]     # {"foc", "physical", "wedge"}: the first-order condition decomposed
res.means["X"]               # the mean of a state or control: a constant here, a path on a finite horizon
res.status                   # {"ok", "flags", "rows"}: the verdict and the failing checks
res.numerics                 # the resolved Numerics
res.to_dict()                # the JSON-ready payload
```

That first solve prints `WINDOW TOO SHORT`, and it is meant to: the example carries the dissertation's
`window: 3.0`, where the state kernel is still 2.4% of its peak at the edge.  The guard is the answer to
"is this number trustworthy", and reading it is the workflow:

```python
res = ns.solve(ns.load("examples/ch3_two_player.yaml").with_horizon(window=9.0).with_numerics(nodes=32))
res.status["ok"]             # True: the tail is 5e-05 and the representation error 9e-12
res.costs["player1"]         # 0.42729 against 0.42895 on the short window -- the truncation, not noise
```

Four of the seven shipped examples flag something, each for a reason noted in its own file: three are
statements about the model (a non-convex best response, a representation error a small grid cannot reach)
and one is the dissertation's window.  `res.check()` does not consult the guards -- it raises only when
the fixed point did not converge -- so read `res.status` as well.

Three objects: a `Model` (the economics, from a file, a dict or `ModelBuilder`), a `Numerics` (how it is
solved: `engine`, `nodes`, `unit`, `unit_range`, `breakpoints`, `continuation_nodes`, `tol`, `damping`,
`max_newton`, `variable`, `settings`; laid over the file's own `numerics:` block) and a `Result`.  Every
engine returns a `Result`: `res.axes` names the coordinates of `kernel()` (`{"age"}` on the stationary
engine, `{"time", "age", "shock_time"}` node-wise on the finite triangle, `{"time", "shock_time"}` for
the cell engine's matrices, and under `"maps"` where each signal row's map values belong), `res.times`
and `res.paths` hold the paths of a finite horizon, `res.extra` the engine's extras.  `res.evaluations`
counts the best-response evaluations, `res.world` is every closed-loop kernel stacked.

The same model from Python, with `ModelBuilder`:

```python
import noisestate as ns

b = ns.ModelBuilder("one_agent", a=1.0, r=0.5)
b.channel("w", "v")
b.state("X", drift={"X": "-a", "D": 1.0}, noise={"w": 1.0})
b.agent("me", controls=["D"], loss=[[1.0, "X", "X"], ["r", "D", "D"]])
b.signal("me", "y", drift={"X": 1.0}, noise={"v": 1.0})
b.stationary(discount=0.1, window=8.0, nodes=16)
res = ns.solve(b)
print(res.summary())
```

`examples/make_ch5_cycle_market.py` builds an N-firm cycle in a loop; `b.finite(T, nodes)` and
`b.transition(T, nodes, past=..., continuation=...)` set the other horizons and `b.numerics(**fields)`
the rest of the block.

## The model file

```yaml
name: ch4_kyle_back
params: {eps: 0.2, rho: 0.5, gamma1: 1.0, sigma_V: 1.0, sigma_Z: 1.0}
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
horizon: {kind: stationary, discount: rho, window: 8.0}
numerics: {nodes: 24}
```

An atom is a state, a control or a definition, `name@tau` its value `tau` earlier.  A state has a
linear `drift` in atoms (plus `const`), a `noise` loading on the channels and, on a finite horizon, an
`initial` value.  A signal row has a `drift`, a `noise` and an optional `delay`.  A loss is a list of
terms `[coef, a, b]` (quadratic) and `[coef, a]` (linear); linear terms, constant drifts and initial
states move the means only.  `ties` make agents share one strategy; `horizon` is the economics of time
(`stationary` with `discount` and `window`, `finite` with `window` = T, `transition` with a `past` and a
`continuation`); `numerics` is how it is solved.  Coefficients may be expressions in the parameters.  The
full reference, every key with its type and default, is
[docs/model_file.md](docs/model_file.md); `noisestate schema model` prints the JSON Schema.

## Models as equations

The same model written as its equations, with `Param`, `State`, `Control`, `Signal`, `Agent` and `shocks`:
a state's drift is assigned (its quantity terms are the drift, its shock terms the noise loading, a constant
becomes `const`), a signal is one linear expression with at least one shock, a loss is quadratic in the
quantities (`(X - 1)**2` is `X^2 - 2X` with the constant dropped and noted in `model.notes`; `x.lag(tau)` is
`x@tau`).  `ns.Model(...)` compiles it to the model file's structure, so `to_dict()` is the file and
`save()` writes it: `examples/expr_examples.py` writes every shipped example this way and
`tests/test_expr.py` checks that each equals its YAML file.

```python
import noisestate as ns
from noisestate import Param, State, Control, Signal, Agent, shocks
r1, r2, p1, p2, sigma = Param.many(r1=0.1, r2=0.1, p1=3.0, p2=3.0, sigma=1.0)
w = shocks("w0", "w1", "w2")
X = State("X"); D1, D2 = Control("D1"), Control("D2")
X.drift = D1 + D2 + sigma * w.w0
player1 = Agent("player1", controls=[D1], signals=[Signal("y1", p1**0.5 * X + w.w1)], loss=X**2 + r1 * D1**2)
player2 = Agent("player2", controls=[D2], signals=[Signal("y2", p2**0.5 * X + w.w2, delay=0.5)], loss=(X - 1)**2 + r2 * D2**2)
game = ns.Model("ch1", states=[X], agents=[player1, player2], horizon=ns.Stationary(window=3.0))
print(game.describe())  # equations, observations and delays, losses, conventions (HTML in a notebook)
eq = game.solve()
eq = game.solve(ns.Numerics(nodes=16, unit=0.5))
eq.status.ok; eq.costs["player1"]; eq.cost_parts["player1"]; eq.means["X"]
k = eq.kernel("X", "w0"); k.values; k.axes; k.at(0.7)
for point in game.sweep(p1=[0.3, 1, 3, 10]): point.value, point.result, point.jump
old = game.solve(); new = game.with_params(p1=10.0).finite(T=6.0); path = new.solve(past=old)
game.save("ch1.yaml"); ns.Model.load("ch1.yaml")
with ns.settings(second_order_tol=1e-3): game.solve()
```

The channels are the shocks used, in `shocks()` order; the parameters are the `Param`s used, with the
values given (`Param.many` returns them in order; a coefficient may use `+ - * / **` and `sqrt exp log sin
cos tanh abs min max`, and renders to the file's expression: `p1**0.5` is `"sqrt(p1)"`); `define(name, expr)`
is a definition, `ns.Finite(T)` and `ns.Transition(T, past=..., continuation=...)` the other horizons,
`numerics=` the file's block.  `res.kernel()` returns a `Kernel`, an ndarray carrying `.axes` and `.at()`
(the engine's own interpolant: an age on the stationary engine, `(t, s)` on the finite triangle, the nearest
cell on the cell engine) and `.plot()`; `sweep()` rows carry `.value`, `.result`, `.jump` (they are still
dicts); `ns.settings(...)` replaces the default `Settings` inside the block for every engine constructed
there (process-wide, not thread-safe).  One caution on the script above: the transition on `[0, 6]` with
player 2's delayed row cuts the strip into pieces half a unit wide, 20k nodes at 16 nodes per piece; solve
it at `ns.Numerics(nodes=4)` or drop the delay.  The YAML form stays the persistence format.

## Transitions

A regime change: the game runs in one stationary equilibrium until time zero, its coefficients change,
and the new path is solved on [0, T] from the old shocks (the *past*: the old model, its converged
result, or a list of initial shocks such as a prior on a state), closed by the new model's stationary
equilibrium on a buffer [T, T + L] (`continuation: stationary`) or by the game's end (`end`).

```python
import noisestate as ns

res = ns.solve("examples/ch3_precision_change.yaml")           # the file form: horizon.past, horizon.continuation
old = ns.solve("examples/ch3_two_player.yaml")
new = ns.load("examples/ch3_two_player.yaml").with_params(p1=10.0)
res = ns.transition(old, new, T=6.0, numerics={"nodes": 12})   # solves the new stationary equilibrium and the transition
res = ns.transition(old, new, settle=1e-4, numerics={"nodes": 12})   # or finds the horizon: a march in T (res.extra["window"], res.march)
res.settled, res.excess_costs, res.loss_path["player1"], res.belief_error("player2", "X")
```

Every transition starts from the continuation's stationary maps.  Two exact identities pin the
construction: a model as its own past and continuation returns its stationary kernels on every node,
and a prior on the state of the one-agent model reproduces the Kalman filter.  The construction, the
result's fields and the two shipped examples are in [docs/transitions.md](docs/transitions.md).

## Sweeps and interactive use

```python
import noisestate as ns

rows = ns.sweep("examples/ch4_kyle_back.yaml", "eps", [0.2, 0.1, 0.05])   # each point warm-started
rows[-1]["result"].summary(); rows[-1]["change"], rows[-1]["jump"]
```

Each point starts from a secant extrapolation of the previous two equilibria, which carries the
Kyle-Back sweep down to a trading cost of 0.01 where a plain restart fails; every row names its
`param`, its `change` from the previous point and whether that is a `jump`.  A warm-started point costs a
handful of best responses, which is what a slider needs, and a solve can be bounded and watched:

```python
import noisestate as ns

log = []
res = ns.solve("examples/ch3_two_player.yaml", max_evaluations=3, deadline=30.0,
               progress=lambda ev: log.append(ev["residual"]), diagnostics=False)
res.converged, res.message          # False, the bound that stopped it; nothing raises, check() does
```

`max_evaluations` caps the best-response evaluations and `deadline` the wall time in seconds; past
either the best iterate comes back with `converged=False` and `res.message` naming the bound.
`progress` is called after every evaluation with `{"evaluation", "residual", "phase", "seconds"}`; an
exception it raises cancels the solve.  `diagnostics=False` skips the checks at the end (`res.foc` and
`res.second_order` stay empty, the summary says `diagnostics skipped`) and halves a warm-started
re-solve.  `start="coarse"` solves first at half the nodes.  `sweep(..., solve_kw={...})` forwards these
to every point.  Grids and their operator caches are shared across solves in a process
(`ns.clear_grid_cache()` releases them).

## The guards

A converged solve is a solution of the discretised, truncated model.  `res.diagnose()` lists every check
as a row `{name, value, threshold, ok, flag, advice}`; `summary()` prints the rows that fail.  The full
descriptions, with thresholds and advice, are in [docs/guards.md](docs/guards.md).

**Resolution.**  The representation error of the action kernels on the seen rows above 1e-6 prints
`UNDER-RESOLVED (representation error 1.3e-05: raise horizon.nodes)`.  `solve(..., refine=True)` or
`res.refine()` re-solves at 1.5 times the nodes and reports the change of every cost and kernel, with
`(NOT RESOLVED)` when either moves more than its tolerance.

**Window.**  On the stationary engine a kernel still moving by more than 2% of its peak over the last
tenth of the window prints `WINDOW TOO SHORT (a kernel still moves by 4.1% of its peak over the last
tenth of the window: raise horizon.window)`: the equilibrium solved is that of the model truncated at
`horizon.window`.  The Kyle-Back example with `rho: 0` is flagged (its kernels are window artefacts);
it ships with `rho: 0.5`.

**Second order.**  A negative curvature is a claim about the model, and the grid can make the same claim
falsely: `res.refine()` re-solves on a finer grid and records the curvature there
(`res.refinement["second_order"]`), and one that shrinks towards zero overturns the verdict --- the
direction is the quadrature's, not a strategy (on the triangle it sits on the diagonal `a = t` and
alternates in sign between neighbouring age nodes).  `examples/kyle_back_prior.yaml` is the case: the
smallest curvature runs -1.45e-02, -1.02e-02, -8.26e-03, -6.71e-03, -5.63e-03 at 8, 12, 16, 20 and 24
nodes, about n^-0.85.  On undiscounted stationary and on every finite-horizon result the objective is a
quadratic form in the agent's strategy, and a smallest curvature below -1e-4 of the largest prints
`NOT A MINIMUM (the best response of 'trader1' is a saddle: its loss is not convex in its own
strategy, smallest curvature -1.8e-03 of the largest)`.  A negative direction that is positive on a
window longer by two lags is reported as `window edge: ... a truncation of the lagged loss terms at the
edge, not a saddle` instead.

**Settled.**  A transition whose maps on [T - L, T] are more than 1e-4 of their peak from the
stationary continuation prints `TRANSITION NOT SETTLED by T - L: raise horizon.window (...)`; the
past's and the continuation's own window tails are echoed as `PAST WINDOW TOO SHORT` and
`CONTINUATION WINDOW TOO SHORT`.

**Stability.**  `res.stability()` (or `solve(..., stability=True)`) reports the spectral radius of the
best-response map, `best-response dynamics UNSTABLE (spectral radius 1.400)` above one: the Kyle-Back
equilibrium converges under Anderson mixing while naive best-response adjustment would not find it.

**Sweeps.**  A row whose `change` is more than five times the sweep's median is a `jump`, so a branch
change between neighbouring points is visible instead of plotted as a curve.

**What the model rejects.**  A misspelled key, an unused channel or parameter, a control that does not
enter its owner's loss, a zero noise loading, a lag or delay not below the window, and a singular
best-response system (a control with no quadratic term in its current value, two rows carrying the same
information) are refused with a message naming the item; `model.notes` and `noisestate validate` say
what the numbers are (flow losses per unit time or discounted integrals, a random walk pinned at 0).

## Settings

The tuning constants live in one frozen dataclass, `noisestate.Settings`, each with its default and a
one-line meaning: the outer fixed point (`anderson_m` 15, `anderson_iters` 150, ...), the best response
(`foc_rcond` 1e-10, `foc_dense_max` 500, ...), the second-order check (`second_order_tol` 1e-4, ...),
the means and the result's checks (`resolution_tol` 1e-6, `window_tail_tol` 0.02, `settled_tol` 1e-4,
...).  They are the `settings` field of `Numerics`: `ns.Numerics(settings=ns.Settings(second_order_tol=1e-3))`,
or `{"settings": {"second_order_tol": 1e-3}}`; the fields that differ from the defaults are recorded in
`res.numerics` and the payload, so `refine()`, `stability()` and a re-solve from the payload keep them.
The table is [docs/settings.md](docs/settings.md).

## Errors

Errors are typed by whose problem they are.  A `ValueError` is a model problem: a file that does not
validate, a parameter that is not the model's, a lag off the panels, a singular best-response system, a
solve bound out of range.  A `TypeError` is a wrong argument: an unknown solve option (the message names
the `Numerics` field it belongs to, or the `Numerics(settings=...)` route for a `Settings` field),
`naive_observers` that is not a mapping.  A `NotImplementedError` is a feature the engine does not have
(leads on the finite engines); a past on the cell engine is a `ValueError` from the numerics (`numerics.engine
'cells' solves a finite horizon only`).  A
`RuntimeError` is a solver problem: a Krylov best response not converging, a non-finite value from the
best-response map, and `ConvergenceError` (a `RuntimeError`) from `res.check()`.  A fixed point that does
not reach `tol`, or is stopped at `max_evaluations` or `deadline`, does not raise: `solve()` returns the
result with `converged=False` and `res.message`, `sweep()` records the row with `converged: False` and
goes on, `check()` is the raise.  `converged` means the residual (the norm of the update over the larger
of one and the norm of the iterate) is at or below `tol`.

## The command line

```bash
noisestate validate examples/ch4_kyle_back.yaml               # the schema, then the model's own checks; prints the notes
noisestate solve examples/ch4_kyle_back.yaml -o kb.json --plot kb.pdf --param rho=0.5 --nodes 32
noisestate solve examples/ch3_two_player.yaml --refine --stability --max-evaluations 50 --deadline 60
noisestate sweep examples/ch4_kyle_back.yaml eps 0.2,0.1,0.05 -o sweep.json
noisestate transition examples/ch3_two_player.yaml new.yaml --window 6 --nodes 12 -o change.json
noisestate transition examples/ch3_two_player.yaml new.yaml --settle 1e-4 --nodes 12     # the horizon found by the march in T
noisestate schema model > model.schema.json               # JSON Schema (draft 2020-12); also: schema payload
noisestate plot kb.json kb.pdf                            # re-solves the payload's model under its recorded options
noisestate --version
```

`solve` takes `--engine`, `--window`, `--tol` and `-v` as well.  The exit status is 0 for a converged
solve (a sweep: every point converged), 1 for a solve that ran but did not converge (the summary is
still printed and `-o` still written), and 2 for a usage error or an error the package raises, printed
as `error: ...` on stderr.  The JSON written by `-o` is `res.to_dict()`, documented key by key in
[docs/payload.md](docs/payload.md); it validates against `noisestate.schema("payload")`.

## Where things are

* [docs/model_file.md](docs/model_file.md): every key of a model file, with its type and default.
* [docs/payload.md](docs/payload.md): every key of `to_dict()` and the CLI's JSON.
* [docs/transitions.md](docs/transitions.md): the transition engine in full, with the two examples.
* [docs/guards.md](docs/guards.md): the checks, their flags, thresholds and advice.
* [docs/settings.md](docs/settings.md): the tuning constants.
* [docs/validation.md](docs/validation.md): what the tests reproduce, with the numbers; the resolved open items.
* [docs/method.md](docs/method.md): how it works, the means, the grids, the stability guarantees.
* [docs/limits.md](docs/limits.md): what the grammar and the engines do not do.
* [docs/architecture.md](docs/architecture.md): the modules, one best response and one transition through them.
* [docs/design/](docs/design/README.md): the design record of the transition, size and consolidation stages.
* [CHANGELOG.md](CHANGELOG.md): every change by release.
