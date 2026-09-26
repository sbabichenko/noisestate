# API reference

Everything `import noisestate as ns` gives you, grouped by the job it does. `ns.__all__` has 47
names; this page covers all of them, plus the methods on the objects they return.

The shortest useful path is three calls:

```python
import noisestate as ns
res = ns.solve("examples/ch1_two_player_finite.yaml")   # solve
res.require_ok()                                        # refuse it unless every required check passed
res.costs, res.kernel("D1")                             # read it
```

`require_ok()` checks convergence and the required diagnostics. A converged result can still fail
those checks:

```python
>>> ns.solve("examples/ch3_two_player.yaml").require_ok()
DiagnosticsError: ch3_two_player: converged, but window [failed] WINDOW TOO SHORT
(a kernel still moves by 2.4% of its peak over the last tenth of the window: raise horizon.window)
```

That solve converged — `res.converged` is `True` and `res.require_converged()` passes. Only
`require_ok()` refuses it, and the fix is in the message.

## 1. Getting a model in

| call | use it when |
|---|---|
| `ns.example(name)` | the path of a shipped model file — `ns.load(ns.example("ch4_kyle_back"))`. `ns.examples()` lists the seven names. They install with the package, so this works from a plain `pip install` |
| `ns.load(path)` | you have a YAML model file. A relative `horizon.past.model` resolves from that file's directory |
| `ns.as_model(x)` | a Model from a Model, a dict (either form) or a path: what every function taking a model does with it |
| `ns.Model.from_dict(d)` | you have the file structure as a dict — generated models, tests, anything programmatic |
| `ns.Model.from_dict(d)` with equations | the same dict written as equations: `shocks: [W0]`, `states: {X: "(D1 + D2) dt + sigma dW0"}`, `observes:`, `loss: "(X - b)^2 + r D^2"` (docs/model_file.md). `load()` reads it from a file |
| `ns.Game(states, agents, T=..., window=..., discount=..., nodes=..., horizon=..., name=..., params=...)` | you wrote the equations in Python (below): a finite game with `T`, a stationary one with `window`, or `horizon=ns.Transition(...)`. Returns a `Model`; `game.solve()` solves it. `ns.Model(...)` itself is not a constructor |

### The expression API

| call | use it when |
|---|---|
| `ns.params(**kw)` (`ns.Param(name, value)`, `ns.Param.many(**kw)`) | named parameters you will sweep or override. Arithmetic on them stays symbolic, so the saved file keeps `sqrt(p1)` rather than `1.732…` |
| `dW0, dW1 = ns.shocks(2)` (or `ns.shocks("w0", "w1")`, a namespace) | the Brownian shocks, named W0, W1, ...; their order is the model's |
| `ns.dt` | the time increment: `(D1 + D2) * dt` is a drift, which with the shocks makes an `ns.Differential` |
| `ns.State("X")`, then `X.d = (D1 + D2) * dt + sigma * dW0` | a state and its law of motion (`X.drift = ...` takes the drift and noise as one expression) |
| `ns.Control("D")` | a control; it belongs to whichever `Agent` lists it |
| `ns.Signal(name, expr, delay=0.0)` | one named, possibly delayed, observed row. The same object `with_signal()` takes |
| `ns.Agent(name, controls, observes=..., loss=..., myopic=False, terminal=None)` | an agent's controls, what it observes (an expression, a list, a dict of named signals, or Signals) and its quadratic loss, whose constant is kept as part of the cost |
| `ns.define(name, expr)` | a named linear expression reported as its own kernel |
| `ns.sqrt exp log sin cos tanh` | these functions of a `Param` expression, kept symbolic |

### The horizon: two quantities, three types

**`window` is the lag-truncation length L. `T` is the terminal time.** They are different objects
and are never aliased — one field used to hold both, which left a transition's lag window with
nowhere to live.

| type | carries | `extent` |
|---|---|---|
| `ns.Stationary(window, discount)` | `window` only | the window |
| `ns.Finite(T, discount)` | `T` only | `T` |
| `ns.Transition(T, past, continuation, discount)` | **both** — its window is the past's | `T` |

`horizon.extent` is the length of the primary computational axis: it *selects* whichever of the two
an operation needs. Grid construction and the lag bounds want it; a transition's continuation wants
`window` and must never take `extent`, which there is `T`. Accessing the length a kind does not have
raises `AttributeError` rather than answering `None`.

## 2. Understanding a model you did not write

| call | use it when |
|---|---|
| `model.describe()` | the whole model as equations, delays, losses and the conventions that apply, with no solve. HTML in a notebook, text elsewhere |
| `model.notes` | just the conventions that are easy to misread here |
| `model.validate()` | every structural rule, in a fixed order. Raises. `from_dict` already runs it |
| `model.to_dict(numeric=False)` | the file structure back out. Carries everything a solve depends on, including definitions and ties; omits only provenance (`source`, `remarks`, `deprecations`). `numeric=True` resolves parameter expressions to numbers |
| `model.state_names` / `.control_names` / `.def_names` / `.shocks` | the names |
| `model.owner_of(control)` | which agent owns a control |
| `model.all_lags()` | every distinct positive lag or observation delay |
| `model.drives_means` | whether anything moves the means at all — a **bool** |
| `model.expand(expr)` / `.constant(expr)` | expanding an expression into primary atoms; its constant term |

## 3. Changing a model without rewriting it

Every one returns a **new** model, shares no mutable structure with it, and leaves it unchanged.

| call | use it when |
|---|---|
| `model.with_params(**values)` | different parameter values; the others keep their expressions |
| `model.with_horizon(horizon)` | a different horizon, **as an object**: `with_horizon(Finite(T=1.0))`. The horizon is replaced, not patched, so nothing carries over by accident |
| `model.with_stationary(window)` / `.with_finite(T)` / `.with_transition(T, past)` | the three kinds, spelled directly |
| `model.with_numerics(numerics_or_fields)` | a different grid, engine or tolerance |
| `model.with_signal(name, equation, delay=0, audience="all")` or `model.with_signal(Signal(...))` | add one observed row, written as in a file's `observes:` (`"(D1 + D2) dt + 0.5 dw_flow"`). A shock it names that the model lacks is added. Adding a name an agent already has is an error |
| `model.with_signals(rows)` | several at once: a mapping of names to equations, or a list of Signals |
| `model.without_signal(name, audience="all")` | remove a row, and the shocks it alone loaded. The inverse of adding one |
| `model.save(path)` | write it back as YAML, parameter expressions intact |

Adding a row an agent already has is an **error**, never a silent replacement — use
`without_signal()` first.

## 4. Solving

| call | use it when |
|---|---|
| `ns.solve(model, numerics=None, **kw)` | the standard entry point. Returns a `Result`; it does **not** raise when the solve fails to converge |
| `model.solve(numerics=None, **kw)` | the same as a method |
| `ns.Numerics(engine=, nodes=, tol=, …)` | the numerical choices. Every field optional; `None` keeps the model's own |
| `ns.Settings(...)` | the full frozen tunable set |
| `ns.engines` | the advanced namespace: `engines.stationary`, `engines.spectral`, and `engines.solver(model, numerics, **kw)` to construct one directly |
| `ns.clear_grid_cache()` | free the cached grids. For measuring memory |

**Warm starts are two arguments because they are two things.** `start_from` takes an **object**
(action kernels or raw maps); `start_policy` takes a **name** (`"zero"`, `"coarse"`,
`"stationary"`). Supplying both raises — a precedence rule would silently discard one of them.

Other keywords: `tol`, `max_evaluations`, `deadline`, `diagnostics=False`, `refine=True`,
`stability=True`, `past`, `continuation`, `verbose`, `progress`.

## 5. Reading a result

Every engine returns a `ns.Result`. The concrete class per engine is internal, but it is where
`summary()` and `plot()` are implemented, so those differ in *contents* between engines while their
signatures and return types do not.

| attribute | use it when |
|---|---|
| `res.costs` | the equilibrium cost per agent. `res.cost_parts` splits it; `res.cost_kind` names the convention |
| `res.kernel(name, shock=None)` | the **closed-loop** response to a shock, as a `Kernel` |
| `res.strategy(control, shock=None)` | the **strategy** on the noise-state, D: the weight the action puts on the agent's estimate of each shock (finite games without delays) |
| `res.estimate(agent, name, shock=None)` | the agent's estimate of a quantity, as a kernel |
| `res.response(q, to=shock, at=s, seen_by=agent).over(t)` | follow one shock through time: the response of q (or an agent's estimate of it) |
| `res.means` / `.mean_times` | the means and the time nodes they sit on. `res.has_means` is a bool |
| `res.paths` / `.times` | paths over time on a finite horizon, and their nodes |
| `res.foc[agent][control]` | the FOC split into `physical` (if nobody reacted) and `wedge` (because they do) |
| `res.axes` / `.map_axes(delay)` | what the kernel indices mean; where a delayed row's map values belong |
| `res.maps` / `.world` | the raw solved objects |
| `res.converged` / `.residual` / `.evaluations` / `.seconds` / `.message` | what the solve did |
| `res.numerics` / `.grid_summary()` | the grid and tolerances it ran with |
| `res.summary()` / `.plot(path)` / `.to_dict()` | readable text, figures, JSON-ready data |

`ns.Kernel` is an ndarray with four additions: `.axes`, `.at(*coords)`, `.plot(path)`, `.values`.

## 6. Assessing a result

The distinction this package cares about most. **A status describes one check; a policy says which
checks a use requires; an assessment is what the two produce together.**

`ns.Status` has six values, and the distinctions are load-bearing:

| status | means |
|---|---|
| `PASSED` / `FAILED` | it ran, and the model met it / did not |
| `SKIPPED` | this engine can compute it here; it was not run (`diagnostics=False`) |
| `UNSUPPORTED` | this engine **cannot** compute it — a property of the engine |
| `NOT_APPLICABLE` | the check has no meaning for this model — a property of the model |
| `MISSING` | applicable and supported, was to run, produced no record — a package defect |

`NOT_APPLICABLE` and `UNSUPPORTED` are never merged: the first means nothing is missing, the second
means something is.

| call | use it when |
|---|---|
| `res.diagnostics.rows` | every emitted check with its presentation fields |
| `res.diagnostics.statuses` | every applicable check by root name, with its `Status` |
| `res.diagnostics.assess(policy)` | an `Assessment`: `accepted`, `policy`, `blocking`, `statuses`, `uncomputed` |
| `res.diagnostics.summary(detailed=False)` | those rows grouped and readable |
| `res.diagnostics.flags` | the flag text of the checks that failed |
| `res.require_converged()` | returns `res`, or raises `ConvergenceError`. Convergence only |
| `res.require_ok(policy=Policy.PUBLICATION)` | returns `res`, or raises. **Every required, applicable check must have PASSED** |
| `res.refine(factor=1.5)` | a `Refinement` carrying the finer `Result` in full |
| `res.stability(untied=True, policy=…)` | a `Stability` — see below |

`Policy.PUBLICATION` requires `converged`, `resolution`, `window`, `second_order`, `settled`;
`Policy.EXPLORATORY` requires only `converged` and must be requested explicitly.

**Exceptions are siblings, not nested:**

```
ResultValidationError          a result is not fit for the use asked of it
├── ConvergenceError           the solve did not reach its tolerance
└── DiagnosticsError           it converged, and the assessment was still not accepted
```

A converged result can fail diagnostics, so `except ConvergenceError` must not catch that.

## 7. Stability: a classification of a *verified* equilibrium

`res.stability()` always returns the spectrum — `radius`, `eigenvalues`, `method`,
`fixed_point_residual`, `residual_norm`. The Jacobian at a non-fixed point is a legitimate object.
What is withheld is the **interpretation**: `full_response`, `adjusted_response` and
`adjusted_radius_bound` are present but `None` unless `verified` is true, and `unverified_reasons`
says why.

```
verified = residual_passed and all(statuses[c] is PASSED
                                   for c in applicable(MINIMUM | policy.required, model))
```

`MINIMUM` is `converged, resolution, second_order, window`. A policy may **strengthen** it and can
never weaken it, so "verified" means one thing regardless of who asked.

**Equilibrium validity and response stability are separate findings.** An equilibrium may be
unstable under best-response iteration, and distinct equilibria may have distinct costs; neither
bears on whether a point *is* an equilibrium.

*Open:* the residual tolerance is not yet defined (`docs/api_spec.txt` D4), so nothing is verified
today and no classification appears. The evidence fields are populated and serialised regardless.

## 8. Families of solves

| call | use it when |
|---|---|
| `ns.sweep(model, param, values, **options)` | one parameter along a path, warm-started from a secant predictor; the options are `solve()`'s (`nodes=12`, `max_evaluations=`, `past=`). Returns `SweepPoint`s |
| `model.sweep(p1=[0.3, 1, 3], nodes=12)` | the same, as a method |
| `ns.compare({name: model, …}, baseline=, stability=, **options)` | several **structurally different** models against one baseline |
| `ns.transition(old, new, T)` | the path from one stationary regime to another |
| `ns.transition_gap(old, new)` | the `T = 0` case: everyone applies the new rules at once, inheriting the old state |

`compare()` returns a `ns.ComparisonResult`: index it by name (`study["balanced"]`), iterate it for
the names, `study.summary()` for the table, `study.to_dict(include_results=False)` for a compact
payload.

The containers share conventions, not identity — all are frozen dataclasses carrying `result`,
`seconds`, `evaluations`, `converged` and `to_dict()`:

| type | is | carries besides |
|---|---|---|
| `ns.SweepPoint` | a continuation point | `param`, `value`, `change`, `jump` |
| `ns.ScenarioResult` | a comparison scenario | `total_cost`, `total_change`, `cost_changes`, `dynamics` |
| `MarchPoint` | a settle-march step, on a transition result`s `.march` | `T`, `gap`, `monitor`, `unknowns`, `polish` |

**Which of the three do I want?** `sweep` moves *one* model along a parameter path. `compare` solves
*several* structurally different models independently against a baseline. `transition` is *two*
models and the path between them.

## 9. Getting data out

| call | use it when |
|---|---|
| `res.to_dict()` | the result as JSON-ready data, `payload_version` 2 |
| `model.to_dict()` / `model.save(path)` | the model as data, or back to a file |
| `ns.schema("model")` / `ns.schema("payload")` | the JSON Schema of either |
| `ns.read_json(path)` | read a result's JSON payload without building anything |

The payload's `options.solve` and `options.solver` accept only their documented keys.
Changes to this contract require a payload version change.

## 10. The command line

Installed as `noisestate`; every subcommand takes `--help`.

| command | use it when |
|---|---|
| `noisestate solve model.yaml` | solve a file; `-o out.json` writes the payload |
| `noisestate validate model.yaml` | check against the schema and the model's own rules |
| `noisestate describe model.yaml` | the model as equations, no solve |
| `noisestate sweep …` / `plot` / `plot-sweep` | a parameter path; plots from a saved payload |
| `noisestate transition old.yaml new.yaml` | the transition between two regimes |
| `noisestate schema` | print either JSON Schema |

`solve` and `transition` take `--require-ok`, which makes a failed **guard** a non-zero exit status
rather than only a failed solve.

The two horizon lengths keep their own names here as in Python: `solve --window L` is the
lag-truncation length and `solve --T` the terminal time, and asking for the one a model's kind does
not have is an error naming the other.  `transition --T` is the terminal time and `--past-window L`
the lag window of the old regime, which the continuation shares.
