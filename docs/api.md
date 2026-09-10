# The API, by what you are trying to do

Everything `import noisestate as ns` gives you, grouped by the job it does.  `ns.__all__` has 44
names; this page covers all of them, plus the methods on the objects they return.

The shortest useful path is three calls:

```python
import noisestate as ns
res = ns.solve("examples/ch1_two_player_finite.yaml")   # solve
res.require_ok()                                        # refuse it unless every guard passed
res.costs, res.kernel("D1")                             # read it
```

Everything else is a variation on getting the model in, or getting more out.

The middle line is the one worth keeping.  A solve can converge to its tolerance and still not be
a number you should quote, and `require_ok()` is what tells them apart:

```python
>>> ns.solve("examples/ch3_two_player.yaml").require_ok()
ConvergenceError: ch3_two_player: converged, but WINDOW TOO SHORT (a kernel still moves by
2.4% of its peak over the last tenth of the window: raise horizon.window)
```

That solve converged.  `res.converged` is `True` and `res.require_converged()` passes; only `require_ok()`
refuses it, and the fix is in the message.  See [guards.md](guards.md) for every such check.

## 1. Getting a model in

A model is a `Model`.  Four spellings produce one, and `solve()` accepts any of them directly, so
these matter when you want the model object itself.

| call | use it when |
|---|---|
| `ns.load(path)` | you have a YAML model file.  A relative `horizon.past.model` resolves from that file's directory |
| `ns.Model.load(path)` | the same thing spelled as a classmethod |
| `ns.Model.from_dict(d)` | you have the file structure as a Python dict — generated models, tests, anything programmatic |
| `ns.Model(name=..., states=..., agents=..., horizon=...)` | you built the pieces with the expression API below |
| `ns.ModelBuilder(name, **params)` | you want the fluent form producing the same structure as the file |

### The expression API

For writing a model as equations instead of a dict.  Build the pieces, then hand them to `ns.Model`.

| call | use it when |
|---|---|
| `ns.Param(name, value)` / `ns.Param.many(**kw)` | a named parameter you will later sweep or override.  Arithmetic on it stays symbolic, so the file keeps the expression |
| `ns.shocks("w0", "w1")` | the Brownian channels, as a namespace: `w = shocks("w0"); w.w0`.  Channel order is the order given here |
| `ns.State("X")`, then `X.drift = D + sigma * w.w0` | a state and its law of motion |
| `ns.Control("D")` | a control; it belongs to whichever `Agent` lists it |
| `ns.Signal("y", sqrt(p) * X + w.w1, delay=0.0)` | one observed row.  `delay` makes it an observation lag |
| `ns.Agent(name, controls, signals, loss, myopic=False, naive_observers=None)` | an agent's controls, information and quadratic loss.  `myopic` ignores its own continuation effects; `naive_observers` is the stationary engine's option |
| `ns.define(name, expr)` | a named linear expression you want reported as its own kernel, e.g. an index of several prices |
| `ns.sqrt exp log sin cos tanh` | these functions of a `Param` expression, kept symbolic (plain numbers when given numbers) |
| `ns.Stationary(window, discount)` | an infinite-horizon problem: lag window `L`, discount 0 meaning average cost |
| `ns.Finite(T, discount)` | a finite horizon `[0, T]` |
| `ns.Transition(T, past, continuation, discount, window)` | a regime change on `[0, T]` from a past stationary model to this one |

## 2. Understanding a model you did not write

| call | use it when |
|---|---|
| `model.describe()` | you want the whole model as equations, delays, losses and the conventions that apply, with no solve.  Renders as HTML in a notebook, plain text elsewhere |
| `model.notes` | just the conventions that are easy to misread for this particular model |
| `model.validate()` | every structural rule, checked in a fixed order.  `from_dict` already runs it; call it after hand-editing |
| `model.to_dict(numeric=False)` | the file structure back out.  `numeric=True` resolves parameter expressions to numbers — the form to fingerprint when you want to prove two models are identical |
| `model.state_names`, `.control_names`, `.def_names`, `.channels` | the names, for building something over them |
| `model.owner_of(control)` | which agent owns a control |
| `model.all_lags()` | every distinct positive lag or observation delay, e.g. to check a window covers them |
| `model.drives_means` | whether anything moves the means at all (a linear loss term, a drift constant).  If false, only second moments matter |
| `model.expand(expr)`, `model.constant(expr)` | expanding an expression into primary atoms, and reading off its constant term |

## 3. Changing a model without rewriting it

Every one returns a **new** model and leaves the original alone.  This is how you run an experiment:
build one model, then vary one thing.

| call | use it when |
|---|---|
| `model.with_params(**values)` | a different parameter value.  The other parameters keep their expressions and are re-evaluated |
| `model.with_horizon(**fields)` | a different window, horizon, discount, past or continuation |
| `model.with_finite(T)` / `model.with_stationary(window)` | the two horizon changes you make most, spelled directly.  `ModelBuilder` keeps plain `.finite()`/`.stationary()`: those mutate the builder, these return a copy, and the `with_` prefix is what marks the difference |
| `model.with_numerics(numerics_or_fields)` | a different grid, engine or tolerance.  The problem is unchanged; only how it is solved |
| `model.with_signal(name, drift=, noise=, audience="all", delay=0)` | an information experiment: add one observed row.  New noise channels are added for you; `audience` is `"all"`, an agent name, or a list |
| `model.with_signals(rows, audience="all")` | several rows at once |
| `model.save(path)` | write it back out as YAML, parameter expressions intact |

## 4. Solving

| call | use it when |
|---|---|
| `ns.solve(model, numerics=None, **kw)` | the normal entry point.  Takes a `Model`, a `ModelBuilder`, a dict or a path |
| `model.solve(numerics=None, **kw)` | the same, as a method.  An expression model's `naive_observers` are passed along for you |
| `ns.Numerics(engine=, nodes=, tol=, ...)` | the numerical choices, separately from the model.  Every field is optional and `None` means "keep the model's own" |
| `ns.using_settings(**overrides)` | a context manager replacing the solver's defaults for a block: `with ns.using_settings(second_order_tol=1e-3): ...` |
| `ns.Settings(...)` | the full frozen tunable set, when you want to hold one rather than a context |
| `ns.solver(model, numerics, **kw)` | you want the engine object itself rather than a result — stepping it by hand, or reusing it |
| `ns.ENGINE_CLASSES`, `ns.engines` | the three engines by name, when selecting one programmatically: `stationary` &rarr; `StationarySolver`, `spectral` &rarr; `SpectralFiniteSolver`, `cells` &rarr; `FiniteSolver` |
| `ns.StationarySolver(model, ...)` | construct the infinite-horizon engine directly, when `ns.solver` is more indirection than you want |
| `ns.SpectralFiniteSolver(model, ...)` | the finite-horizon spectral engine, directly.  It also takes `past` and `continuation` |
| `ns.FiniteSolver(model, ...)` | the cell engine, directly (`numerics.engine="cells"`) |
| `ns.clear_grid_cache()` | free the cached grids.  Relevant when measuring memory, not in normal use |

Useful `solve()` keywords: `init` and `start` (warm starts), `tol`, `max_evaluations`, `deadline`
(bounds), `diagnostics=False` (skip the guards when you are solving thousands of times),
`refine=True` (also solve on a finer grid and report the change), `stability=True` (compute the
best-response spectrum during the solve), `naive_observers`, `past` and `continuation` (engine
options), and `verbose` / `progress` for a running commentary.

## 5. Reading a result

Every engine returns a `ns.Result`, so `isinstance(res, ns.Result)` holds whatever solved it.
The concrete class per engine is an
internal detail, but it is where `summary()` and `plot()` are actually implemented, which is why
those two differ in what they show from one engine to the next.

Most of what you want are attributes, not calls.

| attribute | use it when |
|---|---|
| `res.costs` | the equilibrium cost per agent — usually the number you came for.  `res.cost_parts` splits it |
| `res.kernel(name, channel=None)` | the closed-loop response of a state, control or definition to a shock, as a `Kernel` |
| `res.strategy_kernel(control, channel=None)` | the strategy itself rather than the closed-loop outcome |
| `res.means`, `res.mean_times`, `res.paths`, `res.times` | where the state actually sits, and its path over time on a finite horizon.  `res.has_means` says whether any of it is nonzero |
| `res.foc[agent][control]` | the first-order condition split into `physical` (what it would be if nobody reacted) and `wedge` (what remains because they do).  The wedge is the strategic correction |
| `res.axes` | the coordinate arrays the kernels are indexed by |
| `res.maps`, `res.world` | the raw solved objects, for warm starts and for code that works below the kernel level |
| `res.converged`, `res.residual`, `res.evaluations`, `res.seconds`, `res.message` | what the solve did |
| `res.numerics` | the grid and tolerances it actually ran with, resolved |
| `res.summary()` | all of the above as readable text |
| `res.plot(path)` | the standard figures for this engine (matplotlib) |
| `res.to_dict()` | a JSON-serialisable view, with provenance: version, parameter values, the model |

A `Kernel` is an ndarray with four additions: `.axes` (what its indices mean), `.at(*coords)`
(interpolated at a point, using the engine's own interpolation), `.plot(path)`, and `.values` for
the plain array.

## 6. Deciding whether to believe it

The distinction this package cares about most.  A solve that converged is not the same as a solve
you should quote.

| call | use it when |
|---|---|
| `res.require_converged()` | returns `res`, or raises `ConvergenceError` if it did not reach tolerance.  Convergence only |
| `res.require_ok()` | returns `res`, or raises if it did not converge **or any guard failed**.  This is the one to use before quoting a number |
| `res.status` | `{"ok": ..., "flags": [...]}` without raising |
| `res.diagnostic_rows()` | every check as a row: `{name, value, threshold, ok, flag, advice}` |
| `res.diagnostic_records()` | the same rows with presentation fields added (`category`, `severity`, `meaning`, `action`, `suggested_options`) |
| `res.diagnostic_summary(detailed=False)` | those rows grouped and readable |
| `res.diagnostic_verdict(category, exclude=())` | one category (`"solve"`, `"numerics"`, `"equilibrium"`) collapsed to `True` / `False` / `None`.  `exclude` matches the root of a name, before any `:` |
| `res.resolution_ok` | the single question "is this grid too coarse for the answer it reports" |
| `res.refine(factor=1.5)` | re-solve finer and report the relative change.  A guard standing while its number shrinks means the grid, not the model |
| `res.stability(untied=True)` | the spectral radius of the best-response Jacobian.  Above 1 the equilibrium is not reachable by best-response dynamics — often the finding, not a defect |

`ConvergenceError` is raised only by `require_converged()` and `require_ok()`.  `solve()` itself returns a
result with `converged=False`, so a loop over many models does not abort on one bad case.

## 7. Families of solves

| call | use it when |
|---|---|
| `ns.sweep(model, param, values, ...)` | one parameter along a path.  Warm-starts each point from a secant predictor of the last two, which is what makes it robust at hard points.  Returns `SweepPoint` rows with `result`, `seconds`, `evaluations`, `converged`, `change` and a `jump` flag for a possible branch change |
| `model.sweep(p1=[0.3, 1, 3])` | the same, as a method with the parameter as a keyword |
| `ns.compare({name: model, ...}, baseline=, stability=)` | several **structurally different** models against one baseline.  Requires the same agents, horizon kind, discount and window, so the cost changes compare like with like |
| `ns.transition(old, new, T)` | the path from one stationary regime to another over `[0, T]` |
| `ns.transition_gap(old, new)` | the `T = 0` case: every agent applies the new regime's stationary rules from date zero, inheriting the old regime's state and observations, with no transition solved.  The benchmark a real transition is measured against |

`compare()` returns a `ComparisonResult`: index it by name (`study["balanced"]`), iterate it for the
names, `study.summary()` for the table, `study.to_dict(include_results=False)` for a compact payload.
Each entry is a `ScenarioResult` with `.result` (the ordinary result, nothing hidden), `.total_cost`,
`.total_change`, `.total_change_fraction`, `.cost_changes` (per agent against the baseline) and
`.dynamics` (the response classification when `stability=True`).

## 8. Getting data out

| call | use it when |
|---|---|
| `res.to_dict()` | the result as JSON-ready data |
| `model.to_dict()` / `model.save(path)` | the model as data, or back to a file |
| `ns.schema("model")` / `ns.schema("payload")` | the JSON Schema of either, for validating generated files or typing a front end |
| `ns.read_yaml(path)`, `ns.read_json(path)` | read either without building a model |

## 9. The command line

Installed as `noisestate`.  Every subcommand takes `--help`.

| command | use it when |
|---|---|
| `noisestate solve model.yaml` | solve a file; `-o out.json` writes the payload |
| `noisestate validate model.yaml` | check a file against the schema and the model's own rules, and print its structure |
| `noisestate describe model.yaml` | the model as equations and conventions, no solve |
| `noisestate sweep ...` | solve along one parameter with warm starts, writing a JSON list |
| `noisestate transition old.yaml new.yaml` | the transition between two regimes |
| `noisestate plot result.json` | plot from a saved payload rather than re-solving |
| `noisestate plot-sweep sweep.json` | costs, residuals, strategy changes and runtime from a sweep |
| `noisestate schema` | print either JSON Schema |

`solve` and `transition` take `--require-ok`, which makes a failed guard a non-zero exit status
rather than only a failed solve — the form to use in a script or a CI job.

## Which of these three do I want?

The three ways to run many solves are easy to confuse:

- **`sweep`** — *one* model, one parameter moving along a path.  Warm starts make it cheap; the
  `jump` flag warns you when the equilibrium may have changed branch.
- **`compare`** — *several* models that differ structurally (different information, different
  visibility), each solved independently, costs measured against a named baseline.
- **`transition`** — *two* models and the path between them, as a dynamic problem in its own right.


## Appendix: the 0.7 renames

Names that were correct but inconsistent were renamed in 0.7.  Every old spelling still works,
warns, and names its replacement; they are removed in 0.8.

| old | new | why |
|---|---|---|
| `ns.ENGINES` | `ns.ENGINE_CLASSES` | case-only collision with the `ns.engines` module |
| `ns.settings(...)` | `ns.using_settings(...)` | case-only collision with the `ns.Settings` class, *and* with the `noisestate.settings` submodule |
| `ns.make_solver` | `ns.solver` | the only `make_*`; now parallel to `ns.solve` |
| `ns.BaseResult` | `ns.Result` | an alias since 0.6 |
| `res.check()` | `res.require_converged()` | it raises, so it takes the `require_*` prefix, and now pairs with `require_ok()` |
| `res.diagnose()` | `res.diagnostic_rows()` | joins `diagnostic_records` / `diagnostic_summary` / `diagnostic_verdict` |
| `res.category_verdict()` | `res.diagnostic_verdict()` | same family |
| `res.action_kernel()` | `res.strategy_kernel()` | neither old name said which was the strategy and which the closed loop |
| `res.grid_info()` | `res.grid_summary()` | the only `*_info` |
| `res.means_driven` | `res.has_means` | a bool sitting among `means` and the mean paths, read as data |
| `res.means_t` | `res.mean_times` | it is the *time nodes* of the mean paths, not means over time |
| `model.means_driven` | `model.drives_means` | the structural question, as a predicate |
| `model.finite(T)` | `model.with_finite(T)` | returns a copy, so it joins the `with_*` family |
| `model.stationary(w)` | `model.with_stationary(w)` | the same |
| `model.owner(c)` | `model.owner_of(c)` | reads as a lookup rather than a noun |

The payload key `means_t` is **unchanged**.  The JSON is a wire format read by saved files and by
front ends, so it carries a different compatibility promise from the Python attribute; renaming it
belongs to a payload-schema version, not to this one.
