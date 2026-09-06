# The result payload

What `res.to_dict()`, `noisestate solve -o` and every row of `noisestate sweep -o` carry, generated from
`noisestate.schema("payload")` (`payload_version` 1; `noisestate schema payload` prints the schema,
`noisestate.schema.validate(doc, "payload")` checks a document).  Every key is present unless marked optional;
the optional ones appear when the engine or the options produced them.  `numerics` in the table is the
`numerics` block of the model schema ([model_file.md](model_file.md)); shapes follow `axes`.

## The keys

| key | type | meaning |
|---|---|---|
| `payload_version` | const 1 | the payload format, 1 |
| `version` | string | the package version that wrote it |
| `name` | string | the model's name |
| `engine` | `stationary` \| `spectral` \| `cells` | the engine that solved it (`numerics.engine` resolved) |
| `kind` | `stationary` \| `finite` \| `transition` \| `finite_cells` | the result kind |
| `converged` | boolean | the fixed point reached `tol` |
| `residual` | number | the fixed point's final residual |
| `evaluations` | integer | best-response evaluations (Anderson and the Newton-Krylov polish together) |
| `seconds` | number | wall time of the solve |
| `message` | string | what the outer solver did (empty when converged without event) |
| `params` | map of number | the model's parameters, evaluated |
| `model` | model (see above) | the model spec, `Model.from_dict` rebuilds it |
| `horizon` | object | the model's horizon block (the economics) |
| `numerics` | numerics (see above) | the resolved numerics |
| `axes` | map of list of number | the coordinates of every kernel by name: `age` (stationary); `time`, `age`, `shock_time` node-wise on the spectral triangle (`shock_time` < 0 on a transition's band); `time`, `shock_time` for the cell engine's (N, N) matrices; under `maps`, where each row's map values belong |
| `times` | list of number or null | the time nodes of the paths; null on the stationary engine |
| `options` | object | `numerics` (the resolved Numerics), `solver` (the engine's constructor options) and `solve` (the solve options: start, bounds, diagnostics, ...); `noisestate plot` re-solves under them |
| `grid` | object | the grid's description (`res.grid_info()`): its `kind`, nodes, panels or pieces, window |
| `discount` | number | the discount rate |
| `channels` | list of string | the Brownian channels in kernel column order |
| `agents` | map of object | per agent: its `controls` and its `signals`, each row with its `delay` and the axes of its map (`map_time`, `map_age` or `map_shock_time`, per `map_convention`) |
| `map_convention` | string | one sentence on how a delayed row's map is indexed (`Result.MAP_CONVENTION`) |
| `kernels` | map of map of list of any | per state, control and definition, per channel: the closed-loop kernel on the grid (a list over ages; a list over triangle nodes; an (N, N) list of lists on the cell engine) |
| `maps` | map of map of map of list of any | per agent, per control, per signal row: the raw strategy g[u][r] on the row's map axis |
| `foc` | map of map of map of map of list of number | per agent, per control: `foc`, `physical`, `wedge`, each per channel: the first-order-condition decomposition |
| `costs` | map of number | per agent: the cost (`cost_kind` says what it is) |
| `cost_parts` | map of map of number | per agent: `variance` and `mean` (a transition adds `continuation`, the buffer's cost) |
| `means` | map of number or list of number | per state, control, definition and signal row (`agent.row`): a constant (stationary) or the path on `means_t` |
| `means_t` | list of number or null | the time nodes of the mean paths; null on the stationary engine |
| `representation_error` | map of number | per agent: the representation error of the action kernels on the seen rows |
| `representation_parts` | map of map of number (optional) | a transition's error by region: interior, tip, last window, buffer |
| `diagnostics` | list of object | every row of `res.diagnose()` ([guards.md](guards.md)) |
| `resolution_ok` | boolean or null | the resolution row's verdict (null when not computed) |
| `status` | object | `ok` and the failing `flags` (and the rows) |
| `cost_kind` | string | `flow loss per unit time` (stationary) or the discounted integral (finite) |
| `second_order` | map of object | per agent: `min`, `max`, `ok`, `converged`, and `edge`/`embedded` when the negative direction was re-evaluated on a longer window |
| `notes` | list of string | the model's notes (what the numbers are, deprecations) |
| `refinement` | object (optional) | with `--refine`: `cost_change`, `kernel_change`, `nodes`, `resolved` |
| `window_tail` | number (optional) | stationary: the largest change of a kernel over the last tenth of the window relative to its peak |
| `stability` | object (optional) | with `--stability`: `radius`, `stable`, `method`, `untied` |
| `past` | object (optional) | a transition's past: its provenance (kind, model, params, window, nodes, costs, window tail; no kernels) |
| `settled` | number or null (optional) | a transition's largest relative distance of any map on [T - L, T] from the stationary continuation |
| `continuation` | object (optional) | a transition's continuation: the same provenance |
| `loss_path` | map of list of number (optional) | a transition's E[loss(t)] per agent on `times` |
| `excess_costs` | map of number (optional) | per agent: the discounted integral over [0, T] of E[loss(t)] minus the new stationary flow |
| `old_flows` | map of number (optional) | per agent: the old regime's stationary flow loss |
| `new_flows` | map of number (optional) | per agent: the new regime's stationary flow loss |
| `window` | number (transition) | the horizon T solved on: the file's, or the one the settle march found |
| `march` | array (optional) | the settle march's rows `{T, gap, gap_last, evaluations, seconds, monitor}` (T = 0 first: the pass from the stationary rules) |
| `march_stop` | string or null (optional) | `settled`, `settled at T = 0` or `max_window` (the settled flag then stays) |
| `march_settle` | number or null (optional) | the march's tolerance |
| `excess_windows` | map of list of number (optional) | per agent: the excess's discounted integral per window, the last window [T - L, T] first |
| `excess_costs_tail` | map of number (optional) | per agent: the tail past T, the last window's excess times r / (1 - r) (agents with no factor in (0, 1) are absent) |
| `excess_costs_total` | map of number (optional) | per agent: `excess_costs` + `excess_costs_tail` |
| `excess_tail` | object (optional) | `{source: "loss path" | "march gaps", factor: {agent: r per window}, windows: [[lo, hi], ...]}` |

## Provenance and shapes

A warm-started sweep point costs a handful of best responses, which is what a slider in a front end needs;
`to_dict()` is the payload such a front end would render, and it carries its own provenance: the
package `version`, the `params`, the `model` spec (`Model.from_dict` rebuilds it), the `horizon`,
the `options` (the resolved `numerics`, the engine's `solver` options and the `solve` options), then the grid, kernels per quantity and channel, raw maps, costs
with their variance and mean parts (`cost_parts`), the `means`, and the first-order-condition
decomposition.  `agents` lists each agent's controls and signal rows
with their `delay` and the axes of the row's map (`map_time`, `map_age` or `map_shock_time`, per
`map_convention`), since a delayed row's map is not indexed like an undelayed one: the finite engine
stores it at the shifted time t - delay.  Every row of a sweep's JSON list is one payload plus its `param`, `change` and `jump`.
