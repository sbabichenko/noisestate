# The model file

The reference of every key of a model file, generated from `noisestate.schema("model")` (JSON Schema, draft
2020-12; `noisestate schema model` prints it).  A file is validated against it first (`noisestate validate`,
`noisestate.schema.validate(doc, "model")` lists the violations with their paths), then against the model's
own checks ([guards.md](guards.md), "What the model rejects").  Unknown keys are errors everywhere.  Every
coefficient may be a number or an expression in the parameters (`"sqrt(p1)"`, `"-gamma1"`).

## The keys

| key | type | meaning | default |
|---|---|---|---|
| `name` | string |  | `model` |
| `params` | map of number or expression | parameters, evaluated in order (a later one may use an earlier one) | none |
| `channels` | list of string | the Brownian channels | none (every one listed must load something) |
| `states` | map of object |  |  |
| `states.<name>.drift` | linear expression | a linear expression: {atom: coef} (an atom is name or name@lag; const for a constant), or [[coef, atom], ...] | empty |
| `states.<name>.noise` | linear expression | a linear expression: {atom: coef} (an atom is name or name@lag; const for a constant), or [[coef, atom], ...] | empty |
| `states.<name>.initial` | number or expression | finite horizon only; moves the means | 0 |
| `definitions` | map of linear expression |  | none |
| `agents` | map of object |  |  |
| `agents.<name>.controls` | list of string (required) |  |  |
| `agents.<name>.signals` | map of object |  | none |
| `agents.<name>.signals.<name>.drift` | linear expression | a linear expression: {atom: coef} (an atom is name or name@lag; const for a constant), or [[coef, atom], ...] |  |
| `agents.<name>.signals.<name>.noise` | linear expression | a linear expression: {atom: coef} (an atom is name or name@lag; const for a constant), or [[coef, atom], ...] |  |
| `agents.<name>.signals.<name>.delay` | number or expression | observation delay | 0 |
| `agents.<name>.loss` | list of list of number or expression |  | none (a control must enter its owner's loss) |
| `agents.<name>.myopic` | boolean |  | false |
| `ties` | list of list of string | groups of agents sharing one strategy | none |
| `horizon` | object | the economics of time: the kind, the discount, the window, a transition's past and continuation |  |
| `horizon.kind` | `stationary` \| `finite` \| `transition` \| `finite_cells` | finite_cells is deprecated: kind finite with numerics.engine cells (read until 0.6) | `stationary` |
| `horizon.discount` | number or expression | a number, or an expression in the parameters | 0 |
| `horizon.window` | number or expression | the lag window L (stationary) or the horizon T (finite, transition) | 8.0 (`ModelBuilder.stationary`), 1.0 (`finite`, `transition`) |
| `horizon.past` | object | kind transition only |  |
| `horizon.past.model` | string or object | the old stationary model: a path (relative to the file) or an inline model |  |
| `horizon.past.initial` | list of object | initial shocks {name, loads, rows} |  |
| `horizon.past.initial[].name` | string (required) |  |  |
| `horizon.past.initial[].loads` | map of number or expression |  |  |
| `horizon.past.initial[].rows` | map of number or expression |  |  |
| `horizon.continuation` | `stationary` \| `end` | kind transition only; default stationary | `stationary` |
| `horizon.stationary` | object | kind transition only: the continuation's stationary solve |  |
| `horizon.stationary.window` | number or expression | must equal the past's window | the past's window |
| `horizon.stationary.nodes` | integer >= 2 | **deprecated**: numerics.continuation_nodes (read until 0.6) |  |
| `horizon.nodes` | integer >= 2 | **deprecated**: this key now lives under numerics: (read until 0.6) |  |
| `horizon.unit` | number or expression | **deprecated**: this key now lives under numerics: (read until 0.6) |  |
| `horizon.unit_range` | number or expression | **deprecated**: this key now lives under numerics: (read until 0.6) |  |
| `horizon.breakpoints` | list of number or expression or null | **deprecated**: this key now lives under numerics: (read until 0.6) |  |
| `numerics` | object | how the model is solved: the engine, the grid, the fixed point's options, the settings |  |
| `numerics.engine` | `stationary` \| `spectral` \| `cells` | default from horizon.kind: stationary -> stationary, else spectral | `stationary` for kind stationary, else `spectral` |
| `numerics.nodes` | integer >= 2 | nodes per panel (stationary) or per side of each piece (spectral); cells on the cell engine; default 16 | 16 (12 from `ModelBuilder.transition`, the CLI's `transition`) |
| `numerics.unit` | number or expression | the panel unit: every lag and delay must be a multiple of it | the smallest lag |
| `numerics.unit_range` | number or expression | the age (stationary) or time (spectral) up to which the panels are unit panels | the window |
| `numerics.breakpoints` | list of number or expression | an explicit panel sequence from 0 to the window | the lags' multiples closed under every lag and delay |
| `numerics.continuation_nodes` | integer >= 2 | a transition's stationary continuation solved at this many nodes | `nodes` |
| `numerics.tol` | number or expression | the fixed point's tolerance (default 1e-10 stationary, 1e-8 finite) | 1e-10 stationary, 1e-8 finite |
| `numerics.damping` | number or expression | the Anderson mixing weight | the engine's own |
| `numerics.max_newton` | integer >= 0 | Newton-Krylov polish steps at most | the engine's own |
| `numerics.variable` | `actions` \| `maps` | the fixed point's iterate | `actions` (`maps` with ties) |
| `numerics.settings` | object | the tuning constants (noisestate.Settings): the fields that differ from the defaults | `noisestate.Settings()` ([settings.md](settings.md)) |

The fields of `numerics.settings` are those of `noisestate.Settings`: [settings.md](settings.md).

## What the keys mean

* **Atoms.** `name` is a state, a control, or a definition; `name@tau` is its value
  `tau` earlier (a lag), `name@-tau` its value `tau` later (a lead; allowed only in a loss
  cross term with the agent's own current control, see [limits.md](limits.md)).
* **States.** `drift` is linear in atoms (other states, controls, lagged controls,
  definitions), plus an optional constant under the key `const`
  (`drift: {X: -a, D: 1.0, const: 0.3}`), which moves the means only; `noise` gives the
  loading on each channel; on a finite horizon an optional `initial` value
  (`X: {drift: ..., noise: ..., initial: 1.0}`, zero by default) starts the mean path there
  (a stationary model, which has no initial time, rejects it).
* **Definitions.** Named linear combinations of atoms, usable anywhere:
  `Pidx: {P0@tau: 0.333, P1@tau: 0.333, P2@tau: 0.333}`.
* **Signals.** Each row has a linear `drift` (states, other agents' controls,
  definitions), a `noise` loading, and an optional observation `delay`.  Rows
  with a pure noise loading and no drift make a channel directly observed.
* **Loss.** A list of terms `[coef, a, b]` (quadratic) and `[coef, a]` (linear);
  the flow loss is their sum and the agent minimises `E int e^{-rho t} loss dt`.
  A target `theta` on `X` is `(X - theta)^2` less its constant: `[1, X, X]` and
  `[-2*theta, X]`.  Linear terms move only the means (below).
* **Ties.** `ties: [[firm0, firm1, firm2]]` makes the listed agents share one
  strategy (a symmetric equilibrium): only the first is solved for.
* **Horizon.** The economics of time: `stationary` with `discount` and `window` (lag window L);
  `finite` with `window` = T; `transition` with its `past` and `continuation` ([transitions.md](transitions.md)).
* **Numerics.** How it is solved, an optional block with the fields of `noisestate.Numerics`:
  `engine` (`stationary`, `spectral`, or `cells` for the first-order cell scheme on a finite
  horizon; default from the kind), `nodes` per panel (stationary) or per side of each piece of
  the triangle (12 is usually converged; 6-8 when delays cut the domain into small pieces; the
  panels are the lags' multiples closed under every lag, see [limits.md](limits.md) for a window that is not a
  multiple of them), `unit`/`unit_range`/`breakpoints` (panels are aligned to the delays
  automatically), a transition's `continuation_nodes`, `tol`, `damping`, `max_newton`,
  `variable`, and `settings` ([settings.md](settings.md)).  `solve(model, numerics)` lays a `Numerics` (or a dict
  of its fields) over the file's block.  The keys once nested under `horizon:` (`nodes`, `unit`,
  `unit_range`, `breakpoints`, `stationary: {nodes}`, `kind: finite_cells`) are still read, with
  a deprecation note in `model.notes`, until 0.6.
* Coefficients may be numbers or expressions in the parameters (`"sqrt(p1)"`).

The same structure is available from Python through `ModelBuilder` (see
`examples/make_ch5_cycle_market.py`, which builds an N-firm cycle in a loop); `stationary()`,
`finite()` and `transition()` take the horizon and `nodes`, and `numerics(**fields)` sets the rest
of the block.

## Deprecated keys

The keys once nested under `horizon:` (`nodes`, `unit`, `unit_range`, `breakpoints`, `stationary: {nodes}`) and
the kind `finite_cells` are still read and mapped onto the `numerics:` block, with one deprecation note each
in `model.notes`, until 0.6; a nested key that disagrees with the block is an error.  `kind: finite_cells` is
`kind: finite` with `numerics: {engine: cells}`.
