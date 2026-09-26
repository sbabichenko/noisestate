# The model file

The reference of every key of a model file.  Maintained by hand against `noisestate.schema("model")`
(JSON Schema, draft 2020-12; `noisestate schema model` prints it, and is the authority if the two
ever disagree).  A file is validated against it first (`noisestate validate`,
`noisestate.schema.validate(doc, "model")` lists the violations with their paths), then against the model's
own checks ([guards.md](guards.md), "What the model rejects").  Unknown keys are errors everywhere.  Every
coefficient may be a number or an expression in the parameters (`"sqrt(p1)"`, `"-gamma1"`).

## Written as equations

A file may instead write each block as its equation; `load()` reads either form into the same model,
`model.save()` writes this one, and `model.to_dict()` returns the grammar below:

```yaml
name: ch1_tracking
params: {p1: 3, p2: 10, r1: 0.1, r2: 0.1, b1: 1, b2: -1, sigma: 1, T: 1}
shocks: [W0, W1, W2]                        # the Brownian shocks; dW0 is the increment of W0
states:
  X: "(D1 + D2) dt + sigma dW0"             # or {d: "...", initial: 0.5}
agents:
  player1: {controls: D1, observes: "sqrt(p1) X dt + dW1", loss: "(X - b1)^2 + r1 D1^2"}
  player2: {controls: D2, observes: "sqrt(p2) X dt + dW2", loss: "(X - b2)^2 + r2 D2^2"}
horizon: {T: T}                             # a finite game on [0, T]; {window: 8} is stationary
numerics: {nodes: 24}
```

In an equation, multiplication may be written as a space, `^` is a power, `dt` marks a drift term and
`dW<name>` is a shock; `X@0.5` is `X` half a time unit earlier (a lag) and `X@-0.5` later (a lead, in a
loss only).  `observes` is one equation (the signal `y`), a list (`y1`, `y2`, ...) or a mapping of named
signals, and a signal may be `{d: "...", delay: 0.5}`.  `definitions: {name: "..."}` names a linear
combination.  A loss keeps its constant (`(X - b1)^2` has `b1^2`), reported in the cost as its
`constant` part.  A transition's horizon is `{T: 6, past: old.yaml}` (or `past:` a list of
initial shocks, `settle:` in place of `T`, and `continuation: end` when the game ends at T); its window is the
past's.  `noisestate
validate` reads the equations into the grammar and checks that.

To read a file back as equations rather than as keys, `print(ns.load(path).describe())` lays the loaded
model out as differentials, delays, losses and the conventions that apply, at the current parameter
values and without a solve; a notebook cell shows the same content as HTML.

## The keys

| key | type | meaning | default |
|---|---|---|---|
| `name` | string |  | `model` |
| `params` | map of number or expression | parameters, evaluated in order (a later one may use an earlier one) | none |
| `shocks` | list of string | the Brownian shocks | none (every one listed must load something) |
| `agents.<name>.constant` | number or expression | the loss's constant: part of the cost, moves no strategy | 0 |
| `agents.<name>.terminal` | list of loss terms | the loss paid at T, on the states at T: a finite horizon or a transition ending at T (the equations form writes `terminal: "q (X - b)^2"`) | none |
| `agents.<name>.terminal_constant` | number or expression | the terminal loss's constant | 0 |
| `agents.<name>.instant` | control name or list of them | other agents' controls whose current level this agent sees and reacts to within the instant (the equations form writes `observes: {quote: {level: P}}`); the loading is its loss's `-G^DD^-1 G^DP`; no cycles; stationary engine | none |
| `agents.<name>.monitors` | agent name or list of them | the agents whose deviations this one is privy to (Chapter 6's monitoring relation, transitive); no engine solves it yet | none (all naive) |
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
| `horizon` | object | the economics of time: the kind, the discount, the two lengths, a transition's past and continuation |  |
| `horizon.kind` | `stationary` \| `finite` \| `transition` | | `stationary` |
| `horizon.discount` | number or expression | a number, or an expression in the parameters | 0 |
| `horizon.window` | number or expression | **L, the lag-truncation length**: how far back a strategy may look.  `stationary` only: a `finite` horizon has none, and a transition's is its past's | 8.0 (`ns.Stationary`); a transition's comes from its past |
| `horizon.T` | number or expression | **the terminal time**: when the game ends.  `finite` and `transition` only; a `stationary` horizon has none | 1.0 (`ns.Finite`, `ns.Transition`) |
| `horizon.past` | object | kind transition only |  |
| `horizon.past.model` | string or object | the old stationary model: a path (relative to the file) or an inline model |  |
| `horizon.past.initial` | list of object | initial shocks {name, loads, rows} |  |
| `horizon.past.initial[].name` | string (required) |  |  |
| `horizon.past.initial[].loads` | map of number or expression |  |  |
| `horizon.past.initial[].rows` | map of number or expression |  |  |
| `horizon.continuation` | `stationary` \| `end` | kind transition only; default stationary | `stationary` |
| `horizon.settle` | number or expression | kind transition only, in place of `T` (exactly one): the settle tolerance T is found for by the march in T ([transitions.md](transitions.md)) |  |
| `numerics` | object | how the model is solved: the engine, the grid, the fixed point's options, the settings |  |
| `numerics.engine` | `stationary` \| `spectral` | default from horizon.kind: stationary -> stationary, else spectral | `stationary` for kind stationary, else `spectral` |
| `numerics.nodes` | integer >= 2 | nodes per panel (stationary) or per side of each piece (spectral); default 16 | 16 (12 from the CLI's `transition`) |
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
  loading on each shock; on a finite horizon an optional `initial` value
  (`X: {drift: ..., noise: ..., initial: 1.0}`, zero by default) starts the mean path there
  (a stationary model, which has no initial time, rejects it).
* **Definitions.** Named linear combinations of atoms, usable anywhere:
  `Pidx: {P0@tau: 0.333, P1@tau: 0.333, P2@tau: 0.333}`.
* **Signals.** Each row has a linear `drift` (states, other agents' controls,
  definitions), a `noise` loading, and an optional observation `delay`.  Rows
  with a pure noise loading and no drift make a shock directly observed.
* **Loss.** A list of terms `[coef, a, b]` (quadratic) and `[coef, a]` (linear);
  the flow loss is their sum and the agent minimises `E int e^{-rho t} loss dt`.
  A target `theta` on `X` is `(X - theta)^2` less its constant: `[1, X, X]` and
  `[-2*theta, X]`.  Linear terms move only the means (below).  `terminal:` adds a loss paid at T on the
  states at T (a finite horizon, or a transition that ends at T), discounted by `e^{-rho T}`: it enters the first-order
  conditions as the adjoint's terminal condition `H^X_T = G^XX(T) X_T + G^X_T`.
* **Ties.** `ties: [[firm0, firm1, firm2]]` makes the listed agents share one
  strategy (a symmetric equilibrium): only the first is solved for.
* **Horizon.** The economics of time, carrying **two lengths that are never the same quantity**:
  `window` is the lag-truncation length L (how far back a strategy may look) and `T` is the terminal
  time (when the game ends).  `stationary` has `discount` and `window`; `finite` has `T`;
  `transition` has both, plus its `past` and `continuation` ([transitions.md](transitions.md)).
  Asking a horizon for the length its kind does not have is an error naming the one it does.
* **Numerics.** How it is solved, an optional block with the fields of `noisestate.Numerics`:
  `engine` (`stationary` or `spectral`; default from the kind), `nodes` per panel (stationary) or per side of each piece of
  the triangle (12 is usually converged; 6-8 when delays cut the domain into small pieces; the
  panels are the lags' multiples closed under every lag, see [limits.md](limits.md) for a window that is not a
  multiple of them), `unit`/`unit_range`/`breakpoints` (panels are aligned to the delays
  automatically), a transition's `continuation_nodes`, `tol`, `damping`, `max_newton`,
  `variable`, and `settings` ([settings.md](settings.md)).  `solve(model, numerics)` lays a `Numerics` (or a dict
  of its fields) over the file's block.
* Coefficients may be numbers or expressions in the parameters (`"sqrt(p1)"`).

The same structure can be written as equations (`Param`, `State`, `Control`, `Signal`, `Agent`, `shocks`,
`define`; README, "Models as equations"; `examples/expr_examples.py` writes every shipped example that way), which
compile to this file: `model.to_dict()` is the file, `model.save(path)` writes it
(`examples/make_ch5_cycle_market.py` builds an N-firm cycle in a loop that way).
