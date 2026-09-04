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
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q tests        # regression tests against the chapter solvers
```

Dependencies: numpy, scipy, pyyaml, matplotlib.

## Use

```bash
noisestate validate examples/ch4_kyle_back.yaml
noisestate solve examples/ch4_kyle_back.yaml -o kb.json --plot kb.pdf --param rho=0.5
```

```python
import noisestate as ns
res = ns.solve("examples/ch3_two_player.yaml")
print(res.summary())
res.ages                     # shock ages (Chebyshev nodes on the panels)
res.kernel("X")              # closed-loop kernel of X, one column per channel
res.action_kernel("D1")      # closed-loop kernel of a control
res.maps["player1"]          # raw strategy g[u][r](b) on the agent's signal rows
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
  `tau` earlier (a lag), `name@-tau` its value `tau` later (a lead).
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
  domain into small pieces); `finite_cells` selects the first-order cell scheme.
* Coefficients may be numbers or expressions in the parameters (`"sqrt(p1)"`).

The same structure is available from Python through `ModelBuilder` (see
`examples/make_ch5_cycle_market.py`, which builds an N-firm cycle in a loop).

## How it works

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
(damped iteration, then Newton-Krylov).

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
| 5 | purchase-order market on a 3-cycle with delay | `spectral_market` sweep | 0.5% (quote maps), 1-5% (order maps) |
| 1 | finite-horizon two-player game | `spec_ch1` | converged at 12 nodes per side (cost stable to 1e-8 from 12 to 20); kernels within 1e-3 of the reference except on the diagonal, where the reference's own README reports weakly determined modes; the cell scheme's Richardson limit agrees with the spectral engine there to 1e-3 |

Kyle-Back with two traders: the reference grid solver (`kb_multi.py`), Richardson-
extrapolated, agrees with noisestate to 3-4 decimals; the C++ spectral port
`kb_spectral_q` differs by 2-6% and its solution is not a best response to itself
(see `tests/test_ch4.py` for the diagnostics).

## Limits

Scalar states and controls (write vector models as several scalars); no exact
(noise-free) observation of a state that is not itself a channel; means (targets,
linear loss terms) are not yet solved.
