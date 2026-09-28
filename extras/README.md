# extras

Material tied to the dissertation's own solvers, kept out of the wheel:

* `cpp_cascade_replica.py`, `test_cpp_cascade.py`, `refs/`: a subclass of the stationary engine
  that reproduces the defect of the C++ two-trader Kyle-Back solver (`kb_spectral_q`: residual-flow
  policy rows inside a total-flow feedback cascade), with the published output it reproduces.
* `cells.py` and its tests: the first-order uniform-cell finite-horizon engine, retired from the package
  in 1.1 and kept as an independent discretisation to cross-check the spectral engine (Richardson pairs,
  closed forms).  `FiniteSolver(model.with_numerics(nodes=N)).solve()` on a finite model, N cells.
  Its result's payload (kind `finite_cells`) is not one the package schema accepts.
* `leqg_reference.py`: a brute-force discrete-time reference for risk-averse (CARA, entropic) equilibria of Chapter 1's
  tracking game, independent of the package: linear maps on each player's signal history, the exact log-det entropic
  cost with its analytic gradient, L-BFGS best responses.  `python extras/leqg_reference.py table tests/refs/leqg_ch1.json`
  writes the table `tests/test_cara_finite.py` checks the engine against (n = 50 to 400 steps, about twenty minutes).
* `kyle_reference.py`: the same for a finite-horizon Kyle-Back market with CARA insiders (one, or K identical ones whose
  rivals react to a deviation through their maps) and a competitive market maker (the projection of the insiders'
  current strategy), the insider's realised cost minus its wealth.  `python extras/kyle_reference.py table
  tests/refs/leqg_kyle_prior.json [2]` writes the tables `tests/test_cara_kyle.py` checks (n = 40 to 120, under a minute).
  Cost "dw" (the fundamental-valued flow plus the inventory's Ito sum, with a discount rho) is the wealth written with
  `integrals`; at rho = 0 it equals cost "wealth" path by path.
* `leqg_reference.py` also has `MeanGame` (affine strategies: an initial state, a drift and a target; `python
  extras/leqg_reference.py means tests/refs/leqg_ch1_means.json`) and `Game(xdw=...)` (the integral xdw int X sigma dW0
  in the costs; `... xdw tests/refs/leqg_ch1_xdw.json`), each a few minutes.  `Game(udw=...)` puts a player's own control
  against the shock of its instant, udw int D_i sigma dW0 (a market maker's quote against the noise trades;
  `... udw OUT.json`; tests/refs/leqg_ch1_udw.json holds n = 40 to 200).
* `leqg_reference.py` `MonitorGame`: Chapter 6's privy response under the entropic objective by brute force.  Player 1
  regulates its own state, player 2 tracks it and is privy to player 1's deviations; after a known seed the privy players
  (player 1's own blip continuation included) choose their response columns to minimise their exact precommitment
  objective of the seed world, log det plus the linear part (one linear solve: the objective is quadratic in the columns),
  the on-path maps held.  `udw` adds each player's own control against its state's shock.  `python
  extras/leqg_reference.py monitor tests/refs/leqg_monitor.json`, about two minutes.
* `stationary_cara_reference.py`: a discrete stationary one-agent game under consistent planning (every date's self
  minimises the entropic cost of its own discounted continuation given the later selves' map; the date-0 best response
  in closed form, the fixed point by Newton-Krylov), with the agent seeing the state's shocks or a noisy signal of the
  state.  tests/refs/stationary_cara_lq.json holds its levels (h = 0.1 to 0.0125) and their Richardson limits.
* `patches/`: the fix for that C++ solver.
* `tools/`: comparison scripts against the Chapter 1 solvers.
