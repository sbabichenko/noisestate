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
* `patches/`: the fix for that C++ solver.
* `tools/`: comparison scripts against the Chapter 1 solvers.
