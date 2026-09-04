# extras

Material tied to the dissertation's own solvers, kept out of the wheel:

* `cpp_cascade_replica.py`, `test_cpp_cascade.py`, `refs/`: a subclass of the stationary engine
  that reproduces the defect of the C++ two-trader Kyle-Back solver (`kb_spectral_q`: residual-flow
  policy rows inside a total-flow feedback cascade), with the published output it reproduces.
* `patches/`: the fix for that C++ solver.
* `tools/`: comparison scripts against the Chapter 1 solvers.
