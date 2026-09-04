# Changelog

## 0.2.0 (2026-09-04) — stability

- Model files are validated strictly: unknown keys at any level, channels that nothing loads, and
  controls that do not enter their owner's loss are errors, not silent no-ops.
- `converged` means the relative residual is at or below `tol` in every engine; every result
  carries a `message` describing the outer-solver path, `summary()` shows it when not converged,
  and `result.check()` raises `ConvergenceError`.  No exception is swallowed on the way.
- Reported costs use exact Gram matrices of the nodal bases; the best response is now provably
  optimal against random feasible perturbations in both engines (tests/test_properties.py).
- Tied agents iterate on raw maps (action kernels need the symmetry's channel permutation).
- Property tests: optimality, equivalence of the two iteration variables, invariance to channel
  relabelling and agent order, rejection of typos.
- Performance: batched tensor contractions, cached loss operators, causal block solve in the
  finite engine, warm-started sweeps with a secant predictor.

## 0.1.0 (2026-09-03)

- Stationary engine, spectral finite-horizon engine, cell cross-check engine, CLI, examples and
  regression tests against the dissertation's chapter solvers.
