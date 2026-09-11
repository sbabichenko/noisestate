# Changelog

## 1.0.0 (2026-09-10) — Initial public release

Equilibrium solver for linear-quadratic-Gaussian games with private information,
following the dissertation *Noise-State Calculus for Dynamic Games with Strategic
Information* (Babichenko, 2026).

### Capabilities

- Define models in YAML, with Python expressions, or with `ModelBuilder`.
- Solve for causal linear strategies on stationary and finite horizons, including
  regime transitions and supported observation and action delays.
- Inspect equilibrium costs, shock-response kernels, strategies on signal histories,
  mean paths, and first-order-condition decompositions.
- Assess convergence, resolution, window truncation, and second-order conditions;
  refine solutions and analyse best-response stability.
- Run parameter sweeps and compare observation scenarios.
- Use the Python API or CLI, export results as JSON, and plot results with the
  optional matplotlib dependency.
- Start from seven bundled example models and a worked tutorial.

### Limitations

Models require linear dynamics and observations and quadratic running losses.
Hard control constraints and terminal penalties are not supported. Solutions are
sought within the causal linear strategy class.

Numerical convergence alone does not establish an adequate grid or lag window.
Some bundled benchmarks intentionally trigger diagnostics, and the cell engine
does not support every check. Undiscounted stationary models use the formal
average-cost equations; the stationary verification assumes a positive discount.

See [limits](docs/limits.md) for details and [validation](docs/validation.md)
for comparisons with closed forms and the dissertation's solvers.

The [pre-release development history](docs/design/pre-release-history.md) preserves
the internal 0.x notes and the work leading to 1.0.0.
