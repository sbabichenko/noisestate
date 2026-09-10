# How it works

The method of the solver: the stationary form, the finite-horizon form, the means, the grids and their caches,
and the guarantees the code makes.  The modules and the path of one best response and one transition through
them are in [architecture.md](architecture.md); the checks a result carries are in [guards.md](guards.md).

## The best response and the fixed point

A model whose single tie group is a cycle (the Chapter 5 market) is solved with that symmetry: the
closed loop is block diagonal in the Fourier basis over the cycle, so the world solve is linear in
the number of tied agents, and switching one agent off for its passive world is a low-rank
correction.  The symmetry is found from the ties and verified on the expanded model, and the
result is identical to the general solve.

The two spectral engines share one best response, written against a kernel algebra of seven
operations that each compiled model supplies (convolution with a row, the instantaneous entry
and its adjoint, the response to an action, the discounted continuation, the read of a lagged own
control, the projection onto a row).  The engines keep what differs: the closed-loop solve (dense
on the age grid, causal block substitution on the triangle), the regularisation of the
first-order-condition system, and the projection back to raw maps.  The modules, and the path of one best
response and of one transition through them, are drawn in [docs/architecture.md](docs/architecture.md).

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
(all engines iterate on the action kernels with Tikhonov-regularised Anderson
acceleration, the outer solver of the Chapter 5 market solver, and derive the raw
maps by projection; a Newton-Krylov polish runs if Anderson stalls, about 15 to 25 evaluations of
the map per Newton step: at most 15 fresh Krylov vectors in its inner LGMRES iteration, plus the up
to 10 directions carried from earlier steps, multiplied afresh each step, and the line search.
`solve(variable="maps")` iterates on the raw maps instead, which is what happens with ties in any
case).
Both variables are kept because each fails somewhere the other does not: on the delayed
Chapter 1 finite model the raw maps stall at a residual of 9e-7 after 313 evaluations
where the action kernels converge in 16; with ties only the raw maps carry over between
tied agents.

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
uniform-cell scheme (`numerics.engine: cells`) is kept as a cross-check.

## Means

The kernels are the zero-mean part of the equilibrium: every quantity as a linear functional of the
shocks.  Linear loss terms (targets), constant drifts and, on a finite horizon, initial states move
the means, deterministic paths that are common knowledge; the kernels do not depend on them, and the
means are linear in them.  Every engine solves them at the end of every solve (with
`diagnostics=False` as well; they are part of the answer, not a check).  A control's mean
first-order condition is the kernels' first-order condition applied to a deterministic path, with
no information constraint: the instantaneous derivative of `1/2 z'Qz + q'z` in the control, its
discounted own lagged reads, and the continuation `int_t^T e^{-rho (t' - t)} R(t', t) g(t') dt'`
through the passive-world impulse responses, the very responses the best response computes (a
deviation of one agent's mean is seen by the others through their signals and answered through
their equilibrium kernels), with the targets `q` and the initial state as the driver in place of
the shocks; the mean dynamics close the system, and it is one direct linear solve, no iteration.
In a stationary model the means are constants, one per state and control, the continuation the DC
gain `int_0^L e^{-rho a} R(a) da` and the dynamics `A xbar` plus the inputs at their constants plus
the constant drift equal to zero.  On a finite horizon they are paths on [0, T]: the spectral engine
carries a path on the time nodes of its triangle as a kernel constant in shock age, on which the
kernels' own operators restricted to the line `s = 0` (a kernel's response to a shock at time 0)
are the path's, so the continuation is the same spectral line integral as the kernels' and a lagged
read is the path at `t - lag` (zero before 0), and the state is `xbar(t) = e^{At} x0 + int_0^t
e^{A(t-r)} (inputs + const) dr`; the cell engine does the same on its cells, first order in the cell
length like its kernels, as a cross-check.  The limits are exact: with no information (kernels
zero) the means are the open-loop Nash equilibrium of the deterministic game, with perfect
information the closed-loop (feedback) Nash equilibrium, one agent alone gets the deterministic
optimum; under private information they lie between (the separation failure), which
`tests/test_means.py` and `tests/test_means_finite.py` check against the closed forms, the coupled
Riccati equations and the dissertation's Chapter 1 solver ([validation.md](validation.md)), and
`examples/ch1_mean_sweep.py` shows on the Chapter 1 game with targets.

`res.means` has every state, control and definition and each signal row's mean drift rate
(`"agent.row"`), as a constant (stationary) or as the path on the time nodes `res.mean_times`
(finite; `res.mean(name, t)` interpolates on the spectral engine, and its `plot()` adds the paths
as a last row); `res.cost_parts[agent]` is `{"variance", "mean"}`, the mean part being the flow
`1/2 zbar'Q zbar + q'zbar` per unit time, or its discounted integral over [0, T] (the constant
`theta^2` of a target is not in the model), and `res.costs[agent]` is their sum; the summary,
`to_dict()` and the CLI JSON carry all three.  Without a driver (no linear term, no constant drift,
no initial state) every mean is exactly zero, with no solve.  A random walk with no inputs
(Kyle-Back's `V`, the Chapter 5 market's demand level `q`) has no stationary mean and is pinned at
0, so the means of what it enters are relative to its level (`model.notes` says so); a random walk
with a constant drift and no feedback, and a singular mean system (a state whose mean no
first-order condition determines, a control with no quadratic term in its current value), are
refused with a `ValueError`.  On the stationary engine the continuation integrals are truncated at
the window like the kernels' own, which is what `window_tail` reports; the flag says so when the
means are nonzero.

## Grids and caches

On panels of equal width the convolution and correlation tensors are block-Toeplitz in the panel
index, so the age grid stores a two-piece core on one panel and dense slabs for any non-uniform tail
(20 MB instead of 131 on the Chapter 5 grid) and assembles its operators from them; applying that
structure to the downstream products by FFT was measured and is slower than dense products below
about 300 uniform panels.  Grids and their operator caches are shared across solves in a process (`noisestate.clear_grid_cache()`
releases them; a large stationary grid holds a few hundred MB of convolution tensors).  The cache keeps at
most 32 grids and at most 1.5 GB of their tensors, paths and read matrices (`noisestate.grid_cache.BUDGET_BYTES`),
dropping the least recently used grids beyond that.

## Stability guarantees

* A model file with a misspelled key, an unused channel, or a control that does not enter its
  owner's loss is rejected with a message naming the offending item.
* A singular best-response system is refused on every engine with a `ValueError` naming the agent
  and the usual causes: a control with no quadratic term in its current value whose effect on the
  loss goes through nothing else (Kyle-Back with the market maker's `[1, P, P]` dropped), a
  quadratic only in a lagged read of the control (free within the lag of the window's edge), two
  rows carrying the same information, a zero noise loading.  The finite engines used to fall
  through to a least-squares solve and report a zero strategy as a converged equilibrium with a
  clean `check()`; they now LU-factor the system on its kept unknowns and refuse a reciprocal
  condition estimate below `EngineBase.FOC_RCOND` (1e-10; the worst regular system in the tests
  is at 2.6e-3).  On the cell engine's Krylov branch (above 200 unknowns) the test is one probe per
  control, which catches a control its first-order condition does not respond to; a partial
  deficiency there ends as a non-converged linear solve.  Validation warns (`UserWarning`) when a
  control has no strictly positive quadratic term in its own current value, or in a lagged read of
  it for a non-myopic agent, which is the usual cause.
* `converged` means the residual of the fixed point is at or below `tol`, where the residual is
  the norm of the update divided by the larger of one and the norm of the iterate (so for a
  solution of norm below one it is an absolute residual).  `res.message` says what the outer
  solver did, `res.summary()` shows it when the solve did not converge, and `res.require_converged()` raises
  `ConvergenceError` so a pipeline cannot use a failed solve by accident.
* Errors are typed by whose problem they are.  A `ValueError` is a model problem: a file that does
  not validate, a parameter that is not the model's, a lag off the panels, a singular best-response
  system; a solve bound out of range (`max_evaluations` below one, a negative `deadline`) is one as
  well.  A `TypeError` is a wrong argument (an unknown solve option, `naive_observers` that is not
  a mapping), a `NotImplementedError` a feature the engine does not have (leads on the finite
  engines).  A `RuntimeError` is a solver problem: the cell engine's Krylov best response not
  converging, a best-response map that returns a non-finite value (an overflow; the iteration stops
  at that evaluation instead of running on NaN), and `ConvergenceError` (a `RuntimeError`) from
  `check()`.  A fixed-point iteration
  that does not reach `tol`, or is stopped at `max_evaluations` or `deadline`, does not raise:
  `solve()` returns the result with `converged=False` and
  `res.message` says what happened, `sweep()` records the point as a row with `converged: False` and
  goes on, and `check()` is the raise.  The CLI prints any of these as `error: ...` and exits 2.
* A signal row with a positive `delay` is uninformative about shocks younger than the delay; the
  map on that row is set to zero at ages above `window - delay`, where it reads nothing within the
  window.  A kernel read at a lag (a delayed row, `P@tau`) jumps at the lag, and the panels'
  duplicated breakpoint nodes carry the two one-sided limits: the lower copy reads the left limit
  (zero at the lag), the upper copy the right limit, and a lead is the exact transpose of the lag.
  With that, the breakpoints closed under adding and subtracting every row delay (so the map's
  panels and the action's panels are unions of each other shifted by the delay; beyond `unit_range`
  a delayed model's panels become uniform; the finite triangle is closed under every lag as well,
  so a lagged read is a node-to-node shift, unless `unit_range` is below the window: the cuts then stay
  at every multiple of the unit within unit_range of 0, and of T when the game ends there, and the panels
  grow geometrically between, a lagged read beyond being interpolated; a row observed with a delay keeps
  its map on the action grid shifted by the delay, read node to node on every piece, so this is refused
  without a past), and the map removed where the row reads nothing, the
  delayed problem is discretised exactly: on a one-agent problem with a delayed observation whose
  solution is known in closed form (certainty equivalence and a delay-differential system,
  `tests/test_exact_delay.py`) the cost agrees to 7e-11 and the kernels to 6e-6 at 16 nodes per
  panel, the representation error is 3e-13, the action is exactly zero below the delay, and the
  costs are identical to eight digits between 12 and 24 nodes at a fixed window.
* Costs are integrated with exact Gram matrices, so a converged best response is optimal against
  every feasible perturbation to round-off; `tests/test_properties.py` checks this on both engines
  without any reference solution, together with the equivalence of the two iteration variables
  and invariance to channel relabelling and agent order.
