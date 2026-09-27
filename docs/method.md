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
response and of one transition through them, are drawn in [architecture.md](architecture.md).

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
uniform-cell scheme is kept outside the package as a cross-check (`extras/cells.py`).

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
e^{A(t-r)} (inputs + const) dr`; the cross-check cell engine (`extras/cells.py`) does the same on its cells,
first order in the cell length like its kernels.  The limits are exact: with no information (kernels
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

## Risk-averse agents

An agent with `risk_aversion: theta > 0` minimises the entropic cost `J = theta^-1 log E exp(theta C)` of its
realised cost C (the loss integrated, discounted, plus the terminal loss, as `res.costs` integrates it), instead of
`E C`; `theta = 0` is the risk-neutral model and takes exactly the risk-neutral path.  The spectral finite engine
solves it on a finite horizon without a past, a continuation, monitoring or means (the others refuse it with a
NotImplementedError); the code is `noisestate/risk.py`.

Chapter 1's appendix (thm:risk_sensitive_appendix) gives the first-order condition: the risk-neutral one evaluated
at the risk-adjusted noise-state `W^theta = (I - theta C_t K)^-1 (W_hat + theta C_t k)`, with `C_t` the agent's
conditional covariance of every shock (the filter's posterior for the past ones, the prior for the future ones), `K`
the quadratic kernel of C over the whole horizon, `C = c0 + 1/2 <W, K W>`, and `k` its linear part (zero without
means).  `W^theta` is the point at which the derivative is evaluated: the average of the linear marginal cost over
the outcomes weighted by `exp(theta C) / E exp(theta C)`.  It is not a belief; the agent's filter and its
information are the risk-neutral ones.  Certainty equivalence fails, so the first-order condition keeps the part
of its kernel on the future shocks, which the risk-neutral engine drops because those shocks have mean zero.

On the seen rows the condition is `Pi_t (I - theta K C_t)^-1 f_t = 0`, `f_t` the FOC kernel of a spike at t over all
the shocks.  A kernel whose projection on the rows is zero is left alone by `C_t`, so the condition is the same as
`Pi_t Sigma f_t = 0` with `Sigma = (I - theta K)^-1`, one operator for every t.  The engine keeps its risk-neutral
projection and adds the correction `Delta_t = Sigma f_t - f_t = theta K Sigma f_t` to the FOC kernel (a kernel on
the triangle like the FOC kernel itself).  With `K = A' G A`, `(A g)(tau) = int_0^tau zeta(tau, v) g(v) dv` the loss
atoms' exposure and `G = e^{-rho tau} Q dtau` (plus the terminal loss at T), `h = Delta_t` solves
`h = theta K f + theta K h`:

* `theta K f_t` is computed on the nodes by line integrals on the triangle, with no basis.  `A f_t(tau)` is carried
  as two nodal kernels, for `tau <= t` and for `tau >= t`, read along the projection and response paths.  The
  future part of `f_t` is itself a nodal kernel, `fU(v, t) = f_t(v)` for `v > t`: the continuation path with the
  roles of the spike and the shock exchanged.  Then `K f_t = A' G (A f_t)` along the response and continuation
  paths.
* The rest, `theta K h`, is a Galerkin solve on an orthonormal Legendre basis per time panel and channel
  (`settings.risk_basis` functions, by default the nodes per side plus 4), then Sloan's iterate:
  `c_t = theta (I - theta K_G)^-1 Phi' K f_t` and `Delta_t = theta K f_t + theta (K Phi) c_t`, with
  `K_G = E' G E` built from the exposures `E = A Phi`.  `h` is smooth, so this part converges fast.

A CARA equilibrium is one fixed point.  The kernels, the spike responses and K all come from the closed loop in
which every agent plays its risk-averse strategy, the agent's own included.  A best response takes K (and its
spectrum) in the closed loop of the current profile and solves the FOC with the correction, which is linear in the
world, exactly: the risk-neutral system is factored and preconditions GMRES on the whole system, one application
of the correction per iteration (on the matrix-free path the correction is part of the operator).  A best response
warm-started from the agent's last one stops once its residual is 1e-4 of the warm start's, or at foc_krylov_tol of
the right-hand side if that is larger: early in the fixed point the next best response discards more precision than
that, and near the equilibrium the floor is the one the risk-neutral solve uses.  At the fixed point the profile is
the equilibrium.

`E exp(theta C)` is finite if and only if `theta lambda_max(K) < 1`, and the conditional condition of the theorem
follows from it.  A solve checks this at every best response.  At the zero start (the uncontrolled world,
`lambda_max = 8 / pi^2` on Chapter 1's game) and even at the risk-neutral equilibrium the condition can fail for a
theta that has an equilibrium (theta = 2.5 on Chapter 1's game).  So when some agent's `theta lambda_max` exceeds
0.9 in the uncontrolled world, `solve()` starts from the risk-neutral equilibrium and scales theta up in
warm-started steps.  Each step closes 0.6 of the gap `1 - theta lambda_max` at the last step's equilibrium, and
`res.message` lists the steps.  A step short of the model's theta is solved to 1e-5, since it is only the next
step's start and where its size is measured, and from the second step on it starts from the secant through the
last two step equilibria; the last step is solved to the model's tolerance.  A path whose step equilibrium comes within 1e-3 of the breakdown short of the model's
theta (or that has not reached it in 40 steps) raises `RiskBreakdown` with `reached`, the theta where it stopped (2.90
on Chapter 1's game at 8 nodes; 2.9 itself solves at 12 nodes, with `theta lambda_max` = 0.985).  So does a solution
whose own spectrum is past the breakdown.  An iterate beyond the breakdown inside a
solve is answered at `theta_eff = 0.9 / lambda_max`, a safeguard that a converged equilibrium never uses.

The entropic cost is `J = E C + (2 theta)^-1 sum_i (-log(1 - theta lambda_i) - theta lambda_i)` over the eigenvalues
of K.  The Ritz values of `K_G` give the large ones.  The ones the basis misses (on each channel lambda_i decays
like i^-2) enter through their second-order term `theta / 4 (tr K^2 - sum of the Ritz values squared)`, with
`tr K^2` computed exactly as the triangle kernel K(u, v) squared under the grid's Gram.  The truncation left is of
third order.  The result carries `res.risk[agent]` (theta, the entropic and the expected cost, `lambda_max` and
`theta lambda_max`) and `res.entropic_costs`.  `res.costs` stays the expected cost.

The second-order check is the expected cost's form, as for a risk-neutral agent.  With a positive semidefinite loss
Hessian it bounds the entropic cost's curvature from below (`J'' = E^Q[C''] + theta Var^Q(C') >= tr(Sigma B) >= tr B
= E[C'']`), and the record says so (`"bound": "entropic"`).  Without one the check is not run and says why.
`res.foc[agent][control]` gains `"risk"`, the correction, so that `foc` (the kernel whose projection vanishes) is
`physical + wedge + risk`.  `res.strategy()` is refused for a risk-averse agent: its action weighs the
risk-adjusted noise-state, whose future part the result does not carry.

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

* A model file with a misspelled key, an unused shock, or a control that does not enter its
  owner's loss is rejected with a message naming the offending item.
* A singular best-response system is refused on every engine with a `ValueError` naming the agent
  and the usual causes: a control with no quadratic term in its current value whose effect on the
  loss goes through nothing else (Kyle-Back with the market maker's `[1, P, P]` dropped), a
  quadratic only in a lagged read of the control (free within the lag of the window's edge), two
  rows carrying the same information, a zero noise loading.  The finite engines used to fall
  through to a least-squares solve and report a zero strategy as a converged equilibrium with a
  clean `check()`; they now LU-factor the system on its kept unknowns and refuse a reciprocal
  condition estimate below `EngineBase.FOC_RCOND` (1e-10; the worst regular system in the tests
  is at 2.6e-3).  On the cross-check cell engine's Krylov branch (above 200 unknowns) the test is one probe per
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
  well.  A `TypeError` is a wrong argument (an unknown solve option), a `NotImplementedError` a feature the engine does not have (leads on the finite
  engines).  A `RuntimeError` is a solver problem: a Krylov best response not
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
  and invariance to shock relabelling and agent order.
