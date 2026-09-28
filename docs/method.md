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
solves it on a finite horizon, with no past or a past of initial shocks only (no window), and without a
continuation (the others refuse it with a NotImplementedError); monitored deviations, instant observations and means
are solved; the code is `noisestate/risk.py`.

C must be the agent's realised cost, not merely one with the right mean.  A risk-neutral objective is unchanged by a
term of mean zero; the entropic one is not.  Chapter 4's loss `-D (V - P) + eps D^2` values the insider's flow at the
fundamental: with a moving V it differs from minus the insider's wealth (liquidating at V_T) by `int Q dV`, the
inventory risk a risk-averse trader prices.  Write the wealth: `loss: D P + eps D^2`, `terminal: -Q V`, `Q: D dt`
(tests/test_cara_kyle.py).  With V drawn once at 0- the two are the same random variable.  A market maker's wealth
also holds the noise trades' profit and loss, a stochastic integral `int (P - V) sigma_Z dW_Z`, written with
`integrals` (below).

Chapter 1's appendix (thm:risk_sensitive_appendix) gives the first-order condition: the risk-neutral one evaluated
at the risk-adjusted noise-state `W^theta = (I - theta Sigma_t K)^-1 (W_hat + theta Sigma_t k)`, with `Sigma_t` the agent's
conditional covariance of every shock (the filter's posterior for the past ones, the prior for the future ones), `K`
the quadratic kernel of C over the whole horizon, `C = c0 + 1/2 <W, K W>`, and `k` its linear part (zero without
means).  The theorem is stated for the tracking case, but the argument needs only that the profile is linear: any
quadratic loss in the atoms, cross terms between controls and states included (Kyle's `D V`, `D P`), gives C of that
form, with K indefinite when Q is.  Only a positive eigenvalue can break down, so a K <= 0 (a Kyle insider with a
fixed value: its wealth is never negative along a path) has no breakdown at any theta.  `W^theta` is the point at which the derivative is evaluated: the average of the linear marginal cost over
the outcomes weighted by `exp(theta C) / E exp(theta C)`.  It is not a belief; the agent's filter and its
information are the risk-neutral ones.  Certainty equivalence fails, so the first-order condition keeps the part
of its kernel on the future shocks, which the risk-neutral engine drops because those shocks have mean zero.

On the seen rows the condition is `P_t (I - theta K Sigma_t)^-1 f_t = 0`, `f_t` the FOC kernel of a spike at t over all
the shocks.  A kernel whose projection on the rows is zero is left alone by `Sigma_t`, so the condition is the same as
`P_t S f_t = 0` with `S = (I - theta K)^-1`, one operator for every t.  The engine keeps its risk-neutral
projection and adds the correction `Delta_t = S f_t - f_t = theta K S f_t` to the FOC kernel (a kernel on
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

Initial shocks (a past of shocks drawn at 0-, `horizon.past` without a window) are coordinates of the shocks beside
the paths, their kernels functions of time on the line s = 0.  They join the Galerkin basis as unit vectors, so
their block is exact: `K f_t` gains `k(u) f_xi(t)` on the paths (`k = K e_xi`) and `<k, f_t> + K_xixi f_xi(t)` on the
shocks, the correction gains its part on them (read on the line s = 0 like the FOC kernel's), and `tr K^2` gains
`2 sum ||k_i||^2 + ||K_xixi||^2`.

The entropic cost is `J = E C + (2 theta)^-1 sum_i (-log(1 - theta lambda_i) - theta lambda_i)` over the eigenvalues
of K.  The Ritz values of `K_G` give the large ones.  The ones the basis misses (on each channel lambda_i decays
like i^-2) enter through their second-order term `theta / 4 (tr K^2 - sum of the Ritz values squared)`, with
`tr K^2` computed exactly as the triangle kernel K(u, v) squared under the grid's Gram.  The truncation left is of
third order.  The result carries `res.risk[agent]` (theta, the entropic and the expected cost, `lambda_max` and
`theta lambda_max`) and `res.entropic_costs`.  `res.costs` stays the expected cost.

The second-order check is the expected cost's form, as for a risk-neutral agent.  With a positive semidefinite loss
Hessian it bounds the entropic cost's curvature from below (`J'' = E^Q[C''] + theta Var^Q(C') >= tr(S B) >= tr B
= E[C'']`), and the record says so (`"bound": "entropic"`).  Without one (Kyle's `D P`) nothing bounds it, and the
entropic cost itself is probed (`"bound": "probe"`, finite_free._entropic_probe): its second difference in the world of the
best response along the world of the expected cost's lowest eigenvector (where a minimum is lost first) and, for the
scale, its highest; `"min"` is their ratio and `"expected_min"` the expected cost's own.  One direction: the probe can
find a loss of the minimum, not prove there is none (on Kyle-Back at eps = 0.05 the expected cost's discrete form is
indefinite, -6e-3 at 16 nodes and -4e-3 at 24, while the entropic cost is convex along that direction at theta 1.5).

A risk-averse solve that stalls, or whose best response fails, is retried once from the same start with the best
responses' Krylov solves tightened from 1e-4 of the warm start's residual to 1e-6 (`RISK_KRYLOV_RETRY`): near a fixed
point the looser stop leaves a noise floor in the fixed-point map that Anderson and the Newton polish read as a stall
(Kyle-Back with a prior at eps = 0.05, theta = 1.5, 16 nodes: residual 2e-5; the retry converges, 311 evaluations in all).
A warm-started risk-averse GMRES now runs its restart cycles up to `foc_krylov_maxiter` steps (a cycle can stop early on
the preconditioned residual; two cycles left theta = 1.75 at 6.6e-7 and reported a singular system that is not: its
smallest singular value is 1.4e-4 of the largest and every eigenvalue has a positive real part).  If the retry fails too,
the message carries the entropic probe at the best iterate (or at the start, when a best response raised) and says
whether the risk-averse best response has lost its minimum there.

The retry's best responses are also proximal (`RISK_PROX`, `finite_free.FocSystem._proximal`): argmin of the frozen
objective plus `mu/2 |a(g) - a_prev|^2`, `a(g)` the action kernel the strategy makes and `a_prev` the agent's action
iterate, measured on its rows.  At a fixed point of the action iteration the term vanishes, so the equilibrium is the
same.  It is there because the frozen best response solves `E^Q[C''] g = ...`, the expected cost's Hessian under the
tilted measure at the profile's K, while the entropic cost's own Hessian adds `theta Var^Q(C')`, positive semidefinite,
which freezing K drops.  On Kyle-Back with a prior at eps = 0.05 (16 nodes) the smallest eigenvalue of that frozen system
at the equilibrium, relative to its largest, is 1.2e-2, 6.3e-3, 1.6e-3, 7.3e-4, -2.9e-4 and -3.6e-3 at theta 1, 1.25, 1.5,
1.55, 1.6 and 1.7: it goes through zero at about 1.58, where the best-response map's Jacobian has a pole (the real
eigenvalue of -4.4, -9.1, -20 at theta 1.25, 1.4, 1.5 that `res.stability()` measured, the same at 16 and 20 nodes), and
the plain iteration from the theta = 1.5 equilibrium did not reach 1.6.  With the proximal retry each theta up to 3
converges from the last in 190 to 240 evaluations, the certainty equivalent against `extras/kyle_reference.py` (n = 40
.. 120, Richardson over the five levels): 0.5206574 against 0.5206571 at theta 1.7, 0.4953722 against 0.4953707 at 2.
Anchored at the profile's raw map instead of the action iterate the term did not vanish (the map a projection makes of
an action is not the strategy that made it): Chapter 1's game moved by 4e-4.  A solve that does not converge no longer
raises RiskBreakdown at its last iterate (which can be anywhere); the message says so and the retry goes on.
`res.foc[agent][control]` gains `"risk"`, the correction, so that `foc` (the kernel whose projection vanishes) is
`physical + wedge + risk`.  `res.strategy()` is refused for a risk-averse agent: its action weighs the
risk-adjusted noise-state, whose future part the result does not carry.

### Means: the linear part of the cost

With a target, a constant drift or an initial state the realised cost has a linear part, `C = c0 + <k, W> + 1/2 <W, K W>`,
`k = A' G (Q mbar + q)` (the cross term of the fluctuations and the atoms' mean paths `mbar`; its terminal part
likewise).  `k` is deterministic, so the kernels' condition is unchanged, and the maps with means are the maps without
them, to the bit.  The mean condition of row t gains the tilt of the linear part: the tilted law's mean given `F_t` has
the extra `theta Sigma_t k`, and on the kernels' own condition `Sigma_t` drops out again,
`theta <(I - theta K Sigma_t)^-1 f_t, Sigma_t k> = theta <S f_t, k> = theta <f_t, S k>`.  `k` is linear in the means, so the
mean system stays one linear solve: `S k` for every unit mean path at once (one Galerkin solve, Sloan's iterate
`S k = k + theta K Phi c`, `c = (I - theta K_G)^-1 Phi' k`, `Phi' k` from the same exposures as `K_G`), paired with `f_t` per
time row (`risk.Tilt.linear_part`, `pairing`).  The entropic cost gains `theta / 2 <k, S k>`
(`res.risk[agent]["linear_part"]`).  Checked against `extras/leqg_reference.py` (`MeanGame`: affine strategies, x0 = 1,
drift 0.5, target 0.3; `tests/refs/leqg_ch1_means.json`): costs to 1e-7 and the mean paths to 1e-6 at theta = 0.5 and 1
(tests/test_cara_ext.py).  Risk-averse agents with means on a windowed strip, or with integrals and means together,
are refused.

### Stochastic-integral terms (`integrals`)

`integrals: [[coef, quantity, shock], ...]` adds `coef int e^{-rho t} quantity dW_shock` to the agent's realised cost (an
Ito integral: the quantity is known before the shock).  It has mean zero, so a risk-neutral solve ignores it, to the
bit.  A risk-averse agent prices it: `K` gains `K_x = A_x' e^{-rho} L + e^{-rho} L' A_x` (`L` the coefficients on the
atoms and shocks; the kernel `e^{-rho v} z_x(v, u)' L` for `u < v`, no trace), in `K_G`, `K Phi`, the line integrals and
`tr K^2`, and the FOC kernel gains a world-independent future part, `e^{-rho (v - t)} L' R_x(v, t)` for `v > t` (the spike
moves the quantity, which loads on the later shocks), whose correction enters the right-hand side once
(`Tilt.delta_const`).

The quantity may be the agent's own current control (a market maker's quote against the noise trades), or a control
that reacts to it at once.  A spike at t then moves the integral by `L' dW(t)`, the increment right after the
decision: `f_t` gains a point mass `Ldelta delta_t` on the shock of its own instant (`Tilt.Ldelta`, from the composite
spike).  It is never paired with a seen row (the increment comes after the decision, `Sigma_t delta_t = delta_t`) and
its risk-neutral value is zero; its correction `theta K S (Ldelta delta_t)` is regular and world-independent: `K f_t`
gains `K(., t) Ldelta` at the nodes and `Phi' K f_t` gains `Psi(t)' Ldelta` (`delta_const`).  Checked against
`extras/leqg_reference.py` (`Game(udw)`: `udw sum D_i,k sigma sqrt(dt) g0_k`, `tests/refs/leqg_ch1_udw.json`): at 16 nodes
J to 4e-6 and D1 to 7e-6 at theta = 1, converging like N^-2.4; without the point mass D1 is off by 0.04.  The stationary
engine refuses such an integral.

With integrals, a Kyle insider's wealth on a
moving value is written without a terminal time: `-D V + D P + eps D^2` plus `integrals: [[-sigma_V, Q, wV]]`, `Q: D dt`,
which at rho = 0 is `int D P dt - Q_T V_T` path by path.  Checked against `extras/leqg_reference.py` (`Game(xdw)`,
`tests/refs/leqg_ch1_xdw.json`): costs to 1e-7 and D1 to 3e-5 at theta = 0.5 and 1.

### Monitored deviations (Chapter 6)

A privy player n answers a deviation of origin i it knows: a seed xi at s, zero-variance under the equilibrium law and a
known input from s on.  Its response kernels are fixed by its first-order condition in the seed world, which the
risk-neutral engine writes as `f^xi_t(s) = 0` (the FOC kernel's column on the seed).  The seed world is affine in
`(W, xi)`, so a risk-averse responder's realised cost is `C = c0 + xi <k_s, W> + 1/2 <W, K W>` with the on-path K and the
cross term `k_s = K e_xi(s)`: the seed world's atoms against the profile's,
`k_s(u) = int zeta(tau, u)' G zeta^seed(tau, s) dtau` (+ the terminal loss, + `e^{-rho u} L' x^seed(u, s)` from the
integrals).  Conditioning on F_t, which contains xi, the tilted mean of the shocks is `(I - theta Sigma_t K)^-1 (W_hat +
theta xi Sigma_t k_s)`: the seed enters as a known linear part, like a mean.  The on-path condition makes
`(I - theta K Sigma_t)^-1 f_t` Sigma_t-invariant, so Sigma_t drops out once more and the response condition is

    f^xi_t(s) + theta <S f_t, K e_xi(s)> = 0,      t >= s,

with no conditional covariance and no prior on the seed.  The deviator's own blip continuation is such a response, so a
risk-averse market maker needs it even against a risk-neutral trader.  The term needs `S f_t` over the whole horizon
(its future part, which the on-path correction never forms): with `Phi' (S f_t - f_t) = c_t`, the on-path correction's
Galerkin coefficients, `<S f_t - f_t, k_s> ~ c_t' Phi' k_s` to second order in the basis (the product of two projection
errors), so `theta <S f_t, k_s> = theta [int_s^T a_t' G zeta^seed dtau + terminal + int_s^T e^{-rho u} h_t' L' x^seed du]`
with `a_t = A f_t + E c_t`, `h_t = f_t + Phi c_t`: line integrals along each node's response and continuation paths, exact
in the seed world's kernels and linear in the unknown response kernels (`risk.Tilt.seed_operator`, an N x (n_prim N)
operator added to the responder's rows of the monitoring solve).  The seed spike itself is a point mass that no kernel
on the triangle holds; against the responder's atoms and integrals it adds a constant, `theta e^{-rho s} [(Q a_t(s))_v +
(L (S f_t)(s))_v]` for the controls v the spike moves at its instant (`seed_constant`; the atom part vanishes for a
quantity the responder has seen by s).  The same condition is the precommitment objective with xi a known constant,
minimised over the seed-contingent plan, which is how the reference checks it: `extras/leqg_reference.py`
`MonitorGame` (player 2 tracks player 1's state and is privy to its deviations; the privy players' response columns
minimise their exact seed-world objective, log det plus the linear part; `tests/refs/leqg_monitor.json`).  At 14 nodes
the responses agree to 5e-5 and the entropic costs to 1e-6 (theta 1, theta 1.5); without the term the responses are
off by 0.1 to 1 (tests/test_cara_monitoring.py).

### Consistent planning on the finite engine

`settings.risk_planning = "consistent"` (`ns.solve(m, numerics={"settings": {"risk_planning": "consistent"}})`) replaces
precommitment, `J_0` over the whole strategy, by consistent planning: every date's self minimises
`J_t = theta^-1 log E[exp(theta C_t) | F_t]` of its own continuation `C_t = int_t^T e^{-rho (tau - t)} c_tau dtau` (+ the
terminal loss and the integrals from t on), the later selves playing the equilibrium map.  The tilt argument at date t,
with the kernel `K_t` of `C_t`, gives `P_t S_t f_t^on = 0`, `S_t = (I - theta K_t)^-1`: one operator per date.  `K_t` drops
every cost before t (the date-t self does not bear it; under precommitment those costs still weight the outcomes through
their cross terms with the future), discounts from t (so theta enters the kernels discounted from zero as `theta e^{rho t}`;
precommitment's effective risk aversion at t is `theta e^{-rho t}` of this), and `f_t^on` is the FOC kernel with the
agent's own later reactions on: the later selves do not share the date-t objective, so there is no envelope
(`FocOps(envelope=False)` with `SpectralFiniteSolver._responses_on`, the fixed point `c = G Y (R0 + Resp c)` on the passive
rows, as on the stationary engine).  At theta = 0 both criteria are the risk-neutral one (`P (phi^on - phi^off)` is
5.5e-9 at Chapter 1's risk-neutral equilibrium, 14 nodes, against `|phi^on - phi^off|` = 0.11).

`risk.ConsistentTilt` computes, frozen at the profile, `Delta_t = theta K_t S_t f_t^on` on the seen nodes by Sloan's iterate
with one Galerkin matrix per time row: `K_t f_t` along the continuation path alone (tau >= t), `Phi' K_t f_t =
int_t^T E' G (A f_t) dtau` and `K_{t,G} = int_t^T E' G E dtau` on the geometry's per-row future quadrature, `K_t Phi` at the
nodes by a continuation path of the Galerkin quadrature's order.  The best response adds `(phi^on - phi^off) + Delta_t` to
its risk-neutral FOC kernel, a shift like the stationary engine's: the system is the risk-neutral one (factored), and at a
fixed point the profile is the best response, so the equilibrium solves the condition exactly.  The date-0 self's
objective is the precommitment one, so the breakdown check (`theta lambda_max(K) < 1`) and `res.risk` are unchanged;
later dates need `I - theta K_t` invertible (a singular one is refused with the date's margin).  Refused: a past,
monitoring, instant observations, means.  Checked against `extras/leqg_reference.py` `ConsistentGame`, a brute-force
discrete game solved by backward sweeps over the date selves, each date's row the exact minimiser of its conditional
entropic objective (the unseen shocks integrated out: a quadratic form in the action and the seen signals), with no
gradient and no optimiser (tests/refs/leqg_ch1_consistent.json, tests/test_cara_consistent.py).

### The stationary engine: consistent planning

On the stationary engine each date's self minimises the entropic cost of its own discounted continuation,
`J_t = theta^-1 log E[exp(theta C_t) | F_t]`, `C_t = int_t^inf e^{-rho (s - t)} c_s ds` (plus the integrals), the later selves
playing the stationary strategy.  (Minimising `J_0` over the whole strategy, the finite engine's precommitment, has no
stationary solution: its effective risk aversion at t is `theta e^{-rho t}`.)  The tilt argument at one date gives
`P_0 S f_0 = 0`, `S = (I - theta K_0)^-1` on the shocks born in `(-L, inf)`, `K_0` the kernel of `C_0` and `f_0` the FOC
kernel of a spike at date 0 over the past and the future shocks.  Unlike the risk-neutral condition, `f_0` is taken with
the agent's own later reactions ON: the later selves minimise their own `J_t`, not `J_0`, so a date-0 deviation's
effect through them does not vanish to first order (no envelope).  At theta = 0 the two agree at an equilibrium (checked:
`P (phi^on - phi^off)` is 2e-10 at Chapter 4's risk-neutral equilibrium), and theta = 0 takes the risk-neutral path.
The responses with the own reactions on are the closed loop's with the agent switched off (the only way it carries a
control's spike to the others' rows) plus the agent's reaction kernels from the fixed point `c = G Y (R0 + Resp c)`.

`noisestate/stationary_risk.py` adds to the risk-neutral best response, frozen at the current profile, the shift
`(S f_0^on - f_0^off)` on the past: `(phi^on - phi^off) + theta K_0 S f_0^on`.  At a fixed point the profile is the
best response, so the equilibrium solves the condition exactly; off it the shift is an explicit term that the
Anderson mixing iterates away.  `S f_0` is a Nystrom solve on a uniform lattice through the ages 0 and L (the kernels'
cuts at half weight: the trapezoid rule), `A` and `A'` by FFT convolutions (`K_0` is shift-and-discount covariant), GMRES
for `(I - theta K_0 Sigma) g = f_0` (Sigma the projector off what the agent has seen, on the lattice's past: the same g as
`S f_0` where the condition holds, and well conditioned whenever the conditional entropic cost is finite, which
`I - theta K_0` is not for a Kyle insider, whose seen inventory makes `theta lambda_max(K_0)` 2.9 at theta = 1), then
`theta K_0 Sigma g` read at the engine's age nodes with the tau integral cut exactly at the
window; two lattices (`RISK_LATTICE`, 0.02, and half) and Richardson's h^2 step (`res.risk[agent]["richardson_gap"]`).
`solve()` from no start goes theta x 0, 0.5, then to 1 while `theta_mu_max` at the full theta, estimated at the last step's equilibrium, is at most 0.9, else in the finite engine's steps closing 0.6 of the gap to the breakdown; a step whose fixed point lies past the breakdown (or whose correction's GMRES fails there) is halved, three times at most in all, and a path that ends short of theta raises `RiskBreakdown` with `reached`; a converged solution past the breakdown raises it too.  `res.risk[agent]["entropic"]` is the date-0 self's entropic cost
averaged over the past, `E C_0 + (theta / 2) tr(Pi K_0 B K_0 Pi) + (2 theta)^-1 sum(-log(1 - theta mu) - theta mu)` (`Pi` the
projector on what the agent has seen, `mu` the eigenvalues of `Sigma K_0 Sigma`, `Sigma = I - Pi`, `B = Sigma (I - theta
Sigma K_0 Sigma)^-1 Sigma`), `E C_0` the flow cost over rho, on three lattices (`ENTROPIC_LATTICE`, L / 40, 80, 160, refined by an integer factor when the coarsest would exceed `ENTROPIC_STEP` 0.2) and
the polynomial in h through them (the excess converges like h: the kernels' kinks on the diagonal; about 1e-2 relative
on a fast-decaying kernel, 1e-4 on the signal model of the tests); `theta_mu_max` is the conditional breakdown measure.  The
unconditional `theta^-1 log E e^{theta C_0}` can be infinite when the conditional one is not (a Kyle insider's inventory
times the value's later moves), so it is not what is reported.  Refused: no discount, monitoring, instant
observations, level rows, lagged atoms in a risk-averse agent's loss or integrals, means.  Checked against
`extras/stationary_cara_reference.py`, a discrete one-agent stationary game solved by brute force under consistent
planning (tests/test_cara_ext.py).

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
