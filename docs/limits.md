# Limits

What the grammar and the engines do not do, and what is done only approximately.

Vectors exist in the Python form only (`ns.State("X", 3)`, matrix coefficients), where they expand into scalar
components; the file form has no vector syntax.  An exact observation of a state or of another agent's control
is a level row (`{level: X, filter: true}`): its increments are the row, their drift the kernel's rate of change and
their noise loading its jump at age 0, both set by the equilibrium.  Level rows are solved on the stationary engine only,
without lags, delays or ties, and a quantity that does not jump at age 0 (no noise of its own) is an exact smooth
observation the engine refuses as singular.  A plain `{level: P}` is the instant reaction alone, not information: add
`filter: true` for a trader that learns from the quote (Chapter 6's opaque market).  That market does not converge from
a cold start at a positive inventory weight: solve it by continuation from the cost-free corner,
`solve(m, continue_from={"gamma": 0})` (a long window may also need `start_policy="coarse"` for the corner itself).
A trader who sees a control's level and also what moves it (the order flow behind a quote) is privy: noisestate warns
when such a player is built naive, since its off-path reading is not pinned down.  Means (targets, constant
drifts, initial states) are solved as constants on the stationary engine, where a random walk
with no inputs has no stationary mean and is pinned at 0 and an initial state is rejected, and
as paths on the finite engines (see [method.md](method.md), "Means").  Lead atoms (`X@-0.5`) are accepted only in a
stationary loss cross term with the agent's own current control (`[c, D, X@-0.5]`),
where the covariance is computed exactly (its first-order condition carries the extra
term from flows before *t* that read the quantity after *t*); a led quantity squared,
or a lead on a control, would need the part of the kernel on shocks arriving after *t*,
which the age grid does not carry, and is rejected (write the flow with lags: at
discount 0 the time average of *X(t+tau)^2* equals that of *X(t)^2*).  Under a discount
rate *rho* the flows before *t* that read the quantity after *t* enter that extra term
weighted by up to *exp(rho tau)* relative to the current flow, so a lead with a heavy
discount is a different problem from the undiscounted one: on the Chapter 3 game with a
cross term of 0.1 in *X@-0.5* the cost is unchanged to a few percent up to *rho tau* of
2.5, a thousand times larger at 10, and the fixed point fails above that; the compile warns
when *exp(rho tau)* exceeds 100.  The second-order check IS made at a positive discount, on the
discounted objective's own form: every response to the agent's deviation weighted by
*exp(-rho tau / 2)* at its age *tau* (the average-cost form at *rho* = 0).  With a loss that is not
positive semidefinite (a trader's) its verdict depends on the discount.  The
finite engines reject leads.  With lagged *state* feedback in a drift
(`X@0.5` in the drift of `X`), the map and action-kernel iterations agree only to
first order in the node count (1.6e-5 at 24 nodes per panel on the Chapter 3 game);
the action-kernel path is the default and the more accurate one.  A game can have
several equilibria: `ties` selects the symmetric one, an untied solve from a zero
start may land on another.

**The undiscounted stationary problem is the formal one.**  The dissertation's stationary
verification fixes a *stationary admissible* profile, and admissibility requires *rho > 0* together
with a transversality condition on the discounted belief prices; the proof's terminal term vanishes
as *T* grows by exactly that condition.  At *rho = 0* the argument does not run, and those
computations solve the formal average-cost stationary equations, the *rho = 0* form of that system,
rather than a verified optimum.  The finite lag window stands in for the transversality
numerically, which is why an undiscounted model can converge on a window and still be a window
artefact: the Kyle-Back case in [guards.md](guards.md), whose profit settles neither in the window
nor in the resolution at *rho = 0* and is stable to five digits across windows of 8, 16 and 32 at
*rho = 0.5*.  `horizon.discount` defaults to 0 because the tracking examples are posed at average
cost, not because it is the safer choice.

**A stationary cost at a positive discount is the flow, not the objective.**  `res.costs` on the
stationary engine is the flow loss per unit time at every *rho*, an undiscounted Gram applied to the
equilibrium kernels; `res.cost_kind` says so.  The agent's objective is the discounted integral of
that flow.  The discount is not absent from the solve -- it enters the first-order condition through
the discounted correlation tensor, which is what makes the equilibrium depend on it -- only from the
reported scalar.  A finite horizon reports the discounted integral over [0, T] instead, so a
stationary cost and a finite one are not on the same scale, and `cost_kind` is the field that says
which you are holding.

The finite engines start every state at its `initial` value (a known number, zero when
not given, which moves the mean path only; with a past, a state without one starts at
the past's constant mean, and `initial: 0` overrides it) and integrate flow losses only.
Transitions (see [transitions.md](transitions.md)) run on the spectral finite engine: a model of kind `transition`
compiles to it.  With
a past the map on a row observed with a delay is stored in raw age (zero below the
delay; `map_convention` in the payload); the mean paths with a stationary continuation
are frozen at the new stationary means on the buffer, so their own settling by T is
part of `res.settled`.  Leads with a past are rejected as on every finite horizon.  A
transition with a delayed model is large: the band adds one L-triangle of pieces and
every square above a diagonal is split, so `ch1_delayed_finite.yaml` as its own
transition (T = 1.25, L = 1) has 56 pieces, and the two-firm Chapter 5 market at its
shipped sizes (L = 6, tau = 0.5) has 11 000 nodes x 9 primaries, beyond the dense closed
loop (it is validated at tau = 1, L = T = 2, 5 nodes).  The closure after T assumes the
transition has settled by T - L; a short T gives a biased answer that only the `settled`
row reports.  The strip's quadrature is not exact on the product of an interpolant with its
own Volterra integral along the line s = 0, so at a small trading cost the second-order check
flags the Kyle-Back trader, with or without a prior: -0.0018 at eps 0.2 and 12 nodes (-0.013
relative to the flow map's own largest curvature), not vanishing with nodes, gone at eps 1, a
correct report on the discrete objective and the quadrature's artefact on the D P cross term,
not a saddle of the market ([transitions.md](transitions.md)); a product rule for that term on the diagonal is a
candidate fix.  Terminal losses (`terminal:`) are solved on a finite horizon and on a transition ending at T; a
state with an empty `drift` and `noise` validates and is carried as its initial value.

Chapter 6: monitored deviations (`monitors:`) and instant observations are solved on the stationary engine and on the
finite engine without a past or a continuation (a transition refuses them); instant observations may not form a cycle
and are refused with ties.  The general rectangular system is solved through response kernels rather than
Proposition 6.10's gains, which agree in the linear-quadratic case; `res.foc_residual` checks the kernels.
`risk_aversion` (the entropic objective) is solved by the spectral finite engine (precommitment, J_0 over the whole
strategy, or with `settings.risk_planning = "consistent"` every date's self on its own continuation, then without a past,
monitoring, instant observations or means) with no past or a past of initial shocks only, and without a continuation; monitored deviations and instant
observations are solved (a risk-averse privy player's response gains the seed's tilt, not with initial shocks); means
are solved (without a window, and not together with `integrals`).  With an integral whose quantity is the agent's own
current control (or a control that reacts to it at once) the finite engine converges algebraically, about N^-2.4 (the
spike's point mass on the shock of its instant): 4e-6 on J at 16 nodes on Chapter 1's game.  The stationary engine
solves it under consistent planning (each date's self on its own discounted continuation) with a discount, and
without monitoring, instant observations, level rows, lagged atoms in the risk-averse agent's loss or integrals, or
means; its correction is frozen at the profile inside a best response (the fixed point is exact; the lattice's error
is reported as `richardson_gap`).  On Chapter 4's market with a risk-averse insider on wealth the price's response to a
value shock stalls short of 1 inside the window (0.61 at age 7.9 with L = 8, 0.72 with L = 16, at theta = 1), so the
pricing-error variance and the date-0 entropic cost grow with the window, while the strategies, lambda and the
expected rates agree to 1% between L = 8 and 16.  The random cost is the loss as written: a loss equal to the realised cost only in
expectation (Chapter 4's fundamental-valued flow with a moving value) is a different risk-averse game; stochastic
integrals are written with `integrals` (on the finite engine the quantity may be the agent's own current control, so a
market maker's noise-trade profit and loss `int (P - V) sigma_Z dW_Z` is written; the stationary engine refuses that).  Near the breakdown (theta lambda_max close to 1) the correction is amplified by
1 / (1 - theta lambda_max) and a solve needs more nodes and more continuation steps: on Chapter 1's game at theta = 2.5
(theta lambda_max = 0.94) 12 nodes put the entropic cost 1.4e-5 from 20 nodes, and the solve takes 150 evaluations.  The Chapter 4 example is the stationary variant, where V is a
random walk on the window and the agents keep receiving V shocks.

On the finite spectral engine the time and age panels are the multiples of the
lags closed under every lag and delay, so a lagged read and a delayed row are exact
node-to-node shifts.  A window that is not a multiple of a lag doubles the panels
(the kernels kink at `T - k tau` as well as at `k tau`: a control acting after the
lag is idle within the last lag), which quadruples the pieces and multiplies the
dense operators and the solve time by far more: `examples/ch1_delayed_finite.yaml`
solves in 3 s at `window: 1.0` (5 panels, 10 pieces) and in about 120 s at
`window: 1.1` (9 panels, 45 pieces) at 8 nodes per side.  The compile warns with the
counts when the closure adds panels; a window that is a multiple of every lag, or
fewer nodes per side, keeps the cost down.  Lags that are not multiples of one unit
(0.25 and 0.3 without `horizon.unit: 0.05`) are rejected with the unit to set.

A stationary window much longer than the kernel's support is not free: the truncated
problem can admit a second fixed point, and the solve can land on it while reporting
convergence.  `examples/ch3_two_player.yaml` at 64 nodes is the case.  Up to *L* = 15 the
state kernel decays as it should and the cost is the equilibrium's; from *L* = 18 the solve
still converges to its tolerance, and to something else entirely:

| window | residual | *K(L)* / peak | cost of player1 |
|---|---|---|---|
| 12 | 6.0e-11 | 1.2e-06 | 0.427295 |
| 15 | 4.8e-11 | 3.2e-08 | 0.427295 |
| 18 | 6.1e-11 | 8.7e-01 | 3.137288 |
| 21 | 6.5e-11 | 2.3e-01 | 3.731 |
| 24 | 1.8e-04 (stalls) | 9.2e-01 | --- |

The kernels say what happened: at *L* = 15 the state's response to its own shock falls from
1 to 1.8e-08 over the window; at *L* = 18 it falls almost linearly to *-0.87*, and at *L* = 21
it sits on a plateau near 0.65 out to age 15 before dropping off the edge.  A response that has
not decayed by the end of the window is a closed loop that does not stabilise the state, which
is not the equilibrium of the untruncated game.  At *L* = 24 that branch is ill-conditioned
enough that the fixed point stalls at a residual of 1.8e-04 and the solve reports failure ---
more evaluations do not help (2000 changed nothing: both Anderson and the Newton polish hit the
same noise floor of the map).

The window guard catches every one of these, and it is the only thing that does: `res.require_converged()`
passes at *L* = 18 because the solve did converge.  `res.require_ok()` (or `--require-ok`) is
what refuses a cost of 3.14 for a game whose answer is 0.427.

It is a cold start landing in the wrong basin, not a limit of the formulation: both fixed points
exist at *L* = 18, and warm-starting that solve from the *L* = 15 equilibrium reaches the right one
(cost 0.427295, residual 6.7e-11, tail 8.5e-10).  What separates them is best-response stability ---
`res.stability()` gives spectral radius **0.52** on the equilibrium and **1.16** and **1.09** on the
two spurious branches.  The fixed point is found by Anderson mixing and a Newton polish, which are
root finders: they solve *F(x) = x* whether or not naive best-response adjustment would go there.
That is deliberate (the Kyle-Back equilibrium is best-response unstable and genuine, see "Stability"
in the README), so an unstable radius alone does not condemn a solve --- but an unstable radius
*together with* a kernel that has not decayed at the window's edge is the spurious signature.

Continuation in the window is the remedy, and it is complete: solving at 12 and warm-starting each
larger window from the previous holds the cost at 0.427295 through *L* = 24, with the residual at
3.4e-11 and the tail falling to 7.9e-12.

    prev = ns.solve(m.with_stationary(12.0)).require_converged()
    for L in (15.0, 18.0, 21.0, 24.0):
        wider = m.with_stationary(L)
        prev = ns.solve(wider, start_from=ns.engines.stationary(wider).interpolate_maps(prev)).require_ok()

`extras/tools/ch3_long_window_branch.py` reproduces the table, the figure and the continuation.

**Time scales and long horizons.**  A finite horizon without lags is one panel: every kernel is a polynomial of
`numerics.nodes` degree in each direction over [0, T].  A game whose own time scale is much shorter than T is not
resolved there: the filter settles at the start in a Riccati layer (tanh of its rate), the value function turns at the
end in another, and a polynomial on [0, T] cannot follow a layer of width 1/rate << T.  The one-agent regulator of the
tests (rates of about 3) has its cost 0.13% off at T = 10 on 12 nodes and 190% off at T = 30 (raising nodes does not
help); the tug of war of the website's wedge page at precision p = 1000 (a filter rate of 31.6 on T = 1) has its kernels
5.6% off pointwise.  `solve()` handles this itself (`noisestate/time_panels.py`), for a model with no
`numerics.breakpoints`, no lags or delays and no past window or continuation:

1. *A grid from the time scales, before the first solve.*  The model's linear algebra gives its rates: every agent's
   Kalman filter, the Nash feedback Riccati of the controls under full information (its relaxation rate: sqrt(3/r) for
   two players pulling one integrator, where one alone has 1/sqrt(r)), the closed and open loops and the discount
   (`time_panels.time_scales`).  A tanh layer of rate k has poles pi/(2k) off the real axis, which sets the Chebyshev
   tail of a panel of width w beside it (`layer_tail`: 7.9e-4 predicted at w = 0.25 on the wedge's first panel, 8.8e-4
   measured; 3.0e-2 on its one panel, 1.7e-2 measured).  When the one panel's predicted tail is above 1e-2 (at 8 nodes or
   more), the first solve is on panels graded from both ends, first widths (powers of two) with a predicted tail of 3e-4
   (the filters' layer at 0 counted three times: the shocks born in it carry it along the diagonal), doubling to the
   middle: `[0, 1, 3, 7, 23, 27, 29, 30]` for the regulator at T = 30, `[0, 0.125, 0.375, 1]` for the wedge.
   `time_panels.suggest(model)` returns that grid without solving.  Every other model is solved on its one panel first,
   exactly as before: on the whole test suite and the benchmark every solve predicted above 1e-2 at 8 nodes or more
   failed the check on one panel, and every model the one panel resolves keeps it, bit for bit.
2. *Local refinement.*  When a result fails the resolution check, the check's own error at every node is read onto the
   breakpoint intervals, and only the intervals above `resolution_tol` are bisected, the worst first, each solve
   warm-started from the last one interpolated (a few evaluations where a cold start takes 18).  A one panel that fails
   badly (above 5e-4, or singular) and whose failure a trial at 4 nodes finds grading cuts tenfold (not one more nodes
   fix: Chapter 1's game at 4 nodes; nor one grading leaves: Kyle-Back with a random walk the trader sees) is refined the
   same way from the time-scale grid: T = 5 lands on `[0, 1, 4, 5]` with the cost to 4e-9.
3. *A budget.*  No automatic grid passes `settings.auto_panels_max` (4096 unknowns) or an estimated peak memory of
   `settings.auto_panels_memory` (1536 MB; the estimate, 100 + 8e-5 sum over the answering agents of (nU nR N)^2, is
   within 15% with the checks), and a refinement round's estimate, the best earlier result kept alive, is at most
   `settings.auto_panels_growth` (4) times the first grid's.  Where the next round would pass it, the best result is
   returned with a warning naming the breakpoints of the resolved answer and their cost; `res.panels["suggested"]`
   holds them and `res.sharpen()` re-solves there from the result.  The wedge at p = 1000 refines once (864 -> 1440
   unknowns: kernels to 5e-5 of the converged reference, the cost to 2e-9, 8 s and 0.7 GB, where the earlier version's
   halving took 56 s and 3.7 GB for the same kernels) and warns that 1e-6 needs `[0, 0.125, 0.25, 0.375, 0.53125,
   0.6875, 0.84375, 1]` (4032 unknowns, about 2.7 GB).

`res.numerics.breakpoints` records the grid used (a solve with it reproduces the result), `res.message` how it was
found, and `res.panels` the route (`"one panel"`, `"time scales"`, `"refined"`, or `"given"` for breakpoints passed in),
the rates, the history of (panels, unknowns, representation error), `resolved`, `suggested` and why the refinement
stopped.  `settings.auto_panels_max = 0` turns the automatic grids off; `res.panels["suggested"]` then still names the
time-scale grid, which is the fast-then-sharp route of an interactive page: show the one panel's answer at once, then
`res.sharpen()`.  A model whose rates are many orders of magnitude above 1/T (the regulator with its noise scaled by 1e8,
or r = 1e-8) is singular on the one panel and on the trial's grid, and the singular error, which says to rescale time,
is raised at once.  Sweeps (`ns.sweep`) and transitions keep their grids as given.  The second-order check of a large
grid is the dense form up to `second_order_dense` (8000) unknowns, assembled from the sparse row operators: 10 uniform
panels at 12 nodes (7920 unknowns) take 48 s and 2.9 GB for the whole solve, where Lanczos took 425 s and 5.8 GB.

**A stationary cost that grows with the window.**  The window row asks whether a kernel still moves at L; a random-walk
state passes it (its kernel is constant), yet a loss term in its level squared accrues the same amount at every age, so
the stationary cost is infinite and the one reported is the window's.  The `cost window` row reports the share of each
agent's flow loss that accrues over the last tenth of the window (`res.cost_tail`, 10% for a constant integrand); it is
reported and never required, since the strategies are right when the term moves nothing the agent chooses.  Chapter 4's
market maker is the case (its P^2 - 2 P V leaves out V^2): its reported cost is about 2.1 - L.

**Resolution floors.**  The Chapter 5 market's representation error used to settle at about 1.2e-6 (1.1e-6, 1.9e-6, 1.2e-6
at 12, 14 and 16 nodes), above `resolution_tol`, at ages near the window's edge.  It was not conditioning (the weighted
row operator's condition number is 502; a QR least squares gives the same 1.136e-6): the window cuts every read past L
off, so a kernel read with a lag d jumps at L - d (2.3e-6 of the peak on the prices, 7.7e-6 on the orders at
L - tau = 23.5), and that line lay inside the last geometric panel beyond `unit_range`, an error no node count removes.
The geometric panels are now cut at L - d for every lag d (one panel more on that market): the error falls with the
nodes, 3.6e-7, 1.6e-7, 7.0e-8 and 6.9e-9 at 8, 10, 12 and 14, and the shipped 8-node example passes the check (its
cost moves from 4.5271471 to 4.5271409, the 12- and 14-node values being 4.5271408).
The stationary engine's risk-averse agents (consistent planning) break down where theta times the conditional
cost's largest eigenvalue reaches 1.  The continuation in theta follows the finite engine's steps past its 0.5 step (a jump
from 0.5 to 1 landed the one-agent signal model at theta 1 on a spurious fixed point past the breakdown; the steps reach
its equilibrium, theta mu_max 0.886), a converged solution past the breakdown raises RiskBreakdown, and a path that stops
short raises it with `reached` (theta 2 on that model: 1.17, after halved steps, about two minutes).  The date-0 entropic
cost's lattices are capped at a step of 0.2: at L / 40 .. L / 160 the step, and the gap between the two finest levels,
grew with the window (6e-4, 2.5e-3, 9.3e-3 at L = 4, 8, 16), a discretisation error, not the operator's.
