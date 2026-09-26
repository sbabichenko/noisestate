# Limits

What the grammar and the engines do not do, and what is done only approximately.

Vectors exist in the Python form only (`ns.State("X", 3)`, matrix coefficients), where they expand into scalar
components; the file form has no vector syntax.  No exact (noise-free) observation of a state that is not itself a
shock; an instant observation of another agent's control level (`{level: P}`) is a reaction, not information that
enters the agent's filter, so a market whose trader learns from the quote (Chapter 6's opaque market) is not solved.  Means (targets, constant
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
when *exp(rho tau)* exceeds 100.  The second-order check IS made at a positive discount: the
discounted objective is a quadratic form whose joint running Hessian carries no *rho*, so the
verdict is the same at every discount and the check runs on the average-cost system.  The
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
`risk_aversion` (the entropic objective) is declared and saved but refused by every engine.  The Chapter 4 example is the stationary variant, where V is a
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
