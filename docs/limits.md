# Limits

What the grammar and the engines do not do, and what is done only approximately.

Scalar states and controls (write vector models as several scalars); no exact
(noise-free) observation of a state that is not itself a channel.  Means (targets, constant
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
when *exp(rho tau)* exceeds 100, and the second-order check is not made at a positive
discount.  The finite engines reject leads.  With lagged *state* feedback in a drift
(`X@0.5` in the drift of `X`), the map and action-kernel iterations agree only to
first order in the node count (1.6e-5 at 24 nodes per panel on the Chapter 3 game);
the action-kernel path is the default and the more accurate one.  A game can have
several equilibria: `ties` selects the symmetric one, an untied solve from a zero
start may land on another.

The finite engines start every state at its `initial` value (a known number, zero when
not given, which moves the mean path only; with a past, a state without one starts at
the past's constant mean, and `initial: 0` overrides it) and integrate flow losses only.
Transitions (see [transitions.md](transitions.md)) run on the spectral finite engine only: the cell engine
refuses a past, and a model of kind `transition` compiles to the spectral engine.  With
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
candidate fix.  There is no terminal cost x(T)'Qx(T), so LQ games with a terminal penalty are
outside the grammar; a state with an empty `drift` and `noise` validates and is carried
as its initial value.  The Chapter 4 example is the stationary variant, where V is a
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
