# Validation

What the tests reproduce, with the agreement each reaches.  Every number below is the one the test pins or the
one measured when it was written; the transition and mean rows are described at length in
[transitions.md](transitions.md) and [method.md](method.md).

`tests/` reproduce the dissertation's chapter solvers from the model files:

| chapter | model | reference | agreement |
|---|---|---|---|
| 3 | two-player stationary tracking game | `solve_spectral` | 1e-11 at L = 10 (1e-5 at L = 3, window truncation) |
| 4 | Kyle-Back, one trader, rho = 0 and 0.5; the stationary variant, V a random walk on the window (see [limits.md](limits.md)) | `kb_spectral_q` | 1e-4 at 24 nodes, 1e-5 at 48 |
| 5 | purchase-order market on a 3-cycle with delay | `spectral_market` sweep (16 nodes/panel) | 8 nodes/panel: 0.05-0.2% (quotes), 0.4-1.5% (orders); 6 s at 4 BLAS threads (37 evaluations); flagged under-resolved (representation error 1.3e-5 at 8 nodes, a floor of 1.2e-6 from 12 nodes on, the costs unchanged to 1e-7 across 8 to 16) |
| 1 + delays | control lag and a delayed observation, finite horizon | cell scheme, Richardson-extrapolated | cost within 1e-4, kernels within 1e-3 at smooth ages; exact zero response before the observation delay |
| 1 | finite-horizon two-player game | `spec_ch1` (16 x 16 nodes, Tikhonov 1e-7) and the Chapter 1 grid solver (160 nodes, first order); `tests/test_ch1_refs.py` | converged at 12 nodes per side: cost 0.39690577, stable to 1e-11 to 20 nodes, and the cell scheme's Richardson pairs close on it as h^2 (0.396956 at 80/160 cells, 0.396918 at 160/320); the kernels at lags >= 0.1 agree with the references to their own error, 2e-3 to 4.9e-2 (the largest on a player's response to its own signal noise) for `spec_ch1` and 3e-3 to 3e-2 for the grid solver, which the package is closer to than `spec_ch1` is on every control kernel; near the diagonal `spec_ch1`'s own README reports weakly determined modes (0.35 apart below lag 0.1) and the cell scheme's Richardson limit agrees with the spectral engine to 2e-3; the dissertation's solvers put the cost 3e-4 lower (`spec_ch1` 0.39665; its README estimates the penalty's bias at +6e-4 on 0.39689-0.39690) |
| finite, discounted | one agent, rho = 0.5 (and 0 as the control) on [0, 3] | closed form: discounted Riccati equation, Kalman filter, closed-loop impulse responses (`tests/test_finite_discount.py`) | cost within 2e-8 and kernels within 6e-5 at 12 nodes per side; the cell scheme's error halves from 48 to 96 cells and its Richardson pair is within 5e-4 |
| means, stationary | one agent with a target, dX = (-a X + D) dt + dW; two agents with opposite targets and private signals | closed form ubar = theta a / (1 + r a (a + rho)); the open-loop and the closed-loop (coupled algebraic Riccati) Nash constants of the deterministic game (`tests/test_means.py`) | the windowed closed form to 3e-15, the exact one to the window's truncation e^{-(a + rho) L} (2.8e-8 at a = 1, rho = 0, L = 16); open-loop to 4e-8 at signal precision 1e-6, then monotone toward closed-loop, 0.5622 at precision 1000 against 0.5570 (open-loop 0.6667) |
| transitions, same-model identities | Chapter 3, `examples/ch1_delayed_finite.yaml` (a control lag and a delayed row) and the two-firm Chapter 5 market, each as its own past and continuation (`tests/test_transition.py`, `tests/test_transition_means.py`) | the stationary kernels K(t - s) on every node of the strip, the band, the last window and the buffer included; the stationary means with a target | Chapter 3 to 2e-9 at 16 nodes (settled 7e-10; the loss path equal to the stationary flow to 7e-11, excess costs 5e-11, the mean paths to 4.7e-10), ch1_delayed to 2e-9 at 8 nodes, the two-firm market to 2.3e-4 at 5 nodes (its own closed-loop floor 1.4e-4 to 2.5e-4) |
| transitions, prior start | the one-agent discounted model of `tests/test_finite_discount.py` with X(0) ~ N(0, P0), unobserved and observed at once (`tests/test_transition.py`, `tests/test_transition_result.py`) | the discounted Riccati gain with a Kalman filter from P(0) = P0 or 0, the closed-loop impulse responses, the error variance P(t) | cost within 1e-7 and kernels within 1e-5 at 16 nodes; `belief_error` equal to P(t) to 3.4e-6 on every time node, monotone from P0 = 0.8 to the stationary 0.5466 |
| transitions, Kyle-Back from a prior | `examples/kyle_back_prior.yaml`: the market started from V ~ N(0, Sigma0) seen at once by the insider, the game ending at T = 2 (`tests/test_transition_examples.py`) | the martingale price's identity lambda^2 sigma_Z^2 T = Sigma0 - Sigma_T (the market maker's map a constant, Sigma_T its belief error of V at T) and Back's eps = 0 limit sqrt(Sigma0/T)/sigma_Z = 0.7071 | at 12 nodes lambda = 0.614143 against 0.614142 (eps 0.2), 0.658872 against 0.658865 (0.1), 0.682548 against 0.682514 (0.05), 0.692248 against 0.692185 (0.03), pinned to 1e-4 at 0.2 and 0.1; lambda(0+) equal at 8, 12 and 16 nodes to 2e-6; the eps sweep converging to the limit (0.697 at 0.02) |
| means, finite (Chapter 1 with targets) | the Chapter 1 game with targets b1 = 1, b2 = -1, T = 1, r = 0.1, precisions p1 = p2 = p (`examples/ch1_mean_sweep.py`) | the dissertation's spectral solver (`spec_ch1`, 12 x 16 nodes, Tikhonov 1e-7, its Dbar1(0) converged to 0.03%): Dbar1(0), Dbar1(T/2), Jbar1 at p = 0.1, 1, 10, 100, 1000 and the p = 10 paths (`tests/refs/ch1_mean_p10.txt`); the cell engine's Richardson pairs (40, 80) and (80, 160); the deterministic LQ closed form (Riccati, solve_ivp at rtol 1e-12) for one agent (`tests/test_means_finite.py`) | 12 nodes per side, converged to 2e-5 up to p = 100 (20 nodes at p = 1000): Dbar1(0) = 9.93670, 9.47573, 7.79470, 5.98010, 5.10638 against the reference's 9.93688, 9.47708, 7.79761, 5.98117, 5.11630 (1.9e-5, 1.4e-4, 3.7e-4, 1.8e-4, 1.9e-3), Dbar1(T/2) to 1.1e-5, 7.7e-5, 1.9e-4, 5.3e-4, 1.8e-3, Jbar1 (the reference's includes the target's constant b^2 T = 1) to 5.7e-5, 4.3e-4, 1.1e-3, 1.3e-3, 3.1e-3; the p = 10 path within 1.4e-2 of the reference at every t (1.9e-3 of Dbar1(0), the largest at t = 0.075), where the cell engine's Richardson limits close on the package's path as h^2 (4.8e-3 then 1.2e-3 at T/2) and not on the reference, and the reference's variance cost is off by the same order (9e-5 at p = 10, 2e-4 at 100, 1.1e-3 at 1000): the gaps are the reference's own error, and at p = 3000 the reference is under-resolved on the sharp kernels: 4.9827 against the package's 4.9163 (1.3e-2), which 20 to 32 nodes agree on to 1e-6 and whose gap to the closed-loop limit keeps the 1/sqrt(p) law the stationary test finds; one agent alone within 1e-11 of the Riccati paths and 1e-10 of the cost, with a target, an initial state, and discounted (two states as well); open-loop 10 (1 - t) to 6.5e-9 at precision 1e-8, then monotone toward the closed-loop 4.6469 |

Kyle-Back with two traders: the reference grid solver (`kb_multi.py`), Richardson-
extrapolated, agrees with noisestate to 3-4 decimals; the C++ spectral port
`kb_spectral_q` differs by 2-6% and its solution is not a best response to itself
(the replica of the defect and its reference output are in `extras/`).

## Known open items

* (Resolved.)  A loss cross term between a control and its own lagged read (`[c, D@tau, D]`) gave the
  finite engine a second-order curvature of -3e-5 to -6e-5 relative to the largest at 6 to 8 nodes per
  side for `c = 0.15`, `r = 0.5` (positive at 4 nodes and at `c = 0.05`; the delayed and undelayed rows
  agreed).  The stationary engine never showed it: on delay-aligned panels its shift is an exact
  contraction of the cost's Gram norm, its loss form is positive definite, and the reported curvature is
  the Hessian of the discrete flow loss in the map on the passive rows (rebuilt independently to every
  printed digit; +2.4e-3 at 6 nodes, +1.8e-3 at `c = 0.15`).  On the triangle the lagged atom read
  through `read(lag, lag)`, which copies one source node onto a triangle's whole degenerate corner row:
  a 0/1 matrix of mass norm 4.8 (4 nodes) to 9.0 (6 nodes), so the discrete cross term was not bounded
  by `|c|` times the own term on every nodal vector and the loss form itself was indefinite.  Lagged
  atoms now read through the node-to-node `map_shift` (mass norm exactly 1): the form is positive
  definite, the curvature positive (+1.9e-5 at 6 nodes, +7e-6 at 8), and costs and kernels unchanged.

* (Resolved.)  The Chapter 5 cycle-market example reported a second-order curvature of -3e-5 to
  -4.6e-5, the two-firm variant on a short window -7e-4 to -2e-3.  It is window truncation: a
  firm's own price enters its loss quadratically only through lagged reads, which within the
  largest lag of the window edge fall past it, so on that slab only the current-time cross term
  between orders and prices acts, and it is indefinite.  The direction lives on the last panel,
  its mass-weighted value is identical at windows 24, 36 and 48, it is positive when embedded in a
  wider window, and the untruncated Hessian is positive definite on every grid tried; a quadratic
  in the firm's current price above c^2/r removes it (0.25 for two firms, 0.5 for three).  The
  check now re-evaluates a negative direction on a window longer by two lags and reports the
  truncation as such.
