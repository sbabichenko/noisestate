<!-- Design record, copied verbatim from the transition stage's working directory (transition/design_equations.md). -->

# Transition from a stationary past: design

## 1. Domain and grid

Coordinates stay (t, a), a = t - s the shock age. The finite region becomes the rectangle **[0, T] x [0, L]**, L the past's age window; the diagonal a = t splits it into the new-shock region (a <= t, today's triangle, identical pieces and node order) and the **old-shock region** (t < a <= L, shocks born at s = -b0 = t - a < 0). Shocks are forgotten at age L in both regions, the same truncation the stationary engine applies; with T <= L the new region is today's full triangle. Reasons for (t, a) rather than (t, b0): the same-model solution K(t, a) = kappa(a) is a function of age alone, and the old kernel's kinks (its delays) read through a lag tau surface along age lines a = b_k + tau; in (t, b0) both would be anti-diagonals cutting every piece.

Breakpoints: bp_t on [0, T] as today (lags and delays closed); bp_a on [0, L] = closure under the new lags of (bp_t, the past grid's breakpoints, their shifts by the lags and delays). The two lists agree below min(T, L). Pieces: (p, q) with q < p rectangles (today), q = p the lower Duffy triangle (today) plus an **upper Duffy triangle** {t_p <= t <= a <= t_{p+1}}, q > p rectangles above. Nodes are laid out [today's N nodes; N_up upper nodes], so every index and cached path of the current engine is unchanged. A read at a = t takes the lower side unless asked for the upper (`side_diag`); the two Duffy triangles have separate nodes, so kernels may jump across the diagonal (an information structure that changes at 0).

A third family, **point shocks** (a hand-built prior: one shock at 0- loading v on the states, seen by nobody, or by a past row with a delta loading), lives on the lower triangles' hypotenuse nodes: a 1-D kernel K(t) whose every line integral stays on the line a = t, so its operators are the existing paths restricted to the diagonal nodes.

## 2. Unknowns

Per primary z: Z_new (n N, nW) as today; Z_old (n N_up, nW_past) on the **past's channels** (a past model may have channels the new one lacks; no operator crosses the diagonal, so no channel identification is needed, even for a same-model past); Z_pt (n N_diag, n_pt). Maps: g[u][r](t', b) on the lower region as today, plus **g_past[u][r'](t', b) on the upper region, one per past row r'** with its own delay d' and loading E'. A control at t weights a past raw increment at -c (c in (0, L']) at the node (t - d', c + t - d'), i.e. the map's upper region in the shifted-time convention; the increment matters only while the shocks it carries are within the window, c <= L - t, which is exactly b <= L - d': the stationary engine's `_identified` rule. For a same-model past the past rows are the new rows and the two regions read as one map.

## 3. The closed loop with old shocks

For an old shock (k, s = -b0), kappa the past kernels (zero at negative ages):

X(t) = e^{At} kappa_X(b0) + int_0^t e^{A(t-r)} sum_inputs c K_name(r - tau, s) dr, with K_name(r - tau, s) = kappa_name(b0 + r - tau) for r < tau (known segment) and the upper-region unknown for r >= tau.

u_i(t) = sum_r int_{d_r}^{t} g_{ir}(t, t-u) y_{ir}(u - d_r, s) du + sum_{r'} [ int_{-L'}^{min(0, t-d')} g^past_{ir'}(t, t - v - d') y^past_{ir'k}(v + b0) dv + E'_{r'k} g^past_{ir'}(t, t + b0 - d') ], where y_{ir}(v, s) = sum c K_n(v - l, s) reads the past when v < l.

Given the maps, the old-shock system couples only old nodes (a shock's kernel lives on its own diagonal), so the closed loop is two decoupled causal solves: today's `_solve_causal` on Z_new, untouched, and a second one on Z_old whose right-hand side is the past forcing: map-independent (e^{At} kappa_X, the Volterra known segments of the lagged inputs, the past reads of row drifts) cached per compile, plus the map-dependent part (the upper maps against the past rows) built per call like today's instantaneous entries.

Every place the three assumptions live, and the replacement: `interp` mask a <= t (a <= L, diagonal side); `read(dt, da)` zero at t - dt < 0 (old nodes: a 1-D read of the past at age a - da, returned as a second matrix on the past's nodal vector; `atom_op` -> (M, M_past), used by rows, state inputs, loss atoms); `map_shift` skipping p - k < 0 (unchanged: a control before 0 is history, already inside kappa); `_state_part` B0 = e^{Aa} sigma (old nodes: e^{At} kappa_X(a - t)), Vol path r_lo = s (max(s, 0)) and `if nm in excl: continue` (exclusion applies to the unknown segment only: the excluded agent's **past** actions are history and stay); `conv_left/right`, `response` r_lo = s + d (max(s,0) + d) plus the new `conv_past` path whose known is a 1-D kernel on the past's AgeGrid (`LinePath` gets a `known_grid`; cuts from its breakpoints); `continuation` r_hi = T (min(T, s + L), plus section 5); `projection_op` r_lo = 0 (s from t - L, an old-shock segment, and rows for upper map nodes); `row_weights`, `mass`, `mass_matrix` over [0, L] both regions; `_identified` (also b > L - d' and past rows identically zero); `world_from_actions`, `expected_cost`, `evaluate(t, s < 0)`, `interpolate_maps`, `_second_order` (strategy dimension includes the upper map).

## 4. Passive world, passive rows, FOC, projection

Agent i's information is sigma(past rows on [-L', 0), new rows on [0, t]); its past actions are measurable functionals of the past passive history, so the change of variables to passive rows holds on both sides of 0: the pre-zero passive rows are the **past model's passive world for i** (`closed_loop(past_maps, excluded=i)` on the past engine, known), the post-zero part is Zpass on the upper region: the new closed loop with i off for t >= 0, forced by the full past (state at 0, everybody's past actions in the lagged reads, the others' upper maps reading their past rows). Action kernel c = Gk gamma with Gk = [conv with the passive rows on both regions | conv_past with the past passive rows | instantaneous entries: E on new shocks, E' via map_shift(d') on old ones]; Zfull = Zpass + Resp c, Resp from r_lo = max(s, 0).

FOC per date t and shock: phi_u(t, s) = (Q zeta)_u(t, s) + sum_j int_t^{min(T, s+L)} e^{-rho(tau-t)} R^u_j(tau, tau - t) (Q zeta)_j(tau, s) dtau + own-lag terms + F_T (section 5), where zeta reads the past for lagged atoms at t < tau (a known term in bvec). Projection: E[phi_u(t) dY(v)] = 0 for every increment seen by t, including v < 0: for a lower map node the shock integral runs over s in [t - L, u] (old segment added); for an upper map node it is sum_k int_{s <= v} phi_k(t, s) y^past_{r'k}(v - s) ds + E'_{r'k} phi_k(t, v). The Gram of `maps_from_world` per time row sums over both families and gains the upper-map columns; its old-old block is the past rows' correlation over the shocks still alive (a corr quadrature on the past grid), the known pre-zero part.

Cost: J_i = sum over the three families of int_0^T e^{-rho t} 1/2 zeta' Q zeta with the rectangle's Gram and the past reads, plus the beyond-T part below.

## 5. Stationary continuation at T

Closure: from T on the world is the new model's stationary equilibrium S (kernels kappa_S, maps g_S, agent i's passive impulse responses R_S). A shock alive at T continues until age L, so for a node (t, a) with a + (T - t) <= L (new shocks with s > T - L, old ones with b0 < L - T; the rest have vanished):

F^u_T(t, a) = sum_j int_{T - t}^{L - a} e^{-rho sigma} R^u_{S,j}(sigma) (Q zeta)_{S,j}(a + sigma) dsigma, plus own-lag reads landing past T.

This is the stationary corr integral with a moving lower limit, built by `_corr_rows`-style Gauss quadrature on S's grid (`AgeGrid.corr_partial`), map-independent, cached per (agent, control), entering bvec through H. Beyond-T cost: the same integral of the flow loss over the alive shocks, reported as `continuation_costs`. Diagnostic `transition_tail`: max |K(T, a) - kappa_S(a)| / peak; above 2% the summary says the transition is not over at T (raise T). The exact alternative, a propagation region [T, T + L] under g_S (a map-independent linear operator on the transition kernels), is the upgrade path, not stage one.

## 6. Past object and API

`Past.from_stationary(res, rows_of=...)`: window L, the past AgeGrid, kappa per past channel, per agent the past rows (name, delay, E', full and passive kernels). `Past.zero(L)`, `Past.points([(loadings on states, loadings on past rows)])`. Python: `ns.solve(model, past=..., continuation=res_S | "stationary" | None)`; with a past and no continuation the new model's stationary equilibrium on window L is solved first. YAML: `horizon: {kind: transition, window: T, past: file.json, continuation: stationary}`. L must equal the continuation's window.

x0 and means: a known x0 is a public quantity and enters the **mean layer only**, as the initial condition of the mean ODE; the old regime's constant means give the mean state at 0 (added to x0) and the pre-zero mean increments of the past rows, mu' dv, which the upper maps read: the mean control at t gains sum_{r'} int g^past(t, c) mu'_{r'} dc. The kernel solve is mean-free. A prior variance is a point shock; a prior mean is x0.

## 7. Results

`TransitionResult(TriangleResult)`: `kernel(name, ch, family="new"|"past"|"point")`, `evaluate(name, ch, t, s)` with s < 0 on the old region, `maps` and `maps_past`, `costs` (both families, [0, T]), `continuation_costs`, `transition_tail`, `past`/`continuation` provenance in `to_dict()`, plots by shock time including s < 0.

## 8. Tests

T1 (bit for bit): `solve(model)` equals `solve(model, past=Past.zero(L=T), continuation=None)` by `array_equal` on Z, maps, costs, foc; structurally N_up = 0 is the same code path, and a numerically zero past on a real grid adds only products with zero arrays. T2 (same model): with past = continuation = S, the identities are (i) K = kappa_S(a) solves the extended closed loop with g_S on both regions, since e^{At} kappa(b0) + the Volterra of stationary inputs is kappa(t + b0); (ii) phi_S(a) = (Q zeta)_S(a) + int_0^{L-a} e^{-rho sigma} R_S (Q zeta)_S(a + sigma) equals the finite FOC because the split at sigma = T - t is exact; (iii) the projection on both regions is the stationary projection: assert max |K(t,a) - kappa_S(a)| < 1e-9, maps equal g_S, no dependence on T. T3 (closed form, `test_exact_delay` style): dX = (aX + D)dt + dw0, dY = hX dt + dw1 with h_old before 0 and h_new after, loss X^2 + rD^2: certainty equivalence, K = S_new/r constant (stationary Riccati), the Kalman variance P(t) from P(0) = P_old, kernels the (X, xhat) impulse responses with xhat(0) = the old filter's kernels on old shocks, cost int e^{-rho t}[(1 + rK^2) Sigma + P] with Sigma(0) = Sigma_old; the finite-horizon variant (S(T) = 0) tests stage 2 before the continuation exists. T4: a Kyle-Back point-shock start, refine and FOC/second-order checks. T5: past with different channels and a delayed past row.

## 9. Stages

S0 grid (upper pieces, diagonal side, `known_grid` paths, weights, lower-part invariance test): 3 days. S1 closed loop on old shocks with a given past (Past object, past reads, forcing, decoupled solve; same-model check with g_S): 4 days. S2 best response (passive past rows, Gk, FOC, projection, masks, Gram; T1, T3 finite variant): 5 days. S3 continuation (F_T, costs, `transition_tail`; T2, T3): 3 days. S4 point shocks, TransitionResult, YAML, docs (T4, T5): 3 days. S5 mean-layer interface (x0, mu', upper-map read): 1 day.

## 10. Risks

Panel closure of geometric past panels under new lags; memory (T = L doubles the nodes, quadruples dense operators); the closure's approximation when the transition is not over at T; conditioning of upper map columns on uninformative past rows; bit-for-bit fragility of the lower paths; kernels that do not decay within L (random-walk states) make the window a model feature.

## Key decisions
- Old-shock region in (t, age) coordinates with shared age breakpoints, not (t, b0): the same-model solution is a function of age and old-kernel kinks run along age lines.
- Finite region is [0, T] x [0, L] with the past's window applied to both shock families (forgotten at age L), not the unwindowed triangle: the stationary continuation and the same-model identity need the same truncation; T <= L keeps today's triangle.
- Old shocks are extra nodes (upper pieces) on the past's channels, decoupled from today's nodes in the closed loop, so the lower solve is untouched and bit-for-bit follows by construction; a joint per-panel solve was rejected.
- Pre-zero information is a map on the past rows (own delays and loadings) stored as the map's upper region, not the new rows extended backwards: rows may change at 0.
- Past reads enter as 1-D LinePath reads on the past's AgeGrid (known_grid) and cached forcing, not by interpolating the past onto the rectangle.
- Continuation beyond T is 'the world is stationary from T on': a known forcing F_T from the continuation's passive impulse responses and kernels via a corr quadrature with a moving lower limit, with a transition_tail diagnostic; the exact propagation region [T, T+L] is deferred.
- Hand-built priors are point shocks on the diagonal nodes (1-D kernels in t), not narrow age panels.
- The excluded agent's past actions stay in its passive world (history); only its post-zero actions are switched off, and its pre-zero passive rows are the past model's passive world.
- x0 and the old regime's means enter the mean layer only (initial condition; past mean increments read by the upper maps); the kernel solve stays mean-free.
- Costs are the discounted integral over [0, T] additive over the three families, with the beyond-T stationary part reported separately.

## Risks
- Closing geometric past panels under the new lags can shatter the age panels; require or recommend a uniform-unit past grid (unit_range = L) when lags are present.
- Memory and time: T = L doubles the nodes and quadruples the dense operators; the Gram over old shocks adds a corr quadrature per time row.
- The closure 'stationary from T on' is an approximation when the transition is not over at T; only the transition_tail diagnostic and a longer T guard it, the exact propagation region is future work.
- Upper map columns on past rows that carry no information for the new problem (hand-built pasts, duplicated rows) make the FOC system singular unless masked like the L - d rule.
- Bit-for-bit reproduction of the current engine depends on never touching the lower paths' formulas (r_lo = s + d for s >= 0) or sum orders; a refactor breaks test 1 silently at the 1e-16 level.
- Kernels that do not decay within L (random-walk states) make the window a model feature of the transition, and the same-model test then holds only against the windowed stationary solution.
- A past model with different states or channels needs name-based mapping of kappa; a state that exists only in the past is dropped except through the rows' history.
- Point shocks seen through a past row's delta loading share the map's diagonal node, which needs the upper-side read to be exact.
