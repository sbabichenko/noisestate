<!-- Design record, copied verbatim from the size stage's working directory (size/brief.md). -->

# Size stage brief (draft; launched after the 0.4.0 merge)

Goal: raise the finite engine's size ceiling without changing any number.
1. Closed loop assembled per time panel (never materialise M = (n_prim N)^2): _solve_causal already does block forward substitution over time panels; build each panel's row blocks (p, q <= p) from the line-path operators restricted to the panel's output rows and discard them. Same for _closed_loop_past / the buffer. Peak memory target: O(n_prim^2 N N_panel). Bit identity on every example and the transition tests.
2. Honour unit_range on the triangle grid: unit cuts only within unit_range of the diagonal (kinks weaken at the k-th delay line), coarser beyond; measured accuracy cost on ch1_delayed and the same-model identities; default keeps today's grid (unit_range = window) so nothing moves unless asked.
3. Measure: ch1_delayed as a transition at window 3 / T = 6 (29k nodes; impossible today) and the two-firm Ch5 market at tau 0.5, L = T = 6 (11k nodes x 9 primaries, 80 GB today): peak RSS, time, and the FOC Amat size (nU nR N)^2 per agent — the next ceiling; report whether a matrix-free FOC solve is needed.
