# The slow tests

The fast suite (`python -m pytest -q tests`) runs in under two minutes; every solve above about five
seconds whose pin is repeated at a smaller size, by a closed form or by another test is gated behind
`NOISESTATE_SLOW=1` (`helpers.slow`, `helpers.slow_param`), which also marks the test `slow`:

    NOISESTATE_SLOW=1 python -m pytest -q tests -m slow          # the gated tests only (about 10 minutes)
    NOISESTATE_SLOW=1 python -m pytest -q tests                  # everything

CI runs the fast suite on every push and `-m slow` weekly.  Each gated test, and the one line of what it
pins (the seconds are at four BLAS threads):

| test | pins | s |
|---|---|---|
| test_baseline.py::test_current_package_matches_the_baseline_record | every shipped case against tests/refs/baseline_0.4.json (+ .npz: costs to 1e-12, evaluations, Z within 1e-12 of its peak, the distance reported); the re-baselining instrument, `python extras/compare_baseline.py write tests/refs/baseline_0.4.json` | 11 |
| test_ch5.py::test_ch5_cycle_market_matches_recorded_sweep | the Chapter 5 market's maps against the dissertation's sweep point to 6% | 5 |
| test_exact_delay.py::test_lagged_undelayed_finite_model_is_resolved_at_six_nodes | the undelayed multi-panel grid: 6 against 10 nodes to 1e-6 (the 5-node grid is test_unit_range_finite's) | 9 |
| test_finite_free.py::test_matrix_free_best_response_matches_the_dense_one_without_a_past | the matrix-free best response and fixed point on ch1_delayed at 8 nodes (the with-a-past case at 6 nodes stays fast) | 12 |
| test_means_finite.py::test_ch1_target_sweep_against_the_dissertation | the Chapter 1 target sweep, five precisions, against the dissertation and the package's own limits (p = 10 stays fast) | 6 |
| test_release_review.py::test_delayed_row_stationary_agrees_with_the_finite_engine_in_the_interior | stationary and finite kernels of a delayed row agree at t = 4 of T = 7 to 1e-3 | 20 |
| test_symmetry.py::test_solve_uses_the_symmetric_path_and_reproduces_the_equilibrium | solve() takes the symmetric closed loop and lands on the general path's equilibrium (the closed loops agree to 1e-12 in the fast test) | 6 |
| test_transition.py::test_zero_past_is_the_finite_engine_bit_for_bit[ch1_delayed_finite] | past=None is the finite engine bit for bit on the delayed example (the undelayed example stays fast) | 6 |
| test_transition.py::test_same_model_stationary_past_reproduces_the_stationary_kernels | Chapter 3 as its own past on T = 12 from a coarse start: the kernels are stationary away from the end, the end effect reaches back 3L | 34 |
| test_transition.py::test_same_model_continuation_is_exact_on_the_whole_region | the same-model identity with a continuation at 16 nodes on every node, buffer included, and from zero at 12 (the 6-node copy is test_finite_free's) | 23 |
| test_transition.py::test_regime_change_on_chapter_3_starts_from_the_old_kernels | the p1 = 3 to 10 transition's 12-node costs, settled, representation parts and the past=Model identity (8 nodes stays fast in test_transition_api) | 17 |
| test_transition.py::test_delayed_rows_with_a_past_reproduce_the_stationary_maps | the delayed example as its own transition: one best response at 8 nodes to 1e-8, the fixed point at 6 nodes, from zero too (the 3-node copy is test_unit_range_finite's) | 106 |
| test_transition.py::test_own_lagged_control_read_across_the_band_returns_the_stationary_maps[onlylag-8-floor1] | the lag as the only coercive term at 8 nodes (the loss cross term at 6 nodes stays fast) | 6 |
| test_transition_examples.py::test_ch3_precision_change_example | examples/ch3_precision_change.yaml's excess costs and settled (the T = 9 values stay fast in test_transition_result; the solve is a baseline case) | 9 |
| test_transition_means.py::test_same_model_means_are_the_stationary_constants | the mean paths of a same-model transition are the constants at 16 nodes (12 nodes with a past stays fast in test_transition) | 7 |
| test_transition_means.py::test_target_change_runs_from_the_old_means_to_the_new | the target change's mean paths, their gap at T- in settled, and the game ending at T | 6 |
| test_transition_result.py::test_same_model_loss_path_is_the_stationary_flow | the loss path of a same-model transition is the flow to 1e-9 at 16 nodes (the regime change's path stays fast) | 9 |
| test_unit_range_finite.py::test_unit_range_below_the_window_coarsens_the_grid_within_the_measured_cost | unit_range 0.5 on window 2: the cuts, N 525 for 900, costs to 5e-5 (the default's bit identity and the transition stay fast) | 8 |

Kept in the fast suite above five seconds, one per feature: the regime change's loss path (12 s), the
two-firm market's stability and window-edge second-order check (7 s), the transition sweeps (6 s), the
delayed transition at 3 nodes under unit_range (6 s), the Chapter 1 p = 10 means against the cell engine (5 s).
