# The slow tests

The fast suite (`./run-tests`, never a bare `pytest`: see CLAUDE.md) runs in about three minutes; every solve above about
five seconds whose pin is repeated at a smaller size, by a closed form or by another test is gated behind
`NOISESTATE_SLOW=1` (`helpers.slow`, `helpers.slow_param`), which also marks the test `slow`:

    NOISESTATE_SLOW=1 ./run-tests -m slow          # the gated tests only (about 10 minutes)
    NOISESTATE_SLOW=1 ./run-tests                  # everything

CI runs the fast suite on every push and `-m slow` weekly.

The seconds here and the suite's own are at four BLAS threads, which `tests/conftest.py` now sets (it caps
OMP/OPENBLAS/MKL/NUMEXPR/VECLIB before numpy loads, leaving an explicit setting alone).  The cap is not a
detail: these solves are small enough that a widely threaded GEMM synchronises longer than it computes, and
on a 16-core machine the uncapped suite takes 743 s against 148 s capped, with the heaviest file at 75.9 s
against 16.9 s.  If a plugin that imports numpy is ever loaded before that conftest, the cap stops applying
and the suite silently returns to the slow figures.  Each gated test, and the one line of what it pins:

| test | pins | s |
|---|---|---|
| test_baseline.py::test_current_package_matches_the_baseline_record | every shipped case against tests/refs/baseline_0.4.json (+ .npz: costs to 1e-12, evaluations, Z within 1e-12 of its peak, the distance reported); the re-baselining instrument, `python extras/compare_baseline.py write tests/refs/baseline_0.4.json` | 11 |
| test_cara_finite.py::test_near_the_breakdown | risk aversion 2.5 on Chapter 1's game (theta lambda_max = 0.94): the continuation from the risk-neutral equilibrium, the costs against the reference's five-level limit to 5e-6 and 2e-6 at 16 nodes | 22 |
| test_cara_finite.py::test_the_continuation_ends_at_the_breakdown | risk aversion 3.2: the continuation stops at the breakdown (2.90 at 8 nodes) and raises RiskBreakdown saying where | 10 |
| test_cara_finite.py::test_regenerated_reference_row_agrees | the brute-force CARA reference recomputed at n = 50, 100, 200 for theta = 1: the stored table to 1e-8, its three-level limit against the engine to 2e-5 | 7 |
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
| test_transition.py::test_same_model_identity_holds_below_the_window[1.0] | the same-model identity at T = 1 (one unit, a four-panel unit-cut strip) on every node, buffer and band included (T = 1.5 stays fast) | 13 |
| test_transition_api.py::test_settle_march_finds_the_window_and_equals_the_explicit_solve | the settle march on 3 -> 10 at 12 nodes: T = 3, 6, 9, the gap sequence and its factors, the window found, the maps against the explicit solve at T = 9 (the 6-node march with settle 5e-3 stays fast) | 40 |
| test_transition_api.py::test_settle_march_equals_the_explicit_solve_at_a_tight_tol | the march's maps against the explicit solve's at tol 1e-11 (3e-11) | 55 |
| test_transition_api.py::test_settle_march_by_unit_steps | step=1 with unit 1 visits T = 1, 2, 3 at 6 nodes; the default step is one window | 10 |
| test_transition_api.py::test_excess_cost_sequences_over_the_march_windows | the excess-cost sequences at 12 nodes over T = 3, 6, 15 and the march's gap-factor tail (the 6-node mechanics stay fast) | 30 |
| test_transition_result.py::test_same_model_loss_path_is_the_stationary_flow | the loss path of a same-model transition is the flow to 1e-9 at 16 nodes (the regime change's path stays fast) | 9 |
| test_unit_range_finite.py::test_unit_range_below_the_window_coarsens_the_grid_within_the_measured_cost | unit_range 0.5 on window 2: the cuts, N 525 for 900, costs to 5e-5 (the default's bit identity and the transition stay fast) | 8 |
| test_cara_small_eps.py::test_a_small_trading_cost_converges_with_the_retry | risk-averse Kyle-Back at eps 0.05, theta 1.5, 16 nodes: the stall at the best responses' Krylov floor is retried to convergence, CE 0.53980 (the entropic probe stays fast) | 25 |
| test_cara_small_eps.py::test_the_proximal_retry_crosses_the_frozen_system_s_singularity | risk-averse Kyle-Back at eps 0.05 through theta 1.58, where the frozen best-response system goes singular: the proximal retry reaches 1.7 and 2, the certainty equivalents against extras/kyle_reference.py's five-level limit to 1e-5 | 28 (2 threads) |
| test_limits.py::test_a_thirty_time_constant_horizon_is_graded | the regulator at T = 30 graded to [0, 1, 3, 7, 23, 27, 29, 30]: the closed form to 1e-9, the resolution and second-order checks passed (T = 10 stays fast) | 16 |
| test_limits.py::test_a_hundred_time_units_are_graded_at_eight_nodes | T = 100 at 8 nodes graded to 15 panels, the closed form to 1e-7 (2950 times off on the one panel) | 68 |
| test_limits.py::test_the_chapter_5_market_is_resolved_and_the_error_falls_with_nodes | the shipped Chapter 5 market passes the resolution check at 8 nodes and its error falls threefold by 12 (the L - d cut) | 32 |
| test_limits.py::test_stationary_risk_aversion_near_the_breakdown_is_reached_in_steps | the stationary signal model at theta 1 reaches its equilibrium in the finite engine's steps (theta mu_max 0.886) | 23 |
| test_limits.py::test_stationary_risk_aversion_past_the_breakdown_raises | theta 2 raises RiskBreakdown with reached 1.17 (the halved steps; the steps' unit test stays fast) | 110 |
| test_limits.py::test_the_entropic_lattice_does_not_coarsen_with_the_window | the entropic lattice gap at L = 16 is the L = 8 one (entropic_steps' unit test stays fast) | 35 |
| test_fuzz.py::test_fuzz_campaign | the randomized campaign (extras/fuzz): NOISESTATE_FUZZ_COUNT seeded models (default 40) from NOISESTATE_FUZZ_SEED (default 1000), families drawn per seed, every oracle that applies; no FINDING (a confident wrong answer, a refusal or crash of a valid model, an invalid model accepted); NOISESTATE_FUZZ_OUT=dir writes a YAML repro per finding. `python extras/fuzz/campaign.py --help` runs larger campaigns outside pytest | ~1500 (2 threads) |

Kept in the fast suite above five seconds, one per feature: the regime change's loss path (12 s), the
two-firm market's stability and window-edge second-order check (7 s), the transition sweeps (6 s), the
delayed transition at 3 nodes under unit_range (6 s), the Chapter 1 p = 10 means against the cell engine (5 s).
