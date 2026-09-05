# Reference solutions from the dissertation's own solvers (frozen 2026-09-03/04)

| file | producer | command |
|---|---|---|
| ch3_p3_p10_r1_r1.json | solve_spectral (Ch3 C++) | `solve_spectral 3 10 1 1 --N 24 --L 3 --uniform 61` |
| ch3_L10.json | solve_spectral | `solve_spectral 3 10 1 1 --N 64 --L 10` |
| ch4_N24_L8_e0.2_r0_q1_g1.json | kb_spectral_q (Ch4 C++) | `kb_spectral_q 24 8 0.2 0 1 "1" --uniform 41` |
| ch4_N24_L8_e0.2_r0.5_q1_g1.json | kb_spectral_q | `kb_spectral_q 24 8 0.2 0.5 1 "1" --uniform 41` |
| ch4_fixed_cpp_NT2_N64_e0.001.json | kb_spectral with the cascade patch (extras/patches/) | `kb_spectral 64 8 0.001 0 "1,1" --eps-path 0.2,0.05,0.01,0.003,0.001 --uniform 101` |
| ch5_s1_2.5_u16_endo.txt | spectral_market (Ch5 C++), raw maps | `spec_mkt_blas --N 3 --L 24 --unit 16 --tol 1e-7 s1=2.5 --eig 30` |
| ch1_spec_p3_p3.txt | spec_ch1 (Ch1 spectral prototype; diagonal s=t is unreliable) | `spec_ch1 16 16 10 --p1 3 --p2 3 --r1 0.1 --r2 0.1` |
| ch1_grid_N160.json | solve_interactive (Ch1 grid solver, first order) | `solve_interactive single 3 3 1 -1 0.1 0.1 --N 160` |

`ch1_spec_p3_p3.txt`: 16 x 16 nodes as columns `a k t s g1 g2 X0 X1 X2 calD1_0 calD1_1 calD1_2 calD2_0 calD2_1 calD2_2`
(channels 0, 1, 2 are w0, w1, w2); the run reports Jvar1 = 0.39664911.  `ch1_grid_N160.json`: 160 nodes `t` (t_i = i / 159)
with `X`, `calD1`, `calD2` (and `D1`, `D2`, `Xtilde1`, `Xtilde2`) per channel as (160, 160) arrays indexed [i, j] = (t_i, s_j),
the state kernel carrying the unit impulse on its diagonal; `J1` includes the mean part (b1 = 1, b2 = -1) in another
normalisation.  Both are read by `tests/test_ch1_refs.py` at lags >= 0.1, where the package agrees with each to its own
error (2e-3 to 4.9e-2 for spec_ch1, 3e-3 to 3e-2 for the grid solver).
