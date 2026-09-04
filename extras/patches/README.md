# Patches for the dissertation's Kyle-Back spectral solvers (2026-09-04)

Both multi-trader cascades build each opponent's policy rows on residual-flow observations
(flow net of the opponent's own trades) but then feed the opponent's own reaction back into its
flow observation, a structure that is only correct for total-flow rows. The reaction is therefore
double-counted and the price impact of a trader's order decays too slowly. The patches project the
cascade's policy rows on total-flow rows, the convention kb_multi.py adopted on 2026-08-16, and
leave the agent's own first-order condition (residual rows) unchanged.

    cd ~/Projects/Forecasting-And-Manipulating-The-Forecasts-Of-Others
    patch -p0 kb_spectral.cpp   < ~/Projects/noisestate/patches/kb_spectral_cascade_total_flow_rows.patch
    patch -p0 kb_spectral_q.cpp < ~/Projects/noisestate/patches/kb_spectral_q_cascade_total_flow_rows.patch

After the patch both solvers agree with noisestate and with the Richardson-extrapolated grid solver
(two traders, N=24: kernels to 4e-4, profit to six digits). One-trader results are unchanged.
