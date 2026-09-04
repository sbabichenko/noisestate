import os
"""Compare the finite-horizon engine with spec_ch1's spectral solution (first-order scheme: expect O(h))."""
import sys, time, numpy as np, yaml, noisestate as ns
from noisestate.finite import FiniteSolver
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
SP = REFS
ref = np.loadtxt(f"{SP}/ch1_spec_p3_p3.txt")
t_ref, s_ref = ref[:, 2], ref[:, 3]; X_ref = ref[:, 6:9]; D1_ref = ref[:, 9:12]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
d = yaml.safe_load(open("examples/ch1_two_player_finite.yaml")); d["horizon"]["nodes"] = N
t0 = time.time(); S = FiniteSolver(ns.Model.from_dict(d), verbose=True); res = S.solve(tol=1e-9)
print(res.summary(), f"| reference Jvar1 = 0.39664911")
h = S.c.h
def at(K, t, s):
    """value of a cell kernel K[i, j] (response at t_i to a unit increment in cell j) at (t, s), s < t"""
    i = np.clip(np.round(t / h).astype(int), 0, N - 1); j = np.clip(np.floor(s / h).astype(int), 0, N - 1)
    return K[i, j]
sel = (t_ref > 0.1) & (s_ref < t_ref - 0.1)     # away from the diagonal boundary layer and the corner
for name, R in (("X", X_ref), ("calD1", D1_ref)):
    for k, ch in enumerate(S.c.channels):
        K = res.kernel(name if name == "X" else "D1", ch)
        mine = at(K, t_ref[sel], s_ref[sel]); theirs = R[sel, k]
        print(f"{name} on {ch}: max|diff| {np.abs(mine-theirs).max():.3e}  rms {np.sqrt(np.mean((mine-theirs)**2)):.3e}  scale {np.abs(theirs).max():.3f}")
print(f"N={N} h={h:.4f} done in {time.time()-t0:.1f}s")
