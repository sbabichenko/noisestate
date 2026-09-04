import sys, time, numpy as np, yaml, noisestate as ns
from noisestate.finite_spectral import SpectralFiniteSolver
SP = "/tmp/scratch"
ref = np.loadtxt(f"{SP}/ch1_spec_p3_p3.txt"); t_ref, s_ref = ref[:, 2], ref[:, 3]; X_ref = ref[:, 6:9]; D1_ref = ref[:, 9:12]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
d = yaml.safe_load(open("examples/ch1_two_player_finite.yaml")); d["horizon"]["nodes"] = n
t0 = time.time(); S = SpectralFiniteSolver(ns.Model.from_dict(d), verbose=True); print(f"compiled {time.time()-t0:.1f}s, N={S.c.N}")
t0 = time.time(); res = S.solve(); print(res.summary(), f"| reference Jvar1 = 0.39664911  ({time.time()-t0:.1f}s)")
sel = t_ref > 1e-9
for name, R, key in (("X", X_ref, "X"), ("calD1", D1_ref, "D1")):
    for k, ch in enumerate(S.c.channels):
        mine = res.evaluate(key, ch, t_ref[sel], s_ref[sel]); theirs = R[sel, k]
        print(f"{name} on {ch}: max|diff| {np.abs(mine-theirs).max():.2e}  rms {np.sqrt(np.mean((mine-theirs)**2)):.2e}  scale {np.abs(theirs).max():.3f}")
