"""The Chapter 3 game's spurious long-window branch: the table and the figure of docs/limits.md.

A stationary window much longer than the kernel's support can admit a second fixed point of the
truncated problem.  Up to L = 15 the solve finds the equilibrium; from L = 18 it converges to its
tolerance and to something else, whose kernel has not decayed by the window's edge -- a closed loop
that does not stabilise the state -- and whose cost is seven times the right one.  res.require_converged() passes
on it; only the window guard, and so res.require_ok(), refuses it.

    python extras/tools/ch3_long_window_branch.py [out.png]
"""
import os
import sys

import numpy as np

import noisestate as ns
from noisestate import engines

WINDOWS = (12.0, 15.0, 18.0, 21.0, 24.0)
NODES = 64
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "examples", "ch3_two_player.yaml")


def solve_all(model=MODEL, windows=WINDOWS, nodes=NODES):
    base = ns.load(model)
    for L in windows:
        yield L, ns.solve(base.with_stationary(L).with_numerics(nodes=nodes))


def main(out=None):
    rows = list(solve_all())
    print(f"{'window':>7} {'residual':>10} {'K(L)/peak':>10} {'cost p1':>10}  verdict")
    for L, r in rows:
        K = np.asarray(r.kernel("X")); a = np.asarray(r.ages)
        tail = float(np.abs(K[a > 0.95 * L]).max() / np.abs(K).max())
        verdict = "converged, ok" if r.diagnostics.assess().accepted else ("converged, FLAGGED" if r.converged else "NOT converged")
        print(f"{L:7.1f} {r.residual:10.1e} {tail:10.1e} {r.costs['player1']:10.4f}  {verdict}")
        if r.converged and not r.diagnostics.assess().accepted:
            try:
                r.require_ok()
            except ns.ConvergenceError as exc:
                print(f"{'':7} require_ok() refuses it: {str(exc)[:88]}")
    print()
    print("the same windows, each warm-started from the previous (continuation in L):")
    prev = ns.solve(ns.load(MODEL).with_stationary(12.0).with_numerics(nodes=NODES)).require_converged()
    for L in WINDOWS[1:]:
        wider = ns.load(MODEL).with_stationary(L).with_numerics(nodes=NODES)
        r = ns.solve(wider, start_from=engines.stationary(wider).interpolate_maps(prev))
        K = np.asarray(r.kernel("X")); a = np.asarray(r.ages)
        tail = float(np.abs(K[a > 0.95 * L]).max() / np.abs(K).max())
        print(f"{L:7.1f} {r.residual:10.1e} {tail:10.1e} {r.costs['player1']:10.4f}  "
              f"{'converged, ok' if r.diagnostics.assess().accepted else 'FLAGGED'}")
        if r.converged:
            prev = r
    print()
    print("what separates the branches is best-response stability, not the residual:")
    for label, kw in (("the equilibrium (warm)", {"start_from": engines.stationary(
                           ns.load(MODEL).with_stationary(18.0).with_numerics(nodes=NODES)).interpolate_maps(prev)}),
                      ("the branch a cold start finds", {})):
        m18 = ns.load(MODEL).with_stationary(18.0).with_numerics(nodes=NODES)
        r = ns.solve(m18, **kw)
        st = r.stability()
        print(f"  L=18, {label:30s} cost {r.costs['player1']:8.4f}  spectral radius {st.radius:.4f}"
              f" ({'stable' if st.stable else 'UNSTABLE'})")
    if out is None:
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [(L, r) for L, r in rows if r.converged]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (L, r) in zip(axs.ravel(), shown):
        a = np.asarray(r.ages); K = np.asarray(r.kernel("X"))
        for k, ch in enumerate(r.channels):
            if np.abs(K[:, k]).max() > 1e-12:
                ax.plot(a, K[:, k], lw=1.2, label=ch)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"L={L:g}  cost {r.costs['player1']:.4f}  {'OK' if r.diagnostics.assess().accepted else 'FLAGGED'}",
                     fontsize=10, color=("green" if r.diagnostics.assess().accepted else "crimson"))
        ax.set_xlabel("shock age"); ax.legend(fontsize=7, frameon=False)
    for ax in axs.ravel()[len(shown):]:
        ax.axis("off")
    fig.suptitle(f"ch3_two_player: the state kernel against the lag window ({NODES} nodes)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
