"""Where should a planner spend a fixed precision budget across three asymmetric players?

Sweeps the simplex of private precisions at a fixed total, solving the triangle game of
``examples/triangle_game.py`` at every point, and draws total cost as a ternary map.  The answer in
both regimes is an interior tilt toward the cheapest actor rather than a monopoly, and it sharpens as
effort gets cheaper: 65% of the budget at r = (0.5, 1.0, 1.5), 73% at r = (0.05, 0.10, 0.20).

The finite map is drawn at 6 nodes, where every point carries a resolution flag, so its levels are
not trustworthy; its ordering is, and ``verify()`` re-solves the interesting points at 8 nodes to
show that.  Flagged points are drawn ringed rather than dropped, so the map shows where it is soft.

    python extras/tools/precision_simplex.py stationary        # ~35 s
    python extras/tools/precision_simplex.py finite            # ~140 s
    python extras/tools/precision_simplex.py verify            # ~4 min, the resolution check
    python extras/tools/precision_simplex.py plot [out.png]    # from the saved JSON
"""
import json
import os
import sys
import time

import numpy as np

import noisestate as ns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import examples.triangle_game as tg           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL = 15.0
FLOOR = 0.25            # a signal row needs some precision; 0 would make the row pure noise
GRID = 12               # barycentric subdivisions, so (GRID+1)(GRID+2)/2 = 91 points

REGIMES = {"stationary": dict(kind="stationary", nodes=16, r=(0.5, 1.0, 1.5)),
           "finite": dict(kind="finite", nodes=6, r=(0.05, 0.10, 0.20))}


def corners(i, j, k, n=GRID):
    """The precision triple at barycentric index (i, j, k), on the fixed budget."""
    return tuple(np.array([i, j, k], float) / n * (TOTAL - 3 * FLOOR) + FLOOR)


def sweep(regime, n=GRID, out=None):
    cfg, pts, t0 = REGIMES[regime], [], time.perf_counter()
    for i in range(n + 1):
        for j in range(n + 1 - i):
            p = corners(i, j, n - i - j, n)
            try:
                res = ns.solve(tg.model(cfg["kind"], 3.0, cfg["nodes"], p=p, r=cfg["r"]))
                pts.append(dict(p=list(p), cost=float(sum(res.costs.values())),
                                per=[float(res.costs[a.name]) for a in res.model.agents],
                                conv=bool(res.converged), ok=bool(res.diagnostics.assess().accepted)))
            except Exception as exc:
                pts.append(dict(p=list(p), cost=None, conv=False, ok=False, err=type(exc).__name__))
    report(regime, pts, time.perf_counter() - t0)
    json.dump(pts, open(out or os.path.join(HERE, f"simplex_{regime}.json"), "w"))
    return pts


def report(regime, pts, secs):
    good = [q for q in pts if q["cost"] is not None and q["conv"]]
    print(f"  {regime}: {len(pts)} points in {secs:.0f}s, {len(good)} converged, "
          f"{sum(q['ok'] for q in pts)} unflagged")
    for label, q in (("best", min(good, key=lambda q: q["cost"])),
                     ("worst", max(good, key=lambda q: q["cost"])),
                     ("balanced", min(good, key=lambda q: np.std(q["p"])))):
        print(f"     {label:9s} p = ({q['p'][0]:5.2f}, {q['p'][1]:5.2f}, {q['p'][2]:5.2f})"
              f"   cost {q['cost']:.4f}")


def verify(points=((10.94, 1.44, 2.62), (10.94, 2.62, 1.44), (5.0, 5.0, 5.0),
                   (14.5, 0.25, 0.25), (9.75, 2.62, 2.62)),
           labels=("map optimum", "swap p2/p3", "balanced", "near-monopoly p1", "tilt 65%")):
    """The finite map at 6 nodes is flagged everywhere.  Re-solve its landmarks at 8 and compare."""
    for nodes in (6, 8):
        print(f"  --- {nodes} nodes ---")
        for label, p in zip(labels, points):
            res = ns.solve(tg.model("finite", 3.0, nodes, p=p, r=REGIMES["finite"]["r"]),
                           diagnostics=False)
            print(f"    {label:18s} p=({p[0]:5.2f},{p[1]:5.2f},{p[2]:5.2f})"
                  f"   cost {sum(res.costs.values()):9.4f}")


def plot(out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, regime in zip(axes, ("stationary", "finite")):
        pts = json.load(open(os.path.join(HERE, f"simplex_{regime}.json")))
        good = [q for q in pts if q["cost"] is not None and q["conv"]]
        w = np.array([q["p"] for q in good]) / TOTAL
        xy = w @ np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]])
        cost = np.array([q["cost"] for q in good])
        flagged = ~np.array([q["ok"] for q in good])

        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)], [0, 0]])
        ax.plot(tri[:, 0], tri[:, 1], color="#8a8580", lw=0.8, zorder=1)
        ax.tricontourf(xy[:, 0], xy[:, 1], cost, levels=24, cmap="magma_r", zorder=0)
        if flagged.any():
            ax.scatter(xy[flagged, 0], xy[flagged, 1], s=26, facecolors="none",
                       edgecolors="#2b2724", lw=0.7, zorder=3)
        best = xy[cost.argmin()]
        ax.scatter(*best, s=95, marker="*", color="#ffffff", edgecolors="#2b2724", lw=0.8, zorder=4)
        ctr = np.array([0.5, np.sqrt(0.75) / 3])
        ax.scatter(*ctr, s=34, marker="o", color="#ffffff", edgecolors="#2b2724", lw=0.8, zorder=4)

        b = good[int(cost.argmin())]["p"]
        ax.set_title(f"{regime}   r = {REGIMES[regime]['r']}\n"
                     f"best ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f})   cost {cost.min():.3f}",
                     fontsize=9.5, pad=14)
        for (x, y), name in zip(tri[:3], ("all to p1", "all to p2", "all to p3")):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(0, -13 if y < 0.1 else 7), ha="center", fontsize=8,
                        color="#5c564f")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Total cost over the precision simplex (budget 15, floor 0.25)", fontsize=11)
    fig.text(0.5, 0.035, "star = optimum,  circle = the balanced split,  "
             "ringed dots = a guard flagged that solve", ha="center", fontsize=8.5,
             color="#5c564f")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = out or os.path.join(HERE, "precision_simplex.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if what == "plot":
        plot(*sys.argv[2:3])
    elif what == "verify":
        verify()
    else:
        sweep(what)
