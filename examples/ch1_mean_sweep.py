"""The Chapter 1 game with targets: the separation failure of the mean paths.

Player 1 tracks b1 = 1 and player 2 tracks b2 = -1 on the common state, dX = (D1 + D2) dt + sigma dW0,
loss (X - b_i)^2 + r D_i^2, each seeing its own signal sqrt(p) X dt + dW_i.  The kernels do not depend on
the targets; the targets move the mean paths, which the spectral finite engine solves at the end of every
solve.  With no information (p -> 0) the mean paths are the open-loop Nash equilibrium of the deterministic
game, Dbar1(t) = (T - t) b1 / r = 10 (1 - t); with perfect information the closed-loop (feedback) Nash
equilibrium, Dbar1(0) = 4.646924; under private information they lie between, sliding from one to the other
as the signal precision p grows.  Prints Dbar1(0), Dbar1(T/2) and the mean part of the cost per p, and the
mean paths on a grid with --paths.  Usage: python examples/ch1_mean_sweep.py [--paths] [--nodes N]."""
import argparse, os
import numpy as np
import noisestate as ns

HERE = os.path.dirname(os.path.abspath(__file__))
CLOSED_LOOP = 4.646924          # Dbar1(0) with perfect information (the dissertation's closed-loop solve)


def model(p: float, nodes: int, b=(1.0, -1.0)) -> ns.Model:
    d = ns.read_yaml(os.path.join(HERE, "ch1_two_player_finite.yaml"))
    d["name"] = "ch1_targets"; d["params"].update(p1=p, p2=p); d["horizon"]["nodes"] = nodes
    for i in (1, 2):
        d["agents"][f"player{i}"]["loss"].append([-2.0 * b[i - 1], "X"])          # (X - b_i)^2 less its constant b_i^2
    return ns.Model.from_dict(d)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nodes", type=int, default=12, help="nodes per side of the triangle (12 is converged to 1e-6 up to p = 100)")
    ap.add_argument("--paths", action="store_true", help="print the mean paths on a grid of 11 times")
    ap.add_argument("--p", default="0.1,1,10,100,1000", help="comma-separated signal precisions")
    args = ap.parse_args(argv)
    T = 1.0; r = 0.1
    print(f"open-loop Dbar1(0) = {T / r:.6f}, closed-loop Dbar1(0) = {CLOSED_LOOP:.6f}")
    print(f"{'p':>8} {'Dbar1(0)':>10} {'Dbar1(T/2)':>11} {'Jbar1':>10} {'Jvar1':>10} {'evals':>5} {'s':>5}")
    for p in [float(x) for x in args.p.split(",")]:
        res = ns.solve(model(p, args.nodes)).check()
        d0, dh = res.mean("D1", [0.0, 0.5 * T])
        print(f"{p:8g} {d0:10.6f} {dh:11.6f} {res.cost_parts['player1']['mean']:10.6f} {res.cost_parts['player1']['variance']:10.6f} "
              f"{res.iterations:5d} {res.seconds:5.1f}")
        if args.paths:
            ts = np.linspace(0.0, T, 11)
            print("    t     " + " ".join(f"{t:7.2f}" for t in ts))
            for name in ("X", "D1", "D2"):
                print(f"    {name:5s} " + " ".join(f"{v:7.4f}" for v in res.mean(name, ts)))


if __name__ == "__main__":
    main()
