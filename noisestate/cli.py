"""Command line: `noisestate solve model.yaml [-o out] [--plot] [--nodes N] [--param k=v ...]`."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

from .spec import Model
from .stationary import Result, StationarySolver


def result_to_dict(res: Result) -> dict:
    c = res.compiled
    out = {
        "model": res.model.name, "converged": bool(res.converged), "residual": res.residual,
        "evaluations": res.iterations, "seconds": res.seconds,
        "grid": {"breakpoints": [float(b) for b in c.grid.breakpoints], "nodes_per_panel": c.grid.n, "ages": c.grid.nodes.tolist()},
        "discount": c.rho, "channels": c.channels,
        "kernels": {}, "maps": {}, "foc": {}, "costs": {k: float(v) for k, v in res.costs.items()},
    }
    for name in c.prim:
        out["kernels"][name] = {ch: res.kernel(name)[:, k].tolist() for k, ch in enumerate(c.channels)}
    for a in res.model.agents:
        g = res.maps[a.name]
        out["maps"][a.name] = {u: {r.name: g[ui, ri].tolist() for ri, r in enumerate(a.signals)} for ui, u in enumerate(a.controls)}
        if a.name in res.foc:
            out["foc"][a.name] = {u: {part: {ch: arr[:, k].tolist() for k, ch in enumerate(c.channels)}
                                      for part, arr in dec.items()} for u, dec in res.foc[a.name].items()}
    return out


def save_result(res: Result, path: str) -> None:
    d = result_to_dict(res)
    if path.endswith(".npz"):
        flat = {"ages": res.compiled.grid.nodes}
        for name in res.compiled.prim:
            flat[f"kernel/{name}"] = res.kernel(name)
        for a in res.model.agents:
            flat[f"map/{a.name}"] = res.maps[a.name]
        np.savez(path, **flat)
    else:
        with open(path, "w") as fh:
            json.dump(d, fh)


def plot_result(res: Result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res.compiled
    controls = res.model.control_names
    states = res.model.state_names
    names = states + controls
    ncol = 2
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.6 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        K = res.kernel(name)
        for k, ch in enumerate(c.channels):
            if np.abs(K[:, k]).max() > 1e-12:
                ax.plot(c.grid.nodes, K[:, k], lw=1.1, label=ch)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{name}: kernel by channel", fontsize=10)
        ax.set_xlabel("shock age")
        ax.legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{res.model.name}  (residual {res.residual:.1e})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="noisestate", description="Solve an LQG game with private information from a model file.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve a model file (YAML)")
    s.add_argument("model")
    s.add_argument("-o", "--out", help="write results (.json or .npz)")
    s.add_argument("--plot", help="write a kernel plot (.pdf/.png)")
    s.add_argument("--nodes", type=int, help="override nodes per panel")
    s.add_argument("--window", type=float, help="override the lag window L")
    s.add_argument("--param", action="append", default=[], help="override a parameter, k=v (repeatable)")
    s.add_argument("--tol", type=float, default=1e-10)
    s.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="parse and validate a model file, print its structure")
    v.add_argument("model")
    args = p.parse_args(argv)

    with open(args.model) as fh:
        d = yaml.safe_load(fh)
    if args.cmd == "validate":
        m = Model.from_dict(d)
        print(f"{m.name}: {len(m.channels)} channels, {len(m.states)} states, {len(m.definitions)} definitions, "
              f"{len(m.agents)} agents, {len(m.control_names)} controls; horizon {m.horizon.kind}, "
              f"discount {m.horizon.discount}, window {m.horizon.window}; lags {m.all_lags()}")
        for a in m.agents:
            print(f"  {a.name}: controls {a.controls}; rows {[r.name for r in a.signals]}; {len(a.loss)} loss terms"
                  + ("; myopic" if a.myopic else ""))
        return 0
    for kv in args.param:
        k, v_ = kv.split("=", 1)
        d.setdefault("params", {})[k] = float(v_)
    if args.nodes:
        d.setdefault("horizon", {})["nodes"] = args.nodes
    if args.window:
        d.setdefault("horizon", {})["window"] = args.window
    m = Model.from_dict(d)
    if m.horizon.kind != "stationary":
        print("only the stationary solver is available in this version", file=sys.stderr)
        return 2
    res = StationarySolver(m, verbose=args.verbose).solve(tol=args.tol)
    print(res.summary())
    if args.out:
        save_result(res, args.out)
        print("wrote", args.out)
    if args.plot:
        plot_result(res, args.plot)
        print("wrote", args.plot)
    return 0 if res.converged else 1


if __name__ == "__main__":
    sys.exit(main())
