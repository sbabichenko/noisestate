"""Command line: `noisestate solve model.yaml [-o out] [--plot] [--nodes N] [--param k=v ...]`."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import yaml

from . import solve as _solve
from .spec import Model
from .finite_spectral import SpectralResult
from .sweep import result_to_dict, sweep


def save_result(res, path: str) -> None:
    d = result_to_dict(res)
    if path.endswith(".npz"):
        c = res.compiled
        flat = ({"t": c.g.t, "age": c.g.a} if isinstance(res, SpectralResult) else {"ages": c.grid.nodes})
        for name in res.compiled.prim:
            flat[f"kernel/{name}"] = res.kernel(name)
        for a in res.model.agents:
            flat[f"map/{a.name}"] = res.maps[a.name]
        np.savez(path, **flat)
    else:
        with open(path, "w") as fh:
            json.dump(d, fh)


def plot_result(res, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res.compiled
    if isinstance(res, SpectralResult):
        # finite horizon: plot each kernel as a function of shock time s at a few dates t
        names = res.model.state_names + res.model.control_names
        fig, axes = plt.subplots(len(names), len(c.channels), figsize=(3.6 * len(c.channels), 2.5 * len(names)), squeeze=False)
        for i, name in enumerate(names):
            for k, ch in enumerate(c.channels):
                ax = axes[i, k]
                for t in np.linspace(0.2, 1.0, 5) * c.T:
                    s = np.linspace(0, t, 200)
                    ax.plot(s, res.evaluate(name, ch, np.full_like(s, t), s), lw=1, label=f"t={t:.2f}")
                ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9); ax.set_xlabel("shock time s")
                if i == 0 and k == 0:
                    ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"{res.model.name}  (residual {res.residual:.1e})", fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150)
        return
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
    s.add_argument("--nodes", type=int, help="override nodes per panel (stationary) or per side of each piece (finite)")
    s.add_argument("--window", type=float, help="override the lag window L")
    s.add_argument("--param", action="append", default=[], help="override a parameter, k=v (repeatable)")
    s.add_argument("--tol", type=float, default=1e-10)
    s.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="parse and validate a model file, print its structure")
    v.add_argument("model")
    w = sub.add_parser("sweep", help="solve along one parameter with warm starts; write a JSON list")
    w.add_argument("model"); w.add_argument("param"); w.add_argument("values", help="comma-separated values")
    w.add_argument("-o", "--out", required=True, help="output .json")
    w.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    with open(args.model) as fh:
        d = yaml.safe_load(fh)
    if args.cmd == "sweep":
        rows = sweep(d, args.param, [float(x) for x in args.values.split(",")], verbose=args.verbose)
        with open(args.out, "w") as fh:
            json.dump([{"value": r["value"], "converged": r["converged"], "evaluations": r["evaluations"],
                        "seconds": r["seconds"], "result": result_to_dict(r["result"])} for r in rows], fh)
        print("wrote", args.out, f"({len(rows)} points, {sum(r['seconds'] for r in rows):.1f}s)")
        return 0 if all(r["converged"] for r in rows) else 1
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
    res = _solve(m, verbose=args.verbose, tol=args.tol)
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
