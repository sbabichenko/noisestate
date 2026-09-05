"""Command line: `noisestate solve model.yaml [-o out] [--plot] [--nodes N] [--param k=v ...]`; `--version`."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import yaml

from . import __version__, solve as _solve
from .spec import Model
from .sweep import sweep


def save_result(res, path: str) -> None:
    """Write the result's JSON payload (res.to_dict())."""
    with open(path, "w") as fh:
        json.dump(res.to_dict(), fh)



def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="noisestate", description="Solve an LQG game with private information from a model file.")
    p.add_argument("--version", action="version", version=f"noisestate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve a model file (YAML)")
    s.add_argument("model")
    s.add_argument("-o", "--out", help="write the result as JSON")
    s.add_argument("--plot", help="write a kernel plot (.pdf/.png)")
    s.add_argument("--nodes", type=int, help="override nodes per panel (stationary) or per side of each piece (finite)")
    s.add_argument("--window", type=float, help="override the lag window L")
    s.add_argument("--param", action="append", default=[], help="override a parameter, k=v (repeatable)")
    s.add_argument("--tol", type=float, default=None, help="fixed-point tolerance (default: the engine's own, 1e-10 stationary, 1e-8 finite)")
    s.add_argument("--stability", action="store_true", help="also report the stability of the equilibrium under best-response dynamics")
    s.add_argument("--refine", action="store_true", help="re-solve on a finer grid and report how much costs and kernels move")
    s.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="parse and validate a model file, print its structure")
    v.add_argument("model")
    w = sub.add_parser("sweep", help="solve along one parameter with warm starts; write a JSON list")
    w.add_argument("model"); w.add_argument("param"); w.add_argument("values", help="comma-separated values")
    w.add_argument("-o", "--out", required=True, help="output .json")
    w.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    try:
        return _run(p, args)
    except (ValueError, TypeError, NotImplementedError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:   # a model error: the message
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(p, args) -> int:
    with open(args.model) as fh:
        d = yaml.safe_load(fh)
    if args.cmd == "sweep":
        rows = sweep(d, args.param, [float(x) for x in args.values.split(",")], verbose=args.verbose)
        with open(args.out, "w") as fh:
            json.dump([{"param": r["param"], "value": r["value"], "converged": r["converged"], "evaluations": r["evaluations"],
                        "seconds": r["seconds"], "result": r["result"].to_dict()} for r in rows], fh)
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
        for note in m.notes:
            print("  note:", note)
        return 0
    for kv in args.param:
        if "=" not in kv:
            p.error(f"--param expects name=value, got {kv!r}")
        k, v_ = kv.split("=", 1)
        try:
            d.setdefault("params", {})[k] = float(v_)
        except ValueError:
            p.error(f"--param {k}: {v_!r} is not a number")
    if args.nodes is not None:
        if args.nodes < 2:
            p.error("--nodes must be at least 2")
        d.setdefault("horizon", {})["nodes"] = args.nodes
    if args.window is not None:
        if not args.window > 0:
            p.error("--window must be positive")
        d.setdefault("horizon", {})["window"] = args.window
    m = Model.from_dict(d)
    kw = {} if args.tol is None else {"tol": args.tol}
    res = _solve(m, verbose=args.verbose, refine=args.refine, stability=args.stability, **kw)
    print(res.summary())
    if args.out:
        save_result(res, args.out)
        print("wrote", args.out)
    if args.plot:
        res.plot(args.plot)
        print("wrote", args.plot)
    return 0 if res.converged else 1


if __name__ == "__main__":
    sys.exit(main())
