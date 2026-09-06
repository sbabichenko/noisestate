"""Command line: `noisestate solve model.yaml [-o out] [--plot] [--nodes N] [--param k=v ...] [--max-evaluations N]
[--deadline S]`; `--version`.
Exit status: 0 converged (a sweep: every point), 1 solved but not converged, 2 a usage error (a bad option, a
missing or unreadable model file, bad YAML, an output path that cannot be written) or an error the package
raises (a model or solver problem), printed as `error: ...` on stderr."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

from . import __version__, solve as _solve
from .spec import Model
from .numerics import Numerics
from .sweep import sweep


def transition_lines(m: Model) -> list:
    """The structure of a transition model's horizon: the past, the continuation and its sizing."""
    hz = m.horizon; past = hz.past or {}; out = []
    if isinstance(past.get("model"), str):
        out.append(f"past: the stationary model file {past['model']}")
    elif isinstance(past.get("model"), dict):
        out.append(f"past: the inline stationary model {past['model'].get('name', 'model')!r}")
    for sh in past.get("initial") or []:
        out.append(f"initial shock {sh.get('name', '?')}: loads {sh.get('loads') or {}}, seen on rows {sh.get('rows') or {}}")
    cont = hz.continuation or "stationary"
    if cont == "end":
        out.append(f"continuation: the game ends at T = {hz.window:g}")
    else:
        st = hz.stationary or {}
        out.append(f"continuation: the new model's stationary equilibrium on a buffer of one window after T = {hz.window:g}"
                   + (f" (window {st['window']:g})" if st.get("window") is not None else " (the past's window)")
                   + f", {st.get('nodes', hz.nodes)} nodes per panel (numerics.continuation_nodes)")
    return out


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
    s.add_argument("--nodes", type=int, help="override numerics.nodes: per panel (stationary) or per side of each piece (finite)")
    s.add_argument("--engine", choices=("stationary", "spectral", "cells"), help="override numerics.engine")
    s.add_argument("--window", type=float, help="override the lag window L")
    s.add_argument("--param", action="append", default=[], help="override a parameter, k=v (repeatable)")
    s.add_argument("--tol", type=float, default=None, help="fixed-point tolerance (default: the engine's own, 1e-10 stationary, 1e-8 finite)")
    s.add_argument("--max-evaluations", type=int, metavar="N", help="stop after N best-response evaluations (the result is then not converged; exit 1)")
    s.add_argument("--deadline", type=float, metavar="S", help="stop after S seconds of wall time (likewise)")
    s.add_argument("--stability", action="store_true", help="also report the stability of the equilibrium under best-response dynamics")
    s.add_argument("--refine", action="store_true", help="re-solve on a finer grid and report how much costs and kernels move")
    s.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="parse and validate a model file, print its structure")
    v.add_argument("model")
    w = sub.add_parser("sweep", help="solve along one parameter with warm starts; write a JSON list")
    w.add_argument("model"); w.add_argument("param"); w.add_argument("values", help="comma-separated values")
    w.add_argument("-o", "--out", required=True, help="output .json")
    w.add_argument("--max-evaluations", type=int, metavar="N", help="per point, as for solve")
    w.add_argument("--deadline", type=float, metavar="S", help="per point, as for solve")
    w.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    try:
        return _run(p, args)
    except (ValueError, TypeError, NotImplementedError, RuntimeError, ImportError, np.linalg.LinAlgError,
            OSError, yaml.YAMLError) as exc:              # a model, solver or file error: the message, exit 2
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(p, args) -> int:
    with open(args.model) as fh:
        d = yaml.safe_load(fh)
    base_dir = os.path.dirname(os.path.abspath(args.model))     # a relative horizon.past.model is taken from the file's directory
    bounds = {k: v for k, v in (("max_evaluations", getattr(args, "max_evaluations", None)), ("deadline", getattr(args, "deadline", None)))
              if v is not None}
    if args.cmd == "sweep":
        rows = sweep(d, args.param, [float(x) for x in args.values.split(",")], solve_kw=bounds, verbose=args.verbose)
        with open(args.out, "w") as fh:
            json.dump([{"param": r["param"], "value": r["value"], "converged": r["converged"], "evaluations": r["evaluations"],
                        "seconds": r["seconds"], "result": r["result"].to_dict()} for r in rows], fh)
        print("wrote", args.out, f"({len(rows)} points, {sum(r['seconds'] for r in rows):.1f}s)")
        return 0 if all(r["converged"] for r in rows) else 1
    if args.cmd == "validate":
        m = Model.from_dict(d, base_dir=base_dir)
        print(f"{m.name}: {len(m.channels)} channels, {len(m.states)} states, {len(m.definitions)} definitions, "
              f"{len(m.agents)} agents, {len(m.control_names)} controls; horizon {m.horizon.kind}, "
              f"discount {m.horizon.discount}, window {m.horizon.window}; lags {m.all_lags()}")
        for a in m.agents:
            print(f"  {a.name}: controls {a.controls}; rows {[r.name for r in a.signals]}; {len(a.loss)} loss terms"
                  + ("; myopic" if a.myopic else ""))
        if m.horizon.kind == "transition":
            for line in transition_lines(m):
                print("  " + line)
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
    if args.nodes is not None and args.nodes < 2:
        p.error("--nodes must be at least 2")
    if args.window is not None:
        if not args.window > 0:
            p.error("--window must be positive")
        d.setdefault("horizon", {})["window"] = args.window
    m = Model.from_dict(d, base_dir=base_dir)
    numerics = Numerics(nodes=args.nodes, engine=args.engine, tol=args.tol)
    res = _solve(m, numerics, verbose=args.verbose, refine=args.refine, stability=args.stability, **bounds)
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
