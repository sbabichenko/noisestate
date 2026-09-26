"""Command line: `noisestate solve model.yaml [-o out] [--plot] [--nodes N] [--engine E] [--param k=v ...]
[--max-evaluations N] [--deadline S]`; `validate model.yaml` (the schema, then the model's own checks);
`sweep model.yaml param v1,v2,... -o out.json`; `transition old.yaml new.yaml --T VALUE | --settle TOL [-o out] [--nodes N]`;
`schema {model|payload}`; `plot result.json out.pdf` (plots the saved result; `--re-solve` reproduces the solve
under its recorded options first); `--version`.
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
from .plotting import plot_payload, plot_sweep_payload
from .spec import Model
from .diagnostics import Policy, Status
from .numerics import Numerics
from .sweep import sweep
from .schema import schema, validate as schema_errors


def _exit_code(res, args) -> int:
    """0 when the solve converged, 1 when it did not.  With --require-ok a failing guard is also 1: a
    converged solve is a numerical solution of the discretised, truncated model, and a script that reads only the
    exit status would otherwise take an UNDER-RESOLVED or WINDOW TOO SHORT result as sound."""
    if not res.converged:
        return 1
    policy = {"publication": Policy.PUBLICATION, "exploratory": Policy.EXPLORATORY}[
        getattr(args, "policy", "publication")]
    verdict = res.diagnostics.assess(policy)
    if getattr(args, "require_ok", False) and not verdict.accepted:
        for line in _why_not_accepted(res, verdict):
            print(line, file=sys.stderr)
        return 1
    return 0


def _why_not_accepted(res, verdict) -> list:
    """Why --require-ok is refusing, in the words of what actually happened.

    It used to say "failed diagnostic checks: X" whatever the status was.  On ch4_kyle_back at 40
    nodes that produced "failed diagnostic checks: second_order" above a summary reading "0 failed,
    3 passed" -- two lines about one result contradicting each other, because the status was
    UNSUPPORTED and nothing had failed.  A check this engine cannot run is not a check the model
    failed, and the CLI must not collapse the two any more than the library does.
    """
    by_status: dict = {}
    for b in verdict.blocking:
        by_status.setdefault(b.status, []).append(b.check)
    lines = []
    failed = by_status.pop(Status.FAILED, [])
    if failed:
        lines.append("not ok: failed diagnostic checks: " + ", ".join(sorted(failed)) + " (see summary)")
    unsupported = by_status.pop(Status.UNSUPPORTED, [])
    if unsupported:
        lines.append("not ok: cannot be checked here: " + ", ".join(sorted(unsupported))
                     + " -- this engine cannot compute it for this model, so no setting will change it.")
    for status, checks in sorted(by_status.items(), key=lambda kv: str(kv[0])):
        lines.append(f"not ok: {status} checks: " + ", ".join(sorted(checks)) + " (see summary)")
    if verdict.policy is not Policy.EXPLORATORY:
        lines.append("       a weaker standard is legitimate and has to be named: --policy exploratory "
                     "requires convergence only.")
    return lines


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
        out.append(f"continuation: the game ends at T = {hz.extent:g}")
    else:
        nodes = m.numerics.continuation_nodes or m.numerics.nodes
        out.append(f"continuation: the new model's stationary equilibrium on a buffer of one window after T = {hz.extent:g}"
                   f" (the past's window), {nodes} nodes per panel (numerics.continuation_nodes)")
    return out


def save_result(res, path: str) -> None:
    """Write the result's JSON payload (res.to_dict())."""
    with open(path, "w") as fh:
        json.dump(res.to_dict(), fh)


def _plot_output_path(path: str) -> str:
    """The path savefig will use; Matplotlib appends its configured format when no suffix was given."""
    if os.path.splitext(path)[1]:
        return path
    import matplotlib
    return path + "." + str(matplotlib.rcParams["savefig.format"])



def _horizon_span(hz) -> str:
    """The horizon's own lengths, under the names the model file uses: `window` is the lag-truncation
    length L, `T` the terminal time, and a transition has both.  Not `extent`, which is the derived
    selector between them and is a word the user never writes."""
    parts = []
    if hz.window is not None:
        parts.append(f"window (lag) {hz.window:g}")
    if hz.T is not None:
        parts.append(f"T {hz.T:g}")
    elif hz.settle is not None:
        parts.append(f"T from the settle march ({hz.settle:g})")
    return ", ".join(parts) or "no length set"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="noisestate", description="Solve an LQG game with private information from a model file.")
    p.add_argument("--version", action="version", version=f"noisestate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve a model file (YAML)")
    s.add_argument("model")
    s.add_argument("-o", "--out", help="write the result as JSON")
    s.add_argument("--plot", help="write a kernel plot (.pdf/.png)")
    s.add_argument("--nodes", type=int, help="override numerics.nodes: per panel (stationary) or per side of each piece (finite)")
    s.add_argument("--engine", choices=("stationary", "spectral"), help="override numerics.engine")
    s.add_argument("--window", type=float, metavar="L", help="override horizon.window, the lag-truncation length L "
                   "(stationary and transition models; a finite horizon has none -- use --T)")
    s.add_argument("--T", type=float, dest="T", metavar="T", help="override horizon.T, the terminal time "
                   "(finite and transition models; a stationary model has none -- use --window)")
    s.add_argument("--param", action="append", default=[], help="override a parameter, k=v (repeatable)")
    s.add_argument("--tol", type=float, default=None, help="fixed-point tolerance (default: the engine's own, 1e-10 stationary, 1e-8 finite)")
    s.add_argument("--max-evaluations", type=int, metavar="N", help="stop after N best-response evaluations (the result is then not converged; exit 1)")
    s.add_argument("--deadline", type=float, metavar="S", help="stop after S seconds of wall time (likewise)")
    s.add_argument("--stability", action="store_true", help="also report the stability of the equilibrium under best-response dynamics")
    s.add_argument("--refine", action="store_true", help="re-solve on a finer grid and report how much costs and kernels move")
    s.add_argument("--diagnostics", action="store_true", help="explain every failed diagnostic after the compact verdict")
    s.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="check a model file against the schema and the model's own checks, print its structure")
    v.add_argument("model")
    de = sub.add_parser("describe", help="print the model as equations, delays, losses and the conventions that apply (Model.describe)")
    de.add_argument("model")
    t = sub.add_parser("transition", help="the transition from the stationary regime of old.yaml to the model of new.yaml on [0, T]")
    t.add_argument("old"); t.add_argument("new")
    t.add_argument("--T", type=float, dest="T", metavar="T", help="the terminal time T of the transition (or --settle)")
    t.add_argument("--settle", type=float, metavar="TOL", help="find the horizon by the march in T: stop when the best-response rules "
                   "on the last window are within TOL of the stationary ones (exactly one of --T and --settle)")
    t.add_argument("--step", type=float, metavar="DT", help="the march's step in T (default one window of the past; a unit step "
                   "is available but cannot certify a first window and does not pay before the panels are reused)")
    t.add_argument("--max-window", type=int, metavar="K", help="the march stops at K windows (default 8)")
    t.add_argument("--nodes", type=int, help="numerics.nodes per side of each piece (default 12)")
    t.add_argument("--past-window", type=float, metavar="L",
                   help="solve the old stationary regime with lag window L, which the continuation shares "
                        "(use when PAST WINDOW TOO SHORT or CONTINUATION WINDOW TOO SHORT)")
    t.add_argument("-o", "--out", help="write the result as JSON")
    t.add_argument("--plot", help="write the transition plot (.pdf/.png)")
    t.add_argument("--max-evaluations", type=int, metavar="N", help="as for solve")
    t.add_argument("--deadline", type=float, metavar="S", help="as for solve")
    t.add_argument("-v", "--verbose", action="store_true")
    t.add_argument("--diagnostics", action="store_true", help="explain every failed diagnostic after the compact verdict")
    sc = sub.add_parser("schema", help="print the JSON Schema (draft 2020-12) of the model file or of the result payload")
    sc.add_argument("which", choices=("model", "payload"))
    pl = sub.add_parser("plot", help="plot a result payload written by solve -o, from the saved result")
    pl.add_argument("result"); pl.add_argument("out", help="the figure (.pdf/.png)")
    pl.add_argument("--re-solve", "--resolve", dest="re_solve", action="store_true",
                    help="re-solve the payload's model under its recorded options and plot that instead of the saved result")
    ps = sub.add_parser("plot-sweep", help="plot costs, residuals, strategy changes and runtime from sweep JSON")
    ps.add_argument("result", help="the JSON written by sweep"); ps.add_argument("out", help="the figure (.pdf/.png)")
    for p_ in (s, t):
        p_.add_argument("--require-ok", action="store_true",
                        help="exit non-zero unless the policy accepts the result, not only when the solve does not converge")
        #  The library has had validation policies since 0.8 and the CLI hardcoded PUBLICATION, so a
        #  weaker standard -- which is legitimate, and which the library insists be named -- could
        #  not be asked for from the command line at all.
        p_.add_argument("--policy", choices=("publication", "exploratory"), default="publication",
                        help="which checks --require-ok demands: publication (all of them, the default) "
                             "or exploratory (convergence only). A weaker standard has to be named")
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


def _schema_check(d, path: str) -> None:
    """Raise ValueError naming every schema violation of the model file `d` with its path."""
    errs = schema_errors(d, "model")
    if errs:
        raise ValueError(f"{path} does not match the model schema:\n  " + "\n  ".join(errs))


def _run(p, args) -> int:
    if args.cmd == "schema":
        print(json.dumps(schema(args.which), indent=1))
        return 0
    if args.cmd == "plot":
        with open(args.result) as fh:
            payload = json.load(fh)
        errs = schema_errors(payload, "payload")
        if errs:
            raise ValueError(f"{args.result} does not match the payload schema:\n  " + "\n  ".join(errs))
        plot_path = _plot_output_path(args.out)
        if not args.re_solve:
            try:
                plot_payload(payload, plot_path)                 # the saved kernels; no solve
                print("wrote", plot_path)
                return 0 if payload.get("converged", True) else 1
            except KeyError as exc:                             # a payload older than the fields it needs
                print(f"the saved result does not carry {exc}; re-solving to plot it", file=sys.stderr)
        opts = payload["options"]
        solver_kw = {k: v for k, v in opts["solver"].items() if k in ("verbose",)}
        res = _solve(Model.from_dict(payload["model"]), opts["numerics"], **solver_kw,
                     **{k: v for k, v in opts["solve"].items() if k in ("start_policy", "max_evaluations", "deadline", "diagnostics")})
        res.plot(plot_path)
        print("wrote", plot_path)
        return _exit_code(res, args)
    if args.cmd == "plot-sweep":
        with open(args.result) as fh:
            rows = json.load(fh)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{args.result} is not a non-empty sweep JSON list")
        plot_path = _plot_output_path(args.out)
        plot_sweep_payload(rows, plot_path)
        print("wrote", plot_path)
        return 0 if all(row.get("converged", False) for row in rows) else 1
    bounds = {k: v for k, v in (("max_evaluations", getattr(args, "max_evaluations", None)), ("deadline", getattr(args, "deadline", None)))
              if v is not None}
    if args.cmd == "transition":
        from .transition import transition
        loaded = []
        for path in (args.old, args.new):
            with open(path) as fh:
                data = yaml.safe_load(fh)
            _schema_check(data, path); loaded.append(data)
        if (args.T is None) == (args.settle is None):
            p.error("transition takes exactly one of --T and --settle TOL")
        stationary_window = args.past_window
        old = args.old
        if stationary_window is not None:
            if stationary_window <= 0:
                p.error("the stationary window must be positive")
            old = Model.from_dict(loaded[0], base_dir=os.path.dirname(os.path.abspath(args.old)))._patch_horizon(window=stationary_window)
        if args.T is None:
            res = transition(old, args.new, settle=args.settle, step=args.step, max_window=args.max_window,
                             numerics=Numerics(nodes=args.nodes), verbose=args.verbose, **bounds)
            print(f"settle march: T = {res.extra['T']:g} ({res.march_stop}); " +
                  ", ".join(f"T = {r.T:g}: {max(r.gap.values()):.1e} in {r.evaluations} evaluations" for r in res.march))
        else:
            res = transition(old, args.new, T=args.T, numerics=Numerics(nodes=args.nodes), verbose=args.verbose, **bounds)
        print(res.summary(diagnostics=False))
        print(res.diagnostics.summary(detailed=args.diagnostics))
        if args.out:
            save_result(res, args.out); print("wrote", args.out)
        if args.plot:
            plot_path = _plot_output_path(args.plot); res.plot(plot_path); print("wrote", plot_path)
        return _exit_code(res, args)
    with open(args.model) as fh:
        d = yaml.safe_load(fh)
    _schema_check(d, args.model)
    from . import equations
    if equations.is_equation_form(d):
        d = equations.to_grammar(d)                              # the overrides below edit the grammar's keys
    base_dir = os.path.dirname(os.path.abspath(args.model))     # a relative horizon.past.model is taken from the file's directory
    if args.cmd == "sweep":
        rows = sweep(args.model, args.param, [float(x) for x in args.values.split(",")], verbose=args.verbose, **bounds)
        with open(args.out, "w") as fh:
            json.dump([r.to_dict() for r in rows], fh)
        print(f"{'value':>12}  {'status':<13} {'residual':>10} {'evals':>6} {'seconds':>8} {'change':>10}  jump")
        for r in rows:
            change = "—" if r.change is None else f"{r.change:.2e}"
            status = "ok" if r.converged else "NOT converged"
            print(f"{r.value:12g}  {status:<13} {r.result.residual:10.2e} {r.evaluations:6d} "
                  f"{r.seconds:8.2f} {change:>10}  {'yes' if r.jump else ''}")
        print("wrote", args.out, f"({len(rows)} points, {sum(r.seconds for r in rows):.1f}s)")
        return 0 if all(r.converged for r in rows) else 1
    if args.cmd == "describe":
        print(Model.from_dict(d, base_dir=base_dir).describe())
        return 0
    if args.cmd == "validate":
        m = Model.from_dict(d, base_dir=base_dir)
        print(f"{m.name}: {len(m.shocks)} shocks, {len(m.states)} states, {len(m.definitions)} definitions, "
              f"{len(m.agents)} agents, {len(m.control_names)} controls; horizon {m.horizon.kind}, "
              f"discount {m.horizon.discount}, {_horizon_span(m.horizon)}; lags {m.all_lags()}")
        for a in m.agents:
            print(f"  {a.name}: controls {a.controls}; rows {[r.name for r in a.signals]}; {len(a.loss)} loss terms"
                  + ("; myopic" if a.myopic else "") + (f"; risk aversion {a.risk_aversion:g}" if a.risk_aversion else ""))
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
    for flag, key in (("window", "window"), ("T", "T")):
        value = getattr(args, flag, None)
        if value is None:
            continue
        if not value > 0:
            p.error(f"--{flag} must be positive")
        d.setdefault("horizon", {})[key] = value
    m = Model.from_dict(d, base_dir=base_dir)
    numerics = Numerics(nodes=args.nodes, engine=args.engine, tol=args.tol)
    res = _solve(m, numerics, verbose=args.verbose, refine=args.refine, stability=args.stability, **bounds)
    print(res.summary(diagnostics=False))
    print(res.diagnostics.summary(detailed=args.diagnostics))
    if args.out:
        save_result(res, args.out)
        print("wrote", args.out)
    if args.plot:
        plot_path = _plot_output_path(args.plot)
        res.plot(plot_path)
        print("wrote", plot_path)
    return _exit_code(res, args)


if __name__ == "__main__":
    sys.exit(main())
