"""The shipped cases solved and reduced to a record each, against which a later package is compared.

The cases: the five shipped examples, the Chapter 1 target sweep's p = 10 point (examples/ch1_mean_sweep.py's
model at 12 nodes), Chapter 3 as its own past and continuation at 6 nodes (T = 6, from the stationary maps) and
examples/kyle_back_prior.yaml.  A record (tests/helpers.solve_record) holds the costs and their parts to full
repr, the evaluation count, the residual, `settled`, the means where they are scalars, Z's shape, Z's SHA-256
at 12 significant digits and the SHA-256 of Z's raw bytes.  A last-bit change moves Z_bits and leaves Z_sha12,
the costs (to 1e-12) and the evaluation count alone; a change of the formulation moves them.

    python extras/compare_baseline.py write tests/refs/baseline_0.4.json      # record the current package
    python extras/compare_baseline.py check tests/refs/baseline_0.4.json      # compare; exit 1 on a difference

tests/test_baseline.py runs the check under NOISESTATE_SLOW=1.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for sub in ("tests", "examples"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import noisestate as ns
from helpers import example, example_path, stationary, solve_record
from ch1_mean_sweep import model as ch1_targets

COST_TOL = 1e-12


def _ch3_same_model():
    m = example("ch3_two_player"); stat = stationary(m, 6)
    return ns.solve(m.with_horizon(kind="finite", window=6.0, nodes=6), past=stat, continuation=stat, start="stationary")


CASES = {
    "ch1_two_player_finite": lambda: ns.solve(example_path("ch1_two_player_finite")),
    "ch1_delayed_finite": lambda: ns.solve(example_path("ch1_delayed_finite")),
    "ch3_two_player": lambda: ns.solve(example_path("ch3_two_player")),
    "ch4_kyle_back": lambda: ns.solve(example_path("ch4_kyle_back")),
    "ch5_cycle_market": lambda: ns.solve(example_path("ch5_cycle_market")),
    "ch1_mean_sweep_p10": lambda: ns.solve(ch1_targets(10.0, nodes=12)),
    "ch3_same_model_T6_n6": _ch3_same_model,
    "kyle_back_prior": lambda: ns.solve(example_path("kyle_back_prior")),
}


def run(names=None, log=None):
    """The record of every case (or of `names`), each with the seconds its solve took."""
    out = {}
    for name in names or CASES:
        t0 = time.time(); rec = solve_record(CASES[name]()); rec["seconds"] = round(time.time() - t0, 2)
        out[name] = rec
        if log:
            log(f"{name:24s} {rec['evaluations']:4d} evaluations  {rec['seconds']:6.1f} s  Z {tuple(rec['Z_shape'])}  "
                f"costs {rec['costs']}")
    return out


def differences(ref, got):
    """(failures, notes): the costs beyond COST_TOL, a different evaluation count, shape or 12-digit SHA of Z
    are failures; a different raw-bytes SHA with the same 12-digit SHA is a note (a last-bit change)."""
    fails, notes = [], []
    for name, r in ref.items():
        if name not in got:
            fails.append(f"{name}: not solved"); continue
        g = got[name]
        for k in r["costs"]:
            if k not in g["costs"] or abs(g["costs"][k] - r["costs"][k]) > COST_TOL:
                fails.append(f"{name}: cost {k} {g['costs'].get(k)!r} against {r['costs'][k]!r}")
        if g["evaluations"] != r["evaluations"]:
            fails.append(f"{name}: {g['evaluations']} evaluations against {r['evaluations']}")
        if list(g["Z_shape"]) != list(r["Z_shape"]):
            fails.append(f"{name}: Z shape {g['Z_shape']} against {r['Z_shape']}")
        if g["Z_sha12"] != r["Z_sha12"]:
            fails.append(f"{name}: Z at 12 digits differs ({g['Z_sha12'][:12]} against {r['Z_sha12'][:12]})")
        elif g["Z_bits"] != r["Z_bits"]:
            notes.append(f"{name}: Z's bits differ ({g['Z_bits'][:12]} against {r['Z_bits'][:12]}), the 12-digit SHA holds")
    return fails, notes


def header():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    try:
        from importlib.metadata import version
        pkg = version("noisestate")
    except Exception:
        pkg = None
    return {"commit": commit or None, "noisestate": pkg, "numpy": np.__version__, "python": sys.version.split()[0],
            "date": time.strftime("%Y-%m-%d"), "cost_tol": COST_TOL, "Z_digits": 12}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("write", "check"):
        print(__doc__); return 2
    mode, path = argv
    got = run(log=print)
    if mode == "write":
        with open(path, "w") as fh:
            json.dump({**header(), "cases": got}, fh, indent=1, sort_keys=True)
        print("wrote", path); return 0
    with open(path) as fh:
        ref = json.load(fh)
    fails, notes = differences(ref["cases"], got)
    for line in notes:
        print("note:", line)
    for line in fails:
        print("FAIL:", line)
    print("same as" if not fails else "differs from", ref.get("commit"), f"({len(fails)} differences, {len(notes)} bit notes)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
