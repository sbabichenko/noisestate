"""The shipped cases solved and reduced to a record each, against which a later package is compared.

The cases: the six shipped examples (ch3_precision_change.yaml is the file form of a transition), the Chapter 1 target sweep's p = 10 point (examples/ch1_mean_sweep.py's
model at 12 nodes), Chapter 3 as its own past and continuation at 6 nodes (T = 6, from the stationary maps) and
examples/kyle_back_prior.yaml.  A record (tests/helpers.solve_record) holds the costs and their parts to full
repr, the evaluation count, the residual, `settled`, the means where they are scalars, Z's shape and the
SHA-256 of Z's raw bytes (information: a last-bit change moves it); Z itself is stored as float64 in the .npz
next to the .json.  The check compares the costs to COST_TOL, the evaluation counts and Z's shape exactly, and
Z by max |dZ| / max |Z| against Z_TOL (the value is reported per case: BLAS rounding sits at 1e-16 to 1e-13,
a change of the formulation far above 1e-12).

    python extras/compare_baseline.py write tests/refs/baseline_0.4.json      # record the current package (+ .npz)
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
Z_TOL = 1e-12                    # max |dZ| / max |Z| per case


def npz_path(path: str) -> str:
    """The .npz holding every case's Z, next to the .json record."""
    return os.path.splitext(path)[0] + ".npz"


def _ch3_same_model():
    m = example("ch3_two_player"); stat = stationary(m, 6)
    return ns.solve(m.with_finite(6.0).with_numerics(nodes=6), past=stat, continuation=stat, start="stationary")


CASES = {
    "ch1_two_player_finite": lambda: ns.solve(example_path("ch1_two_player_finite")),
    "ch1_delayed_finite": lambda: ns.solve(example_path("ch1_delayed_finite")),
    "ch3_two_player": lambda: ns.solve(example_path("ch3_two_player")),
    "ch4_kyle_back": lambda: ns.solve(example_path("ch4_kyle_back")),
    "ch5_cycle_market": lambda: ns.solve(example_path("ch5_cycle_market")),
    "ch3_precision_change": lambda: ns.solve(example_path("ch3_precision_change")),
    "ch1_mean_sweep_p10": lambda: ns.solve(ch1_targets(10.0, nodes=12)),
    "ch3_same_model_T6_n6": _ch3_same_model,
    "kyle_back_prior": lambda: ns.solve(example_path("kyle_back_prior")),
}


def run(names=None, log=None):
    """The record of every case (or of `names`), each with the seconds its solve took and, under "Z", the
    solved Z itself (float64; kept out of the JSON, written to the .npz)."""
    out = {}
    for name in names or CASES:
        t0 = time.time(); res = CASES[name](); rec = solve_record(res); rec["seconds"] = round(time.time() - t0, 2)
        rec["Z"] = np.asarray(res.world, dtype=float)
        out[name] = rec
        if log:
            log(f"{name:24s} {rec['evaluations']:4d} evaluations  {rec['seconds']:6.1f} s  Z {tuple(rec['Z_shape'])}  "
                f"costs {rec['costs']}")
    return out


def differences(ref, got, Zref=None):
    """(failures, notes): the costs beyond COST_TOL, a different evaluation count or shape of Z, and Z beyond
    Z_TOL in max |dZ| / max |Z| against Zref[name] are failures; every case's Z distance is a note (with
    "bits differ" when the raw-bytes SHA moved), so a last-bit change is seen and allowed."""
    fails, notes = [], []
    for name, r in ref.items():
        if name not in got:
            fails.append(f"{name}: not solved"); continue
        g = got[name]
        if Zref is not None and name in Zref and "Z" in g and list(g["Z_shape"]) == list(r["Z_shape"]):
            Zr = np.asarray(Zref[name]); Zg = g["Z"]
            dist = float(np.abs(Zg - Zr).max() / max(np.abs(Zr).max(), 1e-300))
            bits = "" if g["Z_bits"] == r["Z_bits"] else ", bits differ"
            (fails if dist > Z_TOL else notes).append(f"{name}: Z max |dZ| / max |Z| = {dist:.2e}{bits}")
        for k in r["costs"]:
            if k not in g["costs"] or abs(g["costs"][k] - r["costs"][k]) > COST_TOL:
                fails.append(f"{name}: cost {k} {g['costs'].get(k)!r} against {r['costs'][k]!r}")
        if g["evaluations"] != r["evaluations"]:
            fails.append(f"{name}: {g['evaluations']} evaluations against {r['evaluations']}")
        if list(g["Z_shape"]) != list(r["Z_shape"]):
            fails.append(f"{name}: Z shape {g['Z_shape']} against {r['Z_shape']}")
    return fails, notes


def load(path: str):
    """(the JSON record, the cases' Z from the .npz or None when it is missing)."""
    with open(path) as fh:
        ref = json.load(fh)
    zp = npz_path(path)
    Zref = dict(np.load(zp)) if os.path.exists(zp) else None
    return ref, Zref


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
            "date": time.strftime("%Y-%m-%d"), "cost_tol": COST_TOL, "Z_tol": Z_TOL}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("write", "check"):
        print(__doc__); return 2
    mode, path = argv
    got = run(log=print)
    if mode == "write":
        Zs = {name: rec.pop("Z") for name, rec in got.items()}
        with open(path, "w") as fh:
            json.dump({**header(), "cases": got}, fh, indent=1, sort_keys=True)
        np.savez(npz_path(path), **Zs)
        print("wrote", path, "and", npz_path(path)); return 0
    ref, Zref = load(path)
    fails, notes = differences(ref["cases"], got, Zref)
    for line in notes:
        print("note:", line)
    for line in fails:
        print("FAIL:", line)
    print("same as" if not fails else "differs from", ref.get("commit"), f"({len(fails)} differences, {len(notes)} notes)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
