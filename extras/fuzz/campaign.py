"""Run a fuzz campaign: generate seeded models, run every oracle that applies, write the records and a repro per finding.

    python extras/fuzz/campaign.py --count 200 --out /tmp/fuzz              # seeds 0..199, the family drawn per seed
    python extras/fuzz/campaign.py --family game_finite --seeds 0:50 --workers 2 --timeout 300
    python extras/fuzz/campaign.py --replay /tmp/fuzz/repros/game_finite-17.yaml

Each case runs in a forked process with a timeout (a hung or runaway solve is recorded as `timeout`, not waited on).
The output directory gets records.jsonl (one JSON record per case), summary.txt (the table by family) and repros/ (the
model of every case with a FINDING, as YAML with the failing checks in a header comment: `noisestate solve` or
`--replay` reproduces it).  Keep --workers at 2 or below on a shared machine; the BLAS threads are capped at 2.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "tests"), os.path.join(HERE, "..", "..", "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

import generate as G  # noqa: E402
import oracles as O  # noqa: E402


def _child(conn, seed, family, depth, model):
    try:
        case = G.generate(seed, family) if model is None else G.Case(seed=seed, family=family, model=model, tags=["replay"])
        rec = O.run_case(case, depth=depth)
        rec["yaml"] = case.yaml()
        conn.send(rec)
    except BaseException as exc:                 # noqa: BLE001
        import traceback
        conn.send({"id": f"{family}-{seed}", "family": family, "seed": seed, "tags": [],
                   "checks": [O.verdict("harness", False, f"{type(exc).__name__}: {exc}", kind="crash")],
                   "traceback": traceback.format_exc()})
    finally:
        conn.close()


def run_one(seed, family, depth="full", timeout=600, model=None):
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, seed, family, depth, model))
    t = time.time(); p.start(); child.close()
    rec = None
    if parent.poll(timeout):
        try:
            rec = parent.recv()
        except EOFError:
            rec = None
    p.join(5)
    if p.is_alive():
        p.kill(); p.join()
    if rec is None:
        case = G.generate(seed, family) if model is None else None
        rec = {"id": f"{family}-{seed}", "family": family, "seed": seed, "tags": case.tags if case else [],
               "checks": [O.verdict("timeout", False, f"no result within {timeout} s (exit code {p.exitcode})", kind="timeout")],
               "yaml": case.yaml() if case else ""}
    rec["wall"] = time.time() - t
    return rec


def family_of(seed, families):
    import numpy as np
    rng = np.random.default_rng([seed, 104729])
    return families[int(rng.integers(len(families)))]


def write_repro(outdir, rec):
    os.makedirs(os.path.join(outdir, "repros"), exist_ok=True)
    bad = [c for c in rec["checks"] if c["verdict"] == "FINDING"]
    head = [f"# fuzz case {rec['id']} (family {rec['family']}, seed {rec['seed']}, tags {rec.get('tags')})",
            "# replay: python extras/fuzz/campaign.py --replay <this file>"]
    for c in bad:
        head.append(f"# FINDING [{c.get('kind')}] {c['check']}: {c['detail']}"[:1000])
    path = os.path.join(outdir, "repros", f"{rec['id']}.yaml")
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n" + rec.get("yaml", ""))
    return path


def table(summary):
    lines = [f"{'family':<18} {'models':>6} {'pass':>5} {'flagged':>7} {'finding':>7} {'checks':>7} {'skipped':>7}"]
    tot = {k: 0 for k in ("models", "pass", "flagged", "finding", "checks", "skip")}
    for fam in G.FAMILIES:
        if fam in summary:
            s = summary[fam]
            lines.append(f"{fam:<18} {s['models']:>6} {s['pass']:>5} {s['flagged']:>7} {s['finding']:>7} {s['checks']:>7} {s['skip']:>7}")
            for k in tot:
                tot[k] += s[k]
    lines.append(f"{'total':<18} {tot['models']:>6} {tot['pass']:>5} {tot['flagged']:>7} {tot['finding']:>7} {tot['checks']:>7} {tot['skip']:>7}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=50, help="number of seeds (from --start)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seeds", help="a:b, overrides --count/--start")
    ap.add_argument("--family", action="append", help="restrict to these families (repeatable)")
    ap.add_argument("--depth", default="full", choices=("full", "smoke"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--budget", type=float, default=None, help="stop starting new cases after this many seconds")
    ap.add_argument("--out", default="fuzz_out")
    ap.add_argument("--replay", help="a repro YAML (or any model file) to run through the oracles")
    a = ap.parse_args(argv)
    if a.replay:
        with open(a.replay) as f:
            model = yaml.safe_load(f)
        fam = "replay"
        for line in open(a.replay):
            if line.startswith("# fuzz case"):
                fam = line.split("family ")[1].split(",")[0]
        rec = run_one(0, fam, a.depth, a.timeout, model=model)
        for c in rec["checks"]:
            print(f"{c['verdict']:<8} {c['check']}: {c['detail']}")
        return 0 if all(c["verdict"] != "FINDING" for c in rec["checks"]) else 1
    families = tuple(a.family) if a.family else G.FAMILIES
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split(":"))
    else:
        lo, hi = a.start, a.start + a.count
    jobs = [(s, families[0] if len(families) == 1 else family_of(s, families)) for s in range(lo, hi)]
    os.makedirs(a.out, exist_ok=True)
    recpath = os.path.join(a.out, "records.jsonl")
    t0 = time.time(); records = []
    from concurrent.futures import ThreadPoolExecutor
    def job(j):
        if a.budget and time.time() - t0 > a.budget:
            return None
        return run_one(j[0], j[1], a.depth, a.timeout)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(recpath, "a") as fout:
        for rec in ex.map(job, jobs):
            if rec is None:
                continue
            records.append(rec)
            fout.write(json.dumps(rec, default=str) + "\n"); fout.flush()
            v = [c["verdict"] for c in rec["checks"]]
            tag = "FINDING" if "FINDING" in v else ("flagged" if "flagged" in v else "pass")
            if tag == "FINDING":
                write_repro(a.out, rec)
            print(f"{rec['id']:<24} {tag:<8} {rec.get('wall', 0):6.1f}s " +
                  "; ".join(f"{c['check']}: {c['detail'][:90]}" for c in rec["checks"] if c["verdict"] == "FINDING")[:300], flush=True)
    summ = O.summarize(records)
    txt = table(summ) + f"\n\n{len(records)} cases in {time.time() - t0:.0f} s\n"
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
