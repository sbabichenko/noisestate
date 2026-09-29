"""Greedy shrinking of a fuzz finding to a smaller model that still shows it.

    python extras/fuzz/shrink.py repros/game_finite-17.yaml ["discrete reference"]

Tries, in turn and repeatedly while one of them keeps the finding: dropping an agent (its controls leave every drift,
row and loss), a signal row (one per agent is kept), a loss term (never an agent's own quadratic on a control), a
state that nothing then needs, a cross term in a drift or a row; then rounding every coefficient to one and to two
significant digits.  A step is kept when the model still validates and the named check (the first FINDING's when
none is named) is still a FINDING.  Writes <repro>.min.yaml.
"""
from __future__ import annotations

import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generate as G  # noqa: E402
import oracles as O  # noqa: E402


def still(model, family, check, depth="full"):
    try:
        import noisestate as ns
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ns.Model.from_dict(copy.deepcopy(model))
    except Exception:                                     # noqa: BLE001
        return False
    rec = O.run_case(G.Case(seed=0, family=family, model=model, tags=["shrink"]), depth=depth)
    return any(c["verdict"] == "FINDING" and c["check"] == check for c in rec["checks"])


def _drop_control(d, u):
    for s in d["states"].values():
        s["drift"] = {k: v for k, v in s.get("drift", {}).items() if k.split("@")[0] != u}
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            r["drift"] = {k: v for k, v in r.get("drift", {}).items() if k.split("@")[0] != u}
        a["loss"] = [t for t in a["loss"] if all(str(x).split("@")[0] != u for x in t[1:])]
        if a.get("instant"):
            a["instant"] = [x for x in a["instant"] if x != u] or None
            if a["instant"] is None:
                a.pop("instant")


def candidates(d):
    """Smaller models, one edit each."""
    out = []
    names = list(d["agents"])
    if len(names) > 1:
        for n in names:
            e = copy.deepcopy(d); ag = e["agents"].pop(n)
            for u in ag["controls"]:
                _drop_control(e, u)
            for a in e["agents"].values():
                if a.get("monitors"):
                    a["monitors"] = [m for m in a["monitors"] if m != n]
                    if not a["monitors"]:
                        a.pop("monitors")
            if e.get("ties"):
                e["ties"] = [[x for x in g if x != n] for g in e["ties"]]
                e["ties"] = [g for g in e["ties"] if len(g) > 1] or None
                if e["ties"] is None:
                    e.pop("ties")
            out.append(e)
    for n, a in d["agents"].items():
        if len(a.get("signals", {})) > 1:
            for r in a["signals"]:
                e = copy.deepcopy(d); e["agents"][n]["signals"].pop(r); out.append(e)
        for i, t in enumerate(a["loss"]):
            if len(t) == 3 and t[1] == t[2] and t[1] in a["controls"]:
                continue
            e = copy.deepcopy(d); e["agents"][n]["loss"].pop(i); out.append(e)
        for key in ("terminal", "risk_aversion", "monitors", "instant", "myopic"):
            if key in a:
                e = copy.deepcopy(d); e["agents"][n].pop(key); out.append(e)
        for r, row in a.get("signals", {}).items():
            for k in list(row.get("drift", {}))[1:]:
                e = copy.deepcopy(d); e["agents"][n]["signals"][r]["drift"].pop(k); out.append(e)
            for k in list(row.get("noise", {}))[1:]:
                e = copy.deepcopy(d); e["agents"][n]["signals"][r]["noise"].pop(k); out.append(e)
    for s, spec in d["states"].items():
        for k in list(spec.get("drift", {})):
            if k != s:
                e = copy.deepcopy(d); e["states"][s]["drift"].pop(k); out.append(e)
        for k in ("initial",):
            if k in spec:
                e = copy.deepcopy(d); e["states"][s].pop(k); out.append(e)
    # shocks nobody loads any more are dropped by _clean
    return [_clean(e) for e in out]


def _clean(d):
    used = set()
    for s in d["states"].values():
        used |= set(s.get("noise", {}))
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            used |= set(r.get("noise", {}))
    d["shocks"] = [w for w in d["shocks"] if w in used]
    return d


def rounded(d, sig):
    e = copy.deepcopy(d)
    r = lambda x: float(f"{float(x):.{sig}g}") if isinstance(x, (int, float)) and not isinstance(x, bool) else x
    for s in e["states"].values():
        s["drift"] = {k: r(v) for k, v in s.get("drift", {}).items()}
        s["noise"] = {k: r(v) for k, v in s.get("noise", {}).items()}
    for a in e["agents"].values():
        for row in a.get("signals", {}).values():
            row["drift"] = {k: r(v) for k, v in row.get("drift", {}).items()}
            row["noise"] = {k: r(v) for k, v in row.get("noise", {}).items()}
        a["loss"] = [[r(t[0])] + t[1:] for t in a["loss"]]
    return e


def shrink(model, family, check, log=print):
    cur = copy.deepcopy(model)
    changed = True
    while changed:
        changed = False
        for e in candidates(cur):
            if still(e, family, check):
                cur = e; changed = True; log(f"  kept an edit: {len(yaml.safe_dump(cur))} bytes")
                break
    for sig in (1, 2):
        e = rounded(cur, sig)
        if still(e, family, check):
            cur = e; log(f"  rounded to {sig} significant digit(s)"); break
    return cur


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = argv[0]
    with open(path) as f:
        text = f.read()
    model = yaml.safe_load(text)
    family = "replay"
    for line in text.splitlines():
        if line.startswith("# fuzz case"):
            family = line.split("family ")[1].split(",")[0]
    check = argv[1] if len(argv) > 1 else None
    if check is None:
        for line in text.splitlines():
            if line.startswith("# FINDING"):
                check = line.split("] ", 1)[1].split(":")[0]
                break
    print(f"shrinking {path} ({family}) on '{check}'")
    small = shrink(model, family, check)
    out = path.replace(".yaml", ".min.yaml")
    with open(out, "w") as f:
        f.write(f"# shrunk from {os.path.basename(path)} on '{check}' (family {family})\n")
        f.write(f"# fuzz case {os.path.basename(path)} (family {family}, seed 0, tags ['shrunk'])\n")
        f.write(yaml.safe_dump(small, sort_keys=False, default_flow_style=None, width=120))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
