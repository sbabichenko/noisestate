"""The oracles of the fuzz campaign (extras/fuzz): every check that applies to a generated case, each returning a verdict.

The rule every check applies: a result must be correct within its stated accuracy or say so.  A disagreement with an
oracle is
  ok        within the tolerance,
  flagged   outside it, but the package said so (not converged, a required diagnostic failed, a warning),
  FINDING   outside it on a result the package accepted (converged and Policy.PUBLICATION accepted), or a crash or
            refusal of a valid model of a supported kind, or an invalid model accepted,
  skip      the oracle does not apply or could not be computed (a reference limitation, recorded with the reason).

Oracles: (a) the independent discrete reference (reference.py, Richardson over three levels), (c) the one-agent
closed form (lqg.py), (b) cross-engine: nodes + 4, the iteration variable, a window 1.5 L, explicit breakpoints, the
matrix-free best response, the stationary maps as a transition's past (one best response on the strip), (d)
invariances: agent order, shock relabelling, a shock split in two, an irrelevant state, a zero-weight term, the units
of a state and of a control, the time unit, the equations form round trip, ties against the untied solve, (e) theta ->
0 against the risk-neutral solve, (f) self-consistency: finite costs and parts summing, a nonnegative cost for a
nonnegative loss, the second-order check on a convex loss, no MISSING status, the best response against random feasible
perturbations.
"""
from __future__ import annotations

import copy
import math
import time
import traceback
import warnings

import numpy as np

import noisestate as ns
from noisestate.diagnostics import Policy, Status

import generate as G
import lqg
from reference import ReferenceError, solve_levels


# ------------------------------------------------------------------------------------------------ solving
class Solved:
    """A solve and what it said: the result or the exception, the warnings, whether it was accepted."""

    def __init__(self, d, numerics=None, **kw):
        self.d = d; self.res = None; self.exc = None; self.warnings = []
        t = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                m = ns.Model.from_dict(copy.deepcopy(d))
                self.res = ns.solve(m, numerics, **kw) if numerics else ns.solve(m, **kw)
            except Exception as exc:          # noqa: BLE001 -- classified by the caller
                self.exc = exc; self.tb = traceback.format_exc(limit=6)
        self.warnings = [str(x.message) for x in w]
        self.seconds = time.time() - t

    @property
    def ok(self):
        return self.res is not None

    @property
    def converged(self):
        return self.ok and bool(self.res.converged)

    @property
    def accepted(self):
        if not self.converged:
            return False
        try:
            return bool(self.res.diagnostics.assess(Policy.PUBLICATION).accepted)
        except Exception:                      # noqa: BLE001
            return False

    # warnings about the model's form, not the result's accuracy
    BENIGN = ("no strictly positive quadratic term",)

    @property
    def accuracy_warnings(self):
        return [w for w in self.warnings if not any(b in w for b in self.BENIGN)]

    @property
    def said_so(self):
        """The package flagged this result: not converged, not accepted, or a warning about it."""
        return (not self.accepted) or bool(self.accuracy_warnings)

    def costs(self):
        return {k: float(v) for k, v in self.res.costs.items()}

    def failed_checks(self):
        if not self.ok:
            return []
        return [k for k, s in self.res.diagnostics.statuses.items() if s is Status.FAILED]

    def brief(self):
        if not self.ok:
            return f"{type(self.exc).__name__}: {str(self.exc)[:200]}"
        return (f"converged={self.res.converged} residual={float(self.res.residual):.1e} failed={self.failed_checks()} "
                f"warnings={[w[:80] for w in self.warnings]}")


def gross_said_so(solved, diff):
    """Whether `solved` said enough about an error of relative size `diff`: a result refused by more than the
    resolution check (not converged, a window, a second-order failure, a warning) said so; one whose only failure is the
    resolution check reported a representation error, which vouches for about three orders of magnitude of it at most:
    an error far beyond max(1e-3, 1000 x the representation error) was not announced."""
    if not solved.said_so:
        return False
    if not solved.converged or solved.accuracy_warnings:
        return True
    failed = set(solved.failed_checks())
    if failed - {"resolution"}:
        return True
    rep = max(solved.res.representation_error.values()) if solved.res.representation_error else 0.0
    return diff <= max(1e-3, 1000.0 * rep)


def verdict(name, ok, detail="", said_so=False, skip=False, kind="accuracy"):
    if skip:
        v = "skip"
    elif ok:
        v = "ok"
    else:
        v = "flagged" if said_so else "FINDING"
    return {"check": name, "verdict": v, "detail": detail, "kind": kind}


def rel(a, b, scale):
    return abs(a - b) / scale


def cost_scale(*dicts):
    vals = [abs(v) for d in dicts for v in d.values() if v is not None and np.isfinite(v)]
    return max([1e-2] + vals)


def compare_costs(name, base, other_costs, tol, other_said=False, detail_prefix="", factor=1.0):
    """base: Solved; other_costs: dict; the check fails when any agent differs by more than tol (relative to the
    largest cost)."""
    bc = base.costs()
    sc = cost_scale(bc, other_costs)
    worst = max(rel(bc[k], factor * other_costs[k], sc) for k in bc)
    detail = f"{detail_prefix}max rel diff {worst:.2e} (tol {tol:.0e}); pkg {fmt(bc)} other {fmt({k: factor * v for k, v in other_costs.items()})}"
    return verdict(name, worst <= tol, detail, said_so=gross_said_so(base, worst) or other_said)


def fmt(d):
    return "{" + ", ".join(f"{k}: {v:.8g}" for k, v in d.items()) + "}"


# ------------------------------------------------------------------------------------------------ transformations
def _map_lin(expr, f):
    """Apply f(coef, atom) -> list of (coef, atom) to a linear expression (dict form out)."""
    items = expr.items() if isinstance(expr, dict) else [(a, c) for c, a in expr]
    out = {}
    for a, c in items:
        for c2, a2 in f(float(c), a):
            out[a2] = out.get(a2, 0.0) + c2
    return out


def _atom_name(a):
    return str(a).split("@")[0]


def scale_quantity(d, name, c, is_state):
    """The same model with the quantity `name` in units c times smaller (X' = c X): every coefficient on it is divided
    by c, and a state's own law is multiplied by c."""
    d = copy.deepcopy(d)
    sub = lambda coef, a: [(coef / c if _atom_name(a) == name else coef, a)]
    for s, spec in d["states"].items():
        spec["drift"] = _map_lin(spec.get("drift", {}), sub)
    for dn in list((d.get("definitions") or {})):
        d["definitions"][dn] = _map_lin(d["definitions"][dn], sub)
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            if "drift" in r:
                r["drift"] = _map_lin(r["drift"], sub)
        for key in ("loss", "terminal"):
            for t in a.get(key, []) or []:
                for x in t[1:]:
                    if _atom_name(x) == name:
                        t[0] = float(t[0]) / c
    if is_state:
        spec = d["states"][name]
        spec["drift"] = {k: v * c for k, v in spec["drift"].items()}
        spec["noise"] = {k: v * c for k, v in _map_lin(spec.get("noise", {}), lambda cc, a: [(cc, a)]).items()}
        if "initial" in spec:
            spec["initial"] = float(spec["initial"]) * c
        for p in (d["horizon"].get("past") or {}).get("initial", []) or []:
            if name in (p.get("loads") or {}):
                p["loads"][name] = float(p["loads"][name]) * c
    return d


def split_shock(d, w):
    """Replace shock w by (w + w') / sqrt 2 everywhere: the same law."""
    d = copy.deepcopy(d); w2 = w + "_split"; k = 1 / math.sqrt(2)
    d["shocks"] = d["shocks"] + [w2]
    f = lambda coef, a: [(coef * k, a), (coef * k, w2)] if a == w else [(coef, a)]
    for spec in d["states"].values():
        spec["noise"] = _map_lin(spec.get("noise", {}), f)
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            if "noise" in r:
                r["noise"] = _map_lin(r["noise"], f)
    return d


def rename_shocks(d):
    d = copy.deepcopy(d)
    ren = {w: f"z{i}_{w}" for i, w in enumerate(d["shocks"])}
    d["shocks"] = list(reversed([ren[w] for w in d["shocks"]]))
    f = lambda coef, a: [(coef, ren.get(a, a))]
    for spec in d["states"].values():
        spec["noise"] = _map_lin(spec.get("noise", {}), f)
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            if "noise" in r:
                r["noise"] = _map_lin(r["noise"], f)
    return d


def reverse_agents(d):
    d = copy.deepcopy(d)
    d["agents"] = dict(reversed(list(d["agents"].items())))
    return d


def irrelevant_state(d):
    d = copy.deepcopy(d)
    d["shocks"] = d["shocks"] + ["w_irr"]
    d["states"]["Z_irr"] = {"drift": {"Z_irr": -1.0}, "noise": {"w_irr": 0.7}}
    return d


def zero_term(d):
    d = copy.deepcopy(d)
    a = next(iter(d["agents"].values()))
    s = next(iter(d["states"]))
    a["loss"] = a["loss"] + [[0.0, s, a["controls"][0]]]
    return d


def rescale_time(d, alpha):
    """The model in a time unit alpha times longer (t = alpha s): drifts and the discount times alpha, noises times
    sqrt(alpha), the flow loss times alpha, lags, delays, T and the window over alpha.  Finite costs are unchanged; a
    stationary flow cost is multiplied by alpha."""
    d = copy.deepcopy(d); sq = math.sqrt(alpha)

    def lagfix(a):
        a = str(a)
        if "@" in a:
            n, l = a.split("@")
            return f"{n}@{float(l) / alpha:g}"
        return a

    f_drift = lambda coef, a: [(coef * alpha, lagfix(a))]
    f_noise = lambda coef, a: [(coef * sq, a)]
    f_atom = lambda coef, a: [(coef, lagfix(a))]
    for spec in d["states"].values():
        spec["drift"] = _map_lin(spec.get("drift", {}), f_drift)
        spec["noise"] = _map_lin(spec.get("noise", {}), f_noise)
    for dn in list((d.get("definitions") or {})):
        d["definitions"][dn] = _map_lin(d["definitions"][dn], f_atom)
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            r["drift"] = _map_lin(r.get("drift", {}), f_drift)
            r["noise"] = _map_lin(r.get("noise", {}), f_noise)
            if r.get("delay"):
                r["delay"] = float(r["delay"]) / alpha
        a["loss"] = [[float(t[0]) * alpha] + [lagfix(x) for x in t[1:]] for t in a["loss"]]
        if "constant" in a:
            a["constant"] = float(a["constant"]) * alpha
        if a.get("risk_aversion"):
            pass
    hz = d["horizon"]
    hz["discount"] = float(hz.get("discount", 0.0)) * alpha
    if "T" in hz:
        hz["T"] = float(hz["T"]) / alpha
    if "window" in hz:
        hz["window"] = float(hz["window"]) / alpha
    nm = d.get("numerics", {})
    for k in ("unit", "unit_range"):
        if k in nm:
            nm[k] = float(nm[k]) / alpha
    if "breakpoints" in nm:
        nm["breakpoints"] = [float(b) / alpha for b in nm["breakpoints"]]
    return d


def with_params(d):
    """The same model with its first few coefficients written as parameter expressions."""
    d = copy.deepcopy(d); params = {}
    i = 0
    for s, spec in d["states"].items():
        for a, c in list(spec["drift"].items()):
            if i < 3 and float(c) != 0:
                params[f"k{i}"] = float(c) / 2
                spec["drift"][a] = f"2 * k{i}"; i += 1
    a0 = next(iter(d["agents"].values()))
    for t in a0["loss"]:
        if i < 5 and float(t[0]) > 0:
            params[f"k{i}"] = float(t[0]) ** 2
            t[0] = f"sqrt(k{i})"; i += 1
    d["params"] = {**d.get("params", {}), **params}
    return d


def strip_ties(d):
    d = copy.deepcopy(d); d.pop("ties", None); return d


def with_theta(d, theta):
    d = copy.deepcopy(d)
    for a in d["agents"].values():
        if a.get("risk_aversion"):
            a["risk_aversion"] = theta
    return d


def has_means(d):
    return any(s.get("initial") or "const" in s.get("drift", {}) for s in d["states"].values()) or \
        any(len(t) == 2 for a in d["agents"].values() for t in a.get("loss", []) + (a.get("terminal") or []))


def lags_of(d):
    out = set()
    def scan(a):
        if "@" in str(a):
            out.add(float(str(a).split("@")[1]))
    for spec in d["states"].values():
        for a in spec.get("drift", {}):
            scan(a)
    for a in d["agents"].values():
        for r in a.get("signals", {}).values():
            for x in r.get("drift", {}):
                scan(x)
            if r.get("delay"):
                out.add(float(r["delay"]))
        for t in a["loss"]:
            for x in t[1:]:
                scan(x)
    return out


def random_walk_free(d):
    """Every state mean-reverts on its own (the stationary reference's truncation is then immaterial)."""
    for s, spec in d["states"].items():
        if float(spec.get("drift", {}).get(s, 0.0)) > -0.2:
            return False
    return True


# ------------------------------------------------------------------------------------------------ the checks
def self_consistency(case, base):
    out = []
    r = base.res
    costs = base.costs()
    finite = all(np.isfinite(v) for v in costs.values())
    if base.converged:
        out.append(verdict("finite costs", finite, fmt(costs), said_so=False, kind="sanity"))
    parts_ok, worst = True, 0.0
    for a, parts in r.cost_parts.items():
        s = sum(float(v) for v in parts.values())
        worst = max(worst, abs(s - costs[a]) / max(1.0, abs(costs[a])))
    out.append(verdict("cost parts sum", worst < 1e-9, f"worst {worst:.1e}", kind="sanity"))
    missing = [k for k, s in r.diagnostics.statuses.items() if s is Status.MISSING]
    out.append(verdict("no missing status", not missing, f"missing {missing}", kind="sanity"))
    psd = G.loss_is_psd(case.model)
    if psd and not has_means(case.model) and base.converged and not any(a.get("risk_aversion") for a in case.model["agents"].values()):
        neg = {k: v for k, v in costs.items() if v < -1e-9 * max(1.0, abs(v))}
        out.append(verdict("nonnegative cost of a nonnegative loss", not neg, fmt(costs), said_so=base.said_so, kind="sanity"))
    so = r.diagnostics.statuses.get("second_order")
    if psd and so is not None and so is not Status.NOT_APPLICABLE and base.converged:
        out.append(verdict("second order on a convex loss", so is not Status.FAILED,
                           f"second_order {so.value}: {getattr(r, 'second_order', None)}", said_so=False, kind="false alarm"))
    return out


def ref_check(case, base, levels=None, tol=5e-4):
    d = case.model
    kind = d["horizon"]["kind"]
    if kind == "stationary" and not random_walk_free(d):
        return [verdict("discrete reference", True, "a state without its own mean reversion", skip=True)]
    lg = lags_of(d)
    if levels is None:
        if kind == "stationary":
            L = float(d["horizon"]["window"])
            h0 = 0.25 if lg else 0.2                  # the coarsest step divides every lag (multiples of 0.25)
            n0 = int(round(L / h0))
            levels = (n0, 2 * n0, 4 * n0)
        else:
            T = float(d["horizon"]["T"])
            n0 = max(10, int(round(T / 0.05)))
            levels = (n0, 2 * n0, 4 * n0)
        # every lag on the coarsest grid
        h = (float(d["horizon"].get("window") or d["horizon"]["T"])) / levels[0]
        if any(abs(l / h - round(l / h)) > 1e-9 for l in lg):
            return [verdict("discrete reference", True, f"lags {lg} off the step {h}", skip=True)]
    t = time.time()
    key = "entropic" if any(a.get("risk_aversion") for a in d["agents"].values()) else "costs"
    try:
        try:
            out = solve_levels(d, levels, key=key)
        except ReferenceError:
            out = solve_levels(d, levels, key=key, beta=0.4)
    except ReferenceError as exc:
        return [verdict("discrete reference", True, f"reference failed: {exc}", skip=True)]
    lim, err = out["limit"], out["error"]
    pkg = base.costs() if key == "costs" else {k: float(v) for k, v in base.res.entropic_costs.items()}
    sc = cost_scale(pkg, lim)
    worst = max(abs(pkg[k] - lim[k]) - 4 * err[k] for k in pkg) / sc
    rawdiff = max(abs(pkg[k] - lim[k]) for k in pkg) / sc
    esterr = max(err.values()) / sc
    detail = (f"levels {levels}: rel diff {rawdiff:.2e}, reference error estimate {esterr:.1e}; pkg {fmt(pkg)} ref {fmt(lim)}"
              f" ({time.time() - t:.1f}s)")
    if esterr > 5e-3:
        return [verdict("discrete reference", True, "reference not converged: " + detail, skip=True)]
    if kind == "stationary":
        tol = window_tol(base, tol)
    return [verdict("discrete reference", worst <= tol, detail, said_so=gross_said_so(base, worst))]


def closed_form_check(case, base):
    d = case.model
    try:
        cf = lqg.solve(d)
    except ReferenceError as exc:
        return [verdict("closed form", True, f"n/a: {exc}", skip=True)]
    tol = 2e-5 if d["horizon"]["kind"] == "finite" else window_tol(base, 1e-4)
    return [compare_costs("closed form", base, cf, tol)]


def window_tol(solved, tol):
    """A stationary result's stated accuracy: the window check passes a kernel that still moves by up to 2% of its
    peak over the last tenth of the window, which leaves the cost up to about 0.03 x that tail from the untruncated
    one (lqg1_stationary-253: tail 1.5%, cost 4.9e-4 off; -414: tail 1.7%, 4.3e-3 off; documented, limits.md): 0.3 x the tail, at least tol."""
    tail = getattr(solved.res, "window_tail", None) if solved.ok else None
    return max(tol, 0.3 * float(tail)) if tail is not None and np.isfinite(tail) else tol


def cross_checks(case, base):
    d = case.model; out = []
    kind = d["horizon"]["kind"]
    nodes = int(d.get("numerics", {}).get("nodes", 16))
    # nodes + 4
    finer = copy.deepcopy(d); finer.setdefault("numerics", {})["nodes"] = nodes + 4
    s2 = Solved(finer)
    if s2.ok:
        if base.accepted and s2.accepted:
            out.append(compare_costs("nodes + 4", base, s2.costs(), 2e-5))
        else:
            out.append(verdict("nodes + 4", True, f"not both accepted ({base.accepted}, {s2.accepted})", skip=True))
    else:
        out.append(verdict("nodes + 4", False, "refused at nodes + 4: " + s2.brief(), said_so=False, kind="refusal"))
    # the iteration variable
    if not d.get("ties") and kind != "transition":
        alt = "maps" if (d.get("numerics", {}).get("variable", "actions") == "actions") else "actions"
        s3 = Solved(d, {"variable": alt})
        if s3.ok and base.converged and s3.converged:
            out.append(compare_costs(f"variable={alt}", base, s3.costs(), 1e-6, other_said=s3.said_so))
        elif s3.ok:
            out.append(verdict(f"variable={alt}", True, "not both converged", skip=True))
    # the order of the best responses (settings.best_responses): the other one reaches the same equilibrium
    instant = any(a.get("instant") for a in d["agents"].values())
    alt = "simultaneous" if instant else "sequential"
    d6 = copy.deepcopy(d); d6.setdefault("numerics", {})
    d6["numerics"]["settings"] = {**(d6["numerics"].get("settings") or {}), "best_responses": alt}
    s6 = Solved(d6)
    if s6.ok and base.converged and s6.converged:
        out.append(compare_costs(f"best_responses={alt}", base, s6.costs(), 1e-6, other_said=s6.said_so))
    elif s6.ok:
        out.append(verdict(f"best_responses={alt}", True, f"not both converged ({base.converged}, {s6.converged}): {s6.brief()}", skip=True))
    else:
        out.append(verdict(f"best_responses={alt}", False, s6.brief(), kind="refusal"))
    # a longer window
    if kind == "stationary":
        wider = copy.deepcopy(d); wider["horizon"]["window"] = 1.5 * float(d["horizon"]["window"])
        s4 = Solved(wider)
        if s4.ok and base.accepted and s4.accepted:
            out.append(compare_costs("window x 1.5", base, s4.costs(), window_tol(base, 2e-4)))
    # explicit breakpoints on a finite horizon without lags: two panels
    if kind == "finite" and not lags_of(d):
        T = float(d["horizon"]["T"])
        bp = copy.deepcopy(d); bp.setdefault("numerics", {})["breakpoints"] = [0.0, T / 2, T]
        s5 = Solved(bp)
        if s5.ok and base.accepted and s5.accepted:
            out.append(compare_costs("breakpoints [0, T/2, T]", base, s5.costs(), 2e-5))
        elif not s5.ok:
            out.append(verdict("breakpoints [0, T/2, T]", False, s5.brief(), kind="refusal"))
    return out


def invariance_checks(case, base, which=None):
    d = case.model; out = []
    kind = d["horizon"]["kind"]
    tol = 1e-6
    tests = []
    if len(d["agents"]) > 1 and not d.get("ties"):
        tests.append(("agent order", reverse_agents(d), 1.0))
    tests.append(("shock relabelling", rename_shocks(d), 1.0))
    s0 = next(iter(d["states"])); w0 = next(iter(d["states"][s0]["noise"]))
    tied = bool(d.get("ties"))                  # the one-agent transformations would break a tie's symmetry
    if not tied:
        tests.append(("shock split", split_shock(d, w0), 1.0))
    tests.append(("irrelevant state", irrelevant_state(d), 1.0))
    if not tied:
        tests.append(("zero-weight term", zero_term(d), 1.0))
        tests.append(("state units x3", scale_quantity(d, s0, 3.0, True), 1.0))
    a0 = next(iter(d["agents"].values()))
    if not tied:
        tests.append(("control units x0.5", scale_quantity(d, a0["controls"][0], 0.5, False), 1.0))
    if not any(a.get("risk_aversion") for a in d["agents"].values()) or kind != "stationary":
        alpha = 2.0
        # a stationary flow cost scales by alpha; risk aversion in time units: theta is per unit of cost, unchanged
        tests.append(("time unit x2", rescale_time(d, alpha), alpha if kind == "stationary" else 1.0))
    if not tied:
        tests.append(("parameter expressions", with_params(d), 1.0))
    for name, d2, factor in tests:
        if which and name not in which:
            continue
        s = Solved(d2)
        if not s.ok:
            out.append(verdict(name, False, "the transformed model was refused: " + s.brief(), kind="refusal"))
            continue
        if not (base.converged and s.converged):
            out.append(verdict(name, True, f"not both converged ({base.converged}, {s.converged})", skip=True))
            continue
        c2 = {k: v / factor for k, v in s.costs().items()}
        v = compare_costs(name, base, c2, tol, other_said=s.said_so)
        # the verdict flips: an invariance broken by more than the tolerance on two converged solves is a finding
        # unless either solve said it was not accurate
        out.append(v)
        if base.accepted != s.accepted and v["verdict"] == "ok":
            # the costs agree but a diagnostic's verdict flips: its measure is not invariant to the reparametrisation
            # (recorded as a note; a finding only if the costs disagree)
            v2 = verdict(name + " (acceptance)", False, f"acceptance differs: base {base.accepted} ({base.failed_checks()}), "
                         f"transformed {s.accepted} ({s.failed_checks()})", said_so=True, kind="consistency")
            v2["verdict"] = "note"
            out.append(v2)
    return out


def theta_zero_check(case, base):
    d = case.model
    if not any(a.get("risk_aversion") for a in d["agents"].values()):
        return []
    rn = Solved(with_theta(d, 0.0))
    tiny = Solved(with_theta(d, 1e-7))
    out = []
    if rn.ok and tiny.ok and rn.converged and tiny.converged:
        out.append(compare_costs("theta 1e-7 vs 0", tiny, rn.costs(), 1e-5, other_said=rn.said_so))
        ent = {k: float(v) for k, v in tiny.res.entropic_costs.items()}
        exp = rn.costs()
        if d["horizon"]["kind"] == "stationary":
            # the stationary entropic cost is the date-0 self's, of the discounted continuation: E C_0 = flow / rho
            rho = float(d["horizon"]["discount"])
            exp = {k: (v / rho if d["agents"][k].get("risk_aversion") else v) for k, v in exp.items()}
        sc = cost_scale(ent, exp)
        w = max(abs(ent[k] - exp[k]) for k in ent) / sc
        out.append(verdict("entropic at theta 1e-7 vs expected", w < 1e-5, f"{w:.1e}", said_so=tiny.said_so))
    else:
        out.append(verdict("theta 1e-7 vs 0", False, f"rn: {rn.brief()} / tiny: {tiny.brief()}",
                           said_so=(rn.ok and tiny.ok), kind="refusal"))
    return out


def ties_check(case, base):
    d = case.model
    if not d.get("ties"):
        return []
    un = Solved(strip_ties(d))
    if not (un.ok and un.converged and base.converged):
        return [verdict("tied vs untied", True, f"untied: {un.brief()}", skip=True)]
    return [compare_costs("tied vs untied", base, un.costs(), 1e-6, other_said=un.said_so)]


def transition_check(case, base):
    """The stationary maps as the past and continuation of a strip [0, L]: one best response from them reproduces them."""
    import helpers
    d = case.model
    if not base.converged:
        return []
    m = ns.Model.from_dict(copy.deepcopy(d))
    L = float(d["horizon"]["window"])
    try:
        dev, _ = helpers.one_shot_from_the_stationary_maps(m, base.res, L, int(d["numerics"]["nodes"]))
    except Exception as exc:                    # noqa: BLE001
        return [verdict("transition identity", False, f"{type(exc).__name__}: {str(exc)[:200]}", kind="refusal",
                        said_so=isinstance(exc, (NotImplementedError,)))]
    worst = max(float(np.max(v)) for v in dev.values())
    return [verdict("transition identity", worst < 1e-3, f"one-shot deviation {worst:.2e}", said_so=base.said_so)]


def optimality_check(case, base, trials=4):
    """The converged best response against random feasible perturbations of its action (test_properties' check,
    generalised): no perturbation lowers the agent's cost."""
    if not base.converged or base.d["horizon"]["kind"] == "transition" or case.model.get("ties"):
        return []
    if any(a.get("risk_aversion") or a.get("myopic") or a.get("monitors") for a in case.model["agents"].values()):
        return []
    from noisestate.stationary import StationarySolver
    from noisestate.finite_spectral import SpectralFiniteSolver
    from noisestate.finite_free import RowOps, RespOps
    r = base.res; out = []
    try:
        S = r._make_solver() if hasattr(r, "_make_solver") else None
    except Exception:                          # noqa: BLE001
        S = None
    if S is None:
        return []
    rng = np.random.default_rng(case.seed)
    worst = 0.0
    try:
        for a in S.model.agents:
            c = S.c; nW = c.nW
            g, o = S.best_response(a, r.maps)
            Zp = c.closed_loop(r.maps, excluded=a.name, impulse_controls=a.controls); Zpass, R = Zp[:, :nW], Zp[:, nW:]
            if isinstance(S, StationarySolver):
                Resp = S._response_operators(a, R)[0]
                cost = lambda cc: S.expected_cost(a, Zpass + Resp @ cc)
                ytil, yinst = S._passive_rows(a, Zpass); Gk = S._row_operator(a, ytil, yinst)
                pert = lambda: np.stack([Gk[k] @ rng.standard_normal(Gk.shape[2]) for k in range(nW)], axis=1)
            elif isinstance(S, SpectralFiniteSolver):
                resp = RespOps(S, a, R); ytil, yinst = S._passive_rows(a, Zpass); rowops = RowOps(S, a, ytil, yinst)
                cost = lambda cc: S.expected_cost(a, Zpass + resp.apply(0, cc).reshape(Zpass.shape))
                pert = lambda: rowops.apply(rng.standard_normal(rowops.nR * rowops.Nm))
            else:
                return []
            c0 = o["action"][0]; L0 = cost(c0)
            for _ in range(trials):
                dc = pert(); dc = dc * (0.02 / max(1e-300, np.abs(dc).max()))
                worst = max(worst, (L0 - cost(c0 + dc)) / max(1.0, abs(L0)), (L0 - cost(c0 - dc)) / max(1.0, abs(L0)))
    except Exception as exc:                   # noqa: BLE001
        return [verdict("optimality vs perturbations", True, f"n/a: {type(exc).__name__}: {str(exc)[:120]}", skip=True)]
    return [verdict("optimality vs perturbations", worst <= 1e-8, f"largest improvement {worst:.1e}", said_so=base.said_so)]


def risk_breakdown_check(case, base):
    """A RiskBreakdown on a small model: does the brute-force entropic reference find the equilibrium at that theta?"""
    d = case.model
    if d["horizon"]["kind"] != "finite" or (d.get("numerics", {}).get("settings") or {}).get("risk_planning") == "consistent":
        return [verdict("breakdown vs reference", True, "not the precommitment finite case", skip=True)]
    T = float(d["horizon"]["T"])
    n0 = max(10, int(round(T / 0.05)))
    try:
        out = solve_levels(d, (n0, 2 * n0), key="entropic", beta=0.5)
    except ReferenceError as exc:
        return [verdict("breakdown vs reference", True, f"the reference fails too: {exc}", skip=True)]
    if not all(np.isfinite(v) for lv in out["values"] for v in lv.values()):
        return [verdict("breakdown vs reference", True, f"the reference finds no finite entropic cost either: {out['values']}", skip=True)]
    return [verdict("breakdown vs reference", False, f"the reference solves it: {fmt(out['limit'])} "
                    f"(levels {out['levels']}, values {out['values']}); package: {base.brief()}", said_so=False, kind="refusal")]


# ------------------------------------------------------------------------------------------------ one case
def run_case(case, depth="full"):
    """All the checks that apply to `case`; returns the record."""
    t0 = time.time()
    d = case.model
    rec = {"id": case.id, "family": case.family, "seed": case.seed, "tags": case.tags, "checks": []}
    if case.family == "invalid":
        s = Solved(d)
        ok = (not s.ok) and isinstance(s.exc, (ValueError, TypeError))
        detail = s.brief() if s.ok else f"{type(s.exc).__name__}: {str(s.exc)[:160]}"
        rec["checks"].append(verdict("invalid model refused", ok, detail, kind="validation"))
        rec["seconds"] = time.time() - t0
        return rec
    base = Solved(d)
    rec["base"] = base.brief(); rec["base_seconds"] = base.seconds
    if not base.ok:
        exc = base.exc
        if isinstance(exc, ns.RiskBreakdown):
            rec["checks"] += risk_breakdown_check(case, base)
        else:
            rec["checks"].append(verdict("solve", False, f"{type(exc).__name__}: {str(exc)[:300]}",
                                         kind="refusal" if isinstance(exc, (ValueError, NotImplementedError)) else "crash"))
            rec["traceback"] = getattr(base, "tb", "")
        rec["seconds"] = time.time() - t0
        return rec
    rec["costs"] = base.costs(); rec["accepted"] = base.accepted; rec["failed"] = base.failed_checks()
    rec["checks"].append(verdict("converged", base.converged, base.brief(), said_so=True, kind="convergence"))
    rec["checks"] += self_consistency(case, base)
    fam = case.family
    if fam == "cara_stationary":
        depth = "smoke"                  # every stationary risk-averse solve is 10-300 s: the base, theta -> 0, a few invariances
    if fam.startswith("lqg1"):
        rec["checks"] += closed_form_check(case, base)
    if fam in ("lqg1_finite", "game_finite", "delay_finite", "means_finite", "myopic", "prior", "cara_finite",
               "lqg1_stationary", "game_stationary", "delay_stationary", "ties"):
        if not (fam == "cara_finite" and "consistent" in case.tags):
            rec["checks"] += ref_check(case, base)
    if depth == "full":
        rec["checks"] += cross_checks(case, base)
        rec["checks"] += invariance_checks(case, base)
        rec["checks"] += optimality_check(case, base)
    else:
        rec["checks"] += invariance_checks(case, base, which={"agent order", "shock split", "state units x3", "time unit x2"})
    rec["checks"] += theta_zero_check(case, base)
    rec["checks"] += ties_check(case, base)
    if fam == "transition":
        rec["checks"] += transition_check(case, base)
    if fam in ("monitor", "ch6_market") and base.converged:
        rec["checks"] += monitor_checks(case, base)
    if fam == "ch6_market":
        rec["checks"] += ch6_checks(case, base)
    rec["seconds"] = time.time() - t0
    return rec


def ch6_checks(case, base):
    """At gamma = 0 the strategic market maker is competitive: the market is the myopic Kyle market (Chapter 6), so
    the trader's cost is that market's."""
    m = case.meta
    if m.get("gamma") != 0.0:
        return []
    comp = Solved(G.competitive_market(m["eps"], m["rho"], m["kap"], m["sz"], m["kind"], case.model["numerics"]["nodes"]))
    if not (comp.converged and base.converged):
        return [verdict("gamma 0 = competitive", True, f"competitive: {comp.brief()}", skip=True)]
    sc = cost_scale({"t": base.costs()["trader"]})
    diff = abs(base.costs()["trader"] - comp.costs()["trader"]) / sc
    return [verdict("gamma 0 = competitive", diff < 1e-5, f"trader {base.costs()['trader']:.8g} vs competitive "
                    f"{comp.costs()['trader']:.8g} ({diff:.1e})", said_so=base.said_so or comp.said_so)]


def monitor_checks(case, base):
    """Chapter 6: every privy responder's first-order condition in each origin's deviation world, by the result's own
    quadrature (independent of the engine's operator), and the all-naive corner is unchanged by a relation among agents
    who cannot see each other."""
    out = []
    d = case.model
    for a, ag in d["agents"].items():
        for origin in ag.get("monitors", []):
            try:
                fr = base.res.foc_residual(a, seed=origin)
                relv = float(fr.relative)
                if np.isnan(relv):
                    continue
                out.append(verdict(f"foc_residual {a}<-{origin}", relv < 1e-4, f"relative {relv:.1e}", said_so=base.said_so))
            except NotImplementedError:
                pass
            except Exception as exc:               # noqa: BLE001
                out.append(verdict(f"foc_residual {a}<-{origin}", False, f"{type(exc).__name__}: {str(exc)[:160]}", kind="crash"))
    return out


def summarize(records):
    """Per family: models, passes, findings, flagged, skipped checks."""
    fams = {}
    for r in records:
        f = fams.setdefault(r["family"], {"models": 0, "pass": 0, "finding": 0, "flagged": 0, "checks": 0, "skip": 0})
        f["models"] += 1
        v = [c["verdict"] for c in r["checks"]]
        f["checks"] += len(v); f["skip"] += v.count("skip")
        if "FINDING" in v:
            f["finding"] += 1
        elif "flagged" in v:
            f["flagged"] += 1
        else:
            f["pass"] += 1
    return fams
