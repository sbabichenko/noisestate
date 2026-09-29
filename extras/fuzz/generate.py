"""Seeded random models for the fuzz campaign (extras/fuzz): small but valid linear-quadratic-Gaussian games in the
grammar's dictionary form, with numeric coefficients (so the independent references can read them), printable as YAML.

`generate(seed, family)` returns a Case: the model dict, its family, the feature tags it exercises and the oracles that
apply.  Every loss is positive semidefinite in its atoms with a strictly positive weight on the agent's own current
controls (so every best response is convex: a failed second-order check is a false alarm), every stationary model is
mean-reverting without control (so a stationary equilibrium exists), every lag and delay is a multiple of 0.25 and of
the horizon, and the finite horizons are 0.5 to 2 time units.  The families:

  lqg1_finite, lqg1_stationary   one agent, 1-2 states, 1-2 controls, 1-2 noisy rows (correlated noise sometimes),
                                 cross terms, discount; finite: terminal loss, initial state, targets, constant drift
  game_finite, game_stationary   2-3 agents on 1-2 shared states, private rows, rows on another agent's control
  delay_finite, delay_stationary control lags in the drift, delayed rows, lagged atoms in the losses
  means_finite                   targets, constant drifts and initial states in a game
  ties                           symmetric 2-3 player games, solved tied, untied and permuted
  myopic                         a Kyle-type market with a myopic competitive market maker (mean-reverting value)
  prior                          a finite game started from initial shocks seen by some rows (continuation: end)
  cara_finite, cara_stationary   risk-averse agents well below the breakdown (precommitment and consistent planning;
                                 stationary: a discount, no lags or means)
  monitor                        Chapter 6 monitoring relations (stationary and finite)
  transition                     a stationary model as its own past and continuation
  ch6_market                     Chapter 6's market: a strategic market maker whose quote the trader sees at once
                                 (`instant`), transparent (the trader monitors it) or not, an inventory cost gamma;
                                 at gamma = 0 it is the competitive (myopic) Kyle market
  invalid                        a valid model with one defect injected: it must be refused
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import yaml

FAMILIES = ("lqg1_finite", "lqg1_stationary", "game_finite", "game_stationary", "delay_finite", "delay_stationary",
            "means_finite", "ties", "myopic", "prior", "cara_finite", "cara_stationary", "monitor", "transition", "invalid",
            "ch6_market")


@dataclass
class Case:
    seed: int
    family: str
    model: dict
    tags: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def id(self):
        return f"{self.family}-{self.seed}"

    def yaml(self):
        return yaml.safe_dump(self.model, sort_keys=False, default_flow_style=None, width=120)


def _r(x, sig=4):
    """Round to `sig` significant digits (short repros)."""
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def _psd_loss(rng, atoms, own, rdiag=(0.1, 1.0), cross=0.6, margin=0.05):
    """Loss terms [c, a, b] of a positive semidefinite form on `atoms`, strictly positive on the `own` atoms.  The
    form is L L' (rank up to len(atoms)) + a diagonal on the own atoms, rounded with a margin on the diagonal so the
    rounded form stays PSD."""
    m = len(atoms)
    L = rng.standard_normal((m, m)) * rng.uniform(0.2, 1.0)
    L[:, rng.random(m) < 0.3] = 0.0
    Q = L @ L.T * cross
    for i, a in enumerate(atoms):
        if a in own:
            Q[i, i] += rng.uniform(*rdiag)
        Q[i, i] += margin
    terms = []
    for i in range(m):
        if Q[i, i] > 0:
            terms.append([_r(Q[i, i] + 1e-3), atoms[i], atoms[i]])
        for j in range(i + 1, m):
            if abs(Q[i, j]) > 1e-3:
                terms.append([_r(2 * Q[i, j] * 0.97), atoms[i], atoms[j]])
    return terms


def _stable(A):
    return np.max(np.linalg.eigvals(A).real) < -0.15


def build(rng, n_agents, n_states, kind, *, lags=False, delays=False, means=False, terminal=False, rows_on_controls=True,
          control_lag_only=False, two_controls=True, correlated=True, lagged_loss=False, discount=None, T=None,
          window=None, nodes=None, signal_scale=1.0):
    """A random game.  `kind` is 'finite' or 'stationary'."""
    states = [f"X{i}" for i in range(n_states)]
    agents = [f"a{i}" for i in range(n_agents)]
    ctrl = {}
    for i, a in enumerate(agents):
        k = 2 if (two_controls and rng.random() < 0.25) else 1
        ctrl[a] = [f"D{i}" if k == 1 else f"D{i}{'ab'[j]}" for j in range(k)]
    tau = float(rng.choice([0.25, 0.5])) if (lags or delays) else None
    if tau is not None and T is not None and tau >= T / 2:
        tau = 0.25                                  # a lag or delay must lie below the horizon (guards.md)
    shocks = []
    d = {"name": "fuzz", "shocks": shocks, "states": {}, "agents": {}}
    # the uncontrolled dynamics: mean-reverting (stationary) or anything mild (finite)
    for _ in range(50):
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            A[i, i] = -rng.uniform(0.3, 1.5) if kind == "stationary" else rng.uniform(-1.2, 0.4)
            for j in range(n_states):
                if i != j and rng.random() < 0.35:
                    A[i, j] = rng.uniform(-0.6, 0.6)
        if kind != "stationary" or _stable(A):
            break
    for i, s in enumerate(states):
        drift = {}
        for j, s2 in enumerate(states):
            if A[i, j]:
                drift[s2] = _r(A[i, j])
        noise = {f"w{s}": _r(rng.uniform(0.5, 1.5))}
        shocks.append(f"w{s}")
        d["states"][s] = {"drift": drift, "noise": noise}
    # every control drives a state (lagged sometimes)
    for a in agents:
        for u in ctrl[a]:
            s = states[rng.integers(n_states)]
            atom = f"{u}@{tau:g}" if (lags and rng.random() < 0.5) else u
            d["states"][s]["drift"][atom] = _r(rng.choice([-1, 1]) * rng.uniform(0.5, 1.5))
            if n_states > 1 and rng.random() < 0.3:
                s2 = [x for x in states if x != s][0]
                d["states"][s2]["drift"][u] = _r(rng.uniform(-0.8, 0.8))
    for a_i, a in enumerate(agents):
        nr = 1 if rng.random() < 0.6 else 2
        rows = {}
        for r in range(nr):
            s = states[rng.integers(n_states)]
            drift = {s: _r(rng.choice([-1, 1]) * rng.uniform(0.5, 2.5) * signal_scale)}
            if n_states > 1 and rng.random() < 0.3:
                s2 = [x for x in states if x != s][0]
                drift[s2] = _r(rng.uniform(-1, 1))
            others = [u for b in agents if b != a for u in ctrl[b]]
            if rows_on_controls and others and rng.random() < 0.25:
                drift[others[rng.integers(len(others))]] = _r(rng.uniform(0.3, 1.2))
            ch = f"w{a}{r}"; shocks.append(ch)
            noise = {ch: _r(rng.uniform(0.5, 1.5))}
            if correlated and rng.random() < 0.15:
                noise[f"w{states[0]}"] = _r(rng.uniform(-0.5, 0.5))
            row = {"drift": drift, "noise": noise}
            if delays and rng.random() < 0.5:
                row["delay"] = tau
            rows[f"y{r}"] = row
        # the loss atoms: own controls, 1-2 states, maybe another agent's control, maybe a lagged state
        own = list(ctrl[a])
        atoms = own + list(rng.choice(states, size=min(n_states, 1 + int(rng.random() < 0.5)), replace=False))
        others = [u for b in agents if b != a for u in ctrl[b]]
        if others and rng.random() < 0.3:
            atoms.append(others[rng.integers(len(others))])
        if lagged_loss and tau is not None and rng.random() < 0.6:
            atoms.append(f"{states[rng.integers(n_states)]}@{tau:g}")
        loss = _psd_loss(rng, atoms, own)
        if means:
            for x in rng.choice(atoms, size=min(2, len(atoms)), replace=False):
                if "@" not in x:
                    loss.append([_r(rng.uniform(-1.5, 1.5)), str(x)])
        ag = {"controls": own, "signals": rows, "loss": loss}
        if terminal and kind == "finite" and rng.random() < 0.6:
            s = states[rng.integers(n_states)]
            ag["terminal"] = [[_r(rng.uniform(0.2, 2.0)), s, s]]
            if means and rng.random() < 0.5:
                ag["terminal"].append([_r(rng.uniform(-1, 1)), s])
        d["agents"][a] = ag
    if means and kind == "finite":
        for s in states:
            if rng.random() < 0.6:
                d["states"][s]["initial"] = _r(rng.uniform(-1.5, 1.5))
            if rng.random() < 0.4:
                d["states"][s]["drift"]["const"] = _r(rng.uniform(-0.8, 0.8))
    rho = discount if discount is not None else (0.0 if rng.random() < 0.4 else _r(rng.uniform(0.1, 1.0)))
    if kind == "finite":
        T = T if T is not None else float(rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]))
        d["horizon"] = {"kind": "finite", "T": T, "discount": rho}
        d["numerics"] = {"nodes": int(nodes or rng.integers(10, 15))}
    else:
        window = window if window is not None else float(rng.choice([6.0, 8.0, 10.0]))
        d["horizon"] = {"kind": "stationary", "window": window, "discount": rho}
        d["numerics"] = {"nodes": int(nodes or rng.integers(16, 21))}
    return d


# --------------------------------------------------------------------------------------------- families
def _lqg1(rng, kind):
    d = build(rng, 1, int(rng.integers(1, 3)), kind, terminal=True, means=(kind == "finite" and rng.random() < 0.5),
              rows_on_controls=False)
    return d, ["one_agent", kind]


def _game(rng, kind):
    d = build(rng, int(rng.integers(2, 4)), int(rng.integers(1, 3)), kind, terminal=rng.random() < 0.3)
    return d, ["game", kind]


def _delay(rng, kind):
    # a finite horizon cut by lags is costly (the triangle's pieces grow with (T / lag)^2): short horizons, 8 nodes
    fin = kind == "finite"
    d = build(rng, int(rng.integers(1, 3)), int(rng.integers(1, 3)), kind, lags=True, delays=True, lagged_loss=True,
              T=float(rng.choice([0.5, 1.0])) if fin else None, nodes=8 if fin else None)
    return d, ["delay", kind]


def _means(rng):
    d = build(rng, int(rng.integers(2, 4)), int(rng.integers(1, 3)), "finite", means=True, terminal=True)
    return d, ["means", "finite"]


def _ties(rng):
    """A symmetric game: n identical players (each with its own state, or all on one state), tied."""
    n = int(rng.integers(2, 4)); kind = "finite" if rng.random() < 0.5 else "stationary"
    shared = rng.random() < 0.5
    a = _r(rng.uniform(0.3, 1.2)); b = _r(rng.uniform(0.5, 1.5)); sig = _r(rng.uniform(0.5, 1.5))
    h = _r(rng.uniform(0.8, 2.5)); s = _r(rng.uniform(0.5, 1.5)); q = _r(rng.uniform(0.3, 1.5)); r = _r(rng.uniform(0.1, 1.0))
    cq = _r(rng.uniform(-0.9, 0.9))
    d = {"name": "fuzz", "shocks": [], "states": {}, "agents": {}}
    if shared:
        d["shocks"].append("wX"); d["states"]["X"] = {"drift": {"X": -a}, "noise": {"wX": sig}}
    for i in range(n):
        st = "X" if shared else f"X{i}"
        if not shared:
            d["shocks"].append(f"wX{i}"); d["states"][st] = {"drift": {st: -a}, "noise": {f"wX{i}": sig}}
        d["states"][st]["drift"][f"D{i}"] = b
        d["shocks"].append(f"w{i}")
        loss = [[q, st, st], [r, f"D{i}", f"D{i}"]]
        if not shared:
            # a private state, and the next player's control in the loss (PSD: |cq| < 2 sqrt(r 0.2))
            nxt = f"D{(i + 1) % n}"
            loss += [[0.2, nxt, nxt], [_r(cq * 2 * np.sqrt(0.2 * r)), f"D{i}", nxt]]
        d["agents"][f"p{i}"] = {"controls": [f"D{i}"], "signals": {"y": {"drift": {st: h}, "noise": {f"w{i}": s}}}, "loss": loss}
    if not shared:
        # each player also sees the next player's control through noise (a cycle of private information)
        for i in range(n):
            d["shocks"].append(f"v{i}")
            d["agents"][f"p{i}"]["signals"]["z"] = {"drift": {f"D{(i + 1) % n}": 1.0}, "noise": {f"v{i}": s}}
    d["ties"] = [[f"p{i}" for i in range(n)]]
    if kind == "finite":
        d["horizon"] = {"kind": "finite", "T": float(rng.choice([0.5, 1.0])), "discount": 0.0}
        d["numerics"] = {"nodes": 12}
    else:
        d["horizon"] = {"kind": "stationary", "window": 8.0, "discount": float(rng.choice([0.0, 0.3]))}
        d["numerics"] = {"nodes": 16}
    return d, ["ties", kind, "shared" if shared else "cycle"]


def _myopic(rng):
    """Kyle-type: a mean-reverting value V, insiders with a noisy view of V - P (or of V), the flow D + noise seen by a
    myopic competitive market maker (P = E[V | flow])."""
    kind = "finite" if rng.random() < 0.5 else "stationary"
    nt = int(rng.integers(1, 3))
    kap = _r(rng.uniform(0.3, 1.0)); sv = _r(rng.uniform(0.5, 1.5)); sz = _r(rng.uniform(0.5, 1.5))
    d = {"name": "fuzz", "shocks": ["wV", "wZ"], "states": {"V": {"drift": {"V": -kap}, "noise": {"wV": sv}}}, "agents": {}}
    flow = {f"D{j}": 1.0 for j in range(nt)}
    d["agents"]["mm"] = {"controls": ["P"], "signals": {"flow": {"drift": flow, "noise": {"wZ": sz}}},
                         "loss": [[1.0, "P", "P"], [-2.0, "P", "V"]], "myopic": True}
    for j in range(nt):
        g = _r(rng.uniform(0.7, 2.0)); eps = _r(rng.uniform(0.15, 0.8))
        d["shocks"].append(f"w{j}")
        row = {"drift": {"V": g, "P": -g}, "noise": {f"w{j}": 1.0}} if rng.random() < 0.5 else {"drift": {"V": g}, "noise": {f"w{j}": 1.0}}
        d["agents"][f"t{j}"] = {"controls": [f"D{j}"], "signals": {"y": row, "flow": {"drift": {}, "noise": {"wZ": sz}}},
                               "loss": [[-1.0, f"D{j}", "V"], [1.0, f"D{j}", "P"], [eps, f"D{j}", f"D{j}"]]}
    rho = _r(rng.uniform(0.2, 0.8))
    if kind == "finite":
        d["horizon"] = {"kind": "finite", "T": float(rng.choice([1.0, 2.0])), "discount": float(rng.choice([0.0, rho]))}
        d["numerics"] = {"nodes": 12}
    else:
        d["horizon"] = {"kind": "stationary", "window": 10.0, "discount": rho}
        d["numerics"] = {"nodes": 20}
    return d, ["myopic", kind, f"traders{nt}"]


def _prior(rng):
    d = build(rng, int(rng.integers(1, 3)), int(rng.integers(1, 3)), "finite", rows_on_controls=False)
    loads = {s: _r(rng.uniform(0.5, 1.5)) for s in list(d["states"])[:1 + int(rng.random() < 0.5)]}
    rows = {}
    for a, ag in d["agents"].items():
        if rng.random() < 0.6:
            rows[f"{a}.{next(iter(ag['signals']))}"] = _r(rng.uniform(0.5, 1.5))
    d["horizon"] = {"kind": "transition", "T": d["horizon"]["T"], "discount": d["horizon"]["discount"],
                    "past": {"initial": [{"name": "xi", "loads": loads, "rows": rows}]}, "continuation": "end"}
    return d, ["prior", "finite"]


def _cara(rng, kind):
    if kind == "finite":
        d = build(rng, int(rng.integers(1, 3)), 1, "finite", terminal=rng.random() < 0.4, means=rng.random() < 0.3,
                  rows_on_controls=False, T=float(rng.choice([0.5, 1.0])), nodes=10)
        tags = ["cara", "finite"]
        if rng.random() < 0.3:
            d["numerics"]["settings"] = {"risk_planning": "consistent"}
            # consistent planning refuses means
            _strip_means(d); tags.append("consistent")
    else:
        d = build(rng, 1, 1, "stationary", rows_on_controls=False, discount=_r(rng.uniform(0.3, 1.0)), window=8.0,
                  nodes=16)
        tags = ["cara", "stationary"]
    for a in d["agents"].values():
        if rng.random() < 0.7 or a is list(d["agents"].values())[0]:
            a["risk_aversion"] = _r(rng.uniform(0.05, 0.6))
    return d, tags


def _strip_means(d):
    for s in d["states"].values():
        s.pop("initial", None); s["drift"].pop("const", None)
    for a in d["agents"].values():
        a["loss"] = [t for t in a["loss"] if len(t) == 3]
        if "terminal" in a:
            a["terminal"] = [t for t in a["terminal"] if len(t) == 3]


def _monitor(rng):
    kind = "stationary" if rng.random() < 0.6 else "finite"
    n = int(rng.integers(2, 4))
    d = build(rng, n, int(rng.integers(1, 3)), kind, rows_on_controls=True, two_controls=False, correlated=False,
              window=6.0, nodes=14 if kind == "stationary" else 10, T=1.0)
    names = list(d["agents"])
    # a transitive relation: a random subset of ordered pairs, closed
    rel = {a: set() for a in names}
    for a in names:
        for b in names:
            if a != b and rng.random() < 0.4:
                rel[a].add(b)
    changed = True
    while changed:
        changed = False
        for a in names:
            for b in list(rel[a]):
                for c in rel[b]:
                    if c != a and c not in rel[a]:
                        rel[a].add(c); changed = True
    if not any(rel.values()):
        rel[names[1]].add(names[0])
    for a in names:
        if rel[a]:
            d["agents"][a]["monitors"] = sorted(rel[a])
    tags = ["monitor", kind]
    if rng.random() < 0.4:
        # an instant observation: a later agent sees an earlier one's control at once and has it in its loss
        i, j = 0, int(rng.integers(1, n))
        u = d["agents"][names[i]]["controls"][0]; aj = d["agents"][names[j]]
        own = aj["controls"][0]
        aj["loss"] = [t for t in aj["loss"] if u not in t[1:]]
        cc = _r(rng.uniform(-0.8, 0.8))
        aj["loss"] += [[0.3, u, u], [cc, own, u]]          # own has weight >= 0.15 (rdiag + margin): PSD
        aj["instant"] = [u]
        tags.append("instant")
        if rng.random() < 0.7:
            # the observer privy to the observed agent (Chapter 6's transparent market), the relation closed again
            rel[names[j]].add(names[i]); rel[names[j]] |= rel[names[i]] - {names[j]}
            for a in names:
                if names[j] in rel[a]:
                    rel[a] |= rel[names[j]] - {a}
            for a in names:
                if rel[a]:
                    d["agents"][a]["monitors"] = sorted(rel[a])
    return d, tags


def ch6_market(gamma, eps, rho, kap, sz, transparent, kind, nodes, T=1.0):
    """Chapter 6's market (tests/test_monitoring.market) with a mean-reverting value (kap >= 0)."""
    tr = {"controls": ["D"], "signals": {"y": {"drift": {"V": 1.0, "P": -1.0}, "noise": {"wY": 1.0}},
                                         "flow": {"drift": {}, "noise": {"wZ": sz}}},
          "loss": [[-1.0, "V", "D"], [1.0, "P", "D"], [eps, "D", "D"]], "instant": ["P"]}
    if transparent:
        tr["monitors"] = ["mm"]
    mm_loss = [[1.0, "V", "D"], [-1.0, "P", "D"]] + ([[gamma, "Q", "Q"]] if gamma else [])
    d = {"name": "fuzz", "shocks": ["wV", "wZ", "wY"],
         "states": {"V": {"drift": ({"V": -kap} if kap else {}), "noise": {"wV": 1.0}},
                    "Q": {"drift": {"D": -1.0}, "noise": {"wZ": -sz}}},
         "agents": {"mm": {"controls": ["P"], "signals": {"flow": {"drift": {"D": 1.0}, "noise": {"wZ": sz}}}, "loss": mm_loss},
                    "trader": tr}}
    if kind == "stationary":
        d["horizon"] = {"kind": "stationary", "window": 8.0, "discount": rho}
    else:
        d["horizon"] = {"kind": "finite", "T": T, "discount": rho}
    d["numerics"] = {"nodes": nodes}
    return d


def competitive_market(eps, rho, kap, sz, kind, nodes, T=1.0):
    """The gamma = 0 limit of ch6_market: Chapter 4's market with a myopic competitive market maker."""
    d = {"name": "fuzz", "shocks": ["wV", "wZ", "wY"],
         "states": {"V": {"drift": ({"V": -kap} if kap else {}), "noise": {"wV": 1.0}}},
         "agents": {"mm": {"controls": ["P"], "signals": {"flow": {"drift": {"D": 1.0}, "noise": {"wZ": sz}}},
                           "loss": [[1.0, "P", "P"], [-2.0, "P", "V"]], "myopic": True},
                    "trader": {"controls": ["D"], "signals": {"y": {"drift": {"V": 1.0, "P": -1.0}, "noise": {"wY": 1.0}},
                                                              "flow": {"drift": {}, "noise": {"wZ": sz}}},
                               "loss": [[-1.0, "V", "D"], [1.0, "P", "D"], [eps, "D", "D"]]}}}
    if kind == "stationary":
        d["horizon"] = {"kind": "stationary", "window": 8.0, "discount": rho}
    else:
        d["horizon"] = {"kind": "finite", "T": T, "discount": rho}
    d["numerics"] = {"nodes": nodes}
    return d


def _ch6(rng):
    kind = "stationary" if rng.random() < 0.7 else "finite"
    gamma = 0.0 if rng.random() < 0.35 else _r(rng.uniform(0.02, 0.3))
    eps = _r(rng.uniform(0.15, 0.6)); rho = _r(rng.uniform(0.3, 0.8)); kap = 0.0 if rng.random() < 0.4 else _r(rng.uniform(0.2, 0.8))
    sz = _r(rng.uniform(0.6, 1.4)); transparent = bool(rng.random() < 0.7)
    d = ch6_market(gamma, eps, rho, kap, sz, transparent, kind, 16 if kind == "stationary" else 10)
    meta = dict(gamma=gamma, eps=eps, rho=rho, kap=kap, sz=sz, transparent=transparent, kind=kind)
    return d, ["ch6_market", kind, "transparent" if transparent else "opaque", "gamma0" if gamma == 0 else "gamma"], meta


def _transition(rng):
    d = build(rng, int(rng.integers(1, 3)), 1, "stationary", rows_on_controls=False, window=4.0, nodes=8,
              two_controls=False)
    return d, ["transition", "stationary"]


# the defects an `invalid` case injects, each a model the package must refuse
DEFECTS = ("nan_coef", "inf_coef", "zero_noise", "negative_delay", "lag_beyond_window", "unused_shock", "unknown_atom",
           "future_drift", "negative_theta", "nan_theta", "unused_param", "nodes_one", "negative_discount",
           "control_not_in_loss", "monitor_unknown", "breakpoints_short", "nan_initial", "stationary_initial",
           "signal_own_level", "lead_in_drift", "string_coef_nan", "inf_T", "nan_window")


def _invalid(rng):
    kind = "finite" if rng.random() < 0.5 else "stationary"
    d = build(rng, int(rng.integers(1, 3)), 1, kind, rows_on_controls=False, two_controls=False)
    defect = str(rng.choice(DEFECTS))
    a0 = next(iter(d["agents"].values())); s0 = next(iter(d["states"])); r0 = next(iter(a0["signals"].values()))
    u0 = a0["controls"][0]
    ext = d["horizon"].get("T") or d["horizon"].get("window")
    if defect == "nan_coef":
        d["states"][s0]["drift"][s0] = float("nan")
    elif defect == "inf_coef":
        a0["loss"][0][0] = float("inf")
    elif defect == "zero_noise":
        ch = next(iter(r0["noise"])); r0["noise"][ch] = 0.0
    elif defect == "negative_delay":
        r0["delay"] = -0.25
    elif defect == "lag_beyond_window":
        d["states"][s0]["drift"][f"{u0}@{ext + 0.5:g}"] = 1.0
    elif defect == "unused_shock":
        d["shocks"].append("w_unused")
    elif defect == "unknown_atom":
        a0["loss"].append([1.0, "Nope", "Nope"])
    elif defect == "future_drift":
        d["states"][s0]["drift"][f"{s0}@-0.25"] = 0.3
    elif defect == "negative_theta":
        a0["risk_aversion"] = -0.5
    elif defect == "nan_theta":
        a0["risk_aversion"] = float("nan")
    elif defect == "unused_param":
        d["params"] = {"zzz": 1.0}
    elif defect == "nodes_one":
        d["numerics"]["nodes"] = 1
    elif defect == "negative_discount":
        d["horizon"]["discount"] = -0.2
    elif defect == "control_not_in_loss":
        a0["loss"] = [t for t in a0["loss"] if u0 not in t[1:]] or [[1.0, s0, s0]]
    elif defect == "monitor_unknown":
        a0["monitors"] = ["nobody"]
    elif defect == "breakpoints_short":
        d["numerics"]["breakpoints"] = [0.0, ext / 2]
    elif defect == "nan_initial":
        d["states"][s0]["initial"] = float("nan")
        if kind == "stationary":
            d["horizon"] = {"kind": "finite", "T": 1.0, "discount": 0.0}
    elif defect == "stationary_initial":
        d["states"][s0]["initial"] = 1.0
        d["horizon"] = {"kind": "stationary", "window": 6.0, "discount": 0.2}
        a0.pop("terminal", None)
        for a in d["agents"].values():
            a.pop("terminal", None)
    elif defect == "signal_own_level":
        a0["signals"]["lvl"] = {"level": u0}
    elif defect == "lead_in_drift":
        d["states"][s0]["drift"][f"{u0}@-0.25"] = 0.5
    elif defect == "string_coef_nan":
        d["params"] = {"k": "nan"}
        d["states"][s0]["drift"][s0] = "k"
    elif defect == "inf_T":
        d["horizon"] = {"kind": "finite", "T": float("inf"), "discount": 0.0}
        for a in d["agents"].values():
            a.pop("terminal", None)
    elif defect == "nan_window":
        d["horizon"] = {"kind": "stationary", "window": float("nan"), "discount": 0.2}
        for a in d["agents"].values():
            a.pop("terminal", None)
    return d, ["invalid", defect]


def generate(seed, family=None):
    """The case of `seed` (family drawn from the seed when None)."""
    rng = np.random.default_rng([seed, 7919])
    if family is None:
        family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    rng = np.random.default_rng([seed, FAMILIES.index(family)])
    if family == "lqg1_finite":
        d, tags = _lqg1(rng, "finite")
    elif family == "lqg1_stationary":
        d, tags = _lqg1(rng, "stationary")
    elif family == "game_finite":
        d, tags = _game(rng, "finite")
    elif family == "game_stationary":
        d, tags = _game(rng, "stationary")
    elif family == "delay_finite":
        d, tags = _delay(rng, "finite")
    elif family == "delay_stationary":
        d, tags = _delay(rng, "stationary")
    elif family == "means_finite":
        d, tags = _means(rng)
    elif family == "ties":
        d, tags = _ties(rng)
    elif family == "myopic":
        d, tags = _myopic(rng)
    elif family == "prior":
        d, tags = _prior(rng)
    elif family == "cara_finite":
        d, tags = _cara(rng, "finite")
    elif family == "cara_stationary":
        d, tags = _cara(rng, "stationary")
    elif family == "monitor":
        d, tags = _monitor(rng)
    elif family == "transition":
        d, tags = _transition(rng)
    elif family == "invalid":
        d, tags = _invalid(rng)
    elif family == "ch6_market":
        d, tags, meta = _ch6(rng)
        c = Case(seed=seed, family=family, model=_plain(d), tags=tags, meta=meta)
        c.model["name"] = f"fuzz_{family}_{seed}"
        return c
    else:
        raise ValueError(f"unknown family {family!r}; the families are {FAMILIES}")
    d["name"] = f"fuzz_{family}_{seed}"
    return Case(seed=seed, family=family, model=_plain(d), tags=[str(t) for t in tags])


def _plain(x):
    """numpy scalars and strings as Python ones (YAML and JSON take them)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.str_):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def loss_is_psd(d):
    """Whether every agent's flow loss is positive semidefinite in its atoms (the generator's promise, rechecked after
    the rounding): a failed second-order check on such a model is a false alarm."""
    for a in d["agents"].values():
        atoms = []
        for t in a["loss"]:
            if len(t) == 3:
                for x in t[1:]:
                    if x not in atoms:
                        atoms.append(x)
        if not atoms:
            continue
        Q = np.zeros((len(atoms), len(atoms)))
        for t in a["loss"]:
            if len(t) == 3:
                i, j = atoms.index(t[1]), atoms.index(t[2])
                Q[i, j] += 0.5 * float(t[0]); Q[j, i] += 0.5 * float(t[0])
        if np.min(np.linalg.eigvalsh(Q)) < -1e-12:
            return False
    return True
