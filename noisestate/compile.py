"""Model compilation shared by the three engines.

Turns a validated Model into the arrays every engine needs and that do not
depend on the discretisation: the primary quantities (states then controls),
the state dynamics split into the contemporaneous matrix A and the remaining
inputs (controls, lagged quantities), the noise loadings, each agent's signal
rows (name, expanded drift, noise loading vector, observation delay), each
agent's loss as a quadratic form over its expanded atoms, and the tie groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .algebra import KernelAlgebra
from .spec import Atom, Model


@dataclass
class Structure:
    model: Model
    channels: List[str]
    prim: List[str]                                  # state names then control names
    index: Dict[str, int]
    nX: int
    nU: int
    A: np.ndarray                                    # (nX, nX) contemporaneous state feedback
    state_inputs: List[Tuple[int, Atom, float]]      # (state index, atom, coef) for every other drift term
    sigma: np.ndarray                                # (nX, nW) noise loadings
    const: np.ndarray                                # (nX,) constant drifts (they move the means only)
    x0: np.ndarray                                   # (nX,) initial states (finite horizon; they move the means only)
    rows: Dict[str, List[Tuple[str, Dict[Atom, float], np.ndarray, float]]]   # agent -> [(name, drift, E, delay)]
    loss: Dict[str, Tuple[List[Atom], np.ndarray, np.ndarray]]               # agent -> (atoms, Q, q): 1/2 z'Qz + q'z
    rep: Dict[str, str]                              # agent -> representative of its tie group
    reps: List[str]
    terminal: Dict[str, Tuple[List[Atom], np.ndarray, np.ndarray]] = None    # agent -> the loss at T, the same form (agents with one)
    terminal_constant: Dict[str, float] = None       # agent -> the terminal loss's constant
    composite: Dict[str, Dict[str, float]] = None     # control u -> {control v: coef}: a spike of u with the instant reactions it draws
    instant_loads: Dict[str, Dict[str, float]] = None  # observer's control v -> {seen control u: h}: v's contemporaneous loading on u

    @property
    def nW(self) -> int:
        return len(self.channels)


class CompiledBase(KernelAlgebra):
    """What every engine's compiled model starts from: the validated model and its Structure, whose
    fields are adopted as attributes (c.rows, c.loss, c.rep, ...); the kernel algebra (algebra.KernelAlgebra)
    each engine's compiled model implements on top."""
    FIELDS = ("channels", "nW", "prim", "index", "nX", "nU", "A", "state_inputs", "sigma", "const", "x0", "rows", "loss", "rep", "reps",
              "terminal", "terminal_constant", "composite", "instant_loads")

    def __init__(self, model: Model):
        model.validate()
        self.model = model
        self.st = compile_structure(model)
        for k in self.FIELDS:
            setattr(self, k, getattr(self.st, k))


def compile_structure(model: Model) -> Structure:
    model.validate()
    channels = list(model.shocks)
    prim = model.state_names + model.control_names
    index = {n: i for i, n in enumerate(prim)}
    nX, nU, nW = len(model.state_names), len(model.control_names), len(channels)
    A = np.zeros((nX, nX))
    state_inputs: List[Tuple[int, Atom, float]] = []
    for i, s in enumerate(model.states):
        for (n, l), c in model.expand(s.drift).items():
            if n in model.state_names and l == 0:
                A[i, model.state_names.index(n)] += c
            else:
                state_inputs.append((i, (n, l), c))
    sigma = np.zeros((nX, nW))
    for i, s in enumerate(model.states):
        for ch, c in s.noise.items():
            sigma[i, channels.index(ch)] = c
    const = np.array([model.constant(s.drift) for s in model.states], dtype=float)
    x0 = np.array([s.initial or 0.0 for s in model.states], dtype=float)
    rows = {}
    for a in model.agents:
        rr = []
        for r in a.signals:
            E = np.zeros(nW)
            for ch, c in r.noise.items():
                E[channels.index(ch)] = c
            rr.append((r.name, model.expand(r.drift), E, float(r.delay)))
        rows[a.name] = rr
    loss = {a.name: _quadratic(model, a.loss) for a in model.agents}
    terminal = {a.name: _quadratic(model, a.terminal) for a in model.agents if a.terminal}
    terminal_constant = {a.name: float(a.terminal_constant) for a in model.agents if a.terminal_constant}
    rep = {a.name: a.name for a in model.agents}
    for group in model.ties:
        for n in group:
            rep[n] = group[0]
    reps = [a.name for a in model.agents if rep[a.name] == a.name]
    return Structure(model=model, channels=channels, prim=prim, index=index, nX=nX, nU=nU, A=A,
                     state_inputs=state_inputs, sigma=sigma, const=const, x0=x0, rows=rows, loss=loss, rep=rep, reps=reps,
                     terminal=terminal, terminal_constant=terminal_constant, composite=_composite(model, loss),
                     instant_loads=_instant_loadings(model, loss))


def _instant_loadings(model: Model, loss) -> Dict[str, Dict[str, float]]:
    """{observer's control v: {seen control u: h}}: the contemporaneous loadings of the instant observations,
    h = -(G^DD)^-1 G^Du from the observer's loss (see _composite)."""
    out: Dict[str, Dict[str, float]] = {}
    for a in model.agents:
        if not a.instant:
            continue
        atoms, Q, _ = loss[a.name]
        ix = [atoms.index((u, 0.0)) for u in a.controls]
        G = Q[np.ix_(ix, ix)]
        for u in a.instant:
            g = np.array([Q[i, atoms.index((u, 0.0))] if (u, 0.0) in atoms else 0.0 for i in ix])
            for v, h in zip(a.controls, -np.linalg.solve(G, g)):
                if h:
                    out.setdefault(v, {})[u] = float(h)
    return out


def _composite(model: Model, loss) -> Dict[str, Dict[str, float]]:
    """For every control u, the spike of u together with the instant reactions it draws: {u: 1, v: coef, ...}.  An
    agent j that sees u's level (Agent.instant) reacts at once with the loading h = -(G^DD_j)^-1 G^{D u}_j from its
    loss's Hessian (the action of Remark 1.13 with u known exactly); an agent that sees one of j's controls reacts to
    that in turn (the instant graph has no cycle, spec._check_instant)."""
    owner = {u: a for a in model.agents for u in a.controls}
    direct: Dict[str, Dict[str, float]] = {u: {} for u in owner}       # u -> {v: h_vu} for the observers' controls v
    for a in model.agents:
        if not a.instant:
            continue
        atoms, Q, _ = loss[a.name]
        own = [(u, 0.0) for u in a.controls]
        if any(x not in atoms for x in own):
            raise ValueError(f"agent {a.name}: an instant observation needs a quadratic term in each of its controls "
                             "(the loading on the level it sees is its loss's -G^DD^-1 G^Du)")
        ix = [atoms.index(x) for x in own]
        G = Q[np.ix_(ix, ix)]
        for u in a.instant:
            g = np.array([Q[i, atoms.index((u, 0.0))] if (u, 0.0) in atoms else 0.0 for i in ix])
            h = -np.linalg.solve(G, g)
            for v, hv in zip(a.controls, h):
                if hv:
                    direct[u][v] = float(hv)
    out: Dict[str, Dict[str, float]] = {}

    def spread(u):                                  # {u: 1} plus everything downstream, coefficients multiplied along the way
        if u in out:
            return out[u]
        acc = {u: 1.0}
        for v, h in direct[u].items():
            for w, cw in spread(v).items():
                acc[w] = acc.get(w, 0.0) + h * cw
        out[u] = acc
        return acc
    for u in owner:
        spread(u)
    return out


def _quadratic(model: Model, terms) -> Tuple[List[Atom], np.ndarray, np.ndarray]:
    """The loss terms [[coef, a, b], [coef, a], ...] as (atoms, Q, q) with the loss 1/2 z'Qz + q'z over the atoms
    (each term's quantities expanded into primaries at their lags)."""
    atoms: List[Atom] = []
    parsed = []
    for term in terms:
        coef = float(term[0])
        ex = [model.expand({s: 1.0}) for s in term[1:]]
        for e in ex:
            for k in e:
                if k not in atoms:
                    atoms.append(k)
        parsed.append((coef, ex))
    m = len(atoms)
    Q = np.zeros((m, m)); q = np.zeros(m)
    for coef, ex in parsed:
        if len(ex) == 1:
            for k, c in ex[0].items():
                q[atoms.index(k)] += coef * c
        else:
            for k1, c1 in ex[0].items():
                for k2, c2 in ex[1].items():
                    i, j = atoms.index(k1), atoms.index(k2)
                    Q[i, j] += coef * c1 * c2
                    Q[j, i] += coef * c1 * c2
    return atoms, Q, q


def reject_leads(model: Model, engine: str) -> None:
    """The finite-horizon engines do not carry the past-date term a lead needs."""
    for a in model.agents:
        for term in a.loss:
            for atom in term[1:]:
                if any(l < 0 for (n, l) in model.expand({atom: 1.0})):
                    raise NotImplementedError(f"{engine}: lead atoms ({atom}) in losses are supported by the stationary engine only")


def close_under_delays(bp, delays, eps: float = 1e-9):
    """The breakpoints with b - d and b + d added for every breakpoint b and row delay d, repeated until
    closed within [0, L].  Both directions: the map on a delayed row over an interval is read by the
    action over that interval shifted by the delay, and each must be a union of panels of the other,
    or the finer side has modes the coarser side cannot see (a singular system)."""
    bp = sorted(float(b) for b in bp); L = bp[-1]
    changed = True
    while changed:
        changed = False
        for d in delays:
            for b in list(bp):
                for c in (b - d, b + d):
                    if eps < c < L - eps and not any(abs(c - x) < eps for x in bp):
                        bp.append(c); changed = True
        bp = sorted(bp)
    return bp
