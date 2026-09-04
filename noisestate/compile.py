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
    rows: Dict[str, List[Tuple[str, Dict[Atom, float], np.ndarray, float]]]   # agent -> [(name, drift, E, delay)]
    loss: Dict[str, Tuple[List[Atom], np.ndarray, np.ndarray]]               # agent -> (atoms, Q, q): 1/2 z'Qz + q'z
    rep: Dict[str, str]                              # agent -> representative of its tie group
    reps: List[str]

    @property
    def nW(self) -> int:
        return len(self.channels)


def compile_structure(model: Model) -> Structure:
    model.validate()
    channels = list(model.channels)
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
    rows = {}
    for a in model.agents:
        rr = []
        for r in a.signals:
            E = np.zeros(nW)
            for ch, c in r.noise.items():
                E[channels.index(ch)] = c
            rr.append((r.name, model.expand(r.drift), E, float(r.delay)))
        rows[a.name] = rr
    loss = {}
    for a in model.agents:
        atoms: List[Atom] = []
        terms = []
        for term in a.loss:
            coef = float(term[0])
            ex = [model.expand({s: 1.0}) for s in term[1:]]
            for e in ex:
                for k in e:
                    if k not in atoms:
                        atoms.append(k)
            terms.append((coef, ex))
        m = len(atoms)
        Q = np.zeros((m, m)); q = np.zeros(m)
        for coef, ex in terms:
            if len(ex) == 1:
                for k, c in ex[0].items():
                    q[atoms.index(k)] += coef * c
            else:
                for k1, c1 in ex[0].items():
                    for k2, c2 in ex[1].items():
                        i, j = atoms.index(k1), atoms.index(k2)
                        Q[i, j] += coef * c1 * c2
                        Q[j, i] += coef * c1 * c2
        loss[a.name] = (atoms, Q, q)
    rep = {a.name: a.name for a in model.agents}
    for group in model.ties:
        for n in group:
            rep[n] = group[0]
    reps = [a.name for a in model.agents if rep[a.name] == a.name]
    return Structure(model=model, channels=channels, prim=prim, index=index, nX=nX, nU=nU, A=A,
                     state_inputs=state_inputs, sigma=sigma, rows=rows, loss=loss, rep=rep, reps=reps)


def reject_leads(model: Model, engine: str) -> None:
    """The finite-horizon engines do not carry the past-date term a lead needs."""
    for a in model.agents:
        for term in a.loss:
            for atom in term[1:]:
                if any(l < 0 for (n, l) in model.expand({atom: 1.0})):
                    raise NotImplementedError(f"{engine}: lead atoms ({atom}) in losses are supported by the stationary engine only")
