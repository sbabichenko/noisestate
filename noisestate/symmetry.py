"""Cyclic symmetry of a tied model: a relabelling of agents, states, controls, definitions and
channels that maps every tied agent to the next one on a cycle and leaves the model unchanged.
Found from the ties by structural matching and verified by rebuilding the model under the
relabelling; the stationary engine uses it to block-diagonalise the closed loop by Fourier modes
over the cycle (the world solve is then linear in the number of tied agents, see
stationary.Compiled.closed_loop)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .spec import Model


@dataclass
class CyclicSymmetry:
    order: int                      # m: the cycle length
    agents: List[str]               # the tied agents in cycle order
    names: Dict[str, str]           # primary/definition/channel -> its image under one step of the cycle
    orbits: List[List[str]]         # orbits of primaries of size m (in cycle order); other primaries are fixed
    channel_orbits: List[List[str]]

    def fixed_primaries(self, prim: List[str]) -> List[str]:
        moved = {n for o in self.orbits for n in o}
        return [n for n in prim if n not in moved]


def _canon(model: Model, expr, ren: Dict[str, str]):
    """An expanded expression with names relabelled, as a sorted tuple."""
    return tuple(sorted((ren.get(n, n), round(l, 9), round(c, 9)) for (n, l), c in expr.items()))


def find_cyclic_symmetry(model: Model, why: Optional[list] = None) -> Optional[CyclicSymmetry]:
    """The cyclic relabelling implied by a single tie group listed in cycle order, or None when the
    model has no such symmetry (no ties, several tie groups, or the relabelling does not reproduce
    the model)."""
    if len(model.ties) != 1 or len(model.ties[0]) < 2:
        if why is not None: why.append('step 1')
        return None
    cycle = list(model.ties[0]); m = len(cycle)
    agents = {a.name: a for a in model.agents}
    ren: Dict[str, str] = {}
    for i, name in enumerate(cycle):
        a, b = agents[name], agents[cycle[(i + 1) % m]]
        ren[a.name] = b.name
        if len(a.controls) != len(b.controls) or len(a.signals) != len(b.signals):
            if why is not None: why.append('step 2')
            return None
        for u, v in zip(a.controls, b.controls):
            ren[u] = v
        for ra, rb in zip(a.signals, b.signals):
            for (ca, wa), (cb, wb) in zip(sorted(ra.noise.items()), sorted(rb.noise.items())):
                ren[ca] = cb
    # private states and definitions: match by the atoms of the rows and losses, canonicalised
    def refs(a):
        out = []
        for r in a.signals:
            out.append(sorted(model.expand(r.drift).items(), key=lambda kv: (round(kv[0][1], 9), round(kv[1], 9), kv[0][0])))
        for t in a.loss:
            for atom in t[1:]:
                out.append(sorted(model.expand({atom: 1.0}).items(), key=lambda kv: (round(kv[0][1], 9), round(kv[1], 9), kv[0][0])))
        return out
    for i, name in enumerate(cycle):
        a, b = agents[name], agents[cycle[(i + 1) % m]]
        for ea, eb in zip(refs(a), refs(b)):
            if len(ea) != len(eb):
                if why is not None: why.append('row/loss atom counts differ')
                return None
            # atoms with equal (lag, coefficient) form a group; the images of the group's names must be the
            # other agent's group, with unmapped names paired only when that pairing is unambiguous
            groups: Dict[tuple, list] = {}
            for ((na, la), ca), ((nb, lb), cb) in zip(ea, eb):
                if abs(la - lb) > 1e-9 or abs(ca - cb) > 1e-9:
                    if why is not None: why.append(f'atom mismatch {na}@{la} x{ca} vs {nb}@{lb} x{cb}')
                    return None
                groups.setdefault((round(la, 9), round(ca, 9)), ([], []))
                groups[(round(la, 9), round(ca, 9))][0].append(na); groups[(round(la, 9), round(ca, 9))][1].append(nb)
            for (A, B) in groups.values():
                for n in A:                               # a name both agents read under the same name is common
                    if n not in ren and n in B:
                        ren[n] = n
                unmapped = [n for n in A if n not in ren]
                image = [ren[n] for n in A if n in ren]
                if any(x not in B for x in image):
                    if why is not None: why.append(f'image {image} not within {B}')
                    return None
                free = [n for n in B if n not in image]
                if len(unmapped) == 1 and len(free) == 1:
                    ren[unmapped[0]] = free[0]
                elif unmapped:
                    if set(unmapped) == set(free):        # common quantities read by both agents under the same name
                        for n in unmapped:
                            ren[n] = n
                    else:
                        if why is not None: why.append(f'ambiguous pairing {unmapped} -> {free}')
                        return None
    # states' own noise channels follow their states; a state's drift may name channels only through noise
    for s in model.states:
        if s.name in ren:
            t = next((x for x in model.states if x.name == ren[s.name]), None)
            if t is None:
                if why is not None: why.append('step 6')
                return None
            for (ca, wa), (cb, wb) in zip(sorted(s.noise.items()), sorted(t.noise.items())):
                ren[ca] = cb
    # verify: the model relabelled must be the model, compared semantically (expanded atoms), since
    # two definitions may have the same expansion
    agents_by = {a.name: a for a in model.agents}
    states_by = {s.name: s for s in model.states}
    for s in model.states:
        t = states_by.get(ren.get(s.name, s.name))
        if t is None or _canon(model, model.expand(s.drift), ren) != _canon(model, model.expand(t.drift), {}) \
                or abs(model.constant(s.drift) - model.constant(t.drift)) > 1e-9 or abs(s.initial - t.initial) > 1e-9 \
                or sorted((ren.get(c, c), round(w, 9)) for c, w in s.noise.items()) != sorted((c, round(w, 9)) for c, w in t.noise.items()):
            if why is not None: why.append(f'state {s.name} does not map onto {ren.get(s.name)}')
            return None
    for a in model.agents:
        b = agents_by.get(ren.get(a.name, a.name))
        ok = b is not None and a.myopic == b.myopic and [ren.get(u, u) for u in a.controls] == list(b.controls) and len(a.signals) == len(b.signals)
        if ok:
            for ra, rb in zip(a.signals, b.signals):
                ok = ok and abs(ra.delay - rb.delay) < 1e-9 and _canon(model, model.expand(ra.drift), ren) == _canon(model, model.expand(rb.drift), {}) \
                    and sorted((ren.get(c, c), round(w, 9)) for c, w in ra.noise.items()) == sorted((c, round(w, 9)) for c, w in rb.noise.items())
            la = sorted((round(float(t[0]), 9), tuple(sorted(_canon(model, model.expand({x: 1.0}), ren) for x in t[1:]))) for t in a.loss)
            lb = sorted((round(float(t[0]), 9), tuple(sorted(_canon(model, model.expand({x: 1.0}), {}) for x in t[1:]))) for t in b.loss)
            ok = ok and la == lb
        if not ok:
            if why is not None: why.append(f'agent {a.name} does not map onto {ren.get(a.name)}')
            return None
    # orbits of primaries and channels
    prim = model.state_names + model.control_names
    seen = set(); orbits = []
    for n in prim:
        if n in seen or n not in ren or ren[n] == n:
            continue
        orb = [n]; x = ren[n]
        while x != n and x not in orb:
            orb.append(x); x = ren.get(x, x)
        if len(orb) != m or x != n:
            if why is not None: why.append('step 9')
            return None
        seen.update(orb); orbits.append(orb)
    seen = set(); ch_orbits = []
    for c in model.channels:
        if c in seen or c not in ren or ren[c] == c:
            continue
        orb = [c]; x = ren[c]
        while x != c and x not in orb:
            orb.append(x); x = ren.get(x, x)
        if len(orb) != m or x != c:
            if why is not None: why.append('step 10')
            return None
        seen.update(orb); ch_orbits.append(orb)
    return CyclicSymmetry(order=m, agents=cycle, names=ren, orbits=orbits, channel_orbits=ch_orbits)
