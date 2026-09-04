"""Diagnostic variants of the stationary engine.

CppCascadeReplica reproduces the multi-trader equilibria of the dissertation's C++ spectral
Kyle-Back solvers (kb_spectral.cpp, kb_spectral_q.cpp) as published on 2026-09-03: their
price-impact cascade applies each opponent's residual-flow policy inside a total-flow feedback,
so the opponent's own reaction is fed back into its flow observation.  The replica uses that
inconsistent reaction both in the continuation and in the reconstruction of the passive world,
exactly as the C++ does; see patches/README.md for the fix.  It exists to pin the defect in a
regression test, not to solve models.
"""
from __future__ import annotations

import json

import numpy as np

from noisestate.spec import Model
from noisestate.stationary import StationarySolver


class CppCascadeReplica(StationarySolver):
    def __init__(self, model: Model, **kw):
        super().__init__(model, **kw)
        d = json.loads(json.dumps(model.to_dict()))
        # every flow row (one that observes other agents' controls) is made gross of the observer's own controls
        for aname, a in d["agents"].items():
            if a.get("myopic"):
                continue                     # a competitive pricing agent has no cascade of its own
            strategic = {u for ag in model.agents if not ag.myopic for u in ag.controls}
            for rname, row in a["signals"].items():
                observed = [k.split("@")[0] for k in row["drift"]]
                # a flow row: observes other strategic agents' controls (not the pricing agent's price)
                if any(o in strategic and o not in a["controls"] for o in observed):
                    for u in a["controls"]:
                        row["drift"][u] = 1.0
        self.gross = StationarySolver(Model.from_dict(d))

    def _impulse_responses(self, agent, maps, R):
        return self.gross.c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)[:, self.c.nW:]

    def _passive_world(self, agent, maps, Zpass, R):
        c = self.c; N = c.N
        Zact = c.closed_loop(maps)
        out = Zact.copy()
        for ui, u in enumerate(agent.controls):
            Ru = R[:, ui].reshape(len(c.prim), N); cu = Zact[c.block(u)]
            for p in range(len(c.prim)):
                if c.prim[p] == u:
                    out[c.block(u)] = 0.0
                elif np.abs(Ru[p]).max() > 0:
                    out[p * N:(p + 1) * N] -= c.grid.conv_op(Ru[p]) @ cu
        return out
