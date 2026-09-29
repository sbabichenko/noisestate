"""Instant observations and monitored deviations (Chapter 6): what the stationary engine and the spectral finite
engine without a past share, as a mixin of both solvers (StationarySolver, SpectralFiniteSolver).

With a monitoring relation, a unit deviation seed of agent i (a control impulse) is resolved by the players privy to
it, P_i (i itself among them, which resumes play after the blip): they do not filter its effects through their maps
but respond through response kernels D^{v<-i}, one per privy control v, in the seed's age.  The naive players filter
it as always.  By linearity the seed world is

    W_i = Z0_i[:, i] + sum_v C_v D^{v<-i},

Z0_i the closed loop with every privy player's map off and an impulse column per privy control, C_v the response
operator of v's spike (the identity on v's own block: v's kernel is D^{v<-i}).  Each privy player j's first-order
condition on W_i vanishes (Definition 6.3 (ii), sequential rationality), with j's continuation through its own
monitored impulse responses R_j^mon, the responses to a seed of j with j's own reaction frozen (Lemma 6.6: the
first-order condition is the same under the blip convention) and P_j \\ {j} responding through their kernels
D^{.<-j}.  The kernels for different origins depend on each other through R^mon, so they are found by iterating:
kernels from the R^mon, R^mon from the kernels.  Every agent's best response then uses its R^mon (Definition 6.3
(i)).  docs/architecture.md walks through the pieces.

The iteration (_monitoring) is written once here; the engines supply the grid's pieces:

    _spikes(c, maps, agent)                    (Zpass, R): the closed loop without the agent, its spike responses
    _seed_setup(maps, origin)                  (ctrls, Z0, C): the privy controls, their spikes, response operators
    _monitor_focs(agent, maps, R, origins, seed_consts)
                                               the agent's first-order-condition operators (N x n_prim N) per
                                               control; a risk-averse one also fills seed_consts
    _frozen_responses(j, setup, Dj, own)       R^mon_j from j's kernels (a Volterra equation in the seed's time)

spike_controls and compose_spikes build a control's spike with the instant reactions it draws (a trader seeing the
quote's level trades at once), which _spikes and _seed_setup of both engines use.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

from .spec import Agent


def spike_controls(ctrls, comp) -> list:
    """The controls whose impulse columns the spikes of `ctrls` need: ctrls, then the instant reactions each draws
    (comp: control -> {control: coefficient}, the compiled model's composite; a control absent from it draws none)."""
    return list(dict.fromkeys(list(ctrls) + [w for v in ctrls for w in comp.get(v, {v: 1.0})]))


def compose_spikes(cols: np.ndarray, need: list, ctrls, comp) -> Dict[str, np.ndarray]:
    """Each control's spike with the instant reactions it draws, {v: column}, from the impulse columns `cols` of the
    controls `need` (spike_controls)."""
    k = {v: i for i, v in enumerate(need)}
    return {v: sum(coef * cols[:, k[w]] for w, coef in comp.get(v, {v: 1.0}).items()) for v in ctrls}


class MonitoredDeviations:
    """The engines' shared part of instant observations and monitored deviations (module docstring)."""

    def _maps_key(self, maps) -> bytes:
        """A digest of every agent's maps: what the caches of one set of maps are keyed on."""
        h = hashlib.sha1()
        for a in self.model.agents:
            h.update(np.ascontiguousarray(maps[a.name]).tobytes())
        return h.digest()

    def _map_part(self, agent: Agent, Z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The part of the agent's action kernels its map carries: the action less its instant loadings on the
        levels it sees (h times their kernels in Z), which the closed loop adds at the same node."""
        loads = self.c.instant_loads or {}
        if not any(u in loads for u in agent.controls):
            return actions
        out = np.array(actions, dtype=float, copy=True)
        for ui, v in enumerate(agent.controls):
            for u, h in loads.get(v, {}).items():
                out[ui] -= h * Z[self.c.block(u)]
        return out

    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """With a monitoring relation, the agent's spike responses with the players privy to its deviations
        responding through their response kernels (Chapter 6); without one, R."""
        if len(self.model.privy(agent.name)) == 1:
            return R
        return self._monitoring(maps)[0][agent.name]

    def _monitoring(self, maps, origins=None):
        """({agent: R^mon (n_prim N, nU)}, {origin: W (n_prim N, n origin controls)}) for the agents with privy
        others, the response kernels found by the iteration of the module docstring, from the naive responses, to
        1e-10 relative or until 10 rounds pass without improving on the best, which is kept (the plain iteration
        reaches rounding and then only wanders).  The non-origin responders' rows are fixed within a call and built
        once.  (Starting from the previous call's kernels carried a trial point's kernels into the next evaluation
        and broke the market of Chapter 6; Anderson on this iteration found other roots.  Neither is used.)
        origins=None: the equilibrium's origins, cached for the last maps; a list (results: the blip continuation
        of a deviator only it is privy to) is solved outside the cache."""
        key = self._maps_key(maps)
        if origins is None and getattr(self, "_monitored", None) is not None and self._monitored[0] == key:
            self._kernel_residual = self._monitored[3]
            return self._monitored[1], self._monitored[2]
        c = self.c; N = c.N
        owner = {a.name: a for a in self.model.agents}
        own = origins is None
        if own:
            origins = [a.name for a in self.model.agents if len(self.model.privy(a.name)) > 1]
        setup = {i: self._seed_setup(maps, i) for i in origins}
        responders = {m for i in origins for m in self.model.privy(i)}
        Rmon = {n: self._spikes(c, maps, owner[n])[1] for n in responders}      # the naive responses
        shapes = {i: (len(owner[i].controls), len(setup[i][0]), N) for i in origins}
        seed_consts: Dict[tuple, dict] = {}             # (responder, control) -> {origin: (N, nO)}: the seed spike's own terms
        Fu = {n: self._monitor_focs(owner[n], maps, Rmon[n], origins, seed_consts)
              for n in responders if n not in origins}                          # fixed within the call
        eqs = {i: [(n, ui) for n in self.model.privy(i) for ui in range(len(owner[n].controls))] for i in origins}   # one FOC per privy control
        # the rows of the non-origin responders are fixed within the call: their blocks of A and B once
        fixed = {}
        for i in origins:
            ctrls, Z0, C = setup[i]
            for r, (n, ui) in enumerate(eqs[i]):
                if n not in origins:
                    fixed[(i, r)] = ([Fu[n][ui] @ C[v] for v in ctrls], Fu[n][ui] @ Z0[:, :shapes[i][0]])
        D = {}
        best = (np.inf, None, None)                     # (change, D, Rmon) of the best round
        best_round = 0
        for it in range(200):
            Fu.update({n: self._monitor_focs(owner[n], maps, Rmon[n], origins, seed_consts)
                       for n in responders if n in origins})
            change = 0.0
            for i in origins:
                ctrls, Z0, C = setup[i]
                nO, nC = shapes[i][0], len(ctrls)
                A = np.empty((len(eqs[i]) * N, nC * N)); B = np.empty((len(eqs[i]) * N, nO))
                for r, (n, ui) in enumerate(eqs[i]):
                    rows = slice(r * N, (r + 1) * N)
                    blocks, rhs = fixed[(i, r)] if (i, r) in fixed else ([Fu[n][ui] @ C[v] for v in ctrls], Fu[n][ui] @ Z0[:, :nO])
                    for k, blk in enumerate(blocks):
                        A[rows, k * N:(k + 1) * N] = blk
                    B[rows] = -rhs
                    if (n, ui) in seed_consts:
                        B[rows] -= seed_consts[(n, ui)][i]
                X = np.linalg.solve(A, B)                                           # one factorisation, every origin control
                Di = X.T.reshape(shapes[i])
                change = max(change, float(np.abs(Di - D[i]).max() / max(1e-300, np.abs(Di).max())) if i in D else np.inf)
                D[i] = Di
            for j in origins:
                Rmon[j] = self._frozen_responses(j, setup[j], D[j], shapes[j][0])
            if change < best[0]:
                best = (change, dict(D), dict(Rmon)); best_round = it
            # the iteration reaches rounding (about 1e-11) and then only wanders: stop at 1e-10, or once 10 rounds
            # have not improved on the best, and keep the best round
            if change < 1e-10 or it - best_round > 10:
                break
        change, D, Rmon = best
        W = {}                                          # the seed worlds of the best round's kernels
        for i in origins:
            ctrls, Z0, C = setup[i]
            X = D[i].reshape(shapes[i][0], -1).T
            W[i] = Z0[:, :shapes[i][0]] + sum(C[v] @ X[k * N:(k + 1) * N] for k, v in enumerate(ctrls))
        # at a trial point of the outer iteration far from the equilibrium the kernels may not settle; the best
        # round is used there and its change kept, and _finish requires them settled at the equilibrium itself
        if own:
            self._kernel_residual = float(change)
            self._monitored = (key, Rmon, W, self._kernel_residual)
        return Rmon, W

    def _require_settled(self, res) -> None:
        """For _finish: with a monitoring relation, the monitored response kernels must be settled at the
        equilibrium's maps (at trial points of the fixed point they may not be, _monitoring), to the solve's own
        tolerance and at least 1e-10 (at a fine grid the kernels' rounding floor can sit just above 1e-10, and a
        solve converged to tol is not failed for it); a result whose kernels did not settle is not converged."""
        if any(len(self.model.privy(a.name)) > 1 for a in self.model.agents):
            self._monitoring(res.maps)                  # the last evaluation's, when it was at these maps
            if self._kernel_residual > max(1e-10, float(res.solve_kw.get("tol") or 0.0)):
                res.converged = False
                res.message += f"; the monitored response kernels did not settle at the equilibrium (residual {self._kernel_residual:.1e})"
