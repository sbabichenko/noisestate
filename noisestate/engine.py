"""Plumbing shared by the three engines.

Every engine iterates on a dict of per-agent arrays (raw maps on the agent's rows, or action
kernels), packs the arrays of the tie-group representatives into one vector for the outer solver,
and fans a best response out over the representatives, copying it to the agents tied to them.
That bookkeeping lives here once; the engines supply `best_response`, the array shapes, and the
conversion from action kernels to maps.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .accel import solve_fixed_point
from .spec import Agent, Model


class EngineBase:
    """The passive-world best response and the outer fixed point, written once against a small
    kernel algebra that each compiled model supplies:

        conv_rows(Y, delay)      (N, m) row kernels -> (m, N, N): gamma on a row -> its action
        instant(age)             (N, N): the instantaneous entry of a row, gamma(. - age)
        instant_adjoint(age)     (N, N): its adjoint, phi(. + age)
        response(Ru, own)        (n_prim, N) impulse responses -> (n_prim N, N): action -> world
        continuation(Rj)         (N, m) -> (m, N, N): discounted continuation through impulse responses
        own_lag_read(lag)        (N, N): the FOC term of a delayed read of the control itself
        projection_rows(Y, d)    (N, nW) row kernels -> (N, nW N): E[phi_t dY_r(t - b)]

    The engines keep what genuinely differs: the closed loop, the solve of the FOC system and its
    regularisation, the projection back to raw maps, and their own extras (decomposition, second
    order, naive observers)."""
    model: Model
    c: object                       # the compiled model: .reps (tie representatives), .rep (agent -> representative), .N, .nW
    shapes: Dict[str, Tuple[int, ...]]      # agent -> shape of its raw maps
    RESULT = None                   # the result class
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.5, 60       # solve() defaults; each engine sets its own
    ACTIONS = True                  # whether the engine can iterate on action kernels
    SECOND_ORDER_TOL = 1e-4         # curvature (relative to the largest) below which a negative value is window truncation
    SECOND_ORDER_QUADRATIC = True   # whether the objective is a quadratic form in the strategy at every discount
    SECOND_ORDER_DENSE = 2500       # strategy dimension up to which the form is built densely (always settles, 2.7 s at 1600); Lanczos above

    def __init__(self, model: Model, verbose: bool = False, **options):
        self.model = model
        self.verbose = verbose
        self.solver_kw = {"verbose": verbose, **options}   # so a result can rebuild the same engine
        self._qa: Dict[str, np.ndarray] = {}                # agent -> (Q zeta) as an operator on the primary kernels
        self._rphys: Dict[str, np.ndarray] = {}             # agent -> physical impulse responses (all reactions off)
        self._second_order_cache: Dict[str, dict] = {}       # representative -> its second-order check, shared with tied agents

    # ------------------------------------------------------------ ties
    def _fill_ties(self, d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Give every agent tied to a representative the representative's array."""
        for a in self.model.agents:
            if a.name not in d:
                d[a.name] = d[self.c.rep[a.name]]
        return d

    def _over_representatives(self, fn: Callable[[Agent], np.ndarray]) -> Dict[str, np.ndarray]:
        """fn on each tie representative, copied to the agents tied to it."""
        return self._fill_ties({a.name: fn(a) for a in self.model.agents if self.c.rep[a.name] == a.name})

    # ---------------------------------------------------------- packing
    def _pack(self, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(arrays[n]).reshape(-1) for n in self.c.reps])

    def _unpack(self, z: np.ndarray, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
        out, pos = {}, 0
        for n in self.c.reps:
            size = int(np.prod(shapes[n])); out[n] = z[pos:pos + size].reshape(shapes[n]); pos += size
        return self._fill_ties(out)

    def pack(self, maps: Dict[str, np.ndarray]) -> np.ndarray:
        """Raw maps of the representatives as one vector."""
        return self._pack(maps)

    def unpack(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        return self._unpack(z, self.shapes)

    def zero_maps(self) -> Dict[str, np.ndarray]:
        return {a.name: np.zeros(self.shapes[a.name]) for a in self.model.agents}

    @property
    def action_shapes(self) -> Dict[str, Tuple[int, int, int]]:
        """Action kernels: (n_controls, N, nW) per agent."""
        return {a.name: (len(a.controls), self.c.N, self.c.nW) for a in self.model.agents}

    def pack_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        return self._pack(actions)

    def unpack_actions(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        return self._unpack(z, self.action_shapes)

    # ---------------------------------------------------------- checks
    def init_kind(self, init: Dict[str, np.ndarray]) -> str:
        """Classify a warm start as "actions" or "maps" by shape; a wrong or ambiguous shape is an error."""
        kinds = set()
        for a in self.model.agents:
            if a.name not in init:
                raise ValueError(f"init has no entry for agent {a.name}")
            v = np.asarray(init[a.name]); sa, sm = self.action_shapes[a.name], tuple(self.shapes[a.name])
            if v.shape == sa and v.shape != sm:
                kinds.add("actions")
            elif v.shape == sm and v.shape != sa:
                kinds.add("maps")
            elif v.shape == sa:
                raise ValueError(f"init for {a.name}: shape {v.shape} could be action kernels or raw maps (N == nR == nW); "
                                 "pass the other representation")
            else:
                raise ValueError(f"init for {a.name}: shape {v.shape}; expected action kernels {sa} or raw maps {sm} "
                                 "(a warm start from a different grid cannot be used directly)")
        if len(kinds) != 1:
            raise ValueError("init mixes action kernels and raw maps across agents")
        return kinds.pop()

    def same_grid(self, other) -> bool:
        """Whether another compiled model lives on the same grid (so its kernels can be a warm start)."""
        g1, g2 = getattr(self.c, "grid", None), getattr(other, "grid", None)
        if g1 is not None or g2 is not None:
            return g1 is g2
        g1, g2 = getattr(self.c, "g", None), getattr(other, "g", None)
        if g1 is not None or g2 is not None:
            return g1 is g2
        return self.c.N == other.N and abs(self.c.h - other.h) < 1e-12

    # ------------------------------------------------------ seen rows
    def _seen_rows(self, agent: Agent, Z: np.ndarray, excluded: set):
        """The agent's signal rows in the world Z: regular kernels (one array per row) and the
        instantaneous entries [(channel index, age, weight)] per row.  Controls in `excluded` are
        switched off (their impulses come through impulse channels instead)."""
        c = self.c; rows, inst = [], []
        blockwise = hasattr(c, "row_blocks")
        for r in range(len(agent.signals)):
            if blockwise:
                blocks, deltas = c.row_blocks(agent.name, r, excluded)
                y = np.zeros((c.N, Z.shape[1]))
                for nm, op in blocks.items():                            # only the primaries the row reads
                    y += op @ Z[c.block(nm)]
                rows.append(y)
            else:
                regular, deltas = c.row(agent.name, r, excluded)
                rows.append(regular @ Z)
            inst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        return rows, inst

    def _passive_rows(self, agent: Agent, Zpass: np.ndarray):
        """The rows in the agent's passive world (its own strategy off)."""
        return self._seen_rows(agent, Zpass, set(agent.controls))

    # ------------------------------------------------- best-response pieces
    def _row_operator(self, agent: Agent, rows, inst):
        """Per channel, the operator (N x nR N) mapping stacked row maps gamma to the action kernel:
        c_k = sum_r (Conv[y_rk] + E_rk S_delta) gamma_r."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        Gk = np.zeros((nW, N, nR * N))
        for r in range(nR):
            Gk[:, :, r * N:(r + 1) * N] += c.conv_rows(rows[r], c.rows[agent.name][r][3])
            for (k, age, w) in inst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.instant(age)
        return Gk

    def _response_operators(self, agent: Agent, R: np.ndarray):
        """Per control, the operator (n_prim N x N) giving the primary kernels' response to that
        control's action kernel: Z = Zpass + Resp_u c_u, with the own block equal to the action."""
        c = self.c; N = c.N
        out = []
        for ui, u in enumerate(agent.controls):
            Cu = c.response(R[:, ui].reshape(len(c.prim), N), c.prim.index(u))
            Cu[c.block(u)] = np.eye(N)
            out.append(Cu)
        return out

    def _qa_operator(self, agent: Agent) -> np.ndarray:
        """(Q zeta) as an operator on the primary kernels: map-independent, cached per agent."""
        if agent.name not in self._qa:
            c = self.c; atoms, Q, q = c.loss[agent.name]
            AO = np.concatenate([c.atom_op(at) for at in atoms], axis=0)
            self._qa[agent.name] = np.kron(Q, np.eye(c.N)) @ AO
        return self._qa[agent.name]

    def _foc_operators(self, agent: Agent, R: np.ndarray):
        """Per control, the operator (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses R, delayed reads of own lagged controls, and the past-date term of a lead.
        Called with the physical impulse responses (all reactions off) for the wedge decomposition."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atoms, Q, q = c.loss[agent.name]
        QA = self._qa_operator(agent)
        Fu = []
        for ui, u in enumerate(agent.controls):
            op = np.zeros((N, n_prim))
            if (u, 0.0) in atoms:
                j0 = atoms.index((u, 0.0))
                op += QA[j0 * N:(j0 + 1) * N]
            if not agent.myopic:
                Rj = np.stack([c.atom_op(at) @ R[:, ui] for at in atoms], axis=1)   # impulse responses of every atom
                CR = c.continuation(Rj)
                for j, (name, lag) in enumerate(atoms):
                    Qj = QA[j * N:(j + 1) * N]
                    if name in agent.controls:
                        if name == u and lag > 0:          # delayed read of the control itself
                            op += np.exp(-c.rho * lag) * c.own_lag_read(lag) @ Qj
                        continue                            # own reactions: envelope
                    op += CR[j] @ Qj
                    if lag < 0:
                        op += self._lead_term(agent, R[:, ui], name, lag) @ Qj
            Fu.append(op)
        return Fu

    def _lead_term(self, agent: Agent, Ru: np.ndarray, name: str, lag: float) -> np.ndarray:
        """The FOC term of a lead (name@lag, lag < 0) from flows before t that read the quantity after t."""
        raise NotImplementedError("leads are supported by the stationary engine only")

    def _projection_operator(self, agent: Agent, rows, inst):
        """H (nR N x nW N): E[phi_t dY_r(t - b)] for every row r and lag b, from the FOC kernels."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            H[r * N:(r + 1) * N] += c.projection_rows(rows[r], c.rows[agent.name][r][3])
            for (k, age, w) in inst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.instant_adjoint(age)
        return H

    # ----------------------------------------------------- best response
    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """Responses of the primary kernels to a unit impulse of each of the agent's controls, with the
        agent's own reaction switched off (hook: naive observers)."""
        return R

    def _passive_world(self, agent: Agent, maps, Zpass: np.ndarray, R: np.ndarray) -> np.ndarray:
        """The agent's passive world, its own strategy off (hook for diagnostics)."""
        return Zpass

    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        """gamma solving Amat gamma = -bvec, with the engine's regularisation."""
        raise NotImplementedError

    def _project(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Raw maps (nU, nR, N) reproducing the action kernels on the agent's closed-loop rows."""
        raise NotImplementedError

    def _decompose(self, agent: Agent, out: dict, Fu, Resp, Gk) -> None:
        """The second-order check and the FOC decomposition (instantaneous/physical/wedge).  Tied agents
        share the second-order check of their representative (the same problem up to relabelling)."""
        c = self.c; nW = c.nW; Zfull = out["Zfull"]
        rep = c.rep[agent.name]
        if rep != agent.name and rep in self._second_order_cache:
            out["second_order"] = self._second_order_cache[rep]
        else:
            out["second_order"] = self._second_order(agent, Resp, Gk, np.tile(self._identified(agent), len(agent.controls)))
            self._second_order_cache[agent.name] = out["second_order"]
        if agent.name not in self._rphys:                                   # physical impulse responses: map-independent
            self._rphys[agent.name] = c.closed_loop(self.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, nW:]
        Fphys = self._foc_operators(agent, self._rphys[agent.name])
        dec = {}
        for ui, u in enumerate(agent.controls):
            phi = np.stack([Fu[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            phi_phys = np.stack([Fphys[ui] @ Zfull[:, k] for k in range(nW)], axis=1)
            dec[u] = {"foc": phi, "physical": phi_phys, "wedge": phi - phi_phys}
        out["decomp"] = dec

    def _identified(self, agent: Agent) -> np.ndarray:
        """Mask over the stacked map nodes of the ages at which the map on each row reads something
        (all of them, unless the engine masks delayed rows)."""
        return np.ones(len(agent.signals) * self.c.N, dtype=bool)

    def _second_order(self, agent: Agent, Resp, Gk, keep) -> Optional[dict]:
        """Second-order condition of the best response: the agent's objective is a quadratic form in its
        strategy, and a first-order condition is a minimum only if that form is positive on the
        feasible strategies (those its rows can express).  The form is computed exactly from the
        cost's own Gram matrix: J(delta) = 1/2 delta' M delta with M = T' G T, T the map from a
        strategy to the world it produces and G the loss form.  Its extreme eigenvalues come from
        Lanczos on matvecs.  Returns {"min", "max", "ok", "converged"} with min/max the eigenvalues
        of M scaled by max; None when the objective is not a quadratic form in the strategy (the
        stationary engine with rho > 0: the discounted objective is not one in the stationary kernel).
        The objective is truncated at the window, so a strategy can push a little loss past the edge:
        curvatures within SECOND_ORDER_TOL of the largest are treated as that, not as a saddle."""
        c = self.c
        if not (self.SECOND_ORDER_QUADRATIC or c.rho == 0):
            return None
        from scipy.sparse.linalg import LinearOperator, eigsh
        N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        atoms, Q, q = c.loss[agent.name]
        AO = np.concatenate([c.atom_op(at) for at in atoms], axis=0)           # (m N, n_prim N)
        QM = np.kron(Q, c.cost_mass())                                           # the loss form on the atoms
        GAO = AO.T @ (QM @ AO)                                                   # symmetric loss form on the world
        idx = np.where(keep)[0]

        def T(delta_full):                     # strategy -> world, per channel: (n_prim N, nW)
            Zd = np.zeros((GAO.shape[0], nW))
            for ui in range(nU):
                du = delta_full[ui * nR * N:(ui + 1) * nR * N]
                for k in range(nW):
                    Zd[:, k] += Resp[ui] @ (Gk[k] @ du)
            return Zd

        def Tt(Zd):                            # its transpose
            out = np.zeros(nU * nR * N)
            for ui in range(nU):
                RZ = Resp[ui].T @ Zd                                            # (N, nW)
                for k in range(nW):
                    out[ui * nR * N:(ui + 1) * nR * N] += Gk[k].T @ RZ[:, k]
            return out

        def matvec(v):
            full = np.zeros(nU * nR * N); full[idx] = np.asarray(v, dtype=float).ravel()
            return Tt(GAO @ T(full))[idx]
        n = idx.size
        if n <= self.SECOND_ORDER_DENSE:
            # the form explicitly: T per channel as a matrix (n_prim N x n), M = sum_k T_k' G T_k
            Mfull = np.zeros((n, n))
            for k in range(nW):
                Tk = np.zeros((GAO.shape[0], n))
                for ui in range(nU):
                    cols = np.where((idx >= ui * nR * N) & (idx < (ui + 1) * nR * N))[0]
                    if cols.size:
                        Tk[:, cols] = Resp[ui] @ Gk[k][:, idx[cols] - ui * nR * N]
                Mfull += Tk.T @ (GAO @ Tk)
            w = np.linalg.eigvalsh((Mfull + Mfull.T) / 2)
            lo, hi = float(w[0]), float(w[-1])
        else:
            op = LinearOperator((n, n), matvec=matvec, dtype=float)
            try:
                hi = float(eigsh(op, k=1, which="LA", tol=1e-6, maxiter=300, return_eigenvectors=False)[0])
                lo = float(eigsh(op, k=1, which="SA", tol=1e-6, maxiter=300, return_eigenvectors=False)[0])
            except Exception as exc:                          # Lanczos did not settle: say so rather than stay silent
                return {"min": None, "max": None, "ok": None, "converged": False, "message": f"{type(exc).__name__}: {exc}"[:120]}
        scale = max(abs(lo), abs(hi), 1e-300)
        return {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -self.SECOND_ORDER_TOL * scale), "converged": True}

    # ------------------------------------------------ maps from kernels

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        """The agent's best response to `maps`: (raw map, {"gamma", "action", "Zfull", ...}).  The
        agent's information is the passive signal history, so its first-order condition is affine
        in its map on the passive rows: one linear solve."""
        c = self.c; N, nW = c.N, c.nW
        nR, nU = len(agent.signals), len(agent.controls)
        Zp = c.closed_loop(maps, excluded=agent.name, impulse_controls=agent.controls)
        Zpass, R = Zp[:, :nW], Zp[:, nW:]
        R = self._impulse_responses(agent, maps, R)
        Zpass = self._passive_world(agent, maps, Zpass, R)
        ytil, yinst = self._passive_rows(agent, Zpass)
        Gk = self._row_operator(agent, ytil, yinst)
        Resp = self._response_operators(agent, R)
        Fu = self._foc_operators(agent, R)
        H = self._projection_operator(agent, ytil, yinst)
        # the FOC is affine in gamma: solve H (Fu (Zpass + sum_v Resp_v Gk gamma_v)) = 0 for all controls
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        Gk_flat = Gk.transpose(1, 0, 2).reshape(N, nW * nR * N)             # (N, [k, gamma]) for one product per control pair
        for ui in range(nU):
            rowsl = slice(ui * nR * N, (ui + 1) * nR * N)
            FRG = np.concatenate([((Fu[ui] @ Resp[vi]) @ Gk_flat).reshape(N, nW, nR * N).transpose(1, 0, 2).reshape(nW * N, nR * N)
                                  for vi in range(nU)], axis=1)              # FR applied per channel, all responding controls
            Amat[rowsl] += H @ FRG                                           # one product over all channels and controls
            bvec[rowsl] += H @ (Fu[ui] @ Zpass).T.reshape(-1)
        gamma = self._solve_foc(agent, Amat, bvec).reshape(nU, nR, N)
        cact = np.stack([np.stack([Gk[k] @ gamma[ui].reshape(-1) for k in range(nW)], axis=1) for ui in range(nU)])
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp[ui] @ cact[ui]
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            self._decompose(agent, out, Fu, Resp, Gk)
        return self._project(agent, Zfull, cact), out

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        """Relative residual of the best-response action kernels after projection on the agent's raw
        rows.  Zero in exact arithmetic; on the grid it measures how well products of kernels are
        resolved, so a value above about 1e-6 means the equilibrium is under-resolved: raise
        horizon.nodes."""
        rows, inst = self._seen_rows(agent, Zfull, set())
        Bk = self._row_operator(agent, rows, inst)
        worst = 0.0
        for ui in range(len(agent.controls)):
            recon = np.stack([Bk[k] @ g[ui].reshape(-1) for k in range(self.c.nW)], axis=1)
            worst = max(worst, float(np.abs(recon - actions[ui]).max() / max(1e-300, np.abs(actions[ui]).max())))
        return worst

    # ------------------------------------------------------ fixed points
    def solve(self, init: Optional[Dict[str, np.ndarray]] = None, tol: Optional[float] = None, damping: Optional[float] = None,
              max_newton: Optional[int] = None, variable: str = "actions"):
        """Find the equilibrium: Anderson mixing on the fixed point of the best-response map (damping is
        the mixing weight), then a Newton-Krylov polish if it stalls.  variable="actions" iterates on
        the agents' action kernels, with the raw maps recovered by projection (better conditioned where
        strategies are weakly identified); "maps" iterates on the raw maps, which is also what happens
        with ties (tied agents' action kernels differ by a channel permutation) and on the cell engine.
        init: action kernels or raw maps per agent, either is accepted."""
        tol = self.TOL if tol is None else tol
        damping = self.DAMPING if damping is None else damping
        max_newton = self.MAX_NEWTON if max_newton is None else max_newton
        t0 = time.time(); evals = [0]
        if variable == "actions" and (self.model.ties or not self.ACTIONS):
            variable = "maps"
        kind = self.init_kind(init) if init is not None else None
        if kind == "actions" and not self.ACTIONS:
            raise ValueError("this engine iterates on raw maps: pass raw maps as init")
        if variable == "actions":
            if kind is None:
                start = {a.name: np.zeros(self.action_shapes[a.name]) for a in self.model.agents}
            else:
                start = init if kind == "actions" else self.actions_from_maps(init)
            pack, unpack, respond = self.pack_actions, self.unpack_actions, self.response_actions
        else:
            if kind is None:
                start = self.zero_maps()
            else:
                start = init if kind == "maps" else self.maps_from_actions(init)
            pack, unpack, respond = self.pack, self.unpack, self.response_map

        def F(zz):
            evals[0] += 1
            return pack(respond(unpack(zz))) - zz
        z, resid, _, converged, message = solve_fixed_point(F, pack(start), tol=tol, verbose=self.verbose, damping=damping,
                                                            max_newton=max_newton)
        maps = self.maps_from_actions(unpack(z)) if variable == "actions" else unpack(z)
        Z = self.c.closed_loop(maps)
        res = self.RESULT(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                          iterations=evals[0], seconds=time.time() - t0, message=message, solver_class=type(self),
                          solver_kw=self.solver_kw,
                          solve_kw={"tol": tol, "damping": damping, "max_newton": max_newton, "variable": variable})
        self._finish(res)
        return res

    def _finish(self, res) -> None:
        """Fill the engine's own outputs on a fresh result: costs, and what else it computes."""
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, res.Z)

    def actions_from_maps(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Action kernels the raw maps produce in their own closed loop."""
        Z = self.c.closed_loop(maps)
        return {a.name: np.stack([Z[self.c.block(u)] for u in a.controls]) for a in self.model.agents}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        raise NotImplementedError

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Raw maps that reproduce the given action kernels in the world they generate."""
        raise NotImplementedError

    def response_map(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Every agent's best-response map against `maps` (the map-iteration fixed-point function)."""
        return self._over_representatives(lambda a: self.best_response(a, maps)[0])

    def response_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Every agent's best-response action kernels against `actions`."""
        maps = self.maps_from_actions(actions)
        return self._over_representatives(lambda a: self.best_response(a, maps)[1]["action"])
