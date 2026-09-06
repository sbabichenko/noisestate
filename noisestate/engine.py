"""Plumbing shared by the three engines.

Every engine iterates on a dict of per-agent arrays (raw maps on the agent's rows, or action
kernels), packs the arrays of the tie-group representatives into one vector for the outer solver,
and fans a best response out over the representatives, copying it to the agents tied to them.
That bookkeeping lives here once; the engines supply `best_response`, the array shapes, and the
conversion from action kernels to maps.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.linalg as sla

from .accel import solve_fixed_point
from .settings import Settings, tunable
from .spec import Agent, Model


def _is_eye(M: np.ndarray) -> bool:
    """Whether M is exactly the identity (a product with it is then skipped: X @ I is X to the bit)."""
    n = M.shape[0]
    return M.ndim == 2 and M.shape[1] == n and np.count_nonzero(M) == n and bool(np.all(np.diagonal(M) == 1.0))


def singular_system_message(name: str) -> str:
    """The error every engine raises on a singular best-response system, naming its usual causes."""
    return (f"the best-response system of {name} is singular: two of its rows may carry the same information, a "
            "control may have no quadratic term in its current value (a quadratic in a lagged read, D@tau, leaves "
            "the strategy free within tau of the window's edge), a row's noise loading may be zero, or the system "
            "may be too ill-conditioned at this resolution (a very small control penalty, a long window)")


class EngineBase:
    """The passive-world best response and the outer fixed point, written once against a small
    kernel algebra that each compiled model supplies and a set of hooks the engines fill in.

    Layout.  A kernel is a nodal vector over the engine's N nodes (shock ages on the stationary
    grid, (time, age) nodes on the triangle).  The world Z is (n_prim N, ncol): rows in
    (primary, node) order, c.block(name) slicing a primary's N nodes; columns the nW Brownian
    channels, then one unit-impulse column per control in `impulse_controls`.  An agent's raw
    maps are (nU, nR, N) (control, signal row, node of the row as the agent sees it: a delayed
    row's map is stored at the shifted time or age); its action kernels are (nU, N, nW); the
    first-order-condition unknown gamma is the maps flattened in (control, row, node) order, and
    a mask over map nodes is (nR N,) in (row, node) order.  The cell engine has its own layouts
    (Z (n_prim, N, ncol) per cell, maps (nU, nR, N, N)) and overrides best_response wholesale, so
    the base's best-response pieces (_seen_rows to _foc_system, _decompose, _second_order) never
    see them; it uses the packing, the fixed point, _finish and the mean hook only.

    Kernel algebra of the compiled model self.c (see each engine's Compiled class):

        closed_loop(maps, excluded=None, impulse_controls=())
                                 Z (n_prim N, nW + len(impulse_controls)) under `maps`, with the agent
                                 `excluded` switched off and a unit impulse of each listed control
        block(name)              slice of the primary's N rows in Z
        atom_op((name, lag))     (N, n_prim N): the kernel of name@lag from the primary vector
        conv_rows(Y, delay)      (N, m) row kernels, as seen (shifted by the delay) -> (m, N, N): the
                                 map gamma on such a row -> the action kernel on that channel
        instant(age, delay)      (N, N): the action's instantaneous read, at `age`, of the map on a row
                                 observed with `delay` (a row's own noise at age = delay; an observed
                                 control's impulse at delay + lag)
        instant_adjoint(age, delay)  (N, N): its adjoint on the FOC kernel phi
        response(Ru, own)        (n_prim, N) impulse responses of the primaries to control `own` ->
                                 (n_prim N, N): action kernel -> world; the base then sets the own
                                 block to the identity
        continuation(Rj)         (N, m) impulse responses of m loss atoms -> (m, N, N): the discounted
                                 continuation of each atom's kernel through its response
        own_lag_read(lag)        (N, N): the FOC term of a delayed read of the control itself
        projection_rows(Y, d)    (N, m) row kernels -> (N, m N), columns (channel, node): E[phi_t dY_r(t - b)]
        cost_mass()              (N, N) Gram matrix under which expected_cost integrates products of kernels
        causal_chunks()          optional: [(lo, hi)] node ranges in increasing age such that projection_rows
                                 is zero from a chunk's ages to nodes of an earlier one (_causal_chunks)
        row_blocks(agent, r, excluded)  optional: ({primary: (N, N)}, {source: [(age, weight)]}), the seen row's
                                 regular part per primary it reads and its instantaneous entries;
                                 else row(agent, r, excluded) -> ((N, n_prim N), the same deltas)
    and its attributes N, nW, prim, index, channels, rows (agent -> [(name, drift, E, delay)]),
    loss (agent -> (atoms, Q, q)), rep, reps, rho, model, and grid / g / h (same_grid).

    Hooks: what an engine overrides (the base's default in brackets) and which engines do:

        hook                 purpose                                                   overridden by
        __init__             build self.c, self.shapes; record the options in solver_kw  all three
        pack, unpack         maps of the tie representatives <-> one vector [flat]     cells
        best_response        (raw map, {"gamma", "action", "Zfull", ...}) [FOC solve]  cells
        _impulse_responses   R with some observers' reactions removed [R]              stationary
        _passive_world       Zpass adjusted [Zpass]                                    none
        _identified          mask of the map nodes that read something [all True]     stationary, spectral
        _solve_foc           gamma from Amat gamma = -bvec [abstract]                  stationary, spectral
        _project             raw maps reproducing action kernels [abstract]            stationary, spectral
        _lead_term           FOC term of a lead [NotImplementedError]                  stationary
        _embedded_curvature  optional: curvature of a direction on a longer window     stationary
        interpolate_maps     a coarser result's maps on this grid [NotImplementedError] stationary, spectral
        maps_from_actions    raw maps reproducing action kernels in their world [abstract]  stationary, spectral
        expected_cost        variance part of an agent's cost from Z [abstract]        all three
        _mean_part           means and the mean part of the costs [no-op]              all three
        _diagnostics         checks at the equilibrium [no-op]                         stationary, spectral

    Class attributes the engines set: RESULT (the result class), TOL / DAMPING / MAX_NEWTON (solve()
    defaults), ACTIONS (whether the engine can iterate on action kernels), SECOND_ORDER_QUADRATIC
    (whether the objective is a quadratic form in the strategy at every discount)."""
    model: Model
    c: object                       # the compiled model: .reps (tie representatives), .rep (agent -> representative), .N, .nW
    shapes: Dict[str, Tuple[int, ...]]      # agent -> shape of its raw maps
    RESULT = None                   # the result class
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.5, 60       # solve() defaults; each engine sets its own
    ANDERSON_M = tunable("anderson_m")              # Anderson memory (settings.anderson_m)
    ACTIONS = True                  # whether the engine can iterate on action kernels
    SECOND_ORDER_TOL = tunable("second_order_tol")      # curvature below which a negative value is window truncation (settings)
    SECOND_ORDER_QUADRATIC = True   # whether the objective is a quadratic form in the strategy at every discount
    SECOND_ORDER_DENSE = tunable("second_order_dense")  # strategy dimension up to which the form is built densely (settings)

    def __init__(self, model: Model, verbose: bool = False, settings=None, **options):
        """Hook (every engine overrides it): an engine's __init__ calls this first, with its own
        constructor options as `options`, then builds self.c (its compiled model) and self.shapes
        (agent -> shape of its raw maps: (nU, nR, N), or (nU, nR, N, N) on the cell engine).  The
        base assumes self.c and self.shapes exist after construction, and that solver_kw holds
        exactly the keywords that rebuild an equal engine: type(self)(model, **solver_kw) is how
        coarse_start, the embedded curvature check and a result's refine()/stability() make one.
        settings: a Settings (or a dict of its fields) with the tuning constants, DEFAULT when None;
        self.settings is what the engine and its result read, and solver_kw records the fields that
        differ from the defaults."""
        self.model = model
        self.verbose = verbose
        self.settings = Settings.of(settings)
        changed = self.settings.changed()
        self.solver_kw = {"verbose": verbose, **options, **({"settings": changed} if changed else {})}   # so a result can rebuild the same engine
        self._qa: Dict[str, np.ndarray] = {}                # agent -> (Q zeta) as an operator on the primary kernels
        self._rphys: Dict[str, np.ndarray] = {}             # agent -> physical impulse responses (all reactions off)
        self._second_order_cache: Dict[str, dict] = {}       # representative -> its second-order check, shared with tied agents
        self._loss_forms: Dict[str, np.ndarray] = {}         # agent -> the loss form on the world (map-independent; built for the
                                                            # tie representatives' second-order check in _finish and released there)

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
        """Raw maps of the tie representatives as one vector (hook: the cell engine packs only the
        causal triangle of its (N, N) maps).  Receives a dict with every agent's maps in self.shapes;
        the base assumes pack(unpack(z)) is z and that unpack fills the tied agents (_fill_ties).
        The fixed point, coarse starts and a result's stability() go through these two."""
        return self._pack(maps)

    def unpack(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """The inverse of pack: every agent's raw maps, tied agents copied from their representative."""
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
        """The agent's signal rows in the world Z: regular kernels (one (N, ncol) array per row, as the
        agent sees them, shifted by the observation delay) and the instantaneous entries
        [(channel index, age, weight)] per row.  Controls in `excluded` are switched off (their
        impulses come through impulse channels instead), and only Brownian sources keep an
        instantaneous entry here: an excluded control's own impulse in a row is dropped (in the
        passive world the agent's own control is off).  Reads c.row_blocks when the compiled model
        has it (the nonzero primary blocks only), else c.row."""
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
    def _row_support(self, agent: Agent, rows):
        """Which (row, channel) regular kernels are not identically zero, (nR, nW); the operators of the
        zero ones are zero and are skipped (a channel the agent's rows never carry, a row that reads
        nothing regular).  Rows are grouped by observation delay so one batched kernel call serves all
        rows of a delay."""
        sup = np.stack([np.any(y != 0, axis=0) for y in rows]) if rows else np.zeros((0, self.c.nW), dtype=bool)
        groups = {}
        for r in range(len(rows)):
            groups.setdefault(float(self.c.rows[agent.name][r][3]), []).append(r)
        return sup, groups

    def _row_operator(self, agent: Agent, rows, inst):
        """Per channel, the operator (N x nR N) mapping stacked row maps gamma to the action kernel:
        c_k = sum_r (Conv[y_rk] + E_rk S_delta) gamma_r."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        Gk = np.zeros((nW, N, nR * N))
        sup, groups = self._row_support(agent, rows)
        for d, rs in groups.items():
            pairs = [(r, k) for r in rs for k in np.where(sup[r])[0]]
            if pairs:
                ops = c.conv_rows(np.stack([rows[r][:, k] for r, k in pairs], axis=1), d)     # one call per delay
                for i, (r, k) in enumerate(pairs):
                    Gk[k, :, r * N:(r + 1) * N] = ops[i]
        for r in range(nR):
            for (k, age, w) in inst[r]:
                Gk[k, :, r * N:(r + 1) * N] += w * c.instant(age, c.rows[agent.name][r][3])
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

    def _foc_operators(self, agent: Agent, R: np.ndarray, atoms: bool = False):
        """Per control, the operator (N x n_prim N) mapping the primary kernels of one channel to the
        first-order-condition kernel: instantaneous derivative, discounted continuation through the
        impulse responses R, delayed reads of own lagged controls, and the past-date term of a lead.
        Called with the physical impulse responses (all reactions off) for the wedge decomposition.
        With atoms=True returns (Fu, Ms), Ms the per-control operators M (n_atoms, N, N) on the loss
        atoms' kernels that Fu contracts with Q (the mean part applies them to the targets q)."""
        c = self.c; N = c.N; n_prim = len(c.prim) * N
        atms, Q, q = c.loss[agent.name]
        AO = [c.atom_op(at) for at in atms]
        # the nonzero N x N blocks of every atom operator (one block, the shift into the atom's primary)
        AO_blocks = [[(p, A[:, p * N:(p + 1) * N]) for p in range(len(c.prim)) if np.any(A[:, p * N:(p + 1) * N])] for A in AO]
        AO_eye = [[_is_eye(blk) for p, blk in blocks] for blocks in AO_blocks]      # an undelayed atom reads through the identity
        Fu, Ms = [], []
        for ui, u in enumerate(agent.controls):
            # op = sum_j M_j (Q zeta)_j with M_j the operator on atom j: identity for the instantaneous
            # term, continuation, delayed own read, lead term; contracted as sum_i (sum_j Q_ji M_j) AO_i
            # so the products are N x N x N per atom block instead of N x N x n_prim N per atom
            M = np.zeros((len(atms), N, N))
            if (u, 0.0) in atms:
                M[atms.index((u, 0.0))] += np.eye(N)
            if not agent.myopic:
                Rj = np.stack([AO[j] @ R[:, ui] for j in range(len(atms))], axis=1)   # impulse responses of every atom
                CR = c.continuation(Rj)
                for j, (name, lag) in enumerate(atms):
                    if name in agent.controls:
                        if name == u and lag > 0:          # delayed read of the control itself
                            M[j] += np.exp(-c.rho * lag) * c.own_lag_read(lag)
                        continue                            # own reactions: envelope
                    M[j] += CR[j]
                    if lag < 0:
                        M[j] += self._lead_term(agent, R[:, ui], name, lag)
            MQ = np.tensordot(Q.T, M, axes=1)              # MQ[i] = sum_j Q[j, i] M_j
            op = np.zeros((N, n_prim))
            for i in range(len(atms)):
                if np.any(MQ[i]):
                    for (p, blk), eye in zip(AO_blocks[i], AO_eye[i]):
                        op[:, p * N:(p + 1) * N] += MQ[i] if eye else MQ[i] @ blk
            Fu.append(op); Ms.append(M)
        return (Fu, Ms) if atoms else Fu

    def _lead_term(self, agent: Agent, Ru: np.ndarray, name: str, lag: float) -> np.ndarray:
        """Hook (stationary only): the FOC term of a lead (name@lag, lag < 0) from the flows before t
        that read the quantity after t.  Receives the impulse responses Ru (n_prim N,) of the
        primaries to one of the agent's controls and the led primary's name; must return an (N, N)
        operator on that atom's (Q zeta) kernel, added to the atom's continuation.  Called only when
        a loss atom has a negative lag, which the finite engines reject at compile time
        (reject_leads), so their NotImplementedError is never reached."""
        raise NotImplementedError("leads are supported by the stationary engine only")

    def _projection_operator(self, agent: Agent, rows, inst):
        """H (nR N x nW N): E[phi_t dY_r(t - b)] for every row r and lag b, from the FOC kernels."""
        c = self.c; N, nW = c.N, c.nW; nR = len(rows)
        H = np.zeros((nR * N, nW * N))
        for r in range(nR):
            H[r * N:(r + 1) * N] += c.projection_rows(rows[r], c.rows[agent.name][r][3])
            for (k, age, w) in inst[r]:
                H[r * N:(r + 1) * N, k * N:(k + 1) * N] += w * c.instant_adjoint(age, c.rows[agent.name][r][3])
        return H

    # ----------------------------------------------------- best response
    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """Hook (stationary: naive observers): the responses R (n_prim N, nU) of the primary kernels
        to a unit impulse of each of the agent's controls, with the agent's own reaction switched
        off and every other agent reacting through `maps`.  Receives the columns closed_loop
        returned for `impulse_controls=agent.controls`; must return an array of the same shape.
        The base uses the result for the response operators, the continuation in the FOC and the
        mean systems; the wedge decomposition uses the physical responses (all maps zero) instead
        and does not go through this hook.  The default returns R unchanged."""
        return R

    def _passive_world(self, agent: Agent, maps, Zpass: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Hook (no engine overrides it): the agent's passive world Zpass (n_prim N, nW), its own
        strategy off, before the passive rows are read from it.  Must return the same shape."""
        return Zpass

    def _solve_foc(self, agent: Agent, Amat: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        """Hook (stationary, spectral): gamma solving Amat gamma = -bvec with the engine's own
        regularisation.  Receives the FOC system of _foc_system on every map node, (nG, nG) and
        (nG,) with nG = nU nR N in (control, row, node) order, including the exactly zero rows and
        columns of the nodes _identified masks out; must return gamma (nG,) in the same order, zero
        at the masked nodes (best_response reshapes it to (nU, nR, N)).  The base assumes a
        singular system raises ValueError with singular_system_message(agent.name) (the stationary
        engine on np.linalg.solve's exact test, the spectral engine on _solve_regular's condition
        estimate), which solve_fixed_point passes through the Newton polish unchanged."""
        raise NotImplementedError

    FOC_RCOND = tunable("foc_rcond")    # a best-response system whose reciprocal condition estimate is below this is singular (settings)

    def _solve_regular(self, agent: Agent, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """x solving A x = b for the best-response system of `agent` on its kept unknowns, refusing a
        singular one: A is LU-factored, the reciprocal of its 1-norm condition number is estimated
        from the factors (LAPACK gecon, O(n^2) after the factorisation), and an estimate below
        FOC_RCOND raises the ValueError naming the usual causes.  Singular systems sit at 1e-16 or
        exactly 0; the worst regular finite-engine system in the tests sits at 2.6e-3.  The finite
        engines use this (a least-squares fallback used to return a zero strategy as converged); the
        stationary engine keeps the exact test of np.linalg.solve, because a deliberately non-convex
        model in the tests solves a system at 1e-23 whose strategy its second-order check must flag."""
        if A.shape[0] == 0:
            return np.zeros(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sla.LinAlgWarning)               # an exactly zero pivot: raised below
            lu, piv = sla.lu_factor(A, check_finite=False)
        gecon, = sla.get_lapack_funcs(("gecon",), (lu,))
        rcond = float(gecon(lu, np.linalg.norm(A, 1))[0])
        if not rcond > self.FOC_RCOND:
            raise ValueError(singular_system_message(agent.name) + f" (reciprocal condition estimate {rcond:.1e})")
        return sla.lu_solve((lu, piv), b, check_finite=False)

    def _project(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Hook (stationary, spectral): raw maps (nU, nR, N) reproducing the action kernels `actions`
        (nU, N, nW) on the agent's closed-loop rows, read from the full world Zfull (n_prim N, nW)
        in which the agent's own strategy is on.  Both engines solve a weighted least-squares
        projection (a Gram over the seen rows, with a small ridge); the base only requires the
        shape, zero where _identified masks a node, and takes the result as the best-response map
        the fixed point iterates on."""
        raise NotImplementedError

    def _decompose(self, agent: Agent, out: dict, Fu, Resp, Gk, maps=None) -> None:
        """The second-order check and the FOC decomposition (instantaneous/physical/wedge).  Tied agents
        share the second-order check of their representative (the same problem up to relabelling)."""
        c = self.c; nW = c.nW; Zfull = out["Zfull"]
        rep = c.rep[agent.name]
        if rep != agent.name and rep in self._second_order_cache:
            out["second_order"] = self._second_order_cache[rep]
        else:
            out["second_order"] = self._second_order(agent, Resp, Gk, np.tile(self._identified(agent), len(agent.controls)), maps)
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
        """Hook (stationary, spectral): boolean mask (nR N,) in (row, node) order of the map nodes at
        which the map on each row reads something within the window; the default keeps all of
        them.  The base assumes a masked node has exactly zero rows and columns in the FOC system
        (a row observed with a delay is uninformative there), so the engines' _solve_foc and
        _project solve on the kept nodes only and leave the map zero elsewhere; the second-order
        check restricts its quadratic form to the kept nodes (tiled over the controls).  Must be
        the same for tied agents up to relabelling, since they share the representative's check."""
        return np.ones(len(agent.signals) * self.c.N, dtype=bool)

    def _second_order(self, agent: Agent, Resp, Gk, keep, maps=None) -> Optional[dict]:
        """Second-order condition of the best response: the agent's objective is a quadratic form in its
        strategy, and a first-order condition is a minimum only if that form is positive on the
        feasible strategies (those its rows can express).  The form is computed exactly from the
        cost's own Gram matrix: J(delta) = 1/2 delta' M delta with M = T' G T, T the map from a
        strategy to the world it produces and G the loss form.  Its extreme eigenvalues come from
        Lanczos on matvecs.  Returns {"min", "max", "ok", "converged"} with min/max the eigenvalues
        of M scaled by max; None when the objective is not a quadratic form in the strategy (the
        stationary engine with rho > 0: the discounted objective is not one in the stationary kernel).
        The objective is truncated at the window, so a strategy can push a little loss past the edge:
        curvatures within SECOND_ORDER_TOL of the largest are treated as that, not as a saddle.
        When the form is not positive and the engine defines _embedded_curvature(agent, maps, idx,
        vmin) (an optional hook, the stationary engine's), that is asked for the curvature of the
        offending direction on a longer window: a float, or None when it cannot say; a positive
        value turns the verdict into ok with "edge" and "embedded" recorded."""
        c = self.c
        if not (self.SECOND_ORDER_QUADRATIC or c.rho == 0):
            return None
        from scipy.sparse.linalg import LinearOperator, eigsh
        N, nW = c.N, c.nW; nR, nU = len(agent.signals), len(agent.controls)
        GAO = self._loss_form(agent)                                             # symmetric loss form on the world
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
            # the form explicitly, M = sum_k T_k' G T_k with T_k = [Resp_u G_k]_u, associated as
            # M[u, v] = sum_k G_k' (Resp_u' G Resp_v) G_k: the inner form H_uv is N x N (through the responding
            # primaries' nodes only), and the sum over channels is one product of the stacked row operators,
            # restricted per channel to the rows whose column block of G_k is not identically zero
            nz = np.where(np.any(np.stack([np.any(Rv != 0, axis=1) for Rv in Resp]), axis=0))[0]   # primaries' nodes that respond
            GAOnz = GAO[np.ix_(nz, nz)]
            Rnz = [np.ascontiguousarray(Rv[nz]) for Rv in Resp]                                    # (nz, N)
            GR = [GAOnz @ Rv for Rv in Rnz]
            rowsof = {}                                                                             # rows with a nonzero block -> channels
            for k in range(nW):
                rows_k = tuple(r for r in range(nR) if np.any(Gk[k][:, r * N:(r + 1) * N]))
                if rows_k:
                    rowsof.setdefault(rows_k, []).append(k)
            parts = []
            for rows_k, ks in rowsof.items():
                cols = np.concatenate([np.arange(r * N, (r + 1) * N) for r in rows_k])
                parts.append((cols, np.ascontiguousarray(Gk[ks][:, :, cols])))                     # (n_k, N, |cols|)
            Mall = np.zeros((nU * nR * N, nU * nR * N))
            for ui in range(nU):
                for vi in range(ui, nU):
                    Huv = Rnz[ui].T @ GR[vi]                                                        # (N, N)
                    Muv = np.zeros((nR * N, nR * N))
                    for cols, Gg in parts:
                        HG = Huv @ Gg                                                               # every channel of the group
                        Muv[np.ix_(cols, cols)] += Gg.reshape(-1, cols.size).T @ HG.reshape(-1, cols.size)
                    Mall[ui * nR * N:(ui + 1) * nR * N, vi * nR * N:(vi + 1) * nR * N] = Muv
                    if vi != ui:
                        Mall[vi * nR * N:(vi + 1) * nR * N, ui * nR * N:(ui + 1) * nR * N] = Muv.T   # H_vu = H_uv'
            Mfull = Mall if n == Mall.shape[0] else Mall[np.ix_(idx, idx)]
            w, V = np.linalg.eigh((Mfull + Mfull.T) / 2)
            lo, hi = float(w[0]), float(w[-1]); vmin = V[:, 0]
        else:
            vmin = None
            op = LinearOperator((n, n), matvec=matvec, dtype=float)
            ltol, lmax = self.settings.second_order_lanczos_tol, self.settings.second_order_lanczos_maxiter
            try:
                hi = float(eigsh(op, k=1, which="LA", tol=ltol, maxiter=lmax, return_eigenvectors=False)[0])
                lo = float(eigsh(op, k=1, which="SA", tol=ltol, maxiter=lmax, return_eigenvectors=False)[0])
            except Exception as exc:                          # Lanczos did not settle: say so rather than stay silent
                return {"min": None, "max": None, "ok": None, "converged": False, "message": f"{type(exc).__name__}: {exc}"[:120]}
        scale = max(abs(lo), abs(hi), 1e-300)
        out = {"min": lo / scale, "max": hi / scale, "ok": bool(lo >= -self.SECOND_ORDER_TOL * scale), "converged": True}
        if not out["ok"] and vmin is not None and maps is not None and hasattr(self, "_embedded_curvature"):
            # the windowed objective omits the flows past the edge that read the strategy within the last lag:
            # a negative direction is a truncation artefact if the same direction, zero-extended onto a window
            # longer by two lags, has positive curvature under the same maps
            emb = self._embedded_curvature(agent, maps, idx, vmin)
            if emb is not None:
                out["embedded"] = float(emb / scale)
                out["edge"] = bool(emb >= 0.0)
                out["ok"] = out["edge"]
        return out

    def _loss_form(self, agent: Agent) -> np.ndarray:
        """The loss form on the primary kernels, AO' kron(Q, mass) AO for the stacked atom operators AO,
        assembled block by block over the atoms' primary blocks (each atom reads one primary through one
        N x N block; an undelayed atom through the identity, whose products are skipped).  Map-independent,
        cached per agent."""
        if agent.name not in self._loss_forms:
            c = self.c; N = c.N; n = len(c.prim) * N
            atoms, Q, q = c.loss[agent.name]
            mass = c.cost_mass()
            AO = [c.atom_op(at) for at in atoms]
            blocks = [[(p, A[:, p * N:(p + 1) * N]) for p in range(len(c.prim)) if np.any(A[:, p * N:(p + 1) * N])] for A in AO]
            GAO = np.zeros((n, n))
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    if Q[i, j] == 0.0:
                        continue
                    W = Q[i, j] * mass
                    for p, Ai in blocks[i]:
                        for p2, Aj in blocks[j]:
                            WA = W if _is_eye(Aj) else W @ Aj
                            GAO[p * N:(p + 1) * N, p2 * N:(p2 + 1) * N] += WA if _is_eye(Ai) else Ai.T @ WA
            self._loss_forms[agent.name] = GAO
        return self._loss_forms[agent.name]

    # ------------------------------------------------ maps from kernels

    def _causal_chunks(self):
        """Node ranges [(lo, hi)] in increasing age such that the regular projection operator of any row is
        zero from ages in one chunk to nodes in an earlier one (a correlation reads only older ages); the
        products over those blocks are skipped.  One chunk when the compiled model declares none
        (c.causal_chunks is optional: the stationary Compiled has it, the triangle does not)."""
        ch = getattr(self.c, "causal_chunks", None)
        return ch() if ch is not None else [(0, self.c.N)]

    def _foc_system(self, agent: Agent, rows, inst, Zpass: np.ndarray, Resp, Fu):
        """The first-order-condition system Amat gamma = -bvec on the passive rows,
        Amat[u, v] = sum_k H_k (Fu_u Resp_v) G_k, bvec[u] = sum_k H_k (Fu_u Zpass)_k, with G_k the row
        operator and H_k the projection operator of channel k.  Both split into a regular part (the
        convolution and correlation with the row kernels, zero for every (row, channel) whose kernel is
        zero) and the instantaneous entries (scaled shifts on one row block); the regular parts are
        assembled over the nonzero rows and channels only, with the projection's columns ordered
        (node, channel) so the product Fu Resp G comes out in the right layout without a transpose, and
        the instantaneous terms are added block by block.  Identical to the dense assembly to round-off."""
        c = self.c; N = c.N; nR, nU = len(rows), len(Fu)
        sup, groups = self._row_support(agent, rows)
        delay = [c.rows[agent.name][r][3] for r in range(nR)]
        Rn = np.where(sup.any(axis=1))[0]; Kn = np.where(sup.any(axis=0))[0]
        nRn, nKn = len(Rn), len(Kn)
        kpos = {int(k): i for i, k in enumerate(Kn)}
        # regular parts: Hs[(ri, a), (j, ki)] and Gs[a, (ki, ri, j)]
        Hs = np.zeros((nRn, N, N, nKn)); Gs = np.zeros((N, nKn, nRn, N))
        for d, rs in groups.items():
            pairs = [(ri, int(k)) for ri, r in enumerate(Rn) if r in rs for k in np.where(sup[r])[0]]
            if not pairs:
                continue
            Y = np.stack([rows[Rn[ri]][:, k] for ri, k in pairs], axis=1)
            Hp = c.projection_rows(Y, d).reshape(N, len(pairs), N)             # (a, pair, j)
            Gp = c.conv_rows(Y, d)                                              # (pair, a, j)
            for i, (ri, k) in enumerate(pairs):
                Hs[ri, :, :, kpos[k]] = Hp[:, i, :]
                Gs[:, kpos[k], ri, :] = Gp[i]
        Hs = Hs.reshape(nRn * N, N * nKn); Gs = Gs.reshape(N, nKn * nRn * N)
        chunks = self._causal_chunks() if nKn else []
        Hc = [np.ascontiguousarray(Hs.reshape(nRn, N, N * nKn)[:, lo:hi, lo * nKn:]).reshape(nRn * (hi - lo), (N - lo) * nKn)
              for lo, hi in chunks]                                             # rows of ages in the chunk, columns of nodes not younger
        # instantaneous entries: (row, channel, weighted shift on the row operator, on the projection); a shift that is the
        # identity (an undelayed row's own noise) is applied as the scaling by its weight, which is what the product gives
        ent = [(r, k, w * c.instant(age, delay[r]), w * c.instant_adjoint(age, delay[r])) for r in range(nR) for (k, age, w) in inst[r]]
        ent = [(r, k, Sg, Sh, (w if _is_eye(c.instant(age, delay[r])) else None), (w if _is_eye(c.instant_adjoint(age, delay[r])) else None))
               for (r, k, Sg, Sh), (r_, k_, age, w) in zip(ent, [(r, k, age, w) for r in range(nR) for (k, age, w) in inst[r]])]
        rmul = lambda X, S, w: X * w if w is not None else X @ S            # X @ S with S = w I
        lmul = lambda S, w, X: w * X if w is not None else S @ X            # S @ X with S = w I
        nG = nU * nR * N
        Amat = np.zeros((nG, nG)); bvec = np.zeros(nG)
        A6 = Amat.reshape(nU, nR, N, nU, nR, N); B3 = bvec.reshape(nU, nR, N)
        for ui in range(nU):
            phi = Fu[ui] @ Zpass                                                # (N, nW): the FOC of the passive world
            if nKn:
                B3[ui][Rn] += (Hs @ phi[:, Kn].reshape(-1)).reshape(nRn, N)
            for (r, k, Sg, Sh, wg, wh) in ent:
                B3[ui, r] += lmul(Sh, wh, phi[:, k])
            for vi in range(nU):
                FR = Fu[ui] @ Resp[vi]
                FRG = (FR @ Gs).reshape(N, nKn, nRn * N) if nKn else None      # [(j, ki), (ri, j')]
                for (lo, hi), H_ in zip(chunks, Hc):
                    T = (H_ @ FRG[lo:].reshape((N - lo) * nKn, nRn * N)).reshape(nRn, hi - lo, nRn, N)
                    for ri, r in enumerate(Rn):
                        A6[ui, r, lo:hi, vi][:, Rn, :] += T[ri]
                for (r, k, Sg, Sh, wg, wh) in ent:
                    if k in kpos:                                               # the channel also has regular kernels
                        ki = kpos[k]
                        X = (Hs.reshape(nRn * N, N, nKn)[:, :, ki] @ rmul(FR, Sg, wg)).reshape(nRn, N, N)     # H_reg FR G_inst
                        for ri, rr in enumerate(Rn):
                            A6[ui, rr, :, vi, r] += X[ri]
                        A6[ui, r, :, vi][:, Rn, :] += lmul(Sh, wh, FRG[:, ki, :]).reshape(N, nRn, N)        # H_inst FR G_reg
                    for (r2, k2, Sg2, Sh2, wg2, wh2) in ent:
                        if k2 == k:
                            A6[ui, r, :, vi, r2] += lmul(Sh, wh, rmul(FR, Sg2, wg2))                       # H_inst FR G_inst
        return Amat, bvec

    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False):
        """The agent's best response to `maps`: (raw map, {"gamma", "action", "Zfull", ...}).  The
        agent's information is the passive signal history, so its first-order condition is affine
        in its map on the passive rows: one linear solve.
        Hook (the cell engine overrides it wholesale, with the signature (agent, maps)).  Receives
        every agent's raw maps in self.shapes; must return the agent's raw map (its shape in
        self.shapes) and a dict with "gamma" (the FOC unknown), "action" (the action kernels,
        (nU, N, nW), what response_actions iterates on) and "Zfull" (the world with the response
        in).  With want_decomp=True the dict also carries "second_order" (the check, or None) and
        "decomp" (control -> {"foc", "physical", "wedge"} kernels (N, nW)), which _diagnostics
        reads; an engine without them must not call the base _diagnostics.  A singular system
        raises the ValueError of singular_system_message."""
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
        # the FOC is affine in gamma: solve H (Fu (Zpass + sum_v Resp_v Gk gamma_v)) = 0 for all controls
        Amat, bvec = self._foc_system(agent, ytil, yinst, Zpass, Resp, Fu)
        gamma = self._solve_foc(agent, Amat, bvec).reshape(nU, nR, N)
        cact = np.stack([(Gk @ gamma[ui].reshape(-1)).T for ui in range(nU)])
        Zfull = Zpass.copy()
        for ui in range(nU):
            Zfull += Resp[ui] @ cact[ui]
        out = {"gamma": gamma, "action": cact, "Zfull": Zfull}
        if want_decomp:
            self._decompose(agent, out, Fu, Resp, Gk, maps)
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
    def interpolate_maps(self, coarse) -> Dict[str, np.ndarray]:
        """Hook (stationary, spectral): the raw maps of `coarse`, a result of the same engine on a
        grid of the same model with fewer nodes (coarse.maps, coarse.compiled), read at this grid's
        nodes: every agent's maps in self.shapes.  coarse_start (solve(start="coarse")) and a
        result's refine() call it; refine() treats NotImplementedError as "no warm start" and
        starts from zero, coarse_start does not catch it (the cell engine, which does not override
        it, has no coarse start)."""
        raise NotImplementedError

    def coarse_start(self, factor: float = 0.5, **solve_kw) -> Dict[str, np.ndarray]:
        """Raw maps to start from: the equilibrium at `factor` times the nodes, interpolated to this
        grid.  A coarse solve costs a few fine evaluations and usually saves many."""
        hz = self.model.horizon
        n0 = max(4, int(round(hz.nodes * factor)))
        if n0 >= hz.nodes:
            return None
        coarse = type(self)(self.model.with_horizon(nodes=n0), **self.solver_kw)
        res = coarse.solve(**{k: v for k, v in solve_kw.items() if k not in ("init", "start")})
        self._coarse_evals, self._coarse_nodes = res.iterations, n0
        return self.interpolate_maps(res)

    def solve(self, init: Optional[Dict[str, np.ndarray]] = None, tol: Optional[float] = None, damping: Optional[float] = None,
              max_newton: Optional[int] = None, variable: str = "actions", start: str = "zero", max_evaluations: Optional[int] = None,
              deadline: Optional[float] = None, progress: Optional[Callable[[dict], None]] = None, diagnostics: bool = True):
        """Find the equilibrium: Anderson mixing on the fixed point of the best-response map (damping is
        the mixing weight), then a Newton-Krylov polish if it stalls.  variable="actions" iterates on
        the agents' action kernels, with the raw maps recovered by projection (better conditioned where
        strategies are weakly identified); "maps" iterates on the raw maps, which is also what happens
        with ties (tied agents' action kernels differ by a channel permutation) and on the cell engine.
        init: action kernels or raw maps per agent, either is accepted.  start="coarse" (with no init)
        solves first at half the nodes and starts from that equilibrium interpolated to this grid.
        max_evaluations bounds the best-response evaluations (Anderson mixing and the polish together, the
        count res.iterations reports) and deadline the wall time of the solve in seconds; at least one
        evaluation is made, and past either bound the best iterate so far is returned with converged=False
        and res.message naming the bound (no exception; check() raises).  A coarse start is bounded the same
        way.  progress(info) is called after every evaluation with {"evaluation": the count so far,
        "residual": the relative residual, "phase": "anderson" or "newton" (prefixed "coarse " during a coarse
        start), "seconds": since the solve began}; an exception it raises propagates, which is how a solve is
        cancelled.  diagnostics=False skips the checks at the end (the first-order-condition decomposition,
        the second-order check and the representation error: res.foc and res.second_order stay empty,
        res.resolution_ok is None and the summary says so) and fills the costs only, for a preview.
        The options as given are recorded in res.solve_kw (the bounds and diagnostics=False when given; the
        progress callable is not, nor is init), so solve(**res.solve_kw) repeats a solve that was not
        warm-started (a sweep row after the first, or refine(), was: its record starts from zero)."""
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(f"max_evaluations must be at least 1, not {max_evaluations}")
        if deadline is not None and deadline < 0:
            raise ValueError(f"deadline must be a number of seconds, not {deadline}")
        tol = self.TOL if tol is None else tol
        damping = self.DAMPING if damping is None else damping
        max_newton = self.MAX_NEWTON if max_newton is None else max_newton
        t0 = time.time(); evals = [0]
        coarse_evals = 0
        if init is None and start == "coarse":
            # the coarse solve's checks are never read; it gets the same bounds (on its own count, on this clock)
            init = self.coarse_start(tol=tol, damping=damping, max_newton=max_newton, variable=variable, diagnostics=False,
                                     max_evaluations=max_evaluations, deadline=deadline, progress=None if progress is None else
                                     (lambda info: progress({**info, "phase": "coarse " + info["phase"], "seconds": time.time() - t0})))
            coarse_evals = getattr(self, "_coarse_evals", 0)
        if variable == "actions" and (self.model.ties or not self.ACTIONS):
            variable = "maps"
        kind = self.init_kind(init) if init is not None else None
        if kind == "actions" and not self.ACTIONS:
            raise ValueError("this engine iterates on raw maps: pass raw maps as init")
        if variable == "actions":
            if kind is None:
                x0 = {a.name: np.zeros(self.action_shapes[a.name]) for a in self.model.agents}
            else:
                x0 = init if kind == "actions" else self.actions_from_maps(init)
            pack, unpack, respond = self.pack_actions, self.unpack_actions, self.response_actions
        else:
            if kind is None:
                x0 = self.zero_maps()
            else:
                x0 = init if kind == "maps" else self.maps_from_actions(init)
            pack, unpack, respond = self.pack, self.unpack, self.response_map

        def F(zz):
            evals[0] += 1
            return pack(respond(unpack(zz))) - zz
        st = self.settings
        z, resid, _, converged, message = solve_fixed_point(F, pack(x0), tol=tol, verbose=self.verbose, damping=damping,
                                                            anderson_iters=st.anderson_iters, max_newton=max_newton, M=self.ANDERSON_M,
                                                            reg=st.anderson_reg, inner_m=st.newton_inner_m, max_evaluations=max_evaluations,
                                                            deadline=deadline, progress=progress, t0=t0)
        maps = self.maps_from_actions(unpack(z)) if variable == "actions" else unpack(z)
        Z = self.c.closed_loop(maps)
        if coarse_evals:
            message = f"coarse start: {coarse_evals} evaluations at {self._coarse_nodes} nodes; " + message
        res = self.RESULT(model=self.model, compiled=self.c, maps=maps, Z=Z, converged=converged, residual=resid,
                          iterations=evals[0], seconds=0.0, message=message, solver_class=type(self),
                          solver_kw=self.solver_kw, settings=self.settings,
                          solve_kw={"tol": tol, "damping": damping, "max_newton": max_newton, "variable": variable, "start": start,
                                    **{k: v for k, v in (("max_evaluations", max_evaluations), ("deadline", deadline)) if v is not None},
                                    **({} if diagnostics else {"diagnostics": False})})
        self._finish(res)
        res.seconds = time.time() - t0            # the diagnostics of _finish are part of the solve's time
        return res

    def _finish(self, res) -> None:
        """Fill the engine's own outputs on a fresh result: the costs (their variance part), the means and the
        mean part of the costs, then, unless the result's options say diagnostics=False, the engine's checks at
        the equilibrium.  The order is the contract: expected_cost first, _mean_part reads res.costs as the
        variance part and adds the mean part, _diagnostics last."""
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, res.Z)
        self._mean_part(res)
        if res.solve_kw.get("diagnostics", True) is not False:
            self._diagnostics(res)

    def _mean_part(self, res) -> None:
        """Hook (every engine overrides it): the means (targets, constant drifts, initial states) and the
        mean part of every cost.  Receives the result with res.maps, res.Z and res.costs already holding the
        variance part of every agent's cost; must fill res.means (name -> a float on the stationary engine, a
        path over res.means_t on the finite engines, for every primary, definition and "agent.row" drift
        rate), res.cost_parts[agent] = {"variance", "mean"} and add the mean part to res.costs[agent].  Part
        of the answer, not a check: it runs with diagnostics=False too.  The default does nothing."""

    def _diagnostics(self, res) -> None:
        """Hook (stationary, spectral; the cell engine keeps the no-op): the checks at the equilibrium,
        filled on the result: res.foc[agent] (the "decomp" of best_response), res.second_order[agent]
        (when the check applies) and res.representation_error[agent].  Runs after _mean_part, skipped
        when the solve was made with diagnostics=False; a result without these has resolution_ok None."""

    def actions_from_maps(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Action kernels the raw maps produce in their own closed loop."""
        Z = self.c.closed_loop(maps)
        return {a.name: np.stack([Z[self.c.block(u)] for u in a.controls]) for a in self.model.agents}

    def expected_cost(self, agent: Agent, Z: np.ndarray) -> float:
        """Hook (every engine overrides it): the variance part of the agent's cost in the world Z, exactly
        what c.closed_loop(maps) returns (so the cell engine receives its (n_prim, N, nW N) layout): the
        stationary flow loss per unit time, or the discounted integral over [0, T], of 1/2 z'Qz over its
        loss atoms, from the shocks alone.  Must return a float; _finish stores it in res.costs before the
        mean part is added."""
        raise NotImplementedError

    def maps_from_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Hook (stationary, spectral; the cell engine iterates on maps only, ACTIONS = False): raw maps
        that reproduce the given action kernels (every agent, (nU, N, nW)) in the world they generate.
        The base uses it to close an iteration on action kernels (the states follow from the actions
        without any map) and to convert an action-kernel warm start; must return every agent's maps in
        self.shapes, tied agents equal."""
        raise NotImplementedError

    def response_map(self, maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Every agent's best-response map against `maps` (the map-iteration fixed-point function)."""
        return self._over_representatives(lambda a: self.best_response(a, maps)[0])

    def response_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Every agent's best-response action kernels against `actions`."""
        maps = self.maps_from_actions(actions)
        return self._over_representatives(lambda a: self.best_response(a, maps)[1]["action"])
