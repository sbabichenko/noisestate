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
from .means import MeanLayer
from ._settings import Settings, tunable
from .spec import Agent, Model


def _is_eye(M: np.ndarray) -> bool:
    """Whether M is exactly the identity (a product with it is then skipped: X @ I is X to the bit)."""
    n = M.shape[0]
    return M.ndim == 2 and M.shape[1] == n and np.count_nonzero(M) == n and bool(np.all(np.diagonal(M) == 1.0))


def dense_curvature_form(Resp, Gk, forms, nU: int, nR: int, Nm: int, idx) -> np.ndarray:
    """The second-order form on the kept strategies `idx`, built explicitly: (len(idx), len(idx)).

    M = sum_k T_k' G_(k) T_k with T_k = [Resp_u G_k]_u and G_(k) the column's loss form, associated
    as M[u, v] = sum_k G_k' (Resp_u' G_(k) Resp_v) G_k: the inner form H_uv is N x N (through the
    responding primaries' nodes only), and the sum over the columns of one loss form is one product
    of the stacked row operators, restricted per column to the rows whose block of G_k is not
    identically zero.

    The two engines reach this with the same three operands and built it in the same twenty-five
    lines each: Engine._second_order from its own structures, finite_free._dense_form from the
    matrix-free operators' dense rows.  Only the assembly is shared -- how Resp, Gk and forms are
    obtained is exactly what differs between them, and stays where it is.
    """
    nz = np.where(np.any(np.stack([np.any(Rv != 0, axis=1) for Rv in Resp]), axis=0))[0]   # primaries' nodes that respond
    Rnz = [np.ascontiguousarray(Rv[nz]) for Rv in Resp]                                    # (nz, N)
    groups = []                                                                             # per loss form: G Resp_v and its column groups
    for G, sl in forms:
        GR = [G[np.ix_(nz, nz)] @ Rv for Rv in Rnz]
        rowsof = {}                                                                         # rows with a nonzero block -> columns
        for k in range(sl.start, sl.stop):
            rows_k = tuple(r for r in range(nR) if np.any(Gk[k][:, r * Nm:(r + 1) * Nm]))
            if rows_k:
                rowsof.setdefault(rows_k, []).append(k)
        parts = []
        for rows_k, ks in rowsof.items():
            cols = np.concatenate([np.arange(r * Nm, (r + 1) * Nm) for r in rows_k])
            parts.append((cols, np.ascontiguousarray(Gk[ks][:, :, cols])))                 # (n_k, N, |cols|)
        groups.append((GR, parts))
    nG = nU * nR * Nm
    Mall = np.zeros((nG, nG))
    for ui in range(nU):
        for vi in range(ui, nU):
            Muv = np.zeros((nR * Nm, nR * Nm))
            for GR, parts in groups:
                Huv = Rnz[ui].T @ GR[vi]                                                    # (N, N)
                for cols, Gg in parts:
                    HG = Huv @ Gg                                                           # every column of the group
                    Muv[np.ix_(cols, cols)] += Gg.reshape(-1, cols.size).T @ HG.reshape(-1, cols.size)
            Mall[ui * nR * Nm:(ui + 1) * nR * Nm, vi * nR * Nm:(vi + 1) * nR * Nm] = Muv
            if vi != ui:
                Mall[vi * nR * Nm:(vi + 1) * nR * Nm, ui * nR * Nm:(ui + 1) * nR * Nm] = Muv.T   # H_vu = H_uv'
    return Mall if idx.size == nG else Mall[np.ix_(idx, idx)]


def symmetrize(M: np.ndarray, block: int = 256) -> np.ndarray:
    """M replaced in place by (M + M')/2, bit for bit the same entries, without the two full temporaries of the
    expression: the lower triangle one row block at a time (the columns a block reads above its own rows are
    not yet written), then the upper from it."""
    n = M.shape[0]
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        M[i0:i1, :i1] = (M[i0:i1, :i1] + M[:i1, i0:i1].T) / 2
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        M[:i0, i0:i1] = M[i0:i1, :i0].T
    return M


def singular_system_message(name: str) -> str:
    """The error every engine raises on a singular best-response system, naming its usual causes."""
    return (f"the best-response system of {name} is singular: two of its rows may carry the same information, a "
            "control may have no quadratic term in its current value (a quadratic in a lagged read, D@tau, leaves "
            "the strategy free within tau of the window's edge), a row's noise loading may be zero, or the system "
            "may be too ill-conditioned at this resolution (a very small control penalty, a long window)")


class EngineBase(MeanLayer):
    """What every engine shares: the packing of the tie representatives, the passive world and rows, the
    outer fixed point, the result, the diagnostics loop, the second-order Lanczos and the mean layer,
    written against a small kernel algebra that each compiled model supplies and a set of hooks the
    engines fill in.  The best response itself is each engine's: the stationary engine's on its kernel
    algebra (stationary.py), the spectral engine's on its operators (finite_free.py), the cell engine's.

    Layout.  A kernel is a nodal vector over the engine's N nodes (shock ages on the stationary
    grid, (time, age) nodes on the triangle).  The world Z is (n_prim N, ncol): rows in
    (primary, node) order, c.block(name) slicing a primary's N nodes; columns the nW Brownian
    channels, then one unit-impulse column per control in `impulse_controls`.  An agent's raw
    maps are (nU, nR, N) (control, signal row, node of the row as the agent sees it: a delayed
    row's map is stored at the shifted time or age); its action kernels are (nU, N, nW); the
    first-order-condition unknown gamma is the maps flattened in (control, row, node) order, and
    a mask over map nodes is (nR N,) in (row, node) order.  The cell engine has its own layouts
    (Z (n_prim, N, ncol) per cell, maps (nU, nR, N, N)); it uses the packing, the fixed point,
    _finish and the mean layer only.  The spectral
    finite engine (finite_spectral.py) overrides best_response, _seen_rows, _representation_error and
    expected_cost with the operator form of finite_free.py on spectral_operators.py (the same pieces as
    applications of the line paths and sparse reads of its compiled model, spectral_compiled.py, whose
    closed loop is closed_loop.py's and whose means are spectral_means.py's), so of the kernel algebra
    below it supplies closed_loop, block and atom_op only.  docs/architecture.md draws the modules.

    Kernel algebra of the compiled model self.c: the interface algebra.KernelAlgebra (closed_loop, block,
    atom_op, expr_op, expr_kernel, row_blocks, conv_rows, instant, instant_adjoint, response, continuation,
    own_lag_read, projection_rows, cost_mass, causal_chunks) with the shapes in its docstrings, and its
    attributes N, nW, prim, index, channels, rows (agent -> [(name, drift, E, delay)]), loss (agent -> (atoms,
    Q, q)), rep, reps, rho, model, and grid / g / h (same_grid).  Each engine's compiled model implements
    what its engine calls and raises NotImplementedError, naming the member, for the rest.

    Hooks: what an engine overrides (the base's default in brackets) and which engines do:

        hook                 purpose                                                   overridden by
        __init__             build self.c, self.shapes; record the options in solver_kw  all three
        pack, unpack         maps of the tie representatives <-> one vector [flat]     cells
        best_response        (raw map, {"gamma", "action", "Zfull", ...}) [abstract]  all three
        _representation_error  relative residual of the projection [abstract]         stationary, spectral
        _impulse_responses   R with some observers' reactions removed [R]              none
        _passive_world       Zpass adjusted [Zpass]                                    none
        _identified          mask of the map nodes that read something [all True]     stationary, spectral
        _project             raw maps reproducing action kernels [abstract]            stationary, spectral
        _embedded_curvature  optional: curvature of a direction on a longer window     stationary
        interpolate_maps     a coarser result's maps on this grid [NotImplementedError] stationary, spectral
        maps_from_actions    raw maps reproducing action kernels in their world [abstract]  stationary, spectral
        expected_cost        variance part of an agent's cost from Z [abstract]        all three
        _diagnostics         checks at the equilibrium [no-op]                         stationary, spectral

    The mean layer (mean_system, solve_means, mean_cost, _mean_part: means.MeanLayer, a base of this class)
    is the base's, over these mean hooks (Nt = 1 on the stationary engine, the time nodes on the finite ones;
    nP primaries, states first):

        hook                 returns                                                   overridden by
        _mean_times          the time nodes (Nt,) of the mean paths [None: floats]     spectral, cells
        _mean_start          the mean state at time zero (nX,) [c.x0]                  stationary (0), spectral (the past)
        _mean_driven         whether anything moves the means [const, q, x0]           spectral (the past, the continuation)
        _mean_dynamics       (Mx (nX Nt, nP Nt), bx (nX Nt,), pinned rows) [abstract]  all three
        _mean_conditions     (Mu (nU Nt, nP Nt), bu (nU Nt,)) per agent [abstract]     all three
        _mean_atoms          the loss atoms' mean paths (m, Nt) [the primaries']       spectral, cells
        _mean_weights        discounted quadrature weights (Nt,) [1]                   spectral, cells

    Class attributes the engines set: RESULT (the result class), TOL / DAMPING / MAX_NEWTON (solve()
    defaults), ACTIONS (whether the engine can iterate on action kernels)."""
    model: Model
    c: object                       # the compiled model: .reps (tie representatives), .rep (agent -> representative), .N, .nW
    shapes: Dict[str, Tuple[int, ...]]      # agent -> shape of its raw maps
    RESULT = None                   # the result class
    TOL, DAMPING, MAX_NEWTON = 1e-10, 0.5, 60       # solve() defaults; each engine sets its own
    ANDERSON_M = tunable("anderson_m")              # Anderson memory (settings.anderson_m)
    ACTIONS = True                  # whether the engine can iterate on action kernels
    MONITORING = False              # whether the engine solves the monitored deviations of Chapter 6 (agents' `monitors`)
    FOC_RCOND = tunable("foc_rcond")    # a best-response system whose reciprocal condition estimate is below this is singular (settings)
    SECOND_ORDER_TOL = tunable("second_order_tol")      # curvature below which a negative value is window truncation (settings)
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
        if any(a.instant for a in model.agents) and not self.MONITORING:
            raise NotImplementedError(f"{type(self).__name__} does not solve instant observations (an agent seeing another's "
                                      "control level, `observes: {quote: {level: P}}`); the stationary engine does")
        if any(a.monitors for a in model.agents) and not self.MONITORING:
            raise NotImplementedError(f"{type(self).__name__} does not solve monitored deviations (Chapter 6: an agent's "
                                      "monitors); a model without `monitors` is the all-naive corner, what it solves")
        self.model = model
        self.verbose = verbose
        self.settings = Settings.of(settings)
        changed = self.settings.changed()
        self.solver_kw = {"verbose": verbose, **options, **({"settings": changed} if changed else {})}   # so a result can rebuild the same engine
        self._rphys: Dict[str, np.ndarray] = {}             # agent -> physical impulse responses (all reactions off)
        self._second_order_cache: Dict[str, dict] = {}       # representative -> its second-order check, shared with tied agents
        self._loss_forms: Dict[tuple, np.ndarray] = {}       # (agent, start_from) -> the loss form on the world (map-independent; built for the
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
    def init_kind(self, start_from: Dict[str, np.ndarray]) -> str:
        """Classify a warm start as "actions" or "maps" by shape; a wrong or ambiguous shape is an error."""
        kinds = set()
        for a in self.model.agents:
            if a.name not in start_from:
                raise ValueError(f"start_from has no entry for agent {a.name}")
            v = np.asarray(start_from[a.name]); sa, sm = self.action_shapes[a.name], tuple(self.shapes[a.name])
            if v.shape == sa and v.shape != sm:
                kinds.add("actions")
            elif v.shape == sm and v.shape != sa:
                kinds.add("maps")
            elif v.shape == sa:
                raise ValueError(f"start_from for {a.name}: shape {v.shape} could be action kernels or raw maps (N == nR == nW); "
                                 "pass the other representation")
            else:
                raise ValueError(f"start_from for {a.name}: shape {v.shape}; expected action kernels {sa} or raw maps {sm} "
                                 "(a warm start from a different grid cannot be used directly)")
        if len(kinds) != 1:
            raise ValueError("start_from mixes action kernels and raw maps across agents")
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
        passive world the agent's own control is off).  Reads c.row_blocks (the nonzero primary blocks
        only)."""
        c = self.c; rows, inst = [], []
        for r in range(len(agent.signals)):
            blocks, deltas = c.row_blocks(agent.name, r, excluded)
            y = np.zeros((c.N, Z.shape[1]))
            for nm, op in blocks.items():                            # only the primaries the row reads
                y += op @ Z[c.block(nm)]
            rows.append(y)
            inst.append([(c.channels.index(src), age, w) for src, dl in deltas.items() if src in c.channels for (age, w) in dl])
        return rows, inst

    def _passive_rows(self, agent: Agent, Zpass: np.ndarray):
        """The rows in the agent's passive world (its own strategy off)."""
        return self._seen_rows(agent, Zpass, set(agent.controls))

    # ------------------------------------------------- best-response pieces
    # ----------------------------------------------------- best response
    def _impulse_responses(self, agent: Agent, maps, R: np.ndarray) -> np.ndarray:
        """Hook (no engine overrides it): the responses R (n_prim N, nU) of the primary kernels
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

    def _shared_second_order(self, agent: Agent, compute) -> dict:
        """The agent's second-order check, compute() unless its tie representative's is cached (tied agents
        face the same problem up to relabelling)."""
        rep = self.c.rep[agent.name]
        if rep != agent.name and rep in self._second_order_cache:
            return self._second_order_cache[rep]
        self._second_order_cache[agent.name] = out = compute()
        return out

    def _physical_responses(self, agent: Agent, ncol: int) -> np.ndarray:
        """The impulse responses to the agent's controls with every reaction off (the FOC decomposition's
        physical part): map-independent, so computed once per agent.  `ncol` is the number of shock columns
        before the impulse columns."""
        if agent.name not in self._rphys:
            self._rphys[agent.name] = self.c.closed_loop(self.zero_maps(), excluded=None, impulse_controls=agent.controls)[:, ncol:]
        return self._rphys[agent.name]

    def free_mask(self, variable: str) -> Optional[np.ndarray]:
        """Hook: the free entries of the packed fixed-point vector (a boolean mask), or None when every entry
        is (the default; the spectral finite engine under freeze_before fixes part of its maps)."""
        return None

    def _identified(self, agent: Agent) -> np.ndarray:
        """Hook (stationary, spectral): boolean mask (nR N,) in (row, node) order of the map nodes at
        which the map on each row reads something within the window; the default keeps all of
        them.  The base assumes a masked node has exactly zero rows and columns in the FOC system
        (a row observed with a delay is uninformative there), so the engines' _solve_foc and
        _project solve on the kept nodes only and leave the map zero elsewhere; the second-order
        check restricts its quadratic form to the kept nodes (tiled over the controls).  Must be
        the same for tied agents up to relabelling, since they share the representative's check."""
        return np.ones(len(agent.signals) * self.c.N, dtype=bool)

    def _lanczos_extremes(self, matvec, n: int) -> dict:
        """The extreme eigenvalues {"lo", "hi"} of the symmetric form given by matvec on n unknowns, by Lanczos
        (settings.second_order_lanczos_tol and _maxiter); when it does not settle, the check's record saying
        so ("converged" False and a "message") rather than silence."""
        from scipy.sparse.linalg import LinearOperator, eigsh
        op = LinearOperator((n, n), matvec=matvec, dtype=float)
        ltol, lmax = self.settings.second_order_lanczos_tol, self.settings.second_order_lanczos_maxiter
        try:
            hi = float(eigsh(op, k=1, which="LA", tol=ltol, maxiter=lmax, return_eigenvectors=False)[0])
            lo = float(eigsh(op, k=1, which="SA", tol=ltol, maxiter=lmax, return_eigenvectors=False)[0])
        except Exception as exc:                          # Lanczos did not settle: say so rather than stay silent
            return {"min": None, "max": None, "ok": None, "converged": False, "message": f"{type(exc).__name__}: {exc}"[:120]}
        return {"lo": lo, "hi": hi}

    # ------------------------------------------------ maps from kernels

    # ------------------------------------------------------ fixed points
    def best_response(self, agent: Agent, maps: Dict[str, np.ndarray], want_decomp: bool = False, project: bool = True):
        """Hook (every engine): the agent's best response to `maps`, (raw map, {"gamma", "action" (nU, N, nW), "Zfull"});
        with want_decomp the dict also carries "second_order" and "decomp", which _fill_diagnostics records; with
        project=False the raw map is None (response_actions needs the action kernels only).  A singular system
        raises the ValueError of singular_system_message."""
        raise NotImplementedError

    def _representation_error(self, agent: Agent, Zfull: np.ndarray, actions: np.ndarray, g: np.ndarray) -> float:
        """Hook (stationary, spectral): how far the raw map g falls short of reproducing the action kernels."""
        raise NotImplementedError

    def interpolate_maps(self, coarse) -> Dict[str, np.ndarray]:
        """Hook (stationary, spectral): the raw maps of `coarse`, a result of the same engine on a
        grid of the same model with fewer nodes (coarse.maps, coarse.compiled), read at this grid's
        nodes: every agent's maps in self.shapes.  coarse_start (solve(start_policy="coarse")) and a
        result's refine() call it; refine() treats NotImplementedError as "no warm start" and
        starts from zero, coarse_start does not catch it (the cell engine, which does not override
        it, has no coarse start)."""
        raise NotImplementedError

    def stationary_start(self) -> Dict[str, np.ndarray]:
        """Hook: the raw maps a solve with start_policy="stationary" begins from (the spectral finite engine with a
        stationary continuation implements it)."""
        raise NotImplementedError("start='stationary' is for the spectral finite engine with a stationary continuation")

    def coarse_start(self, factor: float = 0.5, **solve_kw) -> Dict[str, np.ndarray]:
        """Raw maps to start from: the equilibrium at `factor` times the nodes, interpolated to this
        grid.  A coarse solve costs a few fine evaluations and usually saves many."""
        n0 = max(4, int(round(self.model.numerics.nodes * factor)))
        if n0 >= self.model.numerics.nodes:
            return None
        coarse = type(self)(self.model.with_numerics(nodes=n0), **self.solver_kw)
        res = coarse.solve(**{k: v for k, v in solve_kw.items() if k not in ("start_from", "start_policy")})
        self._coarse_evals, self._coarse_nodes = res.evaluations, n0
        return self.interpolate_maps(res)

    def solve(self, start_from: Optional[Dict[str, np.ndarray]] = None, tol: Optional[float] = None, damping: Optional[float] = None,
              max_newton: Optional[int] = None, variable: str = "actions", start_policy: str = "zero", max_evaluations: Optional[int] = None,
              deadline: Optional[float] = None, progress: Optional[Callable[[dict], None]] = None, diagnostics: bool = True):
        """Find the equilibrium: Anderson mixing on the fixed point of the best-response map (damping is
        the mixing weight), then a Newton-Krylov polish if it stalls.  variable="actions" iterates on
        the agents' action kernels, with the raw maps recovered by projection (better conditioned where
        strategies are weakly identified); "maps" iterates on the raw maps, which is also what happens
        with ties (tied agents' action kernels differ by a channel permutation) and on the cell engine.
        start_from: action kernels or raw maps per agent, either is accepted.  start_policy="coarse" (with no start_from)
        solves first at half the nodes and starts from that equilibrium interpolated to this grid.
        start_policy="stationary" (the spectral finite engine with a stationary continuation) starts from the
        continuation's stationary maps read at every node's age (what noisestate.transition() does).
        max_evaluations bounds the best-response evaluations (Anderson mixing and the polish together, the
        count res.evaluations reports) and deadline the wall time of the solve in seconds; at least one
        evaluation is made, and past either bound the best iterate so far is returned with converged=False
        and res.message naming the bound (no exception; require_converged() raises).  A coarse start is bounded the same
        way.  progress(info) is called after every evaluation with {"evaluation": the count so far,
        "residual": the relative residual, "phase": "anderson" or "newton" (prefixed "coarse " during a coarse
        start), "seconds": since the solve began}; an exception it raises propagates, which is how a solve is
        cancelled.  diagnostics=False skips the checks at the end (the first-order-condition decomposition,
        the second-order check and the representation error: res.foc and res.second_order stay empty,
        those checks report `skipped` in res.diagnostics.statuses) and fills the costs only, for a preview.
        The options as given are recorded in res.solve_kw (the bounds and diagnostics=False when given; the
        progress callable is not, nor is start_from), so solve(**res.solve_kw) repeats a solve that was not
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
        if self.model.ties and any(a.instant for a in self.model.agents):
            # ties iterate on the raw maps, and with instant observations the maps hold an equilibrium but do not
            # reach it: Chapter 6's market (unstable under best responses, radius 1.48) stalls at 0.85 from zero and
            # from a coarse solve alike, where the action kernels (untied) reach it from zero
            raise NotImplementedError("instant observations with ties are not solved yet: ties iterate on the raw maps, "
                                      "which do not reach the equilibrium when a level is observed at once")
        if start_from is None and start_policy == "coarse":
            # the coarse solve's checks are never read; it gets the same bounds (on its own count, on this clock)
            start_from = self.coarse_start(tol=tol, damping=damping, max_newton=max_newton, variable=variable, diagnostics=False,
                                     max_evaluations=max_evaluations, deadline=deadline, progress=None if progress is None else
                                     (lambda info: progress({**info, "phase": "coarse " + info["phase"], "seconds": time.time() - t0})))
            coarse_evals = getattr(self, "_coarse_evals", 0)
        elif start_from is None and start_policy == "stationary":
            start_from = self.stationary_start()
        elif start_policy not in ("zero", "coarse", "stationary"):
            raise ValueError(f"start_policy must be 'zero', 'coarse' or 'stationary', not {start_policy!r}")
        if variable == "actions" and (self.model.ties or not self.ACTIONS):
            variable = "maps"
        kind = self.init_kind(start_from) if start_from is not None else None
        if kind == "actions" and not self.ACTIONS:
            raise ValueError("this engine iterates on raw maps: pass raw maps as start_from")
        if variable == "actions":
            if kind is None:
                x0 = {a.name: np.zeros(self.action_shapes[a.name]) for a in self.model.agents}
            else:
                x0 = start_from if kind == "actions" else self.actions_from_maps(start_from)
            pack, unpack, respond = self.pack_actions, self.unpack_actions, self.response_actions
        else:
            if kind is None:
                x0 = self.zero_maps()
            else:
                x0 = start_from if kind == "maps" else self.maps_from_actions(start_from)
            pack, unpack, respond = self.pack, self.unpack, self.response_map

        z0 = pack(x0)
        free = self.free_mask(variable)                    # an engine with part of the vector fixed iterates on the rest

        def F(zz):
            evals[0] += 1
            if free is None:
                return pack(respond(unpack(zz))) - zz
            full = z0.copy(); full[free] = zz
            return (pack(respond(unpack(full))) - full)[free]
        st = self.settings
        z, resid, _, converged, message = solve_fixed_point(F, z0 if free is None else z0[free], tol=tol, verbose=self.verbose, damping=damping,
                                                            anderson_iters=st.anderson_iters, max_newton=max_newton, M=self.ANDERSON_M,
                                                            reg=st.anderson_reg, inner_m=st.newton_inner_m, max_evaluations=max_evaluations,
                                                            deadline=deadline, progress=progress, t0=t0)
        if free is not None:
            full = z0.copy(); full[free] = z; z = full
        maps = self.maps_from_actions(unpack(z)) if variable == "actions" else unpack(z)
        if coarse_evals:
            message = f"coarse start: {coarse_evals} evaluations at {self._coarse_nodes} nodes; " + message
        solve_kw = {"tol": tol, "damping": damping, "max_newton": max_newton, "variable": variable, "start_policy": start_policy,
                    **{k: v for k, v in (("max_evaluations", max_evaluations), ("deadline", deadline)) if v is not None}}
        res = self._result(maps, converged=converged, residual=resid, evaluations=evals[0], message=message,
                           solve_kw=solve_kw, diagnostics=diagnostics,
                           actions=unpack(z) if variable == "actions" else None)
        res.seconds = time.time() - t0            # the diagnostics of _finish are part of the solve's time
        return res

    def _result(self, maps, *, converged: bool, residual: float, evaluations: int, message: str, solve_kw: dict,
                diagnostics: bool = True, actions=None):
        """The result at `maps`: their closed loop as the world, the engine's own outputs filled by _finish.
        `actions` is the fixed point's iterate when it ran on the action kernels (the maps are its projection),
        kept as a later solve's warm start on the engines whose result has the field."""
        res = self.RESULT(model=self.model, compiled=self.c, maps=maps, world=self.c.closed_loop(maps), converged=converged,
                          residual=residual, evaluations=evaluations, seconds=0.0, message=message, solver_class=type(self),
                          solver_kw=self.solver_kw, settings=self.settings,
                          solve_kw={**solve_kw, **({} if diagnostics else {"diagnostics": False})})
        if actions is not None and "actions" in getattr(res, "__dataclass_fields__", {}):
            res.actions = actions
        self._finish(res)
        return res

    def _finish(self, res) -> None:
        """Fill the engine's own outputs on a fresh result: the costs (their variance part), the means and the
        mean part of the costs, then, unless the result's options say diagnostics=False, the engine's checks at
        the equilibrium.  The order is the contract: expected_cost first, _mean_part reads res.costs as the
        variance part and adds the mean part, _diagnostics last."""
        for a in self.model.agents:
            res.costs[a.name] = self.expected_cost(a, res.world)
        self._mean_part(res)
        if res.solve_kw.get("diagnostics", True) is not False:
            self._diagnostics(res)

    def _diagnostics(self, res) -> None:
        """Hook (stationary, spectral; the cell engine keeps the no-op): the checks at the equilibrium,
        filled on the result: res.foc[agent] (the "decomp" of best_response), res.second_order[agent]
        (when the check applies) and res.representation_error[agent].  Runs after _mean_part, skipped
        when the solve was made with diagnostics=False; the checks then report `skipped`.

        An engine that HAS these calls _fill_diagnostics below; the default stays a no-op so that an
        engine without a decomposition (the cell engine) inherits nothing it cannot honour."""

    def _fill_diagnostics(self, res) -> None:
        """One best response per agent at the equilibrium, recording its decomposition, its curvature
        and its representation error.

        Shared by the stationary and spectral engines, which had it twice, because the agent ORDER is
        an invariant and not a detail: a tied agent's representative must be evaluated before the
        agents that follow it, or the followers read a map the representative has not produced yet.
        """
        self._second_order_cache.clear()                               # the equilibrium's own check, not a stale one
        order = ([a for a in self.model.agents if self.c.rep[a.name] == a.name]
                 + [a for a in self.model.agents if self.c.rep[a.name] != a.name])
        for a in order:
            g, out = self.best_response(a, res.maps, want_decomp=True)
            res.foc[a.name] = out["decomp"]
            if out["second_order"] is not None:
                res.second_order[a.name] = out["second_order"]
            res.representation_error[a.name] = self._representation_error(a, out["Zfull"], out["action"], g)
            self._diagnostics_extra(res, a)
        self._loss_forms.clear()                  # the second-order check is done: its (n_prim N)^2 form is not kept

    def _diagnostics_extra(self, res, agent) -> None:
        """Hook: anything else an engine records per agent (the spectral engine's representation_parts)."""

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
        return self._over_representatives(lambda a: self.best_response(a, maps, project=False)[1]["action"])
