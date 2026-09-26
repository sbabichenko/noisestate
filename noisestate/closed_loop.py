"""The spectral finite engine's closed loop: ClosedLoopSources, the map-independent blocks and forcing the assembly
draws from the compiled model (mixed into SpectralCompiled, spectral_compiled.py, whose reads, paths and row
blocks they use), and ClosedLoopRows, the rows of (I - M) Z = B one time panel at a time and their solve by block
forward substitution (SpectralCompiled.closed_loop calls it).  The operators of the best response are
spectral_operators.py's."""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Dict, FrozenSet, Tuple

import numpy as np

if TYPE_CHECKING:
    from .spectral_compiled import SpectralCompiled


class ClosedLoopSources:
    """The map-independent sources of the assembly on the compiled model (mixed into SpectralCompiled): the rows of
    the Volterra propagation on a time panel, the forcing columns of the state rows and the rows of a control's
    convolution with its map kernel."""

    STATE_ROWS_CACHE_BYTES = 64 * 2**20       # the state rows of the panels kept per excluded set; beyond, recomputed

    def _vol_rows(self, i: int, j: int, p: int) -> np.ndarray:
        """The rows of Vol[i, j] on time panel p against the nodes before the panel's end, (N_p, hi), from the
        path."""
        lo, hi = self._panel_ranges[p]
        if self._vol_E is None:
            return np.zeros((hi - lo, hi))
        return self._vol_path.apply(self._vol_E[:, i, j], rows=(lo, hi))[:, :hi]

    def _state_columns(self, excluded, excl: set, imp: list) -> np.ndarray:
        """The shock and impulse columns of the state rows, B0 (n, ncol + len(imp)), cached per (excluded
        agent, impulse controls): the state's response to each channel's shock (with a past on the new-shock
        region; on the band the past's state kernel at age a - t propagated by e^{At}), the initial shocks'
        loads, the lagged inputs read before zero through the Volterra operator (every control, the excluded
        one's pre-zero actions being history), and each impulse's column (zero on the band: a deviation before
        zero is sunk)."""
        key = (excluded, tuple(imp))
        if key in self._state_cols:
            return self._state_cols[key]
        g = self.g; N = self.N; nW = self.nW; n = len(self.prim) * N; ncol = self.ncol
        B0 = np.zeros((n, ncol + len(imp)))
        EA = self.expA(g.a)                                             # (N, nX, nX)
        if self.past is None:
            for k in range(nW):
                v = self.sigma[:, k]
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), k] += EA[:, i, :] @ v
        else:
            up = g.upper; lower = ~up
            for k in range(nW):
                v = self.sigma[:, k]
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), k] += lower * (EA[:, i, :] @ v)
            if up.any():
                EAt = self.expA(g.t[up])                                     # (n_up, nX, nX)
                iu = np.where(up)[0]
                K0 = np.stack([self.past_at(self.prim[j], iu, g.a[iu] - g.t[iu]) for j in range(self.nX)], axis=1)   # (n_up, nX, nW)
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), :nW][up] += np.einsum("nj,njk->nk", EAt[:, i, :], K0)
            for col in range(self.n_init):
                v = self.init_sigma[:, col]
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), nW + col] += lower * (EA[:, i, :] @ v)
            for si, (nm, lag), c in self.state_inputs:
                if lag > 0 and g.L is not None:
                    pr = self.past_read(nm, lag)
                    if np.any(pr):
                        for i in range(self.nX):
                            B0[self.block(self.prim[i]), :nW] += c * (self._vol_mat(i, si, pr))
        for col, u in enumerate(imp):
            for si, (nm, lag), c in self.state_inputs:
                if nm != u:
                    continue
                v = np.zeros(self.nX); v[si] = c
                EAd = self.expA(np.maximum(g.a - lag, 0.0)) if lag else EA
                on = (g.a0 >= lag - 1e-12) if lag else np.ones(N, dtype=bool)
                if self.past is not None:
                    on = on & lower
                for i in range(self.nX):
                    B0[self.block(self.prim[i]), ncol + col] += on * (EAd[:, i, :] @ v)
        self._state_cols[key] = B0
        return B0

    def state_panel_rows(self, excl: FrozenSet[str], inp, p: int) -> Dict[Tuple[int, int], np.ndarray]:
        """The state rows' blocks of M on time panel p, {(state i, primary): (N_p, hi)}: the Volterra rows times
        the state inputs' operators with the controls in `excl` dropped.  Map-independent, so cached per
        (excl, p) while the cache is under STATE_ROWS_CACHE_BYTES (5% of a transition's solve when recomputed;
        at N = 14700 every panel's rows are 0.9 GB, and those are recomputed)."""
        cache = self.__dict__.setdefault("_state_rows", {})
        key = (excl, p)
        if key in cache:
            return cache[key]
        lo, hi = self._panel_ranges[p]
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for j in range(self.nX):
            if not inp[j]:
                continue
            for i in range(self.nX):
                Vr = self._vol_rows(i, j, p)
                for pi, S in inp[j].items():
                    X = Vr @ S[:hi, :hi]
                    out[(i, pi)] = out[(i, pi)] + X if (i, pi) in out else X
        size = sum(X.nbytes for X in out.values())
        used = self.__dict__.get("_state_rows_bytes", 0)
        if used + size <= self.STATE_ROWS_CACHE_BYTES:
            cache[key] = out; self._state_rows_bytes = used + size
        return out

    def conv_left_rows(self, gker: np.ndarray, delay: float, lo: int, hi: int) -> np.ndarray:
        """Rows [lo, hi) of conv_left(gker, delay)."""
        g = self.g
        lp = self._path(("conv_left", delay), r_lo=g.s + delay, r_hi=g.t, point_fn=lambda k, r: (r, r - g.s[k]),
                        known_fn=lambda k, r: (np.full_like(r, g.t[k] - delay), g.t[k] - r))
        return lp.with_known(gker, rows=(lo, hi))


class ClosedLoopRows:
    """The closed-loop system (I - M) Z = B of SpectralCompiled.closed_loop as a source of row blocks, one time
    panel at a time, and its solve.

    Z (n_prim, N, nc) holds the primaries' kernels at every node; the nc columns are the nW Brownian channels,
    with a past the n_init initial shocks, then one impulse column per control in impulse_controls.  A row on
    time panel p reads the nodes of panels q <= p only (causality), so Z is solved panel by panel: rows(p)
    gives the nonzero blocks {(primary i, primary j): (N_p, hi)} of M on the panel's nodes [lo, hi) against
    every node before hi; the columns before lo multiply the solved Z, the diagonal part is solved by block elimination (_panel_solve),
    and the blocks are discarded, so the (n_prim N)^2 system never exists (the memory is n_prim^2 N N_p for the
    largest panel).  The blocks are the line paths restricted to the panel's output nodes applied to the sparse
    reads of the primaries:
      a state's rows: the Volterra rows of the panel (c._vol_rows, the map-independent part) times the state
        inputs' atom operators (c.state_inputs_sparse; an excluded control's input is dropped, kept where it
        acts on a buffer);
      a control's rows, one term per seen row r of its agent: the conv_left rows of the map kernel
        (c.conv_left_rows) times the row's regular blocks (c.row_blocks_sparse).  With a past the map is the
        frozen stationary one on the buffer (c.with_frozen); the excluded agent's is zero on [0, T] and frozen
        on the buffer (its strategy off on [0, T] only) unless own_frozen=False switches it off there too;
        with `actions` every control's kernels are given and only the states have rows.
    The forcing B, built once: the states' shock, initial-shock and impulse columns (c._state_columns: the
    state's response to each channel's shock, on the band the past's state kernel at age a - t propagated by
    e^{At}, the initial shocks' loads, the lagged inputs read before zero, an impulse's column zero on the band
    where a deviation before zero is sunk); a control row's instantaneous entries (the row's noise at the delay
    under noise_weight, the past's loading on the band; an excluded control's impulse at the delay plus its lag,
    zero on the band); with a past the row's pre-zero increments under the map (past_conv_path) and, added on
    the panel from the conv rows, its lagged atoms read before zero (row_past); and the initial shocks' point
    observations under the discrete weights (disc_embed).
    The same integrals and sums at every size, one assembly; BLAS rounds a row block's product differently
    from the rows of a full product, so the numbers are the 0.4.0 dense assembly's to rounding, not to the bit."""

    def __init__(self, c: SpectralCompiled, maps, excluded=None, impulse_controls=(), own_frozen: bool = True, actions=None):
        self.c = c; N = c.N; nW = c.nW; ncol = c.ncol; g = c.g; nP = len(c.prim)
        imp = list(impulse_controls); self.nc = nc = ncol + len(imp)
        excl = set(next(a for a in c.model.agents if a.name == excluded).controls) if excluded else set()
        past = c.past is not None; self.band = band = g.L is not None; lower = ~g.upper
        self.B = B = np.zeros((nP, N, nc))
        self.inp = None                                 # state -> {primary index: CSR input operator}
        self.excl = frozenset(excl)
        if c.nX:
            B[:] = c._state_columns(excluded, excl, imp).reshape(nP, N, nc)
            self.inp = c.state_inputs_sparse(excl)
        self.rows = []                                  # (control's primary index, agent, row, delay, map kernel, sparse blocks)
        for a in c.model.agents:
            if actions is not None:
                for ui, u in enumerate(a.controls):
                    B[c.index[u]] = actions[a.name][ui]
                continue
            off = a.name == excluded
            if off and (not past or c.cont is None or not own_frozen):
                continue
            gm = maps[a.name]
            for ui, u in enumerate(a.controls):
                bi = c.index[u]
                for r, (rname, drift, E, delay) in enumerate(c.rows[a.name]):
                    blocks, deltas = c.row_blocks_sparse(a.name, r, excl)
                    # an excluded agent's kernel is zero on [0, T], the panels fixed by freeze_before included (its fixed early
                    # actions enter its passive world as a known response, finite_free.best_response: the impulse columns stay
                    # the responses with its reaction off, smooth across the fixed panels' shock times)
                    gker = c.with_frozen(a.name, ui, r, np.zeros(N) if off else gm[ui, r][:N], fixed=not off) if past else gm[ui, r]
                    self.rows.append((bi, a.name, r, delay, gker, blocks))
                    if band:
                        B[bi, :, :nW] += c.past_conv_path().bilinear(gker, c.past_row_kernel(a.name, r))   # increments before zero
                    for src, dl in deltas.items():
                        if src in c.channels:
                            col = c.channels.index(src)
                            for (age, w) in dl:
                                v = c.instant_sparse(age, delay) @ gker
                                B[bi, :, col] += c.noise_weight(a.name, r, col, w) * v if past else w * v
                        elif src in imp:
                            col = ncol + imp.index(src)
                            for (age, w) in dl:
                                v = c.instant_sparse(age, delay) @ gker
                                B[bi, :, col] += w * lower * v if past else w * v
                    if c.n_init and not off:
                        gd = gm[ui, r][N:]
                        for i in range(c.n_init):
                            e = c.init_rows[a.name][r, i]
                            if e:
                                B[bi, :, nW + i] += e * (c.disc_embed(delay) @ gd)

    def panel(self, p: int):
        """(blocks, forcing) of time panel p: the nonzero blocks {(i, j): (N_p, hi)} of M on the panel's nodes
        [lo, hi) against every node before hi, and with a band the lagged atoms read before zero under the
        conv rows, {control's primary index: (N_p, nW)}, to add to B on the panel."""
        c = self.c; lo, hi = c._panel_ranges[p]
        blk: Dict[Tuple[int, int], np.ndarray] = {}

        def add(key, X):
            blk[key] = blk[key] + X if key in blk else X
        if self.inp is not None:
            blk.update(c.state_panel_rows(self.excl, self.inp, p))      # the cached arrays are never written: add makes new ones
        forcing: Dict[int, np.ndarray] = {}
        for (bi, an, r, delay, gker, blocks) in self.rows:
            Cr = c.conv_left_rows(gker, delay, lo, hi)
            for nm, S in blocks.items():
                add((bi, c.index[nm]), Cr[:, :hi] @ S[:hi, :hi])
            if self.band:
                f = Cr @ c.row_past(an, r)
                forcing[bi] = forcing[bi] + f if bi in forcing else f
        return blk, forcing

    def solve(self) -> np.ndarray:
        """Z (n_prim N, nc) by block forward substitution over the time panels."""
        c = self.c; nP = len(c.prim); nc = self.nc
        Z = np.zeros((nP, c.N, nc))
        for p, (lo, hi) in enumerate(c._panel_ranges):
            Np = hi - lo
            blk, forcing = self.panel(p)
            rhs = self.B[:, lo:hi].copy()
            for bi, f in forcing.items():
                rhs[bi, :, :c.nW] += f
            within = {}
            for (i, j), X in blk.items():
                if lo:
                    rhs[i] += X[:, :lo] @ Z[j, :lo]
                if X[:, lo:hi].any():
                    within[(i, j)] = X[:, lo:hi]
            try:
                Z[:, lo:hi] = _panel_solve(within, rhs, nP, Np)
            except np.linalg.LinAlgError:
                raise ValueError(f"the closed loop is singular on time panel {p} (t in [{c.g.t[lo]:g}, {c.g.t[hi - 1]:g}]): "
                                 "the feedback of the strategies makes the world indeterminate there") from None
        return Z.reshape(nP * c.N, nc)


def _uncoupled(nP: int, keys: FrozenSet[Tuple[int, int]], _memo: dict = {}) -> Tuple[int, ...]:
    """A largest set C of primaries with no within-panel block among them (i, j in C, i == j included: M_CC = 0),
    by brute force from the largest size down (nP is the number of primaries, a handful), greedily beyond 16."""
    k = (nP, keys)
    if k in _memo:
        return _memo[k]
    free = [i for i in range(nP) if (i, i) not in keys]
    best: Tuple[int, ...] = ()
    if len(free) <= 16:
        for size in range(len(free), 0, -1):
            for C in itertools.combinations(free, size):
                if not any((i, j) in keys for i in C for j in C):
                    best = C; break
            if best:
                break
    else:
        for i in free:
            if not any((i, j) in keys or (j, i) in keys for j in best):
                best = best + (i,)
    _memo[k] = best
    return best


def _panel_solve(within: Dict[Tuple[int, int], np.ndarray], rhs: np.ndarray, nP: int, Np: int) -> np.ndarray:
    """The panel's diagonal system (I - M) z = rhs, rhs (nP, Np, nc), by block elimination: C the primaries with no
    block among them (_uncoupled), so z_C = rhs_C + M_CS z_S, and the rest S solves the Schur complement
    (I - M_SS - M_SC M_CS) z_S = rhs_S + M_SC rhs_C.  Every primary in C is no solve at all; C empty is the
    one dense solve.  The same solution, to rounding."""
    nc = rhs.shape[-1]
    C = _uncoupled(nP, frozenset(within))
    S = [i for i in range(nP) if i not in C]
    pos = {i: k for k, i in enumerate(S)}; cpos = {i: k for k, i in enumerate(C)}
    nS, nC = len(S) * Np, len(C) * Np
    MSS = np.zeros((nS, nS)); MSC = np.zeros((nS, nC)); MCS = np.zeros((nC, nS))
    for (i, j), X in within.items():
        if i in pos and j in pos:
            MSS[pos[i] * Np:(pos[i] + 1) * Np, pos[j] * Np:(pos[j] + 1) * Np] = X
        elif i in pos:
            MSC[pos[i] * Np:(pos[i] + 1) * Np, cpos[j] * Np:(cpos[j] + 1) * Np] = X
        else:
            MCS[cpos[i] * Np:(cpos[i] + 1) * Np, pos[j] * Np:(pos[j] + 1) * Np] = X
    rS = rhs[S].reshape(nS, nc); rC = rhs[list(C)].reshape(nC, nc)
    z = np.empty_like(rhs)
    if nS:
        A = np.eye(nS) - MSS
        b = rS.copy()
        if nC:
            A -= MSC @ MCS; b += MSC @ rC
        zS = np.linalg.solve(A, b)
        z[S] = zS.reshape(len(S), Np, nc)
        if nC:
            z[list(C)] = (rC + MCS @ zS).reshape(len(C), Np, nc)
    else:
        z[:] = rhs
    return z
