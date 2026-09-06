"""The spectral finite engine's closed loop: ClosedLoopRows, the rows of (I - M) Z = B one time panel at a time
and their solve by block forward substitution (SpectralCompiled.closed_loop calls it; the reads, paths and
row blocks it draws on are spectral_compiled.py's, the operators of the best response spectral_operators.py's)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from .spectral_compiled import SpectralCompiled


class ClosedLoopRows:
    """The closed-loop system (I - M) Z = B of SpectralCompiled.closed_loop as a source of row blocks, one time
    panel at a time, and its solve.

    Z (n_prim, N, nc) holds the primaries' kernels at every node; the nc columns are the nW Brownian channels,
    with a past the n_init initial shocks, then one impulse column per control in impulse_controls.  A row on
    time panel p reads the nodes of panels q <= p only (causality), so Z is solved panel by panel: rows(p)
    gives the nonzero blocks {(primary i, primary j): (N_p, hi)} of M on the panel's nodes [lo, hi) against
    every node before hi; the columns before lo multiply the solved Z, the diagonal part is one dense solve,
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
                    gker = c.with_frozen(a.name, ui, r, np.zeros(N) if off else gm[ui, r][:N]) if past else gm[ui, r]
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
            for j in range(c.nX):
                if not self.inp[j]:
                    continue
                for i in range(c.nX):
                    Vr = c._vol_rows(i, j, p)
                    for pi, S in self.inp[j].items():
                        add((i, pi), Vr @ S[:hi, :hi])
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
            Np = hi - lo; n = nP * Np
            blk, forcing = self.panel(p)
            rhs = self.B[:, lo:hi].reshape(n, nc).copy()
            for bi, f in forcing.items():
                rhs[bi * Np:(bi + 1) * Np, :c.nW] += f
            diag = np.eye(n)
            for (i, j), X in blk.items():
                if lo:
                    rhs[i * Np:(i + 1) * Np] += X[:, :lo] @ Z[j, :lo]
                diag[i * Np:(i + 1) * Np, j * Np:(j + 1) * Np] -= X[:, lo:hi]
            Z[:, lo:hi] = np.linalg.solve(diag, rhs).reshape(nP, Np, nc)
        return Z.reshape(nP * c.N, nc)
