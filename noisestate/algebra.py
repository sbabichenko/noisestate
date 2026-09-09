"""The kernel algebra: the operators the base engine calls on a compiled model, as an explicit interface.

EngineBase (engine.py) writes the passive-world best response once against this algebra; each engine's
compiled model implements the members it supports (the stationary Compiled all of them on its age grid;
SpectralCompiled closed_loop, block, atom_op, expr_op and row_blocks, its engine taking the rest from the
operator form of spectral_operators.py; the cell engine's FiniteCompiled closed_loop and expr_kernel only, its
engine overriding best_response wholesale).  A member an engine lacks raises NotImplementedError naming it.
results.py and past.py read compiled models through these members and the attributes listed below only.

Layout.  A kernel is a nodal vector over the engine's N nodes (shock ages on the stationary grid, (time, age)
nodes on the triangle).  The world Z is (n_prim N, ncol): rows in (primary, node) order, block(name) slicing a
primary's N nodes; columns the nW Brownian channels, then one unit-impulse column per control in
`impulse_controls` (the triangle adds the initial shocks' columns before the impulses: ncol = nW + n_init).
An agent's raw maps are (nU, nR, N) (control, signal row, node of the row as the agent sees it); its action
kernels are (nU, N, nW); the first-order-condition kernel phi is (N,) per control; a mask over map nodes is
(nR N,) in (row, node) order.  The cell engine has its own layouts (Z (n_prim, N, ncol) per cell, maps
(nU, nR, N, N)), which is why it implements the applied forms only."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spec import Atom, Model


def missing(c, name: str) -> NotImplementedError:
    """The error a compiled model raises for a member of the kernel algebra it does not supply."""
    return NotImplementedError(f"{type(c).__name__} does not supply {name} of the kernel algebra (noisestate.algebra."
                               f"KernelAlgebra); its engine does not call it, see the class docstring for what it supplies")


class KernelAlgebra:
    """The interface of a compiled model, in the layout of the module docstring.

    Attributes (CompiledBase adopts them from the model's Structure; every engine has them):

        model                    the validated Model
        N, nW                    nodes of a kernel; Brownian channels
        prim, index              the primaries' names (states then controls) and name -> position
        nX, nU                   states, controls
        A, state_inputs          (nX, nX) contemporaneous feedback; [(state index, atom, coef)] of the other drift terms
        sigma, const, x0         (nX, nW) noise loadings; (nX,) constant drifts; (nX,) initial states
        channels                 the Brownian channels' names, Z's first nW columns
        rows                     agent -> [(name, drift, E, delay)]: the signal rows, E the noise loading (nW,)
        loss                     agent -> (atoms, Q, q): the loss 1/2 z'Qz + q'z over the atoms (name, lag)
        rep, reps                agent -> its tie group's representative; the representatives
        rho                      the discount rate

    Grid attributes, per engine (same_grid compares them): the stationary Compiled's `grid` (the age grid,
    N nodes on [0, L]); SpectralCompiled's `g` (the triangle), `tm` (the Nt time nodes), `T`, `mean_embed`
    ((N, Nt): a path on the time nodes as a kernel constant in age), `init_names` and `n_init` (the initial
    shocks' columns) and `continuation_info`; FiniteCompiled's `times` (the N cell times) and `h`.

    Operators (the shapes of the module docstring; m seen rows or loss atoms as the argument says):"""

    model: Model
    N: int
    nW: int
    prim: List[str]
    index: Dict[str, int]
    nX: int
    nU: int
    A: np.ndarray
    state_inputs: List[Tuple[int, Atom, float]]
    sigma: np.ndarray
    const: np.ndarray
    x0: np.ndarray
    channels: List[str]
    rows: Dict[str, List[Tuple[str, Dict[Atom, float], np.ndarray, float]]]
    loss: Dict[str, Tuple[List[Atom], np.ndarray, np.ndarray]]
    rep: Dict[str, str]
    reps: List[str]
    rho: float

    # ---------------------------------------------------------------- the closed loop
    def closed_loop(self, maps: Dict[str, np.ndarray], excluded: Optional[str] = None, impulse_controls: Sequence[str] = ()) -> np.ndarray:
        """Z (n_prim N, ncol + len(impulse_controls)) under the strategies `maps`, with the agent `excluded`
        switched off and a unit impulse of each listed control as the trailing columns."""
        raise missing(self, "closed_loop")

    def block(self, name: str) -> slice:
        """The slice of the primary's N rows in Z."""
        raise missing(self, "block")

    def atom_op(self, atom: Atom) -> np.ndarray:
        """(N, n_prim N): the kernel of name@lag from the primary vector."""
        raise missing(self, "atom_op")

    def atom_block(self, atom: Atom):
        """(primary index, N x N block) of atom_op's one nonzero block.  Default: slice it out of atom_op;
        an engine that knows the block without building the full width overrides this."""
        name, _ = atom
        return self.index[name], self.atom_op(atom)[:, self.block(name)]

    def expr_op(self, expr: Dict[Atom, float]) -> np.ndarray:
        """(N, n_prim N): the kernel of a linear expression over atoms (a definition, a row's drift)."""
        raise missing(self, "expr_op")

    def expr_kernel(self, Z: np.ndarray, expr: Dict[Atom, float]) -> np.ndarray:
        """The kernel of the expression in the world Z, (N, ncol): expr_op applied [expr_op(expr) @ Z]; the cell
        engine, whose world is (n_prim, N, ncol), supplies the applied form only."""
        return self.expr_op(expr) @ Z

    # ---------------------------------------------------------------- the seen rows
    def row_blocks(self, agent: str, r: int, excluded: set) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[float, float]]]]:
        """({primary: (N, N)}, {source: [(age, weight)]}): the regular part of the agent's row r as seen (shifted
        by its delay), one operator per primary it reads, and its instantaneous entries per source (a channel's
        noise at the delay, an `excluded` control's impulse at delay + lag)."""
        raise missing(self, "row_blocks")

    # ---------------------------------------------------------------- the best-response pieces
    def conv_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(N, m) row kernels, as seen (shifted by the delay) -> (m, N, N): the map gamma on such a row -> the
        action kernel on that channel."""
        raise missing(self, "conv_rows")

    def instant(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N): the action's instantaneous read, at `age`, of the map on a row observed with `delay` (a row's
        own noise at age = delay; an observed control's impulse at delay + lag)."""
        raise missing(self, "instant")

    def instant_adjoint(self, age: float, delay: float = 0.0) -> np.ndarray:
        """(N, N): the adjoint of instant on the FOC kernel phi."""
        raise missing(self, "instant_adjoint")

    def response(self, Ru: np.ndarray, own: int) -> np.ndarray:
        """(n_prim, N) impulse responses of the primaries to control `own` -> (n_prim N, N): action kernel ->
        world; the base engine then sets the own block to the identity."""
        raise missing(self, "response")

    def continuation(self, Rj: np.ndarray) -> np.ndarray:
        """(N, m) impulse responses of m loss atoms -> (m, N, N): the discounted continuation of each atom's
        kernel through its response."""
        raise missing(self, "continuation")

    def own_lag_read(self, lag: float) -> np.ndarray:
        """(N, N): the FOC term of a delayed read of the control itself."""
        raise missing(self, "own_lag_read")

    def projection_rows(self, Y: np.ndarray, delay: float) -> np.ndarray:
        """(N, m) row kernels -> (N, m N), columns (channel, node): E[phi_t dY_r(t - b)], the undiscounted
        correlation of the FOC kernel with the seen rows."""
        raise missing(self, "projection_rows")

    def cost_mass(self) -> np.ndarray:
        """(N, N): the Gram matrix under which expected_cost integrates products of kernels."""
        raise missing(self, "cost_mass")

    def causal_chunks(self, target: int = 4) -> List[Tuple[int, int]]:
        """Optional: [(lo, hi)] node ranges in increasing age such that projection_rows is zero from a chunk's ages
        to nodes of an earlier one, so the base engine skips those products [one chunk, (0, N)]."""
        return [(0, self.N)]
