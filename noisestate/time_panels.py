"""Time panels for a finite horizon that one panel does not resolve: placed from the model's time scales before the
first solve, refined locally where the representation error sits, within a budget.

A finite horizon without lags is one panel in time and in age: every kernel is a polynomial of numerics.nodes degree
in each over [0, T].  That resolves a game whose time scales are of the order of T, and nothing much shorter: the
filter settles at the start (a Riccati layer, tanh(lambda t) for a rate lambda) and the value function turns at the end
(tanh(k (T - t))), each within a few time constants, a shock's effect decays in age at the closed loop's rates, and a
polynomial on [0, T] cannot follow a layer of width 1/lambda << T.  The one-agent regulator of the tests (rates of about
3) has its cost 0.13% off at T = 10 on 12 nodes and 190% off at T = 30; the tug of war of the website's wedge page at
precision p = 1000 (a filter rate of 31.6 on T = 1) has its kernels 5.6% off pointwise.

1. The grid from the time scales (time_scales, scale_breakpoints).  Before any solve, the model's linear algebra gives
   its rates: every agent's Kalman filter (the filter Riccati's closed-loop eigenvalues), the Nash feedback Riccati of the
   agents' controls under full information (its relaxation rate, the Jacobian of the coupled Riccati field at its fixed
   point, halved: sqrt(3/r) for the tug of war, where one player alone has 1/sqrt(r)), the closed loop's and the open
   loop's eigenvalues and the discount.  A tanh layer of rate lambda has poles pi/(2 lambda) off the real axis, and the
   Chebyshev coefficients of degree n - 1 of a panel of width w beside it are rho^-(n-1), rho the Bernstein ellipse
   through the pole (layer_tail: 7.9e-4 predicted at w = 0.25 on the wedge's first panel, 8.8e-4 measured; 1.7e-2 on its
   one panel, 3.0e-2 predicted).  When the one panel's predicted tail is above PRIOR_ABOVE (1e-2) at 8 nodes or more
   (PRIOR_MIN_NODES; a coarser grid is a deliberate choice), the first solve is on panels graded from both ends with first
   widths (powers of two) whose predicted tail is PRIOR_TARGET (3e-4; the filters' layer at 0 counted three times,
   START_WEIGHT), doubling towards the middle: [0, 1, 3, 7, 23, 27, 29, 30] for the regulator at T = 30 (the grid the
   graded search of the earlier version found), [0, 0.125, 0.375, 1] for the wedge at p = 1000; it starts from the one
   panel's equilibrium when that took more than COARSE_EVALUATIONS (5) evaluations.  Every other model is solved on the
   one panel first, exactly as before (bit for bit when it passes, or fails by what more nodes fix).  On the whole test
   suite every solve predicted above 1e-2 at 8 nodes or more failed the check on one panel (the regulator at T = 10, the
   wedge), and none the one panel resolves was predicted above 4e-3 there; below 8 nodes one was (1.2e-2, resolved to
   3e-12: the prediction says a layer exists, not that the kernels carry it), which PRIOR_MIN_NODES keeps on its panel.

2. Local refinement (_refine).  When the result fails the resolution check (settings.resolution_tol), every node's
   representation error (the check's own, per node) is read onto the breakpoint intervals (a piece's error counts for its
   time panel and its age panel) and only the intervals above the tolerance are bisected, the worst first, a few rounds
   (MAX_ROUNDS), each solve warm-started from the last one interpolated onto the new grid (6 to 12 evaluations where a cold
   start takes 18).  A one panel that fails badly (above GRADE_ABOVE, or singular) and whose failure a cheap trial finds
   grading cuts tenfold (_grading_helps, as before) starts this refinement from the time-scale grid (or, where the linear
   algebra gives no rates, from the panels graded from both ends of the earlier version).

3. The budget (_Budget).  No automatic grid passes settings.auto_panels_max unknowns (nU nR N of the largest agent,
   4096) or an estimated peak memory of settings.auto_panels_memory MB (1536; 100 + 8e-5 sum over the answering agents of
   (nU nR N)^2, measured to 15% with the checks), and a refinement round's estimate (the best earlier grid's result
   counted too, as it stays alive) is at most settings.auto_panels_growth (4) times the first automatic grid's: the wedge
   at p = 1000 refines 864 -> 1440 unknowns and stops before 2160 (8 s, 0.7 GB, where the earlier version took 56 s and
   3.7 GB).  A time-scale grid past the caps is coarsened (its end widths doubled) until it fits.  Where the next round would pass the budget, the best result so far is returned with a warning naming the
   breakpoints to re-solve with (res.panels["suggested"]: the intervals above the tolerance bisected until a halving is
   predicted to reach it, at a tenfold fall per halving) and their cost; res.sharpen() re-solves there, warm-started.

res.numerics.breakpoints holds the grid used (a solve with it reproduces the result), res.panels the route ("one panel",
"time scales", "refined"), the rates, the history of (panels, unknowns, representation error), resolved, and suggested
(None when resolved or when nothing is known to help); res.message says how the grid was found.  suggest(model) gives the
time-scale grid without solving (the browser's "fast, then sharpen": solve on one panel with auto_panels_max 0, show it,
then res.sharpen()).  A model whose rates are many orders of magnitude above 1/T (the regulator with its noise scaled by
1e8) is singular on the trial's grid too, and the singular error, which says to rescale time, is raised at once.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import numpy as np


GRADE_ABOVE = 5e-4      # a one-panel representation error above this is refined (1e-3 at T = 5 on the regulator, where the cost is
                        # 2e-5 off; the modest failures more nodes fix, 1e-5 .. 1e-4, stay as they were)
TAIL_ABOVE = 1e-3       # solve(diagnostics=False) asks the representation error (one best response per agent) only of kernels whose
                        # Chebyshev tail (kernel_tail) is above this: 2.6e-3 at T = 5, 5.9e-4 at T = 3, 2.3e-5 on Chapter 1's game
TRIAL_NODES = 4         # the trial of a failed one panel: at this many nodes (against the one panel at 4 and at 8)
TRIAL_SPLIT = 128.0     # ... the trial's graded grid: ends of T / 128 to T / 256 (thirteen panels)
GRADE_GAIN = 0.1        # ... graded when it cuts the error tenfold (else the one panel's result is returned as before: an error
                        # that grading leaves, a random walk the trader sees at 0.8 on Kyle-Back at T = 1, is not a time scale)
PRIOR_ABOVE = 1e-2      # a one panel whose predicted layer tail (layer_tail) is above this is not solved: the time-scale grid is
PRIOR_TARGET = 3e-4     # the predicted layer tail of the time-scale grid's end panels: w = 0.125 for a rate of 31.6 at 12 nodes (3.3e-5;
                        # 0.25 would be 7.9e-4), w = 1 for 3.2 (8.8e-6; 2: 3.2e-4), 0.5 for 10 (1.1e-4): the widths the graded searches
                        # of the earlier version landed on (the regulator at T = 30, and at r = 0.01, T = 5 and 10, the same grids)
START_WEIGHT = 3.0      # the filters' layer at t = 0 counts three times against PRIOR_TARGET (a first width of 0.5 for a filter rate of
                        # 5.5, 3.5e-6; 1 was 1.7e-4 and failed the check at 6e-6 on the regulator at p = 30, T = 5): the shocks
                        # born in the filter's transient carry it along the whole diagonal s = 0, not only near t = 0
PRIOR_MIN_NODES = 8     # below this many nodes the one panel is solved first (a coarse grid is a deliberate choice, and at 4 nodes
                        # Chapter 1's game is predicted at 0.04 and resolved by 8 nodes better than by panels: the trial decides)
MAX_ROUNDS = 4          # local refinement rounds at most
HALVING_GAIN = 10.0     # the fall of an interval's error per bisection assumed for the suggested breakpoints (measured 7 .. 1000)
MEMORY_BASE = 100.0     # MB: the estimate of a solve's peak memory with its checks, MEMORY_BASE + MEMORY_PER * sum (nU nR N)^2
MEMORY_PER = 8e-5       # MB per squared unknown of each answering agent (wedge 1440 .. 3024: 555 .. 1612 MB; regulator 4032: 1349)


# ------------------------------------------------------------------------------------------------ time scales
def _expand(coefs: dict, defs: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in coefs.items():
        if k in defs:
            for k2, v2 in _expand(defs[k], defs).items():
                out[k2] = out.get(k2, 0.0) + float(v) * v2
        else:
            out[k] = out.get(k, 0.0) + float(v)
    return out


def _abs_eigs(M) -> List[float]:
    return [float(abs(e)) for e in np.linalg.eigvals(M)] if M.size else []


def time_scales(model) -> Optional[dict]:
    """The model's rates from its linear algebra, before any solve: {"filter": [...] (every agent's Kalman filter under
    full knowledge of the others' actions: the filter Riccati's closed-loop eigenvalue moduli), "control": [...] (the Nash
    feedback Riccati of the agents' controls under full information: its relaxation rates, half the moduli of the coupled
    Riccati field's Jacobian at the fixed point; each agent alone where the coupled fixed point is not found), "loop": [...]
    (that closed loop's eigenvalue moduli), "open": [...], "discount": rho, "start": the largest Riccati layer at t = 0 (the
    filters'), "end": the largest at T (the controls'), "decay": the largest of the loop's, the open loop's and the
    discount (a shock's decay in age; end_tails counts it at both ends)}; None where the model is outside the linear algebra
    (a lag, a level row, a coefficient on an unknown atom) or nothing is found.  Agents that are myopic, have no controls
    or a loss not positive definite in their controls, and filters or Riccatis the solver refuses (an unobservable
    unstable mode, a loss not positive semi-definite) are left out: the rates are for placing panels, never a result."""
    try:
        return _time_scales(model)
    except Exception:                                  # anything the linear algebra does not cover: no prediction
        return None


def _time_scales(model) -> Optional[dict]:
    if model.all_lags():
        return None
    defs = {d.name: d.expr for d in model.definitions}
    X = [s.name for s in model.states]
    n = len(X)
    if n == 0:
        return None
    W = list(model.shocks)
    ix = {x: i for i, x in enumerate(X)}
    iw = {w: i for i, w in enumerate(W)}
    U = [u for a in model.agents for u in a.controls]
    iu = {u: i for i, u in enumerate(U)}
    A = np.zeros((n, n)); B = np.zeros((n, len(U))); G = np.zeros((n, len(W)))
    for s in model.states:
        for k, v in _expand(s.drift, defs).items():
            if k in ix:
                A[ix[s.name], ix[k]] += v
            elif k in iu:
                B[ix[s.name], iu[k]] += v
            else:
                return None
        for k, v in s.noise.items():
            G[ix[s.name], iw[k]] += float(v)
    rho = float(model.horizon.discount or 0.0)
    out = {"filter": [], "control": [], "loop": [], "open": _abs_eigs(A), "discount": abs(rho)}
    players = []                                      # (B_i, Q_i, N_i, R_i) of the agents with a feedback problem
    for a in model.agents:
        rows = [r for r in a.signals if not r.delay]
        if any(r.level for r in rows):
            return None
        if rows:
            C = np.zeros((len(rows), n)); E = np.zeros((len(rows), len(W)))
            for i, r in enumerate(rows):
                for k, v in _expand(r.drift, defs).items():
                    if k in ix:
                        C[i, ix[k]] += v
                for k, v in r.noise.items():
                    E[i, iw[k]] += float(v)
            R = E @ E.T
            if np.abs(C).max(initial=0.0) > 0 and np.linalg.matrix_rank(R) == len(rows):
                try:
                    Ri = np.linalg.inv(R)
                    Af = A - G @ E.T @ Ri @ C
                    Qf = G @ (np.eye(len(W)) - E.T @ Ri @ E) @ G.T
                    S = _care(Af.T, C.T, 0.5 * (Qf + Qf.T), R)
                    out["filter"] += _abs_eigs(Af - S @ C.T @ Ri @ C)
                except Exception:
                    pass
        if a.myopic or not a.controls:
            continue
        own = a.controls
        Q = np.zeros((n, n)); N = np.zeros((n, len(own))); Rc = np.zeros((len(own), len(own)))
        for term in a.loss:
            if len(term) != 3:
                continue
            c = float(term[0])
            for k1, v1 in _expand({term[1]: 1.0}, defs).items():
                for k2, v2 in _expand({term[2]: 1.0}, defs).items():
                    w = 0.5 * c * v1 * v2
                    for p, q in ((k1, k2), (k2, k1)):
                        if p in ix and q in ix:
                            Q[ix[p], ix[q]] += w
                        elif p in ix and q in own:
                            N[ix[p], own.index(q)] += w
                        elif p in own and q in own:
                            Rc[own.index(p), own.index(q)] += w
        if np.all(np.linalg.eigvalsh(0.5 * (Rc + Rc.T)) > 0):
            players.append((B[:, [iu[u] for u in own]], Q, N, Rc))
    if players:
        rates = _game_rates(A, players, rho) or _solo_rates(A, players, rho)
        if rates is not None:
            out["loop"], out["control"] = rates
    out["start"] = max(out["filter"], default=0.0)
    out["end"] = max(out["control"], default=0.0)
    out["decay"] = max(out["loop"] + out["open"] + [out["discount"]], default=0.0)
    if not all(math.isfinite(out[k]) for k in ("start", "end", "decay")):
        return None
    return out


def _care(A, B, Q, R, s=None):
    """The stabilising solution of A'P + P A - (P B + s) R^-1 (B'P + s') + Q = 0: the stable invariant subspace of the
    Hamiltonian by numpy's eig (tens of microseconds at these sizes, where scipy's QZ-based solver takes 0.4 ms and the
    Nash iteration calls it dozens of times), checked by its residual; scipy's solver where that check fails."""
    n = A.shape[0]
    s = np.zeros_like(B) if s is None else s
    Ri = np.linalg.inv(R)
    Ah, Qh = A - B @ Ri @ s.T, Q - s @ Ri @ s.T
    H = np.block([[Ah, -B @ Ri @ B.T], [-Qh, -Ah.T]])
    try:
        w, V = np.linalg.eig(H)
        stable = np.argsort(w.real)[:n]
        if np.all(w.real[stable] < 0) and np.sum(w.real < 0) == n:
            U1, U2 = V[:n, stable], V[n:, stable]
            P = np.real(U2 @ np.linalg.inv(U1))
            P = 0.5 * (P + P.T)
            res = A.T @ P + P @ A - (P @ B + s) @ Ri @ (B.T @ P + s.T) + Q
            if np.abs(res).max() <= 1e-9 * max(1.0, np.abs(P).max(), np.abs(Q).max()):
                return P
    except np.linalg.LinAlgError:
        pass
    from scipy.linalg import solve_continuous_are
    return solve_continuous_are(A, B, Q, R, s=s)


def _solo_rates(A, players, rho):
    """Each agent's own regulator, the others' controls off: (closed-loop moduli, Riccati rates), or None."""
    n = A.shape[0]
    loop, ctrl = [], []
    for B, Q, N, R in players:
        try:
            P = _care(A - 0.5 * rho * np.eye(n), B, Q, R, s=N)
        except Exception:
            continue
        e = _abs_eigs(A - B @ np.linalg.solve(R, B.T @ P + N.T))
        loop += e; ctrl += e
    return (loop, ctrl) if ctrl else None


def _game_rates(A, players, rho, iters: int = 60):
    """The Nash feedback Riccati under full information: its fixed point by Gauss-Seidel over the agents' AREs, then the
    moduli of the coupled Riccati field's Jacobian there, halved (the rate k of a tanh(k (T - t)) terminal layer: sqrt(3/r)
    for two players pulling one integrator, 1/sqrt(r) for one), with the closed loop's; None when it does not settle."""
    n = A.shape[0]; m = len(players)
    Ar = A - 0.5 * rho * np.eye(n)
    K = [np.zeros((B.shape[1], n)) for B, *_ in players]
    P = [np.zeros((n, n)) for _ in players]
    for _ in range(iters):
        delta = 0.0
        for i, (B, Q, N, R) in enumerate(players):
            Ai = Ar - sum(players[j][0] @ K[j] for j in range(m) if j != i)
            P[i] = _care(Ai, B, Q, R, s=N)
            Ki = np.linalg.solve(R, B.T @ P[i] + N.T)
            delta = max(delta, float(np.abs(Ki - K[i]).max()))
            K[i] = Ki
        if delta <= 1e-7 * (1.0 + max(float(np.abs(k).max()) for k in K)):    # rates to a few digits are all a panel needs
            break
    else:
        return None
    if m == 1:
        e = _abs_eigs(Ar + 0.5 * rho * np.eye(n) - players[0][0] @ K[0])
        return e, _abs_eigs(Ar - players[0][0] @ K[0])

    def field(x):
        Ps = [x[i * n * n:(i + 1) * n * n].reshape(n, n) for i in range(m)]
        Ks = [np.linalg.solve(R, B.T @ Pi + N.T) for (B, Q, N, R), Pi in zip(players, Ps)]
        out = []
        for i, (B, Q, N, R) in enumerate(players):
            Ai = Ar - sum(players[j][0] @ Ks[j] for j in range(m) if j != i)
            out.append((Q + Ps[i] @ Ai + Ai.T @ Ps[i] - (Ps[i] @ B + N) @ Ks[i]).ravel())
        return np.concatenate(out)
    x0 = np.concatenate([p.ravel() for p in P])
    h = 1e-6 * max(1.0, float(np.abs(x0).max()))
    J = np.empty((len(x0), len(x0)))
    for k in range(len(x0)):
        e = np.zeros(len(x0)); e[k] = h
        J[:, k] = (field(x0 + e) - field(x0 - e)) / (2 * h)
    loop = _abs_eigs(A - sum(B @ Ki for (B, *_), Ki in zip(players, K)))
    return loop, [0.5 * r for r in _abs_eigs(J)]


def layer_tail(width: float, rate: float, nodes: int) -> float:
    """The Chebyshev coefficient of degree nodes - 1, relative to the peak, of a tanh layer of `rate` at an end of a panel of
    `width`: rho^-(nodes - 1), rho the Bernstein ellipse of the panel through the layer's pole, pi / (2 rate) off the end."""
    if rate <= 0 or width <= 0:
        return 0.0
    z = complex(-1.0, math.pi / (rate * width))        # the pole, in the panel's [-1, 1], above its left end
    r = abs(z + np.sqrt(z * z - 1))
    r = max(r, 1.0 / r)
    return float(r ** (-(nodes - 1)))


def end_tails(scales: dict, width: float, nodes: int, start_weight: float = 1.0):
    """(the tail at the start, the tail at the end) of a panel of `width` beside each end: its Riccati layer's, and the
    decay's counted as a layer of its rate too.  (As the entire function it is in age, exp(-rate a), the decay asks far
    less: 2 I_11(z) e^-z, 3e-7 for a rate of 3.2 on a panel of 2; but a first panel of 2 on the regulator at T = 30, which
    that allows, fails the check at 6e-6 where the layer's width 1 passes at 5.6e-7: the kernels near t = 0 carry the
    decay and the filter's transient together, along the diagonal s = 0.)"""
    d = layer_tail(width, scales["decay"], nodes)
    return max(start_weight * layer_tail(width, scales["start"], nodes), d), max(layer_tail(width, scales["end"], nodes), d)


def predicted_tail(scales: Optional[dict], T: float, nodes: int) -> float:
    """The larger of end_tails on the one panel [0, T] (0 without scales)."""
    if not scales:
        return 0.0
    return max(end_tails(scales, T, nodes))


def scale_breakpoints(T: float, scales: dict, nodes: int, target: float = PRIOR_TARGET) -> List[float]:
    """Panels graded from both ends by the time scales: first widths w0 at 0 and wT at T, the largest powers of two whose
    layer_tail against the start's (the end's) rate is within `target`, each next panel twice the last, the two
    sequences meeting where their widths match (x + w0 = T - x + wT); a middle panel shorter than half its wider
    neighbour is merged into it.  [0, T] when neither end needs a cut."""
    T = float(T)

    def first(end):                                  # a power of two (the cuts are short binary fractions), or T
        if end_tails(scales, T, nodes, START_WEIGHT)[end] <= target:
            return T
        w = 2.0 ** math.floor(math.log2(T))
        if w >= T:
            w /= 2.0
        while w > T * 2.0 ** -30 and end_tails(scales, w, nodes, START_WEIGHT)[end] > target:
            w /= 2.0
        return w
    return graded_from_ends(T, first(0), first(1))


def graded_from_ends(T: float, w0: float, wT: float) -> List[float]:
    """Panels of widths w0, 2 w0, 4 w0, ... from 0 and wT, 2 wT, ... from T, meeting where their widths match; a middle
    panel shorter than half its wider neighbour is merged into it.  [0, T] when neither width is below T."""
    T = float(T)
    if w0 >= T and wT >= T:
        return [0.0, T]
    m = 0.5 * (T + min(wT, T) - min(w0, T))
    eps = 1e-12 * T
    left, x, step = [0.0], w0, w0
    while x < m - eps:
        left.append(x); step *= 2.0; x += step
    right, y, step = [T], T - wT, wT
    while y > m + eps:
        right.append(y); step *= 2.0; y -= step
    lo, hi = left[-1], right[-1]
    wl = lo - left[-2] if len(left) > 1 else 0.0
    wr = right[-2] - hi if len(right) > 1 else 0.0
    if hi - lo < 0.5 * max(wl, wr) and len(left) + len(right) > 3:
        (left if wl >= wr and len(left) > 1 else right).pop()
    return [float(b) for b in sorted({round(b, 12) for b in left + right})]


def _fitted(T: float, wanted: List[float], scales: dict, nodes: int, budget) -> List[float]:
    """The time-scale grid past the caps, coarsened: its end widths doubled, each end separately (w0 2^i, wT 2^j), and of
    the grids that fit the one whose worse end tail (end_tails, as scale_breakpoints weighs it) is smallest, the finer on
    a tie: the regulator with its noise x5 at T = 30 keeps its end width 1 and widens only the start's 0.5 -> 1."""
    w0 = wanted[1] - wanted[0]
    wT = wanted[-1] - wanted[-2]
    best = None
    for i in range(12):
        for j in range(12):
            a, b = w0 * 2 ** i, wT * 2 ** j
            bp = graded_from_ends(T, a, b)
            if len(bp) <= 2 or not budget.fits(bp):
                continue
            tail = max(end_tails(scales, min(a, T), nodes, START_WEIGHT)[0], end_tails(scales, min(b, T), nodes, START_WEIGHT)[1])
            key = (tail, -budget.unknowns(bp))
            if best is None or key < best[0]:
                best = (key, bp)
    return best[1] if best else [0.0, T]


def suggest(model, numerics=None) -> Optional[List[float]]:
    """The time-scale grid of a finite model one panel is predicted not to resolve (scale_breakpoints; None otherwise, or
    when the model is outside the linear algebra), without solving: solve(model, {"breakpoints": suggest(model)})."""
    from .numerics import Numerics
    from .spec import as_model
    model = as_model(model)
    num = model.numerics.merged(Numerics.of(numerics)).resolved(model.horizon.kind)
    if model.horizon.kind != "finite" or model.all_lags() or num.breakpoints is not None:
        return None
    sc = time_scales(model)
    T, nodes = float(model.horizon.extent), int(num.nodes)
    if predicted_tail(sc, T, nodes) <= PRIOR_ABOVE:
        return None
    bp = scale_breakpoints(T, sc, nodes)
    return bp if len(bp) > 2 else None


# ------------------------------------------------------------------------------------------------ the budget
class _Budget:
    """The caps of the automatic grids (module docstring, 3): unknowns and estimated memory of a breakpoint list."""

    def __init__(self, S):
        c, st = S.c, S.settings
        self.nt, self.na = c.g.nt, c.g.na
        self.width = max(len(a.controls) * len(a.signals) for a in S.model.agents)
        self.answering = [len(a.controls) * len(a.signals) for a in S.model.agents if c.rep[a.name] == a.name]
        self.max_unknowns = int(st.auto_panels_max)
        self.max_memory = float(st.auto_panels_memory)
        self.growth = float(st.auto_panels_growth)
        self.first: Optional[float] = None             # the first automatic grid's estimated memory (the growth's base)

    def nodes(self, bp) -> int:
        P = len(bp) - 1
        return P * (P + 1) // 2 * self.nt * self.na

    def unknowns(self, bp) -> int:
        return self.width * self.nodes(bp)

    def memory(self, bp, held=None) -> float:
        N = self.nodes(bp)
        mem = MEMORY_BASE + MEMORY_PER * sum((w * N) ** 2 for w in self.answering)
        return mem if held is None else mem + self.memory(held) - MEMORY_BASE

    def fits(self, bp, refinement: bool = False, held=None) -> bool:
        """Within the caps; `held`: a grid whose result stays alive while bp is solved (the best so far), counted too."""
        u = self.unknowns(bp)
        if u > self.max_unknowns or self.memory(bp, held) > self.max_memory:
            return False
        return not (refinement and self.first is not None and self.memory(bp, held) > self.growth * self.first)

    def why(self, bp, held=None) -> str:
        u, mem = self.unknowns(bp), self.memory(bp, held)
        out = []
        if u > self.max_unknowns:
            out.append(f"settings.auto_panels_max ({self.max_unknowns})")
        if mem > self.max_memory:
            out.append(f"settings.auto_panels_memory ({self.max_memory:g} MB)")
        if self.first is not None and mem > self.growth * self.first:
            out.append(f"settings.auto_panels_growth ({self.growth:g} x the first grid's {self.first:.0f} MB)")
        return " and ".join(out) or "the budget"


# ------------------------------------------------------------------------------------------------ solve
def plain(S) -> bool:
    """Whether S is the spectral engine on a finite horizon without lags or delays, a window of a past (initial shocks are
    fine) or a continuation: the models whose panels are a free choice (res.panels is filled for them)."""
    c = getattr(S, "c", None)
    g = getattr(c, "g", None)
    if g is None or not hasattr(g, "P") or not hasattr(S, "_rep_by_node"):
        return False
    return not S.model.all_lags() and g.L is None and getattr(c, "cont", None) is None


def eligible(S) -> bool:
    """Whether S may re-cut its panels: plain(S), one panel from the default (numerics.breakpoints not given) and the
    automatic grids on (settings.auto_panels_max > 0)."""
    return plain(S) and S.model.numerics.breakpoints is None and S.c.g.P == 1 and S.settings.auto_panels_max > 0


def _singular(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "singular" in str(exc) and type(exc).__name__ == "ValueError"


def _fmt(bp) -> str:
    return "[" + ", ".join(f"{b:g}" for b in bp) + "]"


def _with_bp(num, bp, **more):
    from .numerics import Numerics
    return Numerics.of(num.to_dict()).merged(Numerics(breakpoints=[float(b) for b in bp], **more))


def solve(S0, num, start_from, start_policy, run: dict, diagnostics: bool, build, verbose: bool = False):
    """solve()'s path for an eligible model (eligible(S0)): the time-scale grid when the one panel is predicted to fail,
    else the one panel as before and, when it fails badly, the refinement.  build(model, numerics) -> (engine, numerics)."""
    from . import engines
    model = S0.model
    T = float(model.horizon.extent)
    nodes = int(S0.c.g.nt)
    scales = time_scales(model)
    pred = predicted_tail(scales, T, nodes)
    budget = _Budget(S0)
    if start_from is None and pred > PRIOR_ABOVE and nodes >= PRIOR_MIN_NODES:
        wanted = scale_breakpoints(T, scales, nodes)
        bp = wanted if budget.fits(wanted) else _fitted(T, wanted, scales, nodes, budget)
        if len(bp) > 2:
            S, num1 = build(model, _with_bp(num, bp))
            if verbose:
                print(f"  -- the one panel is predicted not to resolve the time scales (start rate {scales['start']:.3g}, end "
                      f"rate {scales['end']:.3g} on T = {T:g}): {len(bp) - 1} panels {_fmt(bp)}, {S.c.N} nodes", flush=True)
            coarse = _coarse_start(S0, S, start_policy, run) if start_policy is None else None
            try:
                if coarse is not None:
                    res = S.solve(start_from=S.interpolate_maps(coarse), diagnostics=False, **run)
                    res.evaluations += coarse.evaluations
                else:
                    res = S.solve(start_policy=engines.default_start(S, start_policy), diagnostics=False, **run)
            except ValueError as exc:
                if not _singular(exc):
                    raise
                res = None
            if res is not None:
                budget.first = budget.memory(bp)
                info = {"route": "time scales", "rates": _rates(scales), "predicted_one_panel": pred,
                        "wanted": wanted if wanted != bp else None}
                return _refine(S, res, bp, budget, info, run, diagnostics, build, num, T, verbose)
    # the one panel, as before
    res, err = None, None
    try:
        res = S0.solve(start_from=start_from, start_policy=engines.default_start(S0, start_policy), diagnostics=diagnostics, **run)
    except ValueError as exc:
        if not _singular(exc):
            raise
        err = exc
    return _after_one_panel(S0, num, res, err, scales, pred, budget, run, diagnostics, build, T, verbose)


COARSE_EVALUATIONS = 5  # the time-scale grid starts from the one panel's equilibrium when that took more evaluations than this (the
                        # wedge at p = 1000: 18 -> 12 on three panels, 4.1 -> 2.9 s with the one panel's 0.2 s); a model the fixed
                        # point settles in a few evaluations from zero (the regulator: 3) starts from zero


def _coarse_start(S0, S, start_policy, run):
    """The one panel's equilibrium when it is worth starting the time-scale grid from (COARSE_EVALUATIONS), else None."""
    try:
        coarse = S0.solve(diagnostics=False, **run)
    except ValueError as exc:
        if not _singular(exc):
            raise
        return None
    return coarse if coarse.converged and coarse.evaluations > COARSE_EVALUATIONS else None


def annotate(S, res) -> None:
    """res.panels for a result solved without the automatic grids (settings.auto_panels_max 0, or numerics.breakpoints
    given) on a finite horizon without lags, window or continuation: the grid, whether it is resolved and the suggested
    breakpoints (the time-scale grid when one panel is predicted not to resolve, else the intervals above the tolerance
    bisected; None when resolved or without the diagnostics).  Numbers are not touched: res.sharpen() re-solves there."""
    g = S.c.g
    bp = [float(b) for b in g.bp]
    T = float(S.model.horizon.extent)
    tol = S.settings.resolution_tol
    rep = max(res.representation_error.values()) if res.representation_error else None
    scales = time_scales(S.model)
    info = {"route": "one panel" if g.P == 1 else "given", "rates": _rates(scales),
            "predicted_one_panel": predicted_tail(scales, T, int(g.nt)), "breakpoints": bp,
            "history": [(g.P, S.c.N, rep)], "resolved": None if rep is None else bool(rep <= tol), "suggested": None}
    if g.P == 1 and info["predicted_one_panel"] > PRIOR_ABOVE:
        sug = scale_breakpoints(T, scales, int(g.nt))
        info["suggested"] = sug if len(sug) > 2 and (rep is None or rep > tol) else None
    if info["suggested"] is None and rep is not None and rep > tol and S._rep_by_node:
        info["suggested"] = _suggested(S, bp, tol)
    res.panels = info


def _rates(scales) -> Optional[dict]:
    return None if not scales else {"start": scales["start"], "end": scales["end"], "decay": scales["decay"]}


def _after_one_panel(S0, num, first, first_error, scales, pred, budget, run, diagnostics, build, T, verbose):
    """The one panel's result, returned as it is when it passes or fails by what more nodes fix (or by what grading does
    not cut: the trial), else refined from a graded grid; a singular one panel is graded when the trial's grid is regular."""
    model = S0.model
    tol = S0.settings.resolution_tol
    nodes = int(S0.c.g.nt)
    info = {"route": "one panel", "rates": _rates(scales), "predicted_one_panel": pred}
    if first is not None:
        keep = False
        if not first.converged:
            keep = True                                # a bound or a stall: not a question of resolution
        elif not first.representation_error and kernel_tail(first) <= TAIL_ABOVE:
            keep = True                                # without the diagnostics: resolved kernels need no best response to say so
        elif first.representation_error and max(first.representation_error.values()) <= max(tol, GRADE_ABOVE):
            keep = True                                # resolved, or short of it by what more nodes fix: as before
        elif not _grading_helps(S0, first, num, T, build):
            keep = True                                # an error grading does not cut: not a matter of time scales, as before
        else:
            rep0 = max(first.representation_error.values()) if first.representation_error else S0.representation_probe(first)
            keep = rep0 <= max(tol, GRADE_ABOVE)
        if keep:
            _note_one_panel(S0, first, info, scales, T, nodes)
            return first
        info["history"] = [(1, S0.c.N, rep0)]
    else:
        if not _regular_when_graded(S0, num, T, build):
            raise first_error                          # singular for another reason (graded panels do not change it)
        info["history"] = [(1, S0.c.N, None)]
    bp = scale_breakpoints(T, scales, nodes) if scales else [0.0, T]
    if len(bp) <= 2:
        bp = graded_breakpoints(T, 2.0 ** np.floor(np.log2(T / 4.0)))
    while len(bp) > 2 and not budget.fits(bp):
        bp = bp[:len(bp) // 2] + bp[len(bp) // 2 + 1:] if len(bp) > 3 else [0.0, T]   # drop the middle cut
    if len(bp) <= 2:                                   # no graded grid within the budget
        if first is None:
            raise first_error
        _note_one_panel(S0, first, info, scales, T, nodes)
        return first
    info["route"] = "refined"
    budget.first = budget.memory(bp)
    prev_evals = first.evaluations if first is not None else 0
    try:
        S, num1 = build(model, _with_bp(num, bp))
        start = S.interpolate_maps(first) if first is not None else None
        if verbose:
            print(f"  -- the one panel is under-resolved or singular: {len(bp) - 1} panels {_fmt(bp)}, {S.c.N} nodes", flush=True)
        res = S.solve(start_from=start, diagnostics=False, **run)
    except ValueError as exc:
        if not _singular(exc):
            raise
        res = None
    if res is None or (not res.converged and first is not None):
        if first is None:
            if res is None:
                raise first_error
        else:
            _note_one_panel(S0, first, info, scales, T, nodes)
            return first
    res.evaluations += prev_evals
    return _refine(S, res, bp, budget, info, run, diagnostics, build, num, T, verbose)


def _note_one_panel(S0, res, info, scales, T, nodes):
    """res.panels for a result left on the one panel: the time-scale grid suggested when the check failed."""
    info = dict(info)
    rep = max(res.representation_error.values()) if res.representation_error else None
    tol = S0.settings.resolution_tol
    info.update(breakpoints=[0.0, T], history=[(1, S0.c.N, rep)], resolved=None if rep is None else rep <= tol)
    sug = None
    if rep is not None and rep > tol:
        sug = _suggested(S0, [0.0, T], tol) if S0._rep_by_node else None
    info["suggested"] = sug
    res.panels = info


def _discard(S) -> None:
    """Drop an engine's grid from the shared cache (grid_cache.discard): a grid the refinement has moved past."""
    if S is not None:
        from .grid_cache import triangle_grid
        triangle_grid.discard(S.c.g)


def _indicators(S, P: int) -> np.ndarray:
    """The largest relative representation error on each breakpoint interval: a piece's nodes count for its time panel
    and its age panel (S._rep_by_node, filled by the representation probe or the diagnostics)."""
    vals = [v for v in S._rep_by_node.values() if v is not None]
    e = np.zeros(P)
    if not vals:
        return e
    node = np.max(vals, axis=0)
    for pc in S.c.g.pieces:
        v = float(node[pc.offset:pc.offset + pc.n].max())
        e[pc.p] = max(e[pc.p], v)
        e[pc.q] = max(e[pc.q], v)
    return e


def _split(bp, ks) -> List[float]:
    return [float(b) for b in sorted(set(bp) | {0.5 * (bp[k] + bp[k + 1]) for k in ks})]


def _suggested(S, bp, tol) -> Optional[List[float]]:
    """The breakpoints predicted to pass the check: the intervals above tol bisected, each halving assumed to cut an
    interval's error HALVING_GAIN-fold, until none is predicted above it (at most 12 halvings of any interval)."""
    e = _indicators(S, len(bp) - 1)
    if not (e > tol).any():
        return None
    cur = [(bp[k], bp[k + 1], e[k]) for k in range(len(bp) - 1)]
    for _ in range(12):
        nxt, more = [], False
        for lo, hi, v in cur:
            if v > tol:
                mid = 0.5 * (lo + hi)
                nxt += [(lo, mid, v / HALVING_GAIN), (mid, hi, v / HALVING_GAIN)]
                more = True
            else:
                nxt.append((lo, hi, v))
        cur = nxt
        if not more:
            break
    return [float(b) for b in sorted({lo for lo, _, _ in cur} | {cur[-1][1]})]


def _refine(S, res, bp, budget: _Budget, info: dict, run: dict, diagnostics: bool, build, num, T: float, verbose: bool):
    """The local refinement from the result `res` of engine S on `bp` (module docstring, 2), then the diagnostics of the
    best grid, res.message, res.panels, and the warning when the check is not reached within the budget."""
    model = S.model
    tol = S.settings.resolution_tol
    history = list(info.get("history", []))
    total_evals = res.evaluations
    rep = S.representation_probe(res)
    history.append((len(bp) - 1, S.c.N, rep))
    best = (rep, res, S, bp)
    stop, suggestion, slow = None, None, 0
    for _ in range(MAX_ROUNDS):
        if rep <= tol:
            break
        e = _indicators(S, len(bp) - 1)
        want = [int(k) for k in np.argsort(-e) if e[k] > tol]
        if not want:
            stop = "the error is not on any interval's nodes"
            break
        take = []
        for k in want:                                 # the worst intervals first, as many as the budget holds
            if budget.fits(_split(bp, take + [k]), refinement=True, held=best[3]):
                take.append(k)
        if not take:
            full = _split(bp, want)
            suggestion = _suggested(S, bp, tol)
            stop = f"the next round ({len(full) - 1} panels, {budget.unknowns(full)} unknowns, about " \
                   f"{budget.memory(full, best[3]):.0f} MB with the best grid's result kept) would pass {budget.why(full, best[3])}"
            break
        bp2 = _split(bp, take)
        S2, _ = build(model, _with_bp(num, bp2))
        start = S2.interpolate_maps(res)
        if best[1] is not res:                        # the current grid is not the best: free it before the next solve
            _discard(S); S = res = None
        if verbose:
            print(f"  -- refining {len(take)} of {len(bp) - 1} panels (representation error {rep:.1e}): {len(bp2) - 1} panels "
                  f"{_fmt(bp2)}, {S2.c.N} nodes", flush=True)
        try:
            res2 = S2.solve(start_from=start, diagnostics=False, **run)
        except ValueError as exc:
            if not _singular(exc):
                raise
            history.append((len(bp2) - 1, budget.nodes(bp2), None))
            stop = "the refined grid's best response is singular"
            _discard(S2)
            break
        start = None
        total_evals += res2.evaluations
        if not res2.converged:
            history.append((len(bp2) - 1, S2.c.N, None))
            stop = "the refined solve did not converge"
            _discard(S2)
            break
        rep2 = S2.representation_probe(res2)
        history.append((len(bp2) - 1, S2.c.N, rep2))
        slow = slow + 1 if rep2 > 0.5 * rep else 0
        if rep2 < best[0]:                            # the old best is past: its grid goes (its result with it)
            _discard(best[2])
            best = (rep2, res2, S2, bp2)
        S, res, rep, bp = S2, res2, rep2, bp2
        if slow >= 2:
            stop = "the error stopped falling with the panels (not a matter of time scales)"
            break
    else:
        if rep > tol:
            stop = f"{MAX_ROUNDS} refinement rounds"
    if S is not None and S is not best[2]:
        _discard(S)
    rep, res, S, bp = best
    if rep > tol and suggestion is None and not (stop or "").startswith(("the error stopped", "the refined grid's")):
        suggestion = _suggested(S, bp, tol)
    if diagnostics:
        res.solve_kw.pop("diagnostics", None)
        S._diagnostics(res)
    res.evaluations = total_evals
    tried = "; ".join(f"{p} panel{'s' if p > 1 else ''}: " + ("singular" if r is None else f"{r:.1e}") for p, _, r in history)
    how = ("from the model's time scales (start rate {:.3g}, end rate {:.3g})".format(info["rates"]["start"], info["rates"]["end"])
           if info["route"] == "time scales" else "refined from a one panel of [0, {:g}] under-resolved or singular".format(T))
    rounds = len(history) - (2 if info["route"] == "refined" else 1)
    res.message += (f"; panels graded {how}" + (f", refined locally where the error sat ({rounds} round{'s' if rounds > 1 else ''})"
                                                if rounds > 0 else "")
                    + f": numerics.breakpoints {_fmt(bp)} (representation error by grid: {tried})")
    info = dict(info)
    info.update(breakpoints=list(bp), history=history, resolved=bool(rep <= tol), suggested=suggestion if rep > tol else None,
                stopped=stop if rep > tol else None)
    if info["route"] == "time scales" and len(history) > 1:
        info["route"] = "refined"
    res.panels = info
    if rep > tol:
        _warn_unresolved(model, res, rep, tol, bp, suggestion, stop, budget)
    return res


def _warn_unresolved(model, res, rep, tol, bp, suggestion, stop, budget):
    if suggestion:
        cost = f"{budget.unknowns(suggestion)} unknowns, about {budget.memory(suggestion):.0f} MB"
        todo = (f"re-solve with numerics.breakpoints {_fmt(suggestion)} ({cost}; res.sharpen() does it from this result), "
                "or raise settings.auto_panels_growth / auto_panels_max / auto_panels_memory")
    else:
        todo = "raise numerics.nodes, or rescale time so that the fast rates are of order one"
    warnings.warn(f"{model.name}: representation error {rep:.1e} on {len(bp) - 1} time panels {_fmt(bp)}, above "
                  f"settings.resolution_tol {tol:g}; stopped: {stop}.  For the resolved answer {todo}", stacklevel=4)


def graded_breakpoints(T: float, w: float, growth: float = 2.0) -> List[float]:
    """Breakpoints on [0, T] with panels of width w, growth w, growth^2 w, ... from 0 and, mirrored, from T, meeting in
    one middle panel (merged into its neighbours' when it would be shorter than half the widest of them)."""
    T, w = float(T), float(w)
    if not 0 < w < T / 2:
        return [0.0, T]
    left, step = [0.0], w
    while left[-1] + step < T / 2 - 1e-12 * T:
        left.append(left[-1] + step)
        step *= growth
    # the middle panel [left[-1], T - left[-1]]: too short next to its neighbours, drop the innermost cut on each side
    if len(left) > 2 and T - 2 * left[-1] < 0.5 * (left[-1] - left[-2]):
        left.pop()
    pts = sorted({round(b, 12) for b in left} | {round(T - b, 12) for b in left})
    return [float(b) for b in pts]


def _grading_helps(S0, first, num, T: float, build) -> bool:
    """The trial that keeps a failure grading cannot fix, or that more nodes fix better, from paying for a graded fixed
    point: one best response per agent to the one panel's maps, read on the one panel at TRIAL_NODES (4) nodes, on thirteen
    graded panels (ends of T / 128 to T / 256) at 4, and on the one panel at 8; grading is taken when it
    cuts the representation error by GRADE_GAIN and below what the doubled nodes give.  Kyle-Back with a random-walk value
    the trader sees stays near 0.4 graded at T = 1; Chapter 1's game at 4 nodes is resolved by 8 (a node count, not a time
    scale); the regulator at T = 10 falls from 0.1 to 1e-4 graded, to 2e-2 at 12 nodes."""
    from types import SimpleNamespace
    from .numerics import Numerics
    n = TRIAL_NODES
    reps = []
    for nodes, bp in ((n, None), (n, graded_breakpoints(T, 2.0 ** np.floor(np.log2(T / TRIAL_SPLIT)))), (2 * n, None)):
        S, _ = build(S0.model, Numerics.of(num.to_dict()).merged(Numerics(nodes=nodes) if bp is None else Numerics(nodes=nodes, breakpoints=bp)))
        try:
            reps.append(S.representation_probe(SimpleNamespace(maps=S.interpolate_maps(first))))
        except ValueError as exc:
            if not _singular(exc):
                raise
            reps.append(np.inf)
    return reps[1] < GRADE_GAIN * reps[0] and reps[1] < reps[2]


def _regular_when_graded(S0, num, T: float, build) -> bool:
    """Whether a one panel's singular best response is a matter of time scales: the best responses to the zero start on the
    trial's graded grid (_grading_helps') are regular.  A threshold that refuses every system (settings.foc_rcond 1) is
    singular there too, and the solve raises at once instead of grading down to the budget."""
    from .numerics import Numerics
    S, _ = build(S0.model, Numerics.of(num.to_dict()).merged(
        Numerics(nodes=TRIAL_NODES, breakpoints=graded_breakpoints(T, 2.0 ** np.floor(np.log2(T / TRIAL_SPLIT))))))
    try:
        zero = S.zero_maps()
        for a in S.model.agents:
            if S.c.rep[a.name] == a.name:
                S.best_response(a, zero, project=False)
    except ValueError as exc:
        if not _singular(exc):
            raise
        return False
    return True


def kernel_tail(res) -> float:
    """How far the equilibrium's kernels are from resolved on their pieces: on every piece of the triangle, the largest of the
    Chebyshev coefficients of the two highest degrees in time or in age (the interpolant's own error estimate), relative to
    the kernel's peak, the largest over the primaries and the world's columns.  A kernel resolved to eight digits has it
    near 1e-8 (1.4e-5 on the one-agent regulator at T = 1, 12 nodes); one that varies on a time scale far below the
    panel's width has it near its own size (9e-3 at T = 10, 0.15 at T = 30)."""
    from numpy.polynomial import chebyshev as C
    g = res.compiled.g
    nP = len(res.compiled.prim)
    Z = np.asarray(res.world)[:, :g.N if res.world.ndim == 1 else res.world.shape[1]]
    Z = Z.reshape(nP, g.N, -1)
    peak = np.abs(Z).max(axis=(1, 2))
    inv = {}
    worst = 0.0
    for pc in g.pieces:
        key = (pc.nt, pc.na)
        if key not in inv:
            x = lambda n: np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
            inv[key] = (np.linalg.inv(C.chebvander(x(pc.nt), pc.nt - 1)), np.linalg.inv(C.chebvander(x(pc.na), pc.na - 1)))
        Vt, Va = inv[key]
        V = Z[:, pc.offset:pc.offset + pc.n].reshape(nP, pc.nt, pc.na, -1)
        c = np.einsum("ij,pjkc,lk->pilc", Vt, V, Va)
        top = np.maximum(np.abs(c[:, -2:]).max(axis=(1, 2, 3)), np.abs(c[:, :, -2:]).max(axis=(1, 2, 3)))
        worst = max(worst, float(np.max(top / np.maximum(peak, 1e-300))))
    return worst
