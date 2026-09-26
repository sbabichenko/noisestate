"""Anderson-accelerated fixed-point iteration for the best-response map.

x_{k+1} = x_k + beta r_k  with the increment corrected by a least-squares
combination of the last M secant pairs (Tikhonov-regularised, so identically
zero or degenerate components of the map cannot break the secant system).
The history is reset when the residual grows, and the step is damped back
towards plain mixing if the accelerated step would increase the residual.
This is the outer solver used by the Chapter 5 spectral market solver.
"""
from __future__ import annotations

import time
from typing import Callable, List

import numpy as np


def _finite_or_raise(rn: float, evals: int) -> None:
    """Every decision of the iteration is a comparison with the residual, and every comparison with NaN is
    False: a non-finite residual would run the whole budget on NaN and hand NaN to the polish.  It is an
    overflow in the best-response map, so it is a solver error naming the evaluation."""
    if not np.isfinite(rn):
        raise RuntimeError(f"the best-response map returned a non-finite value at evaluation {evals} (residual {rn}): an "
                           "overflow in the model's dynamics or loss, usually a discount rate or a drift times the window "
                           "above about 700; check the parameter scales")


def anderson(F: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, tol: float = 1e-10, M: int = 6,
             beta: float = 0.5, maxiter: int = 200, reg: float = 1e-8, verbose: bool = False):
    """Solve F(x) = 0 for the residual F(x) = G(x) - x of a fixed-point map G.
    Returns (x, residual_norm_relative, evaluations, converged, stalled); stalled means the best
    residual stopped improving (by less than 30% over 20 iterations), which is what a noise floor
    of F looks like, and the iteration was cut short.  A non-finite residual raises RuntimeError."""
    x = np.array(x0, dtype=float)
    r = F(x); evals = 1
    dX: List[np.ndarray] = []; dR: List[np.ndarray] = []
    rn = np.linalg.norm(r) / max(1.0, np.linalg.norm(x)); _finite_or_raise(rn, evals)
    best_x, best_rn = x.copy(), rn
    best_hist = [rn]
    for k in range(maxiter):
        if verbose:
            print(f"  anderson {k:3d} rel resid {rn:.3e} (history {len(dX)})", flush=True)
        if rn < tol:
            return x, rn, evals, True, False
        if k >= 40 and best_rn > 0.7 * best_hist[-20]:
            return best_x, best_rn, evals, False, True
        if dX:
            Rm = np.stack(dR, axis=1)                                   # (n, m)
            A = Rm.T @ Rm
            A += reg * (np.trace(A) / A.shape[0] + 1e-300) * np.eye(A.shape[0])
            gamma = np.linalg.solve(A, Rm.T @ r)
            Xm = np.stack(dX, axis=1)
            x_new = x + beta * r - (Xm + beta * Rm) @ gamma
        else:
            x_new = x + beta * r
        r_new = F(x_new); evals += 1
        rn_new = np.linalg.norm(r_new) / max(1.0, np.linalg.norm(x_new)); _finite_or_raise(rn_new, evals)
        if rn_new > 2.0 * rn and dX:
            # accelerated step failed: reset history and take a plain damped step
            dX.clear(); dR.clear()
            x_new = x + beta * r
            r_new = F(x_new); evals += 1
            rn_new = np.linalg.norm(r_new) / max(1.0, np.linalg.norm(x_new)); _finite_or_raise(rn_new, evals)
        dX.append(x_new - x); dR.append(r_new - r)
        if len(dX) > M:
            dX.pop(0); dR.pop(0)
        x, r, rn = x_new, r_new, rn_new
        if rn < best_rn:
            best_x, best_rn = x.copy(), rn
        best_hist.append(best_rn)
    return best_x, best_rn, evals, best_rn < tol, False


class ResultValidationError(RuntimeError):
    """A result is not fit for the use asked of it.  The two reasons are siblings, never nested:

        ConvergenceError   the solve did not reach its tolerance
        DiagnosticsError   it converged, and the assessment was still not accepted

    A converged result can fail diagnostics, so DiagnosticsError must not be a ConvergenceError --
    `except ConvergenceError` would then catch failures with nothing to do with convergence.
    A caller wanting either writes `except ResultValidationError`.
    """


class ConvergenceError(ResultValidationError):
    """The solve did not reach its tolerance.  Raised by require_converged(), and by require_ok()
    when the `converged` check is the one that blocked."""


class DiagnosticsError(ResultValidationError):
    """It converged, and require_ok()'s assessment was still not accepted.  Carries the
    Assessment as .assessment."""

    def __init__(self, message, assessment=None):
        super().__init__(message)
        self.assessment = assessment


class _Stop(Exception):
    """Raised inside the fixed-point map when the evaluation budget or the deadline is reached, so the
    iteration (Anderson's or scipy's) unwinds to solve_fixed_point, which returns the best iterate."""


def solve_fixed_point(F, z0, tol: float = 1e-10, verbose: bool = False, damping: float = 0.5,
                      anderson_iters: int = 150, max_newton: int = 30, M: int = 6, reg: float = 1e-8, inner_m: int = 15,
                      max_evaluations=None, deadline=None, progress=None, t0=None):
    """Regularised Anderson mixing on the residual F(z) = G(z) - z, then a Newton-Krylov polish if it
    stalls above tol.  Returns (z, residual, evaluations, converged, message), where the residual is
    norm(F(z)) / max(1, norm(z)) and converged means residual <= tol.
    anderson_iters, M and reg are the Anderson iterations, memory and regularisation; inner_m the fresh
    Krylov vectors per Newton step (the engines pass their Settings' values).
    max_evaluations bounds the evaluations of F over both phases and deadline the wall time in seconds
    since t0 (default: now); at least one evaluation is made, and past either bound the best iterate so
    far is returned, with the message naming the bound.  progress(info) is called after every evaluation
    with {"evaluation", "residual", "phase", "seconds"}; an exception it raises propagates."""
    t0 = time.time() if t0 is None else t0
    phase = ["anderson"]
    state = {"evals": 0, "best_x": None, "best_rn": np.inf}
    if verbose:
        # one line per evaluation, the elapsed time first, whatever the phase (scipy's own verbose output
        # is its own format and is left off); the caller's progress still gets every evaluation
        user_progress, shown = progress, [None]

        def progress(info):
            if info["phase"] != shown[0]:
                shown[0] = info["phase"]
                print(f"  -- {info['phase']}" + (f" polish (anderson stopped at {state['best_rn']:.2e})"
                                                 if info["phase"] == "newton" else ""), flush=True)
            print(f"  [{info['seconds']:7.2f}s] {info['phase']:<8} eval {info['evaluation']:4d}  resid {info['residual']:.3e}",
                  flush=True)
            if user_progress is not None:
                user_progress(info)

    def Fb(x):
        # the map with the bookkeeping: the bounds are checked before an evaluation (so one is always made),
        # the best iterate is kept for a stop, the residual is reported
        k = state["evals"]
        if k and max_evaluations is not None and k >= max_evaluations:
            raise _Stop(f"stopped at the evaluation budget (max_evaluations={max_evaluations})")
        if k and deadline is not None and time.time() - t0 >= deadline:
            raise _Stop(f"stopped at the deadline ({deadline:g} s)")
        try:
            r = F(x)
        except Exception as e:                        # the map's own errors (a singular system) pass through the polish
            e._from_map = True
            raise
        state["evals"] = k + 1
        rn = float(np.linalg.norm(r) / max(1.0, np.linalg.norm(x))); _finite_or_raise(rn, k + 1)
        if rn < state["best_rn"]:
            state["best_x"], state["best_rn"] = np.array(x, dtype=float), rn
        if progress is not None:
            progress({"evaluation": k + 1, "residual": rn, "phase": phase[0], "seconds": time.time() - t0})
        return r

    msg = []
    try:
        z, rn, ev, ok, stalled = anderson(Fb, z0, tol=tol, M=M, beta=damping, maxiter=anderson_iters, reg=reg)
    except _Stop as stop:
        z, rn, ev = state["best_x"], state["best_rn"], state["evals"]
        return z, rn, ev, rn <= tol, f"anderson: {ev} evaluations, residual {rn:.2e}; {stop}"
    msg.append(f"anderson: {ev} evaluations, residual {rn:.2e}" + (" (stalled: the residual stopped improving, a noise floor of the fixed-point map)" if stalled else ""))
    if ok or max_newton <= 0 or (stalled and rn < 100 * tol):
        # a stall within two decades of the tolerance is a floor; a Newton polish cannot beat noise
        return z, rn, ev, rn <= tol, "; ".join(msg)
    phase[0] = "newton"
    from scipy.optimize import newton_krylov, NoConvergence     # here, not at the top: 45 ms a process, and Anderson usually converges
    try:
        try:
            # scipy's KrylovJacobian replaces LGMRES's outer loop by the Newton steps (maxiter 1), so its
            # inner_maxiter is not the budget of a step: inner_m bounds the fresh Krylov vectors (one
            # evaluation of F each); a step also re-multiplies the up to outer_k=10 directions carried from
            # earlier steps (store_outer_Av is False) and makes the line search's, 16 to 26 in all
            z2 = newton_krylov(Fb, z, f_tol=tol * max(1.0, float(np.linalg.norm(z))), maxiter=max_newton,
                               method="lgmres", inner_inner_m=inner_m)
            note = "newton polish: {n} evaluations, residual {r:.2e}"
        except NoConvergence as e:
            z2 = np.asarray(e.args[0])
            note = "newton polish stopped without converging after {n} evaluations (residual {r:.2e})"
        except ValueError as e:                       # scipy's Krylov Jacobian on a step it cannot invert
            if getattr(e, "_from_map", False):
                raise
            z2 = z
            note = f"newton polish failed ({e}) after {{n}} evaluations (residual {{r:.2e}})"
        r2 = float(np.linalg.norm(Fb(z2)) / max(1.0, np.linalg.norm(z2)))
        msg.append(note.format(n=state["evals"] - ev, r=r2))
    except _Stop as stop:
        z2, r2 = state["best_x"], state["best_rn"]
        msg.append(f"newton polish: {state['evals'] - ev} evaluations; {stop}")
    if r2 < rn:
        z, rn = z2, r2
    return z, rn, state["evals"], rn <= tol, "; ".join(msg)
