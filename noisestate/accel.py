"""Anderson-accelerated fixed-point iteration for the best-response map.

x_{k+1} = x_k + beta r_k  with the increment corrected by a least-squares
combination of the last M secant pairs (Tikhonov-regularised, so identically
zero or degenerate components of the map cannot break the secant system).
The history is reset when the residual grows, and the step is damped back
towards plain mixing if the accelerated step would increase the residual.
This is the outer solver used by the Chapter 5 spectral market solver.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


def anderson(F: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, tol: float = 1e-10, M: int = 6,
             beta: float = 0.5, maxiter: int = 200, reg: float = 1e-8, verbose: bool = False):
    """Solve F(x) = 0 for the residual F(x) = G(x) - x of a fixed-point map G.
    Returns (x, residual_norm_relative, evaluations, converged, stalled); stalled means the best
    residual stopped improving (by less than 30% over 20 iterations), which is what a noise floor
    of F looks like, and the iteration was cut short."""
    x = np.array(x0, dtype=float)
    r = F(x); evals = 1
    dX: List[np.ndarray] = []; dR: List[np.ndarray] = []
    rn = np.linalg.norm(r) / max(1.0, np.linalg.norm(x))
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
        rn_new = np.linalg.norm(r_new) / max(1.0, np.linalg.norm(x_new))
        if rn_new > 2.0 * rn and dX:
            # accelerated step failed: reset history and take a plain damped step
            dX.clear(); dR.clear()
            x_new = x + beta * r
            r_new = F(x_new); evals += 1
            rn_new = np.linalg.norm(r_new) / max(1.0, np.linalg.norm(x_new))
        dX.append(x_new - x); dR.append(r_new - r)
        if len(dX) > M:
            dX.pop(0); dR.pop(0)
        x, r, rn = x_new, r_new, rn_new
        if rn < best_rn:
            best_x, best_rn = x.copy(), rn
        best_hist.append(best_rn)
    return best_x, best_rn, evals, best_rn < tol, False


class ConvergenceError(RuntimeError):
    """Raised by Result.check() when a solve did not reach its tolerance."""


def solve_fixed_point(F, z0, tol: float = 1e-10, verbose: bool = False, damping: float = 0.5,
                      anderson_iters: int = 150, max_newton: int = 30, M: int = 6):
    """Regularised Anderson mixing on the residual F(z) = G(z) - z, then a Newton-Krylov polish if it
    stalls above tol.  Returns (z, residual, evaluations, converged, message), where the residual is
    norm(F(z)) / max(1, norm(z)) and converged means residual <= tol."""
    from scipy.optimize import newton_krylov, NoConvergence
    msg = []
    z, rn, ev, ok, stalled = anderson(F, z0, tol=tol, M=M, beta=damping, maxiter=anderson_iters, verbose=verbose)
    msg.append(f"anderson: {ev} evaluations, residual {rn:.2e}" + (" (stalled: the residual stopped improving, a noise floor of the fixed-point map)" if stalled else ""))
    if ok or max_newton <= 0 or (stalled and rn < 100 * tol):
        # a stall within two decades of the tolerance is a floor; a Newton polish cannot beat noise
        return z, rn, ev, rn <= tol, "; ".join(msg)
    cnt = [0]

    def Fc(x):
        cnt[0] += 1
        return F(x)
    try:
        z2 = newton_krylov(Fc, z, f_tol=tol * max(1.0, float(np.linalg.norm(z))), maxiter=max_newton,
                           method="lgmres", inner_maxiter=15, verbose=verbose)
        r2 = float(np.linalg.norm(F(z2)) / max(1.0, np.linalg.norm(z2))); cnt[0] += 1
        msg.append(f"newton polish: {cnt[0]} evaluations, residual {r2:.2e}")
        if r2 < rn:
            z, rn = z2, r2
    except NoConvergence as e:
        z2 = np.asarray(e.args[0]); r2 = float(np.linalg.norm(F(z2)) / max(1.0, np.linalg.norm(z2))); cnt[0] += 1
        msg.append(f"newton polish stopped without converging after {cnt[0]} evaluations (residual {r2:.2e})")
        if r2 < rn:
            z, rn = z2, r2
    return z, rn, ev + cnt[0], rn <= tol, "; ".join(msg)
