"""A brute-force discrete-time reference for a risk-averse (CARA) agent in a STATIONARY discounted game under consistent
planning, independent of the package: every date's self minimises the entropic cost of its own discounted continuation
given what it has seen, the later selves playing the stationary map.  One agent, one state:

  X_{n+1} = (1 + a h) X_n + u_n h + sigma sqrt(h) xi_n                   (the state; X at t_n = n h)
  y_n     = sqrt(h) xi_n                                     obs="noise"  (the agent sees the state's shocks)
          = sqrt(p) X_n h + sqrt(h) eta_n                    obs="signal" (a noisy signal of the state)
  u_n     = sum_{l >= 1} g_l y_{n - l}                                    (the stationary map on the agent's own past)
  C_0     = sum_{n >= 0} e^{-rho n h} h (X_n^2 + r u_n^2)                 (the date-0 continuation; the past is sunk)

The process starts at n = -P from X = 0 (the past, long enough for the stationary law) and runs to n = M - 1 (the
discount makes the rest negligible).  The date-0 self plays u_0 = sum_l a_l y_{-l} and later selves the map g: with the
signal, a deviation at 0 moves X and so the later selves' observations, and they react (no envelope: their objective is
not the date-0 self's).  Every quantity is a row over the shocks g (xi, eta from -P to M - 1) and the date-0 action e.

Consistent planning.  Given the observations y_{<0} (the projector Pi on their span, Sigma = I - Pi), C_0 = g'M g with M
= T'FT, T = [I; a'Y] (Y the rows of y_{-1}, y_{-2}, ...), and theta^-1 log E[e^{theta C_0} | F_0] = g'Pi N Pi g + const
with N = M + 2 theta M Sigma (I - 2 theta Sigma M Sigma)^-1 Sigma M.  Sigma M Sigma does not depend on a (u_0 is seen), so
the expectation of that quadratic over the past, the date-0 self's objective, is quadratic in a:

  phi(a) = const + 2 y'v + (F_ee + 2 theta f'B f) y'y,   y = Y'a,  v = f + 2 theta Pi F_ss B f,  B = Sigma (I - 2 theta
  Sigma F_ss Sigma)^-1 Sigma,

f = F_se, and the best response is a = -(F_ee + 2 theta f'Bf)^-1 (Y Y')^-1 Y v (theta = 0: the risk-neutral projection).
The equilibrium is the fixed point g = a (damped iteration)."""
import numpy as np


class StationaryLQ:
    def __init__(self, h=0.05, past=10.0, future=36.0, a=0.0, sigma=1.0, r=0.1, rho=0.5, theta=0.0, obs="noise", p=3.0):
        self.h = float(h); self.P = int(round(past / h)); self.M = int(round(future / h))
        self.a, self.sigma, self.r, self.rho, self.theta, self.obs, self.p = float(a), float(sigma), float(r), float(rho), float(theta), obs, float(p)
        self.T = self.P + self.M                        # steps n = -P .. M - 1, index i = n + P
        self.nch = 1 if obs == "noise" else 2
        self.D = self.nch * self.T                      # shock coordinates: xi_i (i < T), then eta_i
        self.w = np.exp(-self.rho * self.h * np.arange(self.M))

    def forward(self, g, e_col=True):
        """Rows over (shocks, e) of X_n and u_n for n = -P .. M - 1, the later selves and the past ones playing g (g_l for
        l = 1 .. len(g)), the date-0 action the coordinate e."""
        h, T, D, P = self.h, self.T, self.D, self.P
        X = np.zeros((T + 1, D + 1)); U = np.zeros((T, D + 1)); Y = np.zeros((T, D + 1))
        L = len(g)
        for i in range(T):
            n = i - P
            if self.obs == "noise":
                Y[i, i] = np.sqrt(h)
            else:
                Y[i] = np.sqrt(self.p) * h * X[i]; Y[i, T + i] += np.sqrt(h)
            if n == 0:
                U[i, D] = 1.0
            else:
                lo = max(0, i - L)
                if i > lo:
                    if self.obs == "noise":                                # y_j = sqrt(h) e_j: the row directly
                        U[i, lo:i] = np.sqrt(h) * g[:i - lo][::-1]
                    else:
                        U[i] = g[:i - lo][::-1] @ Y[lo:i]                  # sum_l g_l y_{i - l}
            X[i + 1] = (1.0 + self.a * h) * X[i] + h * U[i]
            X[i + 1, i] += self.sigma * np.sqrt(h)
        return X[:T], U, Y

    def best_response(self, g):
        h, P, D, th = self.h, self.P, self.D, self.theta
        X, U, Y = self.forward(g)
        Xf, Uf = X[P:], U[P:]                            # n = 0 .. M - 1
        wx = (self.w * h)[:, None]
        F = Xf.T @ (wx * Xf) + self.r * Uf.T @ (wx * Uf)     # C_0 = (g, e)' F (g, e)
        Fss, f, Fee = F[:D, :D], F[:D, D], F[D, D]
        Yall = Y[:P, :D]                                 # every past observation: the date-0 self's information
        Q, R = np.linalg.qr(Yall.T)                      # an orthonormal basis of their span (Yall has full row rank)
        Pi = lambda x: Q @ (Q.T @ x)
        Sg = lambda x: x - Pi(x)
        if th:
            A = np.eye(D) - 2.0 * th * (Fss - Pi(Fss) - Pi(Fss.T).T + Q @ (Q.T @ Pi(Fss.T).T))     # I - 2 theta Sigma F Sigma
            Bf = Sg(np.linalg.solve(A, Sg(f)))
            v = f + 2.0 * th * Pi(Fss @ Bf)
            den = Fee + 2.0 * th * f @ Bf
        else:
            v, den = f, Fee
        Ywin = Y[:P][::-1][:len(g), :D]                  # y_{-1}, y_{-2}, ... : the rows the map reads
        a = -np.linalg.lstsq(Ywin.T, v, rcond=None)[0] / den
        return a

    def equilibrium(self, g0=None, tol=1e-11, maxit=400, damping=0.7, verbose=False):
        g = np.zeros(self.P) if g0 is None else np.asarray(g0, float).copy()
        for it in range(maxit):
            a = self.best_response(g)
            ch = float(np.abs(a - g).max())
            g = damping * a + (1 - damping) * g
            if verbose:
                print(it, ch, flush=True)
            if ch < tol:
                break
        return g, ch

    def record(self, g, ages=(0.0, 0.25, 0.5, 1.0, 2.0)):
        """At the last past date n = -1 (every self there plays g): the action's kernel on the state's shock at the ages (u's row
        on xi_{n - 1 - m} over sigma sqrt(h), m = age / h, the shock's step ending m steps before), and the stationary flow
        cost E X^2 + r E u^2."""
        h, P, D = self.h, self.P, self.D
        X, U, Y = self.forward(g)
        n0 = P - 1
        out = {"h": h, "ages": list(ages)}
        out["D_w0"] = [float(U[n0, n0 - 1 - int(round(a / h))] / (self.sigma * np.sqrt(h))) for a in ages]
        out["flow"] = float(X[n0, :D] @ X[n0, :D] + self.r * (U[n0, :D] @ U[n0, :D]))
        # E[J_0 | F_0] - E C_0 over the past: 2 theta tr(Pi M B M Pi) - (2 theta)^-1 log det(I - 2 theta Sigma M Sigma) - tr(Sigma M Sigma)
        # with M the date-0 continuation's form when the date-0 self plays g too (e replaced by its row)
        th = self.theta
        if th:
            Ywin = Y[:P][::-1][:len(g), :D]
            e = g @ Ywin
            Xf = X[P:, :D] + np.outer(X[P:, D], e); Uf = U[P:, :D] + np.outer(U[P:, D], e)
            wx = (self.w * h)[:, None]
            M = Xf.T @ (wx * Xf) + self.r * Uf.T @ (wx * Uf)
            Q, _ = np.linalg.qr(Y[:P, :D].T)
            Sg = np.eye(D) - Q @ Q.T
            SMS = Sg @ M @ Sg
            mu = np.linalg.eigvalsh(0.5 * (SMS + SMS.T))
            A = np.eye(D) - 2.0 * th * SMS
            MQ = M @ Q
            BMQ = Sg @ np.linalg.solve(A, Sg @ MQ)
            out["cond_excess"] = float(2.0 * th * np.sum(MQ * BMQ) + np.sum(-np.log1p(-2.0 * th * mu) - 2.0 * th * mu) / (2.0 * th))
        return out


def richardson(values, hs):
    A = np.array([[h ** k for k in range(len(hs))] for h in hs])
    return float(np.linalg.solve(A, np.asarray(values, float))[0])
