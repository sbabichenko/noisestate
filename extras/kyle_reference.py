"""A brute-force discrete-time reference for a risk-averse (CARA) insider in a finite-horizon Kyle-Back market with a
competitive market maker, independent of the package (the Kyle counterpart of extras/leqg_reference.py).

  V_k = V_0 + sigma_V sqrt(dt) sum_{j<k} gV_j,  V_0 = sqrt(Sigma0) g0      (the value at t_k; V_T = V_n)
  D_k = sum_{j<k} (a_kj gV_j + b_kj gZ_j) + c_k g0                           (an insider's order rate on [t_k, t_k + dt))
  z_k = K D_k dt + sigma_Z sqrt(dt) gZ_k                                     (the order flow of K identical insiders, seen from
                                                                              step k + 1 on; a symmetric equilibrium)
  P_k = sum_{j<k} Lambda_kj z_j                                              (the market maker's quote on step k)

The insider's strategy is any causal linear map of what it has seen: the increments of V before t_k (and V_0 from the
start), and the noise trades before t_k -- equivalently the flow, since it knows its own orders.  The competitive market
maker sets Lambda to the projection P_k = E[V_k | z_0 .. z_{k-1}] of the insider's current strategy (zero-profit
pricing is not an optimisation, so it has no risk attitude).  The insider's realised cost is minus its terminal wealth
marked at V_T, plus the trading cost:

  C = sum_k dt [D_k (P_k - V_T) + eps D_k^2]         ("wealth": the insider liquidates Q_T = sum dt D_k at V_T)
  C = sum_k dt [D_k (P_k - V_k) + eps D_k^2]         ("flow": the loss's fundamental-valued flow; the same mean, a
                                                       different random variable when V moves: they differ by sum Q dV)

C = g' M g in the standardised shocks g, and J = -(2 theta)^-1 log det(I - 2 theta M) (theta > 0), tr M (theta = 0).
The gradient of J in the insider's rows D is dt[(P - 1 V_T') W + Lambda' (dt D) W ... ] (below), W = (I - 2 theta M)^-1,
checked by finite differences in check_gradient().  A best response is L-BFGS over the free entries of the insider's
rows with Lambda fixed; the equilibrium is the fixed point of (insider best response, market maker projection)."""
import numpy as np
from scipy.optimize import minimize


class Kyle:
    def __init__(self, n=50, T=1.0, eps=0.2, sigma_V=1.0, sigma_Z=1.0, Sigma0=0.0, theta=0.0, cost="wealth", insiders=1, rho=0.0):
        self.n, self.T, self.dt = n, float(T), float(T) / n
        self.rho = float(rho)
        self.w = np.exp(-self.rho * self.dt * np.arange(n))            # the discount e^{-rho t_k} of step k
        self.k = int(insiders)                             # identical insiders (a symmetric equilibrium)
        self.rmap = None                                   # the rivals' map (A, c) on what they see: their flow and V_0
        self.eps, self.sV, self.sZ, self.S0, self.theta, self.cost = float(eps), float(sigma_V), float(sigma_Z), float(Sigma0), float(theta), cost
        self.prior = Sigma0 > 0
        # coordinates: gV_0..gV_{n-1}, gZ_0..gZ_{n-1}, (g0)
        self.m = 2 * n + (1 if self.prior else 0)
        self.iV, self.iZ = np.arange(n), n + np.arange(n)
        mask = np.zeros((n, self.m), bool)
        low = np.tril(np.ones((n, n), bool), -1)
        mask[:, :n] = low if self.sV > 0 else False
        mask[:, n:2 * n] = low
        if self.prior:
            mask[:, 2 * n] = True
        self.mask = mask
        dt = self.dt
        V = np.zeros((n + 1, self.m))
        for k in range(1, n + 1):
            V[k] = V[k - 1]; V[k, k - 1] += self.sV * np.sqrt(dt)
        if self.prior:
            V[:, 2 * n] = np.sqrt(self.S0)
        self.V = V                                         # rows of V_0 .. V_n
        self.Nz = np.zeros((n, self.m)); self.Nz[np.arange(n), n + np.arange(n)] = self.sZ * np.sqrt(dt)

    # ------------------------------------------------------------------ the closed loop
    def rival_map(self, D):
        """(A, c): the insiders' strategy D as a map on what an insider sees, the flow less its own orders y = (k - 1) D dt
        + noise (A strictly lower triangular) and V_0 (c): D = A y + c g0."""
        n = self.n; Z = slice(n, 2 * n)
        Y = (self.k - 1) * self.dt * D[:, Z] + self.Nz[:, Z]                    # lower triangular, sigma_Z sqrt(dt) on the diagonal
        A = np.linalg.solve(Y.T, D[:, Z].T).T
        A = np.tril(A, -1)
        c = D[:, 2 * n] - (self.k - 1) * self.dt * A @ D[:, 2 * n] if self.prior else np.zeros(n)
        return A, c

    def rivals(self, D):
        """(rows of each rival, Lt): the k - 1 rivals playing the map self.rmap against this insider's rows D (they see its
        orders in their flow and react), and dP/dD's factor I + (k - 1) dt B A, B = (I - (k - 2) dt A)^-1."""
        n = self.n
        if self.k == 1:
            return np.zeros_like(D), np.eye(n)
        A, c = self.rmap
        B = np.linalg.inv(np.eye(n) - (self.k - 2) * self.dt * A)
        rhs = A @ (self.dt * D + self.Nz)
        if self.prior:
            rhs[:, 2 * n] += c
        return B @ rhs, np.eye(n) + (self.k - 1) * self.dt * B @ A

    def flow(self, D):
        """The order flow's rows: this insider's D, the k - 1 rivals' (reacting through self.rmap; at the symmetric profile when
        rmap is None), the noise."""
        if self.k == 1 or self.rmap is None:
            return self.dt * self.k * D + self.Nz
        return self.dt * (D + (self.k - 1) * self.rivals(D)[0]) + self.Nz

    def projection(self, D):
        """Lambda (n x n, strictly lower): P_k = E[V_k | z_<k] for the insider rows D."""
        n = self.n
        Z = self.flow(D)
        L = np.linalg.cholesky(Z @ Z.T)
        E = np.linalg.solve(L, Z)                           # innovations, orthonormal rows: E = L^-1 Z
        C = self.V[:n] @ E.T                                # Cov(V_k, e_j)
        C = np.tril(C, -1)
        return C @ np.linalg.inv(L)                         # P = C E = C L^-1 Z

    def prices(self, D, Lam):
        return Lam @ self.flow(D)

    def form(self, D, Lam):
        """M (m x m, symmetric) with C = g' M g.  Every step's cost is discounted by w_k = e^{-rho t_k}.  cost "dw": the
        fundamental-valued flow plus the inventory's stochastic integral, sum_k w_k [dt D_k (P_k - V_k) + eps dt D_k^2 -
        Q_{k+1} (V_{k+1} - V_k)], Q_{k+1} = dt sum_{j<=k} D_j (the position after step k's order, adapted to V's step k
        increment: the Ito sum); at rho = 0 it is the "wealth" cost path by path (summation by parts)."""
        P = self.prices(D, Lam)
        ref = np.repeat(self.V[self.n][None, :], self.n, axis=0) if self.cost == "wealth" else self.V[:self.n]
        wD = self.w[:, None] * D
        A = self.dt * (wD.T @ (P - ref) + self.eps * wD.T @ D)
        if self.cost == "dw":
            Qn = self.dt * np.cumsum(D, axis=0)                 # Q_{k+1}
            dV = self.V[1:] - self.V[:-1]
            A = A - (self.w[:, None] * Qn).T @ dV
        return 0.5 * (A + A.T), P, ref

    def objective(self, M):
        th = self.theta
        if th == 0.0:
            return float(np.trace(M)), None
        A = np.eye(self.m) - 2.0 * th * M
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf, None
        return float(-logdet / (2.0 * th)), np.linalg.inv(A)

    def value_and_grad(self, D, Lam):
        M, P, ref = self.form(D, Lam)
        J, W = self.objective(M)
        if not np.isfinite(J):
            return J, np.zeros_like(D)
        if W is None:
            W = np.eye(self.m)
        dt = self.dt
        # dJ = tr(W dM), dM = dt sym(dD'(P - ref) + D' Lt dt dD + 2 eps sym(D' dD)), Lt = Lam (I + (k-1) dt B A) (rivals react)
        Lt = Lam if (self.k == 1 or self.rmap is None) else Lam @ self.rivals(D)[1]
        w = self.w[:, None]
        G = dt * (w * (P - ref) @ W + dt * Lt.T @ (w * D) @ W + 2.0 * self.eps * (w * D) @ W)
        if self.cost == "dw":
            dV = self.V[1:] - self.V[:-1]
            G = G - dt * np.cumsum(((w * dV) @ W)[::-1], axis=0)[::-1]      # - dt sum_{k >= j} w_k dV_k W
        return J, G

    def best_response(self, D, Lam, maxiter=2000):
        mask = self.mask

        def f(v):
            X = np.zeros_like(D); X[mask] = v
            J, G = self.value_and_grad(X, Lam)
            return (J, G[mask]) if np.isfinite(J) else (1e30, np.zeros_like(v))
        if self.k > 1:
            self.rmap = self.rival_map(D)                  # the rivals keep their current map
        try:
            res = minimize(f, D[mask], jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-13, "ftol": 1e-16, "maxcor": 30})
        finally:
            self.rmap = None
        X = np.zeros_like(D); X[mask] = res.x
        return X, res

    def equilibrium(self, D0=None, tol=1e-10, maxit=300, verbose=False, damping=None):
        D = np.zeros((self.n, self.m)) if D0 is None else D0.copy()
        if D0 is None:              # a start that trades on what it knows: D = (V_k - P_k) at P = 0
            D[:, :self.n] = np.tril(np.ones((self.n, self.n)), -1) * self.sV * np.sqrt(self.dt)
            if self.prior:
                D[:, 2 * self.n] = np.sqrt(self.S0)
        change = np.inf
        for it in range(maxit):
            Lam = self.projection(D)
            B, res = self.best_response(D, Lam)
            change = float(np.abs(B - D).max())
            a = damping if damping is not None else (1.0 if self.k == 1 else 0.5)
            D = B if a == 1.0 else a * B + (1 - a) * D
            if verbose:
                print(f"  iteration {it}: change {change:.2e}, J {res.fun:.8f}, |grad| {np.abs(res.jac).max():.1e}", flush=True)
            if change < tol:
                break
        Lam = self.projection(D)
        return D, Lam, change

    # ------------------------------------------------------------------ what is compared
    def record(self, D, Lam, times=(0.1, 0.3, 0.5, 0.8)):
        n, dt = self.n, self.dt
        if self.k > 1:
            self.rmap = self.rival_map(D)
        M, P, ref = self.form(D, Lam)
        J = self.objective(M)[0]
        EC = float(np.trace(M))
        Jn, G = self.value_and_grad(D, Lam)
        self.rmap = None
        ks = [int(round(t * n)) for t in times]
        out = {"n": n, "entropic": J, "expected": EC, "grad": float(np.abs(G[self.mask]).max()),
               "times": list(times)}
        # D's response to V's shock on [0.1 ...]: the order rate at t_k per unit V shock at step j0 (kernel D(t, s) per dW)
        j0 = int(round(0.05 * n))
        if self.sV > 0:
            out["D_wV"] = [float(D[k, j0] / np.sqrt(dt)) for k in ks]
        out["D_wZ"] = [float(D[k, n + j0] / np.sqrt(dt)) for k in ks]
        if self.prior:
            out["D_g0"] = [float(D[k, 2 * n] / np.sqrt(self.S0)) for k in ks]     # per unit of V_0
        # the price's kernel on the noise trades: P_k per unit gZ at step j0, / sqrt(dt)
        out["P_wZ"] = [float(P[k, n + j0] / np.sqrt(dt)) for k in ks]
        # the market maker's residual variance of V_k (information revelation)
        err = self.V[:n] - P
        out["Sigma_mm"] = [float(err[k] @ err[k]) for k in ks]
        return out


def check_gradient(n=8, theta=0.7, cost="wealth", Sigma0=0.0, seed=0, insiders=1, rho=0.0):
    g = Kyle(n=n, theta=theta, cost=cost, Sigma0=Sigma0, insiders=insiders, rho=rho)
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n, g.m)) * 0.3 * g.mask
    if insiders > 1:
        g.rmap = g.rival_map(rng.standard_normal((n, g.m)) * 0.3 * g.mask)
    Lam = np.tril(rng.standard_normal((n, n)), -1) * 0.3
    J, G = g.value_and_grad(D, Lam)
    num = np.zeros_like(D); h = 1e-6
    for idx in zip(*np.nonzero(g.mask)):
        for s in (+1, -1):
            X = D.copy(); X[idx] += s * h
            num[idx] += s * g.value_and_grad(X, Lam)[0] / (2 * h)
    return float(np.abs((G - num)[g.mask]).max() / np.abs(num[g.mask]).max())


def richardson(values, ns):
    A = np.array([[1.0 / n ** k for k in range(len(ns))] for n in ns])
    return float(np.linalg.solve(A, np.asarray(values, dtype=float))[0])


def table(insiders=1, ns=(40, 60, 80, 100, 120), thetas=(0.0, 0.5, 1.0, 2.0, 4.0), verbose=True):
    """tests/refs/leqg_kyle_prior{,2}.json: the prior game (Sigma0 = 1, a fixed value, T = 1, eps = 0.2) at every n, continued
    in theta, and the Richardson limit of every recorded number.  About twenty (one insider) or forty seconds (two)."""
    levels = {}
    for n in ns:
        D = None
        for th in thetas:
            g = Kyle(n=n, theta=th, cost="wealth", Sigma0=1.0, sigma_V=0.0, insiders=insiders)
            D, L, ch = g.equilibrium(D0=D, maxit=200)
            rec = g.record(D, L); rec["change"] = ch
            levels.setdefault(f"{th:g}", []).append(rec)
            if verbose:
                print(n, th, rec["entropic"], rec["expected"], f"grad {rec['grad']:.1e}", flush=True)
    lim = {}
    for th, lv in levels.items():
        nn = [r["n"] for r in lv]
        lim[th] = {k: ([richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))] if isinstance(lv[0][k], list)
                       else richardson([r[k] for r in lv], nn)) for k in lv[0] if k not in ("n", "grad", "times", "change")}
    return {"levels": levels, "limit": lim, "cost": "wealth", "Sigma0": 1.0, "sigma_V": 0.0, "insiders": insiders}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "table":       # python extras/kyle_reference.py table OUT.json [insiders]
        import json
        json.dump(table(int(sys.argv[3]) if len(sys.argv) > 3 else 1), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    print("gradient check:", check_gradient(), check_gradient(theta=0.0), check_gradient(cost="flow", Sigma0=1.0),
          check_gradient(Sigma0=1.0, insiders=2))
