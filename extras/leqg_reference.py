"""A brute-force discrete-time reference for risk-averse (CARA, entropic) equilibria of Chapter 1's tracking game,
independent of the package: strategies are linear maps on each player's own signal history, and each player's best
response minimises its exact entropic cost over its whole map.

  X_{k+1} = X_k + (D1_k + D2_k) dt + sigma sqrt(dt) g0_k,  X_0 = 0
  y_{i,k} = sqrt(p_i) X_k dt + sqrt(dt) g_{i,k}                    (seen from step k + 1 on)
  D_{i,k} = sum_{j<k} Gamma^i[k, j] y_{i,j}                          (Gamma^i strictly lower triangular)
  C_i = sum_k (X_k^2 + r_i D_{i,k}^2) dt = g' M_i g,  g ~ N(0, I) the 3n standardized shocks
  J_i = -(2 theta_i)^-1 log det(I - 2 theta_i M_i)   (theta_i > 0),  tr M_i = E C_i  (theta_i = 0)

Every quantity is a row vector over g, built by one forward pass; the gradient of J_i in Gamma^i comes from one
reverse pass through the same recursion (both players' strategies in the loop, only Gamma^i varied), checked by
finite differences in check_gradient().  A best response is scipy's L-BFGS over the n(n-1)/2 entries of Gamma^i; the
equilibrium is the fixed point of best responses (symmetric games: one map for both).  In an equilibrium found this
way every player's map is its best response to the others' maps, the others' maps themselves risk-averse: the
kernels, adjoints and cost forms all come from the risk-averse closed loop, not from the risk-neutral one."""
import numpy as np
from scipy.optimize import minimize


class Game:
    def __init__(self, n=100, T=1.0, p=(3.0, 3.0), r=(0.1, 0.1), sigma=1.0, theta=(0.0, 0.0)):
        self.n, self.T, self.dt = n, T, T / n
        self.p, self.r, self.sigma, self.theta = np.array(p, float), np.array(r, float), float(sigma), np.array(theta, float)
        self.m = 3 * n                                   # shock coordinates: (step, channel) with channel 0 state, 1 and 2 signals
        self.tril = np.tril_indices(n, -1)

    def e(self, k, ch):
        v = np.zeros(self.m); v[3 * k + ch] = 1.0; return v

    def forward(self, G):
        """Rows over g of X_k, y_{i,k}, D_{i,k} for k < n, given the players' maps G = (Gamma^1, Gamma^2)."""
        n, dt, m = self.n, self.dt, self.m
        X = np.zeros((n, m)); Y = np.zeros((2, n, m)); D = np.zeros((2, n, m))
        for k in range(n):
            if k:
                X[k] = X[k - 1] + dt * (D[0, k - 1] + D[1, k - 1])
                X[k, 3 * (k - 1)] += self.sigma * np.sqrt(dt)
            for i in range(2):
                D[i, k] = G[i][k, :k] @ Y[i, :k] if k else 0.0
                Y[i, k] = np.sqrt(self.p[i]) * dt * X[k]
                Y[i, k, 3 * k + 1 + i] += np.sqrt(dt)
        return X, Y, D

    def cost_form(self, X, D, i):
        return self.dt * (X.T @ X + self.r[i] * D[i].T @ D[i])

    def objective(self, M, i):
        th = self.theta[i]
        if th == 0.0:
            return float(np.trace(M)), np.eye(self.m)
        A = np.eye(self.m) - 2.0 * th * M
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf, None
        return float(-logdet / (2.0 * th)), np.linalg.inv(A)      # dJ = tr(W dM), W = (I - 2 theta M)^-1

    def value_and_grad(self, G, i):
        n, dt = self.n, self.dt
        X, Y, D = self.forward(G)
        M = self.cost_form(X, D, i)
        J, W = self.objective(M, i)
        if not np.isfinite(J):
            return J, np.zeros(len(self.tril[0]))
        Xb = 2.0 * dt * X @ W                               # dJ/dX_k (rows), and dJ/dD_{i,k} from the own-control term
        Db = np.zeros_like(D); Db[i] = 2.0 * dt * self.r[i] * D[i] @ W
        Yb = np.zeros_like(Y)
        for k in range(n - 1, -1, -1):
            if k + 1 < n:
                Db[:, k] += dt * Xb[k + 1]                  # D_k moves X_{k+1}
            for j in range(2):
                if k + 1 < n:
                    Yb[j, k] = G[j][k + 1:, k] @ Db[j, k + 1:]   # y_{j,k} feeds D_{j,m}, m > k
                Xb[k] += np.sqrt(self.p[j]) * dt * Yb[j, k]
            if k + 1 < n:
                Xb[k] += Xb[k + 1]                          # X_k carries into X_{k+1}
        grad = (Db[i] @ Y[i].T)[self.tril]                   # dJ/dGamma^i[k, j] = Db_{i,k} . y_{i,j}
        return J, grad

    def pack(self, Gi):
        return Gi[self.tril]

    def unpack(self, v):
        Gi = np.zeros((self.n, self.n)); Gi[self.tril] = v; return Gi

    def best_response(self, G, i, maxiter=500):
        def f(v):
            H = list(G); H[i] = self.unpack(v)
            J, g = self.value_and_grad(H, i)
            return (J, g) if np.isfinite(J) else (1e30, np.zeros_like(v))
        res = minimize(f, self.pack(G[i]), jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-12, "ftol": 1e-15})
        return self.unpack(res.x), res

    def symmetric_equilibrium(self, G0=None, tol=1e-9, maxit=200, verbose=False):
        """The fixed point Gamma = BR(Gamma, Gamma) of a symmetric game, by damped best-response iteration."""
        G = np.zeros((self.n, self.n)) if G0 is None else G0.copy()
        for it in range(maxit):
            B, res = self.best_response([G, G], 0)
            change = float(np.abs(B - G).max())
            G = 0.5 * G + 0.5 * B if it < 3 else B
            if verbose:
                print(f"  iteration {it}: change {change:.2e}, J {res.fun:.6f}", flush=True)
            if change < tol:
                break
        return G, change

    def expected_cost(self, G, i):
        X, Y, D = self.forward(G)
        return float(np.trace(self.cost_form(X, D, i)))

    def entropic_cost(self, G, i):
        X, Y, D = self.forward(G)
        return self.objective(self.cost_form(X, D, i), i)[0]


def check_gradient(n=12, theta=0.5, seed=0):
    """The reverse pass against central differences, at a random pair of maps."""
    g = Game(n=n, theta=(theta, theta))
    rng = np.random.default_rng(seed)
    G = [np.tril(rng.standard_normal((n, n)), -1) * 0.3 for _ in range(2)]
    J, grad = g.value_and_grad(G, 0)
    num = np.zeros_like(grad); h = 1e-6
    for q in range(len(grad)):
        for s in (+1, -1):
            H = [G[0].copy(), G[1]]; v = g.pack(H[0]); v[q] += s * h; H[0] = g.unpack(v)
            num[q] += s * g.value_and_grad(H, 0)[0] / (2 * h)
    return float(np.abs(grad - num).max() / max(1e-12, np.abs(num).max()))


def equilibrium(g, G0=None, tol=1e-10, maxit=500):
    """The equilibrium maps of g: the symmetric fixed point when the game is symmetric (one map for both), else the
    alternating best responses (Gauss-Seidel) from G0 (zeros).  Returns ([Gamma^1, Gamma^2], the last change)."""
    if g.p[0] == g.p[1] and g.r[0] == g.r[1] and g.theta[0] == g.theta[1]:
        G, change = g.symmetric_equilibrium(G0=None if G0 is None else G0[0], tol=tol)
        return [G, G], change
    Gs = [np.zeros((g.n, g.n)), np.zeros((g.n, g.n))] if G0 is None else [G0[0].copy(), G0[1].copy()]
    for _ in range(maxit):
        B0, _ = g.best_response(Gs, 0)
        B1, _ = g.best_response([B0, Gs[1]], 1)
        change = max(float(np.abs(B0 - Gs[0]).max()), float(np.abs(B1 - Gs[1]).max()))
        Gs = [B0, B1]
        if change < tol:
            break
    return Gs, change


def record(g, Gs, s=0.2, times=(0.3, 0.5, 0.8)):
    """What the engine is compared on: each player's entropic and expected cost, the largest gradient of the entropic cost
    in each player's own map (zero at an equilibrium), and D1's and D2's response at `times` to a unit state shock at
    step round(s n) (the interval [s, s + dt]), D_k being the control on [t_k, t_k + dt)."""
    X, Y, D = g.forward(Gs)
    k0 = int(round(s * g.n)); unit = g.sigma * np.sqrt(g.dt)
    ks = [int(round(t * g.n)) for t in times]
    return {"n": g.n, "entropic": [g.entropic_cost(Gs, i) for i in range(2)], "expected": [g.expected_cost(Gs, i) for i in range(2)],
            "grad": [float(np.abs(g.value_and_grad(Gs, i)[1]).max()) for i in range(2)],
            "D1": [float(D[0][k, 3 * k0] / unit) for k in ks], "D2": [float(D[1][k, 3 * k0] / unit) for k in ks]}


def richardson(values, ns):
    """The limit n -> infinity of values(n) = v + c1/n + ... + c_{m-1}/n^{m-1} through the m points (ns, values)."""
    A = np.array([[1.0 / n ** k for k in range(len(ns))] for n in ns])
    return float(np.linalg.solve(A, np.asarray(values, dtype=float))[0])


CASES = {                                   # name -> the game's parameters; the symmetric ones are reached by continuation in theta
    "theta 0": dict(theta=(0.0, 0.0)), "theta 0.5": dict(theta=(0.5, 0.5)), "theta 1": dict(theta=(1.0, 1.0)),
    "theta 1.5": dict(theta=(1.5, 1.5)), "theta 2": dict(theta=(2.0, 2.0)), "theta 2.5": dict(theta=(2.5, 2.5)),
    "mixed": dict(theta=(1.0, 0.0)),                                          # player 1 risk averse, player 2 neutral
    "asymmetric": dict(p=(3.0, 1.0), r=(0.1, 0.2), theta=(1.0, 0.5)),       # nothing alike
}


def table(ns=(50, 100, 150, 200, 300, 400), verbose=True):
    """Every case of CASES at every n (the symmetric thetas by continuation, 0.25 apart, so that no start lies past the
    breakdown) and the Richardson limit of each number: {case: {"params", "levels": [record per n], "limit": {...}}}.
    A level past the discrete game's own breakdown (theta = 2.5 at n = 50) is left out.  The levels at 150 and 300
    matter near the breakdown, where the series in 1/n is far from its asymptote: at theta = 2.5 the limit of n = 200,
    300, 400 alone is 4.4e-6 off that of all five.  About twenty minutes."""
    out = {}
    for n in ns:
        G = None
        for th in np.round(np.arange(0.0, 2.51, 0.25), 2):
            g = Game(n=n, theta=(th, th))
            Gs, change = equilibrium(g, None if G is None else [G, G])
            G = Gs[0]
            name = f"theta {th:g}"
            if name in CASES:
                rec = record(g, Gs)
                if np.all(np.isfinite(rec["entropic"])):
                    out.setdefault(name, {"params": CASES[name], "levels": []})["levels"].append(rec)
                if verbose:
                    print(n, name, rec["entropic"], rec["expected"], f"grad {max(rec['grad']):.1e}", flush=True)
        for name in ("mixed", "asymmetric"):
            g = Game(n=n, **CASES[name])
            Gs, change = equilibrium(g)
            rec = record(g, Gs)
            out.setdefault(name, {"params": CASES[name], "levels": []})["levels"].append(rec)
            if verbose:
                print(n, name, rec["entropic"], rec["expected"], f"grad {max(rec['grad']):.1e}", flush=True)
    for name, case in out.items():
        lv = case["levels"]; nn = [r["n"] for r in lv]
        case["limit"] = {k: [richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))]
                         for k in ("entropic", "expected", "D1", "D2")}
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "table":
        import json
        json.dump(table(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    print("gradient check (relative error):", f"{check_gradient():.1e}", f"{check_gradient(theta=0.0):.1e}")
