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
    def __init__(self, n=100, T=1.0, p=(3.0, 3.0), r=(0.1, 0.1), sigma=1.0, theta=(0.0, 0.0), xdw=(0.0, 0.0)):
        self.n, self.T, self.dt = n, T, T / n
        self.xdw = np.array(xdw, float)                  # player i's cost gains xdw_i sum_k X_k sigma sqrt(dt) g0_k (Ito: int xdw X sigma dW0)
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
        M = self.dt * (X.T @ X + self.r[i] * D[i].T @ D[i])
        if self.xdw[i]:
            S = np.zeros((self.n, self.m)); S[np.arange(self.n), 3 * np.arange(self.n)] = self.sigma * np.sqrt(self.dt)
            A = self.xdw[i] * X.T @ S
            M = M + 0.5 * (A + A.T)
        return M

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
        if self.xdw[i]:
            Xb += self.xdw[i] * self.sigma * np.sqrt(dt) * W[3 * np.arange(n)]      # the integral: d tr(W sym(dX' S)) = dX_k . W s_k
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


def check_gradient(n=12, theta=0.5, seed=0, xdw=(0.0, 0.0)):
    """The reverse pass against central differences, at a random pair of maps."""
    g = Game(n=n, theta=(theta, theta), xdw=xdw)
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


# ================================================================================================ means (the linear part)
class MeanGame(Game):
    """Game with means: an initial state X_0 = x0, a constant drift a (X_{k+1} = X_k + (D1 + D2 + a) dt + ...), and a
    target xstar in each player's loss, X^2 - 2 xstar X + r D^2 (the target's constant xstar^2 left out, as the model's
    loss leaves it out).  Each strategy is affine: D_{i,k} = sum_{j<k} Gamma^i[k, j] y_{i,j} + d_{i,k}, the open-loop
    part d the mean control.  Every quantity is a row over (g, 1), the last coordinate the constant, and the cost is
    C = c0 + l'g + g'Mg with F = [[M, l/2], [l'/2, c0]]; then

        J = c0 - (2 theta)^-1 log det(I - 2 theta M) + (theta / 2) l'(I - 2 theta M)^-1 l,

    dJ = tr(G_F dF) with G_F = [[W + 4 theta^2 (W f)(W f)', 2 theta W f], [., 1]], f = l / 2, W = (I - 2 theta M)^-1
    (G_F = I at theta = 0: J = E C = tr M + c0).  The gradient in d_{i,k} is the constant coordinate of the adjoint of
    D_{i,k}, the one in Gamma^i as in Game."""

    def __init__(self, n=100, T=1.0, p=(3.0, 3.0), r=(0.1, 0.1), sigma=1.0, theta=(0.0, 0.0), x0=0.0, drift=0.0, xstar=0.0):
        super().__init__(n=n, T=T, p=p, r=r, sigma=sigma, theta=theta)
        self.x0, self.drift, self.xstar = float(x0), float(drift), float(xstar)
        self.M1 = self.m + 1                              # the rows' length: the shocks, then the constant

    def forward(self, G):
        """G = [(Gamma^1, d^1), (Gamma^2, d^2)]: rows over (g, 1) of X_k, y_{i,k}, D_{i,k}."""
        n, dt, m = self.n, self.dt, self.m
        X = np.zeros((n, m + 1)); Y = np.zeros((2, n, m + 1)); D = np.zeros((2, n, m + 1))
        X[0, m] = self.x0
        for k in range(n):
            if k:
                X[k] = X[k - 1] + dt * (D[0, k - 1] + D[1, k - 1])
                X[k, 3 * (k - 1)] += self.sigma * np.sqrt(dt)
                X[k, m] += self.drift * dt
            for i in range(2):
                Gm, d = G[i]
                D[i, k] = Gm[k, :k] @ Y[i, :k] if k else 0.0
                D[i, k, m] += d[k]
                Y[i, k] = np.sqrt(self.p[i]) * dt * X[k]
                Y[i, k, 3 * k + 1 + i] += np.sqrt(dt)
        return X, Y, D

    def cost_form(self, X, D, i):
        F = self.dt * (X.T @ X + self.r[i] * D[i].T @ D[i])
        F[self.m, :] -= self.dt * self.xstar * X.sum(axis=0)
        F[:, self.m] -= self.dt * self.xstar * X.sum(axis=0)
        return F                                          # C = (g, 1)' F (g, 1): -2 xstar X split over the two corners

    def objective(self, F, i):
        m, th = self.m, self.theta[i]
        M, f, c0 = F[:m, :m], F[:m, m], F[m, m]
        if th == 0.0:
            return float(np.trace(M) + c0), np.eye(m + 1)
        A = np.eye(m) - 2.0 * th * M
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf, None
        W = np.linalg.inv(A); Wf = W @ f
        J = float(c0 - logdet / (2.0 * th) + 2.0 * th * f @ Wf)
        GF = np.zeros((m + 1, m + 1))
        GF[:m, :m] = W + 4.0 * th ** 2 * np.outer(Wf, Wf)
        GF[:m, m] = GF[m, :m] = 2.0 * th * Wf
        GF[m, m] = 1.0
        return J, GF

    def value_and_grad(self, G, i):
        n, dt, m = self.n, self.dt, self.m
        X, Y, D = self.forward(G)
        F = self.cost_form(X, D, i)
        J, W = self.objective(F, i)
        if not np.isfinite(J):
            return J, np.zeros(len(self.tril[0]) + n)
        # dJ = tr(W dF), dF = dt sym(2 X' dX + 2 r D' dD) - dt xstar (e_m 1'dX + dX'1 e_m')
        Xb = 2.0 * dt * X @ W
        Xb -= 2.0 * dt * self.xstar * W[m][None, :]           # the target's corners: -2 dt xstar W[m, :] . sum_k dX_k
        Db = np.zeros_like(D); Db[i] = 2.0 * dt * self.r[i] * D[i] @ W
        Yb = np.zeros_like(Y)
        for k in range(n - 1, -1, -1):
            if k + 1 < n:
                Db[:, k] += dt * Xb[k + 1]
            for j in range(2):
                if k + 1 < n:
                    Yb[j, k] = G[j][0][k + 1:, k] @ Db[j, k + 1:]
                Xb[k] += np.sqrt(self.p[j]) * dt * Yb[j, k]
            if k + 1 < n:
                Xb[k] += Xb[k + 1]
        gG = (Db[i] @ Y[i].T)[self.tril]
        gd = Db[i][:, m]
        return J, np.concatenate([gG, gd])

    def pack(self, Gi):
        return np.concatenate([Gi[0][self.tril], Gi[1]])

    def unpack(self, v):
        Gm = np.zeros((self.n, self.n)); k = len(self.tril[0]); Gm[self.tril] = v[:k]
        return (Gm, v[k:].copy())

    def symmetric_equilibrium(self, G0=None, tol=1e-9, maxit=200, verbose=False):
        G = (np.zeros((self.n, self.n)), np.zeros(self.n)) if G0 is None else G0
        for it in range(maxit):
            B, res = self.best_response([G, G], 0)
            change = max(float(np.abs(B[0] - G[0]).max()), float(np.abs(B[1] - G[1]).max()))
            G = (0.5 * G[0] + 0.5 * B[0], 0.5 * G[1] + 0.5 * B[1]) if it < 3 else B
            if verbose:
                print(f"  iteration {it}: change {change:.2e}, J {res.fun:.8f}", flush=True)
            if change < tol:
                break
        return G, change

    def expected_cost(self, G, i):
        X, Y, D = self.forward(G)
        F = self.cost_form(X, D, i)
        return float(np.trace(F[:self.m, :self.m]) + F[self.m, self.m])

    def entropic_cost(self, G, i):
        X, Y, D = self.forward(G)
        return self.objective(self.cost_form(X, D, i), i)[0]


def check_gradient_means(n=10, theta=0.5, seed=1, **kw):
    g = MeanGame(n=n, theta=(theta, theta), x0=0.7, drift=0.4, xstar=0.3, **kw)
    rng = np.random.default_rng(seed)
    G = [(np.tril(rng.standard_normal((n, n)), -1) * 0.3, rng.standard_normal(n) * 0.3) for _ in range(2)]
    J, grad = g.value_and_grad(G, 0)
    num = np.zeros_like(grad); h = 1e-6
    for q in range(len(grad)):
        for s in (+1, -1):
            v = g.pack(G[0]); v[q] += s * h
            num[q] += s * g.value_and_grad([g.unpack(v), G[1]], 0)[0] / (2 * h)
    return float(np.abs(grad - num).max() / max(1e-12, np.abs(num).max()))


def record_means(g, G, s=0.2, times=(0.1, 0.3, 0.5, 0.8)):
    """The entropic and expected cost of player 1, D1's response to a state shock at s (as record), and the mean paths of D1
    and X at `times` (D1's mean on [t_k, t_k + dt), X's at t_k)."""
    X, Y, D = g.forward([G, G])
    m = g.m; k0 = int(round(s * g.n)); unit = g.sigma * np.sqrt(g.dt)
    ks = [int(round(t * g.n)) for t in times]
    return {"n": g.n, "entropic": g.entropic_cost([G, G], 0), "expected": g.expected_cost([G, G], 0),
            "grad": float(np.abs(g.value_and_grad([G, G], 0)[1]).max()),
            "D1": [float(D[0][k, 3 * k0] / unit) for k in ks], "D1_mean": [float(D[0][k, m]) for k in ks],
            "X_mean": [float(X[k, m]) for k in ks], "times": list(times)}


def table_xdw(ns_=(50, 100, 150, 200, 300), thetas=(0.0, 0.5, 1.0), xdw=0.5, verbose=True):
    """tests/refs/leqg_ch1_xdw.json: the symmetric game with the integral xdw int X sigma dW0 in both players' costs, at every n
    and theta (continued in theta), and the Richardson limit of the entropic and expected costs and D1's response."""
    out = {}
    for n in ns_:
        G = None
        for th in thetas:
            g = Game(n=n, theta=(th, th), xdw=(xdw, xdw))
            G, change = g.symmetric_equilibrium(G0=G, tol=1e-10)
            rec = record(g, [G, G]); rec["change"] = change
            out.setdefault(f"{th:g}", {"levels": []})["levels"].append(rec)
            if verbose:
                print(n, th, rec["entropic"], rec["expected"], rec["D1"], f"grad {max(rec['grad']):.1e}", flush=True)
    for th, case in out.items():
        lv = case["levels"]; nn = [r["n"] for r in lv]
        case["limit"] = {k: [richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))] for k in ("entropic", "expected", "D1", "D2")}
    return {"xdw": xdw, "cases": out}


MEAN_CASES = {"x0 drift target": dict(x0=1.0, drift=0.5, xstar=0.3)}


def table_means(ns_=(50, 100, 150, 200, 300), thetas=(0.0, 0.5, 1.0, 2.0), case="x0 drift target", verbose=True):
    """tests/refs/leqg_ch1_means.json: the symmetric game of MEAN_CASES[case] at every n and theta (continued in theta) and the
    Richardson limit of every recorded number."""
    kw = MEAN_CASES[case]
    levels = {}
    for n in ns_:
        G = None
        for th in thetas:
            g = MeanGame(n=n, theta=(th, th), **kw)
            G, change = g.symmetric_equilibrium(G0=G, tol=1e-10)
            rec = record_means(g, G); rec["change"] = change
            levels.setdefault(f"{th:g}", []).append(rec)
            if verbose:
                print(n, th, rec["entropic"], rec["expected"], rec["D1_mean"], f"grad {rec['grad']:.1e}", flush=True)
    lim = {}
    for th, lv in levels.items():
        nn = [r["n"] for r in lv]
        lim[th] = {k: ([richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))] if isinstance(lv[0][k], list)
                       else richardson([r[k] for r in lv], nn)) for k in ("entropic", "expected", "D1", "D1_mean", "X_mean")}
    return {"case": case, "params": kw, "levels": levels, "limit": lim}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "xdw":          # python extras/leqg_reference.py xdw OUT.json
        import json
        json.dump(table_xdw(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "means":        # python extras/leqg_reference.py means OUT.json
        import json
        json.dump(table_means(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "table":
        import json
        json.dump(table(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    print("gradient check (relative error):", f"{check_gradient():.1e}", f"{check_gradient(theta=0.0):.1e}")
