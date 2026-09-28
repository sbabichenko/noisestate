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
    def __init__(self, n=100, T=1.0, p=(3.0, 3.0), r=(0.1, 0.1), sigma=1.0, theta=(0.0, 0.0), xdw=(0.0, 0.0), udw=(0.0, 0.0)):
        self.n, self.T, self.dt = n, T, T / n
        self.xdw = np.array(xdw, float)                  # player i's cost gains xdw_i sum_k X_k sigma sqrt(dt) g0_k (Ito: int xdw X sigma dW0)
        self.udw = np.array(udw, float)                  # ... and udw_i sum_k D_{i,k} sigma sqrt(dt) g0_k (its own control: int udw D_i sigma dW0)
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
        if self.xdw[i] or self.udw[i]:
            S = np.zeros((self.n, self.m)); S[np.arange(self.n), 3 * np.arange(self.n)] = self.sigma * np.sqrt(self.dt)
            A = self.xdw[i] * X.T @ S + self.udw[i] * D[i].T @ S
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
        if self.udw[i]:
            Db[i] += self.udw[i] * self.sigma * np.sqrt(dt) * W[3 * np.arange(n)]
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


def check_gradient(n=12, theta=0.5, seed=0, xdw=(0.0, 0.0), udw=(0.0, 0.0)):
    """The reverse pass against central differences, at a random pair of maps."""
    g = Game(n=n, theta=(theta, theta), xdw=xdw, udw=udw)
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


def table_udw(ns_=(50, 100, 150, 200, 300), thetas=(0.0, 0.5, 1.0), udw=0.5, verbose=True):
    """tests/refs/leqg_ch1_udw.json: the symmetric game with udw int D_i sigma dW0 (a player's OWN control against the shock of
    its own instant, as a market maker's P int (P - V) sigma_Z dW_Z) in both players' costs, at every n and theta (continued in
    theta), and the Richardson limit of the entropic and expected costs and the responses."""
    out = {}
    for n in ns_:
        G = None
        for th in thetas:
            g = Game(n=n, theta=(th, th), udw=(udw, udw))
            G, change = g.symmetric_equilibrium(G0=G, tol=1e-10)
            rec = record(g, [G, G]); rec["change"] = change
            out.setdefault(f"{th:g}", {"levels": []})["levels"].append(rec)
            if verbose:
                print(n, th, rec["entropic"], rec["expected"], rec["D1"], f"grad {max(rec['grad']):.1e}", flush=True)
    for th, case in out.items():
        lv = case["levels"]; nn = [r["n"] for r in lv]
        case["limit"] = {k: [richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))] for k in ("entropic", "expected", "D1", "D2")}
    return {"udw": udw, "cases": out}


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



# ================================================================================================ monitored deviations
class MonitorGame:
    """Chapter 6's privy response under the entropic objective, by brute force: player 1 regulates its own state from a
    signal, player 2 tracks player 1's state with its own from a signal of the gap, and player 2 is privy to player 1's
    deviations (player 1 is unaffected by player 2, so its map is a one-agent problem and no response of player 2 enters
    player 1's own best response):

      X1_{k+1} = X1_k + D1_k dt + sqrt(dt) g0_k,    y1_k = sqrt(p1) X1_k dt + sqrt(dt) g1_k
      X2_{k+1} = X2_k + D2_k dt + s2 sqrt(dt) g2_k, y2_k = sqrt(p2) (X2_k - X1_k) dt + sqrt(dt) g3_k
      C1 = sum_k (X1_k^2 + r1 D1_k^2) dt,  C2 = sum_k ((X2_k - X1_k)^2 + r2 D2_k^2) dt,  J_i = theta_i^-1 log E e^{theta_i C_i}

    The on-path maps (Gamma^i strictly lower triangular on each player's own signal history) are the entropic best
    responses (L-BFGS on the exact log-det, gradient by a reverse pass, checked by finite differences).  A seed is a
    known unit displacement of X1 by player 1 at step s0 (D1_{s0} += 1 / dt); both privy players (player 1 to its own
    seed: its blip continuation; player 2) answer it through response columns d_i[k], k > s0, while their maps go on
    reading the on-path part of their signals (a privy filter does not move).  With the seed a known constant, C_i =
    c0 + l'g + g'M_i g, M_i the on-path form, and each player's precommitment objective over its column is
    J_i = c0 - (2 theta)^-1 log det(I - 2 theta M_i) + (theta / 2) l'(I - 2 theta M_i)^-1 l, quadratic in the columns:
    the two first-order conditions (a Nash in the columns) are one linear solve (seed_response).  Nothing in it uses the
    engine's derivation: no conditional law, no tilt formula, only the exact objective of the seed world."""

    def __init__(self, n=60, T=1.0, p=(3.0, 3.0), r=(0.1, 0.1), s2=1.0, theta=(0.0, 0.0), udw=(0.0, 0.0)):
        self.n, self.T, self.dt = n, T, T / n
        self.p, self.r, self.s2, self.theta = np.array(p, float), np.array(r, float), float(s2), np.array(theta, float)
        self.udw = np.array(udw, float)     # player i's cost gains udw_i sum_k D_{i,k} (its own state's shock increment at k):
        self.m = 4 * n                      # udw_1 int D1 dW0 and udw_2 int D2 s2 dW2, its own control against the shock of its instant
        self.S = []
        for i, (ch, sc) in enumerate(((0, 1.0), (2, self.s2))):
            S = np.zeros((n, self.m)); S[np.arange(n), 4 * np.arange(n) + ch] = sc * np.sqrt(self.dt)
            self.S.append(S)
        self.tril = np.tril_indices(n, -1)

    def forward(self, G):
        n, dt, m = self.n, self.dt, self.m
        sd = np.sqrt(dt)
        X = np.zeros((2, n, m)); Y = np.zeros((2, n, m)); D = np.zeros((2, n, m))
        for k in range(n):
            if k:
                X[:, k] = X[:, k - 1] + dt * D[:, k - 1]
                X[0, k, 4 * (k - 1)] += sd
                X[1, k, 4 * (k - 1) + 2] += self.s2 * sd
            for i in range(2):
                D[i, k] = G[i][k, :k] @ Y[i, :k] if k else 0.0
            Y[0, k] = np.sqrt(self.p[0]) * dt * X[0, k]; Y[0, k, 4 * k + 1] += sd
            Y[1, k] = np.sqrt(self.p[1]) * dt * (X[1, k] - X[0, k]); Y[1, k, 4 * k + 3] += sd
        return X, Y, D

    def _atoms(self, X, D, i):
        """The rows whose squares make C_i: [(rows, weight)]."""
        return [(X[0], 1.0), (D[0], self.r[0])] if i == 0 else [(X[1] - X[0], 1.0), (D[1], self.r[1])]

    def cost_form(self, X, D, i):
        M = self.dt * sum(w * a.T @ a for a, w in self._atoms(X, D, i))
        if self.udw[i]:
            A = self.udw[i] * D[i][:, :self.m].T @ self.S[i]
            M = M + 0.5 * (A + A.T)
        return M

    def W(self, M, i):
        th = self.theta[i]
        return np.eye(self.m) if th == 0.0 else np.linalg.inv(np.eye(self.m) - 2.0 * th * M)

    def objective(self, M, i):
        th = self.theta[i]
        if th == 0.0:
            return float(np.trace(M))
        sign, logdet = np.linalg.slogdet(np.eye(self.m) - 2.0 * th * M)
        return float(-logdet / (2.0 * th)) if sign > 0 else np.inf

    def value_and_grad(self, G, i):
        n, dt = self.n, self.dt
        X, Y, D = self.forward(G)
        M = self.cost_form(X, D, i)
        J = self.objective(M, i)
        if not np.isfinite(J):
            return J, np.zeros(len(self.tril[0]))
        W = self.W(M, i)
        Xb = np.zeros_like(X); Db = np.zeros_like(D)
        if i == 0:
            Xb[0] = 2.0 * dt * X[0] @ W; Db[0] = 2.0 * dt * self.r[0] * D[0] @ W
        else:
            E = 2.0 * dt * (X[1] - X[0]) @ W
            Xb[1] += E; Xb[0] -= E; Db[1] = 2.0 * dt * self.r[1] * D[1] @ W
        if self.udw[i]:
            Db[i] += self.udw[i] * self.S[i] @ W
        Yb = np.zeros_like(Y)
        c1, c2 = np.sqrt(self.p[0]) * dt, np.sqrt(self.p[1]) * dt
        for k in range(n - 1, -1, -1):
            if k + 1 < n:
                Db[:, k] += dt * Xb[:, k + 1]
                for j in range(2):
                    Yb[j, k] = G[j][k + 1:, k] @ Db[j, k + 1:]
            Xb[0, k] += c1 * Yb[0, k] - c2 * Yb[1, k]
            Xb[1, k] += c2 * Yb[1, k]
            if k + 1 < n:
                Xb[:, k] += Xb[:, k + 1]
        return J, (Db[i] @ Y[i].T)[self.tril]

    def pack(self, Gi):
        return Gi[self.tril]

    def unpack(self, v):
        Gi = np.zeros((self.n, self.n)); Gi[self.tril] = v; return Gi

    def best_response(self, G, i, maxiter=3000):
        def f(v):
            H = list(G); H[i] = self.unpack(v)
            J, g = self.value_and_grad(H, i)
            return (J, g) if np.isfinite(J) else (1e30, np.zeros_like(v))
        res = minimize(f, self.pack(G[i]), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxcor": 30, "gtol": 1e-13, "ftol": 1e-16})
        return self.unpack(res.x), res

    def equilibrium(self, G0=None, steps=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Player 1's best response (it faces no one), then player 2's to it, each by continuation in its theta (the zero
        map's entropic cost is infinite past a small theta).  Returns the maps and the largest gradient left."""
        G = [np.zeros((self.n, self.n)), np.zeros((self.n, self.n))] if G0 is None else [G0[0].copy(), G0[1].copy()]
        theta = self.theta.copy()
        try:
            for i in range(2):
                for f in (steps if theta[i] > 0 else (0.0,)):
                    self.theta[i] = f * theta[i]
                    G[i], _ = self.best_response(G, i)
        finally:
            self.theta = theta
        return G, max(float(np.abs(self.value_and_grad(G, i)[1]).max()) for i in range(2))

    def seed_world(self, s0):
        """The seed world's constant columns as linear maps of p = (xi, d1[s0 + 1:], d2[s0 + 1:]): rows (n, len(p)) of X1, X2,
        D1, D2 (the privy maps read only the on-path part of their signals, so the seed moves nothing else)."""
        n, dt = self.n, self.dt
        nd = n - s0 - 1
        npar = 1 + 2 * nd
        X = np.zeros((2, n, npar)); D = np.zeros((2, n, npar))
        D[0, s0, 0] = 1.0 / dt
        for k in range(s0 + 1, n):
            D[0, k, 1 + (k - s0 - 1)] = 1.0
            D[1, k, 1 + nd + (k - s0 - 1)] = 1.0
        for k in range(1, n):
            X[:, k] = X[:, k - 1] + dt * D[:, k - 1]
        return X, D

    def seed_objective_form(self, G, s0, i, onpath=None):
        """J_i of the seed world = p' H p + (terms free of p): H = A + (theta / 2) B' W B, A the constant's square, B the
        cross term's coefficient (l = B p)."""
        X, Y, D = self.forward(G) if onpath is None else onpath
        Xc, Dc = self.seed_world(s0)
        M = self.cost_form(X, D, i)
        dt = self.dt
        g_atoms = self._atoms(X, D, i); c_atoms = self._atoms(Xc, Dc, i)
        A = dt * sum(w * ac.T @ ac for (ac, w) in c_atoms)
        B = 2.0 * dt * sum(w * ag.T @ ac for (ag, w), (ac, _) in zip(g_atoms, c_atoms))
        if self.udw[i]:
            B = B + self.udw[i] * self.S[i].T @ Dc[i]           # the seed's part of the integral's quantity against the shocks
        th = self.theta[i]
        H = A if th == 0.0 else A + 0.5 * th * B.T @ self.W(M, i) @ B
        return 0.5 * (H + H.T)

    def seed_response(self, G, s0):
        """The privy players' response columns (d1, d2) to a unit seed at s0, and the seed world's rows."""
        n = self.n; nd = n - s0 - 1
        on = self.forward(G)
        H = [self.seed_objective_form(G, s0, i, on) for i in range(2)]
        b = [np.arange(1, 1 + nd), np.arange(1 + nd, 1 + 2 * nd)]
        A = np.zeros((2 * nd, 2 * nd)); rhs = np.zeros(2 * nd)
        for i in range(2):
            A[i * nd:(i + 1) * nd, :] = H[i][b[i]][:, 1:]
            rhs[i * nd:(i + 1) * nd] = -H[i][b[i], 0]
        d = np.linalg.solve(A, rhs)
        p = np.concatenate([[1.0], d])
        Xc, Dc = self.seed_world(s0)
        return {"d1": d[:nd], "d2": d[nd:], "X1": Xc[0] @ p, "X2": Xc[1] @ p, "D1": Dc[0] @ p, "D2": Dc[1] @ p, "H": H, "p": p}


def check_gradient_monitor(n=10, theta=0.5, seed=2, udw=(0.0, 0.0)):
    """MonitorGame's reverse pass against central differences, both players, at random maps."""
    g = MonitorGame(n=n, theta=(theta, theta), udw=udw)
    rng = np.random.default_rng(seed)
    G = [np.tril(rng.standard_normal((n, n)), -1) * 0.3 for _ in range(2)]
    worst = 0.0
    for i in range(2):
        J, grad = g.value_and_grad(G, i)
        num = np.zeros_like(grad); h = 1e-6
        for q in range(len(grad)):
            for s in (+1, -1):
                H = list(G); v = g.pack(G[i]); v[q] += s * h; H[i] = g.unpack(v)
                num[q] += s * g.value_and_grad(H, i)[0] / (2 * h)
        worst = max(worst, float(np.abs(grad - num).max() / max(1e-12, np.abs(num).max())))
    return worst


def check_seed_objective(n=12, theta=0.7, s0=4, seed=3, udw=(0.0, 0.0)):
    """The seed world's exact J_i (log det plus the linear part, from the full quadratic form over (g, 1)) against the quadratic
    form seed_objective_form, at a random p: the difference must be a constant in p (checked on two points)."""
    g = MonitorGame(n=n, theta=(theta, theta), udw=udw)
    rng = np.random.default_rng(seed)
    G = [np.tril(rng.standard_normal((n, n)), -1) * 0.2 for _ in range(2)]
    X, Y, D = g.forward(G)
    Xc, Dc = g.seed_world(s0)
    out = 0.0
    for i in range(2):
        H = g.seed_objective_form(G, s0, i)
        vals = []
        for _ in range(2):
            p = rng.standard_normal(Xc.shape[2])
            Xa = np.concatenate([X, (Xc @ p)[:, :, None]], axis=2); Da = np.concatenate([D, (Dc @ p)[:, :, None]], axis=2)
            F = g.dt * sum(w * a.T @ a for a, w in g._atoms(Xa, Da, i))
            if g.udw[i]:                                          # the integral: udw D_i' S_i, D_i with its constant column
                Sx = np.concatenate([g.S[i], np.zeros((g.n, 1))], axis=1)
                A = g.udw[i] * Da[i].T @ Sx
                F = F + 0.5 * (A + A.T)
            M, l, c0 = F[:-1, :-1], 2.0 * F[:-1, -1], F[-1, -1]
            A = np.eye(g.m) - 2.0 * theta * M
            J = c0 - np.linalg.slogdet(A)[1] / (2.0 * theta) + 0.5 * theta * l @ np.linalg.solve(A, l)
            vals.append(J - p @ H @ p)
        out = max(out, abs(vals[0] - vals[1]))
    return out


def record_monitor(g, G, s=0.2, times=(0.3, 0.5, 0.8)):
    """The response at `times` to a unit seed of player 1 at s: X1, D1 (its blip continuation), X2, D2 (the privy
    player's), the rows at step round(t n); and both players' entropic costs."""
    s0 = int(round(s * g.n))
    out = g.seed_response(G, s0)
    ks = [int(round(t * g.n)) for t in times]
    rec = {"n": g.n, "s": s, "times": list(times)}
    for k in ("X1", "D1", "X2", "D2"):
        rec[k] = [float(out[k][kk]) for kk in ks]
    X, Y, D = g.forward(G)
    rec["entropic"] = [g.objective(g.cost_form(X, D, i), i) for i in range(2)]
    return rec


MONITOR_CASES = {"neutral": dict(theta=(0.0, 0.0)), "both 1": dict(theta=(1.0, 1.0)), "responder 1": dict(theta=(0.0, 1.0)),
                 "deviator 1.5": dict(theta=(1.5, 0.0)),
                 "both 1, own integrals": dict(theta=(1.0, 1.0), udw=(0.5, 0.5))}


def table_monitor(ns_=(40, 60, 80, 100, 120), verbose=True):
    """tests/refs/leqg_monitor.json: every case of MONITOR_CASES at every n, and the Richardson limit of every number."""
    out = {}
    for name, kw in MONITOR_CASES.items():
        lv = []
        for n in ns_:
            g = MonitorGame(n=n, **kw)
            G, grad = g.equilibrium()
            rec = record_monitor(g, G); rec["grad"] = grad
            lv.append(rec)
            if verbose:
                print(name, n, rec["entropic"], rec["D2"], rec["D1"], f"grad {grad:.1e}", flush=True)
        nn = [r["n"] for r in lv]
        lim = {k: [richardson([r[k][i] for r in lv], nn) for i in range(len(lv[0][k]))] for k in ("X1", "D1", "X2", "D2", "entropic")}
        out[name] = {"params": {k: list(v) for k, v in kw.items()}, "levels": lv, "limit": lim}
    return out

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
    if len(sys.argv) > 2 and sys.argv[1] == "udw":          # python extras/leqg_reference.py udw OUT.json
        import json
        json.dump(table_udw(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "monitor":      # python extras/leqg_reference.py monitor OUT.json
        import json
        json.dump(table_monitor(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "table":
        import json
        json.dump(table(), open(sys.argv[2], "w"), indent=1)
        sys.exit(0)
    print("gradient check (relative error):", f"{check_gradient():.1e}", f"{check_gradient(theta=0.0):.1e}")
