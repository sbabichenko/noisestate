"""An independent brute-force discrete-time reference for the fuzz campaign (extras/fuzz): it reads a model in the
grammar's dictionary form (numeric coefficients) and solves it with no code from the package.

Time is cut into steps of h.  A state moves by the Euler step X_{k+1} = X_k + h drift_k + sqrt(h) sigma . g_k, a signal
row's increment over step k is y_k = h drift_k + sqrt(h) noise . g_k (the same g_k as the states' step), a lagged atom
reads the quantity lag / h steps earlier (zero before time 0), a row observed with delay d is read by a control at step
k up to the increment k - 1 - d / h.  Every quantity is a row vector over the columns: the standardised shocks g (one
per shock and step on a finite horizon; one per shock at age 0 in the stationary form, whose rows are then the kernels
in age), a deterministic column `1` (targets, constant drifts, initial states) and the initial shocks of a prior.

Finite horizon (risk neutral): an agent's strategy is a map on the history of its rows (a regressor matrix: one
regressor per row and step, the column 1, a prior shock its rows load at 0).  Its best response is computed in its
passive world (its own control an exogenous impulse, every other agent on its raw map): the cost is quadratic in the
action written on the passive rows, whose first-order condition P C S + B Y' = 0 on the causal mask is solved by
conjugate gradients; the raw map follows as C (I + F C)^-1 (F the rows' response to the agent's own actions).  A
myopic agent's best response is the projection of its instant target -Q_uu^-1 (Q_uz z + q_u / 2) on its raw rows.
Risk-averse agents (theta > 0) minimise the exact entropic cost of the discrete realised cost, -(2 theta)^-1 log det
(I - 2 theta M) plus the linear part's tilt, over the whole map (precommitment) by L-BFGS from the risk-neutral best
response.

Stationary: the kernels in age on [0, L] from one shock per channel at age 0 and Toeplitz strategies; the best response
is the date-0 first-order condition with the discounted continuation through the passive world (own later actions held
on the passive rows: the envelope), linear in the action filter; a myopic agent projects its target on its raw rows.
The cost is the flow sum_a z_a' Q z_a (the undiscounted Gram, as the package reports it).

The equilibrium is the fixed point of the best responses over the raw maps (Jacobi with Anderson mixing).  The scheme
is first order in h; `solve_levels` Richardson-extrapolates over a sequence of step counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

CONST = "const"


# ------------------------------------------------------------------------------------------------ parsing
def _lin(expr):
    """A linear expression of the grammar ({atom: coef} or [[coef, atom], ...]) as a list of (coef, atom)."""
    if expr is None:
        return []
    if isinstance(expr, dict):
        return [(float(c), a) for a, c in expr.items()]
    return [(float(c), a) for c, a in expr]


def _atom(s):
    s = str(s)
    if "@" in s:
        n, lag = s.split("@")
        return n, float(lag)
    return s, 0.0


class ReferenceError(Exception):
    """The reference cannot solve this model (a feature it does not model, or its own fixed point failed)."""


@dataclass
class Parsed:
    states: list
    controls: list
    owner: dict
    shocks: list
    drift: dict          # state -> list of (coef, prim, lag); prim may be CONST
    noise: dict          # state -> {shock: coef}
    initial: dict
    agents: list         # dicts: name, controls, rows [(name, drift, noise, delay)], loss (atoms, Q, q), terminal, myopic, theta, constant
    rho: float
    kind: str
    T: float = None
    window: float = None
    priors: list = field(default_factory=list)   # (name, loads {state: c}, rows {(agent, row): c})
    lags: set = field(default_factory=set)


def parse(d):
    defs = {k: _lin(v) for k, v in (d.get("definitions") or {}).items()}

    def expand(terms, lag0=0.0, depth=0):
        """(coef, atom) -> list of (coef, prim, lag), definitions expanded (their lags add)."""
        if depth > 20:
            raise ReferenceError("definition cycle")
        out = []
        for c, a in terms:
            n, lag = _atom(a)
            if n == CONST:
                out.append((c, CONST, 0.0))
            elif n in defs:
                out += [(c * c2, n2, l2) for c2, n2, l2 in expand(defs[n], lag0 + lag, depth + 1)]
            else:
                out.append((c, n, lag0 + lag))
        return out

    states = list(d.get("states", {}))
    agents_in = d["agents"]
    controls, owner = [], {}
    for an, a in agents_in.items():
        for u in a["controls"]:
            controls.append(u); owner[u] = an
    drift, noise, initial = {}, {}, {}
    for s, spec in d.get("states", {}).items():
        drift[s] = expand(_lin(spec.get("drift")))
        noise[s] = {ch: float(c) for c, ch in _lin(spec.get("noise"))}
        initial[s] = float(spec.get("initial", 0.0) or 0.0)
    agents = []
    for an, a in agents_in.items():
        if a.get("instant") or a.get("monitors") or a.get("integrals"):
            raise ReferenceError("instant observations, monitoring and integrals are not modelled")
        rows = []
        for rn, r in (a.get("signals") or {}).items():
            if "level" in r:
                raise ReferenceError("level rows are not modelled")
            rows.append((rn, expand(_lin(r.get("drift"))), {ch: float(c) for c, ch in _lin(r.get("noise"))},
                         float(r.get("delay", 0.0) or 0.0)))
        quad, lin = [], []
        for term in a.get("loss", []):
            if len(term) == 3:
                quad.append((float(term[0]), expand([(1.0, term[1])]), expand([(1.0, term[2])])))
            else:
                lin.append((float(term[0]), expand([(1.0, term[1])])))
        tquad, tlin = [], []
        for term in a.get("terminal", []) or []:
            if len(term) == 3:
                tquad.append((float(term[0]), expand([(1.0, term[1])]), expand([(1.0, term[2])])))
            else:
                tlin.append((float(term[0]), expand([(1.0, term[1])])))
        agents.append(dict(name=an, controls=list(a["controls"]), rows=rows, quad=quad, lin=lin, tquad=tquad, tlin=tlin,
                           myopic=bool(a.get("myopic", False)), theta=float(a.get("risk_aversion", 0.0) or 0.0),
                           constant=float(a.get("constant", 0.0) or 0.0),
                           tconstant=float(a.get("terminal_constant", 0.0) or 0.0)))
    hz = d.get("horizon", {})
    kind = hz.get("kind", "stationary")
    priors = []
    if kind == "transition":
        past = hz.get("past", {})
        if "model" in past or hz.get("continuation", "stationary") != "end":
            raise ReferenceError("a transition with a past model or a stationary continuation is not modelled")
        for p in past.get("initial", []):
            rows = {}
            for key, c in (p.get("rows") or {}).items():
                an, rn = key.split(".")
                rows[(an, rn)] = float(c)
            priors.append((p["name"], {s: float(c) for s, c in (p.get("loads") or {}).items()}, rows))
        kind = "finite"
    P = Parsed(states=states, controls=controls, owner=owner, shocks=list(d["shocks"]), drift=drift, noise=noise,
               initial=initial, agents=agents, rho=float(hz.get("discount", 0.0) or 0.0), kind=kind,
               T=float(hz["T"]) if "T" in hz else None, window=float(hz["window"]) if "window" in hz else None,
               priors=priors)
    for s in states:
        P.lags |= {l for _, _, l in drift[s] if l}
    for a in agents:
        for _, dr, _, dl in a["rows"]:
            P.lags |= {l for _, _, l in dr if l} | ({dl} if dl else set())
        for _, x, y in a["quad"]:
            P.lags |= {l for _, _, l in x + y if l}
        for _, x in a["lin"]:
            P.lags |= {l for _, _, l in x if l}
        if any(l < 0 for _, x, y in a["quad"] for _, _, l in x + y):
            raise ReferenceError("leads are not modelled")
    return P


# ------------------------------------------------------------------------------------------------ the discrete model
class Discrete:
    """The model on n steps (finite: h = T / n; stationary: n steps of age over the window, h = L / n)."""

    def __init__(self, d, n):
        self.d = d
        self.p = p = parse(d)
        self.stationary = p.kind == "stationary"
        self.n = n
        self.h = h = (p.window if self.stationary else p.T) / n
        for l in p.lags:
            if abs(l / h - round(l / h)) > 1e-9:
                raise ReferenceError(f"lag {l} is not a multiple of the step {h}")
        self.steps = lambda l: int(round(l / h))
        self.has_means = any(p.initial[s] for s in p.states) or any(c for s in p.states for c, n_, _ in p.drift[s] if n_ == CONST) \
            or any(a["lin"] or a["tlin"] for a in p.agents)
        # columns: shocks (per step, or per channel at age 0), the constant, the priors
        nsh = len(p.shocks)
        self.nshock_cols = nsh * (1 if self.stationary else n)
        self.ccol = self.nshock_cols if self.has_means else None
        self.xi0 = self.nshock_cols + (1 if self.has_means else 0)
        self.nW = self.xi0 + len(p.priors)
        self.sidx = {s: i for i, s in enumerate(p.shocks)}
        # each agent's regressors: one per row and step, the constant, the priors its rows load
        self.reg = []
        for a in p.agents:
            extra = []
            if self.has_means:
                extra.append(("1", None))
            for pi, (pn, loads, rows) in enumerate(p.priors):
                if any(an == a["name"] for (an, rn) in rows):
                    extra.append(("xi", pi))
            self.reg.append(dict(rows=[r[0] for r in a["rows"]], delay=[self.steps(r[3]) for r in a["rows"]], extra=extra,
                                 nreg=len(a["rows"]) * n + len(extra)))
        # the loss of each agent as atoms, a symmetric Q and a linear q (and the terminal)
        self.loss = [self._loss_form(a["quad"], a["lin"]) for a in p.agents]
        self.tloss = [self._loss_form(a["tquad"], a["tlin"]) for a in p.agents]
        self.w = np.exp(-p.rho * h * np.arange(n + 1))
        if self.stationary and self.has_means:
            raise ReferenceError("stationary means are not modelled")

    def _loss_form(self, quad, lin):
        atoms = []
        def ix(n_, l):
            k = (n_, self.steps(l))
            if k not in atoms:
                atoms.append(k)
            return atoms.index(k)
        entries = []
        for c, x, y in quad:
            for cx, nx, lx in x:
                for cy, ny, ly in y:
                    entries.append((c * cx * cy, ix(nx, lx), ix(ny, ly)))
        lins = []
        for c, x in lin:
            for cx, nx, lx in x:
                lins.append((c * cx, ix(nx, lx)))
        m = len(atoms)
        Q = np.zeros((m, m)); q = np.zeros(m)
        for c, i, j in entries:
            Q[i, j] += 0.5 * c; Q[j, i] += 0.5 * c
        for c, i in lins:
            q[i] += c
        return atoms, Q, q

    # -------------------------------------------------------------------- the mask of an agent's map
    def mask(self, ai):
        """(nc n, nreg) boolean: control u at step k may read regressor r."""
        n, R = self.n, self.reg[ai]; a = self.p.agents[ai]
        nc = len(a["controls"]); M = np.zeros((nc * n, R["nreg"]), bool)
        for r, dl in enumerate(R["delay"]):
            for k in range(n):
                hi = k - dl        # increments j <= k - 1 - dl
                if hi > 0:
                    for u in range(nc):
                        M[u * n + k, r * n: r * n + hi] = True
        for e in range(len(R["extra"])):
            M[:, len(R["rows"]) * n + e] = True
        if self.stationary:
            # Toeplitz strategies read ages 1 + dl .. n - 1; the extras are not used in the stationary form
            M[:, len(R["rows"]) * n:] = False
        return M

    # -------------------------------------------------------------------- the forward pass
    def forward(self, maps, exo=None):
        """Rows over the columns of every state (n+1 steps), control and signal row (n steps).  maps[ai] is the raw
        map (nc n, nreg); agent `exo` plays an exogenous impulse (extra columns, one per control and step; one per
        control at step 0 in the stationary form)."""
        p, n, h = self.p, self.n, self.h
        sq = math.sqrt(h)
        nimp = 0
        if exo is not None:
            nimp = len(p.agents[exo]["controls"]) * (1 if self.stationary else n)
        ncol = self.nW + nimp
        X = {s: np.zeros((n + 1, ncol)) for s in p.states}
        U = {u: np.zeros((n, ncol)) for u in p.controls}
        Y = [np.zeros((len(a["rows"]) * n + len(self.reg[ai]["extra"]), ncol)) for ai, a in enumerate(p.agents)]
        for ai, R in enumerate(self.reg):
            base = len(R["rows"]) * n
            for e, (kind, pi) in enumerate(R["extra"]):
                if kind == "1":
                    Y[ai][base + e, self.ccol] = 1.0
                else:
                    pn, loads, rows = p.priors[pi]
                    c = sum(v for (an, rn), v in rows.items() if an == p.agents[ai]["name"])
                    Y[ai][base + e, self.xi0 + pi] = c
        if not self.stationary:
            for s in p.states:
                if p.initial[s]:
                    X[s][0, self.ccol] = p.initial[s]
            for pi, (pn, loads, rows) in enumerate(p.priors):
                for s, c in loads.items():
                    X[s][0, self.xi0 + pi] += c

        def val(name, k):
            if name in X:
                return X[name][k] if k >= 0 else None
            return U[name][k] if 0 <= k < n else None

        def lin(terms, k):
            out = np.zeros(ncol)
            for c, nm, l in terms:
                if nm == CONST:
                    out[self.ccol] += c
                    continue
                v = val(nm, k - self.steps(l))
                if v is not None:
                    out += c * v
            return out

        def noise_col(ch, k):
            j = self.sidx[ch]
            if self.stationary:
                return j if k == 0 else None
            return j * n + k

        for k in range(n + 1):
            if k > 0:
                for s in p.states:
                    X[s][k] = X[s][k - 1] + h * lin(p.drift[s], k - 1)
                    for ch, c in p.noise[s].items():
                        col = noise_col(ch, k - 1)
                        if col is not None:
                            X[s][k, col] += sq * c
            if k == n:
                break
            for ai, a in enumerate(p.agents):
                for ui, u in enumerate(a["controls"]):
                    if exo == ai:
                        if self.stationary:
                            if k == 0:
                                U[u][k, self.nW + ui] = 1.0
                        else:
                            U[u][k, self.nW + ui * n + k] = 1.0
                    else:
                        U[u][k] = maps[ai][ui * n + k] @ Y[ai]
            for ai, a in enumerate(p.agents):
                for ri, (rn, dr, nz, dl) in enumerate(a["rows"]):
                    y = h * lin(dr, k)
                    for ch, c in nz.items():
                        col = noise_col(ch, k)
                        if col is not None:
                            y[col] += sq * c
                    Y[ai][ri * n + k] = y
        return X, U, Y

    def atoms_at(self, X, U, atoms, k):
        rows = []
        for nm, l in atoms:
            kk = k - l
            if nm in X:
                rows.append(X[nm][kk] if kk >= 0 else np.zeros(X[nm].shape[1]))
            else:
                rows.append(U[nm][kk] if 0 <= kk < self.n else np.zeros(U[nm].shape[1]))
        return np.array(rows) if rows else np.zeros((0, next(iter(X.values())).shape[1] if X else U[next(iter(U))].shape[1]))

    # -------------------------------------------------------------------- costs
    def cost(self, maps, ai, X=None, U=None):
        if X is None:
            X, U, _ = self.forward(maps)
        atoms, Q, q = self.loss[ai]; a = self.p.agents[ai]
        tot = 0.0
        if self.stationary:
            for k in range(self.n):
                Z = self.atoms_at(X, U, atoms, k)
                tot += float(np.einsum("ic,ij,jc->", Z, Q, Z))
            return tot + a["constant"]
        for k in range(self.n):
            Z = self.atoms_at(X, U, atoms, k)
            v = float(np.einsum("ic,ij,jc->", Z, Q, Z))
            if self.ccol is not None:
                v += float(q @ Z[:, self.ccol])
            tot += self.h * self.w[k] * (v + a["constant"])
        tatoms, TQ, tq = self.tloss[ai]
        if tatoms:
            Z = np.array([X[nm][self.n] for nm, l in tatoms])
            v = float(np.einsum("ic,ij,jc->", Z, TQ, Z))
            if self.ccol is not None:
                v += float(tq @ Z[:, self.ccol])
            tot += self.w[self.n] * v
        tot += self.w[self.n] * a["tconstant"]
        return tot

    # -------------------------------------------------------------------- the passive world's quadratic form
    def _passive(self, maps, ai):
        """P (nimp, nimp), B (nimp, nW) and the passive regressors Ytil (nreg, nW), F (nreg, nimp): cost in the passive
        world with impulse weights u (nimp, nW) is tr(u'Pu) + 2 tr(u'B) + const."""
        X, U, Y = self.forward(maps, exo=ai)
        atoms, Q, q = self.loss[ai]; W = self.nW
        nimp = next(iter(X.values())).shape[1] - W if X else U[self.p.controls[0]].shape[1] - W
        P = np.zeros((nimp, nimp)); B = np.zeros((nimp, W))
        for k in range(self.n):
            Z = self.atoms_at(X, U, atoms, k)
            wk = 1.0 if self.stationary else self.h * self.w[k]
            ZI, ZW = Z[:, W:], Z[:, :W]
            QZI = Q @ ZI
            P += wk * ZI.T @ QZI
            B += wk * QZI.T @ ZW
            if self.ccol is not None and not self.stationary:
                B[:, self.ccol] += 0.5 * wk * ZI.T @ q
        if not self.stationary:
            tatoms, TQ, tq = self.tloss[ai]
            if tatoms:
                Z = np.array([X[nm][self.n] for nm, l in tatoms]); wk = self.w[self.n]
                ZI, ZW = Z[:, W:], Z[:, :W]
                P += wk * ZI.T @ TQ @ ZI; B += wk * (TQ @ ZI).T @ ZW
                if self.ccol is not None:
                    B[:, self.ccol] += 0.5 * wk * ZI.T @ tq
        return P, B, Y[ai][:, :W], Y[ai][:, W:], (X, U, Y)

    # -------------------------------------------------------------------- best responses (finite)
    def best_response_finite(self, maps, ai, C0=None):
        a = self.p.agents[ai]
        if a["myopic"]:
            return self._myopic_finite(maps, ai)
        P, B, Yt, F, _ = self._passive(maps, ai)
        M = self.mask(ai)
        S = Yt @ Yt.T
        rhs = -(B @ Yt.T)
        idx = np.nonzero(M.ravel())[0]
        nun = len(idx)
        shape = M.shape

        def op(v):
            Cm = np.zeros(shape).ravel(); Cm[idx] = v
            Cm = Cm.reshape(shape)
            return (P @ Cm @ S).ravel()[idx]

        b = rhs.ravel()[idx]
        if nun <= 2500:
            # dense: the operator's matrix by columns of the Kronecker structure
            rows_u, cols_r = np.unravel_index(idx, shape)
            A = P[np.ix_(rows_u, rows_u)] * S[np.ix_(cols_r, cols_r)]
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                raise ReferenceError("singular best-response system")
        else:
            Aop = LinearOperator((nun, nun), matvec=op)
            dP = np.diag(P); dS = np.diag(S)
            rows_u, cols_r = np.unravel_index(idx, shape)
            dia = dP[rows_u] * dS[cols_r]
            Mpre = LinearOperator((nun, nun), matvec=lambda v: v / dia)
            x0 = None if C0 is None else C0.ravel()[idx]
            x, info = cg(Aop, b, x0=x0, rtol=1e-12, atol=0.0, maxiter=20000, M=Mpre)
            if info != 0:
                raise ReferenceError(f"conjugate gradients did not converge ({info})")
        C = np.zeros(shape).ravel(); C[idx] = x; C = C.reshape(shape)
        # raw map: Gamma = C (I + F C)^-1
        G = C @ np.linalg.inv(np.eye(shape[1]) + F @ C)
        G[~M] = 0.0
        return G

    def _myopic_split(self, ai):
        atoms, Q, q = self.loss[ai]; a = self.p.agents[ai]
        own = [i for i, (nm, l) in enumerate(atoms) if nm in a["controls"] and l == 0]
        rest = [i for i in range(len(atoms)) if i not in own]
        order = [atoms[i][0] for i in own]
        Quu = Q[np.ix_(own, own)]
        return own, rest, order, Quu, Q[np.ix_(own, rest)], q[own]

    def _myopic_target(self, X, U, ai, k):
        atoms, Q, q = self.loss[ai]; a = self.p.agents[ai]
        own, rest, order, Quu, Quz, qu = self._myopic_split(ai)
        Z = self.atoms_at(X, U, atoms, k)
        rhs = Quz @ Z[rest]
        if self.ccol is not None and not self.stationary:
            rhs[:, self.ccol] += 0.5 * qu
        T = -np.linalg.solve(Quu, rhs)
        out = np.zeros((len(a["controls"]), T.shape[1]))
        for i, nm in enumerate(order):
            out[a["controls"].index(nm)] = T[i]
        return out

    def _myopic_finite(self, maps, ai):
        X, U, Y = self.forward(maps)
        M = self.mask(ai); n = self.n; a = self.p.agents[ai]
        Yr = Y[ai]
        G = np.zeros(M.shape)
        for k in range(n):
            T = self._myopic_target(X, U, ai, k)
            for ui in range(len(a["controls"])):
                sel = np.nonzero(M[ui * n + k])[0]
                if len(sel) == 0:
                    continue
                R = Yr[sel]
                coef, *_ = np.linalg.lstsq(R.T, T[ui], rcond=None)
                G[ui * n + k, sel] = coef
        return G

    # -------------------------------------------------------------------- best responses (stationary)
    def _toeplitz(self, gam, nc, nr):
        """Block lower-triangular Toeplitz (nc n, nr n) from filters gam[u][r][lag] (lag 0 .. n-1)."""
        n = self.n
        G = np.zeros((nc * n, nr * n))
        i, j = np.tril_indices(n)
        for u in range(nc):
            for r in range(nr):
                blk = np.zeros((n, n)); blk[i, j] = gam[u, r, i - j]
                G[u * n:(u + 1) * n, r * n:(r + 1) * n] = blk
        return G

    def _filters(self, G, nc, nr):
        n = self.n
        return np.array([[G[u * n:(u + 1) * n, r * n][:n] for r in range(nr)] for u in range(nc)])

    def stationary_map(self, gam, ai):
        """The Toeplitz raw map of agent ai from its filters (nc, nr, n); extras unused."""
        a = self.p.agents[ai]; nc, nr = len(a["controls"]), len(a["rows"])
        G = np.zeros((nc * self.n, self.reg[ai]["nreg"]))
        G[:, :nr * self.n] = self._toeplitz(gam, nc, nr)
        G[~self.mask(ai)] = 0.0
        return G

    @staticmethod
    def _diag_cumsum(M):
        """Cg[t, d + n - 1] = sum_{s < t} M[s, s + d] (M square n x n, zero outside), t = 0 .. n."""
        n = M.shape[0]
        s = np.arange(n)[:, None]; d = np.arange(-(n - 1), n)[None, :]
        a = s + d
        ok = (a >= 0) & (a < n)
        Dg = np.where(ok, M[s, np.clip(a, 0, n - 1)], 0.0)
        return np.vstack([np.zeros((1, 2 * n - 1)), np.cumsum(Dg, axis=0)])

    def best_response_stationary(self, maps, ai):
        """The date-0 first-order condition: E[sum_tau w_tau dflow_tau / du_0 . ytil_{-j}] = 0, own later actions held
        on the passive rows (the envelope).  Returns the new filters (nc, nr, n) on the raw rows.

        With V(tau) = 2 w_tau Q ZI(tau) (the flow's marginal along the response to u_0), Wm = V * Ytil (a convolution in
        age), H_{u,r} = ZI_u * Ytil_r (the atoms' response to u reading row r at lag 0) and ZW the passive kernels:
        FOC(u', r', j') = sum_{s < n - j'} <Wm_{u'r'}(s), Z(s + j')>, Z = ZW + sum gt[u, r, j] H_{ur}(. - j)."""
        a = self.p.agents[ai]; n = self.n; nc, nr = len(a["controls"]), len(a["rows"])
        if a["myopic"]:
            return self._myopic_stationary(maps, ai)
        X, U, Y = self.forward(maps, exo=ai)
        atoms, Q, q = self.loss[ai]; W = self.nW
        Zs = np.array([self.atoms_at(X, U, atoms, k) for k in range(n)])
        ZW = Zs[:, :, :W]                                # (n, A, W)
        ZI = Zs[:, :, W:]                                # (n, A, nc)
        Yt = Y[ai][:nr * n, :W].reshape(nr, n, W)
        F = Y[ai][:nr * n, W:].reshape(nr, n, nc)
        wd = np.exp(-self.p.rho * self.h * np.arange(n))
        dl = self.reg[ai]["delay"]
        V = 2.0 * wd[:, None, None] * np.einsum("ij,tju->tiu", Q, ZI)          # (n, A, nc)

        def conv(P_, Yr):         # (n, A) x (n, W) -> (n, A, W): sum_{b <= s} P_(s - b) Yr(b)
            out = np.zeros((n, P_.shape[1], Yr.shape[1]))
            for b in range(n):
                out[b:] += P_[:n - b, :, None] * Yr[b][None, None, :]
            return out

        Wm = {(u, r): conv(V[:, :, u], Yt[r]).reshape(n, -1) for u in range(nc) for r in range(nr)}
        H = {(u, r): conv(ZI[:, :, u], Yt[r]).reshape(n, -1) for u in range(nc) for r in range(nr)}
        ZWf = ZW.reshape(n, -1)
        unknowns = [(u, r, j) for u in range(nc) for r in range(nr) for j in range(1 + dl[r], n)]
        pos = {k: i for i, k in enumerate(unknowns)}
        nun = len(unknowns)
        if nun == 0:
            raise ReferenceError("no information")
        f0 = np.zeros(nun); J = np.zeros((nun, nun))
        for u2 in range(nc):
            for r2 in range(nr):
                j2 = np.arange(1 + dl[r2], n)
                rows = [pos[(u2, r2, j)] for j in j2]
                C0 = self._diag_cumsum(Wm[(u2, r2)] @ ZWf.T)
                f0[rows] = C0[n - j2, j2 + n - 1]
                for u in range(nc):
                    for r in range(nr):
                        j1 = np.arange(1 + dl[r], n)
                        cols = [pos[(u, r, j)] for j in j1]
                        Cg = self._diag_cumsum(Wm[(u2, r2)] @ H[(u, r)].T)
                        J[np.ix_(rows, cols)] = Cg[(n - j2)[:, None], (j2[:, None] - j1[None, :]) + n - 1]
        try:
            x = np.linalg.solve(J, -f0)
        except np.linalg.LinAlgError:
            raise ReferenceError("singular stationary best response")
        gt = np.zeros((nc, nr, n))
        for i, (u, r, j) in enumerate(unknowns):
            gt[u, r, j] = x[i]
        # raw filters: T(g) = T(gt) (I + T(F) T(gt))^-1
        Tg = self._toeplitz(gt, nc, nr)
        TF = np.zeros((nr * n, nc * n))
        i_, j_ = np.tril_indices(n)
        for r in range(nr):
            for u in range(nc):
                blk = np.zeros((n, n)); blk[i_, j_] = F[r, i_ - j_, u]
                TF[r * n:(r + 1) * n, u * n:(u + 1) * n] = blk
        G = Tg @ np.linalg.inv(np.eye(nr * n) + TF @ Tg)
        return np.array([[G[u * n:(u + 1) * n, r * n] for r in range(nr)] for u in range(nc)])

    def _myopic_stationary(self, maps, ai):
        a = self.p.agents[ai]; n = self.n; nc, nr = len(a["controls"]), len(a["rows"])
        X, U, Y = self.forward(maps)
        Tk = np.array([self._myopic_target(X, U, ai, k) for k in range(n)])     # (age, nc, W): target kernels
        Yr = Y[ai][:nr * n].reshape(nr, n, -1)
        dl = self.reg[ai]["delay"]
        regs = [(r, j) for r in range(nr) for j in range(1 + dl[r], n)]
        # autocovariances: Acf[r2, r][d] = sum_b Yr[r2, b + d] . Yr[r, b], d >= 0
        Acf = np.zeros((nr, nr, n))
        for r2 in range(nr):
            for r in range(nr):
                for d in range(n):
                    Acf[r2, r, d] = float(np.einsum("bc,bc->", Yr[r2, d:], Yr[r, :n - d]))
        m = len(regs)
        rr = np.array([r for r, j in regs]); jj = np.array([j for r, j in regs])
        dd = jj[:, None] - jj[None, :]
        # Cov(y_{r,-j}, y_{r2,-j2}) = Acf[r2, r][j - j2] for j >= j2, else Acf[r, r2][j2 - j]
        Gm = np.where(dd >= 0, Acf[rr[None, :], rr[:, None], np.abs(dd)], Acf[rr[:, None], rr[None, :], np.abs(dd)])
        rhs = np.zeros((m, nc))
        for i, (r, j) in enumerate(regs):
            rhs[i] = np.einsum("buc,bc->u", Tk[j:], Yr[r, :n - j])
        x = np.linalg.lstsq(Gm, rhs, rcond=None)[0]
        g = np.zeros((nc, nr, n))
        for i, (r, j) in enumerate(regs):
            g[:, r, j] = x[i]
        return g

    # -------------------------------------------------------------------- entropic best response (finite)
    def _entropic_parts(self, X, U, ai):
        """Z_r (random cols) and z_c (the constant) of the agent's atoms at every step, and its terminal atoms."""
        atoms, Q, q = self.loss[ai]
        return [self.atoms_at(X, U, atoms, k) for k in range(self.n)]

    def entropic_cost(self, maps, ai):
        X, U, _ = self.forward(maps)
        th = self.p.agents[ai]["theta"]
        M, m, c0 = self._cost_quadratic(X, U, ai)
        return self._entropic(M, m, c0, th)[0]

    def _rand_cols(self):
        cols = [c for c in range(self.nW) if c != self.ccol]
        return np.array(cols, int)

    def _cost_quadratic(self, X, U, ai, dZ=None):
        """C = g'Mg + 2 m'g + c0 over the random columns g."""
        atoms, Q, q = self.loss[ai]; a = self.p.agents[ai]
        rc = self._rand_cols(); nr = len(rc)
        M = np.zeros((nr, nr)); m = np.zeros(nr); c0 = 0.0
        for k in range(self.n):
            Z = self.atoms_at(X, U, atoms, k)
            wk = self.h * self.w[k]
            Zr = Z[:, rc]; zc = Z[:, self.ccol] if self.ccol is not None else np.zeros(len(atoms))
            M += wk * Zr.T @ Q @ Zr
            m += wk * Zr.T @ (Q @ zc + 0.5 * q)
            c0 += wk * (zc @ Q @ zc + q @ zc + a["constant"])
        tatoms, TQ, tq = self.tloss[ai]
        if tatoms:
            Z = np.array([X[nm][self.n] for nm, l in tatoms]); wk = self.w[self.n]
            Zr = Z[:, rc]; zc = Z[:, self.ccol] if self.ccol is not None else np.zeros(len(tatoms))
            M += wk * Zr.T @ TQ @ Zr; m += wk * Zr.T @ (TQ @ zc + 0.5 * tq); c0 += wk * (zc @ TQ @ zc + tq @ zc)
        c0 += self.w[self.n] * a["tconstant"]
        return M, m, c0

    @staticmethod
    def _entropic(M, m, c0, th):
        if th == 0.0:
            return c0 + float(np.trace(M)), np.eye(len(M)), np.zeros(len(m))
        A = np.eye(len(M)) - 2.0 * th * M
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf, None, None
        Wm = np.linalg.solve(A, np.column_stack([np.eye(len(M)), m]))
        W = Wm[:, :-1]; Wmv = Wm[:, -1]
        J = c0 - logdet / (2.0 * th) + 2.0 * th * float(m @ Wmv)
        return J, W, Wmv

    def best_response_entropic(self, maps, ai, maxiter=3000):
        """Precommitment: minimise the exact entropic cost over the agent's map on its passive rows, by L-BFGS from
        the risk-neutral best response."""
        from scipy.optimize import minimize
        th = self.p.agents[ai]["theta"]
        G0 = self.best_response_finite(maps, ai)
        P_, B_, Yt, F, (X, U, Y) = self._passive(maps, ai)
        atoms, Q, q = self.loss[ai]; W = self.nW
        rc = self._rand_cols(); M_mask = self.mask(ai); idx = np.nonzero(M_mask.ravel())[0]; shape = M_mask.shape
        ZWs = [self.atoms_at(X, U, atoms, k) for k in range(self.n)]
        ZI = np.array([z[:, W:] for z in ZWs]); ZW = np.array([z[:, :W] for z in ZWs])
        tatoms, TQ, tq = self.tloss[ai]
        if tatoms:
            ZT = np.array([X[nm][self.n] for nm, l in tatoms]); ZTI, ZTW = ZT[:, W:], ZT[:, :W]
        wk = self.h * self.w[:self.n]
        a = self.p.agents[ai]
        cc = self.ccol

        # C0 from G0: the passive action equivalent of the raw map, C = G (I - F G)^-1
        C_init = G0 @ np.linalg.inv(np.eye(shape[1]) - F @ G0)

        def f(v):
            C = np.zeros(shape).ravel(); C[idx] = v; C = C.reshape(shape)
            u = C @ Yt                                   # (nimp, W)
            Z = ZW + np.einsum("kai,iw->kaw", ZI, u)     # (n, natoms, W)
            Zr = Z[:, :, rc]; zc = Z[:, :, cc] if cc is not None else np.zeros(Z.shape[:2])
            QZr = np.einsum("ij,kjw->kiw", Q, Zr)
            M = np.einsum("k,kiw,kiv->wv", wk, Zr, QZr)
            Qzc = zc @ Q.T
            m = np.einsum("k,kiw,ki->w", wk, Zr, Qzc + 0.5 * q)
            c0 = float(np.einsum("k,ki,ki->", wk, zc, Qzc) + wk @ (zc @ q)) + a["constant"] * wk.sum()
            if tatoms:
                ZTt = ZTW + ZTI @ u; ZTr = ZTt[:, rc]; ztc = ZTt[:, cc] if cc is not None else np.zeros(len(tatoms))
                wT = self.w[self.n]
                M = M + wT * ZTr.T @ TQ @ ZTr; m = m + wT * ZTr.T @ (TQ @ ztc + 0.5 * tq); c0 += wT * (ztc @ TQ @ ztc + tq @ ztc)
            c0 += self.w[self.n] * a["tconstant"]
            J, Wi, Wm = self._entropic(M, m, c0, th)
            if not np.isfinite(J):
                return 1e30, np.zeros_like(v)
            GM = Wi + 4.0 * th * th * np.outer(Wm, Wm)
            gm = 4.0 * th * Wm
            # dJ/dZr_k = 2 wk Q Zr_k GM + wk (Q zc_k + q/2) gm' ; dJ/dzc_k = wk Q Zr_k gm + wk (2 Q zc_k + q)
            dZ = np.zeros_like(Z)
            dZ[:, :, rc] = 2.0 * np.einsum("k,kiw,wv->kiv", wk, QZr, GM) + np.einsum("k,ki,v->kiv", wk, Qzc + 0.5 * q, gm)
            if cc is not None:
                dZ[:, :, cc] = np.einsum("k,kiw,w->ki", wk, QZr, gm) + wk[:, None] * (2.0 * Qzc + q)
            du = np.einsum("kai,kaw->iw", ZI, dZ)
            if tatoms:
                dZT = np.zeros_like(ZTt)
                dZT[:, rc] = 2.0 * wT * (TQ @ ZTr) @ GM + wT * np.outer(TQ @ ztc + 0.5 * tq, gm)
                if cc is not None:
                    dZT[:, cc] = wT * (TQ @ ZTr) @ gm + wT * (2.0 * TQ @ ztc + tq)
                du += ZTI.T @ dZT
            gC = du @ Yt.T
            return J, gC.ravel()[idx]

        res = minimize(f, C_init.ravel()[idx], jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-13, "ftol": 1e-16, "maxcor": 30})
        C = np.zeros(shape).ravel(); C[idx] = res.x; C = C.reshape(shape)
        G = C @ np.linalg.inv(np.eye(shape[1]) + F @ C)
        G[~M_mask] = 0.0
        return G

    # -------------------------------------------------------------------- the equilibrium
    def equilibrium(self, tol=None, maxit=400, m_anderson=6, beta=1.0, verbose=False):
        p = self.p; nA = len(p.agents)
        if tol is None:
            # L-BFGS best responses are exact to about 1e-9: the fixed point to 1e-7 then
            tol = 1e-7 if (not self.stationary and any(a["theta"] > 0 for a in p.agents)) else 1e-10
        if self.stationary:
            shapes = [(len(a["controls"]), len(a["rows"]), self.n) for a in p.agents]
        else:
            shapes = [self.mask(ai).shape for ai in range(nA)]
        sizes = [int(np.prod(s)) for s in shapes]

        def unpack(x):
            out, o = [], 0
            for s, z in zip(shapes, sizes):
                out.append(x[o:o + z].reshape(s)); o += z
            return out

        def to_maps(parts):
            if self.stationary:
                return [self.stationary_map(g, ai) for ai, g in enumerate(parts)]
            return parts

        def br(x):
            parts = unpack(x); maps = to_maps(parts); out = []
            for ai, a in enumerate(p.agents):
                if self.stationary:
                    out.append(self.best_response_stationary(maps, ai).ravel())
                elif a["theta"] > 0:
                    out.append(self.best_response_entropic(maps, ai).ravel())
                else:
                    out.append(self.best_response_finite(maps, ai).ravel())
            return np.concatenate(out)

        x = np.zeros(sum(sizes))
        Xh, Fh = [], []
        res = np.inf
        for it in range(maxit):
            gx = br(x)
            f = gx - x
            res = float(np.abs(f).max() / max(1.0, np.abs(gx).max()))
            if verbose:
                print(f"  ref it {it}: residual {res:.2e}", flush=True)
            if not np.all(np.isfinite(gx)):
                raise ReferenceError("the reference's best response is not finite")
            if res < tol:
                x = gx
                break
            Xh.append(x.copy()); Fh.append(f.copy())
            Xh, Fh = Xh[-m_anderson - 1:], Fh[-m_anderson - 1:]
            if len(Fh) > 1:
                dF = np.array([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)]).T
                dX = np.array([Xh[i + 1] - Xh[i] for i in range(len(Xh) - 1)]).T
                try:
                    gam = np.linalg.lstsq(dF, f, rcond=1e-12)[0]
                    x = x + beta * f - (dX + beta * dF) @ gam
                except np.linalg.LinAlgError:
                    x = x + beta * f
            else:
                x = x + beta * f
        else:
            raise ReferenceError(f"the reference's fixed point did not converge (residual {res:.1e})")
        maps = to_maps(unpack(x))
        return maps

    def solve(self, **kw):
        maps = self.equilibrium(**kw)
        X, U, _ = self.forward(maps)
        costs = {a["name"]: self.cost(maps, ai, X, U) for ai, a in enumerate(self.p.agents)}
        out = {"costs": costs, "maps": maps}
        if any(a["theta"] > 0 for a in self.p.agents):
            out["entropic"] = {a["name"]: (self.entropic_cost(maps, ai) if a["theta"] > 0 else costs[a["name"]])
                               for ai, a in enumerate(self.p.agents)}
        return out


def solve_levels(d, levels, key="costs", **kw):
    """Solve at each step count in `levels` (each a multiple of the last); return the per-agent values, the
    Richardson limits of successive pairs (first order: 2 c(2n) - c(n) for a doubling) and an error estimate."""
    vals = []
    for n in levels:
        vals.append(Discrete(d, n).solve(**kw)[key])
    names = list(vals[0])
    out = {"levels": list(levels), "values": vals, "limit": {}, "error": {}}
    for nm in names:
        v = np.array([x[nm] for x in vals], float)
        if not np.all(np.isfinite(v)):
            out["limit"][nm] = float("nan"); out["error"][nm] = float("inf")
            continue
        ns_ = np.array(levels, float)
        rich = [(ns_[i + 1] * v[i + 1] - ns_[i] * v[i]) / (ns_[i + 1] - ns_[i]) for i in range(len(v) - 1)]
        if len(rich) >= 2:
            # second-order Richardson on the first-order limits (h^2 term)
            r2 = [(ns_[i + 2] ** 2 * (rich[i + 1]) - ns_[i + 1] ** 2 * rich[i]) / (ns_[i + 2] ** 2 - ns_[i + 1] ** 2)
                  for i in range(len(rich) - 1)]
            lim = r2[-1]; err = abs(r2[-1] - rich[-1])
        elif rich:
            lim = rich[-1]; err = abs(rich[-1] - v[-1])
        else:
            lim = v[-1]; err = np.inf
        out["limit"][nm] = float(lim); out["error"][nm] = float(err)
    return out
