"""Closed-form one-agent LQG (separation: a Riccati gain on the Kalman estimate) for the fuzz campaign, independent of
the package.  One agent, no lags or delays, no myopia, signals on the states:

    dx = (A x + B u + c) dt + Sig dW,   dy = H x dt + Gam dW,   flow z'Qz + q'z (z = (x, u)) discounted by rho,
    terminal x'Qt x + qt'x at T, the loss's constant and the terminal constant.

Finite horizon: the discounted Riccati equation with the affine part (the state augmented by a constant 1) backwards
from the terminal loss, the Kalman-Bucy filter from a known initial state (correlated noises allowed), and the second
moments of (x, xhat, 1) forwards; the cost is the discounted integral.  Stationary (no means): the algebraic Riccati
equations and the Lyapunov equation; the cost is the flow (the package's stationary convention).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

from reference import CONST, ReferenceError, _atom, _lin


def matrices(d):
    states = list(d["states"]); agents = d["agents"]
    if len(agents) != 1:
        raise ReferenceError("one agent only")
    (an, a), = agents.items()
    if a.get("myopic") or a.get("instant") or a.get("monitors") or a.get("integrals"):
        raise ReferenceError("plain agent only")
    if d.get("definitions"):
        raise ReferenceError("no definitions")
    ctr = list(a["controls"]); shocks = list(d["shocks"])
    ns_, nc, nw = len(states), len(ctr), len(shocks)
    A = np.zeros((ns_, ns_)); B = np.zeros((ns_, nc)); c = np.zeros(ns_); Sig = np.zeros((ns_, nw)); x0 = np.zeros(ns_)
    for i, s in enumerate(states):
        spec = d["states"][s]
        for coef, atom in _lin(spec.get("drift")):
            nm, lag = _atom(atom)
            if lag:
                raise ReferenceError("lags")
            if nm == CONST:
                c[i] += coef
            elif nm in states:
                A[i, states.index(nm)] += coef
            else:
                B[i, ctr.index(nm)] += coef
        for coef, ch in _lin(spec.get("noise")):
            Sig[i, shocks.index(ch)] += coef
        x0[i] = float(spec.get("initial", 0.0) or 0.0)
    rows = list((a.get("signals") or {}).values())
    H = np.zeros((len(rows), ns_)); Gam = np.zeros((len(rows), nw))
    for r, row in enumerate(rows):
        if float(row.get("delay", 0.0) or 0.0):
            raise ReferenceError("delays")
        for coef, atom in _lin(row.get("drift")):
            nm, lag = _atom(atom)
            if lag or nm not in states:
                raise ReferenceError("a row on a control or a lag")
            H[r, states.index(nm)] += coef
        for coef, ch in _lin(row.get("noise")):
            Gam[r, shocks.index(ch)] += coef
    names = states + ctr
    nz = len(names)
    Q = np.zeros((nz, nz)); q = np.zeros(nz)
    for term in a.get("loss", []):
        ix = []
        for atom in term[1:]:
            nm, lag = _atom(atom)
            if lag:
                raise ReferenceError("lags")
            ix.append(names.index(nm))
        if len(ix) == 2:
            Q[ix[0], ix[1]] += 0.5 * float(term[0]); Q[ix[1], ix[0]] += 0.5 * float(term[0])
        else:
            q[ix[0]] += float(term[0])
    Qt = np.zeros((ns_, ns_)); qt = np.zeros(ns_)
    for term in a.get("terminal", []) or []:
        ix = [states.index(_atom(x)[0]) for x in term[1:]]
        if len(ix) == 2:
            Qt[ix[0], ix[1]] += 0.5 * float(term[0]); Qt[ix[1], ix[0]] += 0.5 * float(term[0])
        else:
            qt[ix[0]] += float(term[0])
    hz = d["horizon"]
    return dict(A=A, B=B, c=c, Sig=Sig, x0=x0, H=H, Gam=Gam, Q=Q, q=q, Qt=Qt, qt=qt, ns=ns_, nc=nc, agent=an,
                const=float(a.get("constant", 0.0) or 0.0), tconst=float(a.get("terminal_constant", 0.0) or 0.0),
                rho=float(hz.get("discount", 0.0) or 0.0), kind=hz.get("kind", "stationary"), T=hz.get("T"))


def solve(d):
    m = matrices(d)
    if m["kind"] == "stationary":
        return _stationary(m)
    if m["kind"] != "finite":
        raise ReferenceError("finite or stationary only")
    return _finite(m)


def _stationary(m):
    A, B, Sig, H, Gam, Q = m["A"], m["B"], m["Sig"], m["H"], m["Gam"], m["Q"]
    if np.any(m["c"]) or np.any(m["q"]) or np.any(m["x0"]):
        raise ReferenceError("stationary means")
    n, k = m["ns"], m["nc"]
    Qxx, Qxu, Quu = Q[:n, :n], Q[:n, n:], Q[n:, n:]
    Arho = A - 0.5 * m["rho"] * np.eye(n)
    try:
        P = solve_continuous_are(Arho, B, Qxx, Quu, s=Qxu)
        Pi = solve_continuous_are(A.T, H.T, Sig @ Sig.T, Gam @ Gam.T, s=Sig @ Gam.T)
    except Exception as exc:
        raise ReferenceError(f"no algebraic Riccati solution: {exc}")
    K = np.linalg.solve(Quu, B.T @ P + Qxu.T)
    L = (Pi @ H.T + Sig @ Gam.T) @ np.linalg.inv(Gam @ Gam.T)
    F = np.block([[A, -B @ K], [L @ H, A - B @ K - L @ H]])
    G = np.vstack([Sig, L @ Gam])
    if np.max(np.linalg.eigvals(F).real) >= -1e-9:
        raise ReferenceError("the closed loop is not stable")
    S = solve_continuous_lyapunov(F, -G @ G.T)
    Mz = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((k, n)), -K]])
    return {m["agent"]: float(np.trace(Mz.T @ Q @ Mz @ S)) + m["const"]}


def _finite(m):
    A, B, c, Sig, H, Gam, Q, q = m["A"], m["B"], m["c"], m["Sig"], m["H"], m["Gam"], m["Q"], m["q"]
    n, k, rho, T = m["ns"], m["nc"], m["rho"], float(m["T"])
    Qxx, Qxu, Quu = Q[:n, :n], Q[:n, n:], Q[n:, n:]
    qx, qu = q[:n], q[n:]
    # augmented y = (x, 1)
    Aa = np.zeros((n + 1, n + 1)); Aa[:n, :n] = A; Aa[:n, n] = c
    Ba = np.vstack([B, np.zeros((1, k))])
    Qa = np.zeros((n + 1, n + 1)); Qa[:n, :n] = Qxx; Qa[:n, n] = Qa[n, :n] = 0.5 * qx
    Na = np.vstack([Qxu, 0.5 * qu[None, :]])
    Qta = np.zeros((n + 1, n + 1)); Qta[:n, :n] = m["Qt"]; Qta[:n, n] = Qta[n, :n] = 0.5 * m["qt"]
    Ri = np.linalg.inv(Quu)

    def ric(t, p):
        P = p.reshape(n + 1, n + 1)
        PBN = P @ Ba + Na
        dP = -(Aa.T @ P + P @ Aa - rho * P + Qa - PBN @ Ri @ PBN.T)
        return dP.ravel()

    sol = solve_ivp(ric, (T, 0.0), Qta.ravel(), rtol=1e-11, atol=1e-13, dense_output=True, method="DOP853")
    if not sol.success:
        raise ReferenceError("Riccati integration failed")
    R_n = Gam @ Gam.T; Rni = np.linalg.inv(R_n); SG = Sig @ Gam.T
    N3 = 2 * n + 1
    Qz = np.zeros((n + k, n + k)); Qz[:] = Q

    def fwd(t, v):
        Pi = v[:n * n].reshape(n, n); S = v[n * n: n * n + N3 * N3].reshape(N3, N3)
        P = sol.sol(t).reshape(n + 1, n + 1)
        K = Ri @ (Ba.T @ P + Na.T); Kx, K1 = K[:, :n], K[:, n]
        L = (Pi @ H.T + SG) @ Rni
        dPi = A @ Pi + Pi @ A.T + Sig @ Sig.T - L @ R_n @ L.T
        F = np.zeros((N3, N3))
        F[:n, :n] = A; F[:n, n:2 * n] = -B @ Kx; F[:n, 2 * n] = c - B @ K1
        F[n:2 * n, :n] = L @ H; F[n:2 * n, n:2 * n] = A - B @ Kx - L @ H; F[n:2 * n, 2 * n] = c - B @ K1
        G = np.vstack([Sig, L @ Gam, np.zeros((1, Sig.shape[1]))])
        dS = F @ S + S @ F.T + G @ G.T
        Mz = np.zeros((n + k, N3)); Mz[:n, :n] = np.eye(n); Mz[n:, n:2 * n] = -Kx; Mz[n:, 2 * n] = -K1
        flow = float(np.trace(Mz.T @ Qz @ Mz @ S)) + float(q @ Mz @ S[:, 2 * n]) + m["const"]
        return np.concatenate([dPi.ravel(), dS.ravel(), [np.exp(-rho * t) * flow]])

    xi0 = np.concatenate([m["x0"], m["x0"], [1.0]])
    v0 = np.concatenate([np.zeros(n * n), np.outer(xi0, xi0).ravel(), [0.0]])
    out = solve_ivp(fwd, (0.0, T), v0, rtol=1e-11, atol=1e-13, method="DOP853")
    if not out.success:
        raise ReferenceError("forward integration failed")
    vT = out.y[:, -1]
    S = vT[n * n: n * n + N3 * N3].reshape(N3, N3)
    term = float(np.trace(m["Qt"] @ S[:n, :n]) + m["qt"] @ S[:n, 2 * n]) + m["tconst"]
    return {m["agent"]: float(vT[-1] + np.exp(-rho * T) * term)}
