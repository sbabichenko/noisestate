"""The shipped examples written as equations in Python, one function each; every function's model compiles to
the model of the YAML file beside it (tests/test_expr.py asserts it), so the two spellings are the same model
and the same numbers.  Read them side by side with the YAML files.

    python examples/expr_examples.py            # solves ch3 and prints its summary
"""
import os

import noisestate as ns
from noisestate import Signal, define, dt, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))


def ch1_two_player_finite() -> ns.Model:
    """Chapter 1: two players track a common state on [0, 1] (examples/ch1_two_player_finite.yaml)."""
    p1, p2, r1, r2, sigma = ns.params(p1=3.0, p2=3.0, r1=0.1, r2=0.1, sigma=1.0)
    dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1 + D2) * dt + sigma * dw0
    player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1}, loss=X**2 + r1 * D1**2)
    player2 = ns.Agent("player2", controls=D2, observes={"y2": sqrt(p2) * X * dt + dw2}, loss=X**2 + r2 * D2**2)
    return ns.Game(X, [player1, player2], T=1.0, name="ch1_two_player_finite", nodes=12)


def ch1_delayed_finite() -> ns.Model:
    """The same game with the controls acting after tau and player 2's signal delayed by tau
    (examples/ch1_delayed_finite.yaml): a lag is `D1.lag(tau)`, a delay the Signal's `delay=`."""
    p1, p2, r1, r2, sigma, tau = ns.params(p1=3.0, p2=3.0, r1=0.1, r2=0.1, sigma=1.0, tau=0.25)
    dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1.lag(tau) + D2.lag(tau)) * dt + sigma * dw0
    player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1}, loss=X**2 + r1 * D1**2)
    player2 = ns.Agent("player2", controls=D2, observes=Signal("y2", sqrt(p2) * X * dt + dw2, delay=tau),
                       loss=X**2 + r2 * D2**2)
    return ns.Game(X, [player1, player2], T=1.0, name="ch1_delayed_finite", nodes=8)


def ch3_two_player() -> ns.Model:
    """Chapter 3: the stationary tracking game at average cost (examples/ch3_two_player.yaml)."""
    p1, p2, r1, r2 = ns.params(p1=3.0, p2=10.0, r1=1.0, r2=1.0)
    dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1 + D2) * dt + dw0
    player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1}, loss=0.5 * X**2 + 0.5 * r1 * D1**2)
    player2 = ns.Agent("player2", controls=D2, observes={"y2": sqrt(p2) * X * dt + dw2}, loss=0.5 * X**2 + 0.5 * r2 * D2**2)
    return ns.Game(X, [player1, player2], window=3.0, name="ch3_two_player", nodes=24)


def ch4_kyle_back() -> ns.Model:
    """Chapter 4: the Kyle-Back market (examples/ch4_kyle_back.yaml).  The market maker's loss is written
    as the file writes it, P^2 - 2 P V (the V^2 of (P - V)^2 moves nothing it chooses and is left out)."""
    eps, rho, gamma1, sigma_V, sigma_Z = ns.params(eps=0.2, rho=0.5, gamma1=1.0, sigma_V=1.0, sigma_Z=1.0)
    dwV, dwZ, dw1 = ns.shocks("wV", "wZ", "w1")
    V = ns.State("V"); P, D1 = ns.Control("P"), ns.Control("D1")
    V.d = sigma_V * dwV
    market_maker = ns.Agent("market_maker", controls=P, myopic=True,
                            observes={"flow": D1 * dt + sigma_Z * dwZ}, loss=P**2 - 2 * P * V)
    trader1 = ns.Agent("trader1", controls=D1,
                       observes={"y1": (gamma1 * V - gamma1 * P) * dt + dw1, "flow": sigma_Z * dwZ},
                       loss=-D1 * V + D1 * P + eps * D1**2)
    return ns.Game(V, [market_maker, trader1], window=8.0, discount=rho, name="ch4_kyle_back", nodes=24)


def ch5_cycle_market(N: int = 3) -> ns.Model:
    """Chapter 5: the purchase-order market on a directed cycle of N firms (examples/ch5_cycle_market.yaml,
    examples/make_ch5_cycle_market.py): the definitions are define()d quantities, the firms are tied."""
    theta, xi, zeta, kappa, m, r, rP, c = ns.params(theta=4.0, xi=0.15, zeta=0.5, kappa=0.3, m=1.0, r=0.2, rP=0.0, c=0.2)
    sigma_u, theta_a, sigma_a, theta_eta, sigma_eta = ns.params(sigma_u=1.0, theta_a=0.5, sigma_a=1.0, theta_eta=0.5, sigma_eta=1.0)
    s1, s2, s3, s4, tau = ns.params(s1=2.5, s2=0.3, s3=0.3, s4=2.0, tau=0.5)
    dw = ns.shocks("w_q", *[n for v in range(N) for n in (f"w_a{v}", f"w_eta{v}", *[f"w_{v}_{k}" for k in range(4)])])
    q = ns.State("q"); q.d = sigma_u * dw.w_q
    a, eta, P, o = [], [], [], []
    for v in range(N):
        a.append(ns.State(f"a{v}")); a[v].d = -theta_a * a[v] * dt + sigma_a * dw[f"w_a{v}"]
        eta.append(ns.State(f"eta{v}")); eta[v].d = -theta_eta * eta[v] * dt + sigma_eta * dw[f"w_eta{v}"]
        P.append(ns.Control(f"P{v}")); o.append(ns.Control(f"o{v}"))
    Pidx = define("Pidx", sum(P[v].lag(tau) for v in range(N)) / N)          # the price index in force
    Pnext = define("Pnext", sum(P[v] for v in range(N)) / N)                 # the quotes' mean
    defs = [Pidx, Pnext]; pi, i, d, yH = [], [], [], []
    for v in range(N):
        cus = (v + 1) % N
        pi.append(define(f"pi{v}", P[v].lag(tau)))                           # the price in force
        i.append(define(f"i{v}", o[v].lag(tau)))                             # the input arriving
        d.append(define(f"d{v}", o[cus].lag(tau)))                           # the deliveries owed
        yH.append(define(f"yH{v}", q + (theta - 1) * Pidx - theta * pi[v] + eta[v]))     # household demand
        defs += [pi[v], i[v], d[v], yH[v]]
    firms = []
    for v in range(N):
        sup, cus = (v - 1) % N, (v + 1) % N
        dev = define(f"dev{v}", pi[v] - (1 - xi - zeta) * Pidx - xi * q - zeta * pi[sup] + zeta * a[v].lag(tau))
        mis = define(f"mis{v}", a[v].lag(tau) + i[v] - d[v] - yH[v])
        bill = define(f"bill{v}", P[sup] - Pnext)
        quote_gap = define(f"quote_gap{v}", P[v] - Pnext)
        defs += [dev, mis, bill, quote_gap]
        loss = (dev**2 - 2 * kappa * d[v] - 2 * kappa * yH[v] + m * mis**2 + r * o[v]**2 + rP * quote_gap**2
                + 2 * c * o[v] * bill)
        firms.append(ns.Agent(f"firm{v}", controls=[P[v], o[v]], loss=loss, observes={
            "sales": yH[v] * dt + s1 * dw[f"w_{v}_0"], "trans_price": P[sup] * dt + s2 * dw[f"w_{v}_1"],
            "order_book": o[cus] * dt + s3 * dw[f"w_{v}_2"], "upstream_order": o[sup] * dt + s4 * dw[f"w_{v}_3"],
            "own_prod": dw[f"w_a{v}"]}))
    return ns.Game([q] + [x for v in range(N) for x in (a[v], eta[v])], firms, window=24.0, definitions=defs,
                   ties=[firms], name="ch5_cycle_market", numerics={"nodes": 8, "unit": 0.5, "unit_range": 8.0})


def ch3_precision_change(past="ch3_two_player.yaml") -> ns.Model:
    """Chapter 3, a change of regime (examples/ch3_precision_change.yaml): player 1's precision rises from 3 to
    10 at time zero; `past` is the old model (the file's path, relative to examples/, or a Model)."""
    p1, p2, r1, r2 = ns.params(p1=10.0, p2=10.0, r1=1.0, r2=1.0)
    dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1 + D2) * dt + dw0
    player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1}, loss=0.5 * X**2 + 0.5 * r1 * D1**2)
    player2 = ns.Agent("player2", controls=D2, observes={"y2": sqrt(p2) * X * dt + dw2}, loss=0.5 * X**2 + 0.5 * r2 * D2**2)
    return ns.Game(X, [player1, player2], horizon=ns.Transition(T=6.0, past=past), name="ch3_precision_change",
                   numerics={"nodes": 12, "continuation_nodes": 12})


def kyle_back_prior() -> ns.Model:
    """The Kyle-Back market started from a prior (examples/kyle_back_prior.yaml): the past is one initial shock,
    V ~ N(0, Sigma0) seen by the trader at once, and the game ends at T."""
    eps, Sigma0, sigma_Z = ns.params(eps=0.1, Sigma0=1.0, sigma_Z=1.0)
    (dwZ,) = ns.shocks("wZ")
    V = ns.State("V"); P, D1 = ns.Control("P"), ns.Control("D1")             # V is constant: no drift, no noise
    market_maker = ns.Agent("market_maker", controls=P, myopic=True, observes={"flow": D1 * dt + sigma_Z * dwZ},
                            loss=P**2 - 2 * P * V)
    trader1 = ns.Agent("trader1", controls=D1, observes={"flow": sigma_Z * dwZ}, loss=-D1 * V + D1 * P + eps * D1**2)
    prior = [{"name": "v0", "loads": {"V": sqrt(Sigma0)}, "rows": {"trader1.flow": 1.0}}]
    return ns.Game(V, [market_maker, trader1], horizon=ns.Transition(T=2.0, past=prior, continuation="end"),
                   name="kyle_back_prior", nodes=12)


EXAMPLES = {"ch1_two_player_finite": ch1_two_player_finite, "ch1_delayed_finite": ch1_delayed_finite, "ch3_two_player": ch3_two_player,
            "ch4_kyle_back": ch4_kyle_back, "ch5_cycle_market": ch5_cycle_market, "ch3_precision_change": ch3_precision_change,
            "kyle_back_prior": kyle_back_prior}


if __name__ == "__main__":
    print(ch3_two_player().solve().summary())
