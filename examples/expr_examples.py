"""The shipped examples written as equations (noisestate.expr), one function each; every function's model
compiles to the dict of the YAML file beside it (tests/test_expr.py asserts it), so the two spellings are the
same model and the same numbers.  Read them side by side with the YAML files.

    python examples/expr_examples.py            # solves ch3 and prints its summary
"""
import os

import noisestate as ns
from noisestate import Param, State, Control, Signal, Agent, shocks, define, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))


def ch1_two_player_finite() -> ns.Model:
    """Chapter 1: two players track a common state on [0, 1] (examples/ch1_two_player_finite.yaml)."""
    p1, p2, r1, r2, sigma = Param.many(p1=3.0, p2=3.0, r1=0.1, r2=0.1, sigma=1.0)
    w = shocks("w0", "w1", "w2")
    X = State("X"); D1, D2 = Control("D1"), Control("D2")
    X.drift = D1 + D2 + sigma * w.w0
    player1 = Agent("player1", controls=[D1], signals=[Signal("y1", sqrt(p1) * X + w.w1)], loss=X**2 + r1 * D1**2)
    player2 = Agent("player2", controls=[D2], signals=[Signal("y2", sqrt(p2) * X + w.w2)], loss=X**2 + r2 * D2**2)
    return ns.Model("ch1_two_player_finite", states=[X], agents=[player1, player2], horizon=ns.Finite(T=1.0),
                    numerics=ns.Numerics(nodes=12))


def ch1_delayed_finite() -> ns.Model:
    """The same game with the controls acting after tau and player 2's signal delayed by tau
    (examples/ch1_delayed_finite.yaml): a lag is `D1.lag(tau)`, a delay the Signal's `delay=`."""
    p1, p2, r1, r2, sigma, tau = Param.many(p1=3.0, p2=3.0, r1=0.1, r2=0.1, sigma=1.0, tau=0.25)
    w = shocks("w0", "w1", "w2")
    X = State("X"); D1, D2 = Control("D1"), Control("D2")
    X.drift = D1.lag(tau) + D2.lag(tau) + sigma * w.w0
    player1 = Agent("player1", controls=[D1], signals=[Signal("y1", sqrt(p1) * X + w.w1)], loss=X**2 + r1 * D1**2)
    player2 = Agent("player2", controls=[D2], signals=[Signal("y2", sqrt(p2) * X + w.w2, delay=tau)], loss=X**2 + r2 * D2**2)
    return ns.Model("ch1_delayed_finite", states=[X], agents=[player1, player2], horizon=ns.Finite(T=1.0),
                    numerics=ns.Numerics(nodes=8))


def ch3_two_player() -> ns.Model:
    """Chapter 3: the stationary tracking game at average cost (examples/ch3_two_player.yaml)."""
    p1, p2, r1, r2 = Param.many(p1=3.0, p2=10.0, r1=1.0, r2=1.0)
    w = shocks("w0", "w1", "w2")
    X = State("X"); D1, D2 = Control("D1"), Control("D2")
    X.drift = D1 + D2 + w.w0
    player1 = Agent("player1", controls=[D1], signals=[Signal("y1", sqrt(p1) * X + w.w1)], loss=0.5 * X**2 + 0.5 * r1 * D1**2)
    player2 = Agent("player2", controls=[D2], signals=[Signal("y2", sqrt(p2) * X + w.w2)], loss=0.5 * X**2 + 0.5 * r2 * D2**2)
    return ns.Model("ch3_two_player", states=[X], agents=[player1, player2], horizon=ns.Stationary(window=3.0, discount=0.0),
                    numerics=ns.Numerics(nodes=24))


def ch4_kyle_back() -> ns.Model:
    """Chapter 4: the Kyle-Back market (examples/ch4_kyle_back.yaml).  The market maker's loss is written
    as the file writes it, P^2 - 2 P V (the V^2 of (P - V)^2 moves nothing it chooses and is left out)."""
    eps, rho, gamma1, sigma_V, sigma_Z = Param.many(eps=0.2, rho=0.5, gamma1=1.0, sigma_V=1.0, sigma_Z=1.0)
    w = shocks("wV", "wZ", "w1")
    V = State("V"); P, D1 = Control("P"), Control("D1")
    V.drift = sigma_V * w.wV
    market_maker = Agent("market_maker", controls=[P], myopic=True,
                         signals=[Signal("flow", D1 + sigma_Z * w.wZ)], loss=P**2 - 2 * P * V)
    trader1 = Agent("trader1", controls=[D1],
                    signals=[Signal("y1", gamma1 * V - gamma1 * P + w.w1), Signal("flow", sigma_Z * w.wZ)],
                    loss=-D1 * V + D1 * P + eps * D1**2)
    return ns.Model("ch4_kyle_back", states=[V], agents=[market_maker, trader1], horizon=ns.Stationary(window=8.0, discount=rho),
                    numerics=ns.Numerics(nodes=24))


def ch5_cycle_market(N: int = 3) -> ns.Model:
    """Chapter 5: the purchase-order market on a directed cycle of N firms (examples/ch5_cycle_market.yaml,
    examples/make_ch5_cycle_market.py): the definitions are define()d quantities, the firms are tied."""
    theta, xi, zeta, kappa, m, r, rP, c = Param.many(theta=4.0, xi=0.15, zeta=0.5, kappa=0.3, m=1.0, r=0.2, rP=0.0, c=0.2)
    sigma_u, theta_a, sigma_a, theta_eta, sigma_eta = Param.many(sigma_u=1.0, theta_a=0.5, sigma_a=1.0, theta_eta=0.5, sigma_eta=1.0)
    s1, s2, s3, s4, tau = Param.many(s1=2.5, s2=0.3, s3=0.3, s4=2.0, tau=0.5)
    w = shocks("w_q", *[n for v in range(N) for n in (f"w_a{v}", f"w_eta{v}", *[f"w_{v}_{k}" for k in range(4)])])
    q = State("q"); q.drift = sigma_u * w.w_q
    a, eta, P, o = [], [], [], []
    for v in range(N):
        a.append(State(f"a{v}")); a[v].drift = -theta_a * a[v] + sigma_a * w[f"w_a{v}"]
        eta.append(State(f"eta{v}")); eta[v].drift = -theta_eta * eta[v] + sigma_eta * w[f"w_eta{v}"]
        P.append(Control(f"P{v}")); o.append(Control(f"o{v}"))
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
        firms.append(Agent(f"firm{v}", controls=[P[v], o[v]], loss=loss, signals=[
            Signal("sales", yH[v] + s1 * w[f"w_{v}_0"]), Signal("trans_price", P[sup] + s2 * w[f"w_{v}_1"]),
            Signal("order_book", o[cus] + s3 * w[f"w_{v}_2"]), Signal("upstream_order", o[sup] + s4 * w[f"w_{v}_3"]),
            Signal("own_prod", w[f"w_a{v}"])]))
    return ns.Model("ch5_cycle_market", states=[q] + [x for v in range(N) for x in (a[v], eta[v])], agents=firms,
                    definitions=defs, ties=[firms], horizon=ns.Stationary(window=24.0, discount=0.0),
                    numerics=ns.Numerics(nodes=8, unit=0.5, unit_range=8.0))


def ch3_precision_change(past="ch3_two_player.yaml") -> ns.Model:
    """Chapter 3, a change of regime (examples/ch3_precision_change.yaml): player 1's precision rises from 3 to
    10 at time zero; `past` is the old model (the file's path, relative to examples/, or a Model)."""
    p1, p2, r1, r2 = Param.many(p1=10.0, p2=10.0, r1=1.0, r2=1.0)
    w = shocks("w0", "w1", "w2")
    X = State("X"); D1, D2 = Control("D1"), Control("D2")
    X.drift = D1 + D2 + w.w0
    player1 = Agent("player1", controls=[D1], signals=[Signal("y1", sqrt(p1) * X + w.w1)], loss=0.5 * X**2 + 0.5 * r1 * D1**2)
    player2 = Agent("player2", controls=[D2], signals=[Signal("y2", sqrt(p2) * X + w.w2)], loss=0.5 * X**2 + 0.5 * r2 * D2**2)
    return ns.Model("ch3_precision_change", states=[X], agents=[player1, player2],
                    horizon=ns.Transition(T=6.0, past=past, continuation="stationary", discount=0.0),
                    numerics=ns.Numerics(nodes=12, continuation_nodes=12))


def kyle_back_prior() -> ns.Model:
    """The Kyle-Back market started from a prior (examples/kyle_back_prior.yaml): the past is one initial shock,
    V ~ N(0, Sigma0) seen by the trader at once, and the game ends at T."""
    eps, Sigma0, sigma_Z = Param.many(eps=0.1, Sigma0=1.0, sigma_Z=1.0)
    w = shocks("wZ")
    V = State("V"); P, D1 = Control("P"), Control("D1")                     # V is constant: no drift, no noise
    market_maker = Agent("market_maker", controls=[P], myopic=True, signals=[Signal("flow", D1 + sigma_Z * w.wZ)],
                         loss=P**2 - 2 * P * V)
    trader1 = Agent("trader1", controls=[D1], signals=[Signal("flow", sigma_Z * w.wZ)], loss=-D1 * V + D1 * P + eps * D1**2)
    prior = [{"name": "v0", "loads": {"V": sqrt(Sigma0)}, "rows": {"trader1.flow": 1.0}}]
    return ns.Model("kyle_back_prior", states=[V], agents=[market_maker, trader1],
                    horizon=ns.Transition(T=2.0, past=prior, continuation="end", discount=0.0), numerics=ns.Numerics(nodes=12))


EXAMPLES = {"ch1_two_player_finite": ch1_two_player_finite, "ch1_delayed_finite": ch1_delayed_finite, "ch3_two_player": ch3_two_player,
            "ch4_kyle_back": ch4_kyle_back, "ch5_cycle_market": ch5_cycle_market, "ch3_precision_change": ch3_precision_change,
            "kyle_back_prior": kyle_back_prior}


if __name__ == "__main__":
    print(ch3_two_player().solve().summary())
