"""Generate the Chapter 5 purchase-order market on a directed cycle of N firms (the dissertation's
Chapter 5 model; the defaults below are the calibration of its `spectral_market` solver, whose sweep
point `tests/refs/ch5_s1_2.5_u16_endo.txt` is what `tests/test_ch5.py` checks the package against).

Firm v quotes P_v and orders o_v (input from supplier v-1, delivered after tau).
Delayed rows: pi_v = P_v@tau (price in force), i_v = o_v@tau (input arriving),
d_v = o_{v+1}@tau (deliveries owed).  Index Pidx = mean pi_v; household demand
yH_v = q - Pidx - theta (pi_v - Pidx) + eta_v.  Flow loss of firm v:
  (pi_v - pistar_v)^2 - 2 kappa (d_v + yH_v) + m (a_v@tau + i_v - d_v - yH_v)^2
  + r o_v^2 + rP (P_v - Pnext)^2 + 2 c o_v (P_{v-1} - Pnext),
  pistar_v = Pidx + xi (q - Pidx) + zeta (pi_{v-1} - Pidx - a_v@tau),  Pnext = mean P_v.
Signals of firm v (noise s1..s4): yH_v, P_{v-1}, o_{v+1}, o_{v-1}; own productivity a_v seen exactly.
"""
import sys
import noisestate as ns
from noisestate import Param, State, Control, Agent, define, shocks, dt


def build(N=3, tau=0.5, L=24.0, nodes=8, unit_range=8.0, **over):
    values = dict(theta=4.0, xi=0.15, zeta=0.5, kappa=0.3, m=1.0, r=0.2, rP=0.0, c=0.2, sigma_u=1.0,
                  theta_a=0.5, sigma_a=1.0, theta_eta=0.5, sigma_eta=1.0, s1=2.5, s2=0.3, s3=0.3, s4=2.0, tau=tau)
    values.update(over)
    p = {k: Param(k, v) for k, v in values.items()}
    theta, xi, zeta, kappa, m, r, rP, c, tau_ = (p[k] for k in ("theta", "xi", "zeta", "kappa", "m", "r", "rP", "c", "tau"))

    # the shocks, in the order the model file lists them: demand, then each firm's own shocks
    wq = shocks("w_q")
    w = [shocks(f"w_a{v}", f"w_eta{v}", *[f"w_{v}_{k}" for k in range(4)]) for v in range(N)]

    q = State("q"); q.d = p["sigma_u"] * wq.w_q                          # the common demand level
    a = [State(f"a{v}") for v in range(N)]; eta = [State(f"eta{v}") for v in range(N)]
    for v in range(N):
        a[v].d = -p["theta_a"] * a[v] * dt + p["sigma_a"] * w[v][f"w_a{v}"]          # productivity
        eta[v].d = -p["theta_eta"] * eta[v] * dt + p["sigma_eta"] * w[v][f"w_eta{v}"]  # a demand shock
    P = [Control(f"P{v}") for v in range(N)]; o = [Control(f"o{v}") for v in range(N)]

    Pidx = define("Pidx", sum((P[v].lag(tau_) * (1.0 / N) for v in range(N)), 0))
    Pnext = define("Pnext", sum((P[v] * (1.0 / N) for v in range(N)), 0))
    pi = [define(f"pi{v}", P[v].lag(tau_)) for v in range(N)]            # the price in force
    i_ = [define(f"i{v}", o[v].lag(tau_)) for v in range(N)]             # input arriving
    d_ = [define(f"d{v}", o[(v + 1) % N].lag(tau_)) for v in range(N)]   # deliveries owed
    yH = [define(f"yH{v}", q + (theta - 1) * Pidx - theta * pi[v] + eta[v]) for v in range(N)]

    firms = []
    for v in range(N):
        sup, cus = (v - 1) % N, (v + 1) % N
        dev = define(f"dev{v}", pi[v] - (1 - xi - zeta) * Pidx - xi * q - zeta * pi[sup] + zeta * a[v].lag(tau_))
        mis = define(f"mis{v}", a[v].lag(tau_) + i_[v] - d_[v] - yH[v])
        bill = define(f"bill{v}", P[sup] - Pnext)
        gap = define(f"quote_gap{v}", P[v] - Pnext)
        loss = dev**2 - 2 * kappa * d_[v] - 2 * kappa * yH[v] + m * mis**2 + r * o[v]**2 + rP * gap**2 + 2 * c * o[v] * bill
        wv = w[v]
        firms.append(Agent(f"firm{v}", controls=[P[v], o[v]], loss=loss, observes={
            "sales": yH[v] * dt + p["s1"] * wv[f"w_{v}_0"],
            "trans_price": P[sup] * dt + p["s2"] * wv[f"w_{v}_1"],
            "order_book": o[cus] * dt + p["s3"] * wv[f"w_{v}_2"],
            "upstream_order": o[sup] * dt + p["s4"] * wv[f"w_{v}_3"],
            "own_prod": wv[f"w_a{v}"]}))
    return ns.Game([q] + [x for v in range(N) for x in (a[v], eta[v])], firms, name="ch5_cycle_market",
                    ties=[firms], horizon=ns.Stationary(window=L, discount=0.0), params=p.values(),
                    numerics={"nodes": nodes, "unit": tau, "unit_range": unit_range})

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/ch5_cycle_market.yaml"
    build().save(path)
    print("wrote", path)
