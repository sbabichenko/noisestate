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
import sys, yaml
from noisestate import ModelBuilder

def build(N=3, tau=0.5, L=24.0, nodes=8, unit_range=8.0, **over):
    p = dict(theta=4.0, xi=0.15, zeta=0.5, kappa=0.3, m=1.0, r=0.2, rP=0.0, c=0.2, sigma_u=1.0,
             theta_a=0.5, sigma_a=1.0, theta_eta=0.5, sigma_eta=1.0, s1=2.5, s2=0.3, s3=0.3, s4=2.0, tau=tau)
    p.update(over)
    b = ModelBuilder("ch5_cycle_market", **p)
    b.channel("w_q")
    b.state("q", drift={}, noise={"w_q": "sigma_u"})
    for v in range(N):
        b.channel(f"w_a{v}", f"w_eta{v}", *[f"w_{v}_{r}" for r in range(4)])
        b.state(f"a{v}", drift={f"a{v}": "-theta_a"}, noise={f"w_a{v}": "sigma_a"})
        b.state(f"eta{v}", drift={f"eta{v}": "-theta_eta"}, noise={f"w_eta{v}": "sigma_eta"})
    b.define("Pidx", {f"P{v}@tau": 1.0 / N for v in range(N)})
    b.define("Pnext", {f"P{v}": 1.0 / N for v in range(N)})
    for v in range(N):
        sup, cus = (v - 1) % N, (v + 1) % N
        b.define(f"pi{v}", {f"P{v}@tau": 1.0})
        b.define(f"i{v}", {f"o{v}@tau": 1.0})
        b.define(f"d{v}", {f"o{cus}@tau": 1.0})
        b.define(f"yH{v}", {"q": 1.0, "Pidx": "theta - 1", f"pi{v}": "-theta", f"eta{v}": 1.0})
    for v in range(N):
        sup, cus = (v - 1) % N, (v + 1) % N
        b.define(f"dev{v}", {f"pi{v}": 1.0, "Pidx": "-(1 - xi - zeta)", "q": "-xi", f"pi{sup}": "-zeta", f"a{v}@tau": "zeta"})
        b.define(f"mis{v}", {f"a{v}@tau": 1.0, f"i{v}": 1.0, f"d{v}": -1.0, f"yH{v}": -1.0})
        b.define(f"bill{v}", {f"P{sup}": 1.0, "Pnext": -1.0})
        b.define(f"quote_gap{v}", {f"P{v}": 1.0, "Pnext": -1.0})
        loss = [[1.0, f"dev{v}", f"dev{v}"], ["-2*kappa", f"d{v}"], ["-2*kappa", f"yH{v}"], ["m", f"mis{v}", f"mis{v}"],
                ["r", f"o{v}", f"o{v}"], ["rP", f"quote_gap{v}", f"quote_gap{v}"], ["2*c", f"o{v}", f"bill{v}"]]
        b.agent(f"firm{v}", [f"P{v}", f"o{v}"], loss)
        b.signal(f"firm{v}", "sales", drift={f"yH{v}": 1.0}, noise={f"w_{v}_0": "s1"})
        b.signal(f"firm{v}", "trans_price", drift={f"P{sup}": 1.0}, noise={f"w_{v}_1": "s2"})
        b.signal(f"firm{v}", "order_book", drift={f"o{cus}": 1.0}, noise={f"w_{v}_2": "s3"})
        b.signal(f"firm{v}", "upstream_order", drift={f"o{sup}": 1.0}, noise={f"w_{v}_3": "s4"})
        b.signal(f"firm{v}", "own_prod", drift={}, noise={f"w_a{v}": 1.0})
    b.tie(*[f"firm{v}" for v in range(N)])
    b.stationary(discount=0.0, window=L, nodes=nodes, unit=tau, unit_range=unit_range)
    return b

if __name__ == "__main__":
    b = build()
    yaml.safe_dump(b.to_dict(), open(sys.argv[1] if len(sys.argv) > 1 else "examples/ch5_cycle_market.yaml", "w"), sort_keys=False)
    print("wrote", sys.argv[1] if len(sys.argv) > 1 else "examples/ch5_cycle_market.yaml")
