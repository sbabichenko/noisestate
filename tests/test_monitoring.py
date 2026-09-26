"""Monitored deviations (Chapter 6): the monitoring relation in the model, and the targets the engines must meet.

The relation: `monitors: [i]` on agent j is j |> i, j privy to deviations originating with i (Definition 6.1);
reflexive implicitly, transitive by Assumption 6.4 (refused otherwise).  No `monitors` anywhere is the all-naive
corner (Proposition 6.11 (ii)), what the engines have always solved.

The stationary engine solves monitored deviations (the spectral engine refuses them).  Each target is independent of
noisestate:

* The all-privy corner of the Chapter 3 tracking game (r = 1, discount 0).  A seed is known to every player and moves
  only conditional means, so the players' responses solve the deterministic game in the deviation, whose Markov
  perfect (feedback Nash) solution is the corner's coupled Riccati system (Proposition 6.11 (i)): the symmetric
  stationary gain K solves 1/2 + K^2 r / 2 - 2 K^2 r = 0, K = 1/sqrt(3 r) = 0.5774, so a unit state impulse from
  player 1 decays as e^{-2 K a} and each player's response to it is -K e^{-2 K a}.
* A privy market maker in the Chapter 4 Kyle-Back market: whatever the monitoring, the trader's equilibrium strategy
  is a best response and trading nothing is feasible at zero cost, so the trader's cost is not positive (the
  withdrawn naive_observers option reported +8.0 here)."""
import numpy as np, pytest
import noisestate as ns
from helpers import example

K_PRIVY = 1.0 / np.sqrt(3.0)


def ch3(monitors):
    d = example("ch3_two_player").to_dict()
    for agent, m in monitors.items():
        d["agents"][agent]["monitors"] = m
    return ns.Model.from_dict(d)


def test_the_relation_reads_writes_and_round_trips(tmp_path):
    m = ch3({"player1": ["player2"], "player2": ["player1"]})
    assert m.privy("player1") == ["player1", "player2"] and m.agents[0].monitors == ["player2"]
    path = str(tmp_path / "m.yaml"); m.save(path)
    assert ns.load(path).to_dict(numeric=True) == m.to_dict(numeric=True)
    assert ns.load(path).to_equations()["agents"]["player1"]["monitors"] == "player2"
    assert ch3({}).privy("player1") == ["player1"]                  # the default: nobody else is privy


def test_the_python_form_takes_agents_or_names():
    from noisestate import dt
    (dw0, dw1, dw2) = ns.shocks("w0 w1 w2")
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1 + D2) * dt + dw0
    p1 = ns.Agent("player1", D1, observes=X * dt + dw1, loss=X**2 + D1**2)
    p2 = ns.Agent("player2", D2, observes=X * dt + dw2, loss=X**2 + D2**2, monitors=p1)
    assert ns.Game(X, [p1, p2], window=3.0).privy("player1") == ["player1", "player2"]


def test_the_relation_is_checked():
    d = example("ch3_two_player").to_dict()
    with pytest.raises(ValueError, match="unknown agent"):
        ns.Model.from_dict({**d, "agents": {**d["agents"], "player1": {**d["agents"]["player1"], "monitors": ["nobody"]}}})
    with pytest.raises(ValueError, match="monitors itself"):
        ns.Model.from_dict({**d, "agents": {**d["agents"], "player1": {**d["agents"]["player1"], "monitors": ["player1"]}}})
    # three agents: k |> j |> i without k |> i is refused (a response seen without its origin)
    three = ns.load(ns.example("ch5_cycle_market")).to_dict()
    three["ties"] = []
    three["agents"]["firm2"]["monitors"] = ["firm1"]; three["agents"]["firm1"]["monitors"] = ["firm0"]
    with pytest.raises(ValueError, match="not transitive"):
        ns.Model.from_dict(three)
    three["agents"]["firm2"]["monitors"] = ["firm1", "firm0"]
    assert ns.Model.from_dict(three).privy("firm0") == ["firm0", "firm1", "firm2"]


def test_no_monitors_is_the_all_naive_corner_the_engines_solve():
    assert ns.solve(ch3({})).costs == ns.solve(example("ch3_two_player")).costs


def test_all_privy_tracking_responds_with_the_feedback_nash_gain():
    """Window 8 at 24 nodes: X and D2 to 1e-7 (measured 4e-9 and 5e-9; at the shipped window 3 the window truncation
    leaves 4e-3 at seed age 2)."""
    res = ns.solve(ch3({"player1": ["player2"], "player2": ["player1"]}).with_stationary(8.0).with_numerics(nodes=24))
    res.require_converged()
    a = np.linspace(0.0, 2.5, 11)
    r = res.deviation_response("player1", ["X", "D2"]).over(a)          # to a unit state impulse from player 1
    assert np.abs(r[:, 0] - np.exp(-2 * K_PRIVY * a)).max() < 1e-7
    assert np.abs(r[:, 1] + K_PRIVY * np.exp(-2 * K_PRIVY * a)).max() < 1e-7
    # the naive corner's response is the frozen one: player 2 filters the impulse, it does not see a deviation
    naive = ns.solve(ch3({}).with_stationary(8.0).with_numerics(nodes=24)).deviation_response("player1", ["D2"]).over(a)
    assert np.abs(naive[:, 0] - r[:, 1]).max() > 0.1


def test_a_privy_market_maker_leaves_the_trader_a_nonpositive_cost():
    """The market maker nets the trader's deviations out of the flow, so they move no price; the trader's cost is
    -0.4999997 (the withdrawn naive_observers gave +8.0, the on-path world built with the deviation's responses)."""
    kb = ns.load(ns.example("ch4_kyle_back")).to_dict()
    kb["agents"]["market_maker"]["monitors"] = "trader1"
    res = ns.solve(kb).require_converged()
    assert res.costs["trader1"] <= 1e-8
    assert res.costs["trader1"] == pytest.approx(-0.4999997, abs=1e-6)
    assert all(res.second_order[a]["ok"] for a in res.second_order)


#  ------------------------------------------------------------------ instant observations: a quote seen at once
#  Chapter 6's market: a strategic market maker quotes P and carries the inventory Q it absorbs; the trader sees the
#  quote's level and trades on it within the instant (`observes: {quote: {level: P}}`), and in the transparent market
#  also sees its deviations (monitors the market maker).  The market maker's loss has no P^2: its curvature is the
#  trader's instant reaction, h = -1/(2 eps), the G^{MM,0} = 2 D^{1<-0,M} of the chapter.

def market(gamma, transparent=True, nodes=16):
    tr = {"controls": "D", "observes": {"y": "(V - P) dt + dwY", "flow": "sigma_Z dwZ", "quote": {"level": "P"}},
          "loss": "-V D + P D + eps D^2"}
    if transparent:
        tr["monitors"] = "mm"
    return ns.Model.from_dict({
        "name": "market", "params": {"eps": 0.2, "gamma": gamma, "rho": 0.5, "sigma_Z": 1.0},
        "shocks": ["wV", "wZ", "wY"], "states": {"V": "dwV", "Q": "-D dt - sigma_Z dwZ"},
        "agents": {"mm": {"controls": "P", "observes": {"flow": "D dt + sigma_Z dwZ"}, "loss": "V D - P D + gamma Q^2"},
                   "trader": tr},
        "horizon": {"window": 8.0, "discount": "rho"}, "numerics": {"nodes": nodes}})


def test_an_instant_observation_is_compiled_from_the_observers_loss():
    from noisestate.compile import compile_structure
    st = compile_structure(market(0.1))
    assert st.instant_loads == {"D": {"P": pytest.approx(-2.5)}}             # -1/(2 eps)
    assert st.composite["P"] == {"P": 1.0, "D": pytest.approx(-2.5)}        # a quote spike draws an order spike
    assert market(0.1).to_equations()["agents"]["trader"]["observes"]["P"] == {"level": "P"}


def test_instant_reactions_must_not_cycle():
    d = market(0.1).to_dict()
    d["agents"]["mm"]["instant"] = ["D"]
    with pytest.raises(ValueError, match="cycle"):
        ns.Model.from_dict(d)


@pytest.mark.parametrize("transparent", [True, False])
def test_with_no_inventory_cost_the_strategic_market_maker_is_competitive(transparent):
    """gamma = 0: the market maker's first-order condition returns the competitive quote E[V | flow] (Chapter 6), so
    the market is Chapter 4's (examples/ch4_kyle_back.yaml at the same eps and rho): the trader's cost -0.82082 and the
    price's kernel on the noise trades to 1e-6.  Counting the trader's reaction to the quote twice -- in its map, where
    the flow already carries the quote, and again as the instant loading -- gave 0.58 for the price's first response."""
    kb = ns.solve(ns.load(ns.example("ch4_kyle_back")).with_numerics(nodes=16))
    res = ns.solve(market(0.0, transparent)).require_converged()
    assert res.costs["trader"] == pytest.approx(kb.costs["trader1"], abs=1e-6)
    a = np.array([0.0, 1.0, 3.0])
    assert np.abs(res.kernel("P", "wZ").at(a) - kb.kernel("P", "wZ").at(a)).max() < 1e-6


def test_the_maps_hold_the_market_but_do_not_reach_it():
    """The market's equilibrium is unstable under best responses (radius 1.48).  The action kernels (the default) reach
    it from zero; the raw maps hold it (a few evaluations from it, the same costs) but from zero stall at 0.85 and say
    so -- not a second equilibrium: converged is False.  Ties, which need the maps, are refused with instant
    observations for that reason."""
    m = market(0.1, nodes=24)
    res = ns.solve(m).require_converged()
    held = ns.engines.solver(m).solve(start_from=res.maps, variable="maps")
    assert held.converged and held.evaluations <= 5
    assert held.costs["trader"] == pytest.approx(res.costs["trader"], abs=1e-8)
    assert not ns.engines.solver(m).solve(variable="maps").converged


#  ------------------------------------------------------------------ the finite horizon (the spectral engine)

def test_all_privy_tracking_on_a_finite_horizon_is_the_finite_feedback_nash_game():
    """On [0, 2] the privy responses follow the finite feedback Nash Riccati P' = -1 + 3 P^2 / r, P(T) = 0: a unit
    impulse from player 1 at s decays as exp(-2 int_s^t P / r), player 2 answering -P(t) / r times it.  At 12 nodes to
    2e-5 (measured 3.2e-6 for X, 2.1e-7 for D2; 3e-4 at 8 nodes, 9e-7 / 7e-6 at 10)."""
    from scipy.integrate import solve_ivp
    T, r = 2.0, 1.0
    Pb = solve_ivp(lambda t, y: [-1.0 + 3.0 * y[0] ** 2 / r], (T, 0.0), [0.0], dense_output=True, rtol=1e-12, atol=1e-14)
    P = lambda t: float(Pb.sol(t)[0])

    def exact(s, t):
        x = solve_ivp(lambda u, y: [-2 * P(u) / r * y[0]], (s, t), [1.0], rtol=1e-12, atol=1e-14).y[0, -1]
        return x, -P(t) / r * x
    m = ch3({"player1": ["player2"], "player2": ["player1"]})
    res = ns.solve(m.with_finite(T).with_numerics(nodes=12)).require_converged()
    pts = [(0.0, 0.5), (0.0, 1.5), (0.5, 1.0), (1.0, 1.9)]
    got = res.deviation_response("player1", ["X", "D2"]).over(np.array([t for s, t in pts]), np.array([s for s, t in pts]))
    assert np.abs(got - np.array([exact(s, t) for s, t in pts])).max() < 2e-5


def finite_market(gamma, transparent, nodes=10):
    tr = {"controls": "D", "observes": {"y": "(V - P) dt + dwY", "flow": "dwZ", "quote": {"level": "P"}},
          "loss": "-V D + P D + eps D^2"}
    if transparent:
        tr["monitors"] = "mm"
    return ns.Model.from_dict({"name": "market", "params": {"eps": 0.5, "gamma": gamma}, "shocks": ["wV", "wZ", "wY"],
        "states": {"V": "dwV", "Q": "-D dt - dwZ"},
        "agents": {"mm": {"controls": "P", "observes": {"flow": "D dt + dwZ"}, "loss": "V D - P D + gamma Q^2"}, "trader": tr},
        "horizon": {"T": 1.0}, "numerics": {"nodes": nodes}})


@pytest.mark.parametrize("transparent", [True, False])
def test_on_a_finite_horizon_the_strategic_market_maker_without_inventory_cost_is_competitive(transparent):
    """gamma = 0 on [0, 1]: the same market with the competitive (myopic, P = E[V | flow]) market maker; the trader's
    cost to 1e-9 (measured 1.3e-11) and the price's kernel on the noise trades to 1e-7 (measured 3.1e-8)."""
    comp = ns.Model.from_dict({"name": "competitive", "params": {"eps": 0.5}, "shocks": ["wV", "wZ", "wY"],
        "states": {"V": "dwV", "Q": "-D dt - dwZ"},
        "agents": {"mm": {"controls": "P", "observes": {"flow": "D dt + dwZ"}, "loss": "P^2 - 2 P V", "myopic": True},
                   "trader": {"controls": "D", "observes": {"y": "(V - P) dt + dwY", "flow": "dwZ"}, "loss": "-V D + P D + eps D^2"}},
        "horizon": {"T": 1.0}, "numerics": {"nodes": 10}})
    rc, r = ns.solve(comp), ns.solve(finite_market(0.0, transparent))
    assert r.converged and abs(r.costs["trader"] - rc.costs["trader"]) < 1e-9
    t = np.array([0.3, 0.6, 0.9]); s = np.array([0.1, 0.2, 0.5])
    assert np.abs(r.evaluate("P", "wZ", t, s) - rc.evaluate("P", "wZ", t, s)).max() < 1e-7


def test_monitoring_with_a_past_is_refused_on_the_finite_engine():
    d = ch3({"player1": ["player2"]}).to_dict()
    with pytest.raises(NotImplementedError, match="without a past"):
        ns.solve(ns.Model.from_dict(d).with_finite(2.0), past=ns.solve(ch3({})), continuation="stationary")
