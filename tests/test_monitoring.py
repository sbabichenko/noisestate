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
