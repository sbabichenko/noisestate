"""res.estimate(agent, name) and res.strategy(control) on the Chapter 1 tracking game (finite, spectral engine)."""
import numpy as np
import pytest

import noisestate as ns

CH1 = {"name": "ch1", "params": {"p1": 3, "p2": 10, "r1": 0.1, "r2": 0.1, "b1": 1, "b2": -1, "sigma": 1, "T": 1},
       "shocks": ["w0", "w1", "w2"], "states": {"X": {"drift": {"D1": 1, "D2": 1}, "noise": {"w0": "sigma"}}},
       "agents": {"player1": {"controls": ["D1"], "signals": {"y1": {"drift": {"X": "sqrt(p1)"}, "noise": {"w1": 1}}},
                              "loss": [[1, "X", "X"], ["-2*b1", "X"], ["r1", "D1", "D1"]]},
                  "player2": {"controls": ["D2"], "signals": {"y2": {"drift": {"X": "sqrt(p2)"}, "noise": {"w2": 1}}},
                              "loss": [[1, "X", "X"], ["-2*b2", "X"], ["r2", "D2", "D2"]]}},
       "horizon": {"kind": "finite", "T": "T"}, "numerics": {"nodes": 16}}


@pytest.fixture(scope="module")
def res():
    return ns.solve(CH1, start_policy="coarse")


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(1e-12, np.abs(np.asarray(b)).max()))


def test_own_control_is_known(res):
    # an agent's estimate of its own action is the action
    assert rel(res.estimate("player1", "D1"), res.kernel("D1")) < 1e-6


def test_estimate_of_the_state(res):
    t = np.array([1.0]); s = np.zeros(1)
    X = float(res.evaluate("X", "w0", t, s)[0])
    x1 = float(res.estimate("player1", "X").at(t, s)[0, 0]); x2 = float(res.estimate("player2", "X").at(t, s)[0, 0])
    # the better-informed player 2 has learned more of the shock; neither knows all of it
    assert 0 < x1 < x2 < X
    # the finite-difference solver (Richardson 2 f(157) - f(79)) gives 0.140 and 0.258 against X = 0.324
    assert abs(x1 - 0.140) < 0.005 and abs(x2 - 0.258) < 0.005


def test_estimate_error_matches_belief_error(res):
    # the variance of X - E[X | agent] over the shocks alive is what belief_error reports, for the transition result;
    # here: the error kernel is not zero, and the estimate is a projection (projecting it again changes nothing)
    E = res.estimate("player1", "X")
    assert rel(res._project("player1", np.asarray(E)), E) < 1e-6


def test_strategy_projects_back_to_the_control(res):
    # the action is the agent's estimate of its strategy's process: projecting the strategy kernel gives D_W
    D = np.asarray(res.strategy("D1"))
    # up to the grid's error on the diagonal s = t in the agent's own noise, which shrinks with the nodes
    # (0.85% at 16 nodes here, 0.1% at 28)
    assert rel(res._project("player1", D), res.kernel("D1")) < 2e-2
    assert rel(D, res.kernel("D1")) > 0.1                     # and it is not D_W itself


def test_strategy_refuses_delays():
    m = dict(CH1); m["agents"] = {k: dict(v) for k, v in CH1["agents"].items()}
    m["agents"]["player1"]["signals"] = {"y1": {"drift": {"X": "sqrt(p1)"}, "noise": {"w1": 1}, "delay": 0.2}}
    m["numerics"] = {"nodes": 8}
    r = ns.solve(m)
    with pytest.raises(NotImplementedError):
        r.strategy("D1")


def test_channel_argument(res):
    t = np.array([0.5]); s = np.array([0.2]); k = res.shocks.index("w0")
    assert abs(float(res.estimate("player2", "X", "w0").at(t, s)[0]) - float(res.estimate("player2", "X").at(t, s)[0, k])) < 1e-12
    assert abs(float(res.strategy("D2", "w0").at(t, s)[0]) - float(res.strategy("D2").at(t, s)[0, k])) < 1e-12


# ---------------------------------------------------------------- the equations form and the 1.1 API

EQ = {"name": "game", "params": {"p1": 3, "p2": 10, "r1": 0.1, "r2": 0.1, "b1": 1, "b2": -1, "sigma": 1, "T": 1},
      "shocks": ["W0", "W1", "W2"], "states": {"X": "(D1 + D2) dt + sigma dW0"},
      "agents": {"player1": {"controls": "D1", "observes": "sqrt(p1) X dt + dW1", "loss": "(X - b1)^2 + r1 D1^2"},
                 "player2": {"controls": "D2", "observes": "sqrt(p2) X dt + dW2", "loss": "(X - b2)^2 + r2 D2^2"}},
      "horizon": {"T": "T"}, "numerics": {"nodes": 10}}


def test_the_three_forms_are_one_model(tmp_path):
    from noisestate import dt, sqrt
    p1, p2, r1, r2, b1, b2, sigma, T = ns.params(p1=3, p2=10, r1=0.1, r2=0.1, b1=1, b2=-1, sigma=1, T=1)
    dW0, dW1, dW2 = ns.shocks(3)
    X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
    X.d = (D1 + D2) * dt + sigma * dW0
    a1 = ns.Agent("player1", controls=D1, observes=sqrt(p1) * X * dt + dW1, loss=(X - b1)**2 + r1 * D1**2)
    a2 = ns.Agent("player2", controls=D2, observes=sqrt(p2) * X * dt + dW2, loss=(X - b2)**2 + r2 * D2**2)
    py = ns.Game(states=X, agents=[a1, a2], T=T, nodes=10)
    eq = ns.Model.from_dict(EQ)
    assert py.to_dict(numeric=True) == eq.to_dict(numeric=True)
    assert ns.schema.validate(EQ, "model") == [] and ns.schema.validate(eq.to_dict(), "model") == []
    path = tmp_path / "game.yaml"; eq.save(str(path))
    back = ns.load(str(path))
    assert back.to_dict(numeric=True) == eq.to_dict(numeric=True) and "observes" in path.read_text()


def test_costs_keep_the_loss_constant():
    r = ns.solve(EQ)
    for a, b in (("player1", 1.0), ("player2", -1.0)):
        parts = r.cost_parts[a]
        assert abs(parts["constant"] - b * b * 1.0) < 1e-12                        # b^2 over [0, T = 1]
        assert abs(r.costs[a] - (parts["variance"] + parts["mean"] + parts["constant"])) < 1e-12


def test_a_drift_term_needs_its_dt():
    X = ns.State("X"); D = ns.Control("D"); (dW,) = ns.shocks(1)
    with pytest.raises(ValueError, match="has no dt"):
        X.d = D + dW
    bad = {**EQ, "states": {"X": "(D1 + D2) + sigma dW0"}}
    with pytest.raises(ValueError, match="has no dt"):
        ns.Model.from_dict(bad)


def test_response_follows_one_shock():
    r = ns.solve(EQ)
    t = np.array([0.0, 0.5, 1.0])
    x = r.response("X", to="W0", at=0.0).over(t)
    assert np.allclose(x, r.kernel("X", "W0").at(t, np.zeros(3)))
    assert r.response("X", to="W0", at=0.5).over(np.array([0.2]))[0] == 0.0           # before the shock
    e = r.response("X", to="W0", seen_by="player2").over(t)
    assert np.allclose(e, r.estimate("player2", "X", "W0").at(t, np.zeros(3)))


def test_describe_writes_the_parameters():
    text = ns.Model.from_dict(EQ).describe()
    assert "sqrt(p1) X dt" in text and "sigma dW0" in text and "b1^2" in text


def test_sweep_resolves_a_relative_past_from_the_file(tmp_path, monkeypatch):
    # sweep() read the file itself, so a transition's relative horizon.past.model resolved against the working
    # directory (FileNotFoundError from anywhere but examples/); it now loads through load()
    monkeypatch.chdir(tmp_path)
    rows = ns.sweep(ns.example("ch3_precision_change"), "p1", [3.0], numerics={"nodes": 6})
    assert rows[0].result is not None


def test_estimates_and_strategies_on_the_stationary_engine():
    r = ns.solve(ns.load(ns.example("ch3_two_player")), nodes=16)
    assert rel(r.estimate("player1", "D1"), r.kernel("D1")) < 1e-9                 # its own action, known
    a = np.array([0.5, 1.0])
    x, x1, x2 = (r.response("X", to="w0", seen_by=who).over(a) for who in (None, "player1", "player2"))
    assert np.all(np.abs(x - x2) < np.abs(x - x1))                                   # player 2 (p = 10) knows more
    D = np.asarray(r.strategy("D1"))
    assert rel(r._project("player1", D), r.kernel("D1")) < 0.05                     # 2.3% at 16 nodes, 0.2% at 48
