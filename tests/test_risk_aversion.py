"""risk_aversion (CARA, the entropic objective theta^-1 log E exp(theta C)): declared on the three surfaces, carried
through save/load, validated, and refused by every engine until one solves it.  theta = 0 is today's model exactly."""
import json

import pytest

import noisestate as ns
from noisestate import dt, sqrt


def _file(theta):
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    d["agents"]["player1"]["risk_aversion"] = theta
    return d


def test_zero_is_the_risk_neutral_model_and_writes_no_key():
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    assert "risk_aversion" not in json.dumps(d)
    a, b = ns.solve(ns.Model.from_dict(_file(0.0))), ns.solve(ns.Model.from_dict(d))     # an explicit 0 solves as before
    assert a.costs == b.costs


def test_file_form_parameter_round_trip(tmp_path):
    p = tmp_path / "m.yaml"
    src = open(ns.example("ch1_two_player_finite")).read()
    src = src.replace("params: {p1: 3.0", "params: {gamma: 0.3, p1: 3.0").replace(
        "    loss: X^2 + r1 D1^2\n", "    loss: X^2 + r1 D1^2\n    risk_aversion: gamma\n")
    p.write_text(src)
    m = ns.load(str(p))
    assert [a.risk_aversion for a in m.agents] == [0.3, 0.0]
    m.save(str(tmp_path / "out.yaml"))
    assert "risk_aversion: gamma" in (tmp_path / "out.yaml").read_text()
    assert ns.load(str(tmp_path / "out.yaml")).to_dict() == m.to_dict()
    assert m.with_params(gamma=0.7).agents[0].risk_aversion == 0.7
    assert any("risk averse" in n for n in m.notes)


def test_python_form_collects_the_parameter():
    r, p, sigma, gam = ns.params(r=0.1, p=3.0, sigma=1.0, gam=0.4)
    dW0, dW1 = ns.shocks(2)
    X = ns.State("X"); D = ns.Control("D")
    X.d = D * dt + sigma * dW0
    me = ns.Agent("me", controls=D, observes=sqrt(p) * X * dt + dW1, loss=X**2 + r * D**2, risk_aversion=gam)
    assert ns.Game(states=X, agents=me, T=1.0).agents[0].risk_aversion == 0.4
    with pytest.raises(ValueError, match="risk_aversion"):
        ns.Agent("x", controls=D, observes=X * dt + dW1, loss=X**2, risk_aversion="big")


@pytest.mark.parametrize("bad", [-1.0, float("inf")])
def test_negative_or_infinite_is_rejected(bad):
    with pytest.raises(ValueError, match="risk_aversion"):
        ns.Model.from_dict(_file(bad))


def test_every_engine_refuses_a_risk_averse_agent():
    with pytest.raises(NotImplementedError, match="risk-averse"):
        ns.solve(ns.Model.from_dict(_file(0.5)))
    d = ns.load(ns.example("ch3_two_player")).to_dict()
    d["agents"][next(iter(d["agents"]))]["risk_aversion"] = 0.5
    with pytest.raises(NotImplementedError, match="risk-averse"):
        ns.solve(ns.Model.from_dict(d))


def test_tied_agents_must_share_it():
    """The tie signature carries theta: two agents alike but for it are not interchangeable."""
    same = ns.Model.from_dict(_file(0.0))
    diff = ns.Model.from_dict(_file(0.5))
    assert same._agent_signature(same.agents[0])[:3] == same._agent_signature(same.agents[1])[:3]
    assert diff._agent_signature(diff.agents[0])[:3] != diff._agent_signature(diff.agents[1])[:3]
