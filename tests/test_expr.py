"""The expression API (noisestate.expr): the shipped examples written as equations compile to their YAML
files' dicts and solve to the baseline's numbers; the README's target script runs; a model saved and loaded
is equal; coefficients render to what spec's evaluator reads; the errors name the object."""
import copy
import json
import os
import random
import sys

import numpy as np
import pytest

import noisestate as ns
from noisestate import Param, State, Control, Signal, Agent, shocks, define, sqrt, exp, dt
from noisestate.spec import safe_eval

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
if EX not in sys.path:
    sys.path.insert(0, EX)
from expr_examples import EXAMPLES                                    # noqa: E402

BASELINE = json.load(open(os.path.join(HERE, "refs", "baseline_0.4.json")))["cases"]


def _canon(d: dict) -> dict:
    """The dict of a model file with the two defaults the builder-written ch5 file spells out removed
    (`myopic: false`, `delay: 0.0`), then normalised through Model.from_dict().to_dict()."""
    d = copy.deepcopy(d)
    for a in d.get("agents", {}).values():
        if a.get("myopic") is False:
            del a["myopic"]
        for r in (a.get("signals") or {}).values():
            if r.get("delay") == 0:
                del r["delay"]
    return ns.Model.from_dict(d).to_dict()


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_compile_to_their_files(name):
    yaml_dict = ns.load(os.path.join(EX, name + ".yaml")).to_dict()
    past = yaml_dict["horizon"].get("past") or {}
    if isinstance(past.get("model"), str):
        past["model"] = os.path.basename(past["model"])          # load() resolved it against the file's directory
    model = EXAMPLES[name]()
    assert _canon(model.to_dict()) == _canon(yaml_dict)
    # and the numeric form, coefficients evaluated, is identical too
    assert ns.Model.from_dict(model.to_dict()).to_dict(numeric=True) == ns.Model.from_dict(yaml_dict).to_dict(numeric=True)


@pytest.mark.parametrize("name", ["ch1_two_player_finite", "ch1_delayed_finite", "ch3_two_player", "ch4_kyle_back", "kyle_back_prior"])
def test_examples_solve_to_the_baseline(name):
    res = EXAMPLES[name]().solve().require_converged()
    for agent, cost in BASELINE[name]["costs"].items():
        assert abs(res.costs[agent] - cost) <= 1e-14 * max(1.0, abs(cost)), (name, agent)


@pytest.mark.skipif(not os.environ.get("NOISESTATE_SLOW"), reason="ch5 (6 s) and the precision-change transition: NOISESTATE_SLOW=1")
def test_slow_examples_solve_to_the_files():
    res = EXAMPLES["ch5_cycle_market"]().solve().require_converged()
    for agent, cost in BASELINE["ch5_cycle_market"]["costs"].items():
        assert abs(res.costs[agent] - cost) <= 1e-14 * max(1.0, abs(cost))
    m = EXAMPLES["ch3_precision_change"](past=EXAMPLES["ch3_two_player"]())
    a = m.solve(); b = ns.solve(os.path.join(EX, "ch3_precision_change.yaml"))
    for agent in a.costs:
        assert abs(a.costs[agent] - b.costs[agent]) <= 1e-14 * max(1.0, abs(b.costs[agent]))


TARGET = '''
import noisestate as ns
from noisestate import Signal, dt, sqrt
r1, r2, p1, p2, sigma = ns.params(r1=0.1, r2=0.1, p1=3.0, p2=3.0, sigma=1.0)
dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
X = ns.State("X"); D1, D2 = ns.Control("D1"), ns.Control("D2")
X.d = (D1 + D2) * dt + sigma * dw0
player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1}, loss=X**2 + r1 * D1**2)
player2 = ns.Agent("player2", controls=D2, observes=Signal("y2", sqrt(p2) * X * dt + dw2, delay=0.5), loss=(X - 1)**2 + r2 * D2**2)
game = ns.Game(X, [player1, player2], window=3.0, name="ch1")
eq = game.solve()
eq = game.solve(ns.Numerics(nodes=16, unit=0.5))
eq.diagnostics.assess().accepted; eq.costs["player1"]; eq.cost_parts["player1"]; eq.means["X"]
k = eq.kernel("X", "w0"); k.values; k.axes; k.at(0.7)
for point in game.sweep(p1=[0.3, 1, 3, 10]): point.value, point.result, point.jump
old = game.solve(); new = game.with_params(p1=10.0).with_finite(T=6.0); path = new.solve(past=old)
game.save("ch1.yaml"); ns.load("ch1.yaml")
game.solve(settings={"second_order_tol": 1e-3})
'''


def test_target_script_runs(tmp_path, monkeypatch):
    """The README's script, verbatim but for the transition line: with player 2's delayed row the strip of
    the transition on [0, 6] at 16 nodes per piece has 20k nodes (an allocation of tens of GB), so the test
    runs that line at Numerics(nodes=4) on the same model; the rest is the text above, saving to tmp_path."""
    monkeypatch.chdir(tmp_path)
    src = TARGET.replace("path = new.solve(past=old)", "path = new.with_numerics(nodes=4).solve(past=old, max_evaluations=2, diagnostics=False)")
    assert src != TARGET
    env = {}
    exec(compile(src, "target", "exec"), env)
    assert env["eq"].diagnostics.assess().accepted in (True, False) and (tmp_path / "ch1.yaml").exists()
    assert isinstance(env["k"].values, np.ndarray) and not isinstance(env["k"].values, ns.Kernel)
    assert env["path"].kind == "transition"


def test_target_script_pieces(tmp_path):
    """The target's objects one by one: the status, the kernel's values and axes, the sweep rows, the horizon
    change, save and load, the settings block."""
    env = {}
    exec(compile(TARGET.split("eq = game.solve()")[0], "head", "exec"), env)
    game = env["game"]
    assert game.to_dict()["agents"]["player2"]["loss"] == [[1, "X", "X"], [-2, "X"], ["r2", "D2", "D2"]]
    assert game.to_dict()["agents"]["player2"]["constant"] == 1          # the loss's constant is kept: it is part of the cost
    eq = game.solve(ns.Numerics(nodes=16, unit=0.5))
    verdict = eq.diagnostics.assess()
    assert verdict.accepted is True and verdict.policy == "publication" and verdict.blocking == ()
    k = eq.kernel("X", "w0")
    assert isinstance(k, np.ndarray) and isinstance(k, ns.Kernel) and k.shape == (len(eq.axes["age"]),)
    assert list(k.axes) == ["age"] and np.array_equal(k.axes["age"], eq.axes["age"])
    assert k.at(0.7) == pytest.approx(float((eq.compiled.grid.interp([0.7]) @ np.asarray(k))[0]))
    assert eq.kernel("X").at([0.7, 0.9]).shape == (2, 3) and np.asarray(k.values).shape == k.shape
    rows = game.sweep(p1=[0.3, 1, 3])
    assert [r.value for r in rows] == [0.3, 1.0, 3.0] and rows[0].param == "p1" and rows[1].result.converged
    assert rows[-1].jump is False and abs(rows[-1].result.costs["player1"] - eq.costs["player1"]) < 1e-8
    with pytest.raises(ValueError, match="exactly one parameter"):
        game.sweep(p1=[1], p2=[1])
    new = game.with_params(p1=10.0).with_finite(T=6.0)
    assert new.horizon.kind == "finite" and new.horizon.T == 6.0 and new.horizon.window is None and new.params["p1"] == 10.0
    assert new.with_stationary(window=4.0).horizon.kind == "stationary"
    path = os.path.join(tmp_path, "ch1.yaml")
    game.save(path)
    loaded = ns.load(path)
    assert loaded == game and loaded.to_dict() == game.to_dict()
    assert ns.solve(loaded, ns.Numerics(nodes=16, unit=0.5)).costs == eq.costs
    res = game.solve(ns.Numerics(nodes=16, unit=0.5), settings={"second_order_tol": 1e-3})
    assert res.settings.second_order_tol == 1e-3 and ns.Settings.of(None).second_order_tol == 1e-4
    assert eq.settings.second_order_tol == 1e-4
    with pytest.raises(TypeError):
        game.solve(settings={"no_such_setting": 1})


def test_round_trip_of_a_model_with_lags_and_definitions(tmp_path):
    m = EXAMPLES["ch1_delayed_finite"]()
    path = os.path.join(tmp_path, "m.yaml"); m.save(path)
    assert ns.load(path) == m
    m = EXAMPLES["ch5_cycle_market"]()
    path = os.path.join(tmp_path, "ch5.yaml"); m.save(path)
    assert ns.load(path) == m and ns.load(path).to_dict() == m.to_dict()


def test_coefficient_rendering_agrees_with_the_evaluator():
    p1, c, theta, r1, rho, tau, a, b = Param.many(p1=3.0, c=0.2, theta=4.0, r1=1.0, rho=0.5, tau=0.25, a=1.5, b=2.5)
    cases = {sqrt(p1): "sqrt(p1)", p1**0.5: "sqrt(p1)", 2 * c: "2*c", -2 * theta: "-2*theta", r1 / 2: "r1/2",
             exp(-rho * tau): "exp(-rho*tau)", theta - 1: "theta - 1", -(1 - a - b): "-(1 - a - b)", 0.5 * r1: "0.5*r1",
             (a + b) * c: "(a + b)*c", a - (b - c): "a - (b - c)", a / (b * c): "a/(b*c)", -a**2: "-a**2", (-a)**2: "(-a)**2",
             a**(b + 1): "a**(b + 1)", 2**a: "2**a", ns.tanh(a) + ns.log(b) * ns.cos(c) - ns.sin(theta): "tanh(a) + log(b)*cos(c) - sin(theta)",
             abs(-a): "abs(-a)", 1 / rho: "1/rho", -(-a): "--a", a - -b: "a - (-b)", -2.5 * a: "-2.5*a", (a * b) ** 2: "(a*b)**2"}
    rng = random.Random(3)
    for coef, text in cases.items():
        assert str(coef) == text, (str(coef), text)
        for _ in range(5):
            params = {p.name: rng.uniform(0.1, 3.0) for p in (p1, c, theta, r1, rho, tau, a, b)}
            assert safe_eval(str(coef), params) == pytest.approx(coef.value(params), rel=1e-14, abs=1e-14)
    # a Param has the functions' arithmetic; a number folds
    assert sqrt(4.0) == 2.0 and isinstance(sqrt(p1), ns.expr.Coef)


def test_errors_name_the_object():
    p = Param("p", 1.0)
    w = shocks("w0", "w1")
    X = State("X"); D = Control("D")
    with pytest.raises(ValueError, match="signal 'y' has no shock term"):
        Signal("y", X * dt)
    with pytest.raises(ValueError, match="cubic"):
        X**2 * D
    with pytest.raises(ValueError, match="not quadratic"):
        (X**2) * (X**2)
    with pytest.raises(ValueError, match="only the square of a linear expression is quadratic"):
        X**3
    with pytest.raises(ValueError, match="shock 'w0' cannot be lagged"):
        w.w0.lag(0.5)
    with pytest.raises(ValueError, match="cannot lag shock 'w0'"):
        (X + w.w0).lag(0.5)
    with pytest.raises(ValueError, match="unknown Param 'q'"):
        q = Param("q")                                       # no value
        X.d = D * dt + q * w.w0
        ns.Game([X], [Agent("me", [D], [Signal("y", X * dt + w.w1)], X**2 + p * D**2)], window=3.0)
    with pytest.raises(ValueError, match="unknown Param 'p'"):
        X.d = D * dt + w.w0
        ns.Game([X], [Agent("me", [D], [Signal("y", X * dt + w.w1)], X**2 + p * D**2)], window=3.0, params=[Param("r", 1.0)])
    with pytest.raises(ValueError, match="two states named 'X'"):
        ns.Game([X, State("X")], [Agent("me", [D], [Signal("y", X * dt + w.w1)], X**2 + p * D**2)], window=3.0)
    with pytest.raises(TypeError, match="ns.Game"):
        ns.Model("m", states=[X], agents=[Agent("me", [D], [Signal("y", X * dt + w.w1)], X**2 + p * D**2)])
    with pytest.raises(ValueError, match="agent me: its loss uses State\\('Y'\\)"):
        Y = State("Y")
        ns.Game([X], [Agent("me", [D], [Signal("y", X * dt + w.w1)], Y**2 + p * D**2)], window=3.0)
    with pytest.raises(ValueError, match="a shock \\(w0\\) cannot enter a loss"):
        (X + w.w0) * D


def test_definitions_lags_leads_and_constants():
    tau, k = Param.many(tau=0.5, k=0.3)
    w = shocks("w0", "w1")
    X = State("X"); D = Control("D")
    Xl = define("Xl", X.lag(tau))
    X.d = (-X + D.lag(0.25) + 0.3) * dt + w.w0                   # a constant drift is `const`
    me = Agent("me", [D], [Signal("y", Xl * dt + w.w1, delay=0.25)], (Xl - k)**2 + D**2 + D * X.lead(0.5))
    m = ns.Game([X], [me], window=3.0, name="m")
    d = m.to_dict()
    assert d["states"]["X"]["drift"] == {"X": -1, "D@0.25": 1, "const": 0.3}
    assert d["definitions"] == {"Xl": {"X@tau": 1}}
    assert d["agents"]["me"]["loss"] == [[1, "Xl", "Xl"], ["-2*k", "Xl"], [1, "D", "D"], [1, "D", "X@-0.5"]]
    assert d["agents"]["me"]["signals"]["y"] == {"drift": {"Xl": 1}, "noise": {"w1": 1}, "delay": 0.25}
    assert d["params"] == {"tau": 0.5, "k": 0.3} and d["shocks"] == ["w0", "w1"]
    assert m.drives_means and m.all_lags() == [0.25, 0.5, 0.75]
    # the file structure builds the same model
    plain = ns.Model.from_dict(d)
    assert plain == m and isinstance(plain, ns.Model)


def test_kernel_on_every_engine():
    m = EXAMPLES["ch1_two_player_finite"]()
    tri = m.solve(ns.Numerics(nodes=6), max_evaluations=2, diagnostics=False)
    k = tri.kernel("X", "w0")
    assert set(k.axes) == {"time", "age", "shock_time"} and k.at(0.8, 0.3) == pytest.approx(float(np.ravel(tri.evaluate("X", "w0", 0.8, 0.3))[0]))
    assert tri.kernel("X").at([0.8, 0.9], [0.3, 0.3]).shape == (2, 3)
    #  the cell engine's (N, N) kernel and its nearest-cell at(): extras/test_cells.py
    with pytest.raises(ValueError, match="lost its nodes"):
        k[1:].at(0.8, 0.3)
    assert tri.to_dict()["kernels"]["X"]["w0"] == np.asarray(k).tolist()
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    fig = k.plot(); assert fig is not None
    fig = EXAMPLES["ch3_two_player"]().solve(ns.Numerics(nodes=6), max_evaluations=1, diagnostics=False).kernel("X").plot()
    assert fig is not None
