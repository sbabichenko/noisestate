"""Level rows: `{level: q, filter: true}`, the exact path of a state or of another agent's control, filtered through its
increments.  Their drift is the kernel's rate of change and their noise loading its jump at age 0, both set by the
equilibrium.  The exact checks: a level row carries the same information as the row of its increments written out,
so the equilibria coincide; then Chapter 6's opaque market, whose trader sees the quote but not the order flow and
reads a quote off the rule as noise flow (the naive corner)."""
import json
import numpy as np
import pytest
import noisestate as ns
from test_monitoring import market


def _ch3(kind):
    d = ns.load(ns.example("ch3_two_player")).to_dict()
    d["horizon"]["window"] = 6.0; d.setdefault("numerics", {})["nodes"] = 24
    d["agents"]["player2"]["signals"]["x"] = ({"level": "X"} if kind == "level" else
                                              {"drift": {"D1": 1.0, "D2": 1.0}, "noise": {"w0": 1.0}})
    return ns.Model.from_dict(d)


def test_a_states_level_is_the_row_of_its_increments():
    """Seeing X exactly is seeing dX = (D1 + D2) dt + dw0: the same equilibrium."""
    a, b = ns.solve(_ch3("row")), ns.solve(_ch3("level"))
    assert a.converged and b.converged
    for u in ("D1", "D2", "X"):
        assert np.abs(a.kernel(u) - b.kernel(u)).max() < 1e-8


@pytest.mark.parametrize("instant", [False, True])
def test_a_competitive_price_level_is_the_order_flow(instant):
    """Kyle-Back: the price is an invertible filter of the flow, so a trader filtering the price level knows what one
    seeing the flow (less its own order) knows; with or without its instant reaction to the price."""
    def model(kind):
        d = ns.load(ns.example("ch4_kyle_back")).to_dict()
        t = d["agents"]["trader1"]
        if instant:
            t["instant"] = ["P"]
        if kind == "level":
            t["signals"].pop("flow"); t["signals"]["quote"] = {"level": "P"}
        return ns.Model.from_dict(d)
    a, b = ns.solve(model("row")), ns.solve(model("level"))
    assert a.converged and b.converged
    for u in ("D1", "P"):
        assert np.abs(a.kernel(u) - b.kernel(u)).max() < 1e-7


def _opaque(gamma, nodes=24):
    d = market(gamma, transparent=False, nodes=nodes).to_dict()
    t = d["agents"]["trader"]; t["signals"].pop("flow"); t["signals"]["quote"] = {"level": "P"}
    return ns.Model.from_dict(d)


def test_the_level_row_round_trips_and_keeps_the_instant_reaction():
    m = _opaque(0.1)
    eq = m.to_equations()
    assert eq["agents"]["trader"]["observes"]["quote"] == {"level": "P", "filter": True}
    back = ns.Model.from_dict(eq)
    assert back.to_dict() == m.to_dict() and back.agents[1].instant == ["P"]
    assert [(r.name, r.level) for r in back.agents[1].signals] == [("y", ""), ("quote", "P")]


def test_level_rows_are_refused_where_they_are_not_solved():
    d = _opaque(0.1).to_dict()
    with pytest.raises(ValueError, match="own control"):
        bad = json.loads(json.dumps(d)); bad["agents"]["trader"]["signals"]["mine"] = {"level": "D"}
        ns.Model.from_dict(bad)
    with pytest.raises(ValueError, match="not a state or a control"):
        bad = json.loads(json.dumps(d)); bad["agents"]["trader"]["signals"]["quote"] = {"level": "nothing"}
        ns.Model.from_dict(bad)
    f = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    f["agents"]["player2"]["signals"]["x"] = {"level": "X"}
    with pytest.raises(ValueError, match="stationary engine only"):
        ns.Model.from_dict(f)
    lagged = ns.load(ns.example("ch1_delayed_finite")).to_dict()
    lagged["horizon"] = {"window": 3.0}; lagged["agents"]["player1"]["signals"]["x"] = {"level": "X"}
    with pytest.raises(ValueError, match="without lags or delays"):
        ns.Model.from_dict(lagged)


def test_the_opaque_market_is_competitive_without_an_inventory_cost():
    """gamma = 0: nothing to manage, so the quote is the competitive one whether or not the trader sees the flow."""
    a, b = ns.solve(market(0.0, nodes=24)), ns.solve(_opaque(0.0))
    assert a.converged and b.converged
    assert np.abs(a.kernel("P") - b.kernel("P")).max() < 1e-7
    assert np.abs(a.kernel("D") - b.kernel("D")).max() < 1e-7


def test_the_opaque_market_by_continuation_and_its_naive_trader():
    """Continued from the competitive market up the inventory weight (a cold start at gamma = 0.1 does not converge),
    then the naive trader's reading of a quote spike: a block at once, of the loss's reaction -1/(2 eps) plus its map on
    the quote at age 0, well short of the privy trader's -2.5 (Chapter 6: the quote read as noise flow)."""
    prev = None
    for g in np.round(np.arange(0.0, 0.1001, 0.01), 3):
        r = ns.solve(_opaque(float(g)), **({"start_from": prev.maps} if prev is not None else {}))
        assert r.converged, (g, r.message)
        prev = r
    c = prev.compiled
    net = float((c.grid.interp([0.0]) @ prev.kernel("P"))[0, c.channels.index("wZ")])
    assert net == pytest.approx(1.212, abs=5e-3)                       # Lambda - P^Q; the chapter's 120 cells: 1.216
    t = ns.solve(market(0.1, nodes=24))
    assert net < float((t.compiled.grid.interp([0.0]) @ t.kernel("P"))[0, 1]) - 0.05     # below the transparent market's
    block = -prev.deviation_response("mm", ["Q"]).over([0.0])[0, 0]
    g0 = float(prev.maps["trader"][0, 1, 0])
    assert block == pytest.approx(-2.5 + g0, rel=1e-8)
    assert -1.0 < block < -0.5                                           # 0.3 of the privy block


def _up(build, top=0.1, step=0.02):
    prev = None
    for g in np.round(np.arange(0.0, top + 1e-9, step), 4):
        r = ns.solve(build(float(g)), **({"start_from": prev.maps} if prev is not None else {}))
        assert r.converged, (g, r.message)
        prev = r
    return prev


def test_transparent_is_seeing_only_the_price_and_being_privy():
    """A trader privy to the market maker who sees only the quote (a level row) is the transparent market's trader, who
    sees the quote and the order flow: on the path the quote gives the flow, and off it the trader knows a deviation for
    what it is.  The privy trader's instant reaction to a quote spike is its loss's alone (a spike is no news to it),
    while on the path its map on the quote reacts at once as well: the two composites (use_maps, seed_composite)."""
    def price_only(g):
        d = market(g, nodes=24).to_dict()
        t = d["agents"]["trader"]; t["signals"].pop("flow"); t["signals"]["quote"] = {"level": "P"}
        return ns.Model.from_dict(d)
    a = ns.solve(market(0.1, nodes=24)); b = _up(price_only)
    ages = np.array([0.0, 0.5, 2.0])
    va = a.deviation_response("mm", ["Q", "D", "P"]).over(ages); vb = b.deviation_response("mm", ["Q", "D", "P"]).over(ages)
    assert np.abs(va - vb).max() < 1e-8
    assert va[0, 2] / va[0, 0] == pytest.approx(-0.325, abs=5e-3)      # P^Q, Chapter 6's -0.33


def test_one_privy_and_one_naive_trader_against_one_market_maker():
    """Both see only the quote; A monitors the market maker, B does not.  On the path they are the same trader; after the
    market maker's quote spike A sells the loss's -1/(2 eps) at once and B less, reading part of it as noise flow."""
    def two(g):
        tr = lambda j, privy: {"controls": f"D{j}", **({"monitors": "mm"} if privy else {}),
                               "observes": {f"y{j}": f"(V - P) dt + dwY{j}", "quote": {"level": "P", "filter": True}},
                               "loss": f"-V D{j} + P D{j} + eps D{j}^2"}
        return ns.Model.from_dict({
            "name": "two", "params": {"eps": 0.2, "gamma": g, "rho": 0.5, "sigma_Z": 1.0},
            "shocks": ["wV", "wZ", "wYA", "wYB"], "states": {"V": "dwV", "Q": "-DA dt - DB dt - sigma_Z dwZ"},
            "agents": {"mm": {"controls": "P", "observes": {"flow": "DA dt + DB dt + sigma_Z dwZ"},
                              "loss": "V DA + V DB - P DA - P DB + gamma Q^2"},
                       "A": tr("A", True), "B": tr("B", False)},
            "horizon": {"window": 10.0, "discount": "rho"}, "numerics": {"nodes": 30}})
    r = _up(two)
    assert np.abs(r.kernel("DA")[:, [0, 1]] - r.kernel("DB")[:, [0, 1]]).max() < 1e-8     # the same trader on the path
    q0 = r.deviation_response("mm", ["Q"]).over([0.0])[0, 0]
    gB = float(r.maps["B"][0, 1, 0])
    assert q0 == pytest.approx(2.5 + (2.5 - gB), rel=1e-8)             # A's block 2.5, B's -(h + g0)
    assert 0.5 < 2.5 - gB < 1.0


def test_the_naive_readers_response_to_a_shading_by_forward_simulation():
    """Independent of the deviation-response machinery: a forward simulation in discrete time of the opaque market with
    no shocks and a narrow unit-area shading of the quote, the market maker quoting by its rule (its map on the flow)
    and the trader acting by its instant reaction plus its maps on its signal and on the quote's increments.  Its
    orders during the shading tend to the block h + g(0+) as the shading narrows (Richardson on two widths), and in all
    it buys back almost everything it sold (Chapter 6's +0.22 in total came from a coarse, path-dependent solver)."""
    prev = None
    for g in np.round(np.arange(0.0, 0.1001, 0.01), 3):
        prev = ns.solve(_opaque(float(g)), **({"start_from": prev.maps} if prev is not None else {}))
    res = prev
    c = res.compiled; names = [s.name for s in res.model.agents[1].signals]
    h = c.instant_loads["D"]["P"]

    def simulate(width, dt):
        n = int(round(4.0 / dt)) + 1
        ages = np.arange(n) * dt
        I = c.grid.interp(ages)
        gy, gq = I @ res.maps["trader"][0, names.index("y")], I @ res.maps["trader"][0, names.index("quote")]
        gm = I @ res.maps["mm"][0, 0]
        s = np.exp(-0.5 * ((ages - 0.5) / width) ** 2); s /= s.sum() * dt
        P, D, dF, dY, dP = (np.zeros(n) for _ in range(5))
        for k in range(n):
            P[k] = (np.dot(gm[1:k + 1][::-1], dF[:k]) if k else 0.0) + s[k]
            dP[k] = P[k] - (P[k - 1] if k else 0.0)
            D[k] = h * P[k] + (np.dot(gy[1:k + 1][::-1], dY[:k]) if k else 0.0) + np.dot(gq[:k + 1][::-1], dP[:k + 1])
            dF[k], dY[k] = D[k] * dt, -P[k] * dt
        during = np.abs(ages - 0.5) <= 6 * width
        return D[during].sum() * dt, D.sum() * dt

    (b1, _), (b2, n2) = simulate(0.01, 0.001), simulate(0.005, 0.0005)
    block = 2 * b2 - b1                                                    # the shading's width to zero
    assert block == pytest.approx(h + float(res.maps["trader"][0, names.index("quote"), 0]), abs=3e-3)
    assert abs(n2) < 0.05 * abs(block)                                   # net: almost everything bought back


def test_a_naive_player_who_also_sees_what_moves_the_quote_is_warned():
    """A player that sees a control's level and shares what its owner observes can tell a quote off the rule, so it is
    privy; built naive, noisestate warns.  The naive trader of the opaque market (the quote alone) and a privy one are quiet."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        market(0.1, transparent=False, nodes=16)                              # sees the quote and the flow, not privy
    assert any("sees the level of mm's P" in str(x.message) for x in w)
    for build in (lambda: market(0.1, nodes=16), lambda: _opaque(0.1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = build().to_dict()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ns.Model.from_dict(d)
        assert not any("sees the level" in str(x.message) for x in w)


def test_continue_from_reaches_a_market_a_cold_start_does_not():
    """solve(m, continue_from={"gamma": 0}): Chapter 6's opaque market at gamma = 0.1 by continuation from the
    cost-free corner, the same point as stepping there by hand; unknown parameters are refused with a hint."""
    cold = ns.solve(_opaque(0.1))
    assert not cold.converged
    r = ns.solve(_opaque(0.1), continue_from={"gamma": 0.0})
    assert r.converged and r.continued[0] == (0.0, "converged") and r.continued[-1][0] == 1.0
    by_hand = _up(_opaque, step=0.01)
    assert np.abs(r.kernel("P") - by_hand.kernel("P")).max() < 1e-7
    with pytest.raises(ValueError, match="did you mean gamma"):
        ns.solve(_opaque(0.1), continue_from={"gama": 0.0})
