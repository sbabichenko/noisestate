"""Regressions from the pre-release adversarial test and code review (0.2.3)."""
import os, json, numpy as np, pytest
import noisestate as ns
from noisestate.stationary import StationarySolver
from helpers import slow
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def _ch3(**hz):
    """kind "finite_cells" names the cell engine's result; the model asks for kind finite with engine cells."""
    d = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml"))
    d["numerics"] = {**d.get("numerics", {}), **{k: hz.pop(k) for k in list(hz) if k in ("nodes", "unit", "unit_range", "breakpoints")}}
    if hz.get("kind") == "finite_cells":
        hz["kind"] = "finite"; d["numerics"]["engine"] = "cells"
    d["horizon"].update(hz); return d


# ---------------------------------------------------------------- delayed rows in the stationary engine
def test_delayed_row_solves_and_is_window_independent():
    d = _ch3(window=3.0, nodes=16); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.5
    r3 = ns.solve(d).check()
    d = _ch3(window=6.0, nodes=16); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.5
    r6 = ns.solve(d).check()
    undelayed = ns.solve(_ch3(window=6.0, nodes=16)).check()
    assert all(r.second_order[a]["ok"] for r in (r3, r6) for a in r.second_order)
    assert abs(r6.costs["player2"] - r3.costs["player2"]) < 5e-3            # window 3 truncates; 6 has settled
    assert r6.costs["player2"] > undelayed.costs["player2"]                   # less information costs more
    g = r6.maps["player2"][0, 0]                                              # map on the delayed row
    assert np.all(g[r6.ages > 6.0 - 0.5 + 1e-9] == 0)                        # zero where it reads nothing


def test_one_agent_delayed_observation_costs_more_and_passes_second_order():
    base = {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": -0.3, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.5}, "noise": {"w1": 1.0}, "delay": 0.5}},
                             "loss": [[1.0, "X", "X"], [0.5, "D", "D"]]}},
            "horizon": {"kind": "stationary", "window": 6.0}, "numerics": {"nodes": 12}}
    r = ns.solve(base).check(); base["agents"]["a"]["signals"]["y"]["delay"] = 0.0; r0 = ns.solve(base).check()
    assert r.costs["a"] > r0.costs["a"] and r.second_order["a"]["ok"]
    assert r.stability()["radius"] == 0.0 and "stable" in r.summary()        # one agent: radius 0, not NaN


def test_cell_engine_dense_branch_with_delays():
    d = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml")); d.setdefault("numerics", {}).update(nodes=16, engine="cells")
    r = ns.solve(d).check()
    assert r.resolution_ok is None and r.to_dict()["resolution_ok"] is None
    r.refine(); assert r.refinement["nodes"] == 32                             # doubled: lags stay aligned


# ---------------------------------------------------------------- guards compare like with like
def test_refine_and_stability_rebuild_the_same_solver():
    kb = os.path.join(EX, "ch4_kyle_back.yaml")
    r = ns.solve(kb, naive_observers={"trader1": ["market_maker"]}, refine=True)
    assert r.refinement["cost_change"] < 1e-4                                 # not the naive-vs-full gap (1.1)
    assert r.solver_class is StationarySolver and r.solver_kw["naive_observers"]


def test_model_is_single_sourced():
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    with pytest.raises(TypeError):                                            # coefficients are numbers once built
        m.params["p1"] = 4.0
    m4 = m.with_params(p1=4.0); base = ns.solve(m); r4 = ns.solve(m4)
    assert m4.params["p1"] == 4.0 and abs(r4.costs["player1"] - base.costs["player1"]) > 1e-4
    fresh = ns.read_yaml(os.path.join(EX, "ch3_two_player.yaml")); fresh["params"]["p1"] = 4.0
    assert abs(r4.costs["player1"] - ns.solve(fresh).costs["player1"]) < 1e-12   # the same model as from a file
    r4.refine(); assert r4.refinement["cost_change"] < 1e-8                    # refine compares like with like
    m.horizon.nodes = 30                                                       # horizon fields are read live
    r = ns.solve(m); assert r.compiled.N == 30 and m.to_dict()["numerics"]["nodes"] == 30
    r.refine(); assert r.refinement["nodes"] == 45
    assert ns.solve(m.with_horizon().with_numerics(nodes=12)).compiled.N == 12
    with pytest.raises(ValueError, match="not parameters"):
        m.with_params(zzz=1.0)


def test_numeric_export_is_loadable_and_equivalent():
    for f in ("ch3_two_player.yaml", "ch5_cycle_market.yaml"):
        m = ns.load(os.path.join(EX, f)); m2 = ns.Model.from_dict(m.to_dict(numeric=True))
        assert m2.all_lags() == m.all_lags() and m2.control_names == m.control_names and not m2.params
    m = ns.load(os.path.join(EX, "ch3_two_player.yaml"))
    a, b = ns.solve(m), ns.solve(ns.Model.from_dict(m.to_dict(numeric=True)))
    assert abs(a.costs["player1"] - b.costs["player1"]) < 1e-12
    # a Model built directly from the dataclasses (no source) refines and sweeps
    direct = ns.Model(name="d", channels=m.channels, states=m.states, agents=m.agents, horizon=m.horizon, params=dict(m.params))
    assert ns.solve(direct, refine=True).refinement["resolved"]


def test_window_tail_ignores_random_walk_states_and_flags_the_undiscounted_kyle_back():
    kb = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml")); kb["params"]["rho"] = 0.0
    bad = ns.solve(kb)
    assert "WINDOW TOO SHORT" in bad.summary()                               # rho = 0: the average-cost artefact
    diagnostic = next(d for d in bad.to_dict()["diagnostics"] if d["name"] == "window")
    assert diagnostic["trend"]["assessment"] == "not_decaying" and diagnostic["suggested_options"] == {}
    kb["params"]["rho"] = 0.5; r = ns.solve(kb)
    assert r.window_tail < 0.01 and "WINDOW TOO SHORT" not in r.summary()     # V is a random walk, P tracks it: not flagged
    assert r.window_tail_extrapolation()["assessment"] == "decaying"


def test_second_order_flags_non_convex_losses():
    d = _ch3(); d["agents"]["player1"]["loss"] = [[0.5, "X", "X"], ["-r1", "D1", "D1"]]
    with pytest.warns(UserWarning, match="unbounded below in D1"):              # validation flags the negative own term
        r = ns.solve(d)
    assert not r.second_order["player1"]["ok"] and "NOT A MINIMUM" in r.summary() and r.second_order["player2"]["ok"]
    d = _ch3(); d["agents"]["player1"]["loss"] = [[0.5, "X", "X"], ["r1", "D1", "X"]]
    with pytest.warns(UserWarning, match="coefficient of D1 squared is 0"):    # and the missing one (a system at rcond 1e-23)
        assert not ns.solve(d).second_order["player1"]["ok"]
    assert all(v["ok"] for v in ns.solve(_ch3()).second_order.values())
    assert json.loads(json.dumps(ns.solve(_ch3()).to_dict()))["second_order"]["player1"]["ok"]


def test_sharp_optimality_the_projected_foc_vanishes_only_at_the_equilibrium():
    m = ns.load(os.path.join(EX, "ch4_kyle_back.yaml")); S = StationarySolver(m); res = S.solve().check(); c = S.c; nW = c.nW

    def projected_foc(maps, a):
        Zp = c.closed_loop(maps, excluded=a.name, impulse_controls=a.controls); Zpass, R = Zp[:, :nW], Zp[:, nW:]
        Fu = S._foc_operators(a, R)
        ytil, yinst = S._passive_rows(a, Zpass); H = S._projection_operator(a, ytil, yinst)
        Z = c.closed_loop(maps)
        worst = 0.0
        for ui in range(len(a.controls)):
            phi = np.stack([Fu[ui] @ Z[:, k] for k in range(nW)], axis=1)
            worst = max(worst, np.linalg.norm(H @ phi.T.reshape(-1)) / max(np.linalg.norm(phi), 1e-300))
        return worst
    for a in m.agents:
        at_eq = projected_foc(res.maps, a)
        scaled = {k: (1.01 * v if k == a.name else v) for k, v in res.maps.items()}
        off = projected_foc(scaled, a)
        assert at_eq < 1e-8 and off > 1e-4 and at_eq < 1e-5 * off, (a.name, at_eq, off)


# ---------------------------------------------------------------- validation and messages
@pytest.mark.parametrize("edit, match", [
    (lambda d: d["states"]["X"]["drift"].update({"X@-0.5": 0.1}), "future value"),
    (lambda d: d["agents"]["player1"]["signals"]["y1"].update(noise={"w1": 0.0}), "zero noise loading"),
    (lambda d: d["horizon"].update(breakpoints=[0, 1]), "breakpoints"),
    (lambda d: d["agents"]["player1"].update(myopic="false"), "myopic must be true or false"),
    (lambda d: d["agents"]["player1"].update(controls="D1"), "controls must be a list"),
    (lambda d: d["agents"]["player1"].update(signals=[{"drift": {"X": 1}}]), "signals must be a mapping"),
    (lambda d: d["params"].update({"p1": "2*base", "base": 1.5}), "defined after it"),
])
def test_validation_messages(edit, match):
    d = _ch3(); edit(d)
    with pytest.raises(ValueError, match=match):
        ns.Model.from_dict(d)


def test_structural_errors_come_before_the_unused_parameter_check():
    d = _ch3(); d["agents"]["player1"]["signals"] = {}                          # removing the rows also strips p1's use
    with pytest.raises(ValueError, match="signal") as e:
        ns.Model.from_dict(d)
    assert "never used" not in str(e.value)


def test_linear_terms_are_noted_and_builder_is_accepted():
    m = ns.load(os.path.join(EX, "ch5_cycle_market.yaml"))
    assert any("linear loss term" in n for n in m.notes)
    from noisestate import ModelBuilder
    b = (ModelBuilder("b", p=2.0).channel("w0", "w1").state("X", drift={"D": 1.0}, noise={"w0": 1.0})
         .agent("a", ["D"], [[1.0, "X", "X"], ["p", "D", "D"]]).stationary(window=4.0, nodes=8))
    b.signal("a", "y", drift={"X": 1.0}, noise={"w1": 1.0})
    assert ns.solve(b).converged and ns.sweep(b, "p", [2.0, 3.0])[1]["converged"]


def test_wrong_kind_warm_start_is_an_error_and_right_kinds_are_converted():
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml"))
    r = ns.solve(d).check(); S = ns.SpectralFiniteSolver(ns.Model.from_dict(d))
    assert S.solve(init=r.maps).converged                                     # raw maps are accepted and converted
    with pytest.raises(ValueError, match="expected action kernels"):
        S.solve(init={k: v[:, :, :5] for k, v in r.maps.items()})


def test_cli_reports_model_errors_as_messages(tmp_path, capsys):
    from noisestate.cli import main
    bad = tmp_path / "bad.yaml"; import yaml
    d = _ch3(); d["agents"]["player1"]["signals"]["y1"]["noise"] = {"w1": 0.0}; yaml.safe_dump(d, open(bad, "w"))
    assert main(["solve", str(bad)]) == 2 and "zero noise loading" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["solve", os.path.join(EX, "ch3_two_player.yaml"), "--nodes", "0"])
    with pytest.raises(SystemExit):
        main(["solve", os.path.join(EX, "ch3_two_player.yaml"), "--param", "p1=abc"])
    d = _ch3(); d["horizon"]["kind"] = "finite"; d["horizon"]["window"] = 1.0; d.setdefault("numerics", {})["nodes"] = 8
    good = tmp_path / "good.yaml"; yaml.safe_dump(d, open(good, "w"))
    assert main(["solve", str(good)]) == 0 and "discounted cost" in capsys.readouterr().out


@slow("slow (20 s); set NOISESTATE_SLOW=1")
def test_delayed_row_stationary_agrees_with_the_finite_engine_in_the_interior():
    """One agent with a delayed noisy observation: the stationary kernels (window 6) against the
    spectral finite engine at t = 4 of T = 7.  Both horizon ends leave transients (the filter
    settles from its start, the control changes near the end; at t = 7 of T = 8 the two engines
    differ by 8% on the control, delayed or not), so the comparison stays a few units from each.
    Both engines are exactly causal (the stationary one since the side-aware shift); the finite
    kernel is read at each stationary node from that node's side of its panel, since kernels jump
    at the delay."""
    base = {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": -0.3, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.5}, "noise": {"w1": 1.0}, "delay": 0.5}},
                             "loss": [[1.0, "X", "X"], [0.5, "D", "D"]]}}}
    import copy
    st = copy.deepcopy(base); st["horizon"] = {"kind": "stationary", "window": 6.0}; st["numerics"] = {"nodes": 16}; rs = ns.solve(st).check()
    fi = copy.deepcopy(base); fi["horizon"] = {"kind": "finite", "window": 7.0}; fi["numerics"] = {"nodes": 5}     # pieces of 0.5: 105 pieces
    rf = ns.solve(fi).check()
    g = rs.compiled.grid; sides = np.where((np.arange(g.N) % g.n) == g.n - 1, -1, 1)
    t_eval = 4.0                    # three units before the end and, for ages up to 1, three after the start (transients ~1e-4)
    I = rf.grid.interp(t_eval + 0 * rs.ages, rs.ages, side_a=sides)
    young = rs.ages <= 1.0
    for name in ("X", "D"):
        for ch in ("w0", "w1"):
            ks = rs.kernel(name, ch); kf = I @ rf.kernel(name, ch)
            assert np.abs(ks - kf)[young].max() < 1e-3 * max(1.0, np.abs(ks).max()), (name, ch, np.abs(ks - kf)[young].max())


def test_leads_only_in_cross_terms_with_the_own_current_control():
    d = _ch3(window=8.0, nodes=10); d["numerics"].update(unit=0.5, unit_range=4.0)
    for bad in ([[0.5, "X@-0.5", "X@-0.5"], ["0.5*r1", "D1", "D1"]], [[0.5, "X", "X"], ["0.5*r1", "D1@-0.5", "D1@-0.5"]],
                [[0.5, "X", "X"], ["0.5*r1", "D1", "D1"], [0.1, "D1@0.5", "X@-0.5"]]):
        d["agents"]["player1"]["loss"] = bad
        with pytest.raises(ValueError, match="lead"):
            ns.Model.from_dict(d)
    # the supported form: [c, D1, X@-tau] is the same flow as [c, D1@tau, X] at discount 0
    d["agents"]["player1"]["loss"] = [[0.5, "X", "X"], ["0.5*r1", "D1", "D1"], [0.2, "D1", "X@-0.5"]]; lead = ns.solve(d).check()
    d["agents"]["player1"]["loss"] = [[0.5, "X", "X"], ["0.5*r1", "D1", "D1"], [0.2, "D1@0.5", "X"]]; lag = ns.solve(d).check()
    assert abs(lead.costs["player1"] - lag.costs["player1"]) < 5e-3 * abs(lag.costs["player1"])


def test_ties_compare_private_state_dynamics_and_channel_sharing():
    def two(th1, s1, share):
        d = {"channels": ["w0", "wa0", "wa1", "wy0", "wy1"],
             "states": {"X": {"drift": {"D0": 1.0, "D1": 1.0}, "noise": {"w0": 1.0}},
                        "a0": {"drift": {"a0": -0.5}, "noise": {"wa0": 1.0}}, "a1": {"drift": {"a1": -th1}, "noise": {"wa1": s1}}},
             "agents": {}, "ties": [["f0", "f1"]], "horizon": {"kind": "stationary", "window": 4.0}, "numerics": {"nodes": 8}}
        for i in range(2):
            d["agents"][f"f{i}"] = {"controls": [f"D{i}"], "signals": {"y": {"drift": {"X": 1.0, f"a{i}": 1.0}, "noise": {(f"wa{i}" if share and i == 0 else f"wy{i}"): 1.0}}},
                                    "loss": [[1.0, "X", "X"], [1.0, f"D{i}", f"D{i}"], [0.5, f"D{i}", f"a{i}"]]}
        return d
    assert ns.Model.from_dict(two(0.5, 1.0, False)).ties
    with pytest.raises(ValueError, match="not structurally identical"):
        ns.Model.from_dict(two(5.0, 3.0, False))
    with pytest.raises(ValueError, match="not structurally identical"):
        ns.Model.from_dict(two(0.5, 1.0, True))


def test_delayed_row_equilibrium_is_insensitive_to_the_least_squares_cutoff():
    d = _ch3(window=6.0, nodes=16); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.5
    m = ns.Model.from_dict(d); orig = np.linalg.lstsq; costs = {}
    try:
        for rc in (1e-9, 1e-6):
            np.linalg.lstsq = lambda A, b, rcond=None, _o=orig, _rc=rc: _o(A, b, rcond=_rc)
            costs[rc] = ns.StationarySolver(m).solve().check().costs["player2"]
    finally:
        np.linalg.lstsq = orig
    assert abs(costs[1e-9] - costs[1e-6]) < 1e-6


# ---------------------------------------------------------------- second round: validation and guards
@pytest.mark.parametrize("edit, match", [
    (lambda d: d["numerics"].update(unit=0.5, unit_range=8.0), "unit_range"),
    (lambda d: d["numerics"].update(nodes=12.7), "must be an integer"),
    (lambda d: d["agents"]["player1"]["signals"]["y1"].update(delay=3.0), "not below the window"),
    (lambda d: d["states"]["X"]["drift"].update({"D1@3.5": 0.1}), "not below the window"),
    (lambda d: d["agents"]["player1"]["loss"].append([0.1, "D1", "X@-9.0"]), "not below the window"),
    (lambda d: d["agents"]["player1"].update(loss=[[0.5, "X", "X"]]), "do not enter its loss"),
    (lambda d: d["agents"].update(ghost=None), "empty block"),
])
def test_second_round_validation(edit, match):
    d = _ch3(); edit(d)
    with pytest.raises(ValueError, match=match):
        ns.Model.from_dict(d)


def test_lag_beyond_horizon_rejected_on_every_engine():
    for kind in ("stationary", "finite", "finite_cells"):
        d = _ch3(kind=kind, window=0.2, nodes=4); d["states"]["X"]["drift"] = {"D1@0.25": 1.0, "D2": 1.0}
        with pytest.raises(ValueError, match="not below the window"):
            ns.Model.from_dict(d)


def test_naive_observers_are_validated():
    kb = ns.load(os.path.join(EX, "ch4_kyle_back.yaml"))
    for bad, exc in (({"trader1": ["playr2"]}, ValueError), ({"playr1": ["market_maker"]}, ValueError),
                     ({"trader1": "market_maker"}, TypeError), ({"trader1": ["trader1"]}, ValueError)):
        with pytest.raises(exc):
            ns.StationarySolver(kb, naive_observers=bad)


def test_wrong_grid_warm_start_is_an_error_on_every_engine():
    d = _ch3(nodes=12); r = ns.solve(d); d.setdefault("numerics", {})["nodes"] = 16
    with pytest.raises(ValueError, match="different grid"):
        ns.StationarySolver(ns.Model.from_dict(d)).solve(init=r.maps)
    dc = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); dc.setdefault("numerics", {}).update(nodes=8, engine="cells")
    rc = ns.solve(dc); dc.setdefault("numerics", {})["nodes"] = 16
    with pytest.raises(ValueError, match="different grid"):
        ns.FiniteSolver(ns.Model.from_dict(dc)).solve(init=rc.maps)


def test_jump_flag_is_quiet_on_a_geometric_sweep():
    rows = ns.sweep(_ch3(), "r1", [2.0 / 2 ** k for k in range(7)])
    assert not any(r["jump"] for r in rows) and all(r["converged"] for r in rows)


def test_refine_on_cells_reports_without_a_verdict():
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d.setdefault("numerics", {}).update(nodes=12, engine="cells")
    r = ns.solve(d); r.refine()
    assert r.refinement["resolved"] is None and "NOT RESOLVED" not in r.summary() and r.refinement["nodes"] == 24


def test_second_order_and_stability_report_their_method():
    r = ns.solve(_ch3(), stability=True)
    assert all(v["converged"] for v in r.second_order.values()) and r.stability_report["method"] == "arnoldi"
    assert r.to_dict()["stability"]["method"] == "arnoldi"


def test_delayed_rows_with_mixed_panels_and_non_dyadic_units():
    """Two cases a skeptic broke: a delayed row (0.5) with unit 0.25 panels ending at unit_range 2 and
    coarser panels beyond (the map's panels must be a union of the action's shifted panels in both
    directions), and a delay of 0.3 on a 0.3 unit grid, where the shifted nodes land 1e-16 off the
    breakpoints and must still read the right side."""
    d = _ch3(window=6.0, nodes=8); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.5; d["numerics"].update(unit=0.25)
    d["agents"]["player1"]["loss"].append([0.05, "D1", "X@0.25"])
    r = ns.solve(d).check()
    assert r.representation_error["player2"] < 1e-9 and np.abs(r.kernel("D2")[r.ages < 0.5 - 1e-12]).max() == 0.0
    d = _ch3(window=12.0, nodes=8); d["agents"]["player2"]["signals"]["y2"]["delay"] = 0.3; d["numerics"].update(unit=0.3, unit_range=2.4)
    r = ns.solve(d).check()
    assert r.representation_error["player2"] < 1e-9 and np.abs(r.kernel("D2")[r.ages < 0.3 - 1e-12]).max() == 0.0


def test_coarse_start_reaches_the_same_equilibrium_with_fewer_fine_evaluations():
    for path in ("ch3_two_player.yaml", "ch1_two_player_finite.yaml"):
        m = ns.load(os.path.join(EX, path))
        cold = ns.make_solver(m).solve().check(); warm = ns.make_solver(m).solve(start="coarse").check()
        assert warm.evaluations < cold.evaluations and "coarse start" in warm.message
        assert max(np.abs(cold.maps[k] - warm.maps[k]).max() for k in cold.maps) < 1e-7
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml")); r.refine()
    assert r.refinement["resolved"]                                            # refine warm-starts from this result
