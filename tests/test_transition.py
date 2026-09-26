"""The spectral finite engine started from a known past (stage 1: the game still ends at T).

In order: (1) a zero past is the finite engine bit for bit; (2) the Chapter 3 stationary equilibrium as
its own past is reproduced on the strip away from the end (the full fixed point under NOISESTATE_SLOW=1;
one best response from the stationary maps in the fast suite); (3) a prior on the state of the discounted
one-agent model against the closed form (the Kalman filter from P(0) = P0, or from 0 when the prior is
observed at once); (4) an unobserved prior's column is the scaled state-noise column; (5) with the
stationary continuation (the maps frozen at the stationary ones on a buffer [T, T + L]) the same-model
identity is exact on the whole region; (6) a precision change on Chapter 3 starts from the old kernels and
is continued by its new stationary equilibrium, with its costs and `settled` pinned; (7) an old row's noise
loading on a channel the new row drops is kept on the band; (8) the mean paths start from the past's means,
`initial: 0` overriding; (9) a model with a control lag and a delayed row (the map in raw age) as its own past and
continuation is exact on every node; (10) an agent's own lagged control read across the band, in a loss cross
term and as the only coercive term, and the two-firm Chapter 5 market, each as its own past and continuation,
return the stationary maps in one best response; (11) the validation errors.
"""
import json
import os
import numpy as np, pytest
from scipy.integrate import solve_ivp
import noisestate as ns
from noisestate import engines
from noisestate.diagnostics import Status
from noisestate.past import Past
from helpers import (IVP, A1, H1, R1, T1, P0, prior_model as model, slow, slow_param, example, stationary,
                     same_model_solver, same_model_setup, two_firm_market, strip_maps, stationary_on_strip,
                     one_shot_from_the_stationary_maps, one_shot_deviation)

a, h, r = A1, H1, R1            # the discounted one-agent model's constants (helpers.prior_model)


@pytest.mark.parametrize("name", ["ch1_two_player_finite", slow_param("ch1_delayed_finite", reason="slow (6 s; the undelayed case is the fast one); set NOISESTATE_SLOW=1")])
def test_zero_past_is_the_finite_engine_bit_for_bit(name):
    """past=None takes the finite engine's own code path: Z, maps, costs and the evaluation count are
    identical (np.array_equal), and the grid is the same cached object."""
    m = example(name)
    r0 = ns.solve(m); r1 = ns.solve(m, past=None)
    assert np.array_equal(r0.world, r1.world)
    assert all(np.array_equal(r0.maps[a], r1.maps[a]) for a in r0.maps)
    assert r0.costs == r1.costs and r0.evaluations == r1.evaluations
    assert r1.grid is r0.grid and engines.spectral(m, past=None).c.g is r0.grid
    assert r1.past is None and r1.settled is None and "past" not in r1.to_dict()


@slow("slow (40 s); set NOISESTATE_SLOW=1")
def test_same_model_stationary_past_reproduces_the_stationary_kernels():
    """Chapter 3 (window L = 3) as its own past on a strip of T = 12, 16 nodes per side, from a coarse start
    (22 evaluations, 40 s): on every node with t < T - 3L the kernels equal K_stat(t - s) on both shock
    families and both agents (measured 1.4e-9 on X, 1.15e-8 on D1 and 1.2e-8 on D2).  The 1e-8 floor at
    t -> 3 is not discretisation but the end reaching back to T - 3L: at 24 nodes per side the interior
    corner drops to 9e-11 while t = 3 stays at 1.3e-8; the identity holds to 1e-10 for t <= 2.5.  The
    end reaches back 3L, not L: the maps within L of T carry the end, the FOC of every shock alive then
    (born after T - 2L) with them, and the agent's own re-optimised end leaks back through the
    forward-backward FOC system at the closed-loop rate: measured 4e-6 on [T - 3L, T - 2L), 1.4e-3 on
    [T - 2L, T - L), 1 at T (the control vanishes there)."""
    m = example("ch3_two_player"); L, T = 3.0, 12.0
    stat = stationary(m, 16)
    res = ns.solve(m.with_finite(T).with_numerics(nodes=16), past=stat, tol=1e-10, start_policy="coarse").require_converged()
    g = res.grid; gs = stat.compiled.grid
    assert g.L == L and g.T == T and int(g.upper.sum()) == 256 and res.compiled.N == 1280
    worst = {"interior": 0.0, "next": 0.0, "end": 0.0}
    for name in res.compiled.prim:
        K = res.kernel(name); Ks = gs.interp(g.a) @ stat.kernel(name)
        dev = np.abs(K - Ks).max(axis=1) / np.abs(Ks).max()
        worst["interior"] = max(worst["interior"], dev[g.t < T - 3 * L - 1e-9].max())
        worst["next"] = max(worst["next"], dev[(g.t >= T - 3 * L - 1e-9) & (g.t < T - 2 * L - 1e-9)].max())
        worst["end"] = max(worst["end"], dev[g.t >= T - 2 * L - 1e-9].max())
    assert worst["interior"] < 2e-8, worst
    assert worst["next"] < 2e-5 and worst["end"] > 1e-2, worst      # the end effect, and only there
    assert res.past is stat or res.past.source is stat
    assert res.to_dict()["past"]["kind"] == "stationary"


def test_same_model_best_response_returns_the_stationary_maps_away_from_the_end():
    """The content of the identity in one evaluation: Chapter 3 as its own past (T = 12, 16 nodes), each agent's
    best response to the stationary maps carried onto the strip returns the projected map itself on the first
    window to 5e-10 (player1) and 1.2e-8 (player2), the band included, and to 1.5e-7 on [3, 6); the end leaks
    back through the one-shot FOC system at the closed-loop rate (1.2e-4 and 2.7e-4 on [6, 9), the control
    vanishing at T)."""
    m = example("ch3_two_player"); L, T = 3.0, 12.0
    stat, solver = same_model_setup(m, T, 16, continuation=False)
    g = solver.c.g; maps = strip_maps(stat, solver)
    for a in m.agents:
        gm, out = solver.best_response(a, maps)
        dev = np.abs(gm - maps[a.name]).max(axis=(0, 1)) / np.abs(maps[a.name]).max()
        wins = [dev[(g.t >= lo - 1e-9) & (g.t < lo + L - 1e-9)].max() for lo in np.arange(0.0, T, L)]
        assert wins[0] < 5e-8 and dev[g.upper].max() < 5e-8, (a.name, wins)
        assert wins[1] < 1e-6 and wins[2] < 1e-3 and wins[3] > 0.5, (a.name, wins)


# ------------------------------------------------ the discounted one-agent model with a prior (helpers.prior_model)
def exact(rho, informed):
    """The closed form of tests/test_finite_discount.py with X(0) ~ N(0, P0): the Kalman filter starts at
    P(0) = P0 (unobserved prior, xhat(0) = 0) or at P(0) = 0 (the prior is observed at once, xhat(0) = X(0)),
    Sigma = E xhat^2 starts at 0 or P0, and the prior's own kernels are the (X, xhat) closed loop from
    (sqrt(P0), 0) or (sqrt(P0), sqrt(P0)) at time 0."""
    Sb = solve_ivp(lambda t, y: [rho * y[0] - 2 * a * y[0] + y[0] ** 2 / r - 1], (T1, 0.0), [0.0], dense_output=True, **IVP)
    Pf = solve_ivp(lambda t, y: [2 * a * y[0] - h * h * y[0] ** 2 + 1], (0.0, T1), [0.0 if informed else P0], dense_output=True, **IVP)
    K = lambda t: float(Sb.sol(t)[0]) / r; P = lambda t: float(Pf.sol(t)[0]); G = lambda t: P(t) * h
    f = solve_ivp(lambda t, y: [2 * (a - K(t)) * y[0] + G(t) ** 2, np.exp(-rho * t) * ((1 + r * K(t) ** 2) * y[0] + P(t))],
                  (0.0, T1), [P0 if informed else 0.0, 0.0], **IVP)

    def kernel(name, channel, t, s):
        out = np.zeros(len(t))
        for si in np.unique(s):
            y0 = {"w0": [1.0, 0.0], "w1": [0.0, G(si)], "xi": [np.sqrt(P0), np.sqrt(P0) if informed else 0.0]}[channel]
            sol = solve_ivp(lambda u, y: [a * y[0] - K(u) * y[1], G(u) * h * y[0] + (a - K(u) - G(u) * h) * y[1]],
                            (si, T1), y0, dense_output=True, **IVP)
            for i in np.flatnonzero(s == si):
                X, xhat = sol.sol(t[i]); out[i] = X if name == "X" else -K(t[i]) * xhat
        return out

    return float(f.y[1, -1]), kernel


TS = np.array([0.5, 1.0, 1.0, 2.0, 2.0, 2.8, 2.8]); SS = TS - np.array([0.1, 0.1, 0.5, 0.5, 1.5, 0.1, 2.0])
T0 = np.array([0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


@pytest.mark.parametrize("rho", [0.0, 0.5])
@pytest.mark.parametrize("informed", [False, True])
def test_prior_start_matches_the_discounted_closed_form(rho, informed):
    """X(0) ~ N(0, P0) as an initial shock (state load sqrt(P0)); unobserved, the engine's projection is the
    Kalman filter from P(0) = P0; with the row loading 1 the agent knows X(0) and the filter starts at 0.
    16 nodes: the cost within 1e-7 (measured 3.6e-8 and 2.3e-8 unobserved, 1e-10 observed) and the kernels
    of X and D on the prior's column and on the channels within 1e-5 (measured 1.7e-6 and 9.7e-7).  The
    prior's initial gain P0 h makes the early transient sharper than the P(0) = 0 start the finite engine
    pins at 12 nodes: there the cost is 2.5e-6 off and the kernels 6e-5, at 20 nodes 6e-10 and 6e-8."""
    J, kernel = exact(rho, informed)
    shock = {"name": "xi", "loads": {"X": np.sqrt(P0)}, **({"rows": {"a.y": 1.0}} if informed else {})}
    res = ns.solve(model(rho, 16), past=[shock]).require_converged()
    assert res.shocks == ["w0", "w1", "xi"] and res.maps["a"].shape == (1, 1, res.compiled.N + res.compiled.Nt)
    assert abs(res.costs["a"] - J) < 1e-7, (res.costs["a"], J)
    assert (res.diagnostics.statuses["resolution"] is Status.PASSED)
    for name in ("X", "D"):
        assert np.abs(res.evaluate(name, "xi", T0, np.zeros_like(T0)) - kernel(name, "xi", T0, np.zeros_like(T0))).max() < 1e-5, name
        for ch in ("w0", "w1"):
            assert np.abs(res.evaluate(name, ch, TS, SS) - kernel(name, ch, TS, SS)).max() < 1e-5, (name, ch)
    d = res.to_dict(); json.dumps(d)
    assert d["past"]["kind"] == "initial" and d["past"]["initial"][0]["name"] == "xi" and "xi" in d["kernels"]["X"]
    assert d["options"]["solver"]["past"]["window"] == 0.0 and len(d["agents"]["a"]["signals"]["y"]["map_init_time"]) == res.compiled.Nt


def test_unobserved_prior_column_is_the_scaled_state_noise_column():
    """A prior on X with no row loading is a shock nobody sees: its column is sqrt(P0) times the column of
    the channel loading X with 1 (w0, in no row), on every node, to round-off."""
    res = ns.solve(model(0.5), past=[{"name": "xi", "loads": {"X": np.sqrt(P0)}}]).require_converged()
    for name in ("X", "D"):
        assert np.abs(res.kernel(name, "xi") - np.sqrt(P0) * res.kernel(name, "w0")).max() < 1e-13


@slow("slow (12 s at 16 nodes; the 6-node copy is tests/test_finite_free.py's); set NOISESTATE_SLOW=1")
def test_same_model_continuation_is_exact_on_the_whole_region():
    """Chapter 3 as its own past and its own continuation (T = 6, L = 3, 16 nodes): with the maps frozen at the
    stationary ones on the buffer [6, 9] and the first-order condition's continuation running to T + L
    through the envelope responses, the identity has no end effect.  One best response from the stationary
    maps returns them on every node (5.2e-10 player1, 1.2e-8 player2, the latter the 16-node floor of the
    band, 1.2e-10 and 6.5e-10 on the last window); the fixed point started from them stays (3 evaluations),
    every kernel equal to K_stat(t - s) on every node, the last window and the buffer included (X 1.9e-9,
    D1 1.4e-9, D2 1.5e-8 on the band and the buffer, 6e-10 on the last window), and res.settled is 6.6e-10;
    under NOISESTATE_SLOW, from zero (12 nodes, 29 evaluations, 10 s) the same fixed point to the 12-node
    floor (1.3e-5), settled 6.3e-7.  The costs are over [0, T]; the buffer's are reported apart."""
    m = example("ch3_two_player"); L, T = 3.0, 6.0
    stat, solver = same_model_setup(m, T, 16)
    c = solver.c; g = c.g; gs = stat.compiled.grid
    assert g.T == T + L and c.T == T and c.Tg == T + L and c.P_T == 2 and int(c.buffer.sum()) == 512 and c.N == 1280
    assert [float(b) for b in g.bp] == [0.0, 3.0, 6.0, 9.0] and g.upper[g.s < -1e-12].all() and not g.upper[g.s > 1e-12].any()
    maps = strip_maps(stat, solver)
    last = ~g.upper & ~c.buffer & (g.t >= T - L - 1e-9)
    for a in m.agents:
        gm, out = solver.best_response(a, maps)
        dev = np.abs(gm - maps[a.name]).max(axis=(0, 1)) / np.abs(maps[a.name]).max()
        assert dev.max() < 5e-8 and dev[last].max() < 5e-9, (a.name, dev.max(), dev[last].max())
    res = solver.solve(start_from=maps, tol=1e-10).require_converged()
    assert res.evaluations <= 4 and res.settled < 1e-8 and res.continuation is stat
    for name in c.prim:
        dev = np.abs(res.kernel(name) - gs.interp(g.a) @ stat.kernel(name)).max(axis=1) / np.abs(stat.kernel(name)).max()
        assert dev.max() < 5e-8 and dev[last].max() < 5e-9 and dev[c.buffer].max() < 5e-8, (name, dev.max())
    parts = res.cost_parts["player1"]
    assert set(parts) == {"variance", "mean", "continuation"} and abs(parts["continuation"] - 3 * stat.costs["player1"]) < 1e-6
    assert res.costs["player1"] == parts["variance"] and 0 < res.costs["player1"] < 2 * parts["continuation"]
    rows = {r["name"]: r for r in res.diagnostics.rows}
    assert rows["settled"]["ok"] and rows["settled"]["threshold"] == 1e-4 and "continuation window" in rows and "past window" in rows
    assert res.to_dict()["continuation"]["window_tail"] == res.past.provenance["window_tail"] and res.grid_summary()["buffer"] == [6.0, 9.0]
    if not os.environ.get("NOISESTATE_SLOW"):
        return
    stat12 = stationary(m, 12)
    r0 = ns.solve(m.with_finite(T).with_numerics(nodes=12), past=stat12, continuation=stat12, tol=1e-10).require_converged()
    g0 = r0.grid; gs0 = stat12.compiled.grid
    for name in c.prim:
        assert np.abs(r0.kernel(name) - gs0.interp(g0.a) @ stat12.kernel(name)).max() / np.abs(stat12.kernel(name)).max() < 5e-5, name
    assert r0.settled < 5e-6


@slow("slow (19 s at 12 nodes; the 8-node copy is tests/test_transition_api.py's fixture); set NOISESTATE_SLOW=1")
def test_regime_change_on_chapter_3_starts_from_the_old_kernels():
    """Chapter 3 with the past at p1 = 3 and the new model at p1 = 10, continued by the new model's stationary
    equilibrium (T = 6, 12 nodes): the state kernels at 0+ are the old kernels at age -s to round-off,
    past=Model equals past=StationaryResult bit for bit, continuation="stationary" is the new model solved at
    horizon.nodes, and the solve converges.  The costs over [0, T] are pinned at their 12-node values
    (2.55448876, 2.55908247; 2.56081481 and 2.56540852 when the game ended at T; they move by 4.2e-5 and
    4.9e-5 at 20 nodes, so 1e-6 catches a change of the formulation and not of the grid), the buffer's cost
    is three times the stationary flow (1.26523081), and `settled` is 4.8e-4 (5.3e-4 at 20 nodes: the
    transition has not settled by T - L = 3, the closed loop decaying by 1e-2 per unit of t), so the guard
    (settled_tol 1e-4) fires; at T = 9 it is 2.7e-6 and the guard is quiet; at T = L = 3 it is 0.58.  The
    past=Model identity is checked at 8 nodes (1.4 s a solve).
    The resolution guard says where its error sits: player2's 8.6e-4 is on the band's tip (3.5e-4 at 20
    nodes: the collapsing corner's floor, not resolution), player1's 4.3e-5 in the interior, the transient
    at 0+ (2.1e-7 at 20 nodes: resolution)."""
    m = example("ch3_two_player")
    old = ns.solve(m.with_params(p1=3.0)).require_converged()
    new = m.with_params(p1=10.0).with_finite(6.0).with_numerics(nodes=12)
    res = ns.solve(new, past=old, continuation="stationary").require_converged()
    assert abs(res.costs["player1"] - 2.55448876) < 1e-6 and abs(res.costs["player2"] - 2.55908247) < 1e-6, res.costs
    assert abs(res.cost_parts["player1"]["continuation"] - 1.26523081) < 1e-6
    assert 4e-4 < res.settled < 6e-4 and not next(r for r in res.diagnostics.rows if r["name"] == "settled")["ok"], res.settled
    assert "TRANSITION NOT SETTLED" in res.summary() and "against settled_tol 0.0001" in res.summary()
    cont = res.continuation
    assert cont.model.horizon.kind == "stationary" and cont.compiled.grid.L == 3.0 and cont.compiled.grid.n == 12
    assert res.solver_kw["continuation"] is cont and res.to_dict()["continuation"]["kind"] == "stationary"
    for a in ("player1", "player2"):
        parts = res.representation_parts[a]
        assert set(parts) == {"interior", "band tip", "last window", "buffer"} and max(parts.values()) == res.representation_error[a]
    p2 = res.representation_parts["player2"]
    assert p2["band tip"] > 10 * max(p2["interior"], p2["last window"], p2["buffer"]) and p2["band tip"] > 5e-4, p2
    assert res.representation_parts["player1"]["interior"] < 1e-4
    flag = next(r for r in res.diagnostics.rows if r["name"] == "resolution")["flag"]
    assert "band tip" in flag and "last window" in flag and "interior" in flag
    assert "representation_parts" in res.to_dict()
    g = res.grid; at0 = g.upper & (g.t < 1e-12)
    assert at0.any()
    for name in ("X",):
        assert np.abs(res.kernel(name)[at0] - old.compiled.grid.interp(g.a[at0]) @ old.kernel(name)).max() < 1e-13
    new8 = new.with_numerics(nodes=8)
    res8 = ns.solve(new8, past=old, continuation="stationary")
    res2 = ns.solve(new8, past=m.with_params(p1=3.0), continuation=res8.continuation)
    assert np.array_equal(res8.world, res2.world) and res8.costs == res2.costs and 6e-4 < res8.settled < 7e-4
    assert all(np.array_equal(res8.maps[k], res2.maps[k]) for k in res8.maps)
    assert res.evaluate("X", "w0", np.array([1.0, 1.0]), np.array([-1.0, 0.5])).shape == (2,)
    assert res.second_order["player1"]["ok"] and res.solver_kw["past"] is res.past
    short = ns.solve(new.with_finite(3.0), past=old, continuation=cont).require_converged()   # a shorter T
    assert short.settled > 0.5 and "TRANSITION NOT SETTLED" in short.summary()


def test_old_noise_loading_on_a_channel_the_new_row_drops_is_kept_on_the_band():
    """The past's row y1 loads w1 and w2 (0.3), the new row y1 loads w1 only: the increments observed before zero
    carry the old loading, so under the stationary maps the closed loop at 0+ returns the old kernels on every
    channel (w2 included: without the union of the two regimes' loadings the entry is dropped and D1's kernel on
    w2 is off by 0.3 g(a), a third of its size) to round-off, the past and the strip sharing their 16 nodes."""
    m = example("ch3_two_player")
    d = m.to_dict(); d["agents"]["player1"]["signals"]["y1"]["noise"] = {"w1": 1.0, "w2": 0.3}
    old = stationary(ns.Model.from_dict(d), 16)
    solver = same_model_solver(m, old, 3.0, 16, continuation=False)
    c = solver.c; g = c.g; gs = old.compiled.grid
    blocks, deltas = c.row_blocks("player1", 0, set())
    assert deltas == {"w1": [(0.0, 1.0)], "w2": [(0.0, 0.0)]}
    assert np.array_equal(c.noise_weight("player1", 0, 2, 0.0), np.where(g.upper, 0.3, 0.0))
    Z = c.closed_loop(strip_maps(old, solver))
    at0 = g.upper & (g.t < 1e-12)
    for name in c.prim:
        K = Z[c.block(name)][at0]; Ks = gs.interp(g.a[at0]) @ old.kernel(name)
        assert np.abs(K - Ks).max() < 1e-12, name
    assert np.abs(Ks[:, 2]).max() > 0.1                           # D2's kernel on w2: the old row's loading is seen


def test_mean_paths_start_from_the_past_means_and_initial_overrides():
    """Chapter 3 with a target for player1 (loss term -2 X), T = L = 3, 12 nodes, its own stationary solution as
    the past: Xbar(0) is the old mean (0.86621649, the initial condition), Dbar1(0+) = 1.80018873 against the
    old constant 1.79893500 (the horizon's end is L away), both controls vanish at T, and the paths are
    pinned at their 12-node values (16 nodes moves them by up to 8e-6).  `initial: 0` on X overrides the
    past's mean: Xbar(0) = 0 and the whole path moves."""
    m = example("ch3_two_player")
    d = m.to_dict(); d["agents"]["player1"]["loss"].append([-2.0, "X"])
    mt = ns.Model.from_dict(d)
    statt = stationary(mt, 16)
    assert abs(statt.means["X"] - 0.86621649) < 1e-8 and abs(statt.means["D1"] - 1.79893500) < 1e-8
    res = ns.solve(mt.with_finite(3.0).with_numerics(nodes=12), past=statt).require_converged()
    at = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert abs(res.mean("X", 0.0)[0] - statt.means["X"]) < 1e-12
    assert np.abs(res.mean("X", at) - [0.86621649, 0.85546733, 0.86347325, 0.91547552, 0.95468024]).max() < 1e-6
    assert np.abs(res.mean("D1", at) - [1.80018873, 1.66937965, 1.49563855, 0.94540394, 0.0]).max() < 1e-5
    assert np.abs(res.mean("D2", at) - [-1.84477844, -1.67009829, -1.46449802, -0.8821671, 0.0]).max() < 1e-5
    assert abs(res.mean("D1", 3.0)[0]) < 1e-12 and abs(res.mean("D2", 3.0)[0]) < 1e-12
    assert abs(res.cost_parts["player1"]["mean"] + 1.82696694) < 1e-6 and abs(res.cost_parts["player2"]["mean"] - 3.46545099) < 1e-6
    d0 = mt.to_dict(); d0["states"]["X"]["initial"] = 0.0
    m0 = ns.Model.from_dict(d0)
    assert m0.states[0].initial == 0.0 and m0.to_dict()["states"]["X"]["initial"] == 0.0 and mt.states[0].initial is None
    r0 = ns.solve(m0.with_finite(3.0).with_numerics(nodes=12), past=statt).require_converged()
    assert abs(r0.mean("X", 0.0)[0]) < 1e-12
    assert np.abs(r0.mean("X", at) - [0.0, 0.39555973, 0.61918718, 0.84513578, 0.92167413]).max() < 1e-6
    assert abs(r0.mean("D1", 0.0)[0] - 2.31569976) < 1e-5
    assert np.array_equal(r0.world, res.world)                             # the kernels do not depend on the means


@slow("slow (33 s; the 3-node copy is tests/test_unit_range_finite.py's transition); set NOISESTATE_SLOW=1")
def test_delayed_rows_with_a_past_reproduce_the_stationary_maps():
    """examples/ch1_delayed_finite.yaml (the controls act after tau = 0.25, player2 sees its row with delay tau)
    with its own stationary equilibrium at window L = 1 as past and continuation, T = 1.25: with a past a delayed
    row's map is stored in raw age, masked below the delay, and the squares above each diagonal are split
    (56 pieces), so the identity is exact: at 7 nodes one best response from the stationary maps returns them
    on every node to 5.1e-8 (player1) and 4.9e-8 (player2, the delayed row), the band and the last window
    included (10 s; at 8 nodes, under NOISESTATE_SLOW, 1.4e-9 and 1.8e-9 in 20 s); at 6 nodes (the floor
    1.2e-6) the fixed point started from them stays in 4 evaluations, every kernel K_stat(t - s) to that
    floor, and res.settled is 1.1e-6.  Under NOISESTATE_SLOW the fixed point from zero reaches the same maps
    (20 evaluations, 66 s)."""
    m = example("ch1_delayed_finite"); L, T, d = 1.0, 1.25, 0.25
    full = bool(os.environ.get("NOISESTATE_SLOW")); n1, pin = (8, 1e-8) if full else (7, 2e-7)
    stat8 = stationary(m, n1, L)
    solver = same_model_solver(m, stat8, T, n1)
    c = solver.c; g = c.g
    assert len(g.pieces) == 56 and c.N == 56 * n1 * n1 and c.row_delays == {"player1": [0.0], "player2": [d]} and c.rows["player2"][0][3] == 0.0
    assert g.upper[g.s < -1e-12].all() and not g.upper[g.s > 1e-12].any() and not g.above[g.upper].all()   # split squares
    keep2 = solver._identified(m.agents[1]).reshape(1, -1)[0, :c.N]
    assert not keep2[g.a1 <= d + 1e-9].any() and keep2[(g.a0 >= d - 1e-9) & ~c.buffer].all()   # whole pieces below the delay
    assert np.abs(c.frozen["player2"][0, 0][g.a1 <= d + 1e-9]).max() == 0.0
    for a in m.agents:
        gm, out = solver.best_response(a, c.frozen)
        keep = solver._identified(a).reshape(len(a.signals), -1)[:, :c.N]
        dev = (np.abs(gm - c.frozen[a.name])[0] * keep / np.abs(c.frozen[a.name]).max())[0]
        assert dev.max() < pin, (a.name, dev.max(), dev[g.upper].max())
    stat = stationary(m, 6, L)
    solver = same_model_solver(m, stat, T, 6)
    c = solver.c; g = c.g
    res = solver.solve(start_from=c.frozen, tol=1e-10).require_converged()
    assert res.evaluations <= 6 and res.settled < 5e-6
    for name in c.prim:
        dev = np.abs(res.kernel(name) - stationary_on_strip(stat, g, name)).max() / np.abs(stat.kernel(name)).max()
        assert dev < 5e-6, (name, dev)
    assert np.abs(res.maps["player2"][0, 0][g.a1 <= d + 1e-9]).max() == 0.0
    ax = res.map_axes(d)
    assert ax["map_time"] == g.t.tolist() and ax["map_age"] == g.a.tolist() and "raw age" in res.MAP_CONVENTION
    assert json.loads(json.dumps(res.to_dict()))["agents"]["player2"]["signals"]["y2"]["delay"] == d
    if full:
        r0 = solver.solve(tol=1e-10).require_converged()
        assert np.abs(r0.world - res.world).max() < 1e-8 and r0.settled < 5e-6


def test_solve_routes_past_and_continuation_and_the_result_rebuilds_them():
    """noisestate.solve(model, past=, continuation=) hands both to the spectral finite engine, which records them
    in solver_kw: a solver rebuilt from the result (refine(), stability()) carries the same Past and the same
    StationaryResult; the payload's options.solver and its top level carry the continuation's provenance
    (kind, window, nodes, costs, window_tail), not its kernels, and it serialises."""
    m = example("ch3_two_player")
    stat = stationary(m, 8)
    res = ns.solve(m.with_finite(6.0).with_numerics(nodes=8), past=stat, continuation="stationary", tol=1e-8).require_converged()
    assert res.solver_kw["past"] is res.past and res.solver_kw["continuation"] is res.continuation
    assert res.continuation is not stat and np.array_equal(res.continuation.world, stat.world)      # solved on the fly at horizon.nodes
    fine = res._make_solver(res.model.with_numerics(nodes=12))
    assert fine.c.past is res.past and fine.c.cont is res.continuation and fine.c.Tg == 9.0
    d = json.loads(json.dumps(res.to_dict()))
    for prov in (d["continuation"], d["options"]["solver"]["continuation"]):
        assert prov["kind"] == "stationary" and prov["window"] == 3.0 and prov["nodes"] == 8 and "kernels" not in prov
        assert abs(prov["window_tail"] - res.continuation.window_tail) < 1e-12 and set(prov["costs"]) == {"player1", "player2"}
    assert d["settled"] == res.settled and d["options"]["solver"]["past"]["kind"] == "stationary"


def chapter3_with_lag(kind, nodes, L=1.5, lag=0.5):
    """Chapter 3 with player1's own control read lag behind: in a loss cross term with the state (`loss`), or as the
    only coercive term (`onlylag`: the instantaneous quadratic 0.01, the lag's 0.5 r1, the Chapter 5 structure)."""
    d = example("ch3_two_player").to_dict()
    if kind == "loss":
        d["agents"]["player1"]["loss"].append([0.3, f"D1@{lag}", "X"])
    else:
        d["agents"]["player1"]["loss"] = [[0.5, "X", "X"], ["0.5*r1", f"D1@{lag}", f"D1@{lag}"], [0.01, "D1", "D1"]]
    d["horizon"] = {"kind": "stationary", "discount": 0.0, "window": L}; d["numerics"] = {"nodes": nodes, "unit": lag, "unit_range": L}
    return ns.Model.from_dict(d)


@pytest.mark.parametrize("kind, nodes, floor", [("loss", 6, (1e-5, 5e-5)),
                                                slow_param("onlylag", 8, (5e-5, 5e-6), reason="slow (6 s; the 6-node loss case is the fast one); set NOISESTATE_SLOW=1")])
def test_own_lagged_control_read_across_the_band_returns_the_stationary_maps(kind, nodes, floor):
    """Player1's own control lagged by 0.5 (unit 0.5, L = 1.5, T = 2), Chapter 3 as its own past and continuation:
    the first-order condition reads D1(t - lag, a - lag) through the band and the buffer, whose corner nodes sit
    on a diagonal or at age exactly L, where the read must take the side the reader's own sides imply.  One
    best response from the stationary maps returns them on every node to the grid's closed-loop floor: with
    the lag in a loss cross term (6 nodes) 2.6e-6 on the band, 2.0e-6 in the interior and 1.2e-6 on the last
    window for player1 (the closed loop itself sits 4e-6 from the stationary kernels there), 2.0e-5 for player2
    (its floor 1.4e-5); with the lag as the only coercive term (8 nodes: at 6 the map is weakly identified and
    sits at 1e-3) 1.4e-5 on the band and 1.4e-8 in the interior for player1, 4.6e-7 for player2.  Before the
    read took its side from side_d at those corners, the band's deviation was 3e-2."""
    m = chapter3_with_lag(kind, nodes)
    stat = ns.solve(m).require_converged()
    dev, solver = one_shot_from_the_stationary_maps(m, stat, 2.0, nodes)
    g = solver.c.g
    assert [float(b) for b in g.bp] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] and len(g.pieces) == 33
    assert dev["player1"].max() < floor[0] and dev["player2"].max() < floor[1], {k: v.max() for k, v in dev.items()}
    assert dev["player1"][g.upper].max() < floor[0] and dev["player1"][~g.upper & (g.t >= 0.5 - 1e-9)].max() < floor[0]


@pytest.mark.parametrize("T", [2.0, 3.0])
def test_two_firm_market_as_its_own_transition_returns_the_stationary_maps(T):
    """examples/make_ch5_cycle_market.build(N=2) (tau = 1, L = 2, rP = 0.1; the ties and the linear terms
    dropped, 5 nodes) as its own past and continuation: the firms' prices and orders enter each other's rows
    and losses a lag behind, so every first-order condition reads controls across the band.  One best response
    from the stationary maps returns them to 2.3e-4 on every node at T = 2 and T = 3, the floor of the 5-node
    closed loop itself (1.4e-4 on P, 2.5e-4 on o against the stationary kernels); pinned at 5e-4."""
    m = two_firm_market(tau=1.0, L=2.0, nodes=5, unit_range=2.0, rP=0.1)
    stat = ns.solve(m).require_converged()
    dev, solver = one_shot_from_the_stationary_maps(m, stat, T, 5, unit=1.0)
    assert len(solver.c.g.pieces) == {2.0: 14, 3.0: 16}[T] and solver.c.N == {2.0: 350, 3.0: 400}[T]
    assert all(v.max() < 5e-4 for v in dev.values()), {k: v.max() for k, v in dev.items()}


def test_past_validation():
    """A finite result is a TypeError, an unconverged stationary one a ValueError, other channels a ValueError,
    an initial shock on an unknown state a ValueError, and a past with a window on a model whose rows are
    observed with a delay is not implemented in this stage."""
    m = example("ch3_two_player")
    with pytest.raises(TypeError, match="stationary"):
        Past.of(ns.solve(m.with_finite(1.0).with_numerics(nodes=6)))
    with pytest.raises(ValueError, match="did not converge"):
        Past.of(ns.solve(m, max_evaluations=1))
    stat = ns.solve(m).require_converged()
    other = m.to_dict(); other["shocks"] = ["w0", "w1", "w9"]
    other["agents"]["player2"]["signals"]["y2"]["noise"] = {"w9": 1.0}
    with pytest.raises(ValueError, match="shocks"):
        engines.spectral(ns.Model.from_dict(other).with_finite(3.0).with_numerics(nodes=6), past=stat)
    with pytest.raises(ValueError, match="not a state"):
        engines.spectral(m.with_finite(3.0).with_numerics(nodes=6), past=[{"name": "v", "loads": {"V": 1.0}}])
    delayed = example("ch1_delayed_finite")
    dpast = stationary(delayed, 6, 1.0)
    with pytest.raises(ValueError, match="differ from the model's .* \\(agent, row, delay\\)"):
        engines.spectral(delayed, past=dpast, continuation=ns.solve(ns.Model.from_dict({**delayed.to_dict(), "agents": {
            **delayed.to_dict()["agents"], "player2": {**delayed.to_dict()["agents"]["player2"], "signals": {"y2": {"drift": {"X": "sqrt(p2)"}, "noise": {"w2": 1.0}}}}}}).with_stationary(1.0).with_numerics(nodes=6)).require_converged())
    fin = m.with_finite(3.0).with_numerics(nodes=6)
    with pytest.raises(TypeError, match="StationaryResult"):
        engines.spectral(fin, past=stat, continuation=ns.solve(fin))
    with pytest.raises(ValueError, match="window"):
        engines.spectral(fin, past=stat, continuation=ns.solve(m.with_stationary(2.0).with_numerics(nodes=6)).require_converged())
    short = engines.spectral(m.with_finite(2.0).with_numerics(nodes=6), past=stat, continuation=stat)   # T < L builds
    assert [float(b) for b in short.c.g.bp] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and int((short.c.g.upper & short.c.buffer).sum()) == 36
    with pytest.raises(ValueError, match="needs a past with a window"):
        engines.spectral(fin, past=[{"name": "v", "loads": {"X": 1.0}}], continuation=stat)
    with pytest.raises(ValueError, match="'stationary', 'end' or a StationaryResult"):
        engines.spectral(fin, past=stat, continuation="tail")
    with pytest.raises(ValueError, match="shocks"):
        engines.spectral(fin, past=stat, continuation=ns.solve(ns.Model.from_dict(other).with_numerics(nodes=6)).require_converged())
    from noisestate.triangle import TriangleGrid
    with pytest.raises(ValueError, match="age panels shifted"):
        TriangleGrid([0.0, 1.0, 2.0, 3.5], 4, 4, T=3.5, window=1.0, buffer=2.0)


def test_second_order_check_sees_the_initial_shock_columns():
    """The check's form includes a past's initial-shock columns under the point form of the line s = 0 (the
    quadrature expected_cost uses for them): on a one-agent problem with a prior seen at once, the point weights
    are no longer a zero direction of the form (min was exactly 0 with the columns skipped; now positive, as the
    finite differences of the cost along a point-weight direction say, +1.2).  The Kyle-Back prior's verdict is
    unchanged (its flagged direction has no weight on the point weights, the raw curvature -1.209e-3 at 8 nodes,
    eps 0.2, the same to 1e-7; the reported value is now relative to the prior column's larger curvature)."""
    d = {"shocks": ["w0"], "states": {"X": {"drift": {"X": -0.3, "D": 1.0}, "noise": {"w0": 1.0}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {}, "noise": {"w0": 1.0}}},
                          "loss": [[1.0, "X", "X"], [0.5, "D", "D"]]}},
         "numerics": {"nodes": 6},
         "horizon": {"kind": "transition", "T": 3.0, "discount": 0.0, "continuation": "end",
                     "past": {"initial": [{"name": "xi", "loads": {"X": 0.9}, "rows": {"a.y": 1.0}}]}}}
    res = ns.solve(d); so = res.second_order["a"]
    assert res.converged and so["ok"] and so["min"] > 0.0, so
    S = res._make_solver(res.model); c = S.c; agent = res.model.agents[0]; gam = res.maps["a"]
    v = np.zeros_like(gam); v[0, 0, c.N:] = 1.0                                  # a direction on the point weights only
    J = lambda h: S.expected_cost(agent, c.closed_loop({"a": gam + h * v}))
    assert (J(1e-2) + J(-1e-2) - 2 * J(0.0)) / 1e-4 > 1.0


@pytest.mark.parametrize("T", [slow_param(1.0), 1.5])
def test_same_model_identity_holds_below_the_window(T):
    """A horizon shorter than the past's window: Chapter 3 (L = 3) as its own past and continuation at T = 1 (one
    unit) and T = 1.5 (L/2), 8 nodes.  The strip is the rectangle [0, T] x [0, L]; the old shocks are still alive
    on the buffer, whose pieces above the line s = 0 carry the band (the cuts below L closed under the shift by
    T so the buffer's panels are the age panels shifted; a node on a diagonal that is not its own piece's reads
    the side its piece lies on, side_ds).  One best response from the stationary maps returns them on every
    node, and the solve returns the stationary kernels on every node, the buffer and its band included, to the
    same floor as at T = 6 (8 nodes: one-shot 6e-4 / 2.8e-3, kernels 2.8e-3 at T = 6; pinned at 1.5x).  T = 1.5 is
    fast (4 s); T = 1, whose unit-cut strip has four panels, is gated (13 s)."""
    m = example("ch3_two_player")
    stat = stationary(m, 8)
    ref = same_model_solver(m, stat, 6.0, 8); rdev = one_shot_deviation(ref); rres = ref.solve(start_policy="stationary")
    solver = same_model_solver(m, stat, T, 8); c = solver.c; g = c.g
    assert c.T == T and g.L == 3.0 and c.buffer.any() and (g.upper & c.buffer).any()          # old shocks alive on the buffer
    assert [float(b) for b in g.bp] == ([0.0, 1.0, 2.0, 3.0, 4.0] if T == 1.0 else [0.0, 1.5, 3.0, 4.5])
    dev = one_shot_deviation(solver)
    for a in dev:
        assert dev[a].max() < 1.5 * rdev[a].max(), (a, dev[a].max(), rdev[a].max())
    res = solver.solve(start_policy="stationary")
    assert res.converged and res.evaluations <= 12 and res.settled < 1.5 * max(rdev[a].max() for a in rdev)
    for name in ("X", "D1", "D2"):
        S = stationary_on_strip(stat, g, name); Sr = stationary_on_strip(stat, ref.c.g, name)
        d = np.abs(res.kernel(name) - S).max(axis=1) / np.abs(S).max()
        dr = (np.abs(rres.kernel(name) - Sr).max() / np.abs(Sr).max())
        assert d.max() < 1.5 * max(dr, 1e-4) and d[c.buffer].max() < 1.5 * max(dr, 1e-4) and d[g.upper & c.buffer].max() < 1.5 * max(dr, 1e-4), (name, d.max(), dr)


# ------------------------------------------------ a terminal loss on a transition with a past
#  The prior start above with a terminal loss q_T X_T^2: the Riccati terminal condition S(T) = q_T, and the cost gains
#  e^{-rho T} q_T (Sigma(T) + P(T)).  The terminal term on the prior's column is read at the corner (T, T), the line
#  s = 0 at T.  Measured at 12, 16, 20 nodes (rho = 0.5, unobserved prior): cost 2.4e-7, 5.1e-8, 8.5e-10; the prior's
#  kernels 4.5e-5, 7.4e-7, 7.4e-8; the channels' 3.2e-3, 1.2e-4, 2.8e-5.
QT_TERM = 2.0


def exact_terminal(rho, informed):
    Sb = solve_ivp(lambda t, y: [rho * y[0] - 2 * a * y[0] + y[0] ** 2 / r - 1], (T1, 0.0), [QT_TERM], dense_output=True, **IVP)
    Pf = solve_ivp(lambda t, y: [2 * a * y[0] - h * h * y[0] ** 2 + 1], (0.0, T1), [0.0 if informed else P0], dense_output=True, **IVP)
    K = lambda t: float(Sb.sol(t)[0]) / r; P = lambda t: float(Pf.sol(t)[0]); G = lambda t: P(t) * h
    f = solve_ivp(lambda t, y: [2 * (a - K(t)) * y[0] + G(t) ** 2, np.exp(-rho * t) * ((1 + r * K(t) ** 2) * y[0] + P(t))],
                  (0.0, T1), [P0 if informed else 0.0, 0.0], **IVP)
    J = float(f.y[1, -1]) + np.exp(-rho * T1) * QT_TERM * (float(f.y[0, -1]) + P(T1))

    def kernel(name, channel, t, s):
        out = np.zeros(len(t))
        for si in np.unique(s):
            y0 = {"w0": [1.0, 0.0], "w1": [0.0, G(si)], "xi": [np.sqrt(P0), np.sqrt(P0) if informed else 0.0]}[channel]
            sol = solve_ivp(lambda u, y: [a * y[0] - K(u) * y[1], G(u) * h * y[0] + (a - K(u) - G(u) * h) * y[1]],
                            (si, T1), y0, dense_output=True, **IVP)
            for i in np.flatnonzero(s == si):
                X, xhat = sol.sol(t[i]); out[i] = X if name == "X" else -K(t[i]) * xhat
        return out
    return J, kernel


@pytest.mark.parametrize("informed", [False, True])
def test_a_terminal_loss_with_a_prior_is_the_riccati_terminal_condition(informed):
    """16 nodes, rho = 0.5: the cost to 2e-7 (measured 5.1e-8 unobserved, 3.5e-8 observed) and the kernels on the
    prior's column to 5e-6 (7.4e-7, 2.2e-9) and on the channels to 3e-4 (1.2e-4)."""
    J, kernel = exact_terminal(0.5, informed)
    d = model(0.5, 16); d["agents"]["a"]["terminal"] = [[QT_TERM, "X", "X"]]
    shock = {"name": "xi", "loads": {"X": np.sqrt(P0)}, **({"rows": {"a.y": 1.0}} if informed else {})}
    res = ns.solve(d, past=[shock]).require_converged()
    assert abs(res.costs["a"] - J) < 2e-7, (res.costs["a"], J)
    assert res.second_order["a"]["ok"]
    for name in ("X", "D"):
        assert np.abs(res.evaluate(name, "xi", T0, np.zeros_like(T0)) - kernel(name, "xi", T0, np.zeros_like(T0))).max() < 5e-6, name
        for ch in ("w0", "w1"):
            assert np.abs(res.evaluate(name, ch, TS, SS) - kernel(name, ch, TS, SS)).max() < 3e-4, (name, ch)


def test_a_terminal_loss_after_a_stationary_past():
    """The band: the one-agent model's stationary equilibrium (window L = 6) as the past of the same model on [0, 3]
    with q_T X_T^2 and the game ending at T, so old shocks are alive at T.  Closed form: the filter stays at its
    stationary P, E xhat^2 starts at the stationary G^2 / 2(K - a) under the old gain K, the Riccati runs back
    from S(T) = q_T; the window truncates at e^{-2 (K - a) L} ~ 2.5e-7.  At 12 nodes the cost is 5.6e-7 off (the
    transition's own floor here: 1.6e-6 with q_T = 0), and a terminal target's mean path is 3.6e-9 off, the two
    mean systems agreeing to rounding."""
    rho, T, L, B = 0.5, 3.0, 6.0, 1.0
    Sinf = max(np.roots([1 / r, rho - 2 * a, -1]).real); Kinf = Sinf / r
    Pinf = max(np.roots([-h * h, 2 * a, 1]).real); G = Pinf * h
    Sb = solve_ivp(lambda t, y: [rho * y[0] - 2 * a * y[0] + y[0] ** 2 / r - 1], (T, 0.0), [QT_TERM], dense_output=True, **IVP)
    S = lambda t: float(Sb.sol(t)[0]); K = lambda t: S(t) / r
    f = solve_ivp(lambda t, y: [2 * (a - K(t)) * y[0] + G * G, np.exp(-rho * t) * ((1 + r * K(t) ** 2) * y[0] + Pinf)],
                  (0.0, T), [G * G / (2 * (Kinf - a)), 0.0], **IVP)
    J = float(f.y[1, -1]) + np.exp(-rho * T) * QT_TERM * (float(f.y[0, -1]) + Pinf)
    ds = model(rho, 12); ds["horizon"] = {"kind": "stationary", "window": L, "discount": rho}
    stat = ns.solve(ds).require_converged()
    d = model(rho, 12); d["agents"]["a"]["terminal"] = [[QT_TERM, "X", "X"]]
    res = ns.solve(d, past=stat, continuation="end").require_converged()
    assert abs(res.costs["a"] - J) < 3e-6, (res.costs["a"], J)
    assert res.second_order["a"]["ok"]
    vb = solve_ivp(lambda t, y: [(rho - a + S(t) / r) * y[0]], (T, 0.0), [-QT_TERM * B], dense_output=True, **IVP)
    xb = solve_ivp(lambda t, y: [a * y[0] - (S(t) * y[0] + float(vb.sol(t)[0])) / r], (0.0, T), [0.0], dense_output=True, **IVP)
    d["agents"]["a"]["terminal"].append([-2 * QT_TERM * B, "X"])
    res = ns.solve(d, past=stat, continuation="end").require_converged()
    t = res.mean_times[res.mean_times <= T]
    assert np.abs(res.means["X"][:t.size] - xb.sol(t)[0]).max() < 1e-7
    S_ = res._make_solver(res.model)
    Md, bd = S_._mean_system_diag(res.maps); Ml, bl = S_._mean_system_line(res.maps)
    assert np.abs(np.linalg.solve(Md, bd) - np.linalg.solve(Ml, bl)).max() < 1e-12
