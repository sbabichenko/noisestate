"""The spectral finite engine started from a known past (stage 1: the game still ends at T).

In order: (1) a zero past is the finite engine bit for bit; (2) the Chapter 3 stationary equilibrium as
its own past is reproduced on the strip away from the end (the full fixed point under NOISESTATE_SLOW=1;
one best response from the stationary maps in the fast suite); (3) a prior on the state of the discounted
one-agent model against the closed form (the Kalman filter from P(0) = P0, or from 0 when the prior is
observed at once); (4) an unobserved prior's column is the scaled state-noise column; (5) a precision
change on Chapter 3 starts from the old kernels, with its costs pinned; (6) an old row's noise loading on a
channel the new row drops is kept on the band; (7) the mean paths start from the past's means, `initial: 0`
overriding; (8) the validation errors.
"""
import json
import os
import numpy as np, pytest
from scipy.integrate import solve_ivp
import noisestate as ns
from noisestate.past import Past

EX = ns.__file__.rsplit("/noisestate/", 1)[0] + "/examples/"
IVP = dict(method="DOP853", rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("example", ["ch1_two_player_finite", "ch1_delayed_finite"])
def test_zero_past_is_the_finite_engine_bit_for_bit(example):
    """past=None takes the finite engine's own code path: Z, maps, costs and the evaluation count are
    identical (np.array_equal), and the grid is the same cached object."""
    m = ns.load(EX + example + ".yaml")
    r0 = ns.solve(m); r1 = ns.solve(m, past=None)
    assert np.array_equal(r0.Z, r1.Z)
    assert all(np.array_equal(r0.maps[a], r1.maps[a]) for a in r0.maps)
    assert r0.costs == r1.costs and r0.iterations == r1.iterations
    assert r1.grid is r0.grid and ns.SpectralFiniteSolver(m, past=None).c.g is r0.grid
    assert r1.past is None and r1.settled is None and "past" not in r1.to_dict()


@pytest.mark.skipif(not os.environ.get("NOISESTATE_SLOW"), reason="slow (40 s); set NOISESTATE_SLOW=1")
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
    m = ns.load(EX + "ch3_two_player.yaml"); L, T = 3.0, 12.0
    stat = ns.solve(m.with_horizon(nodes=16)).check()
    res = ns.solve(m.with_horizon(kind="finite", window=T, nodes=16), past=stat, tol=1e-10, start="coarse").check()
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


def strip_maps(stat, solver):
    """The stationary maps of `stat` carried onto the solver's strip (the map at age a, on both shock families)."""
    g = solver.c.g; gs = stat.compiled.grid
    return {a.name: np.einsum("fn,urn->urf", gs.interp(g.a), stat.maps[a.name]) for a in solver.model.agents}


def test_same_model_best_response_returns_the_stationary_maps_away_from_the_end():
    """The content of the identity in one evaluation: Chapter 3 as its own past (T = 12, 16 nodes), each agent's
    best response to the stationary maps carried onto the strip returns the projected map itself on the first
    window to 5e-10 (player1) and 1.2e-8 (player2), the band included, and to 1.5e-7 on [3, 6); the end leaks
    back through the one-shot FOC system at the closed-loop rate (1.2e-4 and 2.7e-4 on [6, 9), the control
    vanishing at T)."""
    m = ns.load(EX + "ch3_two_player.yaml"); L, T = 3.0, 12.0
    stat = ns.solve(m.with_horizon(nodes=16)).check()
    solver = ns.SpectralFiniteSolver(m.with_horizon(kind="finite", window=T, nodes=16), past=stat)
    g = solver.c.g; maps = strip_maps(stat, solver)
    for a in m.agents:
        gm, out = solver.best_response(a, maps)
        dev = np.abs(gm - maps[a.name]).max(axis=(0, 1)) / np.abs(maps[a.name]).max()
        wins = [dev[(g.t >= lo - 1e-9) & (g.t < lo + L - 1e-9)].max() for lo in np.arange(0.0, T, L)]
        assert wins[0] < 5e-8 and dev[g.upper].max() < 5e-8, (a.name, wins)
        assert wins[1] < 1e-6 and wins[2] < 1e-3 and wins[3] > 0.5, (a.name, wins)


# ------------------------------------------------ the discounted one-agent model with a prior
a, h, r, T1 = -0.3, 1.5, 0.5, 3.0
P0 = 0.8


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


def model(rho, nodes=12):
    return {"channels": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": h}, "noise": {"w1": 1.0}}},
                             "loss": [[1.0, "X", "X"], [r, "D", "D"]]}},
            "horizon": {"kind": "finite", "window": T1, "nodes": nodes, "discount": rho}}


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
    res = ns.solve(model(rho, 16), past=[shock]).check()
    assert res.shocks == ["w0", "w1", "xi"] and res.maps["a"].shape == (1, 1, res.compiled.N + res.compiled.Nt)
    assert abs(res.costs["a"] - J) < 1e-7, (res.costs["a"], J)
    assert res.resolution_ok
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
    res = ns.solve(model(0.5), past=[{"name": "xi", "loads": {"X": np.sqrt(P0)}}]).check()
    for name in ("X", "D"):
        assert np.abs(res.kernel(name, "xi") - np.sqrt(P0) * res.kernel(name, "w0")).max() < 1e-13


def test_regime_change_on_chapter_3_starts_from_the_old_kernels():
    """Chapter 3 with the past at p1 = 3 and the new model at p1 = 10 (T = 6, 12 nodes): the state kernels
    at 0+ are the old kernels at age -s to round-off, past=Model equals past=StationaryResult bit for bit,
    and the solve converges.  The costs are pinned at their 12-node values (2.56081481, 2.56540852; they move
    by 2.4e-5 and 3.1e-5 at 20 nodes, so 1e-6 catches a change of the formulation and not of the grid).
    The resolution guard says where its error sits: player2's 8.6e-4 is on the band's tip (5.4e-4 at 16 nodes: the
    collapsing corner's floor, not resolution), its interior 3.8e-5 (2.3e-5 at 16); player1's 4.4e-5 is in the
    interior, the transient at 0+ (2.2e-6 at 16 nodes: resolution)."""
    m = ns.load(EX + "ch3_two_player.yaml")
    old = ns.solve(m.with_params(p1=3.0)).check()
    new = m.with_params(p1=10.0).with_horizon(kind="finite", window=6.0, nodes=12)
    res = ns.solve(new, past=old).check()
    assert abs(res.costs["player1"] - 2.56081481) < 1e-6 and abs(res.costs["player2"] - 2.56540852) < 1e-6, res.costs
    for a in ("player1", "player2"):
        parts = res.representation_parts[a]
        assert set(parts) == {"interior", "band tip", "last window"} and max(parts.values()) == res.representation_error[a]
    p2 = res.representation_parts["player2"]
    assert p2["band tip"] > 10 * max(p2["interior"], p2["last window"]) and p2["band tip"] > 5e-4, p2
    assert res.representation_parts["player1"]["interior"] < 1e-4
    flag = next(r for r in res.diagnose() if r["name"] == "resolution")["flag"]
    assert "band tip" in flag and "last window" in flag and "interior" in flag
    assert "representation_parts" in res.to_dict()
    g = res.grid; at0 = g.upper & (g.t < 1e-12)
    assert at0.any()
    for name in ("X",):
        assert np.abs(res.kernel(name)[at0] - old.compiled.grid.interp(g.a[at0]) @ old.kernel(name)).max() < 1e-13
    res2 = ns.solve(new, past=m.with_params(p1=3.0))
    assert np.array_equal(res.Z, res2.Z) and res.costs == res2.costs
    assert all(np.array_equal(res.maps[k], res2.maps[k]) for k in res.maps)
    assert res.evaluate("X", "w0", np.array([1.0, 1.0]), np.array([-1.0, 0.5])).shape == (2,)
    assert res.second_order["player1"]["ok"] and res.solver_kw["past"] is res.past


def test_old_noise_loading_on_a_channel_the_new_row_drops_is_kept_on_the_band():
    """The past's row y1 loads w1 and w2 (0.3), the new row y1 loads w1 only: the increments observed before zero
    carry the old loading, so under the stationary maps the closed loop at 0+ returns the old kernels on every
    channel (w2 included: without the union of the two regimes' loadings the entry is dropped and D1's kernel on
    w2 is off by 0.3 g(a), a third of its size) to round-off, the past and the strip sharing their 16 nodes."""
    m = ns.load(EX + "ch3_two_player.yaml")
    d = m.to_dict(); d["agents"]["player1"]["signals"]["y1"]["noise"] = {"w1": 1.0, "w2": 0.3}
    old = ns.solve(ns.Model.from_dict(d).with_horizon(nodes=16)).check()
    solver = ns.SpectralFiniteSolver(m.with_horizon(kind="finite", window=3.0, nodes=16), past=old)
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
    m = ns.load(EX + "ch3_two_player.yaml")
    d = m.to_dict(); d["agents"]["player1"]["loss"].append([-2.0, "X"])
    mt = ns.Model.from_dict(d)
    statt = ns.solve(mt.with_horizon(nodes=16)).check()
    assert abs(statt.means["X"] - 0.86621649) < 1e-8 and abs(statt.means["D1"] - 1.79893500) < 1e-8
    res = ns.solve(mt.with_horizon(kind="finite", window=3.0, nodes=12), past=statt).check()
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
    r0 = ns.solve(m0.with_horizon(kind="finite", window=3.0, nodes=12), past=statt).check()
    assert abs(r0.mean("X", 0.0)[0]) < 1e-12
    assert np.abs(r0.mean("X", at) - [0.0, 0.39555973, 0.61918718, 0.84513578, 0.92167413]).max() < 1e-6
    assert abs(r0.mean("D1", 0.0)[0] - 2.31569976) < 1e-5
    assert np.array_equal(r0.Z, res.Z)                             # the kernels do not depend on the means


def test_past_validation():
    """A finite result is a TypeError, an unconverged stationary one a ValueError, other channels a ValueError,
    an initial shock on an unknown state a ValueError, and a past with a window on a model whose rows are
    observed with a delay is not implemented in this stage."""
    m = ns.load(EX + "ch3_two_player.yaml")
    with pytest.raises(TypeError, match="stationary"):
        Past.of(ns.solve(m.with_horizon(kind="finite", window=1.0, nodes=6)))
    with pytest.raises(ValueError, match="did not converge"):
        Past.of(ns.solve(m, max_evaluations=1))
    stat = ns.solve(m).check()
    other = m.to_dict(); other["channels"] = ["w0", "w1", "w9"]
    other["agents"]["player2"]["signals"]["y2"]["noise"] = {"w9": 1.0}
    with pytest.raises(ValueError, match="channels"):
        ns.SpectralFiniteSolver(ns.Model.from_dict(other).with_horizon(kind="finite", window=3.0, nodes=6), past=stat)
    with pytest.raises(ValueError, match="not a state"):
        ns.SpectralFiniteSolver(m.with_horizon(kind="finite", window=3.0, nodes=6), past=[{"name": "v", "loads": {"V": 1.0}}])
    delayed = ns.load(EX + "ch1_delayed_finite.yaml")
    dpast = ns.solve(delayed.with_horizon(kind="stationary", window=1.0, nodes=6)).check()
    with pytest.raises(NotImplementedError, match="delay"):
        ns.SpectralFiniteSolver(delayed, past=dpast)
