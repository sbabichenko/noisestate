"""Stationary means.  Linear loss terms (targets) and constant drifts move the means of the states and controls,
deterministic paths common knowledge to everyone; the stationary engine solves them from every control's mean
first-order condition (the kernels' condition applied to a constant, no information constraint) and the mean
dynamics, one linear system.  Closed forms, the open-loop and closed-loop limits, invariants, validation."""
import json, os, sys
import numpy as np, pytest
from scipy.optimize import fsolve
import noisestate as ns
from noisestate import engines
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def one_agent(a, r, theta, rho, L=16.0, nodes=16, const=0.0, unit=2.0):
    """dX = (-a X + D [+ const]) dt + dW, loss (X - theta)^2 + r D^2 less theta^2: [1, X, X], [-2 theta, X], [r, D, D]."""
    drift = {"X": -a, "D": 1.0, **({"const": const} if const else {})}
    loss = [[1.0, "X", "X"], [r, "D", "D"]] + ([[-2.0 * theta, "X"]] if theta else [])
    return ns.Model.from_dict({"name": "target", "shocks": ["w0", "w1"], "states": {"X": {"drift": drift, "noise": {"w0": 1.0}}},
                               "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w1": 1.0}}}, "loss": loss}},
                               "horizon": {"kind": "stationary", "discount": rho, "window": L}, "numerics": {"unit": unit, "unit_range": L, "nodes": nodes}})


def two_agents(p, a=1.0, r=1.0, theta=1.0, rho=0.5, L=12.0, nodes=16, opposite=True, tie=False):
    """dX = (-a X + D1 + D2) dt + dW0, agent i sees sqrt(p) X dt + dW_i and has the target theta_i on X."""
    th = [theta, -theta if opposite else theta]
    d = {"name": "targets", "shocks": ["w0", "w1", "w2"],
         "states": {"X": {"drift": {"X": -a, "D1": 1.0, "D2": 1.0}, "noise": {"w0": 1.0}}},
         "agents": {f"a{i}": {"controls": [f"D{i}"], "signals": {f"y{i}": {"drift": {"X": float(np.sqrt(p))}, "noise": {f"w{i}": 1.0}}},
                              "loss": [[1.0, "X", "X"], [-2.0 * th[i - 1], "X"], [r, f"D{i}", f"D{i}"]]} for i in (1, 2)},
         "horizon": {"kind": "stationary", "discount": rho, "window": L}, "numerics": {"breakpoints": [0, 0.05, 0.15, 0.4, 1, 2, 4, 8, L], "nodes": nodes},
         **({"ties": [["a1", "a2"]]} if tie else {})}
    return ns.Model.from_dict(d)


def closed_loop_nash(a, r, theta, rho):
    """Feedback Nash of dx = (-a x + u1 + u2) dt, J_i = int e^{-rho t} [(x - theta_i)^2 + r_i u_i^2] dt, in affine feedback
    u_i = -(P_i x + s_i) / r_i: the coupled algebraic Riccati equations of (P_1, P_2), the linear equations of (s_1, s_2),
    then the rest point of the closed loop."""
    r = np.asarray(r, float); theta = np.asarray(theta, float)
    f = lambda P: 1.0 + P ** 2 / r - 2.0 * P * (a + np.sum(P / r)) - rho * P
    P = fsolve(f, np.full(2, 0.5), xtol=1e-12); assert np.abs(f(P)).max() < 1e-10 and (P > 0).all()
    K = rho + a + np.sum(P / r)
    s = np.linalg.solve(np.diag(K - P / r) + np.outer(P, 1.0 / r), -theta)
    x = -np.sum(s / r) / (a + np.sum(P / r))
    return x, -(P * x + s) / r


def open_loop_nash(a, r, theta, rho):
    """Each agent's derivative of the state mean is the direct dynamics only, the DC gain 1/(a + rho) of e^{-a t}."""
    r = np.asarray(r, float); theta = np.asarray(theta, float); DC = 1.0 / (a + rho)
    x = DC * np.sum(theta / r) / (a + DC * np.sum(1.0 / r))
    return x, (theta - x) * DC / r


@pytest.mark.parametrize("a,r,theta,rho", [(1.0, 1.0, 1.0, 0.0), (2.0, 0.5, -3.0, 0.0), (0.5, 2.0, 2.0, 0.0), (1.0, 1.0, 1.0, 0.5), (2.0, 0.3, 1.5, 1.0)])
def test_one_agent_target_closed_form(a, r, theta, rho):
    """ubar = theta a / (1 + r a (a + rho)), xbar = ubar / a (the discounted deterministic optimum).  The continuation of
    the mean first-order condition is the DC gain of the passive response e^{-(a + rho) t} over ALL ages, 1 / (a + rho),
    whatever the window: the solver reproduces the closed form to round-off on L = 16 and on L = 4, where the window's
    integral (1 - e^{-(a + rho) L}) / (a + rho), which it used before, was 13% short at (a, rho) = (0.5, 0) and moved
    ubar by 3.7e-2.  The mean cost is (xbar - theta)^2 - theta^2 + r ubar^2."""
    u_exact = theta * a / (1.0 + r * a * (a + rho))
    for L in (16.0, 4.0):
        m = one_agent(a, r, theta, rho, L=L, unit=min(2.0, L)); res = ns.solve(m).require_converged()
        ub, xb = res.means["D"], res.means["X"]
        assert abs(ub - u_exact) < 1e-12 * max(1.0, abs(theta)) and abs(xb - ub / a) < 1e-12 * max(1.0, abs(theta))
        assert abs(res.cost_parts["a"]["mean"] - ((xb - theta) ** 2 - theta ** 2 + r * ub ** 2)) < 1e-12
        assert res.costs["a"] == res.cost_parts["a"]["variance"] + res.cost_parts["a"]["mean"] and res.means["a.y"] == xb
        assert res.mean_tail is None                          # the whole response, not the window's integral
    assert f"mean {res.cost_parts['a']['mean']:+.6f}" in res.summary() and "means: X=" in res.summary()


def test_constant_drift_is_the_shifted_target():
    """dX = (-a X + D + kappa) dt with loss X^2 + r D^2 is, in X - kappa / a, the target model with theta = -kappa / a: the same
    ubar and kernels, xbar larger by kappa / a, and a mean cost larger by theta^2, the constant a target leaves out."""
    a, r, kappa = 1.0, 1.0, 0.7
    rc = ns.solve(one_agent(a, r, 0.0, 0.0, const=kappa)).require_converged(); rt = ns.solve(one_agent(a, r, -kappa / a, 0.0)).require_converged()
    assert abs(rc.means["D"] - rt.means["D"]) < 1e-14 and abs(rc.means["X"] - rt.means["X"] - kappa / a) < 1e-14
    assert abs(rc.cost_parts["a"]["mean"] - rt.cost_parts["a"]["mean"] - (kappa / a) ** 2) < 1e-14
    assert rc.cost_parts["a"]["variance"] == rt.cost_parts["a"]["variance"] and np.array_equal(rc.kernel("X"), rt.kernel("X"))
    assert any("constant drift 0.7 moves only the means" in n for n in rc.model.notes)


def test_two_agents_between_open_loop_and_closed_loop_nash():
    """Opposite targets on one state.  With no information (p -> 0, kernels zero) the mean paths are the open-loop Nash of
    the deterministic game, ubar_i = theta_i / (r (a + rho)) = 0.6666667 at a = r = 1, rho = 0.5: the solver is 4.2e-8 from it
    at p = 1e-6, the other agent's answer being O(p) (4.2e-5 at p = 1e-3).  As p grows a deviation of the mean is seen and
    answered through the other agent's kernel, and ubar_1 falls monotonically toward the closed-loop (feedback) Nash from the
    coupled algebraic Riccati equations, 0.5569995 (P = 0.29533): 0.64181 at p = 1, 0.60047 at 10, 0.57297 at 100, 0.56224 at
    1000, the gap shrinking as 1/sqrt(p) (1.6e-2 at 100, 5.2e-3 at 1000).  The state mean is zero throughout by symmetry."""
    a, r, theta, rho = 1.0, 1.0, 1.0, 0.5
    xo, uo = open_loop_nash(a, [r, r], [theta, -theta], rho); xc, uc = closed_loop_nash(a, [r, r], [theta, -theta], rho)
    assert abs(uo[0] - 2.0 / 3.0) < 1e-12 and abs(uc[0] - 0.5569995318) < 1e-9 and abs(xo) < 1e-12 and abs(xc) < 1e-12
    seen = {}
    for p in [1e-6, 1e-3, 1.0, 100.0, 1000.0]:
        res = ns.solve(two_agents(p)).require_converged(); seen[p] = res.means["D1"]
        assert abs(res.means["D2"] + res.means["D1"]) < 1e-12 and abs(res.means["X"]) < 1e-12
    assert abs(seen[1e-6] - uo[0]) < 1e-7 and abs(seen[1e-3] - uo[0]) < 1e-4
    vals = [seen[p] for p in sorted(seen)]
    assert all(v1 > v2 for v1, v2 in zip(vals, vals[1:])) and uc[0] < vals[-1] < uo[0]
    assert abs(seen[1000.0] - uc[0]) < 6e-3 and abs(seen[100.0] - uc[0]) < 2e-2 and abs(seen[1.0] - 0.64181) < 1e-4


def test_examples_unchanged_and_means_zero_without_a_driver():
    """No linear term and no constant drift: every mean is exactly zero, with no solve, and the costs are what they were."""
    for f, costs in (("ch3_two_player.yaml", {"player1": 0.42895400568415, "player2": 0.498143327906243}),
                     ("ch4_kyle_back.yaml", {"market_maker": -9.883485143225153, "trader1": -0.8208711497505051})):
        res = ns.solve(os.path.join(EX, f)).require_converged()
        assert all(abs(res.costs[k] - v) < 1e-12 for k, v in costs.items())
        assert all(v == 0.0 for v in res.means.values()) and set(res.model.state_names + res.model.control_names) <= set(res.means)
        assert all(p["mean"] == 0.0 and p["variance"] == res.costs[k] for k, p in res.cost_parts.items())
        assert "mean" not in res.summary().split("\n")[1] and "means:" not in res.summary()


def test_tied_agents_and_the_cycle_market_get_identical_means():
    """Tied agents share a strategy and therefore a mean; the two-firm Chapter 5 market (linear terms -2 kappa on deliveries
    and sales, a random-walk demand level q pinned at 0) is solved on its symmetric path."""
    res = ns.solve(two_agents(1.0, opposite=False, tie=True)).require_converged()
    assert abs(res.means["D1"] - res.means["D2"]) < 1e-14 and res.means["D1"] > 0.2
    free = ns.solve(two_agents(1.0, opposite=False)).require_converged()
    assert abs(free.means["D1"] - free.means["D2"]) < 1e-10 and abs(free.means["D1"] - res.means["D1"]) < 1e-8
    sys.path.insert(0, EX)
    from make_ch5_cycle_market import build
    m = build(N=2, L=6.0, nodes=6, unit_range=3.0); r5 = ns.solve(m, tol=1e-8).require_converged()
    for u in ("P", "o"):
        assert abs(r5.means[f"{u}0"] - r5.means[f"{u}1"]) < 1e-12 and abs(r5.means[f"{u}0"]) > 0.2
    assert r5.means["q"] == 0.0 and r5.cost_parts["firm0"]["mean"] < 0
    #  The two firms are symmetric, so their cost parts agree -- but each is integrated separately
    #  over its own loss atoms, so the two sums are not the same sequence of floating-point
    #  operations and need not agree to the last bit.  Asserting `==` passed on one machine's BLAS
    #  and failed on a CI runner's by one ULP (4.628987723316478 against ...473).  The claim is
    #  symmetry, which a relative tolerance states without also claiming bit-identity.
    for part in ("variance", "mean"):
        a, b = r5.cost_parts["firm0"][part], r5.cost_parts["firm1"][part]
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (part, a, b)
    assert set(r5.cost_parts["firm0"]) == set(r5.cost_parts["firm1"])
    assert any("random walks with no inputs" in n for n in m.notes) and any("linear loss term" in n for n in m.notes)
    assert "window too short" in repr(r5)                          # its window is flagged (the kernels'; the means' continuation is whole)
    assert r5.mean_tail is None


def test_means_are_part_of_the_answer_payload_and_cli(tmp_path):
    m = one_agent(1.0, 1.0, 1.0, 0.0); S = engines.solver(m)
    off = S.solve(diagnostics=False); assert abs(off.means["D"] - 0.5) < 1e-6 and off.cost_parts["a"]["mean"] < 0
    d = json.loads(json.dumps(off.to_dict()))
    assert d["means"] == off.means and d["cost_parts"]["a"]["mean"] == off.cost_parts["a"]["mean"] and d["costs"]["a"] == off.costs["a"]
    assert ns.Model.from_dict(d["model"]).to_dict() == m.to_dict()
    path = tmp_path / "target.yaml"; out = tmp_path / "target.json"
    import yaml
    path.write_text(yaml.safe_dump(m.to_dict()))
    assert main(["solve", str(path), "-o", str(out)]) == 0 and abs(json.load(open(out))["means"]["D"] - 0.5) < 1e-6
    fin = ns.solve(os.path.join(EX, "ch1_two_player_finite.yaml")).require_converged()     # the spectral finite engine: paths, all zero here
    assert all(np.array_equal(v, np.zeros(len(fin.mean_times))) for v in fin.means.values()) and fin.cost_parts["player1"]["mean"] == 0.0


def test_validation_of_constants_and_singular_mean_systems():
    d = one_agent(1.0, 1.0, 1.0, 0.0, const=0.3).to_dict()
    bad = json.loads(json.dumps(d)); bad["agents"]["a"]["signals"]["y"]["drift"]["const"] = 1.0
    with pytest.raises(ValueError, match="carries no information"):
        ns.Model.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["definitions"] = {"Xs": {"X": 1.0, "const": -1.0}}
    with pytest.raises(ValueError, match="allowed in a state's drift only"):
        ns.Model.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["agents"]["a"]["loss"].append([1.0, "const", "X"])
    with pytest.raises(ValueError, match="reads the constant"):
        ns.Model.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["states"]["X"]["drift"] = {"X": -1.0, "D": 1.0, "const@0.5": 0.3}
    with pytest.raises(ValueError, match="a constant has no lag"):
        ns.Model.from_dict(bad)
    bad = json.loads(json.dumps(d)); bad["states"]["const"] = {"drift": {}, "noise": {"w0": 1.0}}
    with pytest.raises(ValueError, match="reserved"):
        ns.Model.from_dict(bad)
    fin = json.loads(json.dumps(d)); fin["horizon"] = {"kind": "finite", "T": 1.0}; fin["numerics"] = {"nodes": 4}     # the finite engine: paths
    rf = ns.solve(fin).require_converged(); assert rf.means["D"].shape == rf.mean_times.shape and rf.means["D"].ndim == 1
    walk = json.loads(json.dumps(d)); walk["states"]["X"]["drift"] = {"const": 0.3}      # a random walk with a drift and no inputs
    with pytest.raises(ValueError, match="no stationary mean"):
        ns.solve(walk)
    # driven by the control, which offsets the drift: ubar = -0.3, and its first-order condition 2 r ubar + 2 (xbar - theta) DC = 0
    # puts xbar at theta - r ubar / DC with DC the discounted DC gain of a random walk, 1 / rho
    walk["states"]["X"]["drift"] = {"D": 1.0, "const": 0.3}; walk["horizon"]["discount"] = 0.5
    res = ns.solve(walk).require_converged(); DC = 1.0 / 0.5
    assert abs(res.means["D"] + 0.3) < 1e-12 and abs(res.means["X"] - (1.0 + 0.3 / DC)) < 1e-12
    # a random walk driven by a control whose first-order condition never reads it: its mean is undetermined
    sing = {"name": "s", "shocks": ["w0", "w1"], "states": {"V": {"drift": {"D": 1.0}, "noise": {"w0": 1.0}}},
            "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"V": 1.0}, "noise": {"w1": 1.0}}}, "loss": [[1.0, "D", "D"], [-2.0, "D"]]}},
            "horizon": {"kind": "stationary", "window": 4.0}, "numerics": {"nodes": 8}}
    with pytest.raises(ValueError, match="mean system is singular"):
        ns.solve(sing)
    # a random walk driven by a control whose target pins it: well posed, xbar = theta at any discount
    sing["agents"]["a"]["loss"] = [[1.0, "V", "V"], [-2.0 * 1.5, "V"], [1.0, "D", "D"]]; sing["horizon"]["discount"] = 0.5
    res = ns.solve(sing).require_converged(); assert abs(res.means["V"] - 1.5) < 1e-12 and abs(res.means["D"]) < 1e-12


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_lead_cross_term_in_a_driven_model(rho):
    """A lead cross term 2c D X@-tau (allowed: the own current control against a led quantity) enters the mean
    first-order condition through the response of X weighted by e^{rho tau}, the kernels' lead convention (the
    flows before t that read the quantity after t): r ubar + c xbar + DC [(xbar - theta) + e^{rho tau} c ubar] = 0
    with xbar = ubar / a and DC = 1 / (a + rho), so ubar = DC theta / (r + c / a + DC / a + e^{rho tau} c DC); at rho = 0
    the weight is one and the lead is the lag's mean."""
    a, r, theta, c, tau = 1.0, 1.0, 1.0, 0.3, 2.0
    d = one_agent(a, r, theta, rho).to_dict(); d["agents"]["a"]["loss"].append([2.0 * c, "D", f"X@-{tau:g}"])
    res = ns.solve(ns.Model.from_dict(d)).require_converged()
    DC = 1.0 / (a + rho); w = np.exp(rho * tau)
    ub = DC * theta / (r + c / a + DC / a + w * c * DC); xb = ub / a
    assert abs(res.means["D"] - ub) < 1e-10 and abs(res.means["X"] - xb) < 1e-10
    assert abs(res.cost_parts["a"]["mean"] - ((xb - theta) ** 2 - theta ** 2 + r * ub ** 2 + 2.0 * c * ub * xb)) < 1e-10
    assert "-0.000000" not in res.summary()


def tug_of_war(L, nodes):
    """Chapter 1's two players on a random walk with conflicting targets (0 and 1), player 2's row delayed by 0.5, no
    discount: at the mean both push constantly against each other, D1 = -D2 = 1 / (r1 k2 + r2 k1) with k_j the other's
    DC gain on X, and player 1's passive world (player 2's delayed reaction alone) decays slowly."""
    ag = lambda i, d: {"controls": [f"D{i}"], "signals": {f"y{i}": {"drift": {"X": float(np.sqrt(3.0))}, "noise": {f"w{i}": 1.0}, "delay": d}},
                       "loss": [[1.0, "X", "X"], [0.1, f"D{i}", f"D{i}"]] + ([[-2.0, "X"]] if i == 2 else [])}
    return ns.Model.from_dict({"name": "tug", "shocks": ["w0", "w1", "w2"], "states": {"X": {"drift": {"D1": 1.0, "D2": 1.0}, "noise": {"w0": 1.0}}},
                               "agents": {"player1": ag(1, 0.0), "player2": ag(2, 0.5)},
                               "horizon": {"kind": "stationary", "window": L}, "numerics": {"nodes": nodes, "unit": 0.5}})


def test_the_means_continuation_is_whole_so_the_means_do_not_wait_for_the_passive_world():
    """The mean first-order condition weighs the passive response of every loss atom by a constant, so it needs its
    integral over all ages.  In the tug of war player 1's passive response of X is still 42% of its peak at L = 3
    (the equilibrium's kernels 0.5%): over the window alone player 1's mean push |D1| was 7.24 at L = 3, 8.15 at 6,
    8.40 at 12, and the costs 16% apart; over all ages it is 8.4534 / 8.42046 / 8.42044 at L = 3 / 6 / 9, the mean parts
    0.8% apart at L = 3 and 5e-6 from L = 6.  _passive_dc agrees with the grid's integral where the passive response has
    decayed (player 2's at L = 9, 1.1e-4 of its peak left at the edge)."""
    r3, r6, r9 = (ns.solve(tug_of_war(L, n)).require_converged() for L, n in ((3.0, 16), (6.0, 22), (9.0, 28)))
    assert r3.mean_tail is None and r6.mean_tail is None
    assert abs(r6.means["D1"] - r9.means["D1"]) < 1e-5 * abs(r9.means["D1"]) and abs(r6.means["D1"] + r6.means["D2"]) < 1e-10
    assert abs(r3.means["D1"] - r9.means["D1"]) < 1e-2 * abs(r9.means["D1"])
    for a in r9.costs:
        assert abs(r6.cost_parts[a]["mean"] - r9.cost_parts[a]["mean"]) < 1e-5 * abs(r9.cost_parts[a]["mean"])
        assert abs(r3.cost_parts[a]["mean"] - r9.cost_parts[a]["mean"]) < 1e-2 * abs(r9.cost_parts[a]["mean"])
    # player 2's passive world (player 1 reacting at once) has decayed by L = 9: the grid's integral is the transform
    S = engines.solver(tug_of_war(9.0, 28)); res = S.solve(diagnostics=False); c = S.c; a2 = S.model.agents[1]
    R = S._impulse_responses(a2, res.maps, S._spikes(c, res.maps, a2)[1]); H = S._passive_dc(a2, res.maps, R, c.loss[a2.name][0])
    dc = c.grid.discounted_mass(0.0)
    assert abs(H[c.index["X"], 0] - dc @ R[c.block("X"), 0]) < 3e-4 * abs(R[c.block("X"), 0]).max() and R[c.block("X"), 0][-1] < 2e-4   # the tail beyond L and abs(H[c.index["D1"], 0] + 1.0) < 1e-12


def test_the_window_check_reads_the_means_where_their_continuation_is_the_windows():
    """Where the whole-age transform does not apply (here an instant observation) the mean condition keeps the window's
    integral and the result keeps the passive response's level at L, which the `window` check reads."""
    d = {"name": "instant", "shocks": ["w0", "w1", "w2"],
         "states": {"X": {"drift": {"X": -0.2, "D0": 1.0, "D1": 1.0}, "noise": {"w0": 1.0}}},
         "agents": {"a0": {"controls": ["D0"], "signals": {"y": {"drift": {"X": 2.0}, "noise": {"w1": 1.0}}},
                           "loss": [[1.0, "X", "X"], [-2.0, "X"], [1.0, "D0", "D0"]], "monitors": ["a1"]},
                    "a1": {"controls": ["D1"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w2": 1.0}}},
                           "loss": [[1.0, "X", "X"], [1.0, "D1", "D1"], [0.3, "D0", "D0"], [0.6, "D1", "D0"]],
                           "instant": ["D0"], "monitors": ["a0"]}},
         "horizon": {"kind": "stationary", "window": 2.0, "discount": 0.0}, "numerics": {"nodes": 12}}
    res = ns.solve(ns.Model.from_dict(d)).require_converged()
    assert res.mean_tail is not None and res.mean_tail > 0.02
    row = [r for r in res._check_rows() if r["name"] == "window"][0]
    assert row["value"] == res.mean_tail and not row["ok"] and "WINDOW TOO SHORT FOR THE MEANS" in row["flag"]
