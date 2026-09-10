"""Mean paths on the finite engines.  Targets, constant drifts and initial states move the means of the states and
controls, deterministic paths on [0, T] common knowledge to everyone; the spectral engine solves them on the time nodes
from every control's mean first-order condition (the kernels' condition applied to a path, no information constraint,
the targets and the initial state as the driver) and the mean dynamics, one linear system, and the cell engine does the
same on its cells at first order.  Checked against the dissertation's spectral solver (the Chapter 1 game with targets),
the cell engine's Richardson limits, the deterministic finite-horizon LQ closed form (Riccati, solve_ivp), invariants
(linearity in the targets, the kernels untouched, the open-loop limit) and the shipped examples."""
import json, os
import numpy as np, pytest
from scipy.integrate import solve_ivp
import noisestate as ns
from noisestate import engines
from noisestate.cli import main
from noisestate.finite_spectral import SpectralFiniteSolver
from noisestate.finite_free import RespOps
from ch1_mean_sweep import model as ch1_targets, CLOSED_LOOP
from helpers import EX, REFS, slow

# The dissertation's spectral solver (12 x 16 nodes, Tikhonov 1e-7; its Dbar1(0) converged to 0.03%): Dbar1(0), Dbar1(T/2)
# and the mean part of the cost, which there includes the target's constant b^2 T = 1 (the package's Jbar leaves it out).
REF = {0.1: (9.936882, 4.981026, 4.295799), 1.0: (9.477083, 4.850592, 4.033810), 10.0: (7.797607, 4.444278, 3.226008),
       100.0: (5.981174, 3.937315, 2.537043), 1000.0: (5.116303, 3.607183, 2.250005)}
# The package's own converged values (stable from 16 nodes per side, to 1e-7 at 24 and 32; 2026-09-05), Jbar without b^2 T.
OWN = {0.1: (9.936695, 4.980972, 3.295552), 1.0: (9.475728, 4.850218, 3.032079), 10.0: (7.794705, 4.443465, 2.222439),
       100.0: (5.980097, 3.935286, 1.533783), 1000.0: (5.106377, 3.600851, 1.242940)}


@slow("slow (6 s, five solves; the p = 10 point is pinned below and in tests/test_baseline.py); set NOISESTATE_SLOW=1")
def test_ch1_target_sweep_against_the_dissertation():
    """Player 1 tracks +1, player 2 tracks -1 on dX = (D1 + D2) dt + dW0 with private signals of precision p.  Against
    the dissertation's solver, Dbar1(0) and Dbar1(T/2) agree to 0.1% up to p = 100 (1.9e-5 at p = 0.1, 3.7e-4 and 5.3e-4
    at p = 10 and 100) and Jbar to 0.1% up to p = 1 (5.7e-5, 4.3e-4), then 1.1e-3 and 1.3e-3 at p = 10 and 100, the
    reference's own error: the cell engine's Richardson limits close on the package's values as h^2 (the next test) and
    the reference's variance cost is off by the same order (9e-5 at p = 10, 2.0e-4 at p = 100).  At p = 1000 the kernels
    are sharp, the package needs 20 nodes per side (16 is 3.7e-4 off, 12 is 8e-4) and the reference is 2e-3 off on the
    paths (1.9e-3, 1.8e-3), 3.1e-3 on Jbar and 1.1e-3 on its variance cost.  The state mean is zero by symmetry;
    Dbar1(0) falls monotonically from the open-loop 10 toward the closed-loop 4.646924 as the precision grows (the
    separation failure)."""
    seen = {}
    for p, (r0, rh, J) in REF.items():
        res = ns.solve(ch1_targets(p, nodes=20 if p == 1000.0 else 12)).require_converged()
        d0, dh = res.mean("D1", [0.0, 0.5]); Jb = res.cost_parts["player1"]["mean"]; seen[p] = d0
        tol_d, tol_J = (2.5e-3, 3.5e-3) if p == 1000.0 else (1e-3, 2e-3)
        assert abs(d0 - r0) < tol_d * r0 and abs(dh - rh) < tol_d * rh and abs(Jb + 1.0 - J) < tol_J * J
        o0, oh, oJ = OWN[p]
        assert abs(d0 - o0) < 2e-5 * o0 and abs(dh - oh) < 2e-5 * oh and abs(Jb - oJ) < 2e-5 * oJ
        assert np.abs(res.means["D2"] + res.means["D1"]).max() < 1e-12 and np.abs(res.means["X"]).max() < 1e-12
        assert res.cost_parts["player2"]["mean"] == pytest.approx(Jb, abs=1e-12) and res.costs["player1"] == res.cost_parts["player1"]["variance"] + Jb
    vals = [seen[p] for p in sorted(seen)]
    assert all(v1 > v2 for v1, v2 in zip(vals, vals[1:])) and CLOSED_LOOP < vals[-1] < 10.0


def test_ch1_p10_path_against_the_dissertation_and_the_cell_engine():
    """The p = 10 mean paths at every printed t of tests/refs/ch1_mean_p10.txt: within 1.4e-2 of the reference (1.9e-3 of
    Dbar1(0), the largest at t = 0.075, the reference's own error as above; 8e-4 at T/2).  The cell engine, an independent
    first-order discretisation, solves the same mean system on its cells: its Richardson pairs (40, 80) and (80, 160)
    close on the spectral paths as h^2 (4.8e-3 then 1.2e-3 at T/2; the cost 1.4e-3 then 3.3e-4), not on the reference."""
    sp = ns.solve(ch1_targets(10.0, nodes=12)).require_converged()
    data = np.loadtxt(os.path.join(REFS, "ch1_mean_p10.txt"))
    d1 = sp.mean("D1", data[:, 0])
    assert np.abs(d1 - data[:, 2]).max() < 1.4e-2 and abs(d1[100] - data[100, 2]) < 1e-3       # t = 0.5
    assert np.abs(sp.mean("D2", data[:, 0]) + d1).max() < 1e-12 and np.abs(sp.mean("X", data[:, 0])).max() < 1e-12
    assert abs(d1[-1]) < 1e-12                                                                     # Dbar1(T) = 0
    tq = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9]); d1sp = sp.mean("D1", tq); Jsp = sp.cost_parts["player1"]["mean"]
    got = {}
    for N in (40, 80, 160):
        d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "T": 1.0}; d["numerics"] = {"engine": "cells", "nodes": N}
        res = ns.solve(d).require_converged(); idx = np.round(tq / res.compiled.h).astype(int)
        got[N] = (res.means["D1"][idx], res.cost_parts["player1"]["mean"])
        assert res.mean_times.shape == (N,) and np.abs(res.means["D2"] + res.means["D1"]).max() < 1e-12
    assert np.abs(got[160][0] - d1sp).max() < np.abs(got[80][0] - d1sp).max() < np.abs(got[40][0] - d1sp).max()
    r1 = 2 * got[80][0] - got[40][0] - d1sp; r2 = 2 * got[160][0] - got[80][0] - d1sp
    assert np.abs(r2).max() < 1.5e-3 and np.abs(r2).max() < 0.5 * np.abs(r1).max()
    J1 = 2 * got[80][1] - got[40][1] - Jsp; J2 = 2 * got[160][1] - got[80][1] - Jsp
    assert abs(J2) < 5e-4 and abs(J2) < 0.5 * abs(J1)


def riccati(A, B, Qx, theta, r, rho, x0, T, ts):
    """The deterministic finite-horizon LQ problem min int_0^T e^{-rho t} (x'Qx x - 2 theta'x + r u^2) dt, dx/dt = A x + B u,
    x(0) = x0: the Riccati equations of V(t, x) = 1/2 x'Px + s'x + c backward from P(T) = 0, s(T) = 0, c(T) = 0, then the
    closed loop u = -B'(Px + s) / (2r) forward, both at rtol 1e-12.  Returns (x at ts, u at ts, the cost, V(0, x0))."""
    A = np.atleast_2d(A); B = np.atleast_1d(B).astype(float); Qx = np.atleast_2d(Qx); theta = np.atleast_1d(theta)
    x0 = np.atleast_1d(x0); n = A.shape[0]; BB = np.outer(B, B)

    def back(t, y):
        P = y[:n * n].reshape(n, n); s = y[n * n:n * n + n]; c = y[-1]
        dP = rho * P - 2 * Qx - A.T @ P - P @ A + P @ BB @ P / (2 * r)
        ds = rho * s + 2 * theta - A.T @ s + P @ BB @ s / (2 * r)
        return np.concatenate([dP.ravel(), ds, [rho * c + s @ BB @ s / (4 * r)]])
    bw = solve_ivp(back, (T, 0.0), np.zeros(n * n + n + 1), rtol=1e-12, atol=1e-14, dense_output=True)

    def ctrl(t, x):
        y = bw.sol(t); return -B @ (y[:n * n].reshape(n, n) @ x + y[n * n:n * n + n]) / (2 * r)

    def fwd(t, y):
        x = y[:n]; u = ctrl(t, x)
        return np.concatenate([A @ x + B * u, [np.exp(-rho * t) * (x @ Qx @ x - 2 * theta @ x + r * u * u)]])
    fw = solve_ivp(fwd, (0.0, T), np.concatenate([x0, [0.0]]), rtol=1e-12, atol=1e-14, t_eval=ts)
    xs = fw.y[:n]; us = np.array([ctrl(t, xs[:, k]) for k, t in enumerate(ts)])
    y0 = bw.sol(0.0)
    return xs, us, fw.y[n, -1], 0.5 * x0 @ y0[:n * n].reshape(n, n) @ x0 + y0[n * n:n * n + n] @ x0 + y0[-1]


def _kind(kind):
    """kind "finite_cells" names the cell engine's result; the model asks for kind finite with engine cells."""
    return "finite" if kind == "finite_cells" else kind


def _engine(kind):
    return {"engine": "cells"} if kind == "finite_cells" else {}


def one_state(a, r, theta, rho, x0, kind="finite", nodes=12):
    """dX = (a X + D) dt + dW0, loss X^2 - 2 theta X + r D^2, X(0) = x0, one agent with a noisy signal on X."""
    loss = [[1.0, "X", "X"], [r, "D", "D"]] + ([[-2.0 * theta, "X"]] if theta else [])
    return ns.Model.from_dict({"name": "lq1", "channels": ["w0", "w1"], "states": {"X": {"drift": {"X": a, "D": 1.0}, "noise": {"w0": 1.0}, "initial": x0}},
                               "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 2.0 ** 0.5}, "noise": {"w1": 1.0}}}, "loss": loss}},
                               "horizon": {"kind": _kind(kind), **({"window": 1.0} if _kind(kind) == "stationary" else {"T": 1.0}),
                                           "discount": rho},
                               "numerics": {"nodes": nodes, **_engine(kind)}})


@pytest.mark.parametrize("a,r,theta,rho,x0", [(-1.0, 0.5, 1.0, 0.0, 0.0), (0.3, 0.2, 0.0, 0.0, 1.0), (-0.5, 0.3, 2.0, 0.5, -1.0)])
def test_one_agent_is_the_deterministic_optimum(a, r, theta, rho, x0):
    """One agent alone: the mean paths are the deterministic finite-horizon LQ optimum, with a target and x0 = 0, with
    x0 and no target, and discounted with both: paths within 1e-11 of the Riccati closed form at 12 nodes per side
    (6e-14 to 2e-12 seen) and the mean cost within 1e-10 of both the integrated loss and V(0, x0).  The cell engine's
    error halves from 40 to 80 cells (first order): 1.5e-2 to 7.6e-3 on the state at (a, r, theta, rho) = (-0.5, 0.3, 2, 0.5)."""
    res = ns.solve(one_state(a, r, theta, rho, x0)).require_converged()
    xs, us, J, V0 = riccati(a, 1.0, 1.0, theta, r, rho, x0, 1.0, res.mean_times)
    assert np.abs(res.means["X"] - xs[0]).max() < 1e-11 and np.abs(res.means["D"] - us).max() < 1e-11
    assert abs(res.cost_parts["a"]["mean"] - J) < 1e-10 and abs(res.cost_parts["a"]["mean"] - V0) < 1e-10
    assert res.means["X"][0] == x0 and abs(res.means["D"][-1]) < 1e-12 and np.array_equal(res.means["a.y"], 2.0 ** 0.5 * res.means["X"])
    if rho:
        errs = []
        for N in (40, 80):
            rc = ns.solve(one_state(a, r, theta, rho, x0, kind="finite_cells", nodes=N)).require_converged()
            xs, us, J, V0 = riccati(a, 1.0, 1.0, theta, r, rho, x0, 1.0, rc.mean_times)
            errs.append(np.abs(rc.means["X"] - xs[0]).max())
        assert 0.4 < errs[1] / errs[0] < 0.6 and errs[1] < 1e-2


@pytest.mark.parametrize("theta,rho", [(1.0, 0.0), (-0.7, 0.8)])
def test_two_states_matrix_riccati(theta, rho):
    """Two states, dX = (-0.5 X + Y + D) dt + dW0, dY = (-Y + 0.5 D) dt, X(0) = 0.2, Y(0) = -0.4, a target on X and a
    penalty on Y: the matrix Riccati closed form to 1e-11 (2.4e-12 seen), at rho = 0 and 0.8."""
    d = {"name": "lq2", "channels": ["w0", "w1"],
         "states": {"X": {"drift": {"X": -0.5, "Y": 1.0, "D": 1.0}, "noise": {"w0": 1.0}, "initial": 0.2}, "Y": {"drift": {"Y": -1.0, "D": 0.5}, "initial": -0.4}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w1": 1.0}}},
                          "loss": [[1.0, "X", "X"], [-2.0 * theta, "X"], [0.3, "Y", "Y"], [0.4, "D", "D"]]}},
         "horizon": {"kind": "finite", "T": 1.5, "discount": rho}, "numerics": {"nodes": 12}}
    res = ns.solve(d).require_converged()
    xs, us, J, V0 = riccati([[-0.5, 1.0], [0.0, -1.0]], [1.0, 0.5], np.diag([1.0, 0.3]), [theta, 0.0], 0.4, rho, [0.2, -0.4], 1.5, res.mean_times)
    assert np.abs(res.means["X"] - xs[0]).max() < 1e-11 and np.abs(res.means["Y"] - xs[1]).max() < 1e-11 and np.abs(res.means["D"] - us).max() < 1e-11
    assert abs(res.cost_parts["a"]["mean"] - J) < 1e-10


def test_targets_scale_the_means_and_leave_the_kernels_and_the_examples():
    """The means are linear in the targets and the mean cost quadratic (Jbar at target 2 is 4 x Jbar at target 1); the
    kernels and the variance part are identical to the bit, since the solve never sees the targets.  Without a driver
    every mean is exactly zero, with no solve, and the shipped finite examples' costs are what they were (identical to the
    release before the means to every digit, the cell engine included)."""
    r1 = ns.solve(ch1_targets(3.0, nodes=8)).require_converged(); r2 = ns.solve(ch1_targets(3.0, nodes=8, b=(2.0, -2.0))).require_converged()
    assert np.abs(r2.means["D1"] - 2 * r1.means["D1"]).max() < 1e-12 and abs(r2.cost_parts["player1"]["mean"] - 4 * r1.cost_parts["player1"]["mean"]) < 1e-12
    assert np.array_equal(r1.world, r2.world) and r1.cost_parts["player1"]["variance"] == r2.cost_parts["player1"]["variance"]
    off = engines.solver(ch1_targets(3.0, nodes=8)).solve(diagnostics=False)
    assert np.array_equal(off.means["D1"], r1.means["D1"]) and off.costs == r1.costs and off.mean_times.shape == (8,)
    for f, costs in (("ch1_two_player_finite.yaml", {"player1": 0.396905768985999, "player2": 0.396905768985999}),
                     ("ch1_delayed_finite.yaml", {"player1": 0.4838242662682089, "player2": 0.476678011994978})):
        res = ns.solve(os.path.join(EX, f)).require_converged()
        assert all(abs(res.costs[k] - v) < 1e-14 for k, v in costs.items())
        assert all(np.array_equal(v, np.zeros(len(res.mean_times))) for v in res.means.values()) and set(res.model.control_names) <= set(res.means)
        assert all(p["mean"] == 0.0 and p["variance"] == res.costs[k] for k, p in res.cost_parts.items())
        assert "mean" not in res.summary().split("\n")[1] and "means at" not in res.summary()
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d["horizon"] = {"kind": "finite", "T": 1.0}; d["numerics"] = {"engine": "cells", "nodes": 20}
    rc = ns.solve(d).require_converged()
    assert all(np.array_equal(v, np.zeros(20)) for v in rc.means.values()) and rc.cost_parts["player1"]["mean"] == 0.0


def test_no_information_limit_is_the_open_loop_path():
    """With signals that carry nothing (precision 1e-8) the kernels are zero and the mean paths are the open-loop Nash
    equilibrium of the deterministic game, Dbar1(t) = (T - t) b1 / r = 10 (1 - t), to O(p) (6.5e-9 seen); the mean cost
    is r int (10 (1 - t))^2 dt = 10/3."""
    res = ns.solve(ch1_targets(1e-8, nodes=8)).require_converged()
    assert np.abs(res.means["D1"] - 10.0 * (1.0 - res.mean_times)).max() < 1e-7 and abs(res.cost_parts["player1"]["mean"] - 10.0 / 3.0) < 1e-7


def delayed_with_targets():
    """The delayed example (a control lag and a delayed observation) with opposite targets and X(0) = 0.3."""
    d = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml"))
    d["agents"]["player1"]["loss"].append([-2.0, "X"]); d["agents"]["player2"]["loss"].append([2.0, "X"]); d["states"]["X"]["initial"] = 0.3
    return ns.Model.from_dict(d)


def test_delayed_rows_and_lags_keep_the_formulation():
    """A control lag in the drift and a delayed observation do not change the mean paths' formulation: the means are
    finite, the residual of the assembled mean system is at round-off (6e-16 seen), the mean dynamics dX/dt = D1(t - tau)
    + D2(t - tau) from X(0) = 0.3 hold against solve_ivp panel by panel (2.5e-12), and the mean first-order condition is
    the derivative of the mean cost: along three smooth deviations of player 1's path, the world answering through the
    response operator (a different line geometry from the continuation's), the central difference of the mean cost is
    zero to 1e-10 (1e-11 seen).  A control acting after the lag is idle within the last lag: Dbar1 = 0 on (T - tau, T]."""
    m = delayed_with_targets(); S = SpectralFiniteSolver(m); res = S.solve().require_converged(); c = S.c
    assert all(np.isfinite(v).all() for v in res.means.values()) and res.means["X"][0] == 0.3 and np.abs(res.means["D1"]).max() > 1
    M, b = S.mean_system(res.maps); zbar = S.solve_means(res.maps)
    assert np.abs(M @ zbar - b).max() < 1e-13 * np.abs(b).max()
    a1 = m.agents[0]; R = c.closed_loop(res.maps, excluded=a1.name, impulse_controls=a1.controls)[:, c.nW:]
    resp = RespOps(S, a1, R)
    for delta in (np.ones(c.Nt), np.sin(3 * c.tm), (c.tm - 0.5) ** 2):
        dev = np.concatenate([Zp[c.diag] for Zp in resp.apply(0, c.mean_embed @ delta)])
        eps = 1e-3
        assert abs(S.mean_cost(a1, zbar + eps * dev) - S.mean_cost(a1, zbar - eps * dev)) / (2 * eps) < 1e-10
    tau, bp, nt = 0.25, c.g.bp, c.g.nt
    x = np.array([0.3]); worst = 0.0
    for p in range(c.g.P):
        tn = c.tm[p * nt:(p + 1) * nt]
        sol = solve_ivp(lambda t, x: res.mean("D1", t - tau) + res.mean("D2", t - tau), (bp[p], bp[p + 1]), x, rtol=1e-12, atol=1e-14, t_eval=tn)
        worst = max(worst, np.abs(sol.y[0] - res.means["X"][p * nt:(p + 1) * nt]).max()); x = sol.y[:, -1]
    assert worst < 1e-9
    assert np.abs(res.mean("D1", [0.76, 0.9, 1.0])).max() == 0.0 and abs(res.mean("D1", 0.74)[0]) > 0.05
    assert "variance" in res.summary().split("\n")[1] and "means at t = 0" in res.summary()


def test_ties_share_the_means_and_opposite_targets_cannot_be_tied():
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d.setdefault("numerics", {})["nodes"] = 8
    d["agents"]["player1"]["loss"].append([-2.0, "X"]); d["agents"]["player2"]["loss"].append([-2.0, "X"]); d["ties"] = [["player1", "player2"]]
    tied = ns.solve(d).require_converged(); assert np.abs(tied.means["D1"] - tied.means["D2"]).max() < 1e-13 and tied.means["D1"][0] > 1
    d["ties"] = []; free = ns.solve(d).require_converged(); assert np.abs(free.means["D1"] - tied.means["D1"]).max() < 1e-6
    d["agents"]["player2"]["loss"][-1] = [2.0, "X"]; d["ties"] = [["player1", "player2"]]
    with pytest.raises(ValueError, match="not structurally identical"):
        ns.Model.from_dict(d)


def test_initial_state_spec_payload_and_cli(tmp_path):
    m = one_state(-1.0, 0.5, 1.0, 0.0, 0.7)
    assert m.states[0].initial == 0.7 and m.to_dict(numeric=True)["states"]["X"]["initial"] == 0.7 and m.drives_means
    assert ns.Model.from_dict(m.to_dict()).states[0].initial == 0.7 and any("initial value 0.7 moves only the means" in n for n in m.notes)
    assert ns.ModelBuilder("b").channel("w0", "w1").state("X", {"X": -1.0, "D": 1.0}, {"w0": 1.0}, initial=0.7).to_dict()["states"]["X"]["initial"] == 0.7
    d = m.to_dict(); d["states"]["X"]["initial"] = "x0"; d["params"] = {"x0": 0.7}
    assert ns.Model.from_dict(d).states[0].initial == 0.7
    for bad, msg in ((True, "must be a number"), ("nan", "unknown parameter"), ({"X": 1.0}, "coefficient")):
        b = m.to_dict(); b["states"]["X"]["initial"] = bad
        with pytest.raises(ValueError, match=msg):
            ns.Model.from_dict(b)
    n = ns.Model.from_dict(m.to_dict()); n.states[0].initial = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        n.validate()
    b = m.to_dict(); b["states"]["X"]["init"] = 1.0
    with pytest.raises(ValueError, match="unknown key"):
        ns.Model.from_dict(b)
    with pytest.raises(ValueError, match="no meaning in a stationary model"):
        m.with_stationary(4.0)
    res = ns.solve(m).require_converged()
    d = json.loads(json.dumps(res.to_dict()))
    assert d["means"]["X"] == res.means["X"].tolist() and d["mean_times"] == res.mean_times.tolist() and d["cost_parts"]["a"]["mean"] == res.cost_parts["a"]["mean"]
    assert ns.Model.from_dict(d["model"]).states[0].initial == 0.7
    path = tmp_path / "lq.yaml"; out = tmp_path / "lq.json"
    import yaml
    path.write_text(yaml.safe_dump(m.to_dict()))
    assert main(["solve", str(path), "-o", str(out)]) == 0 and json.load(open(out))["means"]["X"][0] == 0.7
    pytest.importorskip("matplotlib")
    res.plot(str(tmp_path / "lq.png")); assert (tmp_path / "lq.png").stat().st_size > 0
