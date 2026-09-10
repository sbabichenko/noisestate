"""TransitionResult (stage 3): the loss paths E[loss(t)] by row quadrature, the excess costs over the new
stationary flow, the belief-error variances, the plot, the payload, and the sweeps over T and over a change
size with warm starts."""
import json
import numpy as np, pytest
from scipy.integrate import solve_ivp
import noisestate as ns
from noisestate.results import TransitionResult
from helpers import EX, IVP, A1 as a, H1 as h, T1, P0, prior_model, slow, example, stationary


@slow("slow (9 s at 16 nodes; the loss path is pinned by the regime change below); set NOISESTATE_SLOW=1")
def test_same_model_loss_path_is_the_stationary_flow():
    """Chapter 3 as its own past and continuation (T = 6, L = 3, 16 nodes) from the stationary maps (3 evaluations):
    E[loss(t)] equals the stationary flow on every time node, [0, T] and the buffer, to 7e-11 for both agents,
    the excess costs are 5e-11 and 4e-11, and the discounted integral of the path over [0, T] (time_mass) is
    res.costs to 3e-14 (the row quadrature integrates products of interpolants exactly)."""
    m = example("ch3_two_player")
    stat = stationary(m, 16)
    res = ns.solve(m.with_horizon(kind="finite", window=6.0).with_numerics(nodes=16), past=stat, continuation=stat, start="stationary", tol=1e-10).require_converged()
    assert isinstance(res, TransitionResult) and res.kind == "transition" and res.evaluations <= 4
    assert res.times.shape == (res.compiled.Nt,) and res.times[0] == 0.0 and res.times[-1] == 9.0 and res.stationary is stat
    assert res.old_flows == stat.costs and res.new_flows == stat.costs
    for ag in ("player1", "player2"):
        lp = res.loss_path[ag]; flow = stat.costs[ag]
        assert np.abs(lp - flow).max() / flow < 1e-9, (ag, np.abs(lp - flow).max() / flow)
        assert abs(res.excess_costs[ag]) < 1e-9, res.excess_costs
        assert abs(res.compiled.time_mass(0.0) @ lp - res.costs[ag]) < 1e-12
    d = json.loads(json.dumps(res.to_dict()))
    assert d["engine"] == "spectral" and d["kind"] == "transition" and d["grid"]["kind"] == "transition" and len(d["times"]) == res.compiled.Nt
    assert set(d["loss_path"]) == {"player1", "player2"} and set(d["excess_costs"]) == {"player1", "player2"}
    assert d["old_flows"] == stat.costs and d["new_flows"] == stat.costs and d["past"]["kind"] == "stationary" and "continuation" in d
    assert "excess cost over the new stationary flow" in res.summary()


def test_regime_change_loss_path_runs_from_the_old_state_to_the_new_flow(tmp_path):
    """Chapter 3, p1 = 3 to 10, T = 9, 12 nodes, from the stationary maps (20 evaluations): at t = 0 the state's
    variance is the old regime's (E X^2 = 0.75748 to 1e-8) while the controls have jumped, so E[loss(0+)]
    (0.43506) is not the old flow (0.42895); at T and at the buffer's end the path is the new stationary flow
    (0.42174) to 1e-7 (settled 2.7e-6); the excess costs are 0.024027 and 0.028619 (0.023848 and 0.028343 at 8
    nodes), the path's discounted integral is res.costs to 1.6e-6 (the transient's product interpolated on the
    time nodes; exact for a constant path); the plot writes."""
    m = example("ch3_two_player")
    old = ns.solve(m).require_converged()
    res = ns.solve(m.with_params(p1=10.0).with_horizon(kind="finite", window=9.0).with_numerics(nodes=12), past=old, continuation="stationary",
                   start="stationary").require_converged()
    assert res.settled < 1e-5
    c = res.compiled; I, w = c.g.row_quadrature(0.0, +1)
    K = res.kernel("X"); Ko = old.kernel("X")
    assert abs(w @ ((I @ K) ** 2).sum(1) - (old.compiled.grid.mass_matrix @ Ko * Ko).sum()) < 1e-8
    iT = int(np.argmin(np.abs(res.times - 9.0)))
    for ag in ("player1", "player2"):
        lp = res.loss_path[ag]; new = res.new_flows[ag]
        assert abs(lp[iT] - new) / new < 1e-7 and abs(lp[-1] - new) / new < 1e-7, (ag, lp[iT], lp[-1], new)
        assert abs(c.time_mass(0.0) @ lp - res.costs[ag]) < 5e-6            # the transient's product interpolated on the time nodes
        assert abs(res.excess_costs[ag] - c.time_mass(0.0) @ (lp - new)) < 1e-12
    assert abs(res.loss_path["player1"][0] - 0.43506) < 1e-4 and abs(res.old_flows["player1"] - 0.42895) < 1e-4
    assert abs(res.loss_path["player2"][0] - 0.47182) < 1e-4
    assert abs(res.excess_costs["player1"] - 0.024027) < 1e-5 and abs(res.excess_costs["player2"] - 0.028619) < 1e-5
    be = res.belief_error("player2", "X")
    assert be.shape == res.times.shape and be.min() > 0 and be[0] > be[iT]                 # a sharper signal: the error falls
    with pytest.raises(KeyError, match="no agent"):
        res.belief_error("player3", "X")
    res.plot(str(tmp_path / "transition.png"))
    assert (tmp_path / "transition.png").stat().st_size > 1000


def test_belief_error_is_the_kalman_variance(P0=P0):
    """The one-agent prior start of tests/test_finite_discount.py's model (X(0) ~ N(0, P0), 16 nodes): the agent's
    belief error of X is the Kalman variance P(t) from P(0) = P0 (unobserved prior; monotone falling from 0.8)
    or from P(0) = 0 (the prior observed at once) to 3.4e-6 on every time node."""
    Pf = {inf: solve_ivp(lambda t, y: [2 * a * y[0] - h * h * y[0] ** 2 + 1], (0.0, T1), [0.0 if inf else P0], dense_output=True, **IVP)
          for inf in (False, True)}
    for informed in (False, True):
        shock = {"name": "xi", "loads": {"X": np.sqrt(P0)}, **({"rows": {"a.y": 1.0}} if informed else {})}
        res = ns.solve(prior_model(0.5, 16), past=[shock]).require_converged()
        assert isinstance(res, TransitionResult) and res.excess_costs == {} and res.new_flows == {} and res.old_flows == {}
        be = res.belief_error("a", "X"); P = Pf[informed].sol(res.times)[0]
        assert np.abs(be - P).max() < 1e-5, np.abs(be - P).max()
        assert abs(be[0] - (0.0 if informed else P0)) < 1e-12
        if not informed:
            assert np.all(np.diff(be) <= 1e-9) and abs(be[-1] - 0.54656) < 1e-4     # the stationary Kalman variance
        assert abs(res.compiled.time_mass(0.5) @ res.loss_path["a"] - res.costs["a"]) < 1e-7      # 2e-8: one panel of 16 nodes
        assert res.to_dict()["excess_costs"] == {}


def test_sweeps_over_the_horizon_and_over_the_change_size():
    """sweep(transition file, "horizon.window", [6, 9]) warm-starts T = 9 from the T = 6 maps read on the new grid
    (the stationary maps beyond): 8 evaluations against 21 from the stationary maps alone, the same equilibrium
    to 2e-8, one past shared; sweep over p1 solves the past once and warm-starts each point on the same grid."""
    m = example("ch3_two_player")
    d = m.with_params(p1=10.0).to_dict()
    d["horizon"] = {"kind": "transition", "window": 6.0, "past": {"model": EX + "ch3_two_player.yaml"}}; d["numerics"] = {"nodes": 8}
    rows = ns.sweep(d, "horizon.window", [6.0, 9.0], solve_kw={"start": "stationary"})
    assert [r["value"] for r in rows] == [6.0, 9.0] and all(r["converged"] for r in rows) and rows[1]["change"] is None
    assert rows[0]["result"].past is rows[1]["result"].past and rows[1]["result"].model.horizon.window == 9.0
    alone = ns.solve({**d, "horizon": {**d["horizon"], "window": 9.0}}, start="stationary")
    assert rows[1]["evaluations"] < alone.evaluations // 2 and np.abs(alone.world - rows[1]["result"].world).max() < 1e-6
    rows = ns.sweep(d, "p1", [7.0, 10.0])
    assert rows[0]["result"].past is rows[1]["result"].past and rows[1]["change"] is not None and all(r["converged"] for r in rows)
    assert rows[0]["result"].continuation is not rows[1]["result"].continuation
    with pytest.raises(ValueError, match="horizon.window"):
        ns.sweep(d, "p9", [1.0])
