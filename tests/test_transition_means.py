"""The mean paths of a transition with a stationary continuation (stage 3): the system on the time line
(the strip is cut at age L below T, so the line s = 0 does not reach the horizon), from the past's constant
means, closed on the buffer with the new stationary means."""
import numpy as np
import noisestate as ns

EX = ns.__file__.rsplit("/noisestate/", 1)[0] + "/examples/"


def with_target(theta):
    d = ns.load(EX + "ch3_two_player.yaml").to_dict()
    d["agents"]["player1"]["loss"].append([-2.0 * theta, "X"])
    return ns.Model.from_dict(d)


def test_time_line_mean_system_equals_the_diagonal_one_where_both_exist():
    """T = L = 3 (no continuation), where the line s = 0 reaches T: the time-line system (the 1-D Volterra
    operator, the conditions on the age-0 line, the lagged atoms' means read at t - lag) and the diagonal one
    (the kernels' operators on the line s = 0) give the same mean paths to 1e-15."""
    mt = with_target(1.0)
    stat = ns.solve(mt.with_horizon(nodes=12)).check()
    S = ns.SpectralFiniteSolver(mt.with_horizon(kind="finite", window=3.0, nodes=12), past=stat)
    res = S.solve(); assert S.c.Nd == S.c.Nt
    Md, bd = S._mean_system_diag(res.maps); Ml, bl = S._mean_system_line(res.maps)
    assert np.abs(np.linalg.solve(Md, bd) - np.linalg.solve(Ml, bl)).max() < 1e-13


def test_same_model_means_are_the_stationary_constants():
    """Chapter 3 with a target -2 X for player1, its stationary solution (16 nodes) as past and continuation,
    T = 6 (twice L; a NotImplementedError before): the mean paths are the stationary means on every time node
    of [0, T] to 4.7e-10 (1.4e-7, 6.3e-7 and 1.1e-7 at 12 nodes: the two discretisations of the continuation)
    and exactly so on the buffer, the mean part of the cost is T times the stationary mean flow to 1e-11, the
    excess costs are 7e-11, and res.mean() reads the path beyond the window."""
    mt = with_target(1.0)
    stat = ns.solve(mt.with_horizon(nodes=16)).check()
    res = ns.solve(mt.with_horizon(kind="finite", window=6.0, nodes=16), past=stat, continuation=stat, start="stationary").check()
    assert res.iterations <= 2 and res.means_t.shape == (res.compiled.Nt,)
    buf = res.means_t >= 6.0 + 1e-9
    for n in ("X", "D1", "D2"):
        assert np.abs(res.means[n] - stat.means[n]).max() < 1e-8, (n, np.abs(res.means[n] - stat.means[n]).max())
        assert np.abs(res.means[n][buf] - stat.means[n]).max() == 0.0
        assert np.abs(res.mean(n, [0.0, 2.5, 4.0, 6.0, 8.0]) - stat.means[n]).max() < 1e-8
    for ag in ("player1", "player2"):
        assert abs(res.cost_parts[ag]["mean"] - 6.0 * (stat.costs[ag] - stat.cost_parts[ag]["variance"])) < 1e-10
        assert abs(res.excess_costs[ag]) < 1e-9
    assert "means at t = 0, T/2, T" in res.summary()


def test_target_change_runs_from_the_old_means_to_the_new():
    """The target moves from 1 to 2 (12 nodes, T = 6): X's mean path starts at the old constant 0.8662 and rises
    monotonically to the new 1.7324, reached at T to 1e-6; D1 jumps at 0+ (4.1115 against the old 1.7989) and
    falls to the new 3.5979; the buffer carries the new constants, and the path's gap at T- (7.7e-4 on X: the
    means settle more slowly than the maps, which are at 6.3e-7) is in `settled`; the game ending at T instead
    (a past shorter than T, no continuation) solves too, with the controls vanishing at T."""
    old = ns.solve(with_target(1.0).with_horizon(nodes=12)).check()
    res = ns.solve(with_target(2.0).with_horizon(kind="finite", window=6.0, nodes=12), past=old, continuation="stationary",
                   start="stationary").check()
    new = res.continuation.means
    assert abs(old.means["X"] - 0.86621659) < 1e-7 and abs(new["X"] - 1.73243318) < 1e-7 and abs(new["D1"] - 3.59787014) < 1e-7
    c = res.compiled; on = np.arange(c.Nt) < c.P_T * c.g.nt              # the time nodes of [0, T]; the rest the buffer's
    X = res.means["X"]
    assert abs(X[0] - old.means["X"]) < 1e-12 and abs(res.mean("X", 6.0)[0] - new["X"]) < 1e-6
    assert np.all(np.diff(X[on]) > -1e-9) and abs(res.mean("X", 2.0)[0] - 1.66309) < 1e-4 and abs(res.mean("X", 3.0)[0] - 1.71283) < 1e-4
    assert abs(res.mean("D1", 0.0)[0] - 4.11152) < 1e-4 and abs(res.mean("D1", 6.0)[0] - new["D1"]) < 1e-6
    assert abs(res.mean("D2", 0.0)[0] + 3.01782) < 1e-4
    for n in ("X", "D1", "D2"):
        assert np.abs(res.means[n][~on] - new[n]).max() == 0.0
    # the means settle more slowly than the maps: X's path at T- is 7.7e-4 below the new constant (the buffer's
    # frozen value), which `settled` reports (1.9e-4 of the largest mean, D1 at 0+) where the maps alone sit at 6.3e-7
    assert abs(X[on][-1] - new["X"] + 7.7e-4) < 1e-5 and abs(res.mean("X", 6.0)[0] - new["X"]) < 1e-12
    assert 1.5e-4 < res.settled < 2.5e-4 and res.cost_parts["player1"]["mean"] > 0
    assert res.settled == max(res._make_solver(res.model).settled(res.maps), res._make_solver(res.model).settled_means(res.means))
    end = ns.solve(with_target(1.0).with_horizon(kind="finite", window=6.0, nodes=12), past=old).check()
    assert isinstance(end, ns.TransitionResult) and end.continuation is None
    assert abs(end.mean("X", 0.0)[0] - old.means["X"]) < 1e-12 and abs(end.mean("D1", 6.0)[0]) < 1e-12
    assert abs(end.mean("D1", 1.0)[0] - 1.7997) < 1e-3 and end.mean("D1", 5.0)[0] < 1.0
