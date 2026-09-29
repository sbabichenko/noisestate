"""An instant observer's best response: its map on its rows, with the reaction its loss fixes in its passive world.

`instant: [u]` gives the observer's control v the loading h = -(G^vv)^-1 G^vu on the level of u at the same instant
(compile.instant_loads) and leaves the rest of v to its map on its rows.  The best response is then the best map with
that reaction in place.  It used to be computed with the reaction switched off together with the map (the closed
loop's `excluded`): the best function of the rows ALONE, from which h u was subtracted and the remainder projected on
the rows.  Where u's level is in the span of the observer's rows (Chapter 6's markets: the trader sees the flow, or
filters the quote as a level row) the two agree.  Where it is not, the remainder h (u - E[u | rows]) is not
representable (representation error 0.37 at every node count), and the two iteration variables settled on two
different fixed points, neither of them the equilibrium (costs 3e-3 apart).

The game: X' = -0.5 X + D0 + D1 + w0; a0 sees 2 X + w1 with loss X^2 + D0^2; a1 sees X + w2 with loss X^2 + D1^2 +
0.3 D0^2 + 0.6 D1 D0 and reacts to D0 at once (h = -0.3).  a1's rows carry no copy of D0, which moves with a0's private
signal.

References, independent of the package: extras/fuzz's discrete brute-force reference with the observer's control
h D0 + its map at the same step (Richardson over 240/480/960 steps, error estimate 1.2e-5; finite 40/80/160 steps,
1.4e-5), and, for the equilibrium property itself, central differences of the agents' expected costs along a change
of one agent's RAW map through the actual closed loop (no first-order operator)."""
import numpy as np
import pytest

import noisestate as ns


def game(kind="stationary", nodes=20, rho=0.3, window=6.0, monitors=()):
    d = {"name": "instant", "shocks": ["w0", "w1", "w2"],
         "states": {"X": {"drift": {"X": -0.5, "D0": 1.0, "D1": 1.0}, "noise": {"w0": 1.0}}},
         "agents": {"a0": {"controls": ["D0"], "signals": {"y": {"drift": {"X": 2.0}, "noise": {"w1": 1.0}}},
                           "loss": [[1.0, "X", "X"], [1.0, "D0", "D0"]]},
                    "a1": {"controls": ["D1"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w2": 1.0}}},
                           "loss": [[1.0, "X", "X"], [1.0, "D1", "D1"], [0.3, "D0", "D0"], [0.6, "D1", "D0"]],
                           "instant": ["D0"]}},
         "numerics": {"nodes": nodes}}
    for who, whom in monitors:
        d["agents"][who]["monitors"] = [whom]
    d["horizon"] = ({"kind": "stationary", "window": window, "discount": rho} if kind == "stationary"
                    else {"kind": "finite", "T": 1.0, "discount": rho})
    with pytest.warns(UserWarning, match="privy") if not monitors else _nothing():
        return ns.Model.from_dict(d)


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("kind, ref, tol", [("stationary", {"a0": 0.74516970, "a1": 0.74047157}, 4e-5),
                                            ("finite", {"a0": 0.2998536, "a1": 0.2994444}, 2e-5)])
def test_the_instant_observers_equilibrium_is_the_references(kind, ref, tol):
    """Measured: stationary 0.745169719 / 0.740471594 (the reference's limit to 2e-9), finite 0.299853628 / 0.299444491
    (7e-8).  Before: stationary actions 0.7477594 / 0.7382325 and maps 0.7471598 / 0.7404032; finite a0 0.2998604."""
    m = game(kind, nodes=20 if kind == "stationary" else 10)
    for variable in ("actions", "maps"):
        res = ns.solve(m, variable=variable).require_converged()
        for a, v in ref.items():
            assert res.costs[a] == pytest.approx(v, abs=tol), (variable, a)


@pytest.mark.parametrize("kind", ["stationary", "finite"])
def test_actions_and_maps_agree_and_the_maps_represent_the_action(kind):
    """The representation error is the projection's again (2e-6, 5e-11 at 14 and 20 nodes; it was 0.37 at every count)
    and the two iteration variables reach one equilibrium."""
    m = game(kind, nodes=20 if kind == "stationary" else 10)
    a, b = ns.solve(m, variable="actions"), ns.solve(m, variable="maps")
    assert max(a.representation_error.values()) < 1e-8 and max(b.representation_error.values()) < 1e-8
    for k in a.costs:
        assert abs(a.costs[k] - b.costs[k]) < 1e-9


def test_no_agent_gains_by_changing_its_raw_map():
    """The equilibrium property, by brute force: each agent's expected cost is flat (to the window's and the quadrature's
    floor, the same as without the instant observation) along a change of its raw map, the others' maps held, the world
    re-solved by the closed loop.  At rho = 0 (the reported flow cost is the objective) on a window of 10.  Before the
    fix the slopes were 8e-3 to 1.6e-2 against curvatures near 1."""
    res = ns.solve(game(nodes=20, rho=0.0, window=10.0)).require_converged()
    S, c = res._solver(), res.compiled
    a = c.grid.nodes
    for agent in res.model.agents:
        g = res.maps[agent.name]
        dg = np.zeros_like(g); dg[0, 0] = np.sin(2 * a) * np.exp(-a)

        def J(e):
            maps = dict(res.maps); maps[agent.name] = g + e * dg
            return S.expected_cost(agent, c.closed_loop(maps))
        slope = (J(1e-3) - J(-1e-3)) / 2e-3
        curvature = (J(1e-3) - 2 * J(0.0) + J(-1e-3)) / 1e-6
        assert curvature > 0.5 and abs(slope) < 1e-5 * curvature, agent.name


@pytest.mark.parametrize("kind", ["stationary", "finite"])
def test_a_privy_instant_observer_answers_the_deviation_it_draws_itself(kind):
    """a0 and a1 privy to each other: a1's frozen spike draws a0's response kernel, which a1 then answers at once (h),
    a fixed reaction and not a choice the envelope drops.  Its first-order condition counted that reaction's effect
    through the others' quantities but not through its own atoms; res.foc_residual (the package's independent
    quadrature of the cost's derivative) was 3.9e-2 (stationary) and 2.5e-2 (finite).  Measured after: 2.5e-12, 3.8e-9."""
    res = ns.solve(game(kind, nodes=20 if kind == "stationary" else 10, monitors=(("a0", "a1"), ("a1", "a0")))).require_converged()
    assert res.foc_residual("a1", seed="a0").relative < 1e-7
    assert res.foc_residual("a0", seed="a1").relative < 1e-7
    assert max(res.representation_error.values()) < 1e-8
