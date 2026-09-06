"""Cyclic symmetry: detection from the ties, and the block-diagonalised closed loop against the dense
and the eliminated solves, for the full world and for every agent's passive world."""
import numpy as np
import noisestate as ns
from noisestate.symmetry import find_cyclic_symmetry
from make_ch5_cycle_market import build
from helpers import example, example_dict, slow


def test_detection():
    for Nf in (2, 3, 4):
        m = build(N=Nf, nodes=4).build(); s = find_cyclic_symmetry(m)
        assert s is not None and s.order == Nf and len(s.orbits) == 4 and s.fixed_primaries(m.state_names + m.control_names) == ["q"]
    assert find_cyclic_symmetry(example("ch3_two_player")) is None                 # no ties
    d = example_dict("ch3_two_player"); d["params"]["p2"] = d["params"]["p1"]; d["params"]["r2"] = d["params"]["r1"]; d["ties"] = [["player1", "player2"]]
    s = find_cyclic_symmetry(ns.Model.from_dict(d)); assert s is not None and s.order == 2 and s.orbits == [["D1", "D2"]]


def test_symmetric_closed_loop_matches_dense_and_eliminated():
    for Nf in (2, 3):
        m = build(N=Nf, nodes=5).build(); S = ns.StationarySolver(m); c = S.c
        g, _ = S.best_response(m.agents[0], S.zero_maps()); maps = {a.name: g.copy() for a in m.agents}
        for a in [None] + list(m.agents):
            kw = {} if a is None else {"excluded": a.name, "impulse_controls": a.controls}
            Zs, Ze, Zd = c.closed_loop_symmetric(maps, **kw), c._closed_loop_eliminated(maps, **kw), c.closed_loop_dense(maps, **kw)
            assert np.abs(Zs - Ze).max() < 1e-12 and np.abs(Ze - Zd).max() < 1e-12
        # asymmetric maps on a tied model fall back to the general solve
        maps2 = dict(maps); maps2[m.agents[1].name] = 1.1 * maps[m.agents[1].name]
        assert np.abs(c.closed_loop(maps2) - c._closed_loop_eliminated(maps2)).max() < 1e-12


@slow("slow (5 s; the symmetric closed loop is pinned to 1e-12 above at 5 nodes); set NOISESTATE_SLOW=1")
def test_solve_uses_the_symmetric_path_and_reproduces_the_equilibrium():
    m = build(N=3, nodes=6).build(); S = ns.StationarySolver(m); calls = [0]
    orig = S.c.closed_loop_symmetric
    def counted(*a, **k):
        calls[0] += 1; return orig(*a, **k)
    S.c.closed_loop_symmetric = counted
    r = S.solve().check(); assert calls[0] > 0
    S2 = ns.StationarySolver(m); S2.c.sym = None; r2 = S2.solve().check()          # the general path
    assert abs(r.costs["firm0"] - r2.costs["firm0"]) < 1e-9 and np.abs(r.maps["firm0"] - r2.maps["firm0"]).max() < 1e-8
