"""The solver's quality-of-life surface: names that are not there say so at once with the nearest ones; the repr and
summary name what failed; a near miss is told from a failure; a result saves and reloads; verbose shows the phases;
the tuning constants stay out of completion; foc_residual checks Chapter 6's response kernels independently."""
import sys, os

import numpy as np
import pytest

import noisestate as ns

sys.path.insert(0, os.path.dirname(__file__))
from test_monitoring import market  # noqa: E402


@pytest.fixture(scope="module")
def two():
    return ns.solve(ns.load(ns.example("ch1_two_player_finite")))


def _monitored(nodes=16):
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    d["params"]["p2"] = 10.0; d["agents"]["player2"]["monitors"] = "player1"
    return ns.solve(ns.Model.from_dict(d).with_numerics(nodes=nodes))


def test_a_misspelled_name_fails_at_once_with_the_nearest(two):
    with pytest.raises(ValueError, match="did you mean 'X'"):
        two.response("Xx", to="w0")                                   # not at .over(): at the call
    with pytest.raises(KeyError, match="did you mean 'w0'"):
        two.response("X", to="W0")
    with pytest.raises(KeyError, match="unknown agent 'player3'"):
        two.response("X", to="w0", seen_by="player3")
    with pytest.raises(KeyError, match="names with a mean are"):
        two.mean("Xx", 0.5)
    with pytest.raises(ValueError, match="did you mean 'D1'"):
        two.kernel("d1")
    try:
        two.kernel("X", "W0")
    except KeyError as e:
        assert str(e).startswith("unknown shock")                     # printed as written, not quoted


def test_solve_suggests_the_option_meant():
    m = ns.load(ns.example("ch1_two_player_finite"))
    with pytest.raises(TypeError, match="did you mean max_evaluations"):
        ns.solve(m, max_iter=2)
    with pytest.raises(TypeError, match="did you mean verbose"):
        ns.solve(m, verbos=True)


def test_status_tells_a_near_miss_from_a_failure(two):
    import copy
    r = copy.copy(two)
    assert r.status == "converged"
    tol = r.solve_kw["tol"]
    r.converged, r.residual = False, 5 * tol
    assert r.status == "near tolerance" and "near tolerance" in repr(r) and "not a failure" in r.summary()
    r.residual = 1e3 * tol
    assert r.status == "not converged" and "NOT converged (residual" in repr(r)


def test_repr_and_summary_name_the_failed_checks():
    res = ns.solve(market(0.1, nodes=16))
    text = repr(res)
    assert "failed:" in text and "window too short" in text and "publication" not in text
    lines = res.summary().splitlines()
    assert any(l.startswith("  - WINDOW TOO SHORT") for l in lines)


def test_tuning_constants_stay_out_of_completion(two):
    names = dir(two)
    assert "response" in names and "status" in names and "STABILITY_K" not in names
    assert two.STABILITY_K == two.settings.stability_k


def test_a_saved_result_reloads_without_a_full_solve(two, tmp_path):
    p = str(tmp_path / "r.json")
    two.save(p)
    back = ns.load_result(p)
    assert back.converged and back.evaluations <= 3 < two.evaluations
    assert back.costs == pytest.approx(two.costs, abs=1e-8)
    t = np.array([0.2, 0.7])
    assert np.allclose(back.response("X", to="w0").over(t), two.response("X", to="w0").over(t), atol=1e-7)


def test_verbose_shows_the_phases_and_the_outcome(capsys):
    ns.solve(ns.load(ns.example("ch3_two_player")), verbose=True)
    out = capsys.readouterr().out
    assert "-- anderson" in out and "eval    1" in out and "-- done in" in out and "converged" in out


def test_foc_residual_holds_for_the_privy_responder_and_converges_with_the_grid():
    coarse, fine = (_monitored(n).foc_residual("player2", seed="player1") for n in (12, 16))
    assert fine.relative < 1e-7 and fine.relative < coarse.relative
    assert _monitored(16).foc_residual("player1", seed="player1").relative < 1e-6


def test_foc_residual_on_the_market_and_its_refusals():
    res = ns.solve(market(0.1, nodes=24))
    assert res.foc_residual("trader", seed="mm").relative < 1e-8
    with pytest.raises(ValueError, match="privy"):
        res.foc_residual("mm", seed="trader")                            # nobody monitors the trader
    empty = ns.solve(market(0.0, nodes=16)).foc_residual("trader", seed="mm")
    assert np.isnan(empty.relative) and "nothing to check" in repr(empty)
