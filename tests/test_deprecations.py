"""The names deprecated since one Result (0.5) warn once each and still work; the cell engine's kernel stack and
its mean solve's rcond guard; the TypeError for a Settings field passed to solve() names the settings route."""
import warnings
import numpy as np
import pytest
import noisestate as ns
from noisestate import Numerics, Settings
from helpers import example, example_path


@pytest.fixture(scope="module")
def stat():
    return ns.solve(example("ch3_two_player").with_numerics(nodes=6))


def _one(match, fn):
    with pytest.warns(DeprecationWarning, match=match) as rec:
        out = fn()
    assert sum(issubclass(w.category, DeprecationWarning) for w in rec) == 1
    return out


def test_result_aliases_warn_once_and_read_the_new_names(stat):
    assert _one("res.iterations", lambda: stat.iterations) == stat.evaluations
    assert _one("res.Z", lambda: stat.Z) is stat.world
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # the new names are silent
        stat.evaluations; stat.world


def test_solve_aliases_warn_once_and_land_in_the_numerics():
    m = example("ch3_two_player")
    r = _one(r"solve\(nodes=\)", lambda: ns.solve(m, nodes=6))
    assert r.numerics.nodes == 6
    r = _one(r"solve\(settings=\)", lambda: ns.solve(m, Numerics(nodes=6), settings=Settings(anderson_m=3)))
    assert r.settings.anderson_m == 3


def test_transition_aliases_warn_before_the_solve():
    m = example("ch3_two_player")
    with pytest.raises(TypeError):                       # the past is not a stationary result, model or Past
        _one(r"transition\(nodes=\)", lambda: ns.transition(object(), m, 4.0, nodes=4))
    with pytest.raises(TypeError):
        _one(r"transition\(stationary=\)", lambda: ns.transition(object(), m, 4.0, stationary={"nodes": 4}))


@pytest.mark.parametrize("name", ["StationaryResult", "TriangleResult", "TransitionResult", "CellResult"])
def test_result_subclasses_by_name_warn_and_stay_importable(name, stat):
    from noisestate import results
    cls = _one(f"noisestate.{name}", lambda: getattr(ns, name))
    assert cls is getattr(results, name) and issubclass(cls, ns.Result)
    assert name not in ns.__all__
    with pytest.raises(AttributeError):
        ns.NoSuchResult


def test_a_settings_field_passed_to_solve_names_the_settings_route():
    with pytest.raises(TypeError, match=r"Numerics\(settings=\{'anderson_m': \.\.\.\}\)"):
        ns.solve(example_path("ch3_two_player"), anderson_m=3)
    with pytest.raises(TypeError, match=r"Numerics\(damping=\.\.\.\)"):
        ns.solve(example_path("ch3_two_player"), damping=0.5)


@pytest.fixture(scope="module")
def cells():
    d = example("ch1_two_player_finite").to_dict(); d["numerics"] = {"engine": "cells", "nodes": 8}
    return ns.solve(d)


def test_cell_engine_kernel_without_a_channel_is_the_stack_over_channels(cells):
    K = cells.kernel("X"); N = cells.compiled.N
    assert K.shape == (N, N, len(cells.channels))
    for k, ch in enumerate(cells.channels):
        assert np.array_equal(K[..., k], cells.kernel("X", ch))
    assert cells.action_kernel("D1").shape == (N, N, len(cells.channels))


def test_cell_engine_mean_solve_refuses_a_singular_mean_system():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from ch1_mean_sweep import model as ch1_targets
    d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "window": 1.0}
    d["numerics"] = {"engine": "cells", "nodes": 8, "settings": {"mean_rcond": 1.0}}      # every system fails a threshold of 1
    with pytest.raises(ValueError, match="the mean system is singular"):
        ns.solve(d)
