"""The spellings 0.5 deprecated are gone in 0.6, and each removal says what replaced it; the cell engine's
kernel stack and its mean solve's rcond guard; the TypeError for a Settings field passed to solve() names the
settings route."""
import warnings
import numpy as np
import pytest
import noisestate as ns
from noisestate import Numerics, Settings
from helpers import example, example_path


@pytest.fixture(scope="module")
def stat():
    return ns.solve(example("ch3_two_player").with_numerics(nodes=6))


def test_the_old_result_attributes_are_gone(stat):
    for name in ("iterations", "Z"):
        assert not hasattr(stat, name), name
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # and the names that replaced them warn about nothing
        assert stat.evaluations >= 1 and stat.world is not None


def test_solve_no_longer_takes_the_numerics_fields_directly():
    m = example("ch3_two_player")
    with pytest.raises(TypeError, match=r"Numerics\(nodes=\.\.\.\)"):
        ns.solve(m, nodes=6)
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(m, Numerics(nodes=6), settings=Settings(anderson_m=3))
    assert ns.solve(m, Numerics(nodes=6)).numerics.nodes == 6


def test_transition_no_longer_takes_the_numerics_fields_directly():
    old = ns.solve(example("ch3_two_player").with_numerics(nodes=6)).require_converged()
    new = example("ch3_two_player").with_params(p1=10.0)
    with pytest.raises(TypeError, match=r"Numerics\(nodes=\.\.\.\)"):
        ns.transition(old, new, 3.0, nodes=6)
    with pytest.raises(TypeError, match="unknown option"):
        ns.transition(old, new, 3.0, stationary={"nodes": 6})


@pytest.mark.parametrize("name", ["StationaryResult", "TriangleResult", "TransitionResult", "CellResult"])
def test_the_result_subclasses_are_no_longer_exported(name):
    with pytest.raises(AttributeError, match="removed in 0.6"):
        getattr(ns, name)
    assert name not in ns.__all__
    with pytest.raises(AttributeError):
        ns.NoSuchResult


@pytest.mark.parametrize("horizon, message", [
    ({"kind": "stationary", "window": 3.0, "nodes": 8}, "moved under numerics"),
    ({"kind": "finite_cells", "window": 1.0}, "numerics.engine"),
    ({"kind": "transition", "window": 3.0, "stationary": {"nodes": 8}}, "numerics.continuation_nodes"),
])
def test_the_old_model_file_spellings_name_what_replaced_them(horizon, message):
    d = {"channels": ["w"], "states": {"X": {"noise": {"w": 1.0}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1.0}, "noise": {"w": 1.0}}},
                          "loss": [[1.0, "X", "X"], [1.0, "D", "D"]]}},
         "horizon": horizon}
    with pytest.raises(ValueError, match=message):
        ns.Model.from_dict(d)


def test_with_horizon_no_longer_takes_the_numerics_fields():
    m = example("ch3_two_player")
    with pytest.raises(ValueError, match="with_numerics"):
        m.with_horizon(nodes=8)
    with pytest.raises(ValueError, match="engine='cells'"):
        m.with_horizon(kind="finite_cells")


def test_the_schema_no_longer_carries_the_old_keys():
    import json
    text = json.dumps(ns.schema("model"))
    assert "deprecat" not in text and "finite_cells" not in text
    assert ns.schema.validate({"horizon": {"nodes": 8}}, "model")


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
    assert cells.strategy_kernel("D1").shape == (N, N, len(cells.channels))


def test_cell_engine_mean_solve_refuses_a_singular_mean_system():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from ch1_mean_sweep import model as ch1_targets
    d = ch1_targets(10.0, nodes=12).to_dict(); d["horizon"] = {"kind": "finite", "window": 1.0}
    d["numerics"] = {"engine": "cells", "nodes": 8, "settings": {"mean_rcond": 1.0}}      # every system fails a threshold of 1
    with pytest.raises(ValueError, match="the mean system is singular"):
        ns.solve(d)


# ---------------------------------------------------------------- the 0.7 renames are gone
#  Deleted rather than aliased: nothing outside this repository imports noisestate, so no
#  transition was owed.  Each removed name still explains itself, as the 0.6 removals do.

RENAMED_07 = [("ENGINES", "ENGINE_CLASSES"), ("make_solver", "solver"),
              ("BaseResult", "Result"), ("settings", "using_settings")]
GONE_ON_RESULT = ["check", "diagnose", "category_verdict", "action_kernel", "grid_info",
                  "means_driven", "means_t"]
GONE_ON_MODEL = ["finite", "stationary", "owner", "means_driven"]


@pytest.mark.parametrize("old,new", RENAMED_07)
def test_the_renamed_top_level_names_are_gone_and_name_the_replacement(old, new):
    with pytest.raises(AttributeError, match=f"renamed noisestate.{new} in 0.7"):
        getattr(ns, old)
    assert getattr(ns, new) is not None


def test_the_settings_submodule_was_renamed_so_the_hook_can_see_the_old_name():
    """noisestate.settings could not explain itself while the submodule of that name was bound --
    __getattr__ is never consulted for a bound name.  The submodule is _settings now."""
    import noisestate._settings as s
    assert s.Settings is ns.Settings
    with pytest.raises(AttributeError, match="using_settings"):
        ns.settings


@pytest.mark.parametrize("old", GONE_ON_RESULT)
def test_the_renamed_result_members_are_gone(stat, old):
    assert not hasattr(stat, old), old


@pytest.mark.parametrize("old", GONE_ON_MODEL)
def test_the_renamed_model_members_are_gone(old):
    assert not hasattr(example("ch3_two_player"), old), old


def test_the_new_names_all_work_and_warn_about_nothing(stat):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stat.require_converged(); stat.diagnostic_rows(); stat.diagnostic_records()
        stat.diagnostic_verdict("numerics"); stat.grid_summary(); stat.strategy_kernel("D1")
        stat.has_means; stat.mean_times
        m = example("ch3_two_player")
        m.with_finite(2.0); m.with_stationary(4.0); m.owner_of("D1"); m.drives_means


def test_the_package_emits_no_deprecation_warning_in_normal_use():
    """A deprecated name used INSIDE noisestate warns code the user did not write.  This caught a
    real one before the layer was deleted: renaming sweep.make_solver to solver left
    Result._make_solver importing the old name, an ImportError on a path no other test reaches."""
    import dataclasses
    m = example("ch3_two_player").with_numerics(nodes=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        res = ns.solve(m)
        res.require_converged(); res.stability(); res.diagnostic_summary(); res.to_dict()
        assert dataclasses.replace(res, solver_class=None)._make_solver(m) is not None
