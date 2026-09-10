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


# ---------------------------------------------------------------- the 0.7 renames
#  Unlike the 0.6 removals above, these still work: each warns and names its replacement.

RENAMED_ON_RESULT = [("check", "require_converged"), ("diagnose", "diagnostic_rows"),
                     ("category_verdict", "diagnostic_verdict"), ("action_kernel", "strategy_kernel"),
                     ("grid_info", "grid_summary")]
RENAMED_ON_MODEL = [("finite", "with_finite"), ("stationary", "with_stationary"), ("owner", "owner_of")]


@pytest.mark.parametrize("old,new", RENAMED_ON_RESULT)
def test_the_renamed_result_methods_still_work_and_name_the_replacement(stat, old, new):
    with pytest.warns(DeprecationWarning, match=f"use Result.{new}"):
        getattr(stat, old)(*{"category_verdict": ("numerics",), "action_kernel": ("D1",)}.get(old, ()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # the new spelling warns about nothing
        getattr(stat, new)(*{"diagnostic_verdict": ("numerics",), "strategy_kernel": ("D1",)}.get(new, ()))


@pytest.mark.parametrize("old,new", [("means_driven", "has_means"), ("means_t", "mean_times")])
def test_the_renamed_result_properties_still_read(stat, old, new):
    with pytest.warns(DeprecationWarning, match=f"use Result.{new}"):
        was = getattr(stat, old)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert was is getattr(stat, new) or was == getattr(stat, new)


@pytest.mark.parametrize("old,new", RENAMED_ON_MODEL)
def test_the_renamed_model_methods_still_work(old, new):
    m = example("ch3_two_player")
    args = {"finite": (4.0,), "stationary": (4.0,), "owner": ("D1",)}[old]
    with pytest.warns(DeprecationWarning, match=f"use Model.{new}"):
        getattr(m, old)(*args)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(m, new)(*args)
    with pytest.warns(DeprecationWarning, match="use Model.drives_means"):
        m.means_driven


@pytest.mark.parametrize("old,new", [("ENGINES", "ENGINE_CLASSES"), ("make_solver", "solver"),
                                     ("BaseResult", "Result"), ("settings", "using_settings")])
def test_the_renamed_top_level_names_still_resolve(old, new):
    with pytest.warns(DeprecationWarning, match=f"use noisestate.{new}"):
        got = getattr(ns, old)(**{}) if old == "settings" else getattr(ns, old)
    if old == "settings":
        assert type(got) is ns.using_settings                # the shim really builds the new object
    else:
        assert got is getattr(ns, new)


def test_a_removed_06_name_still_explains_itself_beside_the_07_renames():
    """The 0.7 module __getattr__ chains to the 0.6 one instead of replacing it: assigning over it
    silently turned these explanations back into bare AttributeErrors."""
    with pytest.raises(AttributeError, match="removed in 0.6"):
        ns.StationaryResult
    with pytest.raises(AttributeError, match="has no attribute"):
        ns.no_such_name_at_all


def test_the_package_never_uses_its_own_deprecated_names():
    """A deprecated name used *inside* noisestate warns code the user did not write, and the warning
    names a fix they cannot apply.  This caught a real one: renaming sweep.make_solver to solver left
    `Result._make_solver` importing the old name, an ImportError on a path no other test reaches."""
    m = example("ch3_two_player").with_numerics(nodes=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        res = ns.solve(m)
        res.require_converged()
        res.stability()
        res.diagnostic_summary()
        res.diagnostic_verdict("numerics")
        res.grid_summary()
        res.to_dict()
        res.has_means, res.mean_times
        m.describe(), m.drives_means, m.owner_of("D1")
        m.with_finite(2.0), m.with_stationary(4.0)

        # _make_solver only takes the import branch when solver_class is None, which a solved
        # result never has -- so reaching the broken line means clearing it deliberately.
        import dataclasses
        bare = dataclasses.replace(res, solver_class=None)
        assert bare._make_solver(m) is not None
