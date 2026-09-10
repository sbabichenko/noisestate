"""The horizon split of docs/api_spec.txt PART 2.

Two distinct quantities, never aliased: `window` is the lag-truncation length L, `T` is the terminal
time.  `extent` is a derived accessor that SELECTS whichever of them a given operation needs -- the
length of the primary computational axis -- and is not a third stored field.

Before the split one `window` field held both, and the field's own comment said so: "L for
stationary; T for finite and transition".  The cost of that was not the name.  It was that the lag
window then had nowhere to live on a transition, whose `window` was T, so it was exiled to the
nested horizon.stationary block -- the same quantity in two different places depending on kind.
"""
import pytest

import noisestate as ns
from helpers import EX, example


# ------------------------------------------------------------------ the quantities are separate

def test_each_kind_carries_only_the_lengths_it_has():
    stat = example("ch3_two_player")
    assert stat.horizon.window == 3.0 and stat.horizon.T is None
    fin = example("ch1_two_player_finite")
    assert fin.horizon.T == 1.0 and fin.horizon.window is None


def test_a_file_spelling_a_finite_length_window_is_refused_by_the_name_that_replaced_it():
    d = example("ch1_two_player_finite").to_dict()
    d["horizon"] = {"kind": "finite", "window": 1.0}
    with pytest.raises(ValueError, match=r"horizon\.window is the lag-truncation length L.*horizon\.T"):
        ns.Model.from_dict(d)
    d["horizon"] = {"kind": "stationary", "T": 1.0}
    with pytest.raises(ValueError, match=r"horizon\.T is the terminal time.*horizon\.window"):
        ns.Model.from_dict(d)


def test_extent_selects_and_is_not_a_third_field():
    assert example("ch3_two_player").horizon.extent == 3.0          # stationary: the lag window
    assert example("ch1_two_player_finite").horizon.extent == 1.0   # finite: the terminal time
    tr = example("ch3_precision_change")
    assert tr.horizon.T == 6.0 and tr.horizon.extent == 6.0         # transition: T, not its L


# ------------------------------------------------------------------ conversions keep what applies

def test_a_kind_change_keeps_the_length_the_new_kind_has_and_drops_the_other():
    """A stationary L must never survive into a finite horizon as its T -- that is the conflation.
    But T carries over from finite to transition, where it means the same thing: the rule is "keep
    what the new kind has", not "drop the old kind's"."""
    stat = example("ch3_two_player")
    assert stat.horizon.window == 3.0
    #  with_horizon DIRECTLY, without the explicit window=None that with_finite passes: this is the
    #  path where the kind-change rule is the only thing stopping the stationary L from riding
    #  along into a horizon that has no lag window at all.
    fin = stat.with_finite(6.0)
    assert fin.horizon.T == 6.0 and fin.horizon.window is None      # L = 3.0 did not survive
    assert "window" not in fin.to_dict()["horizon"]
    assert fin.with_finite(6.0).horizon.window is None              # and the convenience agrees
    tr = fin.with_transition(6.0, past={"model": EX + "ch3_two_player.yaml"}, continuation="stationary")
    assert tr.horizon.T == 6.0                                       # T carried over


def test_the_expression_horizons_express_the_valid_combination_by_type():
    """Finite used to SUBCLASS Stationary and call super().__init__(T), which is how a terminal time
    came to live in a field named window."""
    #  ABSENT, not None.  A class attribute defaulting to None would be optional fields on a
    #  catch-all with extra steps, and omitting a key from serialisation is not the same as
    #  excluding a quantity by type.
    with pytest.raises(AttributeError):
        ns.Stationary(window=3.0).T
    with pytest.raises(AttributeError):
        ns.Finite(T=2.0).window
    assert not hasattr(ns.Finite(T=2.0), "window") and not hasattr(ns.Stationary(window=3.0), "T")
    assert not issubclass(ns.Finite, ns.Stationary)
    assert ns.Finite(T=2.0).compile() == {"kind": "finite", "discount": 0.0, "T": 2.0}
    assert ns.Stationary(window=3.0).compile() == {"kind": "stationary", "discount": 0.0, "window": 3.0}


# ------------------------------------------------------------------ who reads which

def test_a_transitions_continuation_takes_the_pasts_window_not_the_transitions_extent():
    """The continuation is a stationary solve, so its length is a lag window -- the past's.  Taking
    the extent there would give it T, which is a terminal time."""
    old = ns.solve(example("ch3_two_player").with_numerics(nodes=6)).require_converged()
    res = ns.transition(old, example("ch3_two_player").with_params(p1=10.0), 1.0, numerics={"nodes": 6})
    assert res.model.horizon.T == 1.0
    assert res.continuation is not None
    assert res.continuation.model.horizon.window == old.model.horizon.window   # the past's L
    assert res.continuation.model.horizon.T is None                             # and no terminal time


def test_a_sweep_moves_whichever_length_is_named():
    """They are separate parameters because they are separate quantities."""
    rows = ns.sweep(example("ch3_two_player").with_numerics(nodes=6), "horizon.window", [3.0, 4.0],
                    solve_kw={"max_evaluations": 2, "diagnostics": False})
    assert [r["result"].model.horizon.window for r in rows] == [3.0, 4.0]
    with pytest.raises(ValueError, match="horizon.window"):
        ns.sweep(example("ch3_two_player"), "horizon.nonsense", [1.0])


def test_a_settle_template_defers_the_checks_that_need_a_length_but_not_the_others():
    """A transition given `settle` has no T yet -- the march finds it.  Only the extent-dependent
    checks wait; the kind, the discount and the engine pairing are checked for every model, and
    guarding the whole method once hid the kind validator's own error behind this one."""
    d = example("ch3_two_player").with_params(p1=10.0).to_dict()
    d["horizon"] = {"kind": "transition", "settle": 5e-2, "past": {"model": EX + "ch3_two_player.yaml"}}
    m = ns.Model.from_dict(d)                       # validates, with the length checks deferred
    assert m.horizon.T is None and not m.horizon.extent_known
    with pytest.raises(ValueError, match="horizon.kind must be"):
        ns.Model.from_dict({**d, "horizon": {**d["horizon"], "kind": "transitions"}})


def test_with_horizon_replaces_rather_than_patches():
    """The keyword form had to guess which of the old kind's quantities still applied.  Taking an
    object removes the guess: the type states which exist, so nothing carries over by default."""
    stat = example("ch3_two_player")
    fin = stat.with_horizon(ns.Finite(T=6.0))
    assert fin.horizon.T == 6.0 and fin.horizon.window is None
    with pytest.raises(TypeError, match="takes a horizon OBJECT"):
        stat.with_horizon(kind="finite", T=6.0)
    with pytest.raises(TypeError, match="takes a horizon OBJECT"):
        stat.with_horizon()
    #  the internal patcher still refuses a kind change that does not say what each length becomes
    with pytest.raises(ValueError, match="must state both"):
        stat._patch_horizon(kind="finite", T=6.0)
    assert stat._patch_horizon(kind="finite", T=6.0, window=None).horizon.window is None
