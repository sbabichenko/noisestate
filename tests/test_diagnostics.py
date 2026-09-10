"""The diagnostics contract of docs/api_spec.txt PART 3: status, policy, and what they produce.

Each test here is one clause of that contract, and several correspond to defects the specification
was written to close -- an engine passing a verdict it could not test, and a solve with the guards
turned off passing the guards.
"""
import warnings

import pytest

import noisestate as ns
from noisestate.diagnostics import CHECKS, MINIMUM, Policy, Status, applicable
from helpers import example


def model(kind="stationary", engine=None, nodes=8, **horizon):
    d = {"name": "diag", "channels": ["w0", "w1"],
         "states": {"X": {"drift": {"X": -1, "D": 1}, "noise": {"w0": 1}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1}, "noise": {"w1": 1}}},
                          "loss": [[1, "X", "X"], [1, "D", "D"]]}},
         "horizon": {"kind": kind, "window": 4 if kind == "stationary" else 1.0, **horizon},
         "numerics": {"nodes": nodes, **({"engine": engine} if engine else {})}}
    return ns.Model.from_dict(d)


# ------------------------------------------------------------------ status is not acceptance

def test_an_engine_that_cannot_run_a_required_check_is_refused_not_excused():
    """The defect the specification exists to close.  The cell engine computes neither a
    representation error nor a second-order form; before this, it passed the whole verdict while
    running two fewer checks than the others, and absence of evidence read as soundness."""
    res = ns.solve(model("finite", engine="cells"))
    st = res.diagnostics.statuses
    assert st["resolution"] is Status.UNSUPPORTED and st["second_order"] is Status.UNSUPPORTED
    verdict = res.diagnostics.assess()
    assert not verdict.accepted
    assert verdict.uncomputed == ("resolution", "second_order")
    with pytest.raises(ns.DiagnosticsError, match="cannot compute"):
        res.require_ok()


def test_turning_the_guards_off_does_not_pass_the_guards():
    """solve(diagnostics=False) used to leave require_ok() passing with nothing run at all."""
    res = ns.solve(model(), diagnostics=False)
    st = res.diagnostics.statuses
    assert st["resolution"] is Status.SKIPPED and st["second_order"] is Status.SKIPPED
    assert not res.diagnostics.assess().accepted
    with pytest.raises(ns.DiagnosticsError):
        res.require_ok()
    #  the reason is distinguishable from a failure: it was not run, not run-and-failed
    assert {b.status for b in res.diagnostics.assess().blocking} == {Status.SKIPPED}


def test_a_weaker_policy_is_legitimate_and_must_be_named():
    res = ns.solve(model("finite", engine="cells"))
    assert not res.diagnostics.assess(Policy.PUBLICATION).accepted
    assert res.diagnostics.assess(Policy.EXPLORATORY).accepted
    assert res.require_ok(Policy.EXPLORATORY) is res
    assert res.diagnostics.assess(Policy.EXPLORATORY).policy == "exploratory"


# ------------------------------------------------------------------ applicability, from the model

def test_not_applicable_is_a_property_of_the_model_and_never_blocks():
    """A lag-window check has no meaning on a plain finite horizon: nothing is missing, so nothing
    blocks.  That is a different thing from UNSUPPORTED, where something IS missing."""
    fin = ns.solve(model("finite"))
    assert fin.diagnostics.statuses["window"] is Status.NOT_APPLICABLE
    assert fin.diagnostics.assess().accepted
    assert "window" not in fin.diagnostics.assess().uncomputed      # nothing is missing here
    stat = ns.solve(model("stationary"))
    assert stat.diagnostics.statuses["window"] is not Status.NOT_APPLICABLE


def test_the_window_check_applies_to_a_transition_too_not_only_a_stationary_model():
    """Applicability reads the model.  A transition carries a lag-truncation window as well, so
    hard-coding "stationary" would drop transitions out of the check."""
    assert "window" in applicable(CHECKS, example("ch3_two_player"))
    assert "window" not in applicable(CHECKS, example("ch3_two_player").with_finite(1.0))


def test_a_check_with_no_form_to_test_is_not_applicable_rather_than_missing():
    """The stationary engine with a positive discount has no quadratic form in the strategy, so
    Engine._second_order returns None.  Reporting that as MISSING would claim a defect in the
    package where the check simply cannot mean anything for the model."""
    kb = ns.solve(example("ch4_kyle_back"))
    assert kb.model.horizon.discount > 0 and kb.model.horizon.kind == "stationary"
    assert kb.diagnostics.statuses["second_order"] is Status.NOT_APPLICABLE
    assert kb.require_ok() is kb


# ------------------------------------------------------------------ the aggregate is not a status

def test_an_assessment_reports_every_applicable_check_not_only_the_blocking_ones():
    res = ns.solve(model())
    verdict = res.diagnostics.assess()
    assert set(verdict.statuses) == set(CHECKS)             # every check, applicable or not
    assert all(isinstance(v, Status) for v in verdict.statuses.values())
    assert verdict.blocking == () if verdict.accepted else verdict.blocking


def test_a_blocked_check_keeps_the_measurement_and_the_fix():
    """A FAILED check already carries both in its flag text; a generic reason would throw away the
    number and the action, and "raise horizon.window" is the point of reporting it."""
    res = ns.solve(example("ch3_two_player"))
    verdict = res.diagnostics.assess()
    blocked = {b.check: b for b in verdict.blocking}
    assert "window" in blocked and blocked["window"].status is Status.FAILED
    assert "WINDOW TOO SHORT" in blocked["window"].reason and "horizon.window" in blocked["window"].reason


def test_the_exceptions_are_siblings_so_neither_catches_the_other():
    assert issubclass(ns.ConvergenceError, ns.ResultValidationError)
    assert issubclass(ns.DiagnosticsError, ns.ResultValidationError)
    assert not issubclass(ns.DiagnosticsError, ns.ConvergenceError)
    assert not issubclass(ns.ConvergenceError, ns.DiagnosticsError)
    res = ns.solve(example("ch3_two_player"))              # converged, and a guard failed
    assert res.converged
    with pytest.raises(ns.DiagnosticsError) as caught:
        res.require_ok()
    assert caught.value.assessment.blocking                 # the assessment travels with the error


def test_the_payload_carries_every_status_and_the_policy_that_judged_them():
    res = ns.solve(model("finite", engine="cells"))
    payload = res.to_dict()["assessment"]
    assert payload["policy"] == "publication" and payload["accepted"] is False
    assert payload["statuses"]["resolution"] == "unsupported"
    assert payload["uncomputed"] == ["resolution", "second_order"]
    assert ns.schema.validate(res.to_dict(), "payload") == []


def test_the_verification_minimum_is_not_a_policy_and_cannot_be_weakened_by_one():
    """MINIMUM is what "verified" means; a policy may add to it and may never subtract."""
    assert MINIMUM <= Policy.PUBLICATION.required
    assert not MINIMUM <= Policy.EXPLORATORY.required        # a weak policy does not shrink it
    m = model()
    assert applicable(MINIMUM | Policy.EXPLORATORY.required, m) == applicable(MINIMUM, m)
