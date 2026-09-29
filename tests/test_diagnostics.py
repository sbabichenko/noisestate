"""The diagnostics contract of docs/api_spec.txt PART 3: status, policy, and what they produce.

Each test here is one clause of that contract, and several correspond to defects the specification
was written to close -- an engine passing a verdict it could not test, and a solve with the guards
turned off passing the guards.
"""

import pytest

import noisestate as ns
from noisestate.diagnostics import CHECKS, MINIMUM, Policy, Status, applicable
from helpers import example


def model(kind="stationary", engine=None, nodes=8, **horizon):
    d = {"name": "diag", "shocks": ["w0", "w1"],
         "states": {"X": {"drift": {"X": -1, "D": 1}, "noise": {"w0": 1}}},
         "agents": {"a": {"controls": ["D"], "signals": {"y": {"drift": {"X": 1}, "noise": {"w1": 1}}},
                          "loss": [[1, "X", "X"], [1, "D", "D"]]}},
         #  the length goes under the key its kind keeps it in.  Writing one key and branching on
         #  the kind -- as this helper first did -- is the conflation PART 2 of the spec ended.
         "horizon": {"kind": kind, **({"window": 4.0} if kind == "stationary" else {"T": 1.0}), **horizon},
         "numerics": {"nodes": nodes, **({"engine": engine} if engine else {})}}
    return ns.Model.from_dict(d)


# ------------------------------------------------------------------ status is not acceptance

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


def test_applicability_reads_what_the_model_carries_not_its_kind():
    """The check that applies is the one the model gives meaning to, and for the window family that
    is decided by what the horizon HOLDS, not by its kind.

    This test replaces one named "the window check applies to a transition too" which asserted
    exactly that -- and tested it on ch3_two_player, a STATIONARY model, so it passed either way and
    never examined a transition at all.  What a transition actually carries is a past and a
    continuation, each with its own lag window and its own check; its horizon.window is None.
    Deeming the plain `window` check applicable to it produced no row and left the status MISSING --
    "was to run, produced no record" -- on every transition, so none could be accepted.
    """
    stat = example("ch3_two_player")
    assert "window" in applicable(CHECKS, stat)                       # a stationary model has its own
    assert "window" not in applicable(CHECKS, stat.with_finite(1.0))  # a finite horizon has none
    trans = example("ch3_precision_change")
    assert trans.horizon.kind == "transition" and trans.horizon.window is None
    assert "window" not in applicable(CHECKS, trans)                  # its windows are its past's and its continuation's
    assert {"past window", "continuation window", "settled"} <= applicable(CHECKS, trans)


def test_a_transition_closed_by_the_games_end_has_nothing_to_settle_against():
    """continuation "end" means the game stops at T, so there is no stationary buffer for the maps to
    settle TO and no continuation window to be too short.  Both were being demanded regardless."""
    prior = example("kyle_back_prior")
    assert prior.horizon.kind == "transition" and prior.horizon.continuation == "end"
    applies = applicable(CHECKS, prior)
    assert "settled" not in applies and "continuation window" not in applies
    #  and its past is a PRIOR on the state, not an inherited regime, so it has no window either
    assert "model" not in (prior.horizon.past or {}) and "past window" not in applies


def test_no_shipped_example_reports_a_check_as_missing():
    """MISSING means "applicable, supported, was to run, produced no record" -- the status that says
    INVESTIGATE THIS.  It fired routinely on correct models, which is how a status stops meaning
    anything.  Nothing shipped should produce one."""
    coarse = {"ch3_precision_change": {"nodes": 5}, "ch5_cycle_market": {"nodes": 6},
              "kyle_back_prior": {"nodes": 8}}
    for name in ns.examples():
        res = ns.solve(ns.example(name), coarse.get(name))
        missing = [c for c, st in res.diagnostics.statuses.items() if st is Status.MISSING]
        assert not missing, f"{name}: {missing} reported MISSING"


def test_a_failing_past_window_blocks_acceptance():
    """It failed, was printed as PAST WINDOW TOO SHORT, and could not block: `past window` was a row
    the result emitted and not a name any policy knew.  Emitted is not the same as known -- the third
    gap of this shape, after NOT_APPLICABLE-vs-UNSUPPORTED and applicable-vs-emitted.
    """
    res = ns.solve(ns.example("ch3_precision_change"), {"nodes": 5})
    failing = {d["name"].split(":", 1)[0] for d in res.diagnostics.rows if d["ok"] is False}
    assert "past window" in failing                                  # the guard fires
    blocking = {b.check for b in res.diagnostics.assess().blocking}
    assert "past window" in blocking                                 # and acceptance sees it
    assert not (failing - blocking), f"failing rows invisible to acceptance: {sorted(failing - blocking)}"


def test_the_discount_does_not_take_the_second_order_check_away():
    """It was excluded at rho > 0, on the ground that "the discounted objective is not a quadratic
    form in the stationary kernel".  That is wrong, and it made ch4_kyle_back -- the example the
    README prints in full -- permanently unacceptable.

    The discounted stationary objective is a quadratic form in the deviation; at rho > 0 the check
    is made on its own form (the responses weighted by e^{-rho tau / 2}, test_second_order_discount),
    at rho = 0 on the average-cost one.  The Kyle-Back trader's is positive at every rho tried.
    """
    kb = ns.solve(example("ch4_kyle_back"), {"nodes": 20})
    assert kb.model.horizon.discount > 0 and kb.model.horizon.kind == "stationary"
    assert kb.diagnostics.statuses["second_order"] is Status.PASSED
    #  the flagship example passes every check at its shipped L = 12, the costs' window included (the publication
    #  policy's `window cost`, measured by require_ok: a window of 18 moves the trader's cost by 8e-8); on L = 8, where
    #  it shipped before, a window of 12 moves it by 6.5e-5 and require_ok refuses it
    assert {b.check for b in kb.diagnostics.assess().blocking} == {"window cost"}          # skipped until measured
    assert kb.require_ok() is kb and kb.window_check.cost_change < 1e-6
    short = ns.solve(example("ch4_kyle_back").with_stationary(8.0), {"nodes": 20})
    with pytest.raises(ns.DiagnosticsError, match="WINDOW TOO SHORT FOR THE COSTS"):
        short.require_ok()
    #  and the SIGN of the curvature -- the verdict -- does not move with the discount (on the window of 8 this record was
    #  made on: near rho = 0 the undiscounted problem has no solution, and on 12 its window artefact is not a minimum)
    verdicts = {}
    for rho in (1e-9, 0.1, 0.5, 1.0):
        res = ns.solve(example("ch4_kyle_back").with_stationary(8.0).with_params(rho=rho), {"nodes": 16})
        verdicts[rho] = {a: d["ok"] for a, d in res.second_order.items()}
        assert all(d["min"] > 0 for d in res.second_order.values()), f"rho={rho}: {res.second_order}"
    assert all(v == verdicts[1e-9] for v in verdicts.values()), verdicts


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


def test_the_verification_minimum_is_not_a_policy_and_cannot_be_weakened_by_one():
    """MINIMUM is what "verified" means; a policy may add to it and may never subtract."""
    assert MINIMUM <= Policy.PUBLICATION.required
    assert not MINIMUM <= Policy.EXPLORATORY.required        # a weak policy does not shrink it
    m = model()
    assert applicable(MINIMUM | Policy.EXPLORATORY.required, m) == applicable(MINIMUM, m)


# ------------------------------------------------------------------ stability: PART 4

def test_the_spectrum_is_always_computed_only_the_interpretation_is_gated():
    """The Jacobian at a non-fixed point is a legitimate object.  Labelling its spectrum
    "equilibrium stability" is the error, not computing it."""
    res = ns.solve(example("ch3_two_player"))
    st = res.stability()
    assert st.radius > 0 and st.eigenvalues and st.method            # always computed
    assert st.fixed_point_residual >= 0 and st.residual_norm         # the evidence, always
    assert st.verified is False                                      # while D4 is open
    assert (st.full_response, st.adjusted_response, st.adjusted_radius_bound) == (None, None, None)


def test_the_classification_fields_are_present_and_none_never_absent():
    """Removing them conditionally would make attribute access depend on the data."""
    st = ns.solve(example("ch3_two_player")).stability()
    for field in ("full_response", "adjusted_response", "adjusted_radius_bound", "adjustment"):
        assert hasattr(st, field)


def test_verification_names_every_failing_condition_not_only_the_first():
    """ch3_two_player fails the window guard AND waits on D4, so it is unverified twice over.
    Even once D4 lands, that model stays unclassified until its window is fixed."""
    st = ns.solve(example("ch3_two_player")).stability()
    assert len(st.unverified_reasons) >= 2
    assert any("residual criterion" in r for r in st.unverified_reasons)
    assert any("window" in r for r in st.unverified_reasons)


def test_a_weak_policy_cannot_lower_the_verification_bar():
    """verified uses applicable(MINIMUM | policy.required, model).  The union is what stops
    Policy.EXPLORATORY producing verified=True on weaker evidence under the same label."""
    res = ns.solve(example("ch3_two_player"))
    lenient = res.stability(policy=Policy.EXPLORATORY)
    assert lenient.verified is False
    assert any("window" in r for r in lenient.unverified_reasons)     # not in EXPLORATORY.required
    assert lenient.full_response is None


def test_the_payload_carries_the_verification_evidence_not_just_the_radius():
    """A radius with no statement about whether the point is an equilibrium invites the reader to
    treat the spectrum as equilibrium stability, which is the misreading PART 4 exists to prevent."""
    res = ns.solve(example("ch3_two_player"))
    res.stability()
    block = res.to_dict()["stability"]
    assert set(block) >= {"radius", "verified", "fixed_point_residual", "residual_norm",
                          "residual_tolerance", "unverified_reasons"}
    assert block["verified"] is False and block["residual_tolerance"] is None
    assert ns.schema.validate(res.to_dict(), "payload") == []


def test_compare_reports_the_dynamics_rather_than_deciding_them():
    """dynamics used to classify any scenario, including one that never converged, and dropped the
    fixed_point_residual that would have revealed it.  It now reports what stability() produced."""
    m = example("ch3_two_player").with_numerics(nodes=6)
    study = ns.compare({"a": m}, baseline="a", stability=True)
    dyn = study["a"].dynamics
    assert dyn["verified"] is False and dyn["full_response"] is None
    assert dyn["fixed_point_residual"] is not None            # the evidence it used to drop
    assert "not verified" in study.summary()                  # distinct from "not checked"

    #  "not checked" is the third case: stability never ran.  A model whose window guard fails has
    #  its radius computed unasked (noisestate._radius_when_the_window_fails), so the model here is
    #  a finite one, which carries no window check at all.
    quiet = example("ch1_two_player_finite")
    assert "not checked" in ns.compare({"a": quiet}, baseline="a").summary()


@pytest.mark.parametrize("name", ["ch1_two_player_finite", "ch3_two_player"])
def test_a_convex_loss_passes_the_second_order_check_without_its_form(name, monkeypatch):
    """A positive semidefinite loss form makes the second-order form M = T' G T positive semidefinite for any responses:
    the check passes by that certificate and its numbers (min, max) are made on first read (engine.Curvature).  The
    verdicts every solve reads (statuses, status, the window guard, repr) leave them unmade; once read they are the
    eager computation's to the bit, and the record is a plain dict to every copy of it."""
    from noisestate.engine import Curvature, EngineBase
    res = ns.solve(ns.example(name))
    recs = list(res.second_order.values())
    assert recs and all(isinstance(so, Curvature) and so.pending and so["ok"] and so["converged"] for so in recs)
    res.diagnostics.statuses; res.diagnostics.assess(); res.status; repr(res); res.summary()
    assert all(so.pending for so in recs)
    assert res.diagnostics.statuses["second_order"] is Status.PASSED
    monkeypatch.setattr(EngineBase, "_curvature_certified", lambda self, agent: False)
    eager = ns.solve(ns.example(name)).second_order
    for a, so in res.second_order.items():
        assert not isinstance(eager[a], Curvature)
        assert dict(so) == eager[a] and not so.pending
    assert res.to_dict()["second_order"] == {a: dict(v) for a, v in eager.items()}


def test_an_indefinite_loss_is_checked_on_its_form():
    """Kyle's insider (-V D + P D + eps D^2: indefinite in (D, P, V)) gets the eager check; the market maker's loss (P^2 -
    2 P V: indefinite too) likewise."""
    from noisestate.engine import Curvature
    res = ns.solve(ns.example("ch4_kyle_back"))
    assert res.second_order and not any(isinstance(so, Curvature) for so in res.second_order.values())


def test_a_large_certified_form_takes_its_extremes_from_its_cholesky(monkeypatch):
    """Above PSD_EXTREMES_MIN a certified form's extremes come from Lanczos on it and on its Cholesky factor's inverse
    (engine.psd_extremes), not eigh: the same numbers to 1e-12 of the largest.  Forced here on Chapter 1's finite game and
    Chapter 3's stationary one by lowering the threshold; and on a matrix singular to rounding the Cholesky fails and the
    caller is told (None)."""
    import numpy as np
    from noisestate import engine, finite_free, stationary
    eager = {n: ns.solve(ns.example(n)).second_order for n in ("ch1_two_player_finite", "ch3_two_player")}
    for mod in (finite_free, stationary):
        monkeypatch.setattr(mod, "PSD_EXTREMES_MIN", 10)
    for n, ref in eager.items():
        got = ns.solve(ns.example(n)).second_order
        for a in ref:
            assert abs(got[a]["min"] - ref[a]["min"]) < 1e-12 and abs(got[a]["max"] - ref[a]["max"]) < 1e-12, (n, a, got[a], ref[a])
    v = np.random.default_rng(1).standard_normal((50, 3))
    assert engine.psd_extremes(v @ v.T) is None
