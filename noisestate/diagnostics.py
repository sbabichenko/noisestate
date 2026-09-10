"""Check status, validation policy, and the assessment the two produce together.

A STATUS describes one check and says nothing about whether a result is acceptable.  A POLICY
names which checks a particular use requires; it is a property of what the caller intends to do
with the number, not of the check.  An ASSESSMENT is what they produce together, and is never a
single status -- a collection of checks does not have one.

Three separations carry the weight here, and each was a real defect before it was drawn:

  NOT_APPLICABLE vs UNSUPPORTED   the first is a property of the MODEL (a lag-window check on a
      plain finite horizon: nothing is missing).  The second is a property of the ENGINE (the cell
      engine computes no representation error: something IS missing).  Merging them let an engine
      pass a verdict precisely because it could not perform the checks.

  applicable vs emitted   what applies is computed from the model and the policy, never from the
      rows an engine happened to produce.  Otherwise a check that silently produced no row leaves
      the evaluation altogether, which is how the cell engine's absent second-order check used to
      disappear.

  acceptance vs verification   acceptance follows the caller's policy.  Equilibrium verification
      (see Result.stability) adds a mandatory minimum the policy may strengthen and may never
      weaken, so "verified" means one thing regardless of who asked.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple


class Status(enum.Enum):
    PASSED = "passed"                  # ran; the model met it
    FAILED = "failed"                  # ran; the model did not meet it
    SKIPPED = "skipped"                # this engine can compute it here; it was not run
    UNSUPPORTED = "unsupported"        # this engine cannot compute it
    NOT_APPLICABLE = "not_applicable"  # the check has no meaning for this model
    MISSING = "missing"                # applicable and supported, was to run, produced no record

    @property
    def computed(self) -> bool:
        return self in (Status.PASSED, Status.FAILED)

    def __str__(self) -> str:
        return self.value


#  Every check a policy may require, with the question that decides whether it APPLIES.  The
#  predicate reads the model and nothing else: applicability is never hard-coded per horizon kind,
#  because a transition carries a lag-truncation window just as a stationary model does.
def _carries_lag_window(model) -> bool:
    return model.horizon.kind in ("stationary", "transition")


def _is_transition(model) -> bool:
    return model.horizon.kind == "transition"


def _objective_is_quadratic_in_the_strategy(model) -> bool:
    """Whether a second-order check has anything to test.

    On the stationary engine with a positive discount the discounted objective is NOT a quadratic
    form in the stationary kernel, so there is no form whose curvature could be taken -- the check
    is meaningless for the model rather than absent from the run.  Engine._second_order says so and
    returns None; treating that as a missing record would report a defect where there is none.
    """
    return not (model.horizon.kind == "stationary" and float(model.horizon.discount) > 0.0)


CHECKS = {
    "converged": lambda model: True,
    "resolution": lambda model: True,
    "second_order": _objective_is_quadratic_in_the_strategy,
    "window": _carries_lag_window,
    "settled": _is_transition,
}

#  Checks that exist only because diagnostics ran.  With solve(diagnostics=False) they are SKIPPED,
#  which is a reason, not an engine limitation.
DIAGNOSTIC_ONLY = frozenset({"resolution", "second_order"})

#  The minimum an equilibrium must satisfy to be called verified, independent of any policy.
MINIMUM = frozenset({"converged", "resolution", "second_order", "window"})


def applicable(checks: Iterable[str], model) -> frozenset:
    """Those of `checks` the model gives meaning to."""
    return frozenset(c for c in checks if CHECKS.get(c, lambda m: True)(model))


@dataclass(frozen=True)
class Policy:
    """Which checks a use requires.  Weaker sets are legitimate and must be separately named."""
    name: str
    required: frozenset

    def applies(self, model) -> frozenset:
        return applicable(self.required, model)

    def __str__(self) -> str:
        return self.name


Policy.PUBLICATION = Policy("publication",
                            frozenset({"converged", "resolution", "window", "second_order", "settled"}))
Policy.EXPLORATORY = Policy("exploratory", frozenset({"converged"}))


@dataclass(frozen=True)
class Blocker:
    check: str
    status: Status                     # FAILED, SKIPPED, UNSUPPORTED or MISSING
    reason: str

    def __str__(self) -> str:
        return f"{self.check} [{self.status}] {self.reason}".rstrip()


@dataclass(frozen=True)
class Assessment:
    """What a policy makes of a result.  `accepted` is the only question it answers."""
    accepted: bool
    policy: str
    blocking: Tuple[Blocker, ...]           # empty iff accepted
    statuses: Mapping[str, Status]          # every applicable check, by root name
    uncomputed: Tuple[str, ...]             # applicable, no result, for ANY reason

    def to_dict(self) -> dict:
        return {"policy": self.policy, "accepted": self.accepted,
                "statuses": {k: str(v) for k, v in self.statuses.items()},
                "blocking": [{"check": b.check, "status": str(b.status), "reason": b.reason}
                             for b in self.blocking],
                "uncomputed": list(self.uncomputed)}

    def __str__(self) -> str:
        head = f"{self.policy}: {'accepted' if self.accepted else 'NOT accepted'}"
        if self.accepted:
            return head
        return head + "\n" + "\n".join("  " + str(b) for b in self.blocking)


REASON = {
    Status.FAILED: "the check ran and the model did not meet it",
    Status.SKIPPED: "not run (solve(diagnostics=False), a budget or a deadline)",
    Status.UNSUPPORTED: "this engine cannot compute it",
    Status.MISSING: "applicable and supported, but no record was produced -- a defect in the package",
}


def assess(statuses: Mapping[str, Status], policy: Policy, model,
           details: Optional[Mapping[str, str]] = None) -> Assessment:
    """Acceptance: every check the POLICY requires and the MODEL makes applicable has PASSED.

    Nothing else grants acceptance -- not an absent record, not an empty category, not an engine
    limitation.  A check the policy does not require is reported and never blocks.
    """
    required = policy.applies(model)
    details = details or {}

    def why(check: str, status: Status) -> str:
        #  A FAILED check already carries the measurement and the fix in its own flag text; the
        #  generic sentence would throw both away, and "raise horizon.window" is the whole point
        #  of reporting it.  The other statuses have no row, so the generic reason is all there is.
        return details.get(check) or REASON.get(status, "")

    blocking = tuple(Blocker(c, statuses.get(c, Status.MISSING), why(c, statuses.get(c, Status.MISSING)))
                     for c in sorted(required) if statuses.get(c, Status.MISSING) is not Status.PASSED)
    #  applicable and produced no result.  NOT_APPLICABLE is excluded: nothing is missing there.
    uncomputed = tuple(sorted(c for c, s in statuses.items()
                              if not s.computed and s is not Status.NOT_APPLICABLE))
    return Assessment(accepted=not blocking, policy=policy.name, blocking=blocking,
                      statuses=dict(statuses), uncomputed=uncomputed)
