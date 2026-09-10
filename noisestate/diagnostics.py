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


def curvature_is_obtainable(model) -> bool:
    """Whether THIS package can compute the second-order check for `model`.

    A capability question, not an applicability one.  Second-order optimality of a best response is
    meaningful for every model here: the agent's problem has a second-order condition whether or
    not its objective is quadratic.  What is narrower is the method -- Engine._second_order builds
    the curvature as an exact quadratic form M = T' G T and returns None when the objective is not
    one in the strategy, which is the stationary engine with a positive discount (the discounted
    objective is not a quadratic form in the stationary kernel).

    So the check APPLIES and the engine CANNOT RUN IT: that is UNSUPPORTED, and it blocks under a
    policy that requires it.  Calling it NOT_APPLICABLE would claim the condition has no meaning
    for the model, which is a mathematical claim this package has not established and does not
    need -- and it would let a result be accepted for a check nothing performed.
    """
    return not (model.horizon.kind == "stationary" and float(model.horizon.discount) > 0.0)


CHECKS = {
    "converged": lambda model: True,
    "resolution": lambda model: True,
    "second_order": lambda model: True,      # applies everywhere; see curvature_is_obtainable
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


@dataclass(frozen=True)
class Stability:
    """The best-response Jacobian at a point, and -- only if that point is a verified equilibrium --
    what its spectrum means for response dynamics.

    The Jacobian at a NON-fixed point is a legitimate object and is always returned: radius,
    eigenvalues, method and the residual are computed whatever the point.  What is withheld is the
    INTERPRETATION.  Labelling the spectrum at a non-equilibrium "equilibrium stability" is the
    error, not computing it.

    The classification fields are always PRESENT and are None when `verified` is False.  Removing
    them conditionally would make attribute access depend on the data, which is worse than a
    documented None.

    EQUILIBRIUM VALIDITY AND RESPONSE STABILITY ARE SEPARATE FINDINGS.  An equilibrium may be
    unstable under best-response iteration, and distinct equilibria may have distinct costs;
    neither observation bears on whether a point is an equilibrium.
    """
    radius: float
    eigenvalues: Tuple[complex, ...]
    method: str
    fixed_point_residual: float
    residual_norm: str                       # which norm, and how it is scaled
    residual_tolerance: Optional[float]      # None until D4 is settled
    verified: bool
    unverified_reasons: Tuple[str, ...]      # empty iff verified
    full_response: Optional[str]
    adjusted_response: Optional[str]
    adjusted_radius_bound: Optional[float]
    adjustment: Optional[float]
    untied: bool = False
    evaluations: int = 0

    @property
    def stable(self) -> bool:
        """Whether the spectral radius is inside the unit circle.  A statement about the DYNAMICS,
        true or false regardless of whether the point is a verified equilibrium."""
        return self.radius < 1.0

    def to_dict(self) -> dict:
        return {"radius": self.radius, "eigenvalues": [[z.real, z.imag] for z in self.eigenvalues],
                "method": self.method, "stable": self.stable,
                "fixed_point_residual": self.fixed_point_residual,
                "residual_norm": self.residual_norm, "residual_tolerance": self.residual_tolerance,
                "verified": self.verified, "unverified_reasons": list(self.unverified_reasons),
                "full_response": self.full_response, "adjusted_response": self.adjusted_response,
                "adjusted_radius_bound": self.adjusted_radius_bound, "adjustment": self.adjustment,
                "untied": self.untied, "evaluations": self.evaluations}

    #  dict access, so code written against the old report keeps working within this package
    def __getitem__(self, key):
        return self.to_dict()[key]

    def get(self, key, default=None):
        return self.to_dict().get(key, default)


#  D4 is not settled: the norm and scaling that make a residual comparable across grids and model
#  scales are numerical work, not API work.  The FIELDS are specified and populated from the start
#  so the evidence is carried and serialised now; only the CLASSIFICATION waits.
RESIDUAL_NORM = "relative: ||F(z) - z|| / scale, the solver's own scaling"
RESIDUAL_TOLERANCE = None
UNDEFINED_CRITERION = "equilibrium residual criterion not yet defined (api_spec D4)"


def verification(statuses: Mapping[str, Status], policy: Policy, model,
                 residual: float) -> Tuple[bool, Tuple[str, ...]]:
    """Whether a point is a verified equilibrium, and if not, why not.

        required = applicable(MINIMUM | policy.required, model)
        verified = residual_passed and all(statuses[c] is PASSED for c in required)

    The UNION is what makes both halves true at once.  A WEAK policy cannot lower the bar, because
    MINIMUM is always in the union; it also does not by itself cause rejection, since a policy that
    adds nothing simply adds nothing.  A STRONG policy's additional required checks must also pass.
    """
    reasons = []
    if RESIDUAL_TOLERANCE is None:
        reasons.append(UNDEFINED_CRITERION)
    elif not residual <= RESIDUAL_TOLERANCE:
        reasons.append(f"fixed-point residual {residual:.2e} above {RESIDUAL_TOLERANCE:.2e}")
    for check in sorted(applicable(MINIMUM | policy.required, model)):
        status = statuses.get(check, Status.MISSING)
        if status is not Status.PASSED:
            reasons.append(f"{check} is {status}, not passed")
    return (not reasons), tuple(reasons)


def classify(eigenvalues, radius: float, method: str, adjustment: float = 0.5) -> dict:
    """The response-dynamics classification of a VERIFIED equilibrium's spectrum.

    Only ever called for a verified point (Result.stability gates it), because the words it
    produces -- "converges", "oscillates" -- are claims about equilibrium dynamics.

    The damped bound is sound by the triangle inequality: an Arnoldi run returns the leading Ritz
    values, and an omitted mode mu has |mu| <= min|lambda returned|, so its damped image satisfies
    |1 - a + a*mu| <= (1 - a) + a*min|lambda|.  When that cannot be bounded the result says
    "not certified" rather than guessing.
    """
    import numpy as np
    vals = np.asarray([complex(v) for v in eigenvalues])
    dominant = vals[int(np.argmax(np.abs(vals)))]
    damped = (1.0 - adjustment) + adjustment * vals
    sampled = float(np.max(np.abs(damped)))
    stable = radius < 1.0
    if stable:
        full = "converges"
    elif abs(dominant.imag) <= 1e-8 * max(1.0, abs(dominant.real)) and dominant.real < -1:
        full = "oscillates"
    else:
        full = "diverges"
    bound = None
    if method == "zero":
        bound = sampled
    elif stable:
        bound = max(sampled, 1.0 - adjustment + adjustment * radius)
    elif method == "arnoldi" and float(np.min(np.abs(vals))) < 1.0:
        bound = max(sampled, 1.0 - adjustment + adjustment * float(np.min(np.abs(vals))))
    adjusted = ("diverges" if sampled >= 1 else
                "converges" if bound is not None and bound < 1 else "not certified")
    return {"full_response": full, "adjusted_response": adjusted,
            "adjusted_radius_bound": None if bound is None else float(bound),
            "adjustment": float(adjustment), "dominant_eigenvalue": dominant}
