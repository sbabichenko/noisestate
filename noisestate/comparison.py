"""Named comparisons of related models, kept separate from solver internals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional

import numpy as np

from .numerics import Numerics
from .results import Result
from .spec import Model


def _complex(z) -> dict:
    z = complex(z)
    return {"real": float(z.real), "imag": float(z.imag)}


@dataclass
class ScenarioResult:
    """One named model and its ordinary solver result inside a comparison."""
    name: str
    model: Model
    result: Result
    baseline_costs: Mapping[str, float]
    adjustment: float = 0.5

    @property
    def total_cost(self) -> float:
        return float(sum(self.result.costs.values()))

    @property
    def total_change(self) -> float:
        return self.total_cost - float(sum(self.baseline_costs.values()))

    @property
    def total_change_fraction(self) -> float:
        base = float(sum(self.baseline_costs.values()))
        return self.total_change / base if base else float("nan")

    @property
    def cost_changes(self) -> Dict[str, dict]:
        out = {}
        for agent, value in self.result.costs.items():
            base = float(self.baseline_costs[agent])
            out[agent] = {"value": float(value), "change": float(value - base),
                          "change_fraction": float((value - base) / max(abs(base), 1e-300))}
        return out

    @property
    def dynamics(self) -> Optional[dict]:
        rep = getattr(self.result, "stability_report", None)
        if not rep:
            return None
        vals = np.asarray([complex(v) for v in rep["eigenvalues"]])
        dominant = vals[int(np.argmax(np.abs(vals)))]
        adjusted = (1.0 - self.adjustment) + self.adjustment * vals
        sampled_radius = float(np.max(np.abs(adjusted)))
        if rep["stable"]:
            full = "converges"
        elif abs(dominant.imag) <= 1e-8 * max(1.0, abs(dominant.real)) and dominant.real < -1:
            full = "oscillates"
        else:
            full = "diverges"
        bound = None
        if rep["method"] == "zero":
            bound = sampled_radius
        elif rep["stable"]:
            bound = max(sampled_radius, 1.0 - self.adjustment + self.adjustment * float(rep["radius"]))
        elif rep["method"] == "arnoldi" and float(np.min(np.abs(vals))) < 1.0:
            # Arnoldi returned the largest-modulus modes.  Every omitted mode is
            # within the smallest returned disk, which bounds its damped image.
            omitted = 1.0 - self.adjustment + self.adjustment * float(np.min(np.abs(vals)))
            bound = max(sampled_radius, omitted)
        adjusted_response = ("diverges" if sampled_radius >= 1 else
                             "converges" if bound is not None and bound < 1 else "not certified")
        return {"radius": float(rep["radius"]), "dominant_eigenvalue": _complex(dominant),
                "full_response": full, "adjustment": float(self.adjustment),
                "sampled_adjusted_radius": sampled_radius,
                "adjusted_radius_bound": None if bound is None else float(bound),
                "adjusted_response": adjusted_response}

    def to_dict(self, *, include_result: bool = True) -> dict:
        # solve_ok reads `converged` rather than the "solve" category, which holds that one check:
        # the attribute is always there, while the check is absent when the solve ran diagnostics off.
        out = {"name": self.name, "total_cost": self.total_cost, "total_change": self.total_change,
               "total_change_fraction": self.total_change_fraction, "costs": self.cost_changes,
               "solve_ok": bool(self.result.converged),
               "assessment": self.result.diagnostics.assess().to_dict(),
               "dynamics": self.dynamics}
        if include_result:
            out["result"] = self.result.to_dict()
        return out


class ComparisonResult:
    """Results for named scenarios with changes measured against one baseline."""

    def __init__(self, cases: Mapping[str, tuple], baseline: str, adjustment: float):
        self.baseline = baseline
        base_costs = cases[baseline][1].costs
        self.cases = {name: ScenarioResult(name, model, result, base_costs, adjustment)
                      for name, (model, result) in cases.items()}

    def __getitem__(self, name: str) -> ScenarioResult:
        return self.cases[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def to_dict(self, *, include_results: bool = True) -> dict:
        return {"baseline": self.baseline,
                "scenarios": [case.to_dict(include_result=include_results) for case in self.cases.values()]}

    HEAD = ("scenario", "total cost", "vs baseline", "solve", "assessment", "full response")
    ALIGN = "<>><<<"           # the name left, the two numbers right, the verdicts left

    def summary(self, policy=None) -> str:
        """The comparison as a table: cost, change against the baseline, and the verdicts apart.

        The assessment column names the first blocking check rather than collapsing the checks to
        a word: "unsupported: resolution" and "failed: resolution" call for different actions, and
        a single verdict word cannot say which happened.  Response dynamics are a column of their
        own -- an unstable best response is usually the finding, not a fault in the numbers.
        """
        from .diagnostics import Policy
        policy = policy or Policy.PUBLICATION
        rows = []
        for name, case in self.cases.items():
            data = case.to_dict(include_result=False)
            verdict = case.result.diagnostics.assess(policy)
            assessed = ("ok" if verdict.accepted
                        else f"{verdict.blocking[0].status}: {verdict.blocking[0].check}")
            rows.append((name, f"{case.total_cost:.5f}",
                         "baseline" if name == self.baseline else f"{case.total_change_fraction:+.2%}",
                         "ok" if data["solve_ok"] else "failed", assessed,
                         data["dynamics"]["full_response"] if data["dynamics"] else "not checked"))
        width = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                 for i, h in enumerate(self.HEAD)]
        def line(cells):
            return "  ".join(f"{c:{self.ALIGN[i]}{width[i]}}" for i, c in enumerate(cells)).rstrip()
        return "\n".join([line(self.HEAD)] + [line(r) for r in rows])

    def __str__(self) -> str:
        return self.summary()


def compare(models: Mapping[str, object], *, baseline: Optional[str] = None, numerics=None,
            stability: bool = False, adjustment: float = 0.5, solve_kw: Optional[dict] = None) -> ComparisonResult:
    """Solve named models independently and compare their costs with one baseline.

    Scenarios must have the same agents, horizon kind, discount, window and cost
    convention.  ``stability=True`` adds the response-dynamics classification;
    ``adjustment`` is the fraction of each best response applied in the damped
    counterfactual.
    """
    from . import as_model, solve

    if not isinstance(models, Mapping) or not models:
        raise ValueError("compare(): models must be a non-empty mapping of scenario names to models")
    names = list(models)
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("compare(): every scenario name must be a non-empty string")
    baseline = names[0] if baseline is None else baseline
    if baseline not in models:
        raise ValueError(f"compare(): baseline {baseline!r} is not a scenario; choose from {names}")
    if not 0 < adjustment <= 1:
        raise ValueError("compare(): adjustment must be in (0, 1]")

    prepared = {name: as_model(value) for name, value in models.items()}
    first = prepared[baseline]
    signature = (tuple(a.name for a in first.agents), first.horizon.kind,
                 float(first.horizon.discount), float(first.horizon.window))
    for name, model in prepared.items():
        here = (tuple(a.name for a in model.agents), model.horizon.kind,
                float(model.horizon.discount), float(model.horizon.window))
        if here != signature:
            raise ValueError(f"compare(): scenario {name!r} has incompatible agents or horizon; "
                             "cost comparisons require the same agents, kind, discount and window")

    solved = {}
    expected_cost_kind = None
    for name, model in prepared.items():
        result = solve(model, Numerics.of(numerics), **dict(solve_kw or {}))
        if expected_cost_kind is None:
            expected_cost_kind = result.cost_kind
        elif result.cost_kind != expected_cost_kind:
            raise ValueError(f"compare(): scenario {name!r} has incompatible cost convention {result.cost_kind!r}")
        if stability:
            result.stability()
        solved[name] = (model, result)
    return ComparisonResult(solved, baseline, adjustment)
