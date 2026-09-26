"""JSON Schema (draft 2020-12) for the model file and for the result payload, and a validator.

    import noisestate as ns
    ns.schema("model")                     # the model file's schema (the numerics block)
    ns.schema("payload")                   # res.to_dict()'s schema
    ns.schema.validate(doc, "model")       # [] or the errors as "path: message"

`validate` uses the `jsonschema` package when it is importable (a full draft 2020-12 validator); otherwise the
small validator here covers exactly the keywords these two schemas use (type, enum, const, properties,
required, additionalProperties, items, minItems, minimum, exclusiveMinimum, anyOf, $ref into $defs,
) and reports every violation with its path.  The CLI's `validate` runs it before the model's own
checks, and `noisestate schema {model|payload}` prints the schema.
"""
from __future__ import annotations

from dataclasses import fields
from typing import List

from .numerics import ENGINE_NAMES
from ._settings import Settings

DRAFT = "https://json-schema.org/draft/2020-12/schema"

_NUMBER_OR_EXPR = {"anyOf": [{"type": "number"}, {"type": "string"}],
                   "description": "a number, or an expression in the parameters"}
_EXPR = {"anyOf": [{"type": "object", "additionalProperties": _NUMBER_OR_EXPR},
                   {"type": "array", "items": {"type": "array", "minItems": 2}}],
         "description": "a linear expression: {atom: coef} (an atom is name or name@lag; const for a constant), or [[coef, atom], ...]"}
_LOSS_TERM = {"type": "array", "minItems": 2, "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
              "description": "[coef, a, b] (quadratic) or [coef, a] (linear)"}
_SHOCK = {"type": "object", "additionalProperties": False, "required": ["name"],
          "properties": {"name": {"type": "string"},
                         "loads": {"type": "object", "additionalProperties": _NUMBER_OR_EXPR},
                         "rows": {"type": "object", "additionalProperties": _NUMBER_OR_EXPR}}}


def _settings_schema() -> dict:
    props = {}
    for f in fields(Settings):
        props[f.name] = {"type": "integer" if f.type == "int" else "number"}
    return {"type": "object", "additionalProperties": False, "properties": props,
            "description": "the tuning constants (noisestate.Settings): the fields that differ from the defaults"}


def numerics_schema() -> dict:
    """The numerics block (noisestate.Numerics)."""
    return {"type": "object", "additionalProperties": False,
            "description": "how the model is solved: the engine, the grid, the fixed point's options, the settings",
            "properties": {
                "engine": {"enum": list(ENGINE_NAMES), "description": "default from horizon.kind: stationary -> stationary, else spectral"},
                "nodes": {"type": "integer", "minimum": 2, "description": "nodes per panel (stationary) or per side of each piece (spectral); cells on the cell engine; default 16"},
                "unit": {**_NUMBER_OR_EXPR, "description": "the panel unit: every lag and delay must be a multiple of it"},
                "unit_range": {**_NUMBER_OR_EXPR, "description": "the age (stationary) or time (spectral) up to which the panels are unit panels"},
                "breakpoints": {"type": "array", "minItems": 2, "items": _NUMBER_OR_EXPR, "description": "an explicit panel sequence from 0 to the window"},
                "continuation_nodes": {"type": "integer", "minimum": 2, "description": "a transition's stationary continuation solved at this many nodes"},
                "tol": {**_NUMBER_OR_EXPR, "description": "the fixed point's tolerance (default 1e-10 stationary, 1e-8 finite)"},
                "damping": {**_NUMBER_OR_EXPR, "description": "the Anderson mixing weight"},
                "max_newton": {"type": "integer", "minimum": 0, "description": "Newton-Krylov polish steps at most"},
                "variable": {"enum": ["actions", "maps"], "description": "the fixed point's iterate"},
                "settings": _settings_schema()}}


def model_schema() -> dict:
    """The model file (a Model.to_dict() / Model.from_dict() document)."""
    horizon = {"type": "object", "additionalProperties": False,
               "description": "the economics of time: the kind, the discount, the window, a transition's past and continuation",
               "properties": {
                   "kind": {"enum": ["stationary", "finite", "transition"]},
                   "discount": _NUMBER_OR_EXPR,
                   "window": {**_NUMBER_OR_EXPR, "description": "the lag-truncation length L: a stationary "
                              "horizon, and a transition's continuation (which defaults to the past's). A finite "
                              "horizon has none -- its length is T"},
                   "T": {**_NUMBER_OR_EXPR, "description": "the terminal time: a finite horizon or a transition. "
                         "A stationary horizon has none -- its length is the lag window"},
                   "past": {"type": "object", "additionalProperties": False,
                            "properties": {"model": {"anyOf": [{"type": "string"}, {"type": "object"}],
                                                     "description": "the old stationary model: a path (relative to the file) or an inline model"},
                                           "initial": {"type": "array", "items": _SHOCK, "description": "initial shocks {name, loads, rows}"}},
                            "description": "kind transition only"},
                   "continuation": {"enum": ["stationary", "end"], "description": "kind transition only; default stationary"},
                   "settle": {**_NUMBER_OR_EXPR, "description": "kind transition only, in place of window: the settle tolerance the "
                              "horizon T is found for by a march in T (exactly one of window and settle)"},
                   "stationary": {"type": "object", "additionalProperties": False,
                                  "properties": {"window": {**_NUMBER_OR_EXPR, "description": "must equal the past's window"}},
                                  "description": "kind transition only: the continuation's stationary solve"}}}
    state = {"type": "object", "additionalProperties": False,
             "properties": {"drift": _EXPR, "noise": _EXPR, "initial": {**_NUMBER_OR_EXPR, "description": "finite horizon only; moves the means"}}}
    signal = {"type": "object", "additionalProperties": False,
              "properties": {"drift": _EXPR, "noise": _EXPR, "delay": {**_NUMBER_OR_EXPR, "description": "observation delay"}}}
    agent = {"type": "object", "additionalProperties": False, "required": ["controls"],
             "properties": {"controls": {"type": "array", "items": {"type": "string"}},
                            "signals": {"type": "object", "additionalProperties": signal},
                            "loss": {"type": "array", "items": _LOSS_TERM},
                            "myopic": {"type": "boolean"},
                            "constant": {**_NUMBER_OR_EXPR, "description": "the loss's constant: part of the cost, moves no strategy"},
                            "terminal": {"type": "array", "items": _LOSS_TERM, "description": "the loss at T, on the states (finite horizon)"},
                            "terminal_constant": {**_NUMBER_OR_EXPR, "description": "the terminal loss's constant"}}}
    return {"$schema": DRAFT, "$id": "https://noisestate/schema/model", "title": "noisestate model file",
            "type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object", "additionalProperties": _NUMBER_OR_EXPR,
                           "description": "parameters, evaluated in order (a later one may use an earlier one)"},
                "channels": {"type": "array", "items": {"type": "string"}, "description": "the Brownian channels"},
                "states": {"type": "object", "additionalProperties": state},
                "definitions": {"type": "object", "additionalProperties": _EXPR},
                "agents": {"type": "object", "additionalProperties": agent},
                "ties": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                         "description": "groups of agents sharing one strategy"},
                "horizon": horizon,
                "numerics": numerics_schema()}}


PAYLOAD_VERSION = 2          # see payload_schema()'s docstring for what changed


def payload_schema() -> dict:
    """The result payload (Result.to_dict()), payload_version 2.

    VERSION 2 (0.8) changed six things, and a consumer written against version 1 will not read it:

        means_t                 -> mean_times, matching the Python attribute
        status {ok, flags}      -> assessment {policy, accepted, statuses, blocking, uncomputed}
        resolution_ok           -> gone; the resolution check's status is in assessment.statuses
        options.solve.start     -> start_policy
        stability               -> carries verified, fixed_point_residual, residual_norm and the
                                   reasons, and requires them: a radius with no statement about
                                   whether the point is an equilibrium invites the misreading
        refinement              -> the Refinement's to_dict(), not the object

    options.solve and options.solver enumerate their keys and refuse the rest, which is what makes
    the next such change a version bump rather than something nobody notices.
    """
    numbers = {"type": "array", "items": {"type": "number"}}
    by_name = lambda inner: {"type": "object", "additionalProperties": inner}   # noqa: E731
    row = {"type": "object", "required": ["name", "value", "threshold", "ok", "flag", "advice"],
           "properties": {"name": {"type": "string"}, "ok": {"type": ["boolean", "null"]}, "flag": {"type": "string"},
                          "advice": {"type": "string"}, "code": {"type": "string"},
                          "category": {"enum": ["solve", "numerics", "equilibrium"]},
                          "severity": {"enum": ["ok", "info", "error"]}, "meaning": {"type": "string"},
                          "action": {"type": "string"}, "suggested_options": {"type": "object"},
                          "trend": {"type": "object"}}}
    return {"$schema": DRAFT, "$id": "https://noisestate/schema/payload", "title": "noisestate result payload",
            "type": "object",
            "required": ["payload_version", "version", "name", "engine", "kind", "converged", "residual", "evaluations", "seconds",
                         "message", "params", "model", "horizon", "numerics", "axes", "times", "options", "grid", "discount", "channels",
                         "agents", "map_convention", "kernels", "maps", "foc", "costs", "cost_parts", "means", "mean_times",
                         "representation_error", "diagnostics", "assessment", "cost_kind", "second_order", "notes"],
            "properties": {
                "payload_version": {"const": PAYLOAD_VERSION},
                "version": {"type": "string", "description": "the package version that wrote it"},
                "name": {"type": "string"},
                "engine": {"enum": list(ENGINE_NAMES)},
                "kind": {"enum": ["stationary", "finite", "transition", "finite_cells"], "description": "the result kind"},
                "converged": {"type": "boolean"}, "residual": {"type": "number"}, "evaluations": {"type": "integer"},
                "seconds": {"type": "number"}, "message": {"type": "string"},
                "params": by_name({"type": "number"}),
                "model": {"$ref": "#/$defs/model"},
                "horizon": {"type": "object", "description": "the model's horizon block (the economics)"},
                "numerics": {"$ref": "#/$defs/numerics", "description": "the resolved numerics"},
                "axes": by_name(numbers),
                "times": {"anyOf": [numbers, {"type": "null"}], "description": "the time nodes of the paths; null on the stationary engine"},
                #  The two option blocks ENUMERATE their keys and refuse anything else.  Declaring
                #  them as bare objects meant a rename inside them validated silently: `start`
                #  became `start_policy` and nothing noticed, which is the opposite of what a
                #  schema on a wire format is for.  additionalProperties is what does the
                #  refusing -- listing properties alone accepts an obsolete key beside them.
                "options": {"type": "object", "required": ["numerics", "solver", "solve"],
                            "additionalProperties": False,
                            "properties": {
                                "numerics": {"$ref": "#/$defs/numerics"},
                                "solver": {"type": "object", "additionalProperties": False,
                                           "properties": {"verbose": {"type": "boolean"},
                                                          "naive_observers": {"type": ["object", "null"]},
                                                          "past": {},
                                                          "continuation": {}}},
                                "solve": {"type": "object", "additionalProperties": False,
                                          "properties": {"start_from": {},
                                                         "start_policy": {"type": ["string", "null"]},
                                                         "tol": {"type": ["number", "null"]},
                                                         "damping": {"type": ["number", "null"]},
                                                         "max_newton": {"type": ["integer", "null"]},
                                                         "variable": {"type": ["string", "null"]},
                                                         "max_evaluations": {"type": ["integer", "null"]},
                                                         "deadline": {"type": ["number", "null"]},
                                                         "diagnostics": {"type": "boolean"}}}}},
                "grid": {"type": "object", "required": ["kind"]},
                "discount": {"type": "number"},
                "channels": {"type": "array", "items": {"type": "string"}},
                "agents": by_name({"type": "object", "required": ["controls", "signals"],
                                   "properties": {"controls": {"type": "array", "items": {"type": "string"}},
                                                  "signals": by_name({"type": "object", "required": ["delay"],
                                                                      "properties": {"delay": {"type": "number"}},
                                                                      "additionalProperties": numbers})}}),
                "map_convention": {"type": "string"},
                "kernels": by_name(by_name({"type": "array"})),
                "maps": by_name(by_name(by_name({"type": "array"}))),
                "foc": by_name(by_name(by_name(by_name(numbers)))),
                "costs": by_name({"type": "number"}),
                "cost_parts": by_name(by_name({"type": "number"})),
                "means": by_name({"anyOf": [{"type": "number"}, numbers]}),
                "mean_times": {"anyOf": [numbers, {"type": "null"}]},
                "representation_error": by_name({"type": "number"}),
                "representation_parts": by_name(by_name({"type": "number"})),
                "diagnostics": {"type": "array", "items": row},
                "assessment": {"type": "object", "required": ["policy", "accepted", "statuses"],
                               "properties": {"policy": {"type": "string"}, "accepted": {"type": "boolean"},
                                              "statuses": {"type": "object", "additionalProperties": {"type": "string"}},
                                              "blocking": {"type": "array", "items": {"type": "object"}},
                                              "uncomputed": {"type": "array", "items": {"type": "string"}}}},
                "cost_kind": {"type": "string"},
                "second_order": by_name({"type": "object"}),
                "notes": {"type": "array", "items": {"type": "string"}},
                "refinement": {"type": "object"}, "window_tail": {"type": "number"},
                #  verified is required because the evidence is the point: a payload that carried a
                #  radius without saying whether the point is an equilibrium invited the reader to
                #  treat the spectrum as equilibrium stability.
                "stability": {"type": "object", "required": ["radius", "stable", "verified",
                                                            "fixed_point_residual", "residual_norm"]},
                "past": {"type": "object"}, "settled": {"type": ["number", "null"]}, "continuation": {"type": "object"},
                "loss_path": by_name(numbers), "belief_error": by_name(by_name(numbers)),
                "excess_costs": by_name({"type": "number"}),
                "T": {"type": "number"}, "march": {"type": "array"}, "march_stop": {"type": ["string", "null"]},
                "excess_windows": by_name(numbers), "excess_costs_tail": by_name({"type": "number"}),
                "excess_costs_total": by_name({"type": "number"}), "excess_tail": {"type": "object"},
                "march_settle": {"type": ["number", "null"]}, "settle_floor": {"type": ["object", "null"]},
                "old_flows": by_name({"type": "number"}), "new_flows": by_name({"type": "number"})},
            "$defs": {"model": {k: v for k, v in model_schema().items() if k not in ("$schema", "$id")},
                      "numerics": numerics_schema()}}


def schema(which: str = "model") -> dict:
    """The JSON Schema (draft 2020-12) of the model file ("model") or of the result payload ("payload")."""
    if which == "model":
        return model_schema()
    if which == "payload":
        return payload_schema()
    raise ValueError(f"schema() takes 'model' or 'payload', not {which!r}")


# ------------------------------------------------------------------ validation
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(v, t: str) -> bool:
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v == int(v))
    return isinstance(v, _TYPES[t])


def _validate(v, sch: dict, root: dict, path: str, errors: List[str]) -> None:
    if "$ref" in sch:
        node = root
        for part in sch["$ref"].split("/")[1:]:
            node = node[part]
        sch = {**node, **{k: x for k, x in sch.items() if k != "$ref"}}
    t = sch.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_is_type(v, x) for x in ts):
            errors.append(f"{path or '(root)'}: expected {' or '.join(ts)}, got {type(v).__name__}"); return
    if "enum" in sch and v not in sch["enum"]:
        errors.append(f"{path or '(root)'}: {v!r} is not one of {sch['enum']}"); return
    if "const" in sch and v != sch["const"]:
        errors.append(f"{path or '(root)'}: expected {sch['const']!r}, got {v!r}"); return
    if "anyOf" in sch:
        for alt in sch["anyOf"]:
            sub: List[str] = []
            _validate(v, alt, root, path, sub)
            if not sub:
                break
        else:
            errors.append(f"{path or '(root)'}: {v!r} matches none of the allowed forms"); return
    if isinstance(v, dict):
        props = sch.get("properties", {})
        for k in sch.get("required", []):
            if k not in v:
                errors.append(f"{path or '(root)'}: missing required key {k!r}")
        extra = sch.get("additionalProperties", True)
        for k, x in v.items():
            here = f"{path}.{k}" if path else str(k)
            if k in props:
                _validate(x, props[k], root, here, errors)
            elif extra is False:
                errors.append(f"{here}: unknown key (allowed: {sorted(props)})")
            elif isinstance(extra, dict):
                _validate(x, extra, root, here, errors)
    if isinstance(v, list):
        if "minItems" in sch and len(v) < sch["minItems"]:
            errors.append(f"{path or '(root)'}: at least {sch['minItems']} items expected, got {len(v)}")
        if "items" in sch:
            for i, x in enumerate(v):
                _validate(x, sch["items"], root, f"{path}[{i}]", errors)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in sch and v < sch["minimum"]:
            errors.append(f"{path or '(root)'}: {v!r} is below the minimum {sch['minimum']}")
        if "exclusiveMinimum" in sch and v <= sch["exclusiveMinimum"]:
            errors.append(f"{path or '(root)'}: {v!r} must exceed {sch['exclusiveMinimum']}")


def validate(instance, which: str = "model") -> List[str]:
    """The schema errors of `instance` against schema(which), each "path: message"; [] when it validates.
    Uses the jsonschema package when importable, else the validator of this module.  A model written as
    equations (noisestate.equations) is read into the grammar first; an equation it cannot read is the error."""
    if which == "model":
        from . import equations
        if equations.is_equation_form(instance):
            try:
                instance = equations.to_grammar(instance)
            except (ValueError, TypeError) as e:
                return [f"(equations): {e}"]
    sch = schema(which)
    try:
        import jsonschema
    except ImportError:
        errors: List[str] = []
        _validate(instance, sch, sch, "", errors)
        return errors
    out = []
    for e in sorted(jsonschema.Draft202012Validator(sch).iter_errors(instance), key=lambda e: list(e.absolute_path)):
        p = ".".join(str(x) for x in e.absolute_path)
        out.append(f"{p or '(root)'}: {e.message}")
    return out


schema.validate = validate                 # ns.schema.validate(doc, "model")
