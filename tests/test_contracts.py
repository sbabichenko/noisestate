"""The rules of docs/api_spec.txt PART 1, checked across the whole public surface.

These are different in kind from the rest of the suite.  Every other test fixes one behaviour;
these quantify over the API, so a name added later is covered without anyone remembering to cover
it.  That is the point: the rules exist to make a new name placeable without looking one up, and a
rule nothing enforces is a comment.

PART 10 says outright that passing these does not establish compliance -- T1 to T12 were drafted
before implementation and covered neither the absence of a horizon length nor with_horizon's
object-only signature, and both shipped green.  They are a growing record, not a checklist.
"""
import dataclasses
import inspect
import json

import numpy as np
import pytest

import noisestate as ns
from helpers import example

PUBLIC_TYPES = (ns.Model, ns.Result)


def public_names(cls):
    """Everything a caller can reach: no underscore, not inherited from object, not a constant."""
    return [n for n in dir(cls)
            if not n.startswith("_") and n not in dir(object) and not n.isupper()]


# ----------------------------------------------------------------- R1: with_* returns a copy

def _reachable(obj, seen=None, arrays=None, depth=0):
    """Every mutable object reachable from obj, by id, with the ndarrays collected separately."""
    IMMUTABLE = (str, int, float, bool, type(None), bytes, complex)
    if seen is None:
        seen, arrays = {}, []
    frozen = getattr(getattr(obj, "__dataclass_params__", None), "frozen", False)      # a frozen Settings may be shared
    if depth > 14 or isinstance(obj, IMMUTABLE) or frozen or id(obj) in seen:
        return seen, arrays
    if isinstance(obj, np.ndarray):
        seen[id(obj)] = "ndarray"; arrays.append(obj); return seen, arrays
    if isinstance(obj, (list, dict, set, bytearray)) or hasattr(obj, "__dict__"):
        seen[id(obj)] = type(obj).__name__
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reachable(k, seen, arrays, depth + 1); _reachable(v, seen, arrays, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _reachable(v, seen, arrays, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _reachable(v, seen, arrays, depth + 1)
    return seen, arrays


def _fingerprint(model):
    """The receiver's whole state, including what to_dict() leaves out.

    to_dict() carries everything that affects a SOLVE -- definitions and ties included; they are
    absent from ch3_two_player's dict only because that model has none, and a model with 26
    definitions and a tie round-trips identically.  What it omits is PROVENANCE: source (which
    from_dict rebuilds from the dict it is given, and which is what keeps with_params
    re-evaluating expressions).

    That is the right thing for a file format to omit, and the wrong thing for an immutability
    check to omit.
    """
    omitted = {f.name for f in dataclasses.fields(ns.Model)} - set(model.to_dict())
    return json.dumps({"file": model.to_dict(),
                       "omitted": {k: repr(getattr(model, k, None)) for k in sorted(omitted)}},
                      sort_keys=True)


@pytest.mark.parametrize("name", ["ch3_two_player", "ch1_two_player_finite", "ch3_precision_change"])
def test_T1_with_star_returns_an_independent_copy(name):
    """Across the three horizon kinds, and by TRAVERSAL rather than a spot check: an earlier draft
    claimed the contract from one mutated field, which establishes only that field."""
    m = example(name)
    copies = [("with_params", lambda: m.with_params(**{k: float(v) for k, v in list(m.params.items())[:1]})),
              ("with_numerics", lambda: m.with_numerics(nodes=int(m.numerics.nodes) + 2)),
              ("with_signal", lambda: m.with_signal("probe_row", f"{m.state_names[0]} dt + dw_probe"))]
    before_ids, before_arrays = _reachable(m)
    #  R1 has TWO halves and an earlier version of this test checked only one: the copy shares
    #  nothing, AND the receiver is unchanged.  Appending to the receiver's own list creates no
    #  sharing at all, so the id traversal passes while the contract is broken.
    untouched = _fingerprint(m)
    for label, make in copies:
        copy = make()
        assert _fingerprint(m) == untouched, f"{label} mutated the receiver"
        after_ids, after_arrays = _reachable(copy)
        shared = set(before_ids) & set(after_ids)
        assert not shared, f"{label} shares {[before_ids[i] for i in shared][:3]} with the receiver"
        #  distinct ndarray objects can still share writable storage, so ids are not enough
        pairs = [(x, y) for x in before_arrays for y in after_arrays if np.shares_memory(x, y)]
        assert not pairs, f"{label} shares array storage with the receiver"


def test_T1_the_buffer_check_is_live_not_vacuous():
    """It found nothing above because no ndarray is reachable from a Model today.  That is worth
    saying: the check is VACUOUS here, not passed, and it becomes live the moment one is."""
    _, arrays = _reachable(example("ch3_two_player"))
    assert arrays == [], "a Model now reaches an ndarray -- the shares_memory check above is live"


# ----------------------------------------------------------------- R2: properties vs methods

@pytest.mark.parametrize("cls", PUBLIC_TYPES, ids=lambda c: c.__name__)
def test_T2_no_public_property_takes_arguments(cls):
    """A property exposes data; anything parameterized is a method.  A noun METHOD is fine --
    kernel(name, channel) is a parameterized query -- so the rule is about the shape, not the word."""
    for n in public_names(cls):
        member = inspect.getattr_static(cls, n, None)
        if isinstance(member, property):
            params = list(inspect.signature(member.fget).parameters)
            assert params == ["self"], f"{cls.__name__}.{n} is a property taking {params[1:]}"


# ----------------------------------------------------------------- R3: predicates are booleans

def test_T3_every_predicate_name_returns_a_real_bool():
    """is_/has_/drives_ return bool, never None.  A multivalued assessment uses a Status or a
    status object -- that distinction is what res.diagnostics.statuses exists to carry."""
    checked = 0
    for numerics in ({"nodes": 8},):                   # the cell engine's result: extras/test_cells.py
        model = example("ch1_two_player_finite")
        res = ns.solve(model, numerics, diagnostics=False)
        for obj in (res, res.model):
            for n in dir(obj):
                if n.startswith(("is_", "has_", "drives_")) and not n.startswith("_"):
                    value = getattr(obj, n)
                    assert isinstance(value, bool), f"{type(obj).__name__}.{n} is {type(value).__name__}"
                    checked += 1
    assert checked, "no predicate names found -- the check would pass vacuously"


# ----------------------------------------------------------------- R4: require_* raises

def test_T4_every_require_returns_the_receiver_or_raises():
    res = ns.solve(example("ch3_two_player").with_numerics(nodes=6))
    names = [n for n in public_names(ns.Result) if n.startswith("require_")]
    assert names, "no require_* members found -- the check would pass vacuously"
    for n in names:
        try:
            assert getattr(res, n)() is res, f"{n} returned something other than the receiver"
        except ns.ResultValidationError:
            pass                                        # raising is the other half of the contract


# ----------------------------------------------------------------- R6: the *_ok suffix is retired

@pytest.mark.parametrize("cls", PUBLIC_TYPES, ids=lambda c: c.__name__)
def test_T5_no_public_property_carries_the_ok_suffix(cls):
    """It read as two-valued and was not: resolution_ok returned None on the cell engine.  Private
    helpers keep it -- _resolution_ok feeds a diagnostic row whose `ok` field IS tri-state by
    contract -- and require_ok() is a require_* METHOD, exempt by R4."""
    for n in public_names(cls):
        if n.endswith("_ok") and not n.startswith("require_"):
            assert not isinstance(inspect.getattr_static(cls, n, None), property), \
                f"{cls.__name__}.{n} is a public *_ok property"
