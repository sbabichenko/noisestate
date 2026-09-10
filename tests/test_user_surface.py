"""The user-facing surface: the shipped examples, the reprs, the declared methods, and the docs.

These are the things a person meets in the first ten minutes -- the first example, what a notebook
prints, what dir() shows, what the CLI calls the horizon -- and each one here was wrong at some point
without a test noticing.
"""
import json
import os
import re
import subprocess
import sys

import pytest

import noisestate as ns
from noisestate.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#  ----------------------------------------------------------------- the shipped examples
def test_every_shipped_example_resolves_and_loads():
    """ns.example() reaches all seven, and each one is a model the package can build.

    The README's first line used a repository-relative path, which a pip install does not have."""
    names = ns.examples()
    assert len(names) == 7 and "ch1_two_player_finite" in names
    for name in names:
        path = ns.example(name)
        assert os.path.exists(path) and path.endswith(".yaml")
        assert ns.load(path).name                      # builds, validates, has a name
    assert ns.example("ch4_kyle_back.yaml") == ns.example("ch4_kyle_back")   # the suffix is tolerated


def test_unknown_example_lists_the_real_ones():
    with pytest.raises(FileNotFoundError, match="ch1_two_player_finite"):
        ns.example("no_such_model")


def test_examples_are_declared_as_package_data():
    """They install into the wheel; without this the accessor works only from a checkout."""
    text = open(os.path.join(ROOT, "pyproject.toml")).read()
    assert '"noisestate.examples" = "examples"' in text
    assert '"noisestate.examples" = ["*.yaml"]' in text


#  ----------------------------------------------------------------- the first workflow
@pytest.mark.parametrize("name", ["ch1_two_player_finite"])
def test_the_readme_first_workflow_runs_and_is_accepted(name, tmp_path):
    """The README opens with this: it must converge AND pass publication, or the first thing a
    reader sees is a warning they have no context for."""
    model = ns.load(ns.example(name))
    res = ns.solve(model)
    res.require_converged()
    res.require_ok()                                    # publication, the default policy
    assert res.diagnostics.assess().accepted and not res.diagnostics.flags
    assert res.costs["player1"] == pytest.approx(0.39690577, rel=1e-6)
    out = str(tmp_path / "k.png")
    res.kernel("X", "w0").plot(out)
    assert os.path.getsize(out) > 0


#  ----------------------------------------------------------------- what a notebook prints
def test_result_repr_is_one_readable_line():
    """The generated dataclass repr was ~79,000 characters: the compiled engine and every array."""
    res = ns.solve(ns.example("ch1_two_player_finite"))
    text = repr(res)
    assert len(text) < 400 and "\n" not in text
    for expected in ("ch1_two_player_finite", "finite", "converged", "player1", "accepted"):
        assert expected in text
    assert "compiled=" not in text and "array(" not in text


def test_model_repr_is_one_line_naming_the_horizon_lengths():
    stat, fin = ns.load(ns.example("ch3_two_player")), ns.load(ns.example("ch1_two_player_finite"))
    assert "window=3" in repr(stat) and "T=" not in repr(stat)      # a lag window, no terminal time
    assert "T=1" in repr(fin) and "window=" not in repr(fin)        # and the other way round
    assert len(repr(stat)) < 200 and "State(" not in repr(stat)


def test_the_subclasses_inherit_the_short_repr():
    """Each engine's result is a dataclass too, and would regenerate the long form on its own."""
    for name, num in (("ch3_two_player", {"nodes": 8}), ("ch1_two_player_finite", None)):
        assert len(repr(ns.solve(ns.example(name), num))) < 400


#  ----------------------------------------------------------------- discoverability
def test_shared_methods_are_declared_on_the_public_class():
    """The engines' Result subclasses are internal, so a method defined only there appears on
    nothing a user can inspect: help(ns.Result), dir(), an editor's completion."""
    for name in ("summary", "plot", "kernel", "refine", "stability", "require_ok", "to_dict"):
        assert name in dir(ns.Result), name
        assert getattr(ns.Result, name).__doc__, f"{name} has no docstring"


def test_with_signal_accepts_its_three_documented_forms():
    from noisestate import Signal, Control, shocks
    model = ns.load(ns.example("ch1_two_player_finite"))
    w = shocks("w_flow")
    forms = [model.with_signal("flow", drift={"D1": 1}, noise={"w_flow": 1}),     # positional name
             model.with_signal(name="flow", drift={"D1": 1}, noise={"w_flow": 1}),   # keyword name
             model.with_signal(Signal("flow", Control("D1") + w.w_flow))]         # a Signal object
    for built in forms:
        assert "flow" in [r.name for r in built.agents[0].signals] and "w_flow" in built.channels
    #  without_signal() is the inverse, down to the channel the row brought in
    assert forms[0].without_signal("flow").to_dict() == model.to_dict()
    with pytest.raises(TypeError):                       # a Signal AND blocks is none of the forms
        model.with_signal(Signal("flow", Control("D1") + w.w_flow), drift={"D1": 1})


#  ----------------------------------------------------------------- the horizon vocabulary
def test_cli_names_the_two_horizon_lengths_apart(tmp_path, capsys):
    """--window is L and --T is the terminal time, in the CLI as in Python."""
    assert main(["solve", ns.example("ch1_two_player_finite"), "--T", "0.6", "--nodes", "6"]) == 0
    assert "[0, 0.6]" in capsys.readouterr().out
    #  asking for the length the kind does not have fails, and the message names the one it does
    assert main(["solve", ns.example("ch1_two_player_finite"), "--window", "2.0"]) == 2
    assert "horizon.T" in capsys.readouterr().err
    assert main(["solve", ns.example("ch3_two_player"), "--T", "2.0"]) == 2
    assert "horizon.window" in capsys.readouterr().err
    #  transition: --window named the terminal time before 0.8 and is refused by name
    with pytest.raises(SystemExit):
        main(["transition", ns.example("ch3_two_player"), ns.example("ch3_two_player"), "--window", "4"])


def test_validate_reports_the_length_the_kind_actually_has(capsys):
    main(["validate", ns.example("ch3_two_player")])
    assert "window (lag) 3" in capsys.readouterr().out
    main(["validate", ns.example("ch1_two_player_finite")])
    out = capsys.readouterr().out
    assert "T 1" in out and "window" not in out.split("lags")[0]


#  ----------------------------------------------------------------- the docs match the code
def test_payload_doc_documents_exactly_the_schema():
    """payload.md drifted through a whole redesign: it still described `status`, `resolution_ok`
    and `means_t` after the payload had moved to `assessment` and `mean_times`."""
    documented = set(re.findall(r"^\| `([a-zA-Z_]+)` \|", open(os.path.join(ROOT, "docs/payload.md")).read(), re.M))
    declared = set(ns.schema("payload")["properties"])
    assert documented == declared, {"undocumented": sorted(declared - documented),
                                    "not in the schema": sorted(documented - declared)}


def test_the_schema_declares_only_keys_the_payload_can_carry():
    """resolution_ok outlived its removal inside the schema, which is the published contract."""
    for dead in ("status", "resolution_ok", "means_t"):
        assert dead not in ns.schema("payload")["properties"], dead


def test_no_public_docstring_names_a_removed_call():
    """Docstrings kept recommending check(), res.status and resolution_ok after all three were gone."""
    removed = ("res.status", "status[\"ok\"]", "resolution_ok", "diagnostic_rows()", "diagnostic_verdict")
    offenders = []
    for obj in (ns, ns.Result, ns.Model, ns.solve, ns.Result.require_converged, ns.Result.require_ok):
        doc = obj.__doc__ or ""
        offenders += [(getattr(obj, "__name__", obj), bad) for bad in removed if bad in doc]
    assert not offenders, offenders


def test_the_readme_uses_only_calls_that_exist():
    """Every ns.X(...) and res.X(...) the README shows must resolve on the real objects."""
    text = open(os.path.join(ROOT, "README.md")).read()
    for name in sorted(set(re.findall(r"\bns\.([a-zA-Z_][a-zA-Z_0-9]*)", text))):
        assert hasattr(ns, name), f"README uses ns.{name}, which does not exist"
    #  against the classes, not one solve: the README's res.* spans every engine (a transition's
    #  belief_error is not on a finite result), and this stays fast enough to run every time.
    from noisestate import results as _r
    available = set()
    for cls in (ns.Result, _r.StationaryResult, _r.TriangleResult, _r.TransitionResult, _r.CellResult):
        available |= set(dir(cls))
        #  a dataclass field with no default is an ANNOTATION, not a class attribute, so it is
        #  absent from dir(); the annotation is the declaration, and giving it a default merely to
        #  surface it would be changing the API to satisfy an introspection quirk.
        for base in cls.__mro__:
            available |= set(getattr(base, "__annotations__", {}))
    for name in sorted(set(re.findall(r"\bres\.([a-zA-Z_][a-zA-Z_0-9]*)", text))):
        assert name in available, f"README uses res.{name}, which is on no result class"


def test_historical_reviews_are_labelled_as_history():
    """Their recommendations read as instructions until each says what became of it."""
    for name in ("2026-09-09-external-ux-review.txt", "2026-09-10-pre-0.8-glossary.txt"):
        head = open(os.path.join(ROOT, "docs/design/reviews", name)).read()[:1200].upper()
        assert "HISTORICAL" in head or "SUPERSEDED" in head, name
