import json, os, yaml
import noisestate as ns
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_cli_validate_solve_sweep_on_every_engine(tmp_path, capsys):
    assert main(["validate", os.path.join(EX, "ch5_cycle_market.yaml")]) == 0
    cells = ns.load(os.path.join(EX, "ch1_two_player_finite.yaml")).to_dict(); cells["horizon"] = {"kind": "finite", "T": 1.0}; cells["numerics"] = {"engine": "cells", "nodes": 24}
    cells_path = tmp_path / "cells.yaml"; cells_path.write_text(yaml.safe_dump(cells))
    for name, path in (("ch3", os.path.join(EX, "ch3_two_player.yaml")), ("ch1", os.path.join(EX, "ch1_two_player_finite.yaml")), ("cells", str(cells_path))):
        out = tmp_path / f"{name}.json"; plot = tmp_path / f"{name}.png"
        assert main(["solve", path, "-o", str(out), "--plot", str(plot)]) == 0
        d = json.load(open(out)); assert d["converged"] and d["kernels"] and d["grid"]["kind"] in ("stationary", "finite", "finite_cells")
        assert d["version"] == ns.__version__ and d["params"] and d["model"]["name"] == d["name"] and d["options"]["numerics"]["nodes"]
        assert d["agents"]["player1"]["controls"] == ["D1"] and "delay" in d["agents"]["player1"]["signals"]["y1"] and d["map_convention"]
        assert plot.stat().st_size > 1000
    sw = tmp_path / "sw.json"
    assert main(["sweep", os.path.join(EX, "ch3_two_player.yaml"), "p2", "3,5", "-o", str(sw)]) == 0
    sw_rows = json.load(open(sw))        # JSON: dicts, not SweepPoints
    assert len(sw_rows) == 2 and all(r["param"] == "p2" and r["result"]["params"]["p2"] == r["value"] for r in sw_rows)
    assert all("change" in r and "jump" in r for r in sw_rows)
    table = capsys.readouterr().out
    assert "residual" in table and "change" in table and "NOT converged" not in table
    figure = tmp_path / "sweep-figure"
    assert main(["plot-sweep", str(sw), str(figure)]) == 0
    report = capsys.readouterr().out
    assert (tmp_path / "sweep-figure.png").stat().st_size > 1000
    assert f"wrote {figure}.png" in report

def test_cli_file_errors_are_usage_errors(tmp_path, capsys):
    # a missing model file, bad YAML and an output path that cannot be written print `error: ...` and exit 2
    # as the contract says; they raised through main (a traceback and exit 1, the code of an unconverged solve)
    assert main(["solve", str(tmp_path / "missing.yaml")]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    bad = tmp_path / "bad.yaml"; bad.write_text("name: x\nagents: [\n")
    assert main(["validate", str(bad)]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert main(["solve", os.path.join(EX, "ch3_two_player.yaml"), "-o", str(tmp_path / "no_such_dir" / "out.json")]) == 2
    assert capsys.readouterr().err.startswith("error: ")

def test_top_level_solve_rejects_unknown_options():
    import pytest
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(os.path.join(EX, "ch3_two_player.yaml"), tolerance=1e-8)
    res = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), tol=1e-8, verbose=False)
    assert isinstance(res, ns.Result) and res.require_converged() is res


def test_require_ok_covers_the_guards_that_check_does_not(tmp_path, capsys):
    """check() is convergence alone, so it passes on ch3_two_player while a guard fails; require_ok() is
    the whole verdict, and --require-ok gives the command line the same split.  A script that reads only
    the exit status would otherwise take a WINDOW TOO SHORT result as sound."""
    import pytest
    r = ns.solve(os.path.join(EX, "ch3_two_player.yaml"))
    assert r.require_converged() is r and not r.diagnostics.assess().accepted
    with pytest.raises(ns.DiagnosticsError, match="converged, but"):
        r.require_ok()
    #  a model whose every required check actually RAN and passed
    assert ns.solve(os.path.join(EX, "ch1_two_player_finite.yaml")).require_ok().diagnostics.assess().accepted
    #  and one where a required check could not be run at all: the cell engine computes no
    #  second-order form.  Nothing performed the check, so the result is not accepted for it.
    #  (This stanza used to use ch4_kyle_back, on the wrong ground that a positive discount takes
    #  the curvature away; it does not, and that example is accepted now.)
    with pytest.raises(ns.DiagnosticsError, match="second_order"):
        ns.solve(os.path.join(EX, "ch1_two_player_finite.yaml"), {"engine": "cells", "nodes": 10}).require_ok()
    assert main(["solve", os.path.join(EX, "ch3_two_player.yaml")]) == 0
    capsys.readouterr()
    assert main(["solve", os.path.join(EX, "ch3_two_player.yaml"), "--require-ok"]) == 1
    report = capsys.readouterr()
    assert "Diagnostics: 1 failed" in report.out and "Numerics    FAIL — window" in report.out
    assert "WINDOW TOO SHORT" not in report.out
    assert "failed diagnostic checks:" in report.err and "raise horizon.window" not in report.err
    assert main(["solve", os.path.join(EX, "ch3_two_player.yaml"), "--diagnostics"]) == 0
    detailed = capsys.readouterr().out
    assert "meaning: Whether stationary kernels" in detailed and "suggested: --window 6" in detailed
    #  the CLI splits the same way: a model whose required checks all ran and passed exits 0 --
    #  ch4_kyle_back among them, now that the discount no longer excludes its second-order check --
    #  and a model where one of them cannot be run at all exits 1
    assert main(["solve", os.path.join(EX, "ch1_two_player_finite.yaml"), "--require-ok"]) == 0
    capsys.readouterr()
    assert main(["solve", os.path.join(EX, "ch4_kyle_back.yaml"), "--require-ok"]) == 0
    capsys.readouterr()
    assert main(["solve", os.path.join(EX, "ch1_two_player_finite.yaml"), "--engine", "cells",
                 "--nodes", "10", "--require-ok"]) == 1


def test_describe_prints_the_model_as_equations():
    assert main(["describe", os.path.join(EX, "ch3_two_player.yaml")]) == 0


def test_require_ok_says_cannot_be_checked_rather_than_failed(capsys):
    """It used to print "failed diagnostic checks: X" whatever the status was, so a result whose
    summary read "0 failed, 3 passed" was refused for a check that had FAILED according to one line
    and passed according to the next.  A check the engine cannot run is not a check the model
    failed, and the CLI must not collapse the two any more than the library does.
    """
    import noisestate as ns
    assert main(["solve", ns.example("ch1_two_player_finite"), "--engine", "cells",
                 "--nodes", "10", "--require-ok"]) == 1
    err = capsys.readouterr().err
    assert "cannot be checked here" in err and "second_order" in err
    assert "failed diagnostic checks" not in err
    assert "no setting will change it" in err                  # it is the engine, not the tuning
    assert "'spectral'" in err                                 # and the engine that can is named


def test_a_weaker_standard_can_be_asked_for_by_name(capsys):
    """The library has had policies since 0.8 and the CLI hardcoded PUBLICATION, so the weaker
    standard it insists be NAMED could not be named from the command line at all."""
    import noisestate as ns
    argv = ["solve", ns.example("ch1_two_player_finite"), "--engine", "cells", "--nodes", "10", "--require-ok"]
    assert main(argv + ["--policy", "exploratory"]) == 0
    assert main(argv) == 1
    capsys.readouterr()


def test_a_real_failure_is_still_reported_as_a_failure(capsys):
    """Separating unsupported out must not stop a genuine guard failure reading as one."""
    import noisestate as ns
    assert main(["solve", ns.example("ch3_two_player"), "--nodes", "12", "--require-ok"]) == 1
    err = capsys.readouterr().err
    assert "failed diagnostic checks:" in err and "window" in err
    assert "cannot be checked here" not in err


def test_the_flagship_example_is_acceptable(capsys):
    """ch4_kyle_back is the model the README prints in full, and --require-ok refused it at every
    resolution: the second-order check was excluded at a positive discount, wrongly."""
    import noisestate as ns
    assert main(["solve", ns.example("ch4_kyle_back"), "--nodes", "24", "--require-ok"]) == 0
    capsys.readouterr()
