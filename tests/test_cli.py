import json, os, yaml
import noisestate as ns
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_cli_validate_solve_sweep_on_every_engine(tmp_path, capsys):
    assert main(["validate", os.path.join(EX, "ch5_cycle_market.yaml")]) == 0
    cells = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); cells["horizon"] = {"kind": "finite", "window": 1.0}; cells["numerics"] = {"engine": "cells", "nodes": 24}
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
    sw_rows = json.load(open(sw)); assert len(sw_rows) == 2 and all(r["param"] == "p2" and r["result"]["params"]["p2"] == r["value"] for r in sw_rows)
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
    assert ns.solve(os.path.join(EX, "ch4_kyle_back.yaml")).require_ok().diagnostics.assess().accepted
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
    assert main(["solve", os.path.join(EX, "ch4_kyle_back.yaml"), "--require-ok"]) == 0


def test_describe_prints_the_model_as_equations():
    assert main(["describe", os.path.join(EX, "ch3_two_player.yaml")]) == 0
