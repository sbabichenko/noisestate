import json, os, yaml
import noisestate as ns
from noisestate.cli import main
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_cli_validate_solve_sweep_on_every_engine(tmp_path):
    assert main(["validate", os.path.join(EX, "ch5_cycle_market.yaml")]) == 0
    cells = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); cells["horizon"] = {"kind": "finite_cells", "window": 1.0, "nodes": 24}
    cells_path = tmp_path / "cells.yaml"; cells_path.write_text(yaml.safe_dump(cells))
    for name, path in (("ch3", os.path.join(EX, "ch3_two_player.yaml")), ("ch1", os.path.join(EX, "ch1_two_player_finite.yaml")), ("cells", str(cells_path))):
        out = tmp_path / f"{name}.json"; plot = tmp_path / f"{name}.png"; npz = tmp_path / f"{name}.npz"
        assert main(["solve", path, "-o", str(out), "--plot", str(plot)]) == 0
        assert main(["solve", path, "-o", str(npz)]) == 0
        d = json.load(open(out)); assert d["converged"] and d["kernels"] and d["grid"]["kind"] in ("stationary", "finite", "finite_cells")
        assert plot.stat().st_size > 1000 and npz.stat().st_size > 100
    sw = tmp_path / "sw.json"
    assert main(["sweep", os.path.join(EX, "ch3_two_player.yaml"), "p2", "3,5", "-o", str(sw)]) == 0
    assert len(json.load(open(sw))) == 2

def test_top_level_solve_rejects_unknown_options():
    import pytest
    with pytest.raises(TypeError, match="unknown option"):
        ns.solve(os.path.join(EX, "ch3_two_player.yaml"), tolerance=1e-8)
    res = ns.solve(os.path.join(EX, "ch3_two_player.yaml"), tol=1e-8, verbose=False)
    assert isinstance(res, ns.BaseResult) and res.check() is res
