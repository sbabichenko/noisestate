import os, time
import noisestate as ns
from noisestate.stationary import StationarySolver
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_compiles_share_grids_and_their_caches():
    d = ns.load(os.path.join(EX, "ch4_kyle_back.yaml")).to_dict()
    a = StationarySolver(ns.Model.from_dict(d)); d["params"]["eps"] = 0.1; b = StationarySolver(ns.Model.from_dict(d))
    assert a.c.grid is b.c.grid
    df = ns.load(os.path.join(EX, "ch1_delayed_finite.yaml")).to_dict(); df.setdefault("numerics", {})["nodes"] = 5
    s1 = SpectralFiniteSolver(ns.Model.from_dict(df)); s1.solve()
    t0 = time.time(); df["params"]["p1"] = 4.0; s2 = SpectralFiniteSolver(ns.Model.from_dict(df)); dt = time.time() - t0
    assert s2.c.g is s1.c.g and s1.c.g.paths          # quadrature paths carried over
    assert dt < 30.0                                                  # a rebuild of the paths takes minutes
