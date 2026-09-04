import os, time
import noisestate as ns
from noisestate.stationary import StationarySolver
from noisestate.finite_spectral import SpectralFiniteSolver
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")

def test_compiles_share_grids_and_their_caches():
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml"))
    a = StationarySolver(ns.Model.from_dict(d)); d["params"]["eps"] = 0.1; b = StationarySolver(ns.Model.from_dict(d))
    assert a.c.grid is b.c.grid
    df = ns.read_yaml(os.path.join(EX, "ch1_delayed_finite.yaml")); df["horizon"]["nodes"] = 5
    s1 = SpectralFiniteSolver(ns.Model.from_dict(df)); s1.solve()
    t0 = time.time(); df["params"]["p1"] = 4.0; s2 = SpectralFiniteSolver(ns.Model.from_dict(df)); dt = time.time() - t0
    assert s2.c.g is s1.c.g and "_paths" in s1.c.g.__dict__          # quadrature paths carried over
    assert dt < 2.0
