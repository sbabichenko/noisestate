import noisestate as ns
import os
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")

def test_ch5_example_matches_its_generator():
    from make_ch5_cycle_market import build
    generated = build().to_dict()
    committed = ns.load(os.path.join(EX, "ch5_cycle_market.yaml")).to_dict()
    assert generated == committed, "examples/ch5_cycle_market.yaml is stale: rerun examples/make_ch5_cycle_market.py"

def test_every_example_validates():
    for f in sorted(os.listdir(EX)):
        if f.endswith(".yaml"):
            m = ns.load(os.path.join(EX, f)); m.validate()
