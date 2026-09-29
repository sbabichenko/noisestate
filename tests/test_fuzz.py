"""Randomized differential testing (extras/fuzz): seeded random models across the feature families, each run through
every oracle that applies -- the independent discrete brute-force reference, the one-agent closed form, cross-engine
agreement, invariances, theta -> 0, self-consistency -- under one rule: a result is correct within its stated accuracy
or says so (not converged, a failed required check, a warning, a refusal).  A FINDING is a confident wrong answer, a
refusal or crash of a valid model, or an invalid model accepted.

The fast test runs a few fixed seeds per family at depth "smoke"; the slow one (NOISESTATE_SLOW=1) runs the campaign,
NOISESTATE_FUZZ_COUNT models (default 40) from NOISESTATE_FUZZ_SEED (default 1000), writing repros under
NOISESTATE_FUZZ_OUT when set.  `python extras/fuzz/campaign.py --help` runs campaigns outside pytest.
"""
import os
import sys

import pytest

from helpers import slow

HERE = os.path.dirname(os.path.abspath(__file__))
FUZZ = os.path.join(HERE, "..", "extras", "fuzz")
if FUZZ not in sys.path:
    sys.path.insert(0, FUZZ)

import generate as G      # noqa: E402
import oracles as O       # noqa: E402

# (family, seed): fast ones, chosen so every family appears once and the whole set runs in well under a minute
SMOKE = [("lqg1_finite", 1), ("lqg1_stationary", 2), ("game_finite", 1), ("game_stationary", 1), ("delay_finite", 2),
         ("delay_stationary", 3), ("means_finite", 3), ("ties", 2), ("myopic", 1), ("prior", 0), ("cara_finite", 0),
         ("cara_finite", 3), ("monitor", 0), ("monitor", 3), ("transition", 1), ("ch6_market", 0), ("ch6_market", 1)] \
    + [("invalid", s) for s in range(12)]


# the oracle each smoke case must actually have run and passed (a smoke case that silently skips its reference
# would test nothing)
EXPECT = {"lqg1_finite": ("closed form", "discrete reference"), "lqg1_stationary": ("closed form", "discrete reference"),
          "game_finite": ("discrete reference",), "game_stationary": ("discrete reference",),
          "delay_finite": ("discrete reference",), "delay_stationary": ("discrete reference",),
          "means_finite": ("discrete reference",), "myopic": ("discrete reference",), "prior": ("discrete reference",),
          "ties": ("tied vs untied", "discrete reference"), "transition": ("transition identity",),
          "invalid": ("invalid model refused",)}


def _findings(rec):
    return [f"{c['check']}: {c['detail']}" for c in rec["checks"] if c["verdict"] == "FINDING"]


@pytest.mark.parametrize("family,seed", SMOKE, ids=[f"{f}-{s}" for f, s in SMOKE])
def test_fuzz_smoke(family, seed):
    case = G.generate(seed, family)
    rec = O.run_case(case, depth="smoke")
    assert not _findings(rec), "\n".join(_findings(rec)) + "\n--- repro ---\n" + case.yaml()
    verdicts = {c["check"]: c["verdict"] for c in rec["checks"]}
    for check in EXPECT.get(family, ()):
        assert verdicts.get(check) == "ok", f"{check}: {verdicts.get(check)} ({[c for c in rec['checks'] if c['check'] == check]})"


def test_generator_is_reproducible_and_valid_yaml():
    import yaml
    for fam in G.FAMILIES:
        a, b = G.generate(5, fam), G.generate(5, fam)
        assert a.model == b.model
        assert yaml.safe_load(a.yaml()) == a.model


@slow("the randomized campaign; NOISESTATE_FUZZ_COUNT models from NOISESTATE_FUZZ_SEED")
def test_fuzz_campaign():
    import campaign
    count = int(os.environ.get("NOISESTATE_FUZZ_COUNT", "40"))
    start = int(os.environ.get("NOISESTATE_FUZZ_SEED", "1000"))
    out = os.environ.get("NOISESTATE_FUZZ_OUT")
    bad = []
    for seed in range(start, start + count):
        fam = campaign.family_of(seed, G.FAMILIES)
        rec = campaign.run_one(seed, fam, "full", timeout=900)
        f = _findings(rec)
        if f:
            bad.append(f"{rec['id']}: " + "; ".join(f))
            if out:
                campaign.write_repro(out, rec)
    assert not bad, "\n".join(bad)
