"""A singular best-response system is refused on every engine, and validation warns on its usual cause.
Before, both finite engines fell through to a least-squares solve and reported a zero strategy as a
converged equilibrium (cost 0, second_order ok, check() clean); the stationary engine raised."""
import glob, os, warnings, pytest
import noisestate as ns
HERE = os.path.dirname(os.path.abspath(__file__)); EX = os.path.join(HERE, "..", "examples")


def _kyle_back(kind, nodes, mm_loss):
    """Kyle-Back with the market maker's loss replaced; on the finite horizons window 2.0 and no discount."""
    d = ns.read_yaml(os.path.join(EX, "ch4_kyle_back.yaml"))
    d["agents"]["market_maker"]["loss"] = mm_loss
    if kind != "stationary":
        cells = kind == "finite_cells"      # the cell engine's result kind; the model asks for finite + engine cells
        d["horizon"] = {"kind": "finite" if cells else kind, "window": 2.0}
        d["numerics"] = {"nodes": nodes, **({"engine": "cells"} if cells else {})}; del d["params"]["rho"]
    return d


@pytest.mark.parametrize("kind,nodes", [("finite", 6), ("finite_cells", 6), ("finite_cells", 24), ("stationary", 24)])
def test_a_singular_best_response_system_raises_on_every_engine(kind, nodes):
    # without [1, P, P] the market maker's price enters its loss only through the cross term with the exogenous V:
    # its first-order condition does not respond to P at all (cells 6: the dense branch; 24: the Krylov branch)
    with pytest.warns(UserWarning, match="control P has no strictly positive quadratic term"):
        m = ns.Model.from_dict(_kyle_back(kind, nodes, [[-2.0, "P", "V"]]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="the best-response system of market_maker is singular"):
            ns.solve(m)
        res = ns.solve(_kyle_back(kind, nodes, [[1.0, "P", "P"], [-2.0, "P", "V"]])).require_converged()   # the same setup with the term: regular
    assert res.converged and abs(res.costs["market_maker"]) > 0.01       # its cost omits the V^2 term and is negative


def test_a_lagged_only_own_quadratic_is_singular_on_the_finite_engines():
    # [r, D1@0.5, D1@0.5] without [r, D1, D1]: nothing reads the control over the last 0.5 of the horizon
    d = ns.read_yaml(os.path.join(EX, "ch1_two_player_finite.yaml")); d.setdefault("numerics", {})["nodes"] = 6
    d["agents"]["player1"]["loss"] = [[1.0, "X", "X"], ["r1", "D1@0.5", "D1@0.5"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # no validation warning: the lagged read pins a non-myopic control
        m = ns.Model.from_dict(d)
    with pytest.raises(ValueError, match="singular.*lagged read"):
        ns.solve(m)


def test_validation_warns_on_a_control_without_a_positive_own_quadratic_term():
    kb = lambda loss, **kw: dict(_kyle_back("stationary", 24, loss), agents={**_kyle_back("stationary", 24, loss)["agents"]})
    with pytest.warns(UserWarning, match=r"control P has no strictly positive quadratic term .* coefficient of P squared is 0\)"):
        ns.Model.from_dict(kb([[-2.0, "P", "V"]]))
    with pytest.warns(UserWarning, match="unbounded below in P"):
        ns.Model.from_dict(kb([[-1.0, "P", "P"], [-2.0, "P", "V"]]))
    with pytest.warns(UserWarning, match="lagged read P@0.5 does not enter a myopic agent"):
        ns.Model.from_dict(kb([[1.0, "P@0.5", "P@0.5"], [-2.0, "P", "V"]]))
    d = kb([[1.0, "P@0.5", "P@0.5"], [-2.0, "P", "V"]]); d["agents"]["market_maker"]["myopic"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ns.Model.from_dict(d)                                   # a non-myopic agent reads its lagged control
        for path in sorted(glob.glob(os.path.join(EX, "*.yaml"))):
            ns.load(path)                                       # no shipped example warns (the Chapter 5 prices are read with a lag)
