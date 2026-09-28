"""Regression tests for the 2026-09-28 robustness pass: inputs a new user gets wrong, and the answers that were silently
wrong or misleading for them."""
import copy, math
import numpy as np, pytest
import noisestate as ns
from noisestate import Settings, Numerics


def lqg(**over):
    """One agent regulating a state it sees through noise: dX = D dt + sigma dW0, dy = sqrt(p) X dt + dW1, loss X^2 + r D^2
    (the Kalman filter and the LQ regulator separate, so its cost has a closed form, lqg_cost)."""
    d = {"name": "lqg", "params": {"p": 3.0, "r": 0.1, "sigma": 1.0}, "shocks": ["w0", "w1"],
         "states": {"X": "D dt + sigma dw0"},
         "agents": {"a": {"controls": "D", "observes": {"y": "sqrt(p) X dt + dw1"}, "loss": "X^2 + r D^2"}},
         "horizon": {"T": 1.0}, "numerics": {"nodes": 12}}
    for k, v in over.items():
        d[k] = v
    return d


def lqg_cost(p, r, s, T):
    """J = int_0^T Sigma + P p Sigma^2 dt: Sigma(t) = (s / sqrt p) tanh(s sqrt(p) t) the filter's error variance from a known
    start, P(t) = sqrt(r) tanh((T - t) / sqrt r) the regulator's Riccati solution, p Sigma^2 the innovation intensity."""
    from scipy.integrate import quad
    sig = lambda t: s / math.sqrt(p) * math.tanh(s * math.sqrt(p) * t)
    ric = lambda t: math.sqrt(r) * math.tanh((T - t) / math.sqrt(r))
    return quad(lambda t: sig(t) + ric(t) * p * sig(t) ** 2, 0, T, epsabs=0, epsrel=1e-13)[0]


def test_the_single_agent_lqg_matches_its_closed_form():
    """The finite engine against the separation theorem's cost, and the stationary engine against its algebraic limit
    sigma / sqrt(p) + sqrt(r) sigma^2 (the Kalman and the Riccati stationary solutions)."""
    assert abs(ns.solve(ns.Model.from_dict(lqg())).costs["a"] - lqg_cost(3.0, 0.1, 1.0, 1.0)) < 1e-9
    st = ns.solve(ns.Model.from_dict(lqg(horizon={"window": 8.0}, numerics={"nodes": 32})))
    assert abs(st.costs["a"] - (1 / math.sqrt(3.0) + math.sqrt(0.1))) < 1e-9


@pytest.mark.parametrize("where,value,named", [
    (("params", "p"), float("nan"), "parameter p is nan"),
    (("params", "p"), float("inf"), "parameter p is inf"),
    (("horizon", "discount"), float("nan"), "horizon.discount is nan"),
    (("numerics", "unit"), float("nan"), "numerics.unit is nan"),
])
def test_a_non_finite_number_is_named_before_the_solve(where, value, named):
    """A NaN parameter used to solve to a 'converged' result, residual 0, every check passed and a NaN cost; an infinite
    one reached the best response as a 'singular' system; a NaN discount passed the non-negativity check (NaN < 0 is
    False) and failed as singular too."""
    d = lqg()
    d[where[0]][where[1]] = value
    with pytest.raises(ValueError, match=named):
        ns.Model.from_dict(d)
    m = ns.Model.from_dict(lqg())
    if where[0] == "params":
        with pytest.raises(ValueError, match=named):
            m.with_params(**{where[1]: value})


def test_a_non_finite_coefficient_is_named_with_its_field():
    d = ns.Model.from_dict(lqg()).to_dict()                                         # the grammar: loss terms as lists
    d["params"]["big"] = 1e200
    d["agents"]["a"]["loss"] = [[1.0, "X", "X"], ["big * big", "D", "D"]]          # inf, with no exception on the way
    with pytest.raises(ValueError, match="agent a: the coefficient of loss term"):
        ns.Model.from_dict(d)


@pytest.mark.parametrize("expr,why", [("1/0", "division by zero"), ("sqrt(-p)", "outside its domain"),
                                      ("log(0)", "outside its domain"), ("big**2", "out of range"),
                                      ("(-p)**0.5", "not real")])
def test_arithmetic_failures_in_a_coefficient_are_named(expr, why):
    """They reached the caller as a bare ZeroDivisionError, OverflowError or 'math domain error' from inside the
    expression evaluator, or as a complex number float() refused."""
    d = lqg(params={"p": 3.0, "r": 0.1, "sigma": 1.0, "big": 1e200, "q": expr})
    d["agents"]["a"]["loss"] = "X^2 + r D^2 + q X"
    with pytest.raises(ValueError, match=f"cannot evaluate coefficient .*{why}"):
        ns.Model.from_dict(d)


@pytest.mark.parametrize("field,value", [("tol", 0.0), ("tol", -1e-8), ("damping", 0.0), ("damping", float("nan")),
                                         ("max_newton", -1), ("max_newton", 2.5)])
def test_numerics_fixed_point_options_are_checked(field, value):
    """tol 0 or below cannot be met (a negative one reported 'not converged' at the exact answer), damping 0 takes no step."""
    with pytest.raises(ValueError, match=f"numerics.{field}"):
        Numerics(**{field: value})
    with pytest.raises(ValueError, match=f"numerics.{field}"):
        ns.solve(ns.Model.from_dict(lqg()), {field: value})


def test_settings_types_and_the_planning_criterion_are_checked_where_set():
    """A count as a string failed deep in the Anderson loop as a comparison TypeError; a misspelt risk_planning passed
    silently on a risk-neutral model (it was checked only when an agent was risk averse)."""
    with pytest.raises(TypeError, match="settings.anderson_m must be an integer"):
        Settings(anderson_m="15")
    with pytest.raises(ValueError, match="non-negative"):
        Settings(anderson_iters=-1)
    with pytest.raises(TypeError, match="settings.foc_rcond must be a number"):
        Settings(foc_rcond=float("nan"))
    with pytest.raises(ValueError, match="risk_planning"):
        ns.solve(ns.Model.from_dict(lqg()), {"settings": {"risk_planning": "consistant"}})
    assert Settings(anderson_m=np.int64(5)).anderson_m == 5          # a numpy integer is an integer
    assert Settings(second_order_tol=-0.5).second_order_tol == -0.5  # signs are left to the caller (tests move them)


def test_kernel_at_refuses_nan_and_a_date_past_the_horizon():
    """The interpolant gives a point outside its domain a zero row, so at(2.0, 0.5) on [0, 1] and at(nan, 0.5) read as a
    response of exactly 0."""
    k = ns.solve(ns.Model.from_dict(lqg()), diagnostics=False).kernel("X", "w0")
    with pytest.raises(ValueError, match="must not pass 1"):
        k.at(2.0, 0.5)
    with pytest.raises(ValueError, match="NaN"):
        k.at(float("nan"), 0.5)
    assert k.at(0.3, 0.5) == 0.0                     # a later shock: zero by causality, not an error
    assert abs(k.at(1.0, 1.0) - 1.0) < 1e-9


def test_an_initial_shock_may_not_take_a_brownian_shock_s_name():
    """The clash check ran in Past.__init__, before a past of initial shocks only knew the model's channels."""
    d = lqg(horizon={"T": 1.0, "past": [{"name": "w0", "loads": {"X": 1.0}}], "continuation": "end"})
    with pytest.raises(ValueError, match="named like one of the model's shocks"):
        ns.solve(ns.Model.from_dict(d))


def test_a_cost_that_grows_with_the_window_is_reported():
    """A myopic agent leaves a random walk uncontrolled: its stationary cost E X^2 is infinite, and the window's is L.  The
    window check (does a kernel still MOVE at L) passes on a constant kernel; the cost window row says so without
    blocking, since the strategy is right.  Chapter 4's market maker (P^2 - 2 P V) is the same: its cost is about 2.1 - L."""
    d = lqg(horizon={"window": 4.0}, numerics={"nodes": 8})
    d["agents"]["a"]["myopic"] = True
    res = ns.solve(ns.Model.from_dict(d))
    assert abs(res.costs["a"] - 4.0) < 1e-9 and abs(res.cost_tail["a"] - 0.1) < 1e-6
    row = [r for r in res.diagnostics.rows if r["name"] == "cost window"]
    assert row and row[0]["ok"] is False and "COST GROWS WITH THE WINDOW" in res.summary()
    assert res.diagnostics.assess().accepted                   # reported, not required
    ok = ns.solve(ns.Model.from_dict(lqg(horizon={"window": 8.0}, numerics={"nodes": 16})))
    assert ok.cost_tail["a"] < 1e-3 and not [r for r in ok.diagnostics.rows if r["name"] == "cost window"]


def test_an_equations_file_takes_expression_parameters():
    """model_file.md: a parameter may be an expression in earlier ones.  The grammar evaluated them in order, the equations
    form built its Params from the raw strings and refused the file ('the value must be a number')."""
    d = lqg(params={"p": 3.0, "r": "0.2 / 2", "sigma": 1.0, "s2": "2 * sigma"})
    d["agents"]["a"]["loss"] = "X^2 + r D^2 + s2 X"
    m = ns.Model.from_dict(d)
    assert m.params["r"] == 0.1 and m.params["s2"] == 2.0 and m.to_dict()["params"]["s2"] == "2 * sigma"
    assert m.with_params(sigma=3.0).params["s2"] == 6.0          # the dependent parameter follows
    d["params"] = {"p": 3.0, "r": "q", "q": 0.1, "sigma": 1.0, "s2": 1.0}
    with pytest.raises(ValueError, match="defined after it"):
        ns.Model.from_dict(d)


@pytest.mark.parametrize("ties", [[["player1", "player2"], ["player2", "player1"]], [["player1", "player1"]]])
def test_an_agent_belongs_to_one_tie_group_once(ties):
    """The first agent of a group is its representative and the rest copy it: an agent in two groups, or twice in one,
    was accepted and solved by the luck of the order."""
    d = ns.load(ns.example("ch1_two_player_finite")).to_dict()
    d["ties"] = ties
    with pytest.raises(ValueError, match="one tie group"):
        ns.Model.from_dict(d)


def test_a_model_warning_is_shown_once_per_model():
    """validate() runs in from_dict and twice in the compile, each warning from its own line, so Python's once-per-location
    filter showed a negative control penalty's warning three times per solve (five on the stationary engine)."""
    import warnings
    d = lqg(horizon={"window": 4.0}, numerics={"nodes": 8})
    d["params"]["r"] = -0.1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = ns.Model.from_dict(d)
        ns.solve(m, diagnostics=False)
        ns.solve(m, diagnostics=False)
    assert len([x for x in w if "unbounded below in D" in str(x.message)]) == 1


def test_the_krylov_path_solves_a_quote_with_no_square_of_its_own():
    """Chapter 6's finite market: the market maker's loss has no P^2, its curvature is the orders its quote draws at once
    (c' Q c over the spike with the instant reactions).  From 16 nodes the system passes foc_dense_max and goes to GMRES,
    whose time-row preconditioner took the own square alone: a zero block at t = 0, refused as 'singular' although the
    dense system at the same nodes is regular.  The two paths now agree."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_monitoring import finite_market
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        krylov = ns.solve(finite_market(0.5, True, nodes=16), diagnostics=False)
        dense = ns.solve(finite_market(0.5, True, nodes=16), {"settings": {"foc_dense_max": 10 ** 6}}, diagnostics=False)
    assert krylov.converged and dense.converged
    assert all(abs(krylov.costs[a] - dense.costs[a]) < 1e-10 for a in ("mm", "trader"))


def test_a_key_given_twice_in_a_model_file_is_refused(tmp_path):
    """YAML's loaders keep the last of a repeated key silently: a second horizon block, or two agents of one name, replaced
    the first and the file solved as a different model than it reads.  A merge key's override stays legal."""
    import io
    from noisestate.spec import yaml_load
    p = tmp_path / "dup.yaml"
    p.write_text('name: x\nparams: {p: 3, r: 0.1}\nshocks: [w0, w1]\nstates:\n  X: "D dt + dw0"\nagents:\n'
                 '  a: {controls: D, observes: "sqrt(p) X dt + dw1", loss: "X^2 + r D^2"}\nhorizon: {T: 1}\nhorizon: {T: 2}\n')
    with pytest.raises(ValueError, match="line 9: the key 'horizon' is given twice"):
        ns.load(str(p))
    assert yaml_load(io.StringIO("b: &b {x: 1, y: 2}\nd: {<<: *b, y: 3}\n"))["d"] == {"x": 1, "y": 3}


def test_the_lanczos_second_order_check_agrees_with_the_dense_one():
    """Above second_order_dense the check runs Lanczos.  Its lowest eigenvalue was sought with ARPACK's tolerance relative
    to itself, near 0 on a nearly singular form, and ran its whole budget without an answer (a T = 30 game on 5 panels: 3042
    products, 'not converged'); it is now hi less the largest of hi I - A, settled at the tolerance of hi."""
    m = ns.Model.from_dict(lqg())
    dense = ns.solve(m).second_order["a"]
    lanczos = ns.solve(m, {"settings": {"second_order_dense": 10}}).second_order["a"]
    assert lanczos["converged"] and lanczos["ok"] == dense["ok"]
    assert abs(lanczos["min"] - dense["min"]) < 1e-5 and abs(lanczos["max"] - dense["max"]) < 1e-12


def test_a_sweep_point_that_raises_names_its_value_and_keeps_the_points_before_it():
    """sweep() stopped at the first point whose solve raised (here a control penalty of 0: a singular best response) and
    the points already solved were lost with it; the value that failed was left to be guessed."""
    import warnings
    m = ns.Model.from_dict(lqg(numerics={"nodes": 8}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="singular") as info:
            m.sweep(r=[0.2, 0.1, 0.0, 0.05])
    part = info.value.partial
    assert [p.value for p in part] == [0.2, 0.1] and all(p.converged for p in part)
    assert any("r = 0.0, point 3 of 4" in n for n in getattr(info.value, "__notes__", []))


def test_evaluate_and_mean_refuse_a_date_past_the_horizon():
    """res.evaluate and res.mean read the same interpolant as Kernel.at, which gives a point past T a zero row."""
    d = lqg()
    d["agents"]["a"]["loss"] = "(X - 1)^2 + r D^2"
    res = ns.solve(ns.Model.from_dict(d), diagnostics=False)
    with pytest.raises(ValueError, match="must not pass 1"):
        res.evaluate("X", "w0", 1.5, 0.2)
    with pytest.raises(ValueError, match="must not pass 1"):
        res.mean("X", [0.5, 2.0])
    assert np.isfinite(res.evaluate("X", "w0", [0.5, 1.0], [0.2, 0.2])).all() and res.mean("X", [0.5, 1.0]).shape == (2,)


def test_a_stationary_kernel_is_not_read_past_its_window():
    k = ns.solve(ns.Model.from_dict(lqg(horizon={"window": 4.0}, numerics={"nodes": 8})), diagnostics=False).kernel("X", "w0")
    with pytest.raises(ValueError, match=r"window \[0, 4\]"):
        k.at(5.0)
    assert k.at(-0.5) == 0.0 and np.isfinite(k.at([0.0, 2.0, 4.0])).all()
