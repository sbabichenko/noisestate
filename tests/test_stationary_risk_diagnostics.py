"""Risk diagnostics must cover every control of an agent."""
import numpy as np
import noisestate as ns


def test_an_unused_last_control_does_not_erase_the_risk_lattice_gap():
    results=[]
    for controls in (["D1"],["D1","D2"],["D2","D1"]):
        model=ns.Model.from_dict(dict(
            shocks=["w0"],states={"X":"D1 dt + dw0"},
            agents={"p":dict(controls=controls,observes="dw0",
                loss="X^2 + .1 D1^2"+(" + D2^2" if len(controls)==2 else ""),risk_aversion=.5)},
            horizon=dict(window=4.,discount=.5),numerics=dict(nodes=24),
        ))
        result=ns.solve(model,diagnostics=False,tol=1e-10)
        assert result.converged
        results.append(result)
    reference=results[0].risk["p"]["richardson_gap"]
    assert reference>1e-6
    for result in results[1:]:
        assert abs(result.risk["p"]["richardson_gap"]-reference)<1e-12
        assert abs(result.costs["p"]-results[0].costs["p"])<1e-12
        np.testing.assert_allclose(result.kernel("D1").values,results[0].kernel("D1").values,rtol=0,atol=1e-11)
