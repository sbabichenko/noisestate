"""Information quadrature against a continuous-time Kalman filtering formula."""
import numpy as np
import pytest

import noisestate as ns
from noisestate.stationary_risk import StationaryTilt


@pytest.mark.parametrize("length", [1., 4.])
def test_signal_information_covariance_converges_quadratically(length):
    """For dX=dW, dY=sqrt(3)X dt+dV, P'=1-3P^2 and P(0)=0.

    A unit loading on each past W increment is X at the present. Projecting
    those increments off the observed Y increments must approach the exact
    posterior variance tanh(sqrt(3)*L)/sqrt(3). Full drift weight at age zero
    introduced an O(h) bias (about h/2 on the longer interval).
    """
    model = ns.Model.from_dict(dict(
        shocks=["w0", "w1"], states={"X": "D dt + dw0"},
        agents={"p": dict(controls="D", observes="sqrt(3) X dt + dw1",
                           loss="X^2 + .1 D^2", risk_aversion=.5)},
        horizon=dict(window=length, discount=.5), numerics=dict(nodes=16),
    ))
    solver = ns.engines.solver(model)
    maps = {name: np.zeros(shape) for name, shape in solver.shapes.items()}
    tilt = StationaryTilt(solver, model.agents[0], maps, .5)
    exact = np.tanh(np.sqrt(3)*length)/np.sqrt(3)
    errors=[]
    for h in (.2, .1, .05):
        lattice=tilt._lattice(h)
        loading=np.zeros((2,lattice["Nu"]))
        loading[0,:lattice["Na"]]=1.
        unseen=tilt._sigma(lattice,loading)
        variance=h*np.sum(loading*unseen)
        errors.append(abs(variance-exact))
        # The graph shortcut and full QR must describe the same information.
        q=tilt._info_basis(lattice,full=False)
        past=loading[:,:lattice["Na"]+1].ravel()
        qr_unseen=past-q@(q.T@past)
        np.testing.assert_allclose(unseen[:,:lattice["Na"]+1].ravel(),qr_unseen,rtol=0,atol=2e-12)
    assert errors[1] < .28*errors[0]
    assert errors[2] < .28*errors[1]
    assert errors[2] < (8e-5 if length==1 else 1e-8)
