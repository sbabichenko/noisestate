import numpy as np
from noisestate.grid import AgeGrid

def test_mass_and_interp():
    g = AgeGrid([0, 0.5, 1.0, 2.0, 3.0], 14)
    f = np.exp(-0.7 * g.nodes)
    assert abs(g.mass @ f - (1 - np.exp(-2.1)) / 0.7) < 1e-12
    pts = np.array([0.1, 0.5, 1.3, 2.9])
    assert np.allclose(g.evaluate(f, pts), np.exp(-0.7 * pts), atol=1e-12)

def test_shift_exact_on_aligned_panels():
    g = AgeGrid(AgeGrid.breakpoints_from_delays(4.0, [0.5], unit=0.5, unit_range=4.0), 10)
    f = np.where(g.nodes < 1.0, 1.0, np.exp(-(g.nodes - 1.0)))   # jump-free but kinked at 1
    S = g.shift(0.5)
    target = np.where(g.nodes < 0.5, 0.0, np.where(g.nodes - 0.5 < 1.0, 1.0, np.exp(-(g.nodes - 1.5))))
    assert np.allclose(S @ f, target, atol=1e-13)
    Lm = g.shift(-0.5)   # lead
    target2 = np.where(g.nodes + 0.5 > 4.0 + 1e-14, 0.0, np.where(g.nodes + 0.5 < 1.0, 1.0, np.exp(-(g.nodes - 0.5))))
    assert np.allclose(Lm @ f, target2, atol=1e-13)

def test_propagator_ou_and_jump():
    g = AgeGrid([0, 1.0, 2.0, 3.0], 16)
    A = np.array([[-0.8]])
    P0, P = g.propagator(A)
    # x' = -0.8 x + u, u = cos(a), x0 = 1
    u = np.cos(g.nodes)
    x = P0 @ np.array([1.0]) + P @ u
    # closed form: x = e^{-0.8a} + int_0^a e^{-0.8(a-s)} cos s ds
    a = g.nodes
    exact = np.exp(-0.8 * a) + (0.8 * np.cos(a) + np.sin(a) - 0.8 * np.exp(-0.8 * a)) / (1 + 0.64)
    assert np.allclose(x, exact, atol=1e-11)
    J = g.jump_injector(A, 1.0)
    xj = (J @ np.array([2.0]))
    assert np.allclose(xj, np.where(g.nodes >= 1.0, 2.0 * np.exp(-0.8 * (a - 1.0)), 0.0) * (np.arange(g.N) >= g.n), atol=1e-11)
    # 2x2 coupled
    A2 = np.array([[0.0, 1.0], [-1.0, -0.2]])
    P0, P = g.propagator(A2)
    x = (P0 @ np.array([1.0, 0.0])).reshape(g.N, 2)
    from scipy.linalg import expm
    exact = np.array([expm(A2 * t) @ np.array([1.0, 0.0]) for t in a])
    assert np.allclose(x, exact, atol=1e-10)

def test_convolution_and_correlation():
    g = AgeGrid([0, 0.5, 1.0, 2.0, 3.0], 16)
    a = g.nodes
    y = np.exp(-1.3 * a); h = np.exp(-0.4 * a)
    # int_0^a h(b) y(a-b) db = (e^{-0.4a} - e^{-1.3a}) / 0.9
    conv = g.conv_op(y) @ h
    assert np.allclose(conv, (np.exp(-0.4 * a) - np.exp(-1.3 * a)) / 0.9, atol=1e-11)
    assert np.allclose(g.conv_op_left(h) @ y, conv, atol=1e-11)
    # kinked factor: y jumps at 1
    yk = np.zeros(g.N); yk[:2 * g.n] = 1.0     # 1 on [0,1) (two panels), 0 after; left limit at 1 is 1
    conv = g.conv_op(yk) @ h        # = int_{max(0,a-1)}^a h(b) db
    exact = (np.exp(-0.4 * np.maximum(0, a - 1)) - np.exp(-0.4 * a)) / 0.4
    assert np.allclose(conv, exact, atol=1e-11)
    # correlation with discount: int_0^{L-a} e^{-rho s} w(s) z(a+s) ds, w = e^{-0.4 s}, z = e^{-1.3 a}
    rho = 0.5
    corr = g.corr_op(h, rho) @ y
    k = rho + 0.4 + 1.3
    exact = np.exp(-1.3 * a) * (1 - np.exp(-k * (3.0 - a))) / k
    assert np.allclose(corr, exact, atol=1e-11)
    assert np.allclose(g.corr_op_right(y, rho) @ h, corr, atol=1e-11)

def test_breakpoints_from_delays():
    bp = AgeGrid.breakpoints_from_delays(24.0, [0.5])
    assert bp[:3] == [0.0, 0.5, 1.0] and bp[-1] == 24.0 and abs(bp[8] - 4.0) < 1e-12
