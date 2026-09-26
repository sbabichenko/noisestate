"""The examples directory (its model-building scripts) and extras/ (compare_baseline) are importable from every
test; the `slow` marker (tests/helpers.slow, the NOISESTATE_SLOW gate) is registered so `-m slow` selects them;
the BLAS thread count is capped before numpy loads."""
import os, sys

# The solves here are small (a few hundred to a few thousand unknowns), the size at which a widely threaded
# GEMM spends longer synchronising than computing: on a 16-core machine the fast suite takes 743 s at the
# BLAS default and 16.9 s against 75.9 s on its heaviest file alone at four threads.  Four is also where
# tests/SLOW.md's seconds were measured.  This must run before numpy imports its BLAS, so it stays at the
# top of the first conftest pytest loads; an explicit setting in the environment wins.
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "4")

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("examples", "extras"):
    p = os.path.join(HERE, "..", sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs under NOISESTATE_SLOW=1 only (tests/SLOW.md)")


if os.environ.get("NOISESTATE_FOC_FREE") or os.environ.get("NOISESTATE_SIZE_LOG"):
    # NOISESTATE_FOC_FREE=1: the whole suite on the spectral finite engine's matrix-free best response
    # (finite_free), whatever the size of the system; NOISESTATE_SIZE_LOG=path: append every spectral finite
    # engine's largest nU nR N, its N and the model's name to that file (what foc_dense_max is set against)
    import functools
    import noisestate.finite_spectral as _fs
    _init = _fs.SpectralFiniteSolver.__init__

    @functools.wraps(_init)
    def _hooked(self, *a, **k):
        _init(self, *a, **k)
        if os.environ.get("NOISESTATE_FOC_FREE"):
            self.foc_free = True
        if os.environ.get("NOISESTATE_SIZE_LOG"):
            with open(os.environ["NOISESTATE_SIZE_LOG"], "a") as f:
                f.write(f"{max(len(x.controls) * len(x.signals) for x in self.model.agents) * self.Nm} {self.c.N} {self.model.name}\n")
    _fs.SpectralFiniteSolver.__init__ = _hooked
