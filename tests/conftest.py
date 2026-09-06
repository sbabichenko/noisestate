"""The examples directory (its model-building scripts) is importable from every test."""
import os, sys
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
if EX not in sys.path:
    sys.path.insert(0, EX)

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
