"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information."""
from .spec import Model, ModelBuilder
from .stationary import StationarySolver, Result
from .finite import FiniteSolver, FiniteResult
from .finite_spectral import SpectralFiniteSolver, SpectralResult

__all__ = ["Model", "ModelBuilder", "StationarySolver", "Result", "FiniteSolver", "FiniteResult",
           "SpectralFiniteSolver", "SpectralResult", "load", "solve"]

__version__ = "0.1.0"


def load(path: str) -> Model:
    import yaml
    with open(path) as fh:
        return Model.from_dict(yaml.safe_load(fh))


def solve(model, **kw) -> Result:
    if isinstance(model, str):
        model = load(model)
    verbose = kw.pop("verbose", False)
    if model.horizon.kind == "stationary":
        return StationarySolver(model, verbose=verbose).solve(**kw)
    if model.horizon.kind == "finite_cells":
        return FiniteSolver(model, verbose=verbose).solve(**kw)
    return SpectralFiniteSolver(model, verbose=verbose).solve(**kw)
