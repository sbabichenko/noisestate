"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information."""
from .spec import Model, ModelBuilder
from .stationary import StationarySolver, Result
from .finite import FiniteSolver, FiniteResult
from .finite_spectral import SpectralFiniteSolver, SpectralResult

__all__ = ["Model", "ModelBuilder", "StationarySolver", "Result", "FiniteSolver", "FiniteResult",
           "SpectralFiniteSolver", "SpectralResult", "load", "solve", "read_yaml", "read_json"]

__version__ = "0.2.0"


def read_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def read_json(path: str):
    import json
    with open(path) as fh:
        return json.load(fh)


def load(path: str) -> Model:
    return Model.from_dict(read_yaml(path))


def solve(model, **kw) -> Result:
    if isinstance(model, str):
        model = load(model)
    verbose = kw.pop("verbose", False)
    if model.horizon.kind == "stationary":
        return StationarySolver(model, verbose=verbose).solve(**kw)
    if model.horizon.kind == "finite_cells":
        return FiniteSolver(model, verbose=verbose).solve(**kw)
    return SpectralFiniteSolver(model, verbose=verbose).solve(**kw)
