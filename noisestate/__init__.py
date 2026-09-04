"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information."""
from .spec import Model, ModelBuilder
from .stationary import StationarySolver, Result

__version__ = "0.1.0"


def load(path: str) -> Model:
    import yaml
    with open(path) as fh:
        return Model.from_dict(yaml.safe_load(fh))


def solve(model, **kw) -> Result:
    if isinstance(model, str):
        model = load(model)
    if model.horizon.kind == "stationary":
        return StationarySolver(model, verbose=kw.pop("verbose", False)).solve(**kw)
    raise NotImplementedError("finite-horizon solver not yet available")
