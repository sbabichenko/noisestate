"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information."""
import inspect

from .spec import Model, ModelBuilder
from .accel import ConvergenceError
from .settings import Settings
from .results import BaseResult, StationaryResult, TriangleResult, TransitionResult, CellResult
from .stationary import StationarySolver
from .finite import FiniteSolver
from .finite_spectral import SpectralFiniteSolver
from .sweep import sweep, make_solver, ENGINES
from .grid_cache import clear as clear_grid_cache
from .transition import transition

__all__ = ["Model", "ModelBuilder", "ConvergenceError", "Settings", "BaseResult", "StationaryResult", "TriangleResult", "TransitionResult",
           "CellResult", "StationarySolver", "FiniteSolver", "SpectralFiniteSolver", "load", "solve", "sweep",
           "transition", "read_yaml", "read_json", "make_solver", "ENGINES", "clear_grid_cache"]

def _read_version() -> str:
    """The version pyproject.toml declares when the package is imported from a source tree (a checkout on
    the path, an editable install bumped since it was installed), else the installed distribution's."""
    import os, re
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")) as fh:
            text = fh.read()
        if re.search(r'^name\s*=\s*"noisestate"', text, re.M):
            return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    except (OSError, AttributeError):
        pass
    try:
        from importlib.metadata import version
        return version("noisestate")
    except Exception:                                   # not installed as a distribution
        return "unknown"


__version__ = _read_version()


def read_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def read_json(path: str):
    import json
    with open(path) as fh:
        return json.load(fh)


def load(path: str) -> Model:
    """The model of a YAML file; a relative path in horizon.past.model is taken from the file's directory."""
    import os
    return Model.from_dict(read_yaml(path), base_dir=os.path.dirname(os.path.abspath(path)))


def solve(model, refine: bool = False, stability: bool = False, **kw) -> BaseResult:
    """Solve a model (a Model, a dict, or a path to a YAML file) with the engine its horizon selects.
    Keyword arguments go to the engine's constructor (e.g. verbose, naive_observers, settings; on the
    spectral finite engine past=, the known past of a transition, and continuation=, how it goes on
    after T: "end", "stationary" or a stationary result of the model; on a model of kind "transition"
    each overrides the file's block) or to its solve() (e.g. tol, init, start, variable); unknown ones
    are an error.  The constructor options are recorded in
    res.solver_kw, so refine() and stability() rebuild the same engine.  refine=True re-solves on a
    finer grid and reports the change (res.refinement); stability=True adds res.stability()."""
    if isinstance(model, str):
        model = load(model)
    elif isinstance(model, dict):
        model = Model.from_dict(model)
    elif isinstance(model, ModelBuilder):
        model = model.build()
    elif not isinstance(model, Model):
        raise TypeError(f"solve() takes a Model, a ModelBuilder, a dict or a path, not {type(model).__name__}")
    engine = ENGINES[model.horizon.kind]
    init_params = inspect.signature(engine.__init__).parameters
    solve_params = inspect.signature(engine.solve).parameters
    ctor_kw = {k: v for k, v in kw.items() if k in init_params}
    solve_kw = {k: v for k, v in kw.items() if k in solve_params and k not in init_params}
    unknown = sorted(set(kw) - set(ctor_kw) - set(solve_kw))
    if unknown:
        valid = sorted((set(init_params) | set(solve_params)) - {"self", "model", "init"})
        raise TypeError(f"unknown option(s) {unknown} for the {model.horizon.kind!r} engine; valid: {valid}")
    res = engine(model, **ctor_kw).solve(**solve_kw)
    if refine:
        res.refine(**{k: v for k, v in solve_kw.items() if k != "init"})
    if stability:
        res.stability()
    return res
