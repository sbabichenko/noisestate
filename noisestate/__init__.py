"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information.

    import noisestate as ns
    res = ns.solve("examples/ch4_kyle_back.yaml")                    # the model file's own numerics
    res = ns.solve(model, ns.Numerics(nodes=32, tol=1e-12))          # a change of resolution, not of model
    res.summary(); res.kernel("P", "w"); res.costs; res.to_dict()

The model (Model, ModelBuilder, a dict or a path) is the problem; a Numerics (or a dict of its fields) is how it
is solved: the engine, the grid, the tolerances, the settings.  solve() lays the given numerics over the
model's own, builds the engine (noisestate.engines) and returns one Result.
"""
from .spec import Model, ModelBuilder
from .numerics import Numerics
from .accel import ConvergenceError
from .settings import Settings
from .results import BaseResult, StationaryResult, TriangleResult, TransitionResult, CellResult
from .stationary import StationarySolver
from .finite import FiniteSolver
from .finite_spectral import SpectralFiniteSolver
from . import engines
from .engines import ENGINES
from .sweep import sweep, make_solver
from .grid_cache import clear as clear_grid_cache
from .transition import transition

__all__ = ["Model", "ModelBuilder", "Numerics", "ConvergenceError", "Settings", "BaseResult", "StationaryResult", "TriangleResult",
           "TransitionResult", "CellResult", "StationarySolver", "FiniteSolver", "SpectralFiniteSolver", "engines", "load", "solve",
           "sweep", "transition", "read_yaml", "read_json", "make_solver", "ENGINES", "clear_grid_cache"]

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


def as_model(model) -> Model:
    """A Model from what solve() accepts: a Model, a ModelBuilder, a dict or a path to a YAML file."""
    if isinstance(model, str):
        return load(model)
    if isinstance(model, dict):
        return Model.from_dict(model)
    if isinstance(model, ModelBuilder):
        return model.build()
    if isinstance(model, Model):
        return model
    raise TypeError(f"expected a Model, a ModelBuilder, a dict or a path, not {type(model).__name__}")


_ALIASES = {"nodes": "numerics.nodes", "settings": "numerics.settings"}      # accepted until 0.6 (CHANGELOG)


def solve(model, numerics=None, *, init=None, start: str = "zero", tol=None, max_evaluations=None, deadline=None,
          progress=None, diagnostics: bool = True, refine: bool = False, stability: bool = False, verbose: bool = False,
          naive_observers=None, past=None, continuation=None, **deprecated) -> BaseResult:
    """Solve a model (a Model, a ModelBuilder, a dict, or a path to a YAML file) under `numerics` (a Numerics or
    a dict of its fields, laid over the model's own: engine, nodes, unit, unit_range, breakpoints,
    continuation_nodes, tol, damping, max_newton, variable, settings).  The other options are the solve's:
    init (action kernels or raw maps per agent to start from), start ("zero", "coarse", or "stationary" on a
    transition with a continuation), tol (over the numerics'), max_evaluations and deadline (the bounds; past
    either the best iterate is returned not converged), progress (a callable on {"evaluation", "residual",
    "phase", "seconds"} after every evaluation), diagnostics (False skips the checks at the end), refine (re-solve
    on a finer grid and report the change, res.refinement), stability (add res.stability()), verbose;
    naive_observers ({agent: [observers]}, the stationary engine); past and continuation (a transition's, on the
    spectral engine; on a model of kind "transition" each overrides the file's block).  Unknown options are a
    TypeError naming the Numerics field they belong to.  res.numerics is the resolved Numerics."""
    model = as_model(model)
    for k in list(deprecated):
        if k in _ALIASES:
            numerics = Numerics.of(numerics).merged(Numerics.of({k: deprecated.pop(k)}))
    if deprecated:
        bad = sorted(deprecated)
        fields = [k for k in bad if k in Numerics.field_names()]
        raise TypeError(f"unknown option(s) {bad} for solve()" + (f"; {fields} are fields of Numerics: solve(model, Numerics({fields[0]}=...))"
                                                                  if fields else "; see help(noisestate.solve)"))
    S, num = engines.build(model, numerics, verbose=verbose, naive_observers=naive_observers, past=past, continuation=continuation)
    kw = num.solve_kw()
    if tol is not None:
        kw["tol"] = tol
    res = S.solve(init=init, start=start, max_evaluations=max_evaluations, deadline=deadline, progress=progress,
                  diagnostics=diagnostics, **kw)
    if refine:
        res.refine()
    if stability:
        res.stability()
    return res
