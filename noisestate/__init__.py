"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information.

    import noisestate as ns
    res = ns.solve("examples/ch4_kyle_back.yaml")                    # the model file's own numerics
    res = ns.solve(model, ns.Numerics(nodes=32, tol=1e-12))          # a change of resolution, not of model
    res.summary(); res.kernel("P", "w"); res.costs; res.to_dict()

The model (Model, ModelBuilder, a dict or a path) is the problem; a Numerics (or a dict of its fields) is how it
is solved: the engine, the grid, the tolerances, the settings.  solve() lays the given numerics over the
model's own, builds the engine (noisestate.engines) and returns one Result.
"""
import dataclasses

from .spec import Model, ModelBuilder
from .numerics import Numerics
from .accel import ConvergenceError
from .settings import Settings
from .results import Result, BaseResult
from .stationary import StationarySolver
from .finite import FiniteSolver
from .finite_spectral import SpectralFiniteSolver
from . import engines
from .engines import ENGINES
from .sweep import sweep, make_solver
from .grid_cache import clear as clear_grid_cache
from .transition import transition
from .schema import schema
from .expr import Param, shocks, State, Control, define, Signal, Agent, Stationary, Finite, Transition, SweepPoint, settings
from .expr import sqrt, exp, log, sin, cos, tanh
from .kernel import Kernel

__all__ = ["Model", "ModelBuilder", "Numerics", "ConvergenceError", "Settings", "Result", "BaseResult", "StationarySolver",
           "FiniteSolver", "SpectralFiniteSolver", "engines", "load", "solve", "sweep", "transition", "read_yaml", "read_json",
           "make_solver", "ENGINES", "clear_grid_cache", "schema",
           "Param", "shocks", "State", "Control", "define", "Signal", "Agent", "Stationary", "Finite", "Transition", "SweepPoint",
           "settings", "Kernel", "sqrt", "exp", "log", "sin", "cos", "tanh"]

_DEPRECATED_RESULTS = ("StationaryResult", "TriangleResult", "TransitionResult", "CellResult")      # until 0.6 (CHANGELOG)


def __getattr__(name: str):
    """The engines' result subclasses by name (noisestate.StationaryResult, ...) with a DeprecationWarning: they are
    internal since one Result; isinstance(res, noisestate.Result) holds for every result."""
    if name in _DEPRECATED_RESULTS:
        import warnings
        from . import results
        warnings.warn(f"noisestate.{name} is deprecated: every engine returns noisestate.Result (the name goes in 0.6)",
                      DeprecationWarning, stacklevel=2)
        return getattr(results, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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


def solve(model, numerics=None, *, init=None, start=None, tol=None, max_evaluations=None, deadline=None,
          progress=None, diagnostics: bool = True, refine: bool = False, stability: bool = False, verbose: bool = False,
          naive_observers=None, past=None, continuation=None, **deprecated) -> Result:
    """Solve a model (a Model, a ModelBuilder, a dict, or a path to a YAML file) under `numerics` (a Numerics or
    a dict of its fields, laid over the model's own: engine, nodes, unit, unit_range, breakpoints,
    continuation_nodes, tol, damping, max_newton, variable, settings).  The other options are the solve's:
    init (action kernels or raw maps per agent to start from), start ("zero", "coarse", or "stationary": the
    continuation's stationary maps; the default is "stationary" wherever a continuation is given, on the file
    form and the keyword form alike, else "zero"), tol (over the numerics'), max_evaluations and deadline (the bounds; past
    either the best iterate is returned not converged), progress (a callable on {"evaluation", "residual",
    "phase", "seconds"} after every evaluation), diagnostics (False skips the checks at the end), refine (re-solve
    on a finer grid and report the change, res.refinement), stability (add res.stability()), verbose;
    naive_observers ({agent: [observers]}, the stationary engine); past and continuation (a transition's, on the
    spectral engine; on a model of kind "transition" each overrides the file's block).  Unknown options are a
    TypeError naming the Numerics or Settings field they belong to; nodes= and settings= are accepted as aliases of the
    Numerics fields until 0.6, with a DeprecationWarning.  res.numerics is the resolved Numerics."""
    model = as_model(model)
    for k in list(deprecated):
        if k in _ALIASES:
            import warnings
            warnings.warn(f"solve({k}=) is deprecated, pass Numerics({k}=) (the alias goes in 0.6)", DeprecationWarning, stacklevel=2)
            numerics = Numerics.of(numerics).merged(Numerics.of({k: deprecated.pop(k)}))
    if deprecated:
        bad = sorted(deprecated)
        fields = [k for k in bad if k in Numerics.field_names()]
        tuning = [k for k in bad if k in {f.name for f in dataclasses.fields(Settings)}]
        hint = (f"; {fields} are fields of Numerics: solve(model, Numerics({fields[0]}=...))" if fields else "") + \
               (f"; {tuning} are fields of Settings: solve(model, Numerics(settings={{{tuning[0]!r}: ...}}))" if tuning else "")
        raise TypeError(f"unknown option(s) {bad} for solve()" + (hint or "; see help(noisestate.solve)"))
    S, num = engines.build(model, numerics, verbose=verbose, naive_observers=naive_observers, past=past, continuation=continuation)
    kw = num.solve_kw()
    if tol is not None:
        kw["tol"] = tol
    res = S.solve(init=init, start=engines.default_start(S, start), max_evaluations=max_evaluations, deadline=deadline, progress=progress,
                  diagnostics=diagnostics, **kw)
    if refine:
        res.refine()
    if stability:
        res.stability()
    return res
