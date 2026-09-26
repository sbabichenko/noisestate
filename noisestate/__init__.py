"""noisestate: equilibrium solver for linear-quadratic-Gaussian games with private information.

    import noisestate as ns
    res = ns.solve("examples/ch4_kyle_back.yaml")                    # the model file's own numerics
    res = ns.solve(model, ns.Numerics(nodes=32, tol=1e-12))          # a change of resolution, not of model
    res.summary(); res.kernel("P", "w"); res.costs; res.to_dict()

The model (a Model, a dict or a path) is the problem; a Numerics (or a dict of its fields) is how it
is solved: the engine, the grid, the tolerances, the settings.  solve() lays the given numerics over the
model's own, builds the engine (noisestate.engines) and returns one Result.
"""
import dataclasses

from .spec import Model
from .numerics import Numerics
from .accel import ConvergenceError, DiagnosticsError, ResultValidationError
from ._settings import Settings
from .results import Result
from . import engines
from .sweep import sweep, SweepPoint
from .comparison import compare, ComparisonResult, ScenarioResult
from .diagnostics import Assessment, Policy, Status
from .grid_cache import clear as clear_grid_cache
from .transition import transition, transition_gap
from .schema import schema
from .expr import Param, shocks, State, Control, define, Signal, Agent, Stationary, Finite, Transition
from .expr import dt, Differential, params, Game
from .expr import sqrt, exp, log, sin, cos, tanh
from .kernel import Kernel

__all__ = ["Model", "Numerics", "example", "examples", "ConvergenceError", "DiagnosticsError", "ResultValidationError", "Settings", "Result", "engines", "load", "solve", "sweep", "transition", "transition_gap", "read_yaml", "read_json",
           "compare", "ComparisonResult", "ScenarioResult", "Assessment", "Policy", "Status", "clear_grid_cache", "schema",
           "Param", "shocks", "State", "Control", "define", "Signal", "Agent", "Stationary", "Finite", "Transition", "SweepPoint",
           "Kernel", "sqrt", "exp", "log", "sin", "cos", "tanh", "dt", "Differential", "params", "Game"]

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


def examples() -> list:
    """The names of the shipped model files, for ns.example()."""
    import os
    return sorted(f[:-5] for f in os.listdir(_examples_dir()) if f.endswith(".yaml"))


def _examples_dir() -> str:
    """Where the shipped model files are: inside the package on an installed wheel, the repository's
    own examples/ in a checkout (pyproject maps the one to the other, so they are the same files)."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "examples"), os.path.join(os.path.dirname(here), "examples")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("the shipped examples are not installed beside the package")


def example(name: str) -> str:
    """The path of a shipped model file, by name: ns.load(ns.example("ch4_kyle_back")).

    The examples ship INSIDE the wheel, so this works from a pip install as well as a checkout --
    the first line of the README needed the repository before it did.  A path rather than a Model
    because a relative horizon.past.model resolves against the file's own directory, and because
    the file is worth reading.
    """
    import os
    path = os.path.join(_examples_dir(), name[:-5] if name.endswith(".yaml") else name) + ".yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(f"no shipped example named {name!r}; ns.examples() lists them: {examples()}")
    return path


def load(path: str) -> Model:
    """The model of a YAML file; a relative path in horizon.past.model is taken from the file's directory."""
    import os
    d = read_yaml(path)
    if isinstance(d, dict) and "name" not in d:
        d = {"name": os.path.splitext(os.path.basename(path))[0], **d}     # a file without a name is named after itself
    return Model.from_dict(d, base_dir=os.path.dirname(os.path.abspath(path)))


def as_model(model) -> Model:
    """A Model from what solve() accepts: a Model, a dict (the grammar or the equations form) or a path to a YAML file."""
    if isinstance(model, str):
        return load(model)
    if isinstance(model, dict):
        return Model.from_dict(model)
    if isinstance(model, Model):
        return model
    raise TypeError(f"expected a Model, a dict or a path, not {type(model).__name__}")


def _radius_when_the_window_fails(res, asked: bool) -> None:
    """Compute the best-response spectral radius when the window guard fires, even unasked.

    A stationary window longer than the kernel's support can hold a second fixed point, and Anderson
    with a Newton polish is a root finder: it will sit on one that naive best-response adjustment would
    flee (deliberately -- the Kyle-Back equilibrium is unstable and genuine).  What identifies a
    spurious branch is an unstable radius *together with* a kernel that has not decayed at the window's
    edge, so the radius is only informative where the window check has already failed; there it is
    worth the best responses it costs (docs/limits.md).  A stability report that fails is not fatal:
    the flag it would have carried is simply absent.
    """
    if asked or getattr(res, "stability_report", None) is not None:
        return
    if not res.converged or not any(d["name"] == "window" and d["ok"] is False for d in res.diagnostics.rows):
        return
    try:
        res.stability()
    except Exception:                      # an unavailable radius must not fail a solve that converged
        pass


def _after_solve(res, refine: bool, stability: bool, diagnostics: bool):
    """What every solve does once it has a result, whichever branch produced it.

    solve() returns down two paths -- the march in T for a transition with horizon.settle, and the
    ordinary engine solve -- and both ran this same block.  The order matters and is not obvious:
    refine() and stability() are what the caller asked for, and the unasked radius comes last so it
    can see that stability() has already run rather than computing it twice.
    """
    if refine:
        res.refine()
    if stability:
        res.stability()
    if diagnostics:
        _radius_when_the_window_fails(res, stability)
    return res


def solve(model, numerics=None, *, start_from=None, start_policy=None, tol=None, max_evaluations=None, deadline=None,
          progress=None, diagnostics: bool = True, refine: bool = False, stability: bool = False, verbose: bool = False,
          past=None, continuation=None, **unknown) -> Result:
    """Solve a model (a Model, a dict, or a path to a YAML file) under `numerics` (a Numerics or
    a dict of its fields, laid over the model's own: engine, nodes, unit, unit_range, breakpoints,
    continuation_nodes, tol, damping, max_newton, variable, settings).  The other options are the solve's:
    start_from (an OBJECT: action kernels or raw maps per agent to begin at), start_policy (a NAME:
    "zero", "coarse", or "stationary", the continuation's stationary maps; the default is "stationary"
    wherever a continuation is given, on the file form and the keyword form alike, else "zero").  They are
    two arguments because they are two different things, and supplying BOTH is an error rather than a
    precedence rule: a rule would silently discard one of the two things the caller asked for, and the
    caller could not tell which was used.  tol (over the numerics'), max_evaluations and deadline (the bounds; past
    either the best iterate is returned not converged), progress (a callable on {"evaluation", "residual",
    "phase", "seconds"} after every evaluation), diagnostics (False skips the checks at the end), refine (re-solve
    on a finer grid and report the change, res.refinement), stability (add res.stability(); it is also
    computed unasked when the window guard fails, where an unstable radius marks a spurious branch), verbose;
    past and continuation (a transition's, on the
    spectral engine; on a model of kind "transition" each overrides the file's block).  Unknown options are a
    TypeError naming the Numerics or Settings field they belong to.  res.numerics is the resolved Numerics."""
    if start_from is not None and start_policy is not None:
        #  Why there is no precedence rule: docs/design/api_spec.txt.  The caller needs the choice.
        raise TypeError("solve() takes start_from OR start_policy, not both. start_from is an explicit "
                        "starting point (kernels or maps); start_policy is a name (\"zero\", \"coarse\", "
                        "\"stationary\") for one the solver generates. Drop whichever you did not mean.")
    model = as_model(model)
    if "naive_observers" in unknown:
        from .stationary import WITHDRAWN_NAIVE
        raise TypeError(WITHDRAWN_NAIVE)
    grid = {k: unknown.pop(k) for k in list(unknown) if k in Numerics.field_names()}
    if grid:
        # solve(model, nodes=24): a Numerics field given directly is laid over numerics, like numerics={"nodes": 24}
        base = numerics.to_dict() if isinstance(numerics, Numerics) else dict(numerics or {})
        numerics = {**base, **grid}
    if unknown:
        bad = sorted(unknown)
        fields = [k for k in bad if k in Numerics.field_names()]
        tuning = [k for k in bad if k in {f.name for f in dataclasses.fields(Settings)}]
        hint = (f"; {fields} are fields of Numerics: solve(model, Numerics({fields[0]}=...))" if fields else "") + \
               (f"; {tuning} are fields of Settings: solve(model, Numerics(settings={{{tuning[0]!r}: ...}}))" if tuning else "")
        raise TypeError(f"unknown option(s) {bad} for solve()" + (hint or "; see help(noisestate.solve)"))
    if model.horizon.kind == "transition" and model.horizon.settle is not None:      # the horizon is the march's output
        from .transition import march_model
        if start_from is not None or start_policy is not None:
            raise ValueError("a transition with horizon.settle is solved by the march in T, which starts every point itself; "
                             "start_from and start_policy do not apply")
        res = march_model(model, numerics, past=past, continuation=continuation, verbose=verbose, tol=tol, max_evaluations=max_evaluations,
                          deadline=deadline, progress=progress, diagnostics=diagnostics)
        return _after_solve(res, refine, stability, diagnostics)
    S, num = engines._build(model, numerics, verbose=verbose, past=past, continuation=continuation)
    kw = num.solve_kw()
    if tol is not None:
        kw["tol"] = tol
    res = S.solve(start_from=start_from, start_policy=engines.default_start(S, start_policy), max_evaluations=max_evaluations, deadline=deadline, progress=progress,
                  diagnostics=diagnostics, **kw)
    return _after_solve(res, refine, stability, diagnostics)
