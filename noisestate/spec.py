"""Model specification: the game as data.

A model is a finite set of Brownian channels, linear state dynamics, algebraic
definitions, and agents.  Each agent has controls, signal rows (noisy linear
observations of states and of other agents' controls, possibly delayed), and a
quadratic flow loss.  Every coefficient may be a number or a string evaluated
against the model's parameters.

Atoms.  A quantity at a lag is written ``name@tau``: ``P@0.5`` is the control P
half a unit of time ago, ``P@-0.5`` is P half a unit ahead (a lead), ``P`` is
P now.  ``name`` may be a state, a control, or a definition.

Losses.  ``loss`` is a list of terms ``[coef, a, b]`` (quadratic) or
``[coef, a]`` (linear); the flow loss is their sum and the agent minimises
E int e^{-rho t} loss_t dt (rho = 0 is average cost).  A target theta on X is
the terms of (X - theta)^2 less its constant: ``[1, X, X], [-2*theta, X]``.

Means.  Linear loss terms, a constant in a state's drift (the key ``const``:
``drift: {X: -a, D: 1.0, const: 0.3}``) and, on a finite horizon, a state's
``initial`` value (``X: {drift: ..., noise: ..., initial: 1.0}``; zero by
default, and not allowed in a stationary model) move the means of the states
and controls, deterministic and common knowledge; the kernels do not depend
on them.  The stationary engine solves the means as constants and the finite
engines as paths on [0, T] (res.means, res.cost_parts).
"""
from __future__ import annotations

import copy
import os

import ast
import math
import re
import warnings

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union, overload

if TYPE_CHECKING:                       # the overloads below name it; the runtime imports it lazily
    from .expr import Signal

Number = Union[int, float, str]
Atom = Tuple[str, float]          # (primary name, lag); lag > 0 past, < 0 future
Expr = Dict[Atom, float]          # linear combination of primary atoms
CONST = "const"                   # the key of a constant in a state's drift (a quantity may not carry this name)


_ATOM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:@\s*([-+]?[0-9.eE+-]+|[A-Za-z_][A-Za-z0-9_]*))?\s*$")


def parse_atom(s: str, params: Dict[str, float]) -> Atom:
    m = _ATOM_RE.match(s)
    if not m:
        raise ValueError(f"bad atom {s!r}; expected 'name' or 'name@lag'")
    name, lag = m.group(1), m.group(2)
    if lag is None:
        return (name, 0.0)
    try:
        return (name, float(lag))
    except ValueError:
        if lag in params:
            return (name, float(params[lag]))
        raise ValueError(f"unknown lag parameter {lag!r} in atom {s!r}")


_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "sin": math.sin, "cos": math.cos,
          "tanh": math.tanh, "abs": abs, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}
_UNOPS = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def safe_eval(expr: str, params: Dict[str, float]) -> float:
    """Evaluate an arithmetic expression over the parameters: numbers, + - * / **, unary signs,
    the functions sqrt exp log sin cos tanh abs min max, and the constants pi and e.  Anything
    else (attribute access, subscripts, names that are not parameters, calls to other functions)
    is rejected, so model files from untrusted sources cannot run code."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse coefficient {expr!r}: {exc.msg}") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in params:
                return float(params[node.id])
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ValueError(f"unknown parameter {node.id!r} in coefficient {expr!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return float(_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"unsupported expression in coefficient {expr!r}: {ast.dump(node)[:60]}")
    return float(ev(tree))


def eval_coef(c: Number, params: Dict[str, float]) -> float:
    if isinstance(c, bool):
        raise ValueError(f"coefficient must be a number or an expression, got {c!r}")
    if isinstance(c, (int, float)):
        return float(c)
    return safe_eval(str(c), params)


def parse_expr(spec, params: Dict[str, float]) -> Dict[str, float]:
    """A linear expression as {atom string: coef}; accepts a dict or a list of [coef, atom]."""
    out: Dict[str, float] = {}
    if spec is None:
        return out
    if isinstance(spec, dict):
        items = [(k, v) for k, v in spec.items()]
    else:
        items = [(a, c) for c, a in spec]
    for atom, coef in items:
        out[atom] = out.get(atom, 0.0) + eval_coef(coef, params)
    return out


class _Recording(dict):
    """A parameter dict that remembers which keys were looked up."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.used = set()

    def __getitem__(self, k):
        self.used.add(k); return super().__getitem__(k)

    def __contains__(self, k):
        if super().__contains__(k):
            self.used.add(k)
        return super().__contains__(k)


@dataclass
class State:
    name: str
    drift: Dict[str, float] = field(default_factory=dict)   # linear in atoms (states, controls, defs, lags)
    noise: Dict[str, float] = field(default_factory=dict)   # channel -> loading (sigma row)
    initial: Optional[float] = None                         # value at t = 0 (finite horizon; it moves the means only); None: not
                                                            # given (zero, or with a past the past's constant mean)


@dataclass
class Definition:
    name: str
    expr: Dict[str, float]


@dataclass
class SignalRow:
    name: str
    drift: Dict[str, float] = field(default_factory=dict)
    noise: Dict[str, float] = field(default_factory=dict)   # channel -> loading (E row)
    delay: float = 0.0                                      # observed with this delay


@dataclass
class Agent:
    name: str
    controls: List[str]
    signals: List[SignalRow]
    loss: List[list]                  # [[coef, a, b], [coef, a], ...] with atoms as strings
    myopic: bool = False              # competitive: ignore continuation effects of own action
    constant: float = 0.0             # the loss's constant: moves no strategy, but is part of the cost
    terminal: List[list] = field(default_factory=list)   # the loss at T (finite horizon): [[coef, a, b], [coef, a]], states only
    terminal_constant: float = 0.0    # the terminal loss's constant


@dataclass
class Horizon:
    """The horizon block (the economics: kind, discount, window, a transition's past and continuation) and,
    after it, the resolved numerics the engines read (the file's `numerics:` block, or a Numerics laid over it
    by solve(); `Model.numerics` presents them, `Model.with_numerics` changes them)."""
    kind: str = "stationary"          # "stationary" | "finite" (a finite horizon) | "transition"
    discount: float = 0.0
    #  TWO DISTINCT QUANTITIES, never aliased.  Before 0.8 one `window` field held both, and the
    #  comment on it read "L for stationary; T for finite and transition" -- so a transition's own
    #  lag window had nowhere to live and was exiled to the nested `stationary` block.
    window: Optional[float] = None    # L, the lag-truncation length: stationary, and a transition's
    T: Optional[float] = None         # the terminal time: finite and transition
    # kind "transition" only: the past ({"model": a path or an inline stationary model dict, "initial": [shocks]}),
    # the continuation ("stationary", the default, or "end") and the sizing of the new model's stationary solve
    # for the continuation ({"window": the past's, "nodes": numerics.continuation_nodes})
    past: Optional[dict] = None
    continuation: Optional[str] = None
    stationary: Optional[dict] = None
    settle: Optional[float] = None    # kind "transition" only: in place of T, the settle tolerance the horizon T is found for (a march in T)

    @property
    def extent_known(self) -> bool:
        """Whether the primary axis has a length yet.

        False for exactly one thing: a transition given `settle` instead of T, whose terminal time
        the march finds.  The extent-dependent checks (lags, leads, unit_range, breakpoints) are
        deferred there rather than run against a stand-in -- the march builds a real model at each
        T, and each of those is validated.
        """
        return self.window is not None if self.kind == "stationary" else (self.T is not None or self.window is not None)

    @property
    def extent(self) -> float:
        """The length of the PRIMARY COMPUTATIONAL AXIS: T for a finite horizon or a transition,
        the lag window L for a stationary model.

        A derived quantity, not a third field and not an alias: it SELECTS whichever of T and
        window a given operation needs.  Grid construction, breakpoints and unit_range want this;
        a transition's continuation wants `window` and must never take extent, which there is T.
        """
        if self.kind == "stationary":
            return float(self.window)
        if self.T is not None:
            return float(self.T)
        #  a transition given `settle` instead of T: the terminal time is not determined yet -- the
        #  march finds it -- so the template's extent is its continuation window when it has one.
        if self.window is not None:
            return float(self.window)
        raise ValueError("this transition has neither a terminal time T nor a continuation window: "
                         "its extent is not determined until the settle march sets T")
    # ---- the resolved numerics (numerics.py): the grid
    breakpoints: Optional[List[float]] = None
    unit: Optional[float] = None
    unit_range: Optional[float] = None
    nodes: int = 16
    engine: Optional[str] = None      # "stationary" | "spectral" | "cells"; None: the kind's default
    # ---- the resolved numerics: the fixed point's options and the tuning constants
    tol: Optional[float] = None
    damping: Optional[float] = None
    max_newton: Optional[int] = None
    variable: Optional[str] = None
    settings: object = None           # a Settings; None: the defaults



@dataclass(init=False)
class Model:
    """The game as data, in one of two spellings.  The field form is the file's structure (from_dict builds it:
    channels, states, agents, horizon, definitions, ties, params); the expression form (noisestate.expr) is the
    equations, `Model("ch1", states=[X], agents=[player1, player2], horizon=Stationary(window=3.0))` with
    State, Control, Signal, Agent and Param objects, compiled to the same structure through from_dict, so the
    two are one class and one model (`to_dict()` is the file in both cases)."""
    name: str
    channels: List[str]
    states: List[State]
    agents: List[Agent]
    horizon: Horizon
    definitions: List[Definition] = field(default_factory=list)
    ties: List[List[str]] = field(default_factory=list)      # groups of agents sharing one strategy
    params: Dict[str, float] = field(default_factory=dict)
    source: Optional[dict] = field(default=None, repr=False)  # the file structure with its expressions, if built from one

    def __init__(self, name: str = "model", channels=None, states=None, agents=None, horizon=None, definitions=None,
                 ties=None, params=None, source=None, *, numerics=None):
        from . import expr
        if expr.is_expression_form(states, agents, horizon, definitions):
            if channels is not None:
                raise ValueError("Model(): the channels of an expression model are the shocks it uses; do not give channels=")
            d = expr.compile_model(name, states, agents, definitions=definitions, ties=ties, horizon=horizon,
                                   numerics=numerics, params=params)
            built = Model.from_dict(d)
            self.__dict__.update(built.__dict__)
            self.source = built.to_dict()                 # the normalised file (as load(save()) reads it back)
            return
        if numerics is not None:
            raise TypeError("Model(): numerics= belongs to the expression form; the field form carries them in its Horizon")
        self.name = name; self.channels = list(channels or []); self.states = list(states or []); self.agents = list(agents or [])
        self.horizon = horizon if horizon is not None else Horizon()
        self.definitions = list(definitions or []); self.ties = list(ties or [])
        self.params = params if params is not None else {}
        self.source = source

    def __repr__(self) -> str:
        """One line: the name, the horizon with its own lengths, and what the model holds.

        The generated dataclass repr expands every state, agent, signal row and loss term, which is
        the whole model file on one line in a notebook.  describe() is the readable long form and
        to_dict() the exact one; this is for the cell that just names a model.
        """
        hz = self.horizon
        parts = [hz.kind]
        if hz.kind == "stationary" and hz.window is not None:
            parts.append(f"window={hz.window:g}")            # L, the lag-truncation length
        else:
            if hz.T is not None:
                parts.append(f"T={hz.T:g}")
            elif hz.settle is not None:
                parts.append(f"settle={hz.settle:g}")        # T is the march's output, not yet known
            if hz.window is not None:
                parts.append(f"window={hz.window:g}")
        if hz.discount:
            parts.append(f"discount={hz.discount:g}")
        def _plural(n, word):
            return f"{n} {word}" + ("" if n == 1 else "s")
        held = [_plural(len(self.states), "state"), _plural(len(self.agents), "agent"),
                _plural(len(self.channels), "channel")]
        if self.definitions:
            held.append(_plural(len(self.definitions), "definition"))
        if self.ties:
            held.append(_plural(len(self.ties), "tie"))
        return f"<Model {self.name!r} {' '.join(parts)}: {', '.join(held)}>"

    # ------------------------------------------------------------ the expression form's conveniences
    def describe(self):
        """Return a readable summary of the model as it now stands.

        Uses the current evaluated coefficients, without compiling or solving.
        The value is a string: print it in a terminal for aligned plain text,
        or return it from a notebook cell for the same content as HTML.
        The notes describe conventions, not a certification of solvability.
        """
        from .description import describe
        return describe(self)

    def solve(self, numerics=None, **solve_kw):
        """noisestate.solve(self, numerics, **solve_kw): `game.solve(nodes=24)`."""
        from . import solve
        return solve(self, numerics, **solve_kw)

    def sweep(self, numerics=None, solver_kw=None, solve_kw=None, verbose: bool = False, **values):
        """noisestate.sweep over one parameter given by keyword, `model.sweep(p1=[0.3, 1, 3])`: the rows, each
        with .value, .result, .jump (a SweepPoint, a dict)."""
        from . import sweep
        if len(values) != 1:
            raise ValueError(f"sweep() takes exactly one parameter by keyword, e.g. sweep(p1=[...]); got {sorted(values)}")
        (param, vals), = values.items()
        return sweep(self, param, list(vals), numerics=numerics, solver_kw=solver_kw, solve_kw=solve_kw, verbose=verbose)

    def with_finite(self, T: float, **fields) -> "Model":
        """The same model on the finite horizon [0, T].  T is the terminal time, not a lag window."""
        return self._patch_horizon(kind="finite", T=T, window=None, **fields)

    def with_stationary(self, window: float, **fields) -> "Model":
        """The same model on a stationary horizon with lag-truncation length L = `window`."""
        return self._patch_horizon(kind="stationary", window=window, T=None, **fields)

    def with_transition(self, T: float, past, window: Optional[float] = None, **fields) -> "Model":
        """The same model as a transition on [0, T] from `past`.

        A transition carries BOTH quantities: T is when the game ends, and `window` is the
        continuation's lag-truncation length L.  `window` defaults to the past's, which is what a
        transition without one has always used.
        """
        extra = {} if window is None else {"window": window}
        return self._patch_horizon(kind="transition", T=T, past=past, window=extra.get("window"), **fields)

    def to_equations(self) -> dict:
        """The model written as equations (noisestate.equations): the form a person reads, and save() writes."""
        from . import equations
        return equations.from_grammar(self.to_dict())

    def save(self, path: str, form: str = "equations") -> None:
        """Write the model file as YAML, the parameter expressions intact: written as equations (to_equations(),
        the default, the form a person reads) or in the grammar (form="grammar", to_dict()).  load() reads both.

        A transition's past that lies under this file's own directory is written relative to it, so the
        pair travels together; a past elsewhere keeps its absolute path, which is what it means (that
        file, not whatever happens to sit beside the copy)."""
        import yaml
        if form not in ("equations", "grammar"):
            raise ValueError(f"save(): form is 'equations' or 'grammar', not {form!r}")
        d = self.to_equations() if form == "equations" else self.to_dict()
        past = (d.get("horizon") or {}).get("past")
        if isinstance(past, dict) and isinstance(past.get("model"), str):
            base = os.path.dirname(os.path.abspath(path)) or os.curdir
            try:
                rel = os.path.relpath(past["model"], base)
            except ValueError:                    # another drive on Windows: no relative path exists
                rel = None
            if rel is not None and not rel.startswith(os.pardir + os.sep) and rel != os.pardir:
                past["model"] = rel
        with open(path, "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Model":
        """The model of a YAML file (noisestate.load)."""
        from . import load
        return load(path)

    # ------------------------------------------------------------ numerics
    @property
    def numerics(self):
        """The numerics this model carries (its file's block, or what solve() laid over it), as a Numerics; the
        engine is None until a solve resolves it from the horizon kind (res.numerics is resolved)."""
        from .numerics import Numerics
        hz = self.horizon
        st = hz.stationary or {}
        return Numerics(engine=hz.engine, nodes=hz.nodes, unit=hz.unit, unit_range=hz.unit_range,
                        breakpoints=None if hz.breakpoints is None else list(hz.breakpoints),
                        continuation_nodes=st.get("nodes"), tol=hz.tol, damping=hz.damping, max_newton=hz.max_newton,
                        variable=hz.variable, settings=hz.settings)

    def _rebuilt(self, d: dict) -> "Model":
        """from_dict(d).  source is not carried: from_dict() rebuilds it from d, which is how with_params() keeps
        the parameter expressions intact."""
        return Model.from_dict(d)

    def with_numerics(self, numerics=None, **fields) -> "Model":
        """A new model with these numerics fields (a Numerics or dict, and/or keywords) laid over its own:
        the model's problem is unchanged, its resolution, engine or tolerances are not."""
        from .numerics import Numerics
        num = Numerics.of(numerics).merged(Numerics.of(fields)) if fields else Numerics.of(numerics)
        d = self.to_dict()
        d["numerics"] = self.numerics.merged(num).to_dict()
        return self._rebuilt(d)

    # ------------------------------------------------------------ lookups
    @property
    def state_names(self) -> List[str]:
        return [s.name for s in self.states]

    @property
    def control_names(self) -> List[str]:
        return [u for a in self.agents for u in a.controls]

    @property
    def def_names(self) -> List[str]:
        return [d.name for d in self.definitions]

    def owner_of(self, control: str) -> Agent:
        for a in self.agents:
            if control in a.controls:
                return a
        raise KeyError(control)

    # ---------------------------------------------------------- expansion
    def expand(self, expr: Dict[str, float]) -> Expr:
        """Expand an expression over atom strings into primary atoms (states, controls): the linear part;
        a constant (the key `const`, allowed in a state's drift, see constant()) is left out."""
        defs = {d.name: d.expr for d in self.definitions}
        out: Expr = {}

        def add(name: str, lag: float, coef: float, depth: int):
            if depth > 50:
                raise ValueError(f"definition cycle involving {name!r}")
            if name == CONST:
                if lag != 0:
                    raise ValueError(f"a constant has no lag: write {CONST}, not {CONST}@{lag:g}")
            elif name in defs:
                for sub, c2 in defs[name].items():
                    n2, l2 = parse_atom(sub, self.params)
                    add(n2, lag + l2, coef * c2, depth + 1)
            elif name in self.state_names or name in self.control_names:
                key = (name, float(lag))
                out[key] = out.get(key, 0.0) + coef
            else:
                raise ValueError(f"unknown quantity {name!r}")

        for atom, coef in expr.items():
            n, l = parse_atom(atom, self.params)
            add(n, l, coef, 0)
        return {k: v for k, v in out.items() if v != 0.0}

    def constant(self, expr: Dict[str, float]) -> float:
        """The constant part of an expression: the coefficient under the key `const` (zero when absent)."""
        return float(sum(c for atom, c in expr.items() if parse_atom(atom, self.params)[0] == CONST))

    @property
    def drives_means(self) -> bool:
        """Whether anything moves the means: a linear loss term, a constant in a state's drift, an initial state."""
        return (any(len(t) == 2 for a in self.agents for t in a.loss) or any(self.constant(s.drift) != 0 for s in self.states)
                or any(s.initial for s in self.states))

    def all_lags(self) -> List[float]:
        """Every distinct positive lag or observation delay in the model."""
        lags = set()
        for s in self.states:
            for (n, l) in self.expand(s.drift):
                if l > 0:
                    lags.add(l)
        for a in self.agents:
            for r in a.signals:
                if r.delay > 0:
                    lags.add(r.delay)
                for (n, l) in self.expand(r.drift):
                    if l > 0:
                        lags.add(l)
                    if r.delay + l > 0:
                        lags.add(round(r.delay + l, 12))        # a delayed row reads the quantity at delay + lag
            for term in a.loss:
                for atom in term[1:]:
                    for (n, l) in self.expand({atom: 1.0}):
                        if l != 0:
                            lags.add(abs(l))
        return sorted(lags)

    def _agent_signature(self, a: "Agent"):
        """Structure of an agent's problem up to relabelling of its own controls, rows, channels and of
        the other agents' controls: what a tie must preserve."""
        own = {u: f"own{i}" for i, u in enumerate(a.controls)}
        others = set(self.control_names) - set(a.controls)
        # states referenced by one agent only are that agent's private states: compared by role, not name
        users: Dict[str, set] = {}
        for ag in self.agents:
            exprs = [self.expand(r.drift) for r in ag.signals] + [self.expand({s: 1.0}) for t in ag.loss for s in t[1:]]
            for e in exprs:
                for (n, l) in e:
                    if n in self.state_names:
                        users.setdefault(n, set()).add(ag.name)

        states = {s.name: s for s in self.states}

        def cls(name, depth=0):
            if name in own:
                return own[name]
            if name in others:
                return "other_control"
            if len(users.get(name, set())) <= 1:
                # a private state is compared by its dynamics: drift (canonicalised, with its constant) and noise loadings
                s = states[name]
                drift = canon(self.expand(s.drift), depth + 1) if depth < 3 else "..."
                return "private_state:" + repr((drift, round(self.constant(s.drift), 9), round(s.initial or 0.0, 9),
                                                tuple(sorted(round(v, 9) for v in s.noise.values()))))
            return "state:" + name

        def canon(expr, depth=0):
            return tuple(sorted((cls(n, depth), round(l, 9), round(c, 9)) for (n, l), c in expr.items()))

        def noise_sig(noise):
            # a row's noise channel may also drive a state: shared with a private state of this agent, a
            # public state, or nothing; the tie must preserve that
            out = []
            for ch, v in noise.items():
                drives = [s.name for s in self.states if ch in s.noise]
                kind = tuple(sorted(("private" if len(users.get(n, set())) <= 1 else "state:" + n) for n in drives))
                out.append((round(v, 9), kind))
            return tuple(sorted(out))
        rows = tuple((canon(self.expand(r.drift)), noise_sig(r.noise), round(r.delay, 9)) for r in a.signals)
        loss = []
        for term in a.loss:
            ex = [canon(self.expand({s: 1.0})) for s in term[1:]]
            loss.append((round(float(term[0]), 9), tuple(sorted(ex))))
        return (len(a.controls), a.myopic, rows, tuple(sorted(loss)))

    @property
    def notes(self) -> List[str]:
        """Conventions that apply to this particular model and are easy to misread."""
        out = []
        for a in self.agents:
            for r in a.signals:
                obs = [n for (n, l) in self.expand(r.drift) if n in self.control_names and l == 0]
                if obs:
                    out.append(f"row {a.name}.{r.name} observes the control(s) {obs}: only their regular (predictable) part is "
                               "seen; a quantity with a white component (a price that loads on a noise channel) is observed "
                               "through that channel, e.g. a row with the channel as its noise")
            if a.myopic:
                out.append(f"agent {a.name} is myopic: it ignores the effect of its action on future flows (a competitive pricing agent)")
            lin = [t for t in a.loss if len(t) == 2]
            if lin:
                out.append(f"agent {a.name}: the linear loss term(s) {lin} move only the means (the kernels do not depend on them)"
                           + self._means_note())
        for s in self.states:
            k = self.constant(s.drift)
            if k != 0:
                out.append(f"state {s.name}: the constant drift {k:g} moves only the means (the kernels do not depend on it)" + self._means_note())
            if s.initial:
                out.append(f"state {s.name}: the initial value {s.initial:g} moves only the means (the kernels do not depend on it)" + self._means_note())
        if self.horizon.kind == "stationary" and self.drives_means:
            walks = [s.name for s in self.states if not self.expand(s.drift) and self.constant(s.drift) == 0]
            if walks:
                out.append(f"state(s) {walks} are random walks with no inputs, which have no stationary mean: their means "
                           "are taken as 0, so the means of the quantities they enter are relative to their levels")
        if self.horizon.kind == "stationary":
            out.append("costs are stationary flow losses per unit time" + (" (the discount rate enters the best responses, not the reported cost)" if self.horizon.discount else ""))
        else:
            out.append("costs are discounted integrals over [0, T]")
        return out

    def _means_note(self) -> str:
        if self.horizon.kind == "stationary":
            return ("; the stationary engine solves the means of every state and control from each control's mean first-order "
                    "condition and the mean dynamics, one linear system (res.means; res.cost_parts splits each cost into its "
                    "variance and mean parts)")
        return ("; the finite engines solve the mean paths of every state and control on [0, T] from each control's mean "
                "first-order condition and the mean dynamics, one linear system (res.means on the time nodes res.mean_times; "
                "res.cost_parts splits each cost into its variance and mean parts)")

    # --------------------------------------------------------- validation
    def validate(self) -> None:
        """Every structural rule of a model, checked in a fixed order, each by one named check below (the
        first failing rule raises its ValueError; the one warning is a control with no positive own quadratic
        term).  Compile calls this on every engine; from_dict calls it before the unused-parameter check."""
        self._check_names()
        self._check_states()
        self._check_definitions()
        self._check_agents()
        self._check_ties()
        self._check_channels_used()
        self._check_control_terms()
        self._check_horizon()
        self._check_transition()

    def _check_names(self) -> None:
        """Quantity names (states, controls, definitions) are distinct and none is the reserved `const`;
        channel names and agent names are distinct."""
        names = self.state_names + self.control_names + self.def_names
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate quantity names {dup}")
        if CONST in names:
            raise ValueError(f"{CONST!r} is reserved for a constant in a state's drift; name the quantity otherwise")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("duplicate channel names")
        agent_names = [a.name for a in self.agents]
        if len(set(agent_names)) != len(agent_names):
            raise ValueError("duplicate agent names")

    def _check_states(self) -> None:
        """Each state loads known channels only, its drift is causal (no lead), its initial value is a finite
        number, and an initial value appears only on a finite horizon."""
        for s in self.states:
            for ch in s.noise:
                if ch not in self.channels:
                    raise ValueError(f"state {s.name}: unknown channel {ch}")
            for (n, l) in self.expand(s.drift):
                if l < 0:
                    raise ValueError(f"state {s.name}: its drift depends on the future value {n}@{l}; drifts must be causal")
            if s.initial is not None and (isinstance(s.initial, bool) or not isinstance(s.initial, (int, float)) or not math.isfinite(s.initial)):
                raise ValueError(f"state {s.name}: initial must be a finite number, got {s.initial!r}")
            if s.initial and self.horizon.kind == "stationary":
                raise ValueError(f"state {s.name}: an initial value ({s.initial:g}) has no meaning in a stationary model, which "
                                 "has no initial time; drop it, or solve a finite horizon")

    def _check_definitions(self) -> None:
        """Each definition expands (known quantities, no cycle) and carries no constant."""
        for d in self.definitions:
            self.expand({d.name: 1.0})
            if self.constant(d.expr) != 0:
                raise ValueError(f"definition {d.name}: a constant ({CONST}) is allowed in a state's drift only; a target is a "
                                 "linear loss term, [-2*theta, X] beside [1, X, X]")

    def _check_agents(self) -> None:
        """Each agent, in order: its controls are a non-empty list of names and myopic is a bool, then its
        signal rows (_check_signals) and its loss terms (_check_losses)."""
        for a in self.agents:
            if not isinstance(a.controls, list) or not all(isinstance(u, str) for u in a.controls):
                raise ValueError(f"agent {a.name}: controls must be a list of names, got {a.controls!r}")
            if not a.controls:
                raise ValueError(f"agent {a.name} has no controls")
            if not isinstance(a.myopic, bool):
                raise ValueError(f"agent {a.name}: myopic must be true or false, got {a.myopic!r}")
            self._check_signals(a)
            self._check_losses(a)

    def _check_signals(self, a: "Agent") -> None:
        """The agent has at least one signal row, and each row loads known channels, has a nonzero noise
        loading (exact rows are not supported), a non-negative delay, a causal drift and no constant."""
        if not a.signals:
            raise ValueError(f"agent {a.name} has no signal rows: a strategy reads its rows, so there is "
                             f"nothing to solve for (give it a row, or drop the agent and its controls)")
        for r in a.signals:
            for ch in r.noise:
                if ch not in self.channels:
                    raise ValueError(f"row {a.name}.{r.name}: unknown channel {ch}")
            if not r.noise:
                raise ValueError(f"row {a.name}.{r.name} needs a noise loading (exact rows are not supported)")
            if all(v == 0 for v in r.noise.values()):
                raise ValueError(f"row {a.name}.{r.name} has a zero noise loading (exact rows are not supported)")
            if r.delay < 0:
                raise ValueError(f"row {a.name}.{r.name}: delay must be non-negative")
            # own controls may appear in own rows (the agent knows them; they drop out of its passive rows)
            for (n, l) in self.expand(r.drift):
                if l < 0:
                    raise ValueError(f"row {a.name}.{r.name} observes a future quantity {n}@{l}")
            if self.constant(r.drift) != 0:
                raise ValueError(f"row {a.name}.{r.name}: a constant in a signal row carries no information (the agent "
                                 "knows it); leave it out")

    def _check_terminal(self, a: "Agent") -> None:
        """A terminal loss is paid at T: a finite horizon ending there (no continuation after T), and terms in the
        states or in definitions of them, at T itself (no lag, no control: a control at one instant costs nothing)."""
        if not a.terminal and not a.terminal_constant:
            return
        hz = self.horizon
        if hz.kind != "finite" and not (hz.kind == "transition" and (hz.continuation or "stationary") == "end"):
            raise ValueError(f"agent {a.name}: a terminal loss needs a game that ends at T (horizon kind finite, or a "
                             f"transition with continuation 'end'), not {hz.kind!r}")
        for term in a.terminal:
            if len(term) not in (2, 3):
                raise ValueError(f"agent {a.name}: terminal term {term} must be [coef, a] or [coef, a, b]")
            for atom in term[1:]:
                for (n, l) in self.expand({str(atom): 1.0}):
                    if n == CONST:
                        raise ValueError(f"agent {a.name}: terminal term {term} reads the constant; write it as terminal_constant")
                    if l != 0:
                        raise ValueError(f"agent {a.name}: terminal term {term} reads {n} at a lag; a terminal loss is on the states at T")
                    if n in self.control_names:
                        raise ValueError(f"agent {a.name}: terminal term {term} reads the control {n}; a terminal loss is on the "
                                         "states at T (a control at one instant costs nothing)")

    def _check_losses(self, a: "Agent") -> None:
        """Each loss term of the agent is [coef, a] or [coef, a, b], reads no constant, and a lead appears only
        in a cross term with the agent's own current control."""
        self._check_terminal(a)
        for term in a.loss:
            if len(term) not in (2, 3):
                raise ValueError(f"agent {a.name}: loss term {term} must be [coef, a] or [coef, a, b]")
            if any(parse_atom(str(atom), self.params)[0] == CONST for atom in term[1:]):
                raise ValueError(f"agent {a.name}: loss term {term} reads the constant; a linear term is [coef, X] and a "
                                 "constant in the loss moves nothing")
            ex = [self.expand({atom: 1.0}) for atom in term[1:]]
            led = [i for i, e in enumerate(ex) if any(l < 0 for (n, l) in e)]
            if led:
                # a value tau ahead also loads on shocks that arrive after t, which the age grid does not
                # carry; only its covariance with the agent's own current action is computed exactly
                other = [e for i, e in enumerate(ex) if i not in led]
                ok = (len(term) == 3 and len(led) == 1
                      and all(n in a.controls and l == 0 for (n, l) in other[0]))
                if not ok:
                    raise ValueError(f"agent {a.name}: loss term {term} uses a lead (name@-tau) outside a cross term with "
                                     "the agent's own current control; a led quantity squared, or a lead on a control, "
                                     "is not supported: write the flow with lags instead (at discount 0 the time "
                                     "average of X(t+tau)^2 equals that of X(t)^2)")

    def _check_ties(self) -> None:
        """Each tie group names known agents that are structurally identical up to relabelling."""
        agent_names = [a.name for a in self.agents]
        for group in self.ties:
            for n in group:
                if n not in agent_names:
                    raise ValueError(f"tie group names unknown agent {n}")
            ag = [next(a for a in self.agents if a.name == n) for n in group]
            sigs = [self._agent_signature(a) for a in ag]
            for a, sig in zip(ag[1:], sigs[1:]):
                if sig != sigs[0]:
                    raise ValueError(f"tied agents {group[0]} and {a.name} are not structurally identical "
                                     f"(same rows, losses and coefficients up to relabelling); untie them or fix the model")

    def _check_channels_used(self) -> None:
        """Every channel is loaded by a state or a signal row."""
        used = {ch for s in self.states for ch in s.noise} | {ch for a in self.agents for r in a.signals for ch in r.noise}
        unused = [ch for ch in self.channels if ch not in used]
        if unused:
            raise ValueError(f"channel(s) {unused} are never loaded by a state or a signal row (misspelled?)")

    def _check_control_terms(self) -> None:
        """Every control of an agent enters its loss (else the best response is undetermined), and a control with
        no strictly positive quadratic term in its own current value (nor, for a non-myopic agent, in a lagged
        read) gets a UserWarning: its best-response system is usually singular."""
        for a in self.agents:
            atoms = {n for term in a.loss for atom in term[1:] for (n, l) in self.expand({atom: 1.0})}
            missing = [u for u in a.controls if u not in atoms]
            if missing:
                raise ValueError(f"agent {a.name}: control(s) {missing} do not enter its loss; the best response would be undetermined")
            for u in a.controls:
                own = {}                                           # lag -> coefficient of the square of u@lag
                for term in a.loss:
                    if len(term) != 3:
                        continue
                    try:
                        coef = float(term[0])
                    except (TypeError, ValueError):
                        continue
                    e1, e2 = (self.expand({atom: 1.0}) for atom in term[1:])
                    for (n, l), c1 in e1.items():
                        if n == u and l >= 0 and (n, l) in e2:
                            own[l] = own.get(l, 0.0) + coef * c1 * e2[(n, l)]
                cur = own.get(0.0, 0.0)
                lagged = sorted(l for l, v in own.items() if l > 0 and v > 0)
                if cur > 0 or (cur == 0 and lagged and not a.myopic):     # a lagged read pins a non-myopic agent's control
                    continue
                why = (f"; a negative coefficient makes the loss unbounded below in {u}" if cur < 0 else
                       f"; its quadratic term in the lagged read {u}@{lagged[0]:g} does not enter a myopic agent's "
                       "first-order condition" if lagged else "")
                warnings.warn(f"agent {a.name}: control {u} has no strictly positive quadratic term in its own current "
                              f"value in the loss (the coefficient of {u} squared is {cur:g}{why}); the first-order "
                              "condition then has no term in the control itself and determines it only through the "
                              "quantities it moves, so the best-response system is usually singular (every engine "
                              "refuses it) or the problem ill-posed", UserWarning)

    def _check_horizon(self) -> None:
        """The horizon: nodes an integer of at least 2, every lag, delay and lead below the primary axis's
        extent, unit_range within it, unit positive, breakpoints increasing from 0 to it, each length
        positive where the kind has one, discount non-negative, kind one of the three engines."""
        hz = self.horizon
        if hz.nodes != int(hz.nodes):
            raise ValueError(f"numerics.nodes must be an integer, got {hz.nodes!r}")
        if hz.nodes < 2:
            raise ValueError("numerics.nodes must be at least 2")
        if hz.unit is not None and not hz.unit > 0:
            raise ValueError("numerics.unit must be positive")
        if hz.discount < 0:
            raise ValueError("horizon.discount must be non-negative")
        if hz.kind not in ("stationary", "finite", "transition"):
            raise ValueError("horizon.kind must be 'stationary', 'finite' or 'transition' (the engine, spectral or cells, is numerics.engine)")
        if hz.engine == "cells" and hz.kind != "finite":
            raise ValueError(f"numerics.engine 'cells' solves a finite horizon only, not horizon.kind {hz.kind!r}")
        if hz.engine == "stationary" and hz.kind != "stationary":
            raise ValueError(f"numerics.engine 'stationary' solves horizon.kind 'stationary' only, not {hz.kind!r}")
        if hz.engine == "spectral" and hz.kind == "stationary":
            raise ValueError("numerics.engine 'spectral' solves a finite horizon or a transition, not horizon.kind 'stationary'")

        #  Each LENGTH is checked where the kind has one: a stationary model has no T and a finite
        #  one has no lag window, so a blanket check on either would test None for one of the kinds.
        if hz.window is not None and not hz.window > 0:
            raise ValueError("horizon.window (the lag-truncation length L) must be positive")
        if hz.T is not None and not hz.T > 0:
            raise ValueError("horizon.T (the terminal time) must be positive")
        if hz.settle is None and hz.extent_known and not hz.extent > 0:
            raise ValueError(f"a {hz.kind} horizon needs a positive "
                             + ("window (the lag-truncation length L)" if hz.kind == "stationary" else "T"))

        #  The checks that need the primary axis's LENGTH.  A settle march's template has none yet
        #  -- the march finds T -- so these are deferred to the models it builds, each of which is
        #  validated in turn.  Only these are deferred: the kind, the discount and the engine
        #  pairing above are checked for every model, and guarding the whole method once hid the
        #  kind's own error behind this one.
        if not hz.extent_known:
            return
        far = [l for l in self.all_lags() if l >= hz.extent - 1e-12]
        if far:
            raise ValueError(
                f"lag(s)/delay(s) {far} are not below the horizon's extent {hz.extent}. This is a "
                "restriction of the current engines, not a statement that the model is ill-posed: a "
                "stationary kernel is truncated at the window, and the finite engines carry no "
                "history before t = 0. A transition does inherit a past, so whether a delayed row "
                "may cross the regime boundary is an open question (docs/api_spec.txt, "
                "INVESTIGATIONS): the past being represented does not establish that such a row is "
                "resolved correctly, which depends on the signal definitions and on how the two "
                "regimes join.")
        leads = [-l for s in self.states for (n, l) in self.expand(s.drift) if l < 0]
        for a in self.agents:
            for term in a.loss:
                leads += [-l for atom in term[1:] for (n, l) in self.expand({atom: 1.0}) if l < 0]
        if any(l >= hz.extent - 1e-12 for l in leads):
            raise ValueError(f"lead(s) {sorted(set(l for l in leads if l >= hz.extent - 1e-12))} are not below the horizon's extent {hz.extent}")
        if hz.unit_range is not None and hz.unit_range > hz.extent + 1e-12:
            raise ValueError(f"numerics.unit_range ({hz.unit_range}) must not exceed the horizon's extent ({hz.extent})")
        if hz.breakpoints is not None:
            bp = list(hz.breakpoints)
            if len(bp) < 2 or abs(bp[0]) > 1e-12 or abs(bp[-1] - hz.extent) > 1e-9 * max(1.0, hz.extent) or any(b2 <= b1 for b1, b2 in zip(bp, bp[1:])):
                raise ValueError(f"numerics.breakpoints {bp} must increase from 0 to the horizon's extent ({hz.extent})")

    def _check_transition(self) -> None:
        """The transition blocks: kind 'transition' requires a past block (a `model`, a list of `initial` shocks,
        or both), its continuation is 'stationary' or 'end', its `stationary` block has a positive window and
        at least 2 nodes; the other kinds refuse all three blocks (a keyword past goes to solve(past=))."""
        hz = self.horizon
        if hz.kind != "transition":
            for k in ("past", "continuation", "stationary", "settle"):
                if getattr(hz, k) is not None:
                    raise ValueError(f"horizon.{k} belongs to horizon.kind 'transition', not {hz.kind!r} (a past given by keyword goes "
                                     "to solve(model, past=...))")
            return
        if not isinstance(hz.past, dict) or not hz.past:
            raise ValueError("horizon.kind 'transition' needs a past block: horizon.past: {model: old.yaml | an inline stationary "
                             "model, initial: [{name, loads: {state: coef}, rows: {agent.row: coef}}, ...]}")
        self._check_keys("horizon.past", hz.past, {"model", "initial"})
        if hz.past.get("model") is None and not hz.past.get("initial"):
            raise ValueError("horizon.past needs a `model` (a path or an inline stationary model) or a list of `initial` shocks")
        if hz.past.get("model") is not None and not isinstance(hz.past["model"], (str, dict)):
            raise ValueError(f"horizon.past.model must be a path or an inline model dict, not {type(hz.past['model']).__name__}")
        if hz.past.get("initial") is not None and not isinstance(hz.past["initial"], list):
            raise ValueError("horizon.past.initial must be a list of shocks {name, loads, rows}")
        if hz.continuation is not None and hz.continuation not in ("stationary", "end"):
            raise ValueError(f"horizon.continuation must be 'stationary' or 'end', not {hz.continuation!r}")
        if hz.settle is not None:
            if not isinstance(hz.settle, (int, float)) or isinstance(hz.settle, bool) or not hz.settle > 0:
                raise ValueError("horizon.settle must be a positive tolerance (the relative distance of the best-response rules from "
                                 "the stationary ones the horizon T is found for)")
            if hz.continuation == "end":
                raise ValueError("horizon.settle needs continuation 'stationary': the march measures the rules against the new model's "
                                 "stationary equilibrium")
        if hz.stationary is not None:
            self._check_keys("horizon.stationary", hz.stationary, {"window", "nodes"})
            if hz.stationary.get("window") is not None and not hz.stationary["window"] > 0:
                raise ValueError("horizon.stationary.window must be positive")
            n = hz.stationary.get("nodes")
            if n is not None and (n != int(n) or n < 2):
                raise ValueError("numerics.continuation_nodes must be an integer of at least 2")

    # ------------------------------------------------------- construction
    _KEYS = {
        "model": {"name", "params", "channels", "states", "definitions", "agents", "ties", "horizon", "numerics"},
        "horizon": {"kind", "discount", "window", "T", "past", "continuation", "stationary", "settle"},
        "numerics": {"engine", "nodes", "unit", "unit_range", "breakpoints", "continuation_nodes", "tol", "damping", "max_newton",
                     "variable", "settings"},
        "state": {"drift", "noise", "initial"},
        "agent": {"controls", "signals", "loss", "myopic", "constant", "terminal", "terminal_constant"},
        "signal": {"drift", "noise", "delay"},
    }

    @staticmethod
    def _check_keys(what: str, d: dict, allowed: set, where: str = "") -> None:
        if not isinstance(d, dict):
            raise ValueError(f"{what}{where} must be a mapping, got {type(d).__name__}")
        bad = sorted(set(d) - allowed)
        if bad:
            raise ValueError(f"unknown key(s) {bad} in {what}{where}; allowed: {sorted(allowed)}")

    def to_dict(self, numeric: bool = False) -> dict:
        """The file structure of this model.  When the model was built from a file or dict, that
        source (with its parameter expressions) is returned, so re-parametrising it works; with
        numeric=True, or when there is no source, coefficients are returned as numbers."""
        hz = self.horizon
        #  each kind writes the quantities it HAS; a finite horizon has no lag window and a
        #  stationary model has no terminal time, so neither is emitted as a null
        horizon = {"kind": hz.kind, "discount": hz.discount}
        if hz.window is not None:
            horizon["window"] = hz.window
        if hz.T is not None:
            horizon["T"] = hz.T
        if hz.settle is not None:                             # the horizon is the march's output: the file says settle, not T
            horizon = {"kind": hz.kind, "discount": hz.discount, "settle": hz.settle}
        for k in ("past", "continuation"):
            if getattr(hz, k) is not None:
                horizon[k] = copy.deepcopy(getattr(hz, k))
        if hz.stationary and hz.stationary.get("window") is not None:
            horizon["stationary"] = {"window": hz.stationary["window"]}
        numerics = self.numerics.to_dict()
        if self.source is not None and not numeric:
            # the source (parameter expressions intact) with the live horizon: horizon fields may be changed
            # on the object (the engines read them at compile time); parameters may not (coefficients are
            # numbers once built), so params come from the source and with_params() makes a new model
            d = copy.deepcopy(self.source)
            hsrc = d.get("horizon") or {}
            nsrc = dict(d.get("numerics") or {})
            seen: Dict[str, float] = {}
            for k, v in (d.get("params") or {}).items():
                seen[k] = eval_coef(v, seen)

            def same(src, v) -> bool:
                """Whether the source's spelling `src` (a number, an expression, a list, a block) evaluates to v."""
                if src == v:
                    return True
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if isinstance(src, str):
                        try:
                            return abs(eval_coef(src, seen) - v) <= 1e-12 * max(1.0, abs(v))
                        except ValueError:
                            return False
                    return not isinstance(src, (list, str, bool, dict)) and src is not None and abs(float(src) - v) <= 1e-12 * max(1.0, abs(v))
                if isinstance(v, list) and isinstance(src, list) and len(src) == len(v):
                    return all(same(a, b) for a, b in zip(src, v))
                return False
            hout = {}
            for k, v in horizon.items():
                if k == "past" and k in hsrc and _eval_past_block(copy.deepcopy(hsrc[k]), seen) == v:
                    hout[k] = hsrc[k]; continue           # the past block with its expressions (the loadings evaluated agree)
                if k == "stationary" and isinstance(hsrc.get(k), dict) and same(hsrc[k].get("window"), v["window"]):
                    hout[k] = {"window": hsrc[k]["window"]}; continue
                hout[k] = hsrc[k] if k in hsrc and same(hsrc[k], v) else v      # unchanged: keep the source's spelling (an expression)
            nout = {}
            for k, v in numerics.items():
                nout[k] = nsrc[k] if k in nsrc and same(nsrc[k], v) else v
            d["horizon"] = hout
            d["numerics"] = nout
            return d
        # numeric form: parameter values are inlined, including the lags written as name@param
        p = self.params

        def atom(s: str) -> str:
            n, l = parse_atom(s, p)
            return n if l == 0 else f"{n}@{l:g}"

        def ex(e: Dict[str, float]) -> Dict[str, float]:
            return {atom(k): float(v) for k, v in e.items()}
        d = {"name": self.name, "channels": list(self.channels),
             "states": {s.name: {"drift": ex(s.drift), "noise": ex(s.noise), **({"initial": float(s.initial)} if s.initial is not None else {})}
                        for s in self.states},
             "definitions": {x.name: ex(x.expr) for x in self.definitions},
             "agents": {}, "ties": [list(g) for g in self.ties], "horizon": horizon, "numerics": numerics}
        for a in self.agents:
            d["agents"][a.name] = {"controls": list(a.controls), "myopic": a.myopic,
                                   "signals": {r.name: {"drift": ex(r.drift), "noise": ex(r.noise), "delay": r.delay} for r in a.signals},
                                   "loss": [[float(t[0])] + [atom(x) for x in t[1:]] for t in a.loss]}
            if a.constant:
                d["agents"][a.name]["constant"] = float(a.constant)
            if a.terminal:
                d["agents"][a.name]["terminal"] = [[float(t[0])] + [atom(x) for x in t[1:]] for t in a.terminal]
            if a.terminal_constant:
                d["agents"][a.name]["terminal_constant"] = float(a.terminal_constant)
        return d

    def with_params(self, **values) -> "Model":
        """A new model with these parameter values (numbers) replacing the source's; the other parameters
        keep their expressions and are re-evaluated."""
        d = self.to_dict()
        unknown = sorted(set(values) - set(d.get("params") or {}))
        if unknown:
            raise ValueError(f"{unknown} are not parameters of the model (params: {sorted(d.get('params') or {})})")
        for k, v in values.items():
            d["params"][k] = float(v)
        return self._rebuilt(d)

    #  Three calling forms, spelled out for a reader and for a type checker.  The single
    #  implementation signature admits combinations none of them allows (a Signal AND a drift), and
    #  the runtime rejects those; these say which calls are actually meant.
    @overload
    def with_signal(self, signal: "Signal", *, audience: Union[str, Iterable[str]] = "all") -> "Model": ...
    @overload
    def with_signal(self, signal: str, *, drift: Mapping[str, Any], noise: Mapping[str, Any],
                    audience: Union[str, Iterable[str]] = "all", delay: float = 0.0) -> "Model": ...
    @overload
    def with_signal(self, *, name: str, drift: Mapping[str, Any], noise: Mapping[str, Any],
                    audience: Union[str, Iterable[str]] = "all", delay: float = 0.0) -> "Model": ...

    def with_signal(self, signal=None, *, name=None, drift=None, noise=None, audience="all", delay=0.0) -> "Model":
        """Return a copy with one observation row added to the selected agents.

        TWO SPELLINGS OF ONE ROW, and the same type as the expression form:

            model.with_signal(Signal("flow", D1 + sigma * w.wZ))     # a Signal object
            model.with_signal(name="flow", drift={"D1": 1}, noise={"wZ": 0.5})

        The object form is the one to reach for when the model was built as equations -- it is the
        same Signal the Agent takes, so a row is written once and used in either workflow.  The
        block form stays because a model loaded from a FILE has no Signal object to hand, and
        building one only to take it apart again would be ceremony.

        ``audience`` is ``"all"`` (the default), one agent name, or an iterable of agent names.
        Noise channels named by the row are added to the model when absent.  Adding a name an
        agent already has is an error, never a silent replacement -- use without_signal() first.
        The operation is model construction only: it does not mutate or solve this model.
        """
        from .expr import Signal as _Signal
        #  overloaded on the FIRST argument: a Signal, or the row's name followed by its blocks
        if isinstance(signal, _Signal):
            if any(v is not None for v in (name, drift, noise)) or delay:
                raise TypeError("with_signal(): give a Signal OR name/drift/noise, not both -- the "
                                "Signal already carries its drift, noise and delay")
            return self.with_signals([signal], audience=audience)
        if isinstance(signal, str):
            if name is not None:
                raise TypeError("with_signal(): the row's name was given twice, positionally and as name=")
            name = signal
        elif signal is not None:
            raise TypeError(f"with_signal(): the first argument is a Signal or the row's name, not "
                            f"{type(signal).__name__}")
        if name is None or drift is None or noise is None:
            raise TypeError("with_signal(): needs a Signal, or a name with drift= and noise=")
        return self.with_signals({name: {"drift": drift, "noise": noise, "delay": delay}}, audience=audience)

    def with_signals(self, rows, *, audience="all") -> "Model":
        """Return a copy with several observation rows added to an audience.

        `rows` is a SEQUENCE OF SIGNALS, or a mapping of row name to its model-file block
        (``drift``, ``noise`` and an optional ``delay``).  Mixing the two in one call is an error:
        a call says which form it is written in.

        Rebuilding through :meth:`from_dict` gives transformed models the same validation and
        serialisation guarantees as models loaded from files.
        """
        from .expr import Signal as _Signal
        if isinstance(rows, (list, tuple)):
            if not rows:
                raise ValueError("with_signals(): no signals given")
            bad = [r for r in rows if not isinstance(r, _Signal)]
            if bad:
                raise TypeError("with_signals(): a sequence must hold Signals, not "
                                f"{type(bad[0]).__name__}; a mapping holds file-form blocks")
            #  Signal.compile() already produces the block this method consumes -- the same one the
            #  expression form writes into a model file -- so the two workflows share the type
            #  rather than each having their own way of saying a row.
            rows = {r.name: r.compile() for r in rows}
        if not isinstance(rows, dict) or not rows:
            raise ValueError("with_signals(): rows must be a non-empty mapping of signal names to "
                             "signal blocks, or a non-empty sequence of Signals")
        known = [a.name for a in self.agents]
        if audience == "all":
            selected = known
        elif isinstance(audience, str):
            selected = [audience]
        else:
            try:
                selected = list(audience)
            except TypeError:
                raise TypeError("with_signals(): audience must be 'all', an agent name, or an iterable of agent names") from None
        if not selected:
            raise ValueError("with_signals(): audience is empty")
        if len(set(selected)) != len(selected):
            raise ValueError(f"with_signals(): audience repeats an agent name: {selected}")
        unknown = sorted(set(selected) - set(known))
        if unknown:
            raise ValueError(f"with_signals(): unknown agent(s) {unknown}; agents: {known}")

        d = self.to_dict()
        d.setdefault("channels", [])
        for row_name, block in rows.items():
            if not isinstance(row_name, str) or not row_name:
                raise ValueError(f"with_signals(): signal names must be non-empty strings, not {row_name!r}")
            if not isinstance(block, dict):
                raise ValueError(f"with_signals(): signal {row_name!r} must be a mapping")
            self._check_keys("signal", block, self._KEYS["signal"], f" {row_name!r}")
            if not isinstance(block.get("drift") or {}, dict) or not isinstance(block.get("noise") or {}, dict):
                raise ValueError(f"with_signals(): signal {row_name!r} drift and noise must be mappings")
            for channel in (block.get("noise") or {}):
                if channel not in d["channels"]:
                    d["channels"].append(channel)
            clean = copy.deepcopy(block)
            if clean.get("delay") == 0:
                clean.pop("delay", None)
            for agent in selected:
                signals = d["agents"][agent].setdefault("signals", {})
                if row_name in signals:
                    raise ValueError(f"with_signals(): agent {agent!r} already has a signal named {row_name!r}")
                signals[row_name] = copy.deepcopy(clean)
        return self._rebuilt(d)

    def without_signal(self, name: str, *, audience="all") -> "Model":
        """Return a copy with an observation row removed from the selected agents.

        The partner of with_signal(), which refuses a name an agent already has rather than
        replacing it silently: replacing is two decisions -- remove this, add that -- and a caller
        who means both should say both.  Removing a name an agent does not have is an error too,
        for the same reason a typo should not pass quietly.
        """
        d = self.to_dict()
        known = [a.name for a in self.agents]
        selected = known if audience == "all" else ([audience] if isinstance(audience, str) else list(audience))
        unknown = sorted(set(selected) - set(known))
        if unknown:
            raise ValueError(f"without_signal(): unknown agent(s) {unknown}; agents: {known}")
        for agent in selected:
            signals = d["agents"][agent].get("signals") or {}
            if name not in signals:
                raise ValueError(f"without_signal(): agent {agent!r} has no signal named {name!r}; "
                                 f"its rows are {sorted(signals)}")
            del signals[name]
            if not signals:
                raise ValueError(f"without_signal(): removing {name!r} would leave agent {agent!r} with no "
                                 "rows at all, and an agent that observes nothing cannot act")
        #  with_signals ADDS the channels a row loads, so removal drops the ones that are now
        #  unloaded -- the model refuses a channel nothing uses, and the two operations have to be
        #  each other's inverse or a round trip does not validate.
        used = {c for st in d["states"].values() for c in (st.get("noise") or {})}
        for spec in d["agents"].values():
            for row in (spec.get("signals") or {}).values():
                used |= set(row.get("noise") or {})
        d["channels"] = [c for c in d.get("channels", []) if c in used]
        return self._rebuilt(d)

    def with_horizon(self, horizon=None, **fields) -> "Model":
        """A new model on `horizon`: Stationary(...), Finite(...) or Transition(...).

        The horizon's ECONOMICS are REPLACED, not patched.  That is the point of taking an object:
        there is no rule about which of the old kind's quantities to retain, no way for a stationary
        lag window to survive as a finite horizon's terminal time, and no way for a transition's
        past to be carried into a kind that cannot hold one.  The type states which quantities
        exist, so the question of what to keep does not arise.

        The grid stays with with_numerics(); a change of kind drops the old kind's panel sizing
        (numerics.breakpoints, unit_range, engine), which a stationary grid does not share with a
        finite one.
        """
        from .expr import _Horizon
        if fields or horizon is None:
            #  The reasoning behind the object form is in this method's docstring, not in the
            #  message: what a caller who got here needs is the call that works.
            raise TypeError(
                "with_horizon() takes a horizon object, not keywords"
                + (f" (got {sorted(fields)})" if fields else "") + ". "
                "To change one field on the same kind: with_stationary(window), with_finite(T), "
                "with_transition(T, past). To replace the horizon: with_horizon(Stationary(window=3.0)), "
                "with_horizon(Finite(T=1.0)), with_horizon(Transition(T=1.0, past=...)).")
        if not isinstance(horizon, _Horizon):
            raise TypeError("with_horizon() takes a horizon object -- Stationary(window=...), "
                            "Finite(T=...) or Transition(T=..., past=...) -- not "
                            f"{type(horizon).__name__}. To change one field on the same kind, use "
                            "with_stationary(), with_finite() or with_transition().")
        d = self.to_dict(); d.setdefault("numerics", {})
        if horizon.kind != self.horizon.kind:
            for k in ("breakpoints", "unit_range", "engine"):
                d["numerics"].pop(k, None)
        d["horizon"] = horizon.compile()
        return self._rebuilt(d)

    def _patch_horizon(self, **fields) -> "Model":
        """Internal: change horizon fields, keeping the rest.

        A kind change must state BOTH lengths, so the caller says what each becomes rather than
        leaving the method to guess which of the old kind's quantities still apply.  That guess is
        what with_horizon(**fields) used to make, and it is why with_horizon now takes an object.
        """
        d = self.to_dict(); d.setdefault("horizon", {}); d.setdefault("numerics", {})
        bad = sorted(set(fields) - self._KEYS["horizon"])
        if bad:
            raise ValueError(f"unknown horizon field(s) {bad}; the numerics fields go to with_numerics()")
        changes_kind = fields.get("kind") is not None and fields["kind"] != self.horizon.kind
        if changes_kind:
            if not {"window", "T"} <= set(fields):
                raise ValueError("a kind change must state both horizon.window and horizon.T (either may be "
                                 "None): the lag-truncation length and the terminal time are different "
                                 "quantities and neither carries over by default")
            for k in ("breakpoints", "unit_range", "engine"):
                if k not in fields:
                    d["numerics"].pop(k, None)
        d["horizon"].update({k: v for k, v in fields.items() if v is not None})
        for key, value in fields.items():
            if value is None:
                d["horizon"].pop(key, None)
        return self._rebuilt(d)

    @classmethod
    def from_dict(cls, d: dict, base_dir: Optional[str] = None) -> "Model":
        """A Model from the file structure.  base_dir: the directory a relative path in horizon.past.model is
        resolved against (the model file's own directory when loaded from a file); None leaves it as given.
        A file written as equations (noisestate.equations: shocks, "(D1 + D2) dt + sigma dW0", ...) is read too."""
        from . import equations
        if equations.is_equation_form(d):
            d = equations.to_grammar(d)
        cls._check_keys("model", d, cls._KEYS["model"])
        cls._check_keys("horizon", d.get("horizon") or {}, cls._KEYS["horizon"])
        for k, v in (d.get("states") or {}).items():
            cls._check_keys("state", v or {}, cls._KEYS["state"], f" '{k}'")
        for k, v in (d.get("agents") or {}).items():
            if not v:
                raise ValueError(f"agent {k}: empty block; give it controls, signals and a loss")
            cls._check_keys("agent", v or {}, cls._KEYS["agent"], f" '{k}'")
            if not isinstance(v.get("signals") or {}, dict):
                raise ValueError(f"agent {k}: signals must be a mapping of row name to {{drift, noise, delay}}")
            if not isinstance(v.get("loss") or [], list):
                raise ValueError(f"agent {k}: loss must be a list of [coef, a, b] terms")
            if isinstance(v.get("controls"), str):
                raise ValueError(f"agent {k}: controls must be a list of names, e.g. [{v['controls']}]")
            for rk, rv in (v.get("signals") or {}).items():
                cls._check_keys("signal", rv or {}, cls._KEYS["signal"], f" '{k}.{rk}'")
        params: Dict[str, float] = {}
        pdict = d.get("params") or {}
        if not isinstance(pdict, dict):
            raise ValueError("params must be a mapping of name to number or expression")
        for k, v in pdict.items():                            # a parameter may be an expression in earlier ones
            try:
                params[k] = eval_coef(v, params)
            except ValueError as exc:
                later = [n for n in pdict if n not in params and n != k and f"'{n}'" in str(exc)]
                if later:
                    raise ValueError(f"parameter {k!r} uses {later[0]!r}, which is defined after it; parameters are "
                                     "evaluated in order, so move it up") from None
                raise
        params = _Recording(params)                            # records which parameters the model references
        hz = d.get("horizon") or {}
        if hz.get("settle") is not None and hz.get("T") is not None:
            raise ValueError("horizon takes exactly one of T (the terminal time) and settle (the tolerance the "
                             "horizon is found for by a march in T), not both")
        nm, kind = _numerics_block(d)
        from ._settings import Settings
        cls._check_keys("numerics", nm, cls._KEYS["numerics"])
        stationary = hz.get("stationary")
        if isinstance(stationary, dict):
            stationary = {k: eval_coef(v, params) if k == "window" else v for k, v in stationary.items()}
            if nm.get("continuation_nodes") is not None:
                stationary["nodes"] = nm["continuation_nodes"]
        elif stationary is None and nm.get("continuation_nodes") is not None:
            stationary = {"nodes": nm["continuation_nodes"]}
        try:
            settings = Settings.of(nm.get("settings"))
        except TypeError as exc:
            raise ValueError(f"numerics.settings: {exc}") from None
        horizon = Horizon(kind=kind,
                          discount=eval_coef(hz.get("discount", 0.0), params),
                          #  the two quantities are read separately, and each kind is given the one
                          #  it has.  A file that spells a finite horizon's length `window` is
                          #  refused by name in _horizon_length below.
                          window=_horizon_length(hz, kind, params, "window"),
                          T=_horizon_length(hz, kind, params, "T"),
                          past=copy.deepcopy(hz["past"]) if hz.get("past") is not None else None,
                          continuation=hz.get("continuation"),
                          stationary=stationary,
                          settle=None if hz.get("settle") is None else float(eval_coef(hz["settle"], params)),
                          breakpoints=[eval_coef(b, params) for b in nm["breakpoints"]] if nm.get("breakpoints") else None,
                          unit=eval_coef(nm["unit"], params) if nm.get("unit") is not None else None,
                          unit_range=eval_coef(nm["unit_range"], params) if nm.get("unit_range") is not None else None,
                          nodes=nm.get("nodes", 16), engine=nm.get("engine"),
                          tol=None if nm.get("tol") is None else float(eval_coef(nm["tol"], params)),
                          damping=None if nm.get("damping") is None else float(eval_coef(nm["damping"], params)),
                          max_newton=nm.get("max_newton"), variable=nm.get("variable"),
                          settings=None if settings == Settings() else settings)
        if base_dir and isinstance(horizon.past, dict) and isinstance(horizon.past.get("model"), str) \
                and not os.path.isabs(horizon.past["model"]):
            horizon.past["model"] = os.path.normpath(os.path.join(base_dir, horizon.past["model"]))
        _eval_past_block(horizon.past, params)                 # an initial shock's loadings may be parameter expressions
        states = [State(name=k, drift=parse_expr(v.get("drift"), params), noise=parse_expr(v.get("noise"), params),
                        initial=eval_coef(v["initial"], params) if v.get("initial") is not None else None)
                  for k, v in (d.get("states") or {}).items()]
        defs = [Definition(name=k, expr=parse_expr(v, params)) for k, v in (d.get("definitions") or {}).items()]
        agents = []
        for k, v in (d.get("agents") or {}).items():
            rows = []
            for rk, rv in (v.get("signals") or {}).items():
                rows.append(SignalRow(name=rk, drift=parse_expr(rv.get("drift"), params),
                                      noise=parse_expr(rv.get("noise"), params),
                                      delay=eval_coef(rv.get("delay", 0.0), params)))
            loss = []
            for term in (v.get("loss") or []):
                loss.append([eval_coef(term[0], params)] + [str(x) for x in term[1:]])
            agents.append(Agent(name=k, controls=list(v.get("controls") or []), signals=rows, loss=loss,
                                myopic=v.get("myopic", False), constant=eval_coef(v.get("constant", 0.0), params),
                                terminal=[[eval_coef(t[0], params)] + [str(x) for x in t[1:]] for t in (v.get("terminal") or [])],
                                terminal_constant=eval_coef(v.get("terminal_constant", 0.0), params)))
        from types import MappingProxyType
        m = cls(name=d.get("name", "model"), channels=list(d.get("channels") or []), states=states,
                agents=agents, horizon=horizon, definitions=defs, ties=[list(g) for g in (d.get("ties") or [])],
                params=MappingProxyType(params), source=copy.deepcopy(d))     # read-only: see with_params()
        m.validate()                                           # structural errors first (its expansions also record
        m.horizon.nodes = int(m.horizon.nodes)                 # the lag parameters, 'P@tau'); then the parameter check
        if m.horizon.engine is not None and m.horizon.engine not in ("stationary", "spectral", "cells"):
            raise ValueError(f"numerics.engine must be 'stationary', 'spectral' or 'cells', not {m.horizon.engine!r}")
        if m.horizon.variable is not None and m.horizon.variable not in ("actions", "maps"):
            raise ValueError(f"numerics.variable must be 'actions' or 'maps', not {m.horizon.variable!r}")
        for k, v in pdict.items():
            if isinstance(v, str):
                safe_eval(v, params)                           # a parameter used inside another one counts as used
        unused = sorted(set(params) - params.used)
        if unused:
            raise ValueError(f"parameter(s) {unused} are defined but never used in the model (misspelled somewhere?)")
        return m




def _horizon_length(hz: dict, kind: str, params, which: str):
    """`window` (the lag-truncation length L) or `T` (the terminal time), per kind.

    Before 0.8 one key carried both, so `window: 1.0` on a finite horizon meant T.  A file still
    spelling it that way is refused by the name that replaced it rather than silently read as an
    L the finite engines have no use for.
    """
    if kind not in ("stationary", "finite", "transition"):
        return None            # an unknown kind is the kind validator's to report, not this one
    wants_window = kind in ("stationary", "transition")
    wants_T = kind in ("finite", "transition")
    if which == "window":
        if not wants_window:
            if hz.get("window") is not None:
                raise ValueError(f"horizon.window is the lag-truncation length L and a {kind} horizon has none; "
                                 f"its length is horizon.T")
            return None
        if kind == "transition":
            #  a transition's L is its CONTINUATION's, which comes from the past unless the file
            #  states one.  Defaulting it to a number would put a meaningless 8.0 on every
            #  transition that never asked for one.
            return None if hz.get("window") is None else eval_coef(hz["window"], params)
        return eval_coef(hz.get("window", 8.0), params)
    if not wants_T:
        if hz.get("T") is not None:
            raise ValueError(f"horizon.T is the terminal time and a {kind} horizon has none; "
                             f"its length is horizon.window (the lag-truncation length L)")
        return None
    if hz.get("T") is None and hz.get("settle") is None:
        raise ValueError(f"a {kind} horizon needs horizon.T (the terminal time)"
                         + ("; horizon.window there is the continuation's lag-truncation length L"
                            if kind == "transition" else ""))
    return None if hz.get("T") is None else eval_coef(hz["T"], params)


def _numerics_block(d: dict):
    """The numerics of a model dict (its `numerics:` block) and the horizon kind."""
    hz = d.get("horizon") or {}
    nm = dict(d.get("numerics") or {}) if isinstance(d.get("numerics"), dict) or d.get("numerics") is None else d["numerics"]
    if not isinstance(nm, dict):
        raise ValueError(f"numerics must be a mapping of its fields, got {type(nm).__name__}")
    return nm, hz.get("kind", "stationary")


def _eval_past_block(block, params):
    """Evaluate the initial shocks' loadings (numbers or parameter expressions) of a horizon.past block in place;
    returns the block."""
    if isinstance(block, dict) and isinstance(block.get("initial"), list):
        for sh in block["initial"]:
            if isinstance(sh, dict):
                for key in ("loads", "rows"):
                    if isinstance(sh.get(key), dict):
                        sh[key] = {k: eval_coef(v, params) for k, v in sh[key].items()}
    return block
