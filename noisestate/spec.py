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
from typing import Dict, List, Optional, Tuple, Union

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


@dataclass
class Horizon:
    kind: str = "stationary"          # "stationary" | "finite" (spectral triangle) | "finite_cells" | "transition"
    discount: float = 0.0
    window: float = 8.0               # L for stationary; T for finite and transition
    breakpoints: Optional[List[float]] = None
    unit: Optional[float] = None
    unit_range: Optional[float] = None
    nodes: int = 16
    # kind "transition" only: the past ({"model": a path or an inline stationary model dict, "initial": [shocks]}),
    # the continuation ("stationary", the default, or "end") and the sizing of the new model's stationary solve
    # for the continuation ({"window", "nodes"}; default the past's window and horizon.nodes)
    past: Optional[dict] = None
    continuation: Optional[str] = None
    stationary: Optional[dict] = None

    @property
    def is_transition(self) -> bool:
        return self.kind == "transition"


@dataclass
class Model:
    name: str
    channels: List[str]
    states: List[State]
    agents: List[Agent]
    horizon: Horizon
    definitions: List[Definition] = field(default_factory=list)
    ties: List[List[str]] = field(default_factory=list)      # groups of agents sharing one strategy
    params: Dict[str, float] = field(default_factory=dict)
    source: Optional[dict] = field(default=None, repr=False)  # the file structure with its expressions, if built from one

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

    def owner(self, control: str) -> Agent:
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
    def means_driven(self) -> bool:
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
        if self.horizon.kind == "stationary" and self.means_driven:
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
                "first-order condition and the mean dynamics, one linear system (res.means on the time nodes res.means_t; "
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
        """Each signal row of the agent loads known channels, has a nonzero noise loading (exact rows are not
        supported), a non-negative delay, a causal drift and no constant."""
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

    def _check_losses(self, a: "Agent") -> None:
        """Each loss term of the agent is [coef, a] or [coef, a, b], reads no constant, and a lead appears only
        in a cross term with the agent's own current control."""
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
        """The horizon: nodes an integer of at least 2, every lag, delay and lead below the window, unit_range
        within the window, unit positive, breakpoints increasing from 0 to the window, window positive,
        discount non-negative, kind one of the three engines."""
        hz = self.horizon
        if hz.nodes != int(hz.nodes):
            raise ValueError(f"horizon.nodes must be an integer, got {hz.nodes!r}")
        far = [l for l in self.all_lags() if l >= hz.window - 1e-12]
        if far:
            raise ValueError(f"lag(s)/delay(s) {far} are not below the window {hz.window}: a quantity read that far back, or "
                             "a row delayed that much, carries nothing within the window")
        leads = [-l for s in self.states for (n, l) in self.expand(s.drift) if l < 0]
        for a in self.agents:
            for term in a.loss:
                leads += [-l for atom in term[1:] for (n, l) in self.expand({atom: 1.0}) if l < 0]
        if any(l >= hz.window - 1e-12 for l in leads):
            raise ValueError(f"lead(s) {sorted(set(l for l in leads if l >= hz.window - 1e-12))} are not below the window {hz.window}")
        if hz.unit_range is not None and hz.unit_range > hz.window + 1e-12:
            raise ValueError(f"horizon.unit_range ({hz.unit_range}) must not exceed the window ({hz.window})")
        if hz.unit is not None and not hz.unit > 0:
            raise ValueError("horizon.unit must be positive")
        if hz.breakpoints is not None:
            bp = list(hz.breakpoints)
            if len(bp) < 2 or abs(bp[0]) > 1e-12 or abs(bp[-1] - hz.window) > 1e-9 * max(1.0, hz.window) or any(b2 <= b1 for b1, b2 in zip(bp, bp[1:])):
                raise ValueError(f"horizon.breakpoints {bp} must increase from 0 to horizon.window ({hz.window})")
        if hz.nodes < 2:
            raise ValueError("horizon.nodes must be at least 2")
        if not hz.window > 0:
            raise ValueError("horizon.window must be positive")
        if hz.discount < 0:
            raise ValueError("horizon.discount must be non-negative")
        if self.horizon.kind not in ("stationary", "finite", "finite_cells", "transition"):
            raise ValueError("horizon.kind must be 'stationary', 'finite' (spectral triangle), 'finite_cells' or 'transition'")

    def _check_transition(self) -> None:
        """The transition blocks: kind 'transition' requires a past block (a `model`, a list of `initial` shocks,
        or both), its continuation is 'stationary' or 'end', its `stationary` block has a positive window and
        at least 2 nodes; the other kinds refuse all three blocks (a keyword past goes to solve(past=))."""
        hz = self.horizon
        if hz.kind != "transition":
            for k in ("past", "continuation", "stationary"):
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
        if hz.stationary is not None:
            self._check_keys("horizon.stationary", hz.stationary, {"window", "nodes"})
            if hz.stationary.get("window") is not None and not hz.stationary["window"] > 0:
                raise ValueError("horizon.stationary.window must be positive")
            n = hz.stationary.get("nodes")
            if n is not None and (n != int(n) or n < 2):
                raise ValueError("horizon.stationary.nodes must be an integer of at least 2")

    # ------------------------------------------------------- construction
    _KEYS = {
        "model": {"name", "params", "channels", "states", "definitions", "agents", "ties", "horizon"},
        "horizon": {"kind", "discount", "window", "breakpoints", "unit", "unit_range", "nodes", "past", "continuation", "stationary"},
        "state": {"drift", "noise", "initial"},
        "agent": {"controls", "signals", "loss", "myopic"},
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
        horizon = {"kind": hz.kind, "discount": hz.discount, "window": hz.window, "nodes": hz.nodes}
        for k in ("breakpoints", "unit", "unit_range", "past", "continuation", "stationary"):
            if getattr(hz, k) is not None:
                horizon[k] = copy.deepcopy(getattr(hz, k))
        if self.source is not None and not numeric:
            # the source (parameter expressions intact) with the live horizon: horizon fields may be changed
            # on the object (the engines read them at compile time); parameters may not (coefficients are
            # numbers once built), so params come from the source and with_params() makes a new model
            d = copy.deepcopy(self.source)
            hsrc = d.get("horizon") or {}
            seen: Dict[str, float] = {}
            for k, v in (d.get("params") or {}).items():
                seen[k] = eval_coef(v, seen)
            hout = {}
            for k, v in horizon.items():
                if k == "past" and k in hsrc and _eval_past_block(copy.deepcopy(hsrc[k]), seen) == v:
                    hout[k] = hsrc[k]; continue           # the past block with its expressions (the loadings evaluated agree)
                if k in hsrc and (hsrc[k] == v or (isinstance(hsrc[k], str) and abs(eval_coef(hsrc[k], seen) - v) <= 1e-12 * max(1.0, abs(v)))
                                  or (isinstance(v, float) and not isinstance(hsrc[k], (list, str, bool)) and abs(float(hsrc[k]) - v) <= 1e-12 * max(1.0, abs(v)))):
                    hout[k] = hsrc[k]                     # unchanged: keep the source's spelling (an expression)
                else:
                    hout[k] = v
            d["horizon"] = hout
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
             "agents": {}, "ties": [list(g) for g in self.ties], "horizon": horizon}
        for a in self.agents:
            d["agents"][a.name] = {"controls": list(a.controls), "myopic": a.myopic,
                                   "signals": {r.name: {"drift": ex(r.drift), "noise": ex(r.noise), "delay": r.delay} for r in a.signals},
                                   "loss": [[float(t[0])] + [atom(x) for x in t[1:]] for t in a.loss]}
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
        return Model.from_dict(d)

    def with_horizon(self, **fields) -> "Model":
        """A new model with these horizon fields (nodes, window, discount, kind, breakpoints, unit, unit_range)."""
        d = self.to_dict(); d.setdefault("horizon", {})
        bad = sorted(set(fields) - self._KEYS["horizon"])
        if bad:
            raise ValueError(f"unknown horizon field(s) {bad}")
        d["horizon"].update(fields)
        return Model.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict, base_dir: Optional[str] = None) -> "Model":
        """A Model from the file structure.  base_dir: the directory a relative path in horizon.past.model is
        resolved against (the model file's own directory when loaded from a file); None leaves it as given."""
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
        horizon = Horizon(kind=hz.get("kind", "stationary"),
                          discount=eval_coef(hz.get("discount", 0.0), params),
                          window=eval_coef(hz.get("window", 8.0), params),
                          breakpoints=[eval_coef(b, params) for b in hz["breakpoints"]] if hz.get("breakpoints") else None,
                          unit=eval_coef(hz["unit"], params) if hz.get("unit") is not None else None,
                          unit_range=eval_coef(hz["unit_range"], params) if hz.get("unit_range") is not None else None,
                          nodes=hz.get("nodes", 16),
                          past=copy.deepcopy(hz["past"]) if hz.get("past") is not None else None,
                          continuation=hz.get("continuation"),
                          stationary={k: eval_coef(v, params) if k == "window" else v for k, v in hz["stationary"].items()}
                          if isinstance(hz.get("stationary"), dict) else hz.get("stationary"))
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
                                myopic=v.get("myopic", False)))
        from types import MappingProxyType
        m = cls(name=d.get("name", "model"), channels=list(d.get("channels") or []), states=states,
                agents=agents, horizon=horizon, definitions=defs, ties=[list(g) for g in (d.get("ties") or [])],
                params=MappingProxyType(params), source=copy.deepcopy(d))     # read-only: see with_params()
        m.validate()                                           # structural errors first (its expansions also record
        m.horizon.nodes = int(m.horizon.nodes)                 # the lag parameters, 'P@tau'); then the parameter check
        for k, v in pdict.items():
            if isinstance(v, str):
                safe_eval(v, params)                           # a parameter used inside another one counts as used
        unused = sorted(set(params) - params.used)
        if unused:
            raise ValueError(f"parameter(s) {unused} are defined but never used in the model (misspelled somewhere?)")
        return m


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


# --------------------------------------------------------------- builder
class ModelBuilder:
    """Fluent Python interface producing the same structure as the YAML file."""

    def __init__(self, name: str = "model", **params):
        self.d = {"name": name, "params": dict(params), "channels": [], "states": {}, "definitions": {},
                  "agents": {}, "ties": [], "horizon": {}}

    def param(self, **kw):
        self.d["params"].update(kw); return self

    def channel(self, *names):
        self.d["channels"].extend(names); return self

    def state(self, name, drift=None, noise=None, initial=None):
        self.d["states"][name] = {"drift": drift or {}, "noise": noise or {}, **({"initial": initial} if initial is not None else {})}; return self

    def define(self, name, expr):
        self.d["definitions"][name] = expr; return self

    def agent(self, name, controls, loss, myopic=False):
        self.d["agents"][name] = {"controls": list(controls), "signals": {}, "loss": list(loss), "myopic": myopic}
        return self

    def signal(self, agent, name, drift=None, noise=None, delay=0.0):
        self.d["agents"][agent]["signals"][name] = {"drift": drift or {}, "noise": noise or {}, "delay": delay}
        return self

    def tie(self, *agents):
        self.d["ties"].append(list(agents)); return self

    def stationary(self, discount=0.0, window=8.0, nodes=16, breakpoints=None, unit=None, unit_range=None):
        self.d["horizon"] = {"kind": "stationary", "discount": discount, "window": window, "nodes": nodes,
                             "breakpoints": breakpoints, "unit": unit, "unit_range": unit_range}
        return self

    def finite(self, T=1.0, nodes=16, discount=0.0):
        self.d["horizon"] = {"kind": "finite", "window": T, "nodes": nodes, "discount": discount}; return self

    def transition(self, T=1.0, nodes=12, past=None, continuation="stationary", discount=0.0, stationary=None, unit=None):
        """A transition on [0, T] from `past` (a path to the old stationary model file, its dict, a Model or
        ModelBuilder (their dict is inlined), or a list of initial shocks) continued by 'stationary' (the new
        model's stationary equilibrium, sized by `stationary` = {"window", "nodes"}) or ending at T ('end')."""
        if isinstance(past, ModelBuilder):
            past = past.to_dict()
        elif isinstance(past, Model):
            past = past.to_dict()
        block = {"initial": list(past)} if isinstance(past, (list, tuple)) else {"model": past}
        self.d["horizon"] = {"kind": "transition", "window": T, "nodes": nodes, "discount": discount, "past": block,
                             "continuation": continuation, **({"stationary": dict(stationary)} if stationary else {}),
                             **({"unit": unit} if unit is not None else {})}
        return self

    def build(self) -> Model:
        return Model.from_dict(self.d)

    def to_dict(self) -> dict:
        import copy
        return copy.deepcopy(self.d)          # a copy: mutating it must not alter the builder
