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
E int e^{-rho t} loss_t dt (rho = 0 is average cost).
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

Number = Union[int, float, str]
Atom = Tuple[str, float]          # (primary name, lag); lag > 0 past, < 0 future
Expr = Dict[Atom, float]          # linear combination of primary atoms


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
    kind: str = "stationary"          # "stationary" | "finite" (spectral triangle) | "finite_cells"
    discount: float = 0.0
    window: float = 8.0               # L for stationary; T for finite
    breakpoints: Optional[List[float]] = None
    unit: Optional[float] = None
    unit_range: Optional[float] = None
    nodes: int = 16


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
        """Expand an expression over atom strings into primary atoms (states, controls)."""
        defs = {d.name: d.expr for d in self.definitions}
        out: Expr = {}

        def add(name: str, lag: float, coef: float, depth: int):
            if depth > 50:
                raise ValueError(f"definition cycle involving {name!r}")
            if name in defs:
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

        def cls(name):
            if name in own:
                return own[name]
            if name in others:
                return "other_control"
            if len(users.get(name, set())) <= 1:
                return "private_state"
            return "state:" + name

        def canon(expr):
            return tuple(sorted((cls(n), round(l, 9), round(c, 9)) for (n, l), c in expr.items()))
        rows = tuple((canon(self.expand(r.drift)), tuple(sorted(round(v, 9) for v in r.noise.values())), round(r.delay, 9))
                     for r in a.signals)
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
        if self.horizon.kind == "stationary":
            out.append("costs are stationary flow losses per unit time" + (" (the discount rate enters the best responses, not the reported cost)" if self.horizon.discount else ""))
        else:
            out.append("costs are discounted integrals over [0, T]")
        return out

    # --------------------------------------------------------- validation
    def validate(self) -> None:
        names = self.state_names + self.control_names + self.def_names
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate quantity names {dup}")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("duplicate channel names")
        agent_names = [a.name for a in self.agents]
        if len(set(agent_names)) != len(agent_names):
            raise ValueError("duplicate agent names")
        for s in self.states:
            for ch in s.noise:
                if ch not in self.channels:
                    raise ValueError(f"state {s.name}: unknown channel {ch}")
            self.expand(s.drift)
        for d in self.definitions:
            self.expand({d.name: 1.0})
        for a in self.agents:
            if not a.controls:
                raise ValueError(f"agent {a.name} has no controls")
            for r in a.signals:
                for ch in r.noise:
                    if ch not in self.channels:
                        raise ValueError(f"row {a.name}.{r.name}: unknown channel {ch}")
                if not r.noise:
                    raise ValueError(f"row {a.name}.{r.name} needs a noise loading (exact rows are not supported)")
                for (n, l) in self.expand(r.drift):
                    pass   # own controls may appear in own rows (the agent knows them; they drop out of its passive rows)
                    if l < 0:
                        raise ValueError(f"row {a.name}.{r.name} observes a future quantity {n}@{l}")
            for term in a.loss:
                if len(term) not in (2, 3):
                    raise ValueError(f"agent {a.name}: loss term {term} must be [coef, a] or [coef, a, b]")
                for atom in term[1:]:
                    self.expand({atom: 1.0})
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
        used = {ch for s in self.states for ch in s.noise} | {ch for a in self.agents for r in a.signals for ch in r.noise}
        unused = [ch for ch in self.channels if ch not in used]
        if unused:
            raise ValueError(f"channel(s) {unused} are never loaded by a state or a signal row (misspelled?)")
        for a in self.agents:
            atoms = {n for term in a.loss for atom in term[1:] for (n, l) in self.expand({atom: 1.0})}
            missing = [u for u in a.controls if u not in atoms]
            if missing:
                raise ValueError(f"agent {a.name}: control(s) {missing} do not enter its loss; the best response would be undetermined")
        hz = self.horizon
        if hz.nodes < 2:
            raise ValueError("horizon.nodes must be at least 2")
        if not hz.window > 0:
            raise ValueError("horizon.window must be positive")
        if hz.discount < 0:
            raise ValueError("horizon.discount must be non-negative")
        if self.horizon.kind not in ("stationary", "finite", "finite_cells"):
            raise ValueError("horizon.kind must be 'stationary', 'finite' (spectral triangle) or 'finite_cells'")

    # ------------------------------------------------------- construction
    _KEYS = {
        "model": {"name", "params", "channels", "states", "definitions", "agents", "ties", "horizon"},
        "horizon": {"kind", "discount", "window", "L", "T", "breakpoints", "unit", "unit_range", "nodes"},
        "state": {"drift", "noise"},
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
        if self.source is not None and not numeric:
            import copy
            return copy.deepcopy(self.source)
        d = {"name": self.name, "params": dict(self.params), "channels": list(self.channels),
             "states": {s.name: {"drift": dict(s.drift), "noise": dict(s.noise)} for s in self.states},
             "definitions": {x.name: dict(x.expr) for x in self.definitions},
             "agents": {}, "ties": [list(g) for g in self.ties],
             "horizon": {"kind": self.horizon.kind, "discount": self.horizon.discount, "window": self.horizon.window,
                         "nodes": self.horizon.nodes, "breakpoints": self.horizon.breakpoints, "unit": self.horizon.unit,
                         "unit_range": self.horizon.unit_range}}
        for a in self.agents:
            d["agents"][a.name] = {"controls": list(a.controls), "myopic": a.myopic,
                                   "signals": {r.name: {"drift": dict(r.drift), "noise": dict(r.noise), "delay": r.delay} for r in a.signals},
                                   "loss": [list(t) for t in a.loss]}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        cls._check_keys("model", d, cls._KEYS["model"])
        cls._check_keys("horizon", d.get("horizon") or {}, cls._KEYS["horizon"])
        for k, v in (d.get("states") or {}).items():
            cls._check_keys("state", v or {}, cls._KEYS["state"], f" '{k}'")
        for k, v in (d.get("agents") or {}).items():
            cls._check_keys("agent", v or {}, cls._KEYS["agent"], f" '{k}'")
            for rk, rv in (v.get("signals") or {}).items():
                cls._check_keys("signal", rv or {}, cls._KEYS["signal"], f" '{k}.{rk}'")
        params: Dict[str, float] = {}
        for k, v in (d.get("params") or {}).items():          # a parameter may be an expression in earlier ones
            params[k] = eval_coef(v, params)
        params = _Recording(params)                            # records which parameters the model references
        hz = d.get("horizon") or {}
        horizon = Horizon(kind=hz.get("kind", "stationary"),
                          discount=eval_coef(hz.get("discount", 0.0), params),
                          window=eval_coef(hz.get("window", hz.get("L", hz.get("T", 8.0))), params),
                          breakpoints=[eval_coef(b, params) for b in hz["breakpoints"]] if hz.get("breakpoints") else None,
                          unit=eval_coef(hz["unit"], params) if hz.get("unit") is not None else None,
                          unit_range=eval_coef(hz["unit_range"], params) if hz.get("unit_range") is not None else None,
                          nodes=int(hz.get("nodes", 16)))
        states = [State(name=k, drift=parse_expr(v.get("drift"), params), noise=parse_expr(v.get("noise"), params))
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
                                myopic=bool(v.get("myopic", False))))
        import copy
        # atoms ('P@tau') are parsed lazily, so scan them for lag parameters; a parameter used inside
        # another parameter's expression also counts as used
        for s in states:
            for atom in list(s.drift) + list(s.noise):
                parse_atom(atom, params)
        for df in defs:
            for atom in df.expr:
                parse_atom(atom, params)
        for a in agents:
            for r in a.signals:
                for atom in list(r.drift) + list(r.noise):
                    parse_atom(atom, params)
            for term in a.loss:
                for atom in term[1:]:
                    parse_atom(atom, params)
        for k, v in (d.get("params") or {}).items():
            if isinstance(v, str):
                safe_eval(v, params)
        unused = sorted(set(params) - params.used)
        if unused:
            raise ValueError(f"parameter(s) {unused} are defined but never used in the model (misspelled somewhere?)")
        m = cls(name=d.get("name", "model"), channels=list(d.get("channels") or []), states=states,
                agents=agents, horizon=horizon, definitions=defs, ties=[list(g) for g in (d.get("ties") or [])],
                params=dict(params), source=copy.deepcopy(d))
        m.validate()
        return m


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

    def state(self, name, drift=None, noise=None):
        self.d["states"][name] = {"drift": drift or {}, "noise": noise or {}}; return self

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

    def build(self) -> Model:
        return Model.from_dict(self.d)

    def to_dict(self) -> dict:
        import copy
        return copy.deepcopy(self.d)          # a copy: mutating it must not alter the builder
