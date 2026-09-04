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


def eval_coef(c: Number, params: Dict[str, float]) -> float:
    if isinstance(c, (int, float)):
        return float(c)
    ns = {k: v for k, v in params.items()}
    ns.update({n: getattr(math, n) for n in ("sqrt", "exp", "log", "pi", "e", "sin", "cos", "tanh")})
    try:
        return float(eval(str(c), {"__builtins__": {}}, ns))
    except Exception as exc:
        raise ValueError(f"cannot evaluate coefficient {c!r}: {exc}") from exc


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
    kind: str = "stationary"          # "stationary" | "finite"
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
            shapes = {(len(a.controls), len(a.signals)) for a in ag}
            if len(shapes) != 1:
                raise ValueError(f"tied agents {group} must have the same numbers of controls and signal rows")
        used = {ch for s in self.states for ch in s.noise} | {ch for a in self.agents for r in a.signals for ch in r.noise}
        unused = [ch for ch in self.channels if ch not in used]
        if unused:
            raise ValueError(f"channel(s) {unused} are never loaded by a state or a signal row (misspelled?)")
        for a in self.agents:
            atoms = {n for term in a.loss for atom in term[1:] for (n, l) in self.expand({atom: 1.0})}
            missing = [u for u in a.controls if u not in atoms]
            if missing:
                raise ValueError(f"agent {a.name}: control(s) {missing} do not enter its loss; the best response would be undetermined")
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

    def to_dict(self) -> dict:
        """The file structure of this model (coefficients numeric); from_dict(to_dict()) round-trips."""
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
        params = {k: float(v) for k, v in (d.get("params") or {}).items()}
        # allow parameters defined in terms of earlier ones
        for k, v in (d.get("params") or {}).items():
            if isinstance(v, str):
                params[k] = eval_coef(v, params)
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
        m = cls(name=d.get("name", "model"), channels=list(d.get("channels") or []), states=states,
                agents=agents, horizon=horizon, definitions=defs, ties=[list(g) for g in (d.get("ties") or [])],
                params=params)
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
        return self.d
