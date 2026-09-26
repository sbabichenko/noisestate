"""Models as equations: symbols whose arithmetic compiles to the model file's grammar (spec.py).

    import noisestate as ns
    from noisestate import dt, sqrt
    r, p, sigma = ns.params(r=0.1, p=3.0, sigma=1.0)
    dW0, dW1 = ns.shocks(2)                                     # named W0, W1
    X = ns.State("X"); D = ns.Control("D")
    X.d = D * dt + sigma * dW0                                  # drift {D: 1}, noise {W0: "sigma"}
    me = ns.Agent("me", controls=D, observes=sqrt(p) * X * dt + dW1, loss=X**2 + r * D**2)
    game = ns.Game(states=X, agents=me, window=8.0)
    game.to_dict()                                              # the model file, exactly

Three layers.  A *coefficient* is a number or an expression in Params (`Coef`), rendered to the string the
evaluator in spec.py reads ("sqrt(p)", "0.5*r", "-2*theta").  A *linear expression* (`Linear`) is a sum of
coefficient x atom, an atom being a quantity (a State, a Control, a define()d quantity, each possibly
lagged: `X.lag(0.5)` is the atom "X@0.5"), a shock, or the constant 1.  A *differential* is a linear
expression times `dt` plus shock terms; assigned to `X.d`, or observed, its dt part is the drift dict, its
shock terms the noise dict and its constant `const`.
A *quadratic expression* (`Quad`) is a product of two linear expressions, or `expr**2`, plus linear terms; an
Agent's loss compiles it to the term list [coef, a, b] / [coef, a]: the loss is the literal sum of the
terms, so the cross term c * a * b is the one term [c, a, b] (and (X - theta)**2 is [1, X, X], [-2*theta, X]
and the constant theta**2, which moves no strategy but is part of the cost: the agent's `constant`).
Everything the grammar refuses (a lead in a drift, an unused channel, a control outside its owner's loss)
is still refused by spec.Model.from_dict, which every expression model is built through.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from . import spec as _spec

Number = Union[int, float]

# --------------------------------------------------------------------------------------------- coefficients

_counter = itertools.count()


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num_str(v: Number) -> str:
    if isinstance(v, int):
        return str(v)
    if v == int(v) and abs(v) < 1e15:
        return f"{v:.1f}"
    return repr(v)


class Coef:
    """A coefficient expression in Params: rendered by str() to the grammar spec.safe_eval reads, evaluated
    by value(params).  Built by arithmetic on Params and numbers; every operation on two numbers stays a
    number (Python folds it), so a Coef always involves at least one Param."""
    __slots__ = ("op", "args")
    _PREC = {"num": 5, "sym": 5, "call": 5, "pow": 4, "neg": 3, "mul": 2, "div": 2, "add": 1, "sub": 1}

    def __init__(self, op: str, *args):
        self.op = op; self.args = args

    # -- rendering
    def _render(self) -> Tuple[str, int]:
        op = self.op
        if op == "num":
            v = self.args[0]
            return (_num_str(v), 5 if v >= 0 else 3)
        if op == "sym":
            return (self.args[0].name, 5)
        if op == "call":
            return (f"{self.args[0]}({', '.join(_render(a)[0] for a in self.args[1:])})", 5)
        if op == "neg":
            s, p = _render(self.args[0])
            return ("-" + (f"({s})" if p < 2 else s), 3)          # -b*r needs no brackets, -(a + b) does
        a, b = self.args
        if op == "pow":
            if _is_number(b) and b == 0.5:
                return (f"sqrt({_render(a)[0]})", 5)
            sa, pa = _render(a); sb, pb = _render(b)
            return ((f"({sa})" if pa <= 4 else sa) + "**" + (f"({sb})" if pb < 4 else sb), 4)
        sa, pa = _render(a); sb, pb = _render(b)
        if op in ("mul", "div"):
            left = f"({sa})" if pa < 2 else sa
            right = f"({sb})" if (pb < 2 or (op == "div" and pb == 2) or pb == 3) else sb
            return (left + ("*" if op == "mul" else "/") + right, 2)
        left = f"({sa})" if pa < 1 else sa
        right = f"({sb})" if (pb < 1 or (op == "sub" and pb == 1) or pb == 3) else sb
        return (left + (" + " if op == "add" else " - ") + right, 1)

    def __str__(self) -> str:
        return self._render()[0]

    def __repr__(self) -> str:
        return f"Coef({self})"

    def value(self, params: Dict[str, float]) -> float:
        """Evaluate against parameter values (spec.safe_eval of str(self) gives the same number)."""
        op = self.op
        if op == "num":
            return float(self.args[0])
        if op == "sym":
            name = self.args[0].name
            if name not in params:
                raise ValueError(f"unknown parameter {name!r} in coefficient {self}")
            return float(params[name])
        if op == "call":
            return float(_spec._FUNCS[self.args[0]](*[_value(a, params) for a in self.args[1:]]))
        if op == "neg":
            return -_value(self.args[0], params)
        a, b = (_value(x, params) for x in self.args)
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b if b != 0 else math.inf, "pow": a ** b}[op]

    def params(self) -> List["Param"]:
        """The Params this coefficient uses, in order of first appearance."""
        out: List[Param] = []
        if self.op == "sym":
            out.append(self.args[0])
        else:
            for a in self.args:
                if isinstance(a, Coef):
                    for p in a.params():
                        if p not in out:
                            out.append(p)
        return out

    # -- arithmetic (a Coef with a linear expression is that expression scaled: Linear handles it)
    def _bin(self, op, other, rev=False):
        if isinstance(other, Linear):
            return NotImplemented
        if not (_is_number(other) or isinstance(other, Coef)):
            return NotImplemented
        a, b = (other, self) if rev else (self, other)
        if op == "mul":
            # keep products readable: (-a)(-b) is ab, (-a)b is -(ab), and a*a is a**2 ((X - b)**2's constant is b**2)
            na = isinstance(a, Coef) and a.op == "neg"; nb = isinstance(b, Coef) and b.op == "neg"
            if na or nb:
                prod = (a.args[0] if na else a) * (b.args[0] if nb else b)
                return prod if na and nb else -prod
            if isinstance(a, Coef) and isinstance(b, Coef) and str(a) == str(b):
                return Coef("pow", a, 2)
            # a number inside a product comes to the front: q*(-2*b) is -2*q*b
            for x, y in ((a, b), (b, a)):
                if isinstance(y, Coef) and y.op == "mul" and _is_number(y.args[0]) and not _is_number(x):
                    return _times(y.args[0], x * y.args[1])
        return Coef(op, a, b)

    def __add__(self, o): return self._bin("add", o)
    def __radd__(self, o): return self._bin("add", o, True)
    def __sub__(self, o): return self._bin("sub", o)
    def __rsub__(self, o): return self._bin("sub", o, True)
    def __mul__(self, o): return self._bin("mul", o)
    def __rmul__(self, o): return self._bin("mul", o, True)
    def __truediv__(self, o): return self._bin("div", o)
    def __rtruediv__(self, o): return self._bin("div", o, True)
    def __pow__(self, o): return self._bin("pow", o)
    def __rpow__(self, o): return self._bin("pow", o, True)
    def __neg__(self): return Coef("neg", self)
    def __pos__(self): return self
    def __abs__(self): return Coef("call", "abs", self)
    __array_ufunc__ = None            # numpy scalars defer to these operators


def _render(x) -> Tuple[str, int]:
    return x._render() if isinstance(x, Coef) else Coef("num", x)._render()


def _value(x, params) -> float:
    return x.value(params) if isinstance(x, Coef) else float(x)


def _coef_str(c) -> Union[Number, str]:
    """A coefficient as the model file writes it: a number stays a number, a Coef is its string."""
    return str(c) if isinstance(c, Coef) else c


def _fn(name):
    def f(x):
        if isinstance(x, Coef):
            return Coef("call", name, x)
        return float(_spec._FUNCS[name](x))
    f.__name__ = name; f.__doc__ = f"{name}() of a Param expression (a number when given a number)."
    return f


sqrt, exp, log, sin, cos, tanh = (_fn(n) for n in ("sqrt", "exp", "log", "sin", "cos", "tanh"))


class Param(Coef):
    """A named parameter with a value (a number, or None until Model(params=...) supplies it); arithmetic on
    it builds a Coef.  `Param.many(a=1.0, b=2.0)` returns Params in the given order."""
    __slots__ = ("name", "given", "order")

    def __init__(self, name: str, value: Optional[Number] = None):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"a parameter name must be an identifier, not {name!r}")
        if value is not None and not _is_number(value):
            raise ValueError(f"Param {name}: the value must be a number, not {value!r}")
        super().__init__("sym", self)
        self.name = name; self.given = value; self.order = next(_counter)

    @staticmethod
    def many(**values) -> Tuple["Param", ...]:
        return tuple(Param(k, v) for k, v in values.items())

    def __repr__(self) -> str:
        return f"Param({self.name!r}, {self.given!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other


# --------------------------------------------------------------------------------------------------- atoms

@dataclass(frozen=True)
class _Atom:
    """A term's atom: kind "q" (a quantity `name`, at `lag`: a number, a Param, or None for now), "w" (a shock
    of `name`), or "1" (the constant)."""
    kind: str
    name: str = ""
    lag: object = None

    def key(self) -> str:
        if self.kind == "1":
            return _spec.CONST
        if self.kind == "w" or self.lag is None:
            return self.name
        return f"{self.name}@{self.lag.name if isinstance(self.lag, Param) else format(float(self.lag), 'g')}"

    def lagged(self, tau, sign: int) -> "_Atom":
        if self.kind != "q":
            what = "the constant" if self.kind == "1" else f"shock {self.name!r}"
            raise ValueError(f"cannot lag {what}: only a state, a control or a definition has a lagged value")
        if isinstance(tau, Param):
            if sign < 0:
                raise ValueError(f"{self.name}.lead({tau.name}): a lead by a Param is not supported; give a number")
            lag = tau
        elif _is_number(tau) and math.isfinite(tau):
            lag = sign * float(tau)
        else:
            raise ValueError(f"the lag of {self.name!r} must be a number or a Param, not {tau!r}")
        if self.lag not in (None, 0.0):
            raise ValueError(f"{self.name!r} is already lagged ({self.key()}); lag the quantity itself")
        return _Atom("q", self.name, None if lag == 0 else lag)


class Shock:
    """One Brownian channel; `shocks("w0", "w1")` makes a namespace of them."""
    __slots__ = ("name", "space", "index")

    def __init__(self, name: str, space: "_Shocks", index: int):
        self.name = name; self.space = space; self.index = index

    def _linear(self) -> "Linear":
        return Linear({_Atom("w", self.name): 1}, shocks={self.name: self})

    def lag(self, tau):
        raise ValueError(f"shock {self.name!r} cannot be lagged: lag the quantity it drives instead")

    lead = lag

    def __repr__(self) -> str:
        return f"Shock({self.name!r})"

    def __add__(self, o): return self._linear() + o
    def __radd__(self, o): return o + self._linear()
    def __sub__(self, o): return self._linear() - o
    def __rsub__(self, o): return o - self._linear()
    def __mul__(self, o): return self._linear() * o
    def __rmul__(self, o): return o * self._linear()
    def __truediv__(self, o): return self._linear() / o
    def __neg__(self): return -self._linear()
    def __pos__(self): return self._linear()
    def __pow__(self, o): return self._linear() ** o
    __array_ufunc__ = None


class _Shocks:
    """The namespace shocks() returns: attribute access to its shocks, in order."""

    def __init__(self, names: Sequence[str]):
        if len(set(names)) != len(names):
            raise ValueError(f"shocks(): duplicate channel names in {list(names)}")
        for n in names:
            if not isinstance(n, str) or not n.isidentifier():
                raise ValueError(f"shocks(): a channel name must be an identifier, not {n!r}")
        self.order = next(_counter)
        self.names = list(names)
        self._by_name = {n: Shock(n, self, i) for i, n in enumerate(names)}

    def __getattr__(self, name: str) -> Shock:
        try:
            return self.__dict__["_by_name"][name]
        except KeyError:
            raise AttributeError(f"no shock {name!r}; the shocks are {self.names}") from None

    def __getitem__(self, name: str) -> Shock:
        return getattr(self, name)

    def __iter__(self):
        return iter(self._by_name[n] for n in self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"shocks({', '.join(map(repr, self.names))})"


def shocks(*names) -> _Shocks:
    """The Brownian shocks: `dW0, dW1, dW2 = shocks(3)` (named W0, W1, W2), `shocks("W0 V")` (names in one
    string), or `w = shocks("w0", "w1"); w.w0` (a namespace, which also unpacks).  The model's shocks are the ones
    it uses, in this order."""
    if len(names) == 1 and isinstance(names[0], int) and not isinstance(names[0], bool):
        names = tuple(f"W{i}" for i in range(names[0]))
    elif len(names) == 1 and isinstance(names[0], str) and (" " in names[0].strip() or "," in names[0]):
        names = tuple(names[0].replace(",", " ").split())
    return _Shocks(names)


def params(**values) -> Tuple["Param", ...]:
    """Parameters with their values, in order: `p, r = params(p=3, r=0.1)` (Param.many)."""
    return Param.many(**values)


def Game(states, agents, *, T=None, window=None, discount=0.0, horizon=None, definitions=None, ties=None,
         name: str = "game", nodes=None, numerics=None, params=None):
    """A model from its equations, the Python form's one constructor: states (a State or a list), agents, and the
    horizon, either T (a finite game on [0, T]), or window (a stationary game, its kernels cut at that lag), or an
    explicit horizon (Finite, Stationary, Transition); discount the rate on future losses.  `nodes` (or numerics)
    sets the grid; `params` lists the Params in the order the file should write them (default: as first used).
    Returns a Model: game.solve() solves it."""
    from .spec import Model
    given = [x for x in (T, window, horizon) if x is not None]
    if len(given) != 1:
        raise ValueError("Game(): give exactly one of T (finite), window (stationary) or horizon")
    if horizon is None:
        horizon = Finite(T=T, discount=discount) if T is not None else Stationary(window=window, discount=discount)
    if nodes is not None:
        numerics = dict(numerics or {}, nodes=nodes)
    states = [states] if isinstance(states, State) else list(states)
    agents = [agents] if isinstance(agents, Agent) else list(agents)
    if not is_expression_form(states, agents, horizon, definitions):
        raise TypeError("Game() takes the Python form's objects (ns.State, ns.Agent, ...); a model from its file "
                        "structure is Model.from_dict(d) or ns.load(path)")
    built = Model.from_dict(compile_model(name, states, agents, definitions=definitions, ties=ties, horizon=horizon,
                                          numerics=numerics, params=params))
    built.source = built.to_dict()                  # the normalised file (as load(save()) reads it back)
    return built


# --------------------------------------------------------------------------------------- linear expressions

def _scalar(x) -> bool:
    return _is_number(x) or isinstance(x, Coef)


def _times(a, b):
    """The product of two coefficients, numbers folded, 1 dropped."""
    if _is_number(a) and _is_number(b):
        return a * b
    if (_is_number(a) and a == 0) or (_is_number(b) and b == 0):
        return 0
    if _is_number(a) and a == 1:
        return b
    if _is_number(b) and b == 1:
        return a
    if _is_number(b) and not _is_number(a):
        a, b = b, a
    if _is_number(a):                        # a number times -(x) is (-number)*x, times (number*x) folds the numbers
        if b.op == "neg":
            return _times(-a, b.args[0])
        if b.op == "mul" and _is_number(b.args[0]):
            return _times(a * b.args[0], b.args[1])
        return Coef("mul", a, b)
    return a * b                             # two expressions: Coef's product folds signs and squares (-b)(-b) = b**2


def _plus(a, b):
    if _is_number(a) and _is_number(b):
        return a + b
    if _is_number(a) and a == 0:
        return b
    if _is_number(b) and b == 0:
        return a
    if isinstance(a, Coef) and isinstance(b, Coef) and str(a) == str(b):      # the same term twice: 2 x
        return _times(2, a)
    return Coef("add", a, b)


def _negate(a):
    """Minus a coefficient: a number negated, -(2*c) written -2*c, -(-x) written x."""
    if _is_number(a):
        return -a
    if a.op == "neg":
        return a.args[0]
    if a.op == "mul" and _is_number(a.args[0]):
        return Coef("mul", -a.args[0], a.args[1])
    return Coef("neg", a)


class Linear:
    """A linear expression: {atom: coefficient}, with the quantities and shocks it refers to."""

    def __init__(self, terms: Dict[_Atom, object], quantities: Optional[dict] = None, shocks: Optional[dict] = None):
        self.terms = dict(terms)
        self.quantities: Dict[str, Quantity] = dict(quantities or {})   # name -> the object (for definitions and names)
        self.shocks: Dict[str, Shock] = dict(shocks or {})

    # -- construction
    @staticmethod
    def of(x) -> "Linear":
        if isinstance(x, Linear):
            return x
        if isinstance(x, Shock):
            return x._linear()
        if _scalar(x):
            return Linear({_Atom("1"): x}) if not (_is_number(x) and x == 0) else Linear({})
        raise TypeError(f"cannot use {x!r} in a linear expression")

    def _merge(self, other: "Linear", sign: int) -> "Linear":
        terms = dict(self.terms)
        for a, c in other.terms.items():
            c = c if sign > 0 else _negate(c)
            terms[a] = _plus(terms[a], c) if a in terms else c
        return Linear(terms, {**self.quantities, **other.quantities}, {**self.shocks, **other.shocks})

    def scaled(self, k) -> "Linear":
        return Linear({a: _times(k, c) for a, c in self.terms.items()}, self.quantities, self.shocks)

    def lag(self, tau) -> "Linear":
        """The expression `tau` earlier (every atom lagged; a number or a Param)."""
        return Linear({a.lagged(tau, +1): c for a, c in self.terms.items()}, self.quantities, self.shocks)

    def lead(self, tau) -> "Linear":
        """The expression `tau` later (a lead; allowed only in a loss cross term with the agent's own control)."""
        return Linear({a.lagged(tau, -1): c for a, c in self.terms.items()}, self.quantities, self.shocks)

    # -- operators
    def __add__(self, o):
        if isinstance(o, Quad):
            return o + self
        try:
            return self._merge(Linear.of(o), +1)
        except TypeError:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Quad):
            return (-o) + self
        try:
            return self._merge(Linear.of(o), -1)
        except TypeError:
            return NotImplemented

    def __rsub__(self, o):
        try:
            return Linear.of(o)._merge(self, -1)
        except TypeError:
            return NotImplemented

    def __neg__(self):
        return self.scaled(-1)

    def __pos__(self):
        return self

    def __mul__(self, o):
        if _scalar(o):
            return self.scaled(o)
        if isinstance(o, (Linear, Shock)):
            return Quad.product(self, Linear.of(o))
        if isinstance(o, Quad):
            raise ValueError(f"({self}) * ({o}) is cubic: a loss is at most quadratic")
        return NotImplemented

    def __rmul__(self, o):
        if _scalar(o):
            return self.scaled(o)
        return NotImplemented

    def __truediv__(self, o):
        if _scalar(o):
            return self.scaled(1 / o if _is_number(o) else Coef("div", 1, o))
        return NotImplemented

    def __pow__(self, n):
        if _is_number(n) and n == 2:
            return Quad.product(self, self)
        if _is_number(n) and n == 1:
            return self
        raise ValueError(f"({self})**{n!r}: only the square of a linear expression is quadratic")

    __array_ufunc__ = None

    # -- compilation
    def split(self, what: str) -> Tuple[dict, dict, object]:
        """(drift, noise, constant) in the grammar's spelling; `what` names the object for the errors."""
        drift: Dict[str, object] = {}; noise: Dict[str, object] = {}; const = 0
        for a, c in self.terms.items():
            if _is_number(c) and c == 0:
                continue
            if a.kind == "q":
                drift[a.key()] = _coef_str(c)
            elif a.kind == "w":
                noise[a.name] = _coef_str(c)
            else:
                const = c
        return drift, noise, const

    def to_expr(self, what: str) -> dict:
        """The {atom: coef} dict of a definition or a drift with its constant under `const`."""
        drift, noise, const = self.split(what)
        if noise:
            raise ValueError(f"{what}: a shock ({', '.join(noise)}) has no place here; shocks drive states and signals")
        if not (_is_number(const) and const == 0):
            drift[_spec.CONST] = _coef_str(const)
        return drift

    def atoms_str(self) -> str:
        parts = []
        for a, c in self.terms.items():
            cs, prec = _render(c)
            parts.append(("" if (_is_number(c) and c == 1) else f"({cs})*" if prec < 2 else cs + "*") + a.key())
        return " + ".join(parts) if parts else "0"

    def __repr__(self) -> str:
        return self.atoms_str()

    def __str__(self) -> str:
        return self.atoms_str()


class Quantity(Linear):
    """A named quantity: a State, a Control, or a define()d definition.  As an expression it is itself."""
    role = "quantity"

    def __init__(self, name: str):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"a {self.role} name must be an identifier, not {name!r}")
        if name == _spec.CONST:
            raise ValueError(f"{name!r} is reserved for a constant; name the {self.role} otherwise")
        super().__init__({_Atom("q", name): 1}, quantities={name: self})
        self.name = name
        self.order = next(_counter)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other


class State(Quantity):
    """A state: `X = State("X"); X.d = D * dt + sigma * dW0` (its differential: the dt terms are the drift, a
    constant among them under const, and the shock terms the noise); `initial` moves the mean on a finite horizon."""
    role = "state"

    def __init__(self, name: str, initial: Optional[Number] = None):
        super().__init__(name)
        self.initial = initial
        self._drift = None                  # the differential's drift and noise as one linear expression

    @property
    def d(self) -> "Differential":
        """The state's differential: `X.d = (D1 + D2) * dt + sigma * dW0` (drift terms with dt, shocks without)."""
        L = self._drift or Linear({})
        drift = Linear({a: c for a, c in L.terms.items() if a.kind != "w"}, L.quantities)
        noise = Linear({a: c for a, c in L.terms.items() if a.kind == "w"}, {}, L.shocks)
        return Differential(drift, noise)

    @d.setter
    def d(self, expr) -> None:
        self._drift = Differential.of(expr).combined()

    def compile(self) -> dict:
        drift, noise, const = (self._drift or Linear({})).split(f"state {self.name}")
        if not (_is_number(const) and const == 0):
            drift[_spec.CONST] = _coef_str(const)
        out = {"drift": drift, "noise": noise}
        if self.initial is not None:
            out["initial"] = _coef_str(self.initial)
        return out


class Control(Quantity):
    """A control: `D = Control("D")`; it belongs to the Agent whose controls list it."""
    role = "control"


class Definition(Quantity):
    """A named linear combination (`define(name, expr)`), usable wherever a quantity is."""
    role = "definition"

    def __init__(self, name: str, expr):
        super().__init__(name)
        if isinstance(expr, Quad):
            raise ValueError(f"definition {name}: must be linear, not {expr}")
        self.expr = Linear.of(expr)
        if _Atom("1") in self.expr.terms:
            raise ValueError(f"definition {name}: a constant is allowed in a state's drift only")

    def compile(self) -> dict:
        return self.expr.to_expr(f"definition {self.name}")


def define(name: str, expr) -> Definition:
    """A definition: a named linear expression (`Pidx = define("Pidx", (P0.lag(tau) + P1.lag(tau)) / 2)`)."""
    return Definition(name, expr)


# ------------------------------------------------------------------------------------ quadratic expressions

class Quad:
    """A quadratic expression: {(atom, atom) or (atom,): coef} in the order the terms were written, plus a
    constant.  A product's cross term keeps the order its factors were written in ((a, b) from a * b); a term
    met again in either order is added to the first."""

    def __init__(self, terms: dict, const, quantities: dict):
        self.terms = dict(terms); self.const = const; self.quantities = dict(quantities)

    @staticmethod
    def _add(terms: dict, key: tuple, c) -> None:
        if len(key) == 2 and key not in terms and (key[1], key[0]) in terms:
            key = (key[1], key[0])
        terms[key] = _plus(terms[key], c) if key in terms else c

    @staticmethod
    def product(x: Linear, y: Linear) -> "Quad":
        if x.shocks or y.shocks:
            names = sorted(set(x.shocks) | set(y.shocks))
            raise ValueError(f"({x}) * ({y}): a shock ({', '.join(names)}) cannot enter a loss; a loss is quadratic in states and controls")
        terms: dict = {}; const = 0
        for a, ca in x.terms.items():
            for b, cb in y.terms.items():
                c = _times(ca, cb)
                if a.kind == "1" and b.kind == "1":
                    const = _plus(const, c)
                elif a.kind == "1" or b.kind == "1":
                    Quad._add(terms, (b if a.kind == "1" else a,), c)
                else:
                    Quad._add(terms, (a, b), c)
        return Quad(terms, const, {**x.quantities, **y.quantities})

    def _merge(self, other: "Quad", sign: int) -> "Quad":
        terms = dict(self.terms)
        for key, c in other.terms.items():
            Quad._add(terms, key, c if sign > 0 else _negate(c))
        const = _plus(self.const, other.const if sign > 0 else _negate(other.const))
        return Quad(terms, const, {**self.quantities, **other.quantities})

    @staticmethod
    def of(x) -> "Quad":
        if isinstance(x, Quad):
            return x
        if isinstance(x, (Linear, Shock)) or _scalar(x):
            l = Linear.of(x)
            if l.shocks:
                raise ValueError(f"{l}: a shock cannot enter a loss")
            terms = {(a,): c for a, c in l.terms.items() if a.kind == "q"}
            return Quad(terms, l.terms.get(_Atom("1"), 0), l.quantities)
        raise TypeError(f"cannot use {x!r} in a quadratic expression")

    def scaled(self, k) -> "Quad":
        return Quad({key: _times(k, c) for key, c in self.terms.items()}, _times(k, self.const), self.quantities)

    def __add__(self, o):
        try:
            return self._merge(Quad.of(o), +1)
        except TypeError:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, o):
        try:
            return self._merge(Quad.of(o), -1)
        except TypeError:
            return NotImplemented

    def __rsub__(self, o):
        try:
            return Quad.of(o)._merge(self, -1)
        except TypeError:
            return NotImplemented

    def __neg__(self):
        return self.scaled(-1)

    def __pos__(self):
        return self

    def __mul__(self, o):
        if _scalar(o):
            return self.scaled(o)
        if isinstance(o, (Linear, Shock)):
            raise ValueError(f"({self}) * ({o}) is cubic: a loss is at most quadratic in the quantities")
        if isinstance(o, Quad):
            raise ValueError(f"({self}) * ({o}) is not quadratic: a loss is at most quadratic in the quantities")
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, o):
        if _scalar(o):
            return self.scaled(1 / o if _is_number(o) else Coef("div", 1, o))
        return NotImplemented

    def __pow__(self, n):
        raise ValueError(f"({self})**{n!r} is not quadratic: a loss is at most quadratic in the quantities")

    __array_ufunc__ = None

    def compile(self, what: str) -> Tuple[List[list], object]:
        """(the loss term list [coef, a, b] / [coef, a] in the order written, the dropped constant)."""
        terms = [[_coef_str(c)] + [a.key() for a in key] for key, c in self.terms.items() if not (_is_number(c) and c == 0)]
        return terms, self.const

    def __repr__(self) -> str:
        parts = ["*".join([_render(c)[0]] + [a.key() for a in key]) for key, c in self.terms.items()]
        if not (_is_number(self.const) and self.const == 0):
            parts.append(_render(self.const)[0])
        return " + ".join(parts) if parts else "0"

    __str__ = __repr__


# ---------------------------------------------------------------------------------------- differentials

class Differential:
    """An SDE right-hand side, `(D1 + D2) * dt + sigma * dW0`: a drift (the terms multiplied by dt) and a noise
    loading (the shocks).  Assigned to a state (`X.d = ...`) or observed by an agent (`observes=...`), it is the
    state's drift and noise, or the signal's.  Every quantity term needs its dt; a shock never has one."""
    __slots__ = ("drift", "noise")

    def __init__(self, drift=None, noise=None):
        self.drift = drift if drift is not None else Linear({})
        self.noise = noise if noise is not None else Linear({})

    @staticmethod
    def of(x) -> "Differential":
        if isinstance(x, Differential):
            return x
        if _is_number(x) and x == 0:
            return Differential()
        L = Linear.of(x)
        stray = [a.key() if a.kind == "q" else "a constant" for a, c in L.terms.items() if a.kind != "w" and not (_is_number(c) and c == 0)]
        if stray:
            raise ValueError(f"{', '.join(stray)} has no dt: a drift term is written with dt (X * dt), "
                             f"a shock without (sigma * dW0)")
        return Differential(noise=L)

    def __add__(self, o):
        try:
            o = Differential.of(o)
        except TypeError:
            return NotImplemented
        return Differential(self.drift + o.drift, self.noise + o.noise)

    __radd__ = __add__

    def __sub__(self, o):
        try:
            o = Differential.of(o)
        except TypeError:
            return NotImplemented
        return Differential(self.drift - o.drift, self.noise - o.noise)

    def __rsub__(self, o):
        return Differential.of(o) - self

    def __neg__(self):
        return Differential(-self.drift, -self.noise)

    def __mul__(self, k):
        if not _scalar(k):
            return NotImplemented
        return Differential(self.drift * k, self.noise * k)

    __rmul__ = __mul__

    def combined(self) -> Linear:
        """The drift and the noise as one linear expression, the form a State and a Signal store."""
        return self.drift + self.noise

    def __repr__(self) -> str:
        parts = []
        if self.drift.terms:
            parts.append(f"({self.drift}) dt")
        if self.noise.terms:
            parts.append(str(self.noise))
        return " + ".join(parts) or "0"

    __array_ufunc__ = None


class _Dt:
    """dt, the time increment: multiplying a linear expression by it makes the drift of a Differential."""
    __slots__ = ()

    def __mul__(self, o):
        if isinstance(o, Shock) or (isinstance(o, Linear) and o.shocks):
            raise ValueError("a shock times dt has no meaning here: write the shock alone (sigma * dW0)")
        if isinstance(o, (Differential, Quad)):
            raise ValueError(f"dt multiplies a linear expression, not {o!r}")
        try:
            return Differential(drift=Linear.of(o))
        except TypeError:
            return NotImplemented

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return "dt"

    __array_ufunc__ = None


dt = _Dt()


# -------------------------------------------------------------------------------------- signals and agents

class Signal:
    """One observed row, the differential of what is observed: `Signal("y", sqrt(p) * X * dt + dW1, delay=0.5)`;
    the dt terms are the row's drift, the shock terms its noise loading (at least one is required)."""

    def __init__(self, name: str, expr, delay=0.0):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"a signal name must be an identifier, not {name!r}")
        if isinstance(expr, Quad):
            raise ValueError(f"signal {name}: a signal is linear, not {expr}")
        self.name = name
        self.expr = Differential.of(expr).combined()
        if not self.expr.shocks:
            raise ValueError(f"signal {name!r} has no shock term: every row needs a noise loading, e.g. Signal({name!r}, {self.expr} dt + dV)")
        if not (_is_number(delay) or isinstance(delay, Coef)) or (_is_number(delay) and delay < 0):
            raise ValueError(f"signal {name}: delay must be a non-negative number or a Param, not {delay!r}")
        self.delay = delay

    def compile(self) -> dict:
        drift, noise, const = self.expr.split(f"signal {self.name}")
        if not (_is_number(const) and const == 0):
            raise ValueError(f"signal {self.name}: a constant carries no information (the agent knows it); leave it out")
        out = {"drift": drift, "noise": noise}
        if not (_is_number(self.delay) and self.delay == 0):
            out["delay"] = _coef_str(self.delay)
        return out

    def __repr__(self) -> str:
        return f"Signal({self.name!r}, {self.expr})"


class Agent:
    """An agent: its controls, what it observes, its quadratic loss; `myopic` ignores the continuation effects
    of its own actions (a competitive agent).  `observes` is one differential (the signal "y"), a list (y1, y2,
    ...), a dict {name: differential}, or Signals (for a name and a delay) in any of these."""

    def __init__(self, name: str, controls: Sequence[Control], observes=None, loss=None, myopic: bool = False,
                 terminal=None):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"an agent name must be an identifier, not {name!r}")
        if observes is None:
            raise ValueError(f"agent {name}: observes= is required (what the agent sees)")
        if isinstance(observes, dict):
            signals = [o if isinstance(o, Signal) else Signal(k, o) for k, o in observes.items()]
        elif isinstance(observes, (list, tuple)):
            signals = [o if isinstance(o, Signal) else Signal(f"y{i + 1}", o) for i, o in enumerate(observes)]
        else:
            signals = [observes if isinstance(observes, Signal) else Signal("y", observes)]
        controls = list(controls) if not isinstance(controls, Control) else [controls]
        for u in controls:
            if not isinstance(u, Control):
                raise ValueError(f"agent {name}: controls must be Control objects, not {u!r}")
        signals = list(signals) if not isinstance(signals, Signal) else [signals]
        for s in signals:
            if not isinstance(s, Signal):
                raise ValueError(f"agent {name}: signals must be Signal objects, not {s!r}")
        if loss is None:
            raise ValueError(f"agent {name}: a loss is required (quadratic in the quantities)")
        try:
            self.loss = Quad.of(loss)
        except TypeError:
            raise ValueError(f"agent {name}: the loss must be a quadratic expression, not {loss!r}") from None
        if not self.loss.terms:
            raise ValueError(f"agent {name}: the loss {loss!r} has no term in a quantity")
        self.name = name; self.controls = controls; self.signals = signals
        self.myopic = bool(myopic)
        # the loss paid at T, on the states: terminal=q * (X - b)**2
        try:
            self.terminal = None if terminal is None else Quad.of(terminal)
        except TypeError:
            raise ValueError(f"agent {name}: the terminal loss must be a quadratic expression, not {terminal!r}") from None

    def compile(self) -> Tuple[dict, object]:
        """(the agent block, the loss's dropped constant)."""
        names = [s.name for s in self.signals]
        if len(set(names)) != len(names):
            raise ValueError(f"agent {self.name}: two signals share a name ({names})")
        terms, const = self.loss.compile(f"agent {self.name}")
        block = {"controls": [u.name for u in self.controls], "signals": {s.name: s.compile() for s in self.signals}, "loss": terms}
        if not (_is_number(const) and const == 0):
            block["constant"] = _coef_str(const)
        if self.terminal is not None:
            tterms, tconst = self.terminal.compile(f"agent {self.name}, terminal loss")
            if tterms:
                block["terminal"] = tterms
            if not (_is_number(tconst) and tconst == 0):
                block["terminal_constant"] = _coef_str(tconst)
        if self.myopic:
            block["myopic"] = True
        return block, const

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, controls={[u.name for u in self.controls]})"


# ------------------------------------------------------------------------------------------------- horizons

class _Horizon:
    """A horizon in the expression form.  The three kinds are separate types so that the valid
    combination of quantities is expressed by the TYPE, rather than by optional fields on one
    object -- a stationary model has no terminal time and a finite one has no lag window.
    """
    kind = "stationary"
    #  Neither quantity is declared here.  Each subclass sets ONLY the one its kind has, so
    #  Finite(...).window raises AttributeError rather than answering None: the valid combination is
    #  expressed by the TYPE, and a class attribute defaulting to None would be optional fields on a
    #  catch-all with extra steps -- exactly what the split replaced.
    LENGTHS = ()                 # the length attributes this kind carries, in file order

    @property
    def extent(self):
        """The length of the primary computational axis: T where the kind has one, else the window."""
        return self.window if self.kind == "stationary" else self.T

    def compile(self) -> dict:
        out = {"kind": self.kind, "discount": _coef_str(self.discount)}
        for name in self.LENGTHS:
            out[name] = _coef_str(getattr(self, name))
        return out


class Stationary(_Horizon):
    """A stationary horizon: the lag-truncation length L and the discount rate (0 is average cost)."""
    kind = "stationary"
    LENGTHS = ("window",)

    def __init__(self, window: float = 8.0, discount=0.0):
        self.window = window; self.discount = discount


class Finite(_Horizon):
    """A finite horizon [0, T].  T is the terminal time; a finite horizon has no lag window."""
    kind = "finite"
    LENGTHS = ("T",)

    def __init__(self, T: float = 1.0, discount=0.0):
        self.T = T; self.discount = discount


class Transition(_Horizon):
    """A transition on [0, T] from `past` (a stationary Model, its dict, a path, or a list of initial shocks
    {name, loads, rows}) continued by "stationary" (the new model's equilibrium, on the past's window) or "end"."""
    kind = "transition"
    LENGTHS = ("T",)

    def __init__(self, T: float = 1.0, past=None, continuation: str = "stationary", discount=0.0):
        self.T = T; self.discount = discount
        if past is None:
            raise ValueError("Transition(): a past is required (a stationary Model, a path, or a list of initial shocks)")
        self.past = past; self.continuation = continuation

    def compile(self) -> dict:
        past = self.past
        if isinstance(past, _spec.Model):
            past = past.to_dict()
        if isinstance(past, (list, tuple)):
            block = {"initial": [{k: ({a: _coef_str(c) for a, c in v.items()} if isinstance(v, dict) else v) for k, v in sh.items()}
                                 for sh in past]}
        else:
            block = {"model": past}
        return {**super().compile(), "past": block, "continuation": self.continuation}


# ------------------------------------------------------------------------------------------- the compiler

def is_expression_form(states, agents, horizon, definitions) -> bool:
    """Whether Model(...) was given expression objects (else spec's own field form)."""
    return (any(isinstance(s, State) for s in (states or [])) or any(isinstance(a, Agent) for a in (agents or []))
            or isinstance(horizon, Stationary) or any(isinstance(d, Definition) for d in (definitions or [])))


def _no_repeats(names, message, model):
    """Names used once each; `message` names the offender as {name}."""
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Model {model}: " + message.format(name=repr(dup[0])))


def _check_kinds(name, states, agents, definitions, horizon):
    """The arguments are the objects the expression form builds, and the horizon is one of the three."""
    for items, kind, what in ((states, State, "states must be State objects"),
                              (agents, Agent, "agents must be Agent objects"),
                              (definitions, Definition, "definitions must be define()d quantities")):
        for x in items:
            if not isinstance(x, kind):
                raise ValueError(f"Model {name}: {what}, not {x!r}")
    #  _Horizon, not Stationary: the three kinds are siblings now.  Finite used to SUBCLASS
    #  Stationary -- which is how a terminal time came to be stored in a field called window --
    #  and this check was the last thing that inheritance was carrying.
    if not isinstance(horizon, _Horizon):
        raise ValueError(f"Model {name}: horizon must be Stationary(...), Finite(...) or Transition(...), not {horizon!r}")


class _Walk:
    """One pass over every expression of a model, gathering what the file needs: the definitions reached
    (a define()d quantity that appears, and whatever its own expression reaches), the shocks loaded, and
    the Params used by a coefficient, a lag or a delay."""

    def __init__(self):
        self.defs: Dict[str, Definition] = {}
        self.shocks: Dict[str, Shock] = {}
        self.params: Dict[int, Param] = {}

    def coef(self, c):
        if isinstance(c, Coef):
            for p in c.params():
                self.params.setdefault(id(p), p)

    def _lag(self, atom):
        if isinstance(atom.lag, Param):
            self.params.setdefault(id(atom.lag), atom.lag)

    def _reach(self, quantities):
        for q in quantities:
            if isinstance(q, Definition) and q.name not in self.defs:
                self.defs[q.name] = q
                self.linear(q.expr)

    def linear(self, l: Linear):
        for a, c in l.terms.items():
            self.coef(c); self._lag(a)
        for w in l.shocks.values():
            self.shocks.setdefault(w.name, w)
        self._reach(l.quantities.values())

    def quad(self, q: Quad):
        for key, c in q.terms.items():
            self.coef(c)
            for x in key:
                self._lag(x)
        self.coef(q.const)
        self._reach(q.quantities.values())

    def model(self, states, agents, definitions, horizon):
        self._reach(definitions)
        for s in states:
            if s._drift is not None:
                self.linear(s._drift)
            self.coef(s.initial)
        for a in agents:
            for sg in a.signals:
                self.linear(sg.expr); self.coef(sg.delay)
            self.quad(a.loss)
            if a.terminal is not None:
                self.quad(a.terminal)
        self.coef(horizon.extent); self.coef(horizon.discount)
        if isinstance(horizon, Transition):
            if isinstance(horizon.past, (list, tuple)):
                for sh in horizon.past:
                    for key in ("loads", "rows"):
                        for c in (sh.get(key) or {}).values():
                            self.coef(c)
        return self


def _check_quantities(name, states, controls, defs, agents):
    """Every atom names a state, a control or a definition of this model, and the very object this model
    lists: a State of the right name built elsewhere is a different quantity (Quantity equality is identity)."""
    known = set(defs) | {x.name for x in states} | {x.name for x in controls}
    own = {id(x) for x in states} | {id(x) for x in controls}

    def check(where, quantities):
        """`where` is the whole lead-in, up to but not including the verb."""
        for q in quantities:
            if q.name not in known or (isinstance(q, (State, Control)) and id(q) not in own):
                raise ValueError(f"{where} uses {q!r}, which is not a state, control or definition of the model {name!r}")

    for s in states:
        if s._drift is not None:
            check(f"state {s.name}: its drift", s._drift.quantities.values())
    for a in agents:
        for sg in a.signals:
            check(f"signal {a.name}.{sg.name}:", sg.expr.quantities.values())
        check(f"agent {a.name}: its loss", a.loss.quantities.values())
    for d in defs.values():
        check(f"definition {d.name}:", d.expr.quantities.values())


def _param_values(name, used, params):
    """The parameters used, in creation order, with the values to write: as each Param was given, or as
    `params` supplies them (a mapping name -> value, or an iterable of Params)."""
    plist = sorted(used.values(), key=lambda p: p.order)
    seen: Dict[str, Param] = {}
    for p in plist:
        if seen.setdefault(p.name, p) is not p:
            raise ValueError(f"Model {name}: two different Params are named {p.name!r}")
    if params is None:
        values = {p.name: p.given for p in plist}
    else:
        as_map = isinstance(params, dict)
        if not as_map:
            for p in params:
                if not isinstance(p, Param):
                    raise ValueError(f"Model {name}: params must be Params or a mapping of name to value, not {p!r}")
        supplied = dict(params) if as_map else {p.name: p.given for p in params}
        for p in plist:
            if p.name not in supplied:
                raise ValueError(f"unknown Param {p.name!r} in a coefficient of the model {name!r}: it is not among params={sorted(supplied)}")
        values = {p.name: supplied[p.name] for p in plist}
        for k, v in supplied.items():
            if k not in values:
                if as_map:
                    raise ValueError(f"Model {name}: params gives {k!r}, which no coefficient of the model uses")
                values[k] = v
    for k, v in values.items():
        if v is None:
            raise ValueError(f"unknown Param {k!r} in a coefficient of the model {name!r}: it has no value; give Param({k!r}, value) or params={{{k!r}: value}}")
        if not _is_number(v):
            raise ValueError(f"Model {name}: the value of {k!r} must be a number, not {v!r}")
    return values


def compile_model(name: str, states, agents, definitions=None, ties=None, horizon=None, numerics=None, params=None):
    """The model file dict of an expression model.  Channels are the shocks used, in the order
    of their shocks() namespaces; params are the Params used (in creation order, values as given, or the
    values `params` supplies: a mapping name -> value or an iterable of Params); definitions are the given
    ones then every define()d quantity that appears."""
    states = list(states or []); agents = list(agents or []); definitions = list(definitions or [])
    horizon = Stationary() if horizon is None else horizon
    _check_kinds(name, states, agents, definitions, horizon)
    controls = [u for a in agents for u in a.controls]
    _no_repeats([s.name for s in states], "two states named {name}", name)
    _no_repeats([a.name for a in agents], "two agents named {name}", name)
    _no_repeats([u.name for u in controls], "control {name} is listed by two agents", name)

    walk = _Walk().model(states, agents, definitions, horizon)
    _check_quantities(name, states, controls, walk.defs, agents)
    values = _param_values(name, walk.params, params)

    d: dict = {"name": name}
    if values:
        d["params"] = dict(values)
    d["shocks"] = [w.name for w in sorted(walk.shocks.values(), key=lambda w: (w.space.order, w.index))]
    d["states"] = {s.name: s.compile() for s in states}
    if walk.defs:
        d["definitions"] = {n: q.compile() for n, q in walk.defs.items()}
    d["agents"] = {}
    for a in agents:
        block, _ = a.compile()
        d["agents"][a.name] = block
    if ties:
        d["ties"] = [[x.name if isinstance(x, Agent) else str(x) for x in group] for group in ties]
    d["horizon"] = horizon.compile()
    if numerics is not None:
        from .numerics import Numerics
        d["numerics"] = Numerics.of(numerics).to_dict()
    return d


__all__ = ["Param", "Coef", "shocks", "Shock", "State", "Control", "define", "Definition", "Quantity", "Linear", "Quad",
           "Signal", "Agent", "Stationary", "Finite", "Transition",
           "sqrt", "exp", "log", "sin", "cos", "tanh"]
