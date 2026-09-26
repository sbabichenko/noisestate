"""Model files written as equations: the same model as the grammar of spec.py, in the notation of the paper.

    params: {p1: 3, r1: 0.1, b1: 1, sigma: 1, T: 1}
    shocks: [W0, W1]
    states:
      X: "(D1 + D2) dt + sigma dW0"             # dX = (D1 + D2) dt + sigma dW0
    agents:
      player1: {controls: D1, observes: "sqrt(p1) X dt + dW1", loss: "(X - b1)^2 + r1 D1^2"}
    horizon: {T: T}                             # finite on [0, T]; {window: 8} is stationary
    numerics: {nodes: 24}

Each string is read into the objects of the Python equations form (expr.py) and the model is compiled from them,
so the two forms are one model.  Multiplication may be written as a space ("r1 D1^2"), ^ is a power, dt marks the
drift and dW<name> is the shock <name>; X@0.5 is X half a time unit earlier (a lag), X@-0.5 later (a lead, in a
loss only).  A state may be {d: "...", initial: 1}; an agent's observes is one string (the signal y), a list
(y1, y2, ...) or a mapping {name: string}, and any signal may be {d: "...", delay: 0.5}.  A horizon with a kind
(a transition, say) is taken as the grammar's horizon block unchanged.  Model.to_equations() writes this form.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List

from . import expr as E

_FUNCS = {"sqrt": E.sqrt, "exp": E.exp, "log": E.log, "sin": E.sin, "cos": E.cos, "tanh": E.tanh}
_KEYS = {"name", "params", "shocks", "states", "agents", "definitions", "horizon", "numerics", "ties"}


def is_equation_form(d: dict) -> bool:
    """A model dict in this form: it writes a state, an agent's observations or a loss as equations (both forms
    list their shocks under shocks:)."""
    if not isinstance(d, dict):
        return False
    states = d.get("states") or {}
    if any(isinstance(v, str) or (isinstance(v, dict) and "d" in v) for v in states.values()):
        return True
    return any(isinstance(a, dict) and ("observes" in a or isinstance(a.get("loss"), str)) for a in (d.get("agents") or {}).values())


# ------------------------------------------------------------------------------------------ the strings

_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"
                    r"|(?P<name>[A-Za-z_]\w*)(?:@(?P<lag>-?(?:\d+(?:\.\d*)?|\.\d+|[A-Za-z_]\w*)))?"
                    r"|(?P<op>\*\*|[-+*/^(),]))")


def _python(text: str, what: str) -> str:
    """The equation as a Python expression: implicit products made explicit, ^ as **, name@tau as a lag call."""
    out: List[str] = []; pos = 0; prev = None           # prev: the kind of the last token ("value", "func", "op", "(")
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"{what}: cannot read {text[pos:pos + 12]!r} in {text!r}")
        pos = m.end()
        if m.group("num") is not None or m.group("name") is not None:
            name = m.group("name")
            if prev in ("value", ")"):
                out.append("*")                          # "r1 D1" is r1 * D1, "2 X" is 2 * X, "(a) b" is (a) * b
            if name is not None and m.group("lag") is not None:
                out.append(f"__lag__({name}, {m.group('lag')})"); prev = "value"
            elif name is not None and name in _FUNCS and text[pos:].lstrip().startswith("("):
                out.append(name); prev = "func"
            else:
                out.append(m.group("num") if name is None else name); prev = "value"
            continue
        op = m.group("op")
        if op == "(":
            if prev in ("value", ")"):
                out.append("*")
            out.append("("); prev = "("
        elif op == ")":
            out.append(")"); prev = ")"
        else:
            out.append("**" if op in ("^", "**") else op); prev = "op"
    return " ".join(out)


def _lag(x, tau):
    tau = float(tau) if not isinstance(tau, E.Coef) else tau
    if isinstance(tau, float) and tau < 0:
        return x.lead(-tau)
    return x.lag(tau)


def _eval(text, env: dict, what: str):
    """Evaluate one equation string over the model's symbols (numbers, parameters, quantities, shocks, dt)."""
    if not isinstance(text, str):
        return text                                     # a number written as a number
    src = _python(text, what)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise ValueError(f"{what}: {text!r} is not an expression") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"{what}: {node.id!r} is not a parameter, state, control, definition or shock "
                                 f"of this model (a shock is written dW<name>; the shocks are {env['__shocks__']})")
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}
            for k, f in ops.items():
                if isinstance(node.op, k):
                    return f()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "__lag__" and len(node.args) == 2:
                return _lag(ev(node.args[0]), ev(node.args[1]))
            if node.func.id in _FUNCS and len(node.args) == 1:
                return _FUNCS[node.func.id](ev(node.args[0]))
        raise ValueError(f"{what}: {text!r} uses something an equation cannot ({ast.dump(node)[:50]})")

    return ev(tree)


# ------------------------------------------------------------------------------------------ the model

def _shock_names(d: dict) -> List[str]:
    """The shocks: as listed, or the dW names in the order the equations first use them."""
    if d.get("shocks") is not None:
        s = d["shocks"]
        return s.replace(",", " ").split() if isinstance(s, str) else [str(x) for x in s]
    seen: List[str] = []
    def scan(v):
        if isinstance(v, str):
            for n in re.findall(r"\bd(W\w*)\b", v):
                if n not in seen:
                    seen.append(n)
        elif isinstance(v, dict):
            for x in v.values():
                scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
    scan(d.get("states")); scan(d.get("agents")); scan(d.get("definitions"))
    return seen


def _in_dependency_order(defs: dict) -> list:
    """The definitions as (name, text) pairs, each after the definitions it reads: a map has no order, so
    `yH: q + Pidx` may come before `Pidx`.  A cycle is left to the evaluator, which names the first unknown."""
    pending = dict(defs); out = []
    while pending:
        ready = [k for k, v in pending.items()
                 if not (set(re.findall(r"[A-Za-z_]\w*", str(v))) & (set(pending) - {k}))]
        for k in ready or list(pending)[:1]:
            out.append((k, pending.pop(k)))
    return out


def _as_list(x) -> list:
    return [] if x is None else [x] if isinstance(x, str) else list(x)


def _environment(d: dict):
    """The names an equation of the equations dict `d` may use: (env, params, states, controls), env mapping
    each parameter, d<shock>, state, control and definition (evaluated, in dependency order) to its object."""
    values = dict(d.get("params") or {})
    params = {k: E.Param(k, v) for k, v in values.items()}
    shock_names = _shock_names(d)
    sh = E.shocks(*shock_names) if shock_names else None
    env: Dict[str, object] = dict(params)
    env["__shocks__"] = shock_names
    env["dt"] = E.dt
    for n in shock_names:
        env["d" + n] = sh[n]
    states = {k: E.State(k) for k in (d.get("states") or {})}
    controls = {}
    for a, spec in (d.get("agents") or {}).items():
        for u in _as_list((spec or {}).get("controls")):
            controls[u] = E.Control(u)
    env.update(states); env.update(controls)
    names = set(params) | set(states) | set(controls) | set(d.get("definitions") or {})
    clash = sorted(n for n in shock_names if "d" + n in names)
    if clash:
        raise ValueError(f"shock(s) {clash}: 'd' + the shock's name is also the name of a parameter, state, control or "
                         "definition, so an equation could not tell the increment from the quantity; rename one of them")
    for k, v in _in_dependency_order(d.get("definitions") or {}):
        env[k] = E.define(k, _eval(v, env, f"definition {k}"))
    return env, params, states, controls


def _signal(sname: str, v, env: dict, where: str):
    """One observed row, "..." or {d: "...", delay: tau}, as a Signal; {level: P} is the instant observation of another
    agent's control P (a Level)."""
    if isinstance(v, dict) and "level" in v:
        if set(v) != {"level"}:
            raise ValueError(f"{where}, {sname}: an instant observation is {{level: P}} alone")
        q = env.get(str(v["level"]))
        if not isinstance(q, E.Control):
            raise ValueError(f"{where}, {sname}: {v['level']!r} is not a control; {{level: ...}} observes another agent's control")
        return E.Level(q)
    if isinstance(v, dict):
        ex = sorted(set(v) - {"d", "delay"})
        if ex:
            raise ValueError(f"{where}, signal {sname}: unknown key(s) {ex}")
        delay = v.get("delay", 0.0)
        delay = _eval(delay, env, where) if isinstance(delay, str) else delay
        return E.Signal(sname, _eval(v.get("d"), env, f"{where}, signal {sname}"), delay=delay)
    return E.Signal(sname, _eval(v, env, f"{where}, signal {sname}"))


def signal_block(d: dict, name: str, row) -> dict:
    """The grammar's block of one row written as an equation ("..." or {d: "...", delay: tau}) in the context of
    the equations dict `d` (its parameters, shocks, states, controls and definitions): Model.with_signal's reader."""
    env = _environment(d)[0]
    return _signal(name, row, env, f"signal {name}").compile()


def to_grammar(d: dict) -> dict:
    """The model file of spec.py's grammar that this equations dict means (compiled through expr.py)."""
    bad = sorted(set(d) - _KEYS)
    if bad:
        raise ValueError(f"unknown key(s) {bad} in an equations model; allowed: {sorted(_KEYS)}")
    name = d.get("name", "model")
    values = dict(d.get("params") or {})
    env, params, states, controls = _environment(d)
    defs = [env[k] for k in (d.get("definitions") or {})]           # the file's order: load(save(m)) == m

    for k, v in (d.get("states") or {}).items():
        where = f"state {k}"
        if isinstance(v, dict):
            extra = sorted(set(v) - {"d", "initial"})
            if extra:
                raise ValueError(f"{where}: unknown key(s) {extra}; a state is \"...\" or {{d: \"...\", initial: ...}}")
            if "initial" in v:
                states[k].initial = _eval(v["initial"], env, where) if isinstance(v["initial"], str) else v["initial"]
            v = v.get("d")
        states[k].d = _eval(v, env, where)

    agents = []
    for a, spec in (d.get("agents") or {}).items():
        spec = dict(spec or {}); where = f"agent {a}"
        extra = sorted(set(spec) - {"controls", "observes", "loss", "myopic", "terminal", "monitors"})
        if extra:
            raise ValueError(f"{where}: unknown key(s) {extra}; allowed: controls, observes, loss, terminal, myopic, monitors")

        def signal(sname, v):
            return _signal(sname, v, env, where)

        obs = spec.get("observes")
        if obs is None:
            raise ValueError(f"{where}: observes is required (what the agent sees)")
        if isinstance(obs, dict) and not ("d" in obs and len(set(obs) - {"d", "delay"}) == 0):
            signals = [signal(k, v) for k, v in obs.items()]
        elif isinstance(obs, list):
            signals = [signal(f"y{i + 1}", v) for i, v in enumerate(obs)]
        else:
            signals = [signal("y", obs)]
        loss = spec.get("loss")
        if loss is None:
            raise ValueError(f"{where}: a loss is required")
        agents.append(E.Agent(a, controls=[controls[u] for u in _as_list(spec.get("controls"))], observes=signals,
                              loss=_eval(loss, env, f"{where}, loss"), myopic=bool(spec.get("myopic", False)),
                              monitors=_as_list(spec.get("monitors")),
                              terminal=_eval(spec["terminal"], env, f"{where}, terminal") if spec.get("terminal") is not None else None))

    hz = d.get("horizon") or {}
    if "kind" not in hz and "past" in hz:               # a transition: {T (or settle), past, continuation, discount}
        ex = sorted(set(hz) - {"T", "settle", "past", "continuation", "discount"})
        if ex:
            raise ValueError(f"horizon: unknown key(s) {ex} for a transition; it takes T (or settle), past, continuation "
                             "and discount (its window is the past's)")
        past = hz["past"]
        block = ({"model": past} if isinstance(past, str) else {"initial": past} if isinstance(past, list) else dict(past))
        hz = {"kind": "transition", "continuation": "stationary", **{k: v for k, v in hz.items() if k != "past"}, "past": block}
    if "kind" in hz:
        horizon = None                                  # the grammar's own block, laid in below
    else:
        ex = sorted(set(hz) - {"T", "window", "discount"})
        if ex or ("T" in hz) == ("window" in hz):
            raise ValueError("horizon: give T (a finite game on [0, T]) or window (a stationary game), and optionally "
                             "discount; a transition adds past: (the old model's file, or initial shocks) to T")
        val = lambda x: _eval(x, env, "horizon") if isinstance(x, str) else x
        disc = val(hz.get("discount", 0.0))
        horizon = E.Finite(T=val(hz["T"]), discount=disc) if "T" in hz else E.Stationary(window=val(hz["window"]), discount=disc)

    out = E.compile_model(name, list(states.values()), agents, definitions=defs, ties=d.get("ties"),
                             horizon=horizon or E.Finite(T=1.0), params=params.values())
    out["params"] = values                              # every parameter, used or not (the horizon may use one)
    if horizon is None:
        out["horizon"] = hz
    if d.get("numerics") is not None:
        out["numerics"] = d["numerics"]
    return out


# ------------------------------------------------------------------------------------------ writing it

def _is_sum(c: str) -> bool:
    """Whether the coefficient text has a + or - outside parentheses other than a leading sign ("a - b" does,
    "-(1 - xi)" and "2e-3*r" do not), so that it needs parentheses as a factor."""
    depth = 0
    for i, ch in enumerate(c):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and ch in "+-" and i > 0 and c[i - 1] not in "eE*/^(" and c[:i].strip():
            return True
    return False


def _coef_text(c) -> str:
    if isinstance(c, str):
        c = c.replace("**", "^")
        if re.fullmatch(r"-?[\w.]+(\*[\w.]+)+", c):
            return c.replace("*", " ")                  # a plain product reads as the equations write it: 2 kappa
        return f"({c})" if _is_sum(c) else c
    if isinstance(c, float):
        return str(int(c)) if c.is_integer() and abs(c) < 1e15 else repr(c)      # exact: load(save(m)) is m
    return str(c)


def _terms(pairs) -> str:
    """'a X + b Y' from (coefficient, atom text) pairs, signs folded, unit coefficients left out."""
    out = ""
    for c, atom in pairs:
        if isinstance(c, (int, float)) and c == 0:
            continue
        text = _coef_text(c)
        neg = text.startswith("-")
        body = text[1:] if neg else text
        if body in ("1", "1.0") and atom:
            body = ""
        term = " ".join(x for x in (body, atom) if x)
        out += (" - " if neg else " + ") + term if out else ("-" if neg else "") + term
    return out or "0"


def _differential(row: dict) -> str:
    drift = row.get("drift") or {}; noise = row.get("noise") or {}
    parts = []
    dr = _terms([(c, "" if a == "const" else a) for a, c in drift.items()])
    if dr != "0":
        parts.append((f"({dr})" if (" + " in dr or " - " in dr) else dr) + " dt")
    nz = _terms([(c, "d" + w) for w, c in noise.items()])
    if nz != "0":
        parts.append(nz)
    s = " + ".join(parts) or "0"
    return s.replace("+ -", "- ")


def from_grammar(g: dict) -> dict:
    """The equations form of a grammar dict (Model.to_dict()): the same model, each block written as an equation.
    A loss is written expanded ((X - b)^2 comes back as X^2 - 2 b X + b^2)."""
    out: dict = {"name": g.get("name", "model")}
    if g.get("params"):
        out["params"] = dict(g["params"])
    out["shocks"] = list(g.get("shocks") or [])
    out["states"] = {}
    for k, v in (g.get("states") or {}).items():
        eq = _differential(v)
        out["states"][k] = {"d": eq, "initial": v["initial"]} if v.get("initial") is not None else eq
    if g.get("definitions"):
        out["definitions"] = {k: _terms([(c, "" if a == "const" else a) for a, c in v.items()]) for k, v in g["definitions"].items()}
    out["agents"] = {}
    for a, v in (g.get("agents") or {}).items():
        sig = {}
        for u in _as_list(v.get("instant")):
            sig[u] = {"level": u}
        for sname, row in (v.get("signals") or {}).items():
            eq = _differential(row)
            delay = row.get("delay", 0)
            sig[sname] = {"d": eq, "delay": delay} if delay not in (0, 0.0, None) else eq
        loss = _terms([(t[0], f"{t[1]}^2" if len(t) == 3 and t[1] == t[2] else " ".join(str(x) for x in t[1:])) for t in v.get("loss") or []]
                      + ([(v["constant"], "")] if v.get("constant") not in (None, 0, 0.0) else []))
        block = {"controls": v["controls"][0] if len(v.get("controls") or []) == 1 else list(v.get("controls") or []),
                 "observes": next(iter(sig.values())) if list(sig) == ["y"] else sig, "loss": loss}
        if v.get("terminal") or v.get("terminal_constant") not in (None, 0, 0.0):
            block["terminal"] = _terms([(t[0], f"{t[1]}^2" if len(t) == 3 and t[1] == t[2] else " ".join(str(x) for x in t[1:])) for t in v.get("terminal") or []]
                                       + ([(v["terminal_constant"], "")] if v.get("terminal_constant") not in (None, 0, 0.0) else []))
        if v.get("myopic"):
            block["myopic"] = True
        mons = _as_list(v.get("monitors"))
        if mons:
            block["monitors"] = mons[0] if len(mons) == 1 else mons
        out["agents"][a] = block
    hz = dict(g.get("horizon") or {})
    if hz.get("kind") == "finite" and set(hz) <= {"kind", "T", "discount"}:
        out["horizon"] = {"T": hz["T"], **({"discount": hz["discount"]} if hz.get("discount") not in (None, 0, 0.0) else {})}
    elif hz.get("kind") == "stationary" and set(hz) <= {"kind", "window", "discount"}:
        out["horizon"] = {"window": hz["window"], **({"discount": hz["discount"]} if hz.get("discount") not in (None, 0, 0.0) else {})}
    elif hz.get("kind") == "transition" and set(hz) <= {"kind", "T", "settle", "past", "continuation", "discount"}:
        past = hz.get("past") or {}
        past = past["model"] if set(past) == {"model"} else past["initial"] if set(past) == {"initial"} else past
        out["horizon"] = {**({"T": hz["T"]} if hz.get("T") is not None else {"settle": hz["settle"]}), "past": past,
                          **({"continuation": hz["continuation"]} if hz.get("continuation") not in (None, "stationary") else {}),
                          **({"discount": hz["discount"]} if hz.get("discount") not in (None, 0, 0.0) else {})}
    else:
        out["horizon"] = hz
    if g.get("ties"):
        out["ties"] = g["ties"]
    if g.get("numerics"):
        out["numerics"] = g["numerics"]
    return out
