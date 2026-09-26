"""Messages for a name that is not there: what was asked for, the nearest names, and the names there are.

One helper so that every reader (res.kernel, res.response, res.mean, ...), the model's expansion and solve()'s
options say the same thing the same way: a misspelling should cost one glance, not a trip to the model file."""
import difflib
from typing import Iterable


def _one_edit(a: str, b: str) -> bool:
    """Whether a and b differ by one substitution, insertion, deletion or swap of neighbours (r3 for r1, sigam)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def nearest(name: str, choices: Iterable[str], n: int = 3) -> list:
    """The choices closest to `name`: an exact match up to case first, then one edit away (what difflib misses on
    short names: r3 for r1), then the choices it begins (ch1_two_player), then difflib's."""
    choices = [str(c) for c in choices]; name = str(name)
    out = [c for c in choices if c.lower() == name.lower()]
    out += [c for c in choices if c not in out and _one_edit(name.lower(), c.lower())]
    out += [c for c in choices if c not in out and len(name) >= 3 and c.lower().startswith(name.lower())]
    out += [c for c in difflib.get_close_matches(name, choices, n=n, cutoff=0.6) if c not in out]
    return out[:n]


def unknown(kind: str, name, choices: Iterable[str], plural: str = None) -> str:
    """'unknown quantity 'Xx'; did you mean 'X'? the quantities are [...]'."""
    choices = list(choices)
    near = nearest(name, choices)
    hint = f"; did you mean {' or '.join(repr(c) for c in near)}?" if near else ""
    if len(choices) > 40:
        return f"unknown {kind} {name!r}{hint}"
    return f"unknown {kind} {name!r}{hint}" + (" The" if near else "; the") + f" {plural or kind + 's'} are {choices}"


class NameNotFound(KeyError):
    """A KeyError (what the readers raised before) whose message prints as written, not quoted."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""
