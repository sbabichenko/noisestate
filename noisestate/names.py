"""Messages for a name that is not there: what was asked for, the nearest names, and the names there are.

One helper so that every reader (res.kernel, res.response, res.mean, ...), the model's expansion and solve()'s
options say the same thing the same way: a misspelling should cost one glance, not a trip to the model file."""
import difflib
from typing import Iterable


def nearest(name: str, choices: Iterable[str], n: int = 3) -> list:
    """The choices closest to `name`: an exact match up to case first, then difflib's."""
    choices = [str(c) for c in choices]
    folded = [c for c in choices if c.lower() == str(name).lower()]
    return (folded + [c for c in difflib.get_close_matches(str(name), choices, n=n, cutoff=0.6) if c not in folded])[:n]


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
