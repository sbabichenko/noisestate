"""Solver-free presentation of the evaluated model specification.

``Model.describe()`` returns a string that prints as aligned plain text in a
terminal and renders as HTML in a notebook.  Both views are built from the same
extracted pieces, so they cannot drift apart.
"""
import os
from html import escape
from textwrap import wrap

WIDTH = 88
NOTATION = ('written as the model file writes it: coefficients in the parameters above, name@tau is that '
            'quantity tau time units earlier (negative tau is a lead), and d<shock> is that shock\'s Brownian '
            'increment')


# ---------------------------------------------------------------- the model's pieces, rendered once

def _align(equations):
    """Line up the equals signs of a block of differentials."""
    width = max((eq.index('=') for eq in equations), default=0)
    return [' ' * (width - eq.index('=')) + eq for eq in equations]


def _inline(mapping):
    return ', '.join(f'{k}' if v == 1 else f'{v:g} * {k}' for k, v in mapping.items())


def _path(path):
    """Show a model file relative to the working directory when it sits under it."""
    try:
        relative = os.path.relpath(path)
    except ValueError:
        return str(path)
    return relative if not relative.startswith('..') else str(path)


def _past(past):
    """Name the past the transition starts from: a model file, or the prior it carries."""
    if isinstance(past, dict) and set(past) == {'model'}:
        return _path(past['model'])
    if isinstance(past, dict) and set(past) == {'initial'}:
        return '; '.join(f'{p.get("name", "prior")} loads {_inline(p.get("loads", {}))} '
                         f'into {_inline(p.get("rows", {}))}' for p in past['initial'])
    return str(past)


def _pieces(model):
    """The model's equations as the file writes them (model.to_equations(), the one renderer), with the
    resolved horizon and notes."""
    hz = model.horizon
    span = f'{"lag window" if hz.kind == "stationary" else "T"} {hz.extent:g}'
    eq = model.to_equations()

    def text(v):
        return v["d"] if isinstance(v, dict) else v
    agents = []
    for agent in model.agents:
        observes = eq["agents"][agent.name]["observes"]
        if not isinstance(observes, dict) or "d" in observes:
            observes = {agent.signals[0].name: observes}
        agents.append(dict(
            name=agent.name, controls=list(agent.controls), loss=eq["agents"][agent.name]["loss"],
            signals=list(zip(_align([f'd{row.name} = {text(observes[row.name])}' for row in agent.signals]),
                             [row.delay for row in agent.signals]))))
    return dict(
        name=model.name,
        horizon=f'{hz.kind}, {span}, discount {hz.discount:g}',
        params=[f'{k} = {v:g}' for k, v in model.params.items()],
        states=list(zip(_align([f'd{s.name} = {text(eq["states"][s.name])}' for s in model.states]),
                        [s.initial for s in model.states])),
        definitions=[f'{d.name} = {eq["definitions"][d.name]}' for d in model.definitions],
        agents=agents,
        ties=[list(group) for group in model.ties],
        transition=(dict(past=_past(hz.past), continuation=str(hz.continuation or 'stationary'))
                    if hz.kind == 'transition' else None),
        notes=list(model.notes))


# ---------------------------------------------------------------------------------- the text layout

def _lead(label, indent, label_width):
    return ' ' * indent + label.ljust(label_width)


def _field(label, text, indent=2, label_width=11):
    """One labelled field, wrapped with its continuation lines under the text."""
    lead = _lead(label, indent, label_width)
    return wrap(text, WIDTH, initial_indent=lead, subsequent_indent=' ' * len(lead)) or [lead.rstrip()]


def _items(items, indent=2, label=''):
    """Wrap a comma-separated list without ever breaking one item across lines."""
    lines, lead = [], ' ' * indent
    for item in items:
        if lines and len(lines[-1]) + len(item) + 2 <= WIDTH:
            lines[-1] += ', ' + item
        else:
            if lines:
                lines[-1] += ','
            lines.append(lead + item)
    lines = lines or [lead]
    return [_lead(label, 2, indent - 2) + lines[0][indent:]] + lines[1:] if label else lines


def _bullets(items, indent=2):
    lines = []
    for item in items:
        lines += wrap(str(item), WIDTH, initial_indent=' ' * indent + '- ',
                      subsequent_indent=' ' * (indent + 2))
    return lines


def _text(p):
    lines = [f'Model: {p["name"]}'] + _field('horizon', p['horizon'])
    if p['params']:
        lines += _items(p['params'], indent=13, label='parameters')
    lines += _field('notation', NOTATION)

    lines += ['', 'States']
    for equation, initial in p['states']:
        lines.append('  ' + equation + (f'   [initial mean {initial:g}]' if initial is not None else ''))
    if p['definitions']:
        lines += ['', 'Definitions'] + ['  ' + d for d in p['definitions']]

    lines += ['', 'Agents']
    for agent in p['agents']:
        lines += ['  ' + agent['name']]
        lines += _field('controls', ', '.join(agent['controls']), indent=4)
        width = max((len(equation) for equation, _ in agent['signals']), default=0)
        rows = [equation + (f'{"":{width - len(equation) + 3}}observed with delay {delay:g}' if delay else '')
                for equation, delay in agent['signals']]
        lines += _field('observes', rows[0] if rows else 'nothing (no signal rows)', indent=4)
        lines += [' ' * len(_lead('observes', 4, 11)) + row for row in rows[1:]]
        lines += _field('flow loss', agent['loss'], indent=4)
        lines += ['']
    if lines[-1] == '':
        lines.pop()

    for group in p['ties']:
        lines += ['', 'Shared strategy'] + _items(group)
    if p['transition']:
        lines += ['', 'Transition']
        lines += _field('past', p['transition']['past'], label_width=14)
        lines += _field('continuation', p['transition']['continuation'], label_width=14)
    if p['notes']:
        lines += ['', 'Notes and conventions'] + _bullets(p['notes'])
    return '\n'.join(lines)


# ------------------------------------------------------------------------- the notebook's HTML view

STYLE = ('font-family:system-ui,sans-serif;line-height:1.5;max-width:60em;'
         'color:inherit;background:transparent')
MONO = 'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre'
RULE = '1px solid rgba(128,128,128,.45)'
CELL = f'border-top:{RULE};padding:.35em .8em .35em 0;vertical-align:top;text-align:left'
HEAD = f'border-bottom:{RULE};padding:.35em .8em .35em 0;text-align:left;font-weight:600'
DIM = 'opacity:.7'


def _mono(text):
    return f'<span style="{MONO}">{escape(str(text))}</span>'


def _rows(pairs):
    """A borderless label/value grid for the header and the transition."""
    cells = ''.join(f'<tr><td style="padding:.1em .8em .1em 0;{DIM}">{escape(label)}</td>'
                    f'<td style="padding:.1em 0">{value}</td></tr>' for label, value in pairs)
    return f'<table style="border-collapse:collapse;margin:.4em 0">{cells}</table>'


def _section(title, body):
    return (f'<div style="margin:1.1em 0 .3em;{DIM};text-transform:uppercase;'
            f'letter-spacing:.06em;font-size:.82em">{escape(title)}</div>{body}')


def _block(lines):
    return f'<div style="{MONO}">' + '<br>'.join(lines) + '</div>'


def _html(p):
    header = [('horizon', escape(p['horizon']))]
    if p['params']:
        header.append(('parameters', ' &nbsp; '.join(_mono(item) for item in p['params'])))
    header.append(('notation', f'<span style="{DIM}">{escape(NOTATION)}</span>'))
    out = [f'<div style="{STYLE}">',
           f'<div style="font-size:1.15em;font-weight:600">{escape(p["name"])}</div>', _rows(header)]

    out.append(_section('States', _block(
        escape(equation) + (f'<span style="{DIM}">   [initial mean {initial:g}]</span>'
                            if initial is not None else '')
        for equation, initial in p['states'])))
    if p['definitions']:
        out.append(_section('Definitions', _block(escape(d) for d in p['definitions'])))

    header_row = ''.join(f'<th style="{HEAD}">{label}</th>' for label in ('Agent', 'Controls', 'Observes', 'Flow loss'))
    body = ''
    for agent in p['agents']:
        signals = '<br>'.join(
            _mono(equation) + (f'<span style="{DIM}">&nbsp; observed with delay {delay:g}</span>' if delay else '')
            for equation, delay in agent['signals']) or f'<span style="{DIM}">nothing (no signal rows)</span>'
        body += (f'<tr><td style="{CELL}">{escape(agent["name"])}</td>'
                 f'<td style="{CELL}">{" ".join(_mono(c) for c in agent["controls"])}</td>'
                 f'<td style="{CELL}">{signals}</td>'
                 f'<td style="{CELL}">{_mono(agent["loss"])}</td></tr>')
    out.append(_section('Agents', f'<table style="border-collapse:collapse;width:100%">'
                                  f'<tr>{header_row}</tr>{body}</table>'))

    for group in p['ties']:
        out.append(_section('Shared strategy', ' &nbsp; '.join(_mono(name) for name in group)))
    if p['transition']:
        out.append(_section('Transition', _rows([('past', _mono(p['transition']['past'])),
                                                 ('continuation', _mono(p['transition']['continuation']))])))
    if p['notes']:
        out.append(_section('Notes and conventions', f'<ul style="margin:.2em 0;padding-left:1.2em;{DIM}">'
                            + ''.join(f'<li>{escape(note)}</li>' for note in p['notes']) + '</ul>'))
    return '\n'.join(out) + '\n</div>'


class ModelDescription(str):
    """Aligned plain text when printed; the same content as HTML in a notebook."""

    def __new__(cls, pieces):
        self = str.__new__(cls, _text(pieces))
        self._pieces = pieces
        return self

    def _repr_html_(self):
        return _html(self._pieces)


def describe(model):
    return ModelDescription(_pieces(model))
