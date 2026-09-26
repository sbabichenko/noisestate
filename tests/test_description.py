"""Descriptions expose the actual model without running an equilibrium solve.

The printed form is plain text; the notebook form is HTML.  Both come from the
same extracted pieces, so the tests check the pair together.
"""
from xml.etree import ElementTree

import noisestate as ns


def _model():
    w = ns.shocks('w0', 'w1')
    x, u = ns.State('X'), ns.Control('D')
    x.d = (-x + u) * ns.dt + 2 * w.w0
    agent = ns.Agent('player', controls=[u],
                     observes=ns.Signal('y', x * ns.dt + w.w1, delay=0.5),
                     loss=(x - 1)**2 + u**2)
    return ns.Game(x, agent, window=3, name='tracking')


def _parsed(html):
    """Well-formedness, with the two HTML spellings XML does not share."""
    return ElementTree.fromstring(html.replace('&nbsp;', '&#160;').replace('<br>', '<br/>'))


def test_description_information_and_loss():
    model = _model()
    before = model.to_dict()
    summary = model.describe()
    assert 'dX = (-X + D) dt + 2 dw0' in summary
    assert 'dy = X dt + dw1' in summary
    assert 'observed with delay 0.5' in summary
    assert 'X^2 - 2 X + D^2' in summary
    assert model.to_dict() == before


def test_printed_description_is_plain_text():
    summary = _model().describe()
    assert not any(markup in summary for markup in ('<br>', '&#', '**', '|', '`'))
    assert all(len(line) <= 88 for line in summary.splitlines())


def test_notebook_description_is_html_carrying_the_same_content():
    summary = _model().describe()
    html = summary._repr_html_()
    _parsed(html)
    text = ''.join(_parsed(html).itertext())
    for fragment in ('tracking', 'dX = (-X + D) dt + 2 dw0', 'dy = X dt + dw1',
                     'observed with delay 0.5', 'X^2 - 2 X + D^2'):
        assert fragment in text
    assert 'Flow loss' in text
    assert not hasattr(summary, '_repr_markdown_')


def test_description_uses_current_fields_and_exposes_control_notes():
    model = ns.load('examples/ch4_kyle_back.yaml')
    model.states[0].noise['wV'] = 7
    summary = model.describe()
    flat = ' '.join(summary.split())
    assert '7 dwV' in flat and '7 dwV' in ''.join(_parsed(summary._repr_html_()).itertext())
    assert 'predictable' in flat
    assert 'myopic' in flat
    assert 'flow loss' in flat


def test_description_names_the_transition_past_and_shared_strategies():
    summary = ns.load('examples/ch3_precision_change.yaml').describe()
    assert 'Transition' in summary and 'ch3_two_player.yaml' in summary
    assert 'continuation  stationary' in summary
    assert 'ch3_two_player.yaml' in ''.join(_parsed(summary._repr_html_()).itertext())
    cycle = ns.load('examples/ch5_cycle_market.yaml').describe()
    assert 'Shared strategy\n  firm0, firm1, firm2' in cycle
    assert 'firm0' in ''.join(_parsed(cycle._repr_html_()).itertext())
