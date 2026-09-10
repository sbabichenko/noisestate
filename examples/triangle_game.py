"""Three players pulling one two-dimensional state toward their own corner of a triangle.

Each player has a private signal on the state whose precision is theirs alone, and a control cost
that differs across players, so the game is asymmetric in both what the players know and what acting
costs them.  With ``flow=True`` there is also a public row per dimension whose drift is a weighted
sum of all three controls and which carries its own noise -- aggregate order flow.  Observing a
control means seeing only its predictable part, the Kyle-Back convention the package implements.

``vis`` sets how much of each player's control reaches that public row, so ``vis=(1, 0, 0)`` makes
player 1 the only visible actor.  Every quantity that a study varies is an argument: nothing here
reads a module-level default at call time.

    import examples.triangle_game as tg
    res = ns.solve(tg.model(p=(9.75, 2.62, 2.62)))
"""
import numpy as np

import noisestate as ns

P = (3.0, 10.0, 6.0)            # private signal precisions, one per player
R = (0.5, 1.0, 1.5)             # control costs, cheapest player first
V = ((1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254))     # the three vertices
SIG = 1.0                       # state noise
SIG_F = 1.0                     # noise on the public flow row


def model(kind="stationary", window=3.0, nodes=12, flow=False, p=P, r=R, sig_f=SIG_F,
          vis=(1.0, 1.0, 1.0)):
    chans = ["wx1", "wx2"] + [f"w{i + 1}{k + 1}" for i in range(3) for k in range(2)]
    d = {"name": "triangle_flow" if flow else "triangle", "channels": chans,
         "states": {f"X{k + 1}": {"drift": {f"D{i + 1}{k + 1}": 1.0 for i in range(3)},
                                  "noise": {f"wx{k + 1}": SIG}} for k in range(2)},
         "agents": {}, "horizon": {"kind": kind, "discount": 0.0, "window": window},
         "numerics": {"nodes": nodes}}
    for i, (pi, ri, v) in enumerate(zip(p, r, V)):
        loss, rows = [], {}
        for k in range(2):
            loss += [[0.5, f"X{k + 1}", f"X{k + 1}"], [-float(v[k]), f"X{k + 1}"],
                     [0.5 * ri, f"D{i + 1}{k + 1}", f"D{i + 1}{k + 1}"]]
            rows[f"y{i + 1}{k + 1}"] = {"drift": {f"X{k + 1}": float(np.sqrt(pi))},
                                        "noise": {f"w{i + 1}{k + 1}": 1.0}}
        d["agents"][f"player{i + 1}"] = {"controls": [f"D{i + 1}{k + 1}" for k in range(2)],
                                         "signals": rows, "loss": loss}
    base = ns.Model.from_dict(d)
    if not flow:
        return base
    return base.with_signals({
        f"flow{k + 1}": {"drift": {f"D{j + 1}{k + 1}": float(vis[j])
                                     for j in range(3) if vis[j] != 0.0},
                          "noise": {f"wf{k + 1}": sig_f}}
        for k in range(2)
    }, audience="all")
