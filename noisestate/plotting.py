"""Rendering: every figure the package draws, from a live Result or from a saved payload.

This is the only module that imports matplotlib, which is why it is a module.  It sat inside
results.py as 266 lines under the four Result classes plus a plot() body in each of them, so a file
about what a solve returns was a fifth rendering code and matplotlib was reachable from anywhere
that touched a result.  Nothing here is imported at package import time; `_pyplot()` raises the
install hint if matplotlib is missing.

Two entry points, and they are not the same thing:

    plot_<engine>(res, path)     from a live Result -- it can evaluate the kernel anywhere, so the
                                 curves are drawn at the engine's own interpolant
    plot_payload(payload, path)  from the JSON alone -- no solve, no engine, only what was saved

`noisestate plot result.json out.pdf` takes the second (it used to re-solve); `--re-solve` asks for
the first.  Kernel.plot() draws one kernel and lives with the Kernel, in kernel.py.
"""
from __future__ import annotations

import numpy as np


def plot_stationary(res, path: str) -> None:
    """Kernels by shock for every state and control, one panel per quantity (needs matplotlib)."""
    plt = _pyplot(); c = res.compiled
    names = res.model.state_names + res.model.control_names
    nrow = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(10.4, 2.6 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        K = res.kernel(name)
        for k, ch in enumerate(res.shocks):
            if np.abs(K[:, k]).max() > 1e-12:
                ax.plot(c.grid.nodes, K[:, k], lw=1.1, label=ch)
        ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name}: kernel by shock", fontsize=10); ax.set_xlabel("shock age")
        ax.legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(_result_plot_title(res), fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150)

def plot_triangle(res, path: str) -> None:
    """Each kernel as a function of the shock time s at five dates t (needs matplotlib)."""
    T = res.compiled.T
    def curves(name, ch):
        for t in np.linspace(0.2, 1.0, 5) * T:
            s = np.linspace(0, t, 200)
            yield t, s, res.evaluate(name, ch, np.full_like(s, t), s)
    _plot_by_shock_time(res, curves, path)

def plot_transition(res, path: str) -> None:
    """The kernels against the shock time s from -L at five dates (the band s < 0 shaded), a row with
    E[loss(t)] per agent and the old and new stationary flows as horizontal lines, a row with each agent's
    belief-error variance of every state, and the mean paths when driven (needs matplotlib)."""
    _plot_transition(res, path)

def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("plotting needs matplotlib: pip install 'noisestate[plot]'") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_title(name: str, residual: float, rows, suffix: str = "") -> str:
    """A figure title that cannot lose the result's numerical trust verdict when it is shared alone."""
    failed = [row["name"] for row in rows if row.get("ok") is False]
    title = f"{name}  (residual {residual:.1e}{suffix})"
    return title + ("\nWARNING: failed checks — " + ", ".join(failed) if failed else "")


def _result_plot_title(res, suffix: str = "") -> str:
    return _plot_title(res.model.name, res.residual, res.diagnostics.rows, suffix)


def _mean_paths_row(axes, res, names) -> None:
    """The bottom row: every mean path on one axis, the rest of the row blank.  Drawn the same way
    from the transition figure and from the by-shock-time figure, which each had their own copy."""
    ax = axes[-1, 0]
    for name in names:
        ax.plot(res.mean_times, res.means[name], lw=1, label=name)
    ax.axhline(0, color="k", lw=0.4); ax.set_title("mean paths", fontsize=9); ax.set_xlabel("t")
    ax.legend(fontsize=6, frameon=False)
    for blank in axes[-1, 1:]:
        blank.axis("off")


def _plot_transition(res, path: str) -> None:
    plt = _pyplot()
    names = res.model.state_names + res.model.control_names; chans = res.shocks
    agents = [a.name for a in res.model.agents]; states = res.model.state_names
    c = res.compiled; g = res.grid; T = c.T; L = g.L or 0.0
    means = res.has_means and res.mean_times is not None
    ncol = max(len(chans), len(agents), 1)
    fig, axes = plt.subplots(len(names) + 2 + means, ncol, figsize=(3.6 * ncol, 2.5 * (len(names) + 2 + means)), squeeze=False)
    for i, name in enumerate(names):
        for k, ch in enumerate(chans):
            ax = axes[i, k]
            for t in np.linspace(0.2, 1.0, 5) * T:
                s = np.linspace(max(-L, t - L) if L else 0.0, t, 300)
                ax.plot(s, res.evaluate(name, ch, np.full_like(s, t), s), lw=1, label=f"t={t:.2f}")
            if L:
                ax.axvspan(-L, 0.0, color="0.85", alpha=0.6, lw=0)
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9); ax.set_xlabel("shock time s")
            if i == 0 and k == 0:
                ax.legend(fontsize=6, frameon=False)
        for ax in axes[i, len(chans):]:
            ax.axis("off")
    old, new = res.old_flows, res.new_flows
    for k, a in enumerate(agents):
        ax = axes[len(names), k]
        if a in res.loss_path:
            ax.plot(res.times, res.loss_path[a], lw=1, label="E[loss(t)]")
        if a in old:
            ax.axhline(old[a], color="C1", lw=0.8, ls="--", label="old flow")
        if a in new:
            ax.axhline(new[a], color="C2", lw=0.8, ls=":", label="new flow")
        ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{a}: expected loss", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
        ax = axes[len(names) + 1, k]
        for name in states:
            ax.plot(res.times, res.belief_error(a, name), lw=1, label=name)
        ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{a}: belief error variance", fontsize=9); ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
    for r in (len(names), len(names) + 1):
        for ax in axes[r, len(agents):]:
            ax.axis("off")
    if means:
        _mean_paths_row(axes, res, names)
    suffix = f", settled {res.settled:.1e}" if res.settled is not None else ""
    fig.suptitle(_result_plot_title(res, suffix), fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_by_shock_time(res, curves, path: str) -> None:
    """Finite horizon: one panel per (quantity, shock), the kernel against the shock time s at a few dates t;
    a last row with the mean paths against t when they are nonzero."""
    plt = _pyplot()
    names = res.model.state_names + res.model.control_names; chans = res.shocks
    means = res.has_means and res.mean_times is not None
    fig, axes = plt.subplots(len(names) + means, len(chans), figsize=(3.6 * len(chans), 2.5 * (len(names) + means)), squeeze=False)
    for i, name in enumerate(names):
        for k, ch in enumerate(chans):
            ax = axes[i, k]
            for t, s, y in curves(name, ch):
                ax.plot(s, y, lw=1, label=f"t={t:.2f}")
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9); ax.set_xlabel("shock time s")
            if i == 0 and k == 0:
                ax.legend(fontsize=6, frameon=False)
    if means:
        _mean_paths_row(axes, res, names)
    fig.suptitle(_result_plot_title(res), fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150)


def _terminal_time(payload: dict) -> float:
    """The end of the time axis a triangle payload is drawn on.

    Not horizon.window.  0.8 split the lag-truncation length L from the terminal time T, and a
    finite horizon carries only T -- so `payload["horizon"]["window"]` raised KeyError here and
    `noisestate plot` could not draw a finite or cell payload at all.  A transition keeps its T (for a
    march, the T it found) in the top-level `T` key (`window` before 1.1).
    """
    if payload["kind"] == "transition":
        return float(payload["T"] if "T" in payload else payload["window"])
    horizon = payload.get("horizon") or {}
    for key in ("T", "window"):                  # "window" only for a payload written before 0.8
        if horizon.get(key) is not None:
            return float(horizon[key])
    raise KeyError("the payload's horizon carries neither T nor window: cannot place the time axis")


def plot_payload(payload: dict, path: str):
    """Plot a saved result from the payload alone, without re-solving it.

    `to_dict()` stores every kernel by name and shock, and the node coordinates beside them (`age` on
    the stationary grid, `time`/`age`/`shock_time` on the triangle).  Curves are drawn through the nodes
    the solve actually produced, not through an interpolant of them: on the triangle that means the
    shock-time slices are the node rows at a few dates, so a jump at a delay line shows where it is instead
    of being smoothed across.  A transition's expected losses, belief errors and means are serialized beside
    its kernels and plotted from those saved paths too.

    Raises KeyError if the payload predates the fields it needs; the caller can fall back to re-solving.
    """
    plt = _pyplot()
    kernels = payload["kernels"]; chans = payload["shocks"]; axes_of = payload["axes"]
    names = list(kernels)                                             # a zero kernel is still a result
    suffix = ", from the saved result"

    if payload["kind"] == "stationary":
        age = np.asarray(axes_of["age"], dtype=float)
        nrow = (len(names) + 1) // 2
        fig, axs = plt.subplots(nrow, 2, figsize=(10.4, 2.6 * nrow), squeeze=False)
        for ax, name in zip(axs.ravel(), names):
            for ch in chans:
                K = np.asarray(kernels[name][ch], dtype=float)
                if np.abs(K).max() > 1e-12:
                    ax.plot(age, K, lw=1.1, label=ch)
            ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name}: kernel by shock", fontsize=10)
            ax.set_xlabel("shock age")
            if ax.lines[1:]:
                ax.legend(fontsize=7, frameon=False, ncol=2)
        for ax in axs.ravel()[len(names):]:
            ax.axis("off")
    else:
        t = np.asarray(axes_of["time"], dtype=float); s = np.asarray(axes_of["shock_time"], dtype=float)
        T = _terminal_time(payload)
        available = np.unique(np.round(t[t <= T + 1e-12], 12))
        targets = np.linspace(0.2, 1.0, min(5, len(available))) * T if len(available) else np.array([])
        dates = np.unique([available[np.abs(available - x).argmin()] for x in targets]) if len(available) else available
        transition = payload["kind"] == "transition"
        agents = list(payload["agents"]); states = list(payload["model"].get("states", {}))
        means = payload.get("mean_times") is not None and any(np.any(np.asarray(v, dtype=float)) for v in payload.get("means", {}).values())
        extra_rows = (2 if transition else 0) + (1 if means else 0)
        ncol = max(len(chans), len(agents) if transition else 0, 1)
        fig, axs = plt.subplots(len(names) + extra_rows, ncol,
                                figsize=(3.6 * ncol, 2.5 * (len(names) + extra_rows)), squeeze=False)
        L = float(payload.get("grid", {}).get("window", 0.0) or 0.0)
        for i, name in enumerate(names):
            for k, ch in enumerate(chans):
                ax = axs[i, k]; K = np.asarray(kernels[name][ch], dtype=float)
                for d in dates:
                    sel = np.abs(t - d) < 1e-12
                    if sel.sum() > 1:
                        o = np.argsort(s[sel], kind="stable")
                        ax.plot(s[sel][o], K[sel][o], lw=1, marker=".", ms=2.5, label=f"t={d:.2f}")
                if transition and L:
                    ax.axvspan(-L, 0.0, color="0.85", alpha=0.6, lw=0)
                ax.axhline(0, color="k", lw=0.4); ax.set_title(f"{name} on {ch}", fontsize=9)
                ax.set_xlabel("shock time s (saved nodes)")
                if i == 0 and k == 0:
                    ax.legend(fontsize=6, frameon=False)
            for ax in axs[i, len(chans):]:
                ax.axis("off")
        row = len(names)
        if transition:
            times = np.asarray(payload["times"], dtype=float)
            old, new, losses = payload.get("old_flows", {}), payload.get("new_flows", {}), payload.get("loss_path", {})
            for k, agent in enumerate(agents):
                ax = axs[row, k]
                if agent in losses:
                    ax.plot(times, losses[agent], lw=1, label="E[loss(t)]")
                if agent in old:
                    ax.axhline(old[agent], color="C1", lw=0.8, ls="--", label="old flow")
                if agent in new:
                    ax.axhline(new[agent], color="C2", lw=0.8, ls=":", label="new flow")
                ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{agent}: expected loss", fontsize=9)
                ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
            for ax in axs[row, len(agents):]:
                ax.axis("off")
            row += 1
            belief = payload["belief_error"]
            for k, agent in enumerate(agents):
                ax = axs[row, k]
                for name in states:
                    ax.plot(times, belief[agent][name], lw=1, label=name)
                ax.axvline(T, color="k", lw=0.4); ax.set_title(f"{agent}: belief error variance", fontsize=9)
                ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
            for ax in axs[row, len(agents):]:
                ax.axis("off")
            row += 1
            settled = payload.get("settled")
            if settled is not None:
                suffix += f", settled {settled:.1e}"
        if means:
            mt = np.asarray(payload["mean_times"], dtype=float); ax = axs[row, 0]
            for name in names:
                if name in payload["means"]:
                    ax.plot(mt, payload["means"][name], lw=1, label=name)
            ax.axhline(0, color="k", lw=0.4); ax.set_title("mean paths", fontsize=9)
            ax.set_xlabel("t"); ax.legend(fontsize=6, frameon=False)
            for ax in axs[row, 1:]:
                ax.axis("off")
    title = _plot_title(payload["name"], payload["residual"], payload.get("diagnostics", []), suffix)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return fig


def plot_sweep_payload(rows: list, path: str):
    """Plot the main decision measures from the JSON list written by the sweep CLI.

    `rows` are the DICTS in that file, not SweepPoints: this reads a payload, and a
    payload that has been through JSON has no attributes.
    """
    if not rows:
        raise ValueError("a sweep plot needs at least one row")
    plt = _pyplot()
    values = np.asarray([row["value"] for row in rows], dtype=float)
    results = [row["result"] for row in rows]
    param = rows[0].get("param", "parameter")
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.5), squeeze=False)

    agents = list(results[0].get("agents", {}))
    for agent in agents:
        costs = [result.get("costs", {}).get(agent, np.nan) for result in results]
        axs[0, 0].plot(values, costs, marker="o", ms=3, lw=1.1, label=agent)
    axs[0, 0].set_title("costs"); axs[0, 0].legend(fontsize=8, frameon=False)

    residuals = np.asarray([result.get("residual", np.nan) for result in results], dtype=float)
    axs[0, 1].semilogy(values, np.maximum(residuals, np.finfo(float).tiny), marker="o", ms=3, lw=1.1)
    axs[0, 1].set_title("solver residual")

    changes = np.asarray([np.nan if row.get("change") is None else row["change"] for row in rows], dtype=float)
    axs[1, 0].plot(values, changes, marker="o", ms=3, lw=1.1)
    jumps = np.asarray([bool(row.get("jump", False)) for row in rows])
    if np.any(jumps):
        axs[1, 0].scatter(values[jumps], changes[jumps], marker="x", s=65, linewidths=1.8,
                          color="crimson", label="possible branch jump", zorder=3)
        axs[1, 0].legend(fontsize=8, frameon=False)
    axs[1, 0].set_title("relative strategy change")

    seconds = [row.get("seconds", np.nan) for row in rows]
    axs[1, 1].plot(values, seconds, marker="o", ms=3, lw=1.1, label="seconds")
    axs[1, 1].set_title("solve time")
    for ax in axs.ravel():
        ax.set_xlabel(param); ax.grid(alpha=0.2)
    failed = [str(row["value"]) for row in rows if not row.get("converged", False)]
    title = f"sweep of {param} ({len(rows)} points)"
    if failed:
        title += "\nWARNING: unconverged values — " + ", ".join(failed)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return fig
