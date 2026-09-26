"""Kernel: the ndarray res.kernel() returns, with its coordinates and the engine's own interpolant.

    k = res.kernel("X", "w0")       # (N,) on the stationary engine: the response at the shock ages k.axes["age"]
    k.at(0.7)                       # interpolated at age 0.7 (the grid's barycentric interpolant)
    k = res.kernel("X", "w0")       # a triangle result: (N,) at the nodes (k.axes["time"], k.axes["shock_time"])
    k.at(0.8, 0.3)                  # the response at time 0.8 to a shock at time 0.3 (the triangle's interpolant)
    k = res.kernel("X", "w0")       # the cell engine: the (N, N) matrix over (time, shock_time)
    k.at(0.8, 0.3)                  # the nearest cell (k.note says so); no interpolation on the cell scheme
    k.plot("k.png")                 # matplotlib, optional

A Kernel is an ndarray: indexing, arithmetic and .tolist() work as before; a slice or an arithmetic result
keeps the class but not a meaningful .at() (it checks the node count and refuses).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Kernel(np.ndarray):
    """An ndarray with .axes, .at(*coords), .plot(path=None); see the module docstring."""

    axes: dict = {}
    result = None
    name: str = ""
    shock: Optional[str] = None
    note: str = ""

    @classmethod
    def of(cls, values, result, name: str, shock: Optional[str]) -> "Kernel":
        k = np.asarray(values).view(cls)
        k.result = result; k.name = name; k.shock = shock
        k.axes = {a: np.asarray(v) for a, v in result._node_axes().items()}
        k.note = getattr(result, "KERNEL_NOTE", "")
        return k

    @property
    def values(self) -> np.ndarray:
        """The plain ndarray."""
        return np.asarray(self)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for a in ("axes", "result", "name", "shock", "note"):
            setattr(self, a, getattr(obj, a, getattr(type(self), a)))

    def __reduce__(self):                          # pickle as a plain array (the result is not pickled with it)
        return (np.asarray, (np.asarray(self),))

    def _nodes(self) -> int:
        return len(next(iter(self.axes.values()))) if self.axes else -1

    def at(self, *coords):
        """The kernel interpolated at the coordinates of res.axes, with the engine's own interpolant: at(age)
        on the stationary engine; at(t, s) (time, shock time; arrays broadcast) on the finite triangle and on
        a transition; the nearest cell on the cell engine (self.note).  Returns one value per point, or one
        row per point over the shocks when the kernel holds every shock."""
        r = self.result
        if r is None or self.shape[0] != self._nodes():
            raise ValueError("at() needs a kernel as res.kernel() returned it (a slice or a product has lost its nodes)")
        K = np.asarray(self)
        names = [a for a in self.axes]
        if names == ["age"]:
            if len(coords) != 1:
                raise TypeError("at(age) on the stationary engine")
            age = np.atleast_1d(np.asarray(coords[0], dtype=float))
            out = r.compiled.grid.interp(age) @ K
        elif "shock_time" in names and "age" in names:
            if len(coords) != 2:
                raise TypeError("at(t, s): the time and the shock time")
            t, s = np.broadcast_arrays(np.atleast_1d(np.asarray(coords[0], dtype=float)), np.atleast_1d(np.asarray(coords[1], dtype=float)))
            out = r.grid.interp(t.ravel(), (t - s).ravel()) @ K
            out = out.reshape(t.shape + K.shape[1:])
        else:                                      # the cell matrices: nearest node on each axis
            if len(coords) != 2:
                raise TypeError("at(t, s): the time and the shock time")
            t, s = np.broadcast_arrays(np.atleast_1d(np.asarray(coords[0], dtype=float)), np.atleast_1d(np.asarray(coords[1], dtype=float)))
            times = self.axes["time"]
            i = np.abs(times[None, :] - t.ravel()[:, None]).argmin(axis=1)
            j = np.abs(times[None, :] - s.ravel()[:, None]).argmin(axis=1)
            out = K[i, j].reshape(t.shape)
        return out[0] if np.ndim(coords[0]) == 0 and (len(coords) == 1 or np.ndim(coords[1]) == 0) else out

    def plot(self, path: Optional[str] = None):
        """Plot the kernel (matplotlib, optional): against age on the stationary engine, against the shock time
        at five dates on the triangle, as an image on the cell engine.  Saved to `path` when given; returns
        the figure."""
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        K = np.asarray(self); names = [a for a in self.axes]
        label = self.name + (f" / {self.shock}" if self.shock else "")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        if names == ["age"]:
            ax.plot(self.axes["age"], K, lw=1.2)
            ax.set_xlabel("age"); ax.set_ylabel(label)
            if K.ndim == 2 and self.result is not None:
                ax.legend(self.result.shocks, fontsize=7)
        elif "shock_time" in names and "age" in names:
            T = float(self.axes["time"].max())
            for t in np.linspace(0.2, 1.0, 5) * T:
                s = np.linspace(0, t, 200)
                v = self.at(np.full_like(s, t), s)
                ax.plot(s, v if v.ndim == 1 else v[:, 0], lw=1.0, label=f"t = {t:.2g}")
            ax.set_xlabel("shock time"); ax.set_ylabel(label); ax.legend(fontsize=7)
        else:
            shown = K if K.ndim == 2 else K[..., 0]
            t = self.axes["time"]
            s = self.axes["shock_time"]
            future = s[None, :] > t[:, None] + 1e-12
            shown = np.ma.array(shown, mask=future)
            cmap = plt.get_cmap("RdBu_r").with_extremes(bad="0.88")
            scale = float(np.max(np.abs(np.asarray(K if K.ndim == 2 else K[..., 0])[~future]))) if np.any(~future) else 0.0
            im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cmap,
                           **({"vmin": -scale, "vmax": scale} if scale > 0 else {"vmin": -1, "vmax": 1}),
                           extent=[self.axes["shock_time"][0], self.axes["shock_time"][-1], self.axes["time"][0], self.axes["time"][-1]])
            if scale > 0:
                fig.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, "zero on causal cells", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_xlabel("shock time"); ax.set_ylabel("time"); ax.set_title(label)
        if self.result is not None:
            failed = [row["name"] for row in self.result.diagnostics.rows if row.get("ok") is False]
            if failed:
                fig.suptitle("WARNING: failed checks — " + ", ".join(failed), fontsize=10)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130); plt.close(fig)
        return fig
