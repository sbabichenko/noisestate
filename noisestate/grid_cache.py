"""Grids are pure geometry; share them between compiles so a sweep or a slider reuses their
operator caches (tensors, quadrature paths, shifts) instead of rebuilding them per point.

The cache is bounded by count and by bytes: when the grids' operator caches (measured at each
request, since they fill lazily) exceed BUDGET_BYTES, the least recently used grids are dropped
until they fit (never the grid just requested)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np

from .grid import AgeGrid
from .triangle import TriangleGrid

BUDGET_BYTES = 1_500_000_000        # total footprint of the cached grids' tensors, paths and read matrices
MAXSIZE = 32


def grid_bytes(grid) -> int:
    """Bytes of the arrays a grid holds (its own attributes and the caches on it, sparse matrices by their
    data/indices/indptr), each buffer counted once (a view of a stored tensor adds nothing)."""
    seen = set(); total = 0
    stack = [grid]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            key = (base.__array_interface__["data"][0], base.nbytes)
            if key not in seen:
                seen.add(key); total += base.nbytes
        elif hasattr(obj, "indptr") and hasattr(obj, "data"):            # scipy sparse
            stack.extend([obj.data, obj.indices, obj.indptr])
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return total


class _GridCache:
    """An LRU of grids keyed by their constructor arguments, bounded by MAXSIZE entries and BUDGET_BYTES."""

    def __init__(self, factory):
        self.factory = factory
        self._grids: "OrderedDict[tuple, object]" = OrderedDict()

    def __call__(self, *key):
        if key in self._grids:
            self._grids.move_to_end(key)
        else:
            self._grids[key] = self.factory(*key)
        self._evict(key)
        return self._grids[key]

    def _evict(self, keep: tuple) -> None:
        while len(self._grids) > MAXSIZE:
            self._grids.popitem(last=False)
        sizes = {k: grid_bytes(g) for k, g in self._grids.items()}
        total = sum(sizes.values())
        for k in list(self._grids):                                    # least recently used first
            if total <= BUDGET_BYTES:
                break
            if k == keep:
                continue
            total -= sizes[k]; del self._grids[k]

    def cache_clear(self) -> None:
        self._grids.clear()

def _age_grid(breakpoints: Tuple[float, ...], nodes: int) -> AgeGrid:
    return AgeGrid(list(breakpoints), nodes)


def _triangle_grid(breakpoints: Tuple[float, ...], nt: int, na: int, T: float = None, window: float = None,
                   buffer: float = None) -> TriangleGrid:
    """The key is (breakpoints, nt, na) for today's triangle and (breakpoints, nt, na, T, L[, T_b]) for the strip
    of a transition (with the buffer's start T_b under a continuation), so a grid without a past is the same
    cached object it always was."""
    return TriangleGrid(list(breakpoints), nt, na, T=T, window=window, buffer=buffer)


age_grid = _GridCache(_age_grid)
triangle_grid = _GridCache(_triangle_grid)


def clear():
    age_grid.cache_clear(); triangle_grid.cache_clear()
