"""Grids are pure geometry; share them between compiles so a sweep or a slider reuses their
operator caches (tensors, quadrature paths, shifts) instead of rebuilding them per point."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from .grid import AgeGrid
from .triangle import TriangleGrid


@lru_cache(maxsize=32)
def age_grid(breakpoints: Tuple[float, ...], nodes: int) -> AgeGrid:
    return AgeGrid(list(breakpoints), nodes)


@lru_cache(maxsize=32)
def triangle_grid(breakpoints: Tuple[float, ...], nt: int, na: int) -> TriangleGrid:
    return TriangleGrid(list(breakpoints), nt, na)


def clear():
    age_grid.cache_clear(); triangle_grid.cache_clear()
