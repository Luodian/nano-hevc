"""
Block view abstraction for accessing rectangular regions of a plane.

Uses __slots__ for reduced memory overhead when iterating over many blocks.
"""

from __future__ import annotations
from typing import Tuple, Optional, Iterator
import numpy as np

from nano_hevc.frame import Plane


class BlockView:
    """
    A view into a rectangular block within a Plane (no data copy).

    Uses __slots__ to reduce memory overhead, which is important when
    creating many BlockView instances during block iteration.
    """
    __slots__ = ('plane', 'x', 'y', 'size')

    def __init__(self, plane: Plane, x: int, y: int, size: int):
        self.plane = plane
        self.x = x
        self.y = y
        self.size = size

    @property
    def pixels(self) -> np.ndarray:
        return self.plane.data[self.y:self.y + self.size,
                               self.x:self.x + self.size]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.size, self.size)

    def get_top_neighbors(self, count: Optional[int] = None) -> np.ndarray:
        """Get reference pixels from the row above this block."""
        n = count if count is not None else self.size
        if self.y == 0:
            return np.full(n, 128, dtype=self.plane.data.dtype)
        return self.plane.data[self.y - 1, self.x:self.x + n].copy()

    def get_left_neighbors(self, count: Optional[int] = None) -> np.ndarray:
        """Get reference pixels from the column to the left of this block."""
        n = count if count is not None else self.size
        if self.x == 0:
            return np.full(n, 128, dtype=self.plane.data.dtype)
        return self.plane.data[self.y:self.y + n, self.x - 1].copy()

    def get_top_left_neighbor(self) -> int:
        if self.y == 0 or self.x == 0:
            return 128
        return int(self.plane.data[self.y - 1, self.x - 1])

    def copy_pixels(self) -> np.ndarray:
        return self.pixels.copy()

    def write_pixels(self, data: np.ndarray) -> None:
        self.plane.data[self.y:self.y + self.size,
                        self.x:self.x + self.size] = data

    def __repr__(self) -> str:
        return f"BlockView(x={self.x}, y={self.y}, size={self.size})"


def iterate_blocks(plane: Plane, block_size: int) -> Iterator[BlockView]:
    """Iterate over all non-overlapping blocks in a plane."""
    for y in range(0, plane.height, block_size):
        for x in range(0, plane.width, block_size):
            actual_size = min(block_size, plane.height - y, plane.width - x)
            if actual_size == block_size:
                yield BlockView(plane=plane, x=x, y=y, size=block_size)
