"""
Frame and Plane abstractions for YUV420p video data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Plane:
    """A single color plane (Y, U, or V)."""
    data: np.ndarray

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape[:2]

    @classmethod
    def zeros(cls, height: int, width: int, dtype: np.dtype = np.int16) -> Plane:
        return cls(data=np.zeros((height, width), dtype=dtype))

    @classmethod
    def from_buffer(cls, buffer: bytes, height: int, width: int,
                    dtype: np.dtype = np.uint8) -> Plane:
        data = np.frombuffer(buffer, dtype=dtype).reshape(height, width)
        return cls(data=data.copy())


@dataclass
class Frame:
    """A video frame in YUV420p format."""
    y: Plane
    u: Plane
    v: Plane

    @property
    def height(self) -> int:
        return self.y.height

    @property
    def width(self) -> int:
        return self.y.width

    @classmethod
    def zeros(cls, height: int, width: int, dtype: np.dtype = np.int16) -> Frame:
        return cls(
            y=Plane.zeros(height, width, dtype),
            u=Plane.zeros(height // 2, width // 2, dtype),
            v=Plane.zeros(height // 2, width // 2, dtype),
        )

    @classmethod
    def from_yuv420p(cls, buffer: bytes, height: int, width: int) -> Frame:
        """
        Create a Frame from raw YUV420p bytes.

        Args:
            buffer: raw bytes in planar YUV420p layout (Y, then U, then V)
            height: luma height in pixels
            width: luma width in pixels

        Returns:
            Frame with Y at full resolution, U/V at half resolution
        """
        y_size = height * width
        uv_height = height // 2
        uv_width = width // 2
        uv_size = uv_height * uv_width

        return cls(
            y=Plane.from_buffer(buffer[:y_size], height, width),
            u=Plane.from_buffer(buffer[y_size:y_size + uv_size], uv_height, uv_width),
            v=Plane.from_buffer(buffer[y_size + uv_size:y_size + 2 * uv_size], uv_height, uv_width),
        )

    def to_yuv420p(self) -> bytes:
        """Convert Frame back to raw YUV420p bytes."""
        return (self.y.data.astype(np.uint8).tobytes() +
                self.u.data.astype(np.uint8).tobytes() +
                self.v.data.astype(np.uint8).tobytes())
