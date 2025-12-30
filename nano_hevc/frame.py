"""
Frame and Plane abstractions for YUV420p video data.

Memory optimization features:
- __slots__ to reduce per-instance memory overhead
- C-contiguous memory layout for cache-friendly access
- PackedFrame for contiguous Y/U/V allocation
- FrameBufferPool for buffer reuse
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np


class Plane:
    """
    A single color plane (Y, U, or V).

    Uses __slots__ to reduce memory overhead by ~40-50% compared to
    a regular class with __dict__.
    """
    __slots__ = ('data',)

    def __init__(self, data: np.ndarray):
        self.data = data

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
        """Create a zero-filled plane with C-contiguous memory layout."""
        return cls(data=np.zeros((height, width), dtype=dtype, order='C'))

    @classmethod
    def from_buffer(cls, buffer: bytes, height: int, width: int,
                    dtype: np.dtype = np.uint8) -> Plane:
        """Create a plane from raw bytes, ensuring C-contiguous layout."""
        data = np.frombuffer(buffer, dtype=dtype).reshape(height, width)
        # np.ascontiguousarray ensures C-contiguous and copies if needed
        return cls(data=np.ascontiguousarray(data))

    def __repr__(self) -> str:
        return f"Plane(shape={self.shape}, dtype={self.data.dtype})"


class Frame:
    """
    A video frame in YUV420p format.

    Uses __slots__ to reduce memory overhead.
    """
    __slots__ = ('y', 'u', 'v')

    def __init__(self, y: Plane, u: Plane, v: Plane):
        self.y = y
        self.u = u
        self.v = v

    @property
    def height(self) -> int:
        return self.y.height

    @property
    def width(self) -> int:
        return self.y.width

    @classmethod
    def zeros(cls, height: int, width: int, dtype: np.dtype = np.int16) -> Frame:
        """Create a zero-filled frame with C-contiguous planes."""
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

    def __repr__(self) -> str:
        return f"Frame(height={self.height}, width={self.width})"


class PackedFrame:
    """
    A video frame with contiguous memory layout for Y/U/V planes.

    Benefits:
    - Single allocation vs three separate allocations
    - Y/U/V planes are contiguous in memory, improving cache efficiency
    - Zero-copy export to YUV420p bytes

    Memory layout: [Y plane data][U plane data][V plane data]
    """
    __slots__ = ('_buffer', 'y', 'u', 'v', 'height', 'width', '_y_size', '_uv_size')

    def __init__(self, height: int, width: int, dtype: np.dtype = np.int16):
        self.height = height
        self.width = width

        self._y_size = height * width
        uv_height, uv_width = height // 2, width // 2
        self._uv_size = uv_height * uv_width
        total = self._y_size + 2 * self._uv_size

        # Single contiguous allocation for all planes
        self._buffer = np.zeros(total, dtype=dtype, order='C')

        # Views into the same buffer (no copy)
        self.y = self._buffer[:self._y_size].reshape(height, width)
        self.u = self._buffer[self._y_size:self._y_size + self._uv_size].reshape(uv_height, uv_width)
        self.v = self._buffer[self._y_size + self._uv_size:].reshape(uv_height, uv_width)

    @classmethod
    def from_yuv420p(cls, buffer: bytes, height: int, width: int) -> PackedFrame:
        """Create a PackedFrame from raw YUV420p bytes."""
        frame = cls(height, width, dtype=np.uint8)
        data = np.frombuffer(buffer, dtype=np.uint8)
        np.copyto(frame._buffer, data[:len(frame._buffer)])
        return frame

    @classmethod
    def from_frame(cls, frame: Frame) -> PackedFrame:
        """Create a PackedFrame from an existing Frame (copies data)."""
        packed = cls(frame.height, frame.width, dtype=frame.y.data.dtype)
        np.copyto(packed.y, frame.y.data)
        np.copyto(packed.u, frame.u.data)
        np.copyto(packed.v, frame.v.data)
        return packed

    def to_yuv420p(self) -> bytes:
        """
        Convert to raw YUV420p bytes.

        More efficient than Frame.to_yuv420p() as data is already contiguous.
        """
        return self._buffer.astype(np.uint8).tobytes()

    def to_frame(self) -> Frame:
        """Convert to a regular Frame (copies data)."""
        return Frame(
            y=Plane(data=self.y.copy()),
            u=Plane(data=self.u.copy()),
            v=Plane(data=self.v.copy()),
        )

    def clear(self) -> None:
        """Zero out all pixel data for reuse."""
        self._buffer.fill(0)

    def __repr__(self) -> str:
        return f"PackedFrame(height={self.height}, width={self.width}, dtype={self._buffer.dtype})"


class FrameBufferPool:
    """
    A pool of reusable frame buffers to avoid repeated allocation.

    Video encoders repeatedly allocate same-sized buffers. This pool
    reduces allocation overhead (syscall + page fault) by reusing buffers.

    Usage:
        pool = FrameBufferPool(1080, 1920, pool_size=4)

        # Acquire a buffer
        idx, frame = pool.acquire()

        # Use the frame...

        # Release back to pool
        pool.release(idx)
    """
    __slots__ = ('_pool', '_available', '_in_use', 'height', 'width', 'dtype')

    def __init__(self, height: int, width: int, pool_size: int = 4,
                 dtype: np.dtype = np.int16, use_packed: bool = True):
        """
        Initialize a frame buffer pool.

        Args:
            height: Frame height in pixels
            width: Frame width in pixels
            pool_size: Number of frames to pre-allocate
            dtype: NumPy dtype for pixel values
            use_packed: If True, use PackedFrame; otherwise use Frame
        """
        self.height = height
        self.width = width
        self.dtype = dtype

        if use_packed:
            self._pool: List[PackedFrame | Frame] = [
                PackedFrame(height, width, dtype=dtype)
                for _ in range(pool_size)
            ]
        else:
            self._pool = [
                Frame.zeros(height, width, dtype=dtype)
                for _ in range(pool_size)
            ]

        self._available: List[int] = list(range(pool_size))
        self._in_use: set[int] = set()

    def acquire(self, clear: bool = True) -> Tuple[int, PackedFrame | Frame]:
        """
        Acquire a frame buffer from the pool.

        Args:
            clear: If True, zero out the buffer before returning

        Returns:
            Tuple of (buffer_index, frame)

        Raises:
            RuntimeError: If no buffers are available
        """
        if not self._available:
            raise RuntimeError(
                f"No buffers available in pool. "
                f"In use: {len(self._in_use)}, Total: {len(self._pool)}"
            )

        idx = self._available.pop()
        self._in_use.add(idx)
        frame = self._pool[idx]

        if clear:
            if isinstance(frame, PackedFrame):
                frame.clear()
            else:
                frame.y.data.fill(0)
                frame.u.data.fill(0)
                frame.v.data.fill(0)

        return idx, frame

    def release(self, idx: int) -> None:
        """
        Release a frame buffer back to the pool.

        Args:
            idx: Buffer index returned by acquire()

        Raises:
            ValueError: If idx is not currently in use
        """
        if idx not in self._in_use:
            raise ValueError(f"Buffer {idx} is not currently in use")

        self._in_use.remove(idx)
        self._available.append(idx)

    @property
    def available_count(self) -> int:
        """Number of available buffers."""
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        """Number of buffers currently in use."""
        return len(self._in_use)

    @property
    def pool_size(self) -> int:
        """Total number of buffers in pool."""
        return len(self._pool)

    def __repr__(self) -> str:
        return (f"FrameBufferPool(height={self.height}, width={self.width}, "
                f"available={self.available_count}/{self.pool_size})")
