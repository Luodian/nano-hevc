"""
Intra prediction modes for HEVC.

HEVC defines 35 intra modes:
- Mode 0: Planar
- Mode 1: DC
- Modes 2-34: Angular (directional)
"""

from __future__ import annotations
import numpy as np


def intra_dc_predict_4x4(top: np.ndarray, left: np.ndarray) -> np.ndarray:
    """
    DC intra prediction for a 4x4 block.
    DC = floor((sum(top) + sum(left) + 4) / 8)
    """
    dc_value = (int(top.sum()) + int(left.sum()) + 4) >> 3
    return np.full((4, 4), dc_value, dtype=np.int16)


def intra_dc_predict(top: np.ndarray, left: np.ndarray, size: int) -> np.ndarray:
    """
    DC intra prediction for an NxN block.

    Fills the block with the average of neighboring reference pixels.
    Formula: DC = floor((sum(top) + sum(left) + size) / (2 * size))

    Args:
        top: reference pixels from row above, shape (size,)
        left: reference pixels from column to left, shape (size,)
        size: block size (4, 8, 16, or 32)

    Returns:
        prediction block of shape (size, size), dtype int16
    """
    dc_value = (int(top.sum()) + int(left.sum()) + size) // (2 * size)
    return np.full((size, size), dc_value, dtype=np.int16)


def residual_block(orig: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute residual block (original - prediction)."""
    return orig.astype(np.int16) - pred.astype(np.int16)


def reconstruct_block(pred: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Reconstruct block from prediction and residual."""
    return pred.astype(np.int16) + residual.astype(np.int16)


def clip_to_pixel_range(block: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Clip pixel values to valid range [0, 2^bit_depth - 1]."""
    max_val = (1 << bit_depth) - 1
    return np.clip(block, 0, max_val).astype(np.int16)
