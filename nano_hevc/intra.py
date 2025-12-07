"""
Intra prediction modes for HEVC.

HEVC defines 35 intra modes:
- Mode 0: Planar
- Mode 1: DC
- Modes 2-34: Angular (directional)

Angular mode geometry:
- Mode 10: Pure horizontal (angle = 0)
- Mode 26: Pure vertical (angle = 0)
- Mode 2-9: Horizontal-like, upward angles
- Mode 11-17: Horizontal-like, downward angles
- Mode 18: Diagonal (45 degrees)
- Mode 19-25: Vertical-like, leftward angles
- Mode 27-34: Vertical-like, rightward angles
"""

from __future__ import annotations
import numpy as np


# HEVC spec Table 8-4: intraPredAngle for modes 2-34
# Index 0-32 maps to modes 2-34
# Angle in 1/32 pixel units
INTRA_PRED_ANGLE = [
    32, 26, 21, 17, 13, 9, 5, 2, 0,      # modes 2-10
    -2, -5, -9, -13, -17, -21, -26, -32, # modes 11-17
    -26, -21, -17, -13, -9, -5, -2, 0,   # modes 18-25 (note: 18 starts new)
    2, 5, 9, 13, 17, 21, 26, 32          # modes 26-34 (note: 26 starts new)
]

# Inverse angles for reference sample extension (when angle < 0)
# invAngle = round(256 * 32 / angle) for negative angles
INV_ANGLE = {
    -2: -4096, -5: -1638, -9: -910, -13: -630,
    -17: -482, -21: -390, -26: -315, -32: -256
}


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


def intra_planar_predict(
    top: np.ndarray,
    left: np.ndarray,
    top_right: int,
    bottom_left: int,
    size: int
) -> np.ndarray:
    """
    Planar intra prediction for an NxN block.

    Creates smooth gradient via bilinear interpolation between edges.
    HEVC spec 8.4.4.2.4.

    Args:
        top: reference pixels above, shape (size,)
        left: reference pixels to left, shape (size,)
        top_right: pixel at top[size] position
        bottom_left: pixel at left[size] position
        size: block size (4, 8, 16, or 32)

    Returns:
        prediction block of shape (size, size), dtype int16
    """
    pred = np.empty((size, size), dtype=np.int16)
    log2_size = int(np.log2(size))

    for y in range(size):
        for x in range(size):
            h = (size - 1 - x) * left[y] + (x + 1) * top_right
            v = (size - 1 - y) * top[x] + (y + 1) * bottom_left
            pred[y, x] = (h + v + size) >> (log2_size + 1)

    return pred


def intra_angular_predict(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
    mode: int,
    size: int
) -> np.ndarray:
    """
    Angular intra prediction for an NxN block.

    Projects reference pixels along a direction defined by mode.
    HEVC spec 8.4.4.2.6.

    Args:
        top: reference pixels above, shape (2*size + 1,)
             top[0] is top-left, top[1..size] are above block
        left: reference pixels to left, shape (2*size + 1,)
              left[0] is top-left, left[1..size] are left of block
        top_left: corner pixel (redundant but explicit)
        mode: prediction mode 2-34
        size: block size (4, 8, 16, or 32)

    Returns:
        prediction block of shape (size, size), dtype int16
    """
    pred = np.empty((size, size), dtype=np.int16)
    angle = INTRA_PRED_ANGLE[mode - 2]

    # Modes 2-17 are horizontal-like, modes 18-34 are vertical-like
    is_vertical = mode >= 18

    if is_vertical:
        # Vertical modes: project onto top reference
        # For pixel (x, y): ref position = x + 1 + ((y+1)*angle >> 5)
        ref = _build_ref_array(top, left, top_left, angle, size)
        for y in range(size):
            for x in range(size):
                pred[y, x] = _project_sample_at(ref, x, y, angle, size)
    else:
        # Horizontal modes: project onto left reference
        # For pixel (x, y): ref position = y + 1 + ((x+1)*angle >> 5)
        ref = _build_ref_array(left, top, top_left, angle, size)
        for y in range(size):
            for x in range(size):
                pred[y, x] = _project_sample_at(ref, y, x, angle, size)

    return pred


def _build_ref_array(
    primary: np.ndarray,
    secondary: np.ndarray,
    corner: int,
    angle: int,
    size: int
) -> np.ndarray:
    """
    Build 1D reference array for angular prediction.

    For vertical modes: primary=top, secondary=left
    For horizontal modes: primary=left, secondary=top

    The reference array is indexed from -size to 2*size.
    Index 0 corresponds to the corner pixel.
    Positive indices are the primary reference.
    Negative indices are extended from secondary (when angle < 0).
    """
    # Allocate reference array: indices -size to 2*size
    # We use offset indexing: ref[i + size] accesses logical index i
    ref = np.zeros(3 * size + 1, dtype=np.int16)
    offset = size

    # Copy primary reference (indices 1 to 2*size)
    # primary[0] is corner, primary[1..2*size] are the reference pixels
    ref[offset] = corner
    for i in range(1, 2 * size + 1):
        if i < len(primary):
            ref[offset + i] = primary[i]
        else:
            ref[offset + i] = primary[-1]

    # Extend with secondary reference when angle is negative
    if angle < 0:
        inv_angle = INV_ANGLE[angle]
        # Number of samples to extend
        num_extend = (size * angle) >> 5
        for i in range(-1, num_extend - 1, -1):
            # Project back to secondary reference
            proj = ((i + 1) * inv_angle + 128) >> 8
            if proj < len(secondary):
                ref[offset + i] = secondary[proj]

    return ref


def _project_sample_at(
    ref: np.ndarray,
    base: int,
    scan: int,
    angle: int,
    size: int
) -> int:
    """
    Project a sample position onto reference array and interpolate.

    For vertical modes: base=x (column), scan=y (row)
    For horizontal modes: base=y (row), scan=x (column)

    Args:
        ref: reference array with offset indexing (ref[i + size] = logical i)
        base: base position in reference (column for vertical, row for horizontal)
        scan: scan position that determines angle offset
        angle: prediction angle in 1/32 units
        size: block size (used as offset)

    Returns:
        Interpolated pixel value
    """
    offset = size

    # Calculate projection offset: (scan + 1) * angle
    # Result is in 1/32 pixel units
    proj = (scan + 1) * angle

    # Integer and fractional parts
    int_part = proj >> 5
    frac = proj & 31

    # Reference position: base + 1 + integer offset
    ref_idx = offset + base + 1 + int_part

    if frac == 0:
        return ref[ref_idx]
    else:
        # Linear interpolation: (32 - frac) * ref[i] + frac * ref[i+1]
        return (
            (32 - frac) * ref[ref_idx] + frac * ref[ref_idx + 1] + 16
        ) >> 5
