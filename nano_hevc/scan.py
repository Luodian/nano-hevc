"""
Coefficient scan patterns for HEVC.

HEVC uses three scan orders for transform coefficients:
- Diagonal scan (default for most cases)
- Horizontal scan (for horizontal intra modes)
- Vertical scan (for vertical intra modes)
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


SCAN_DIAG = 0
SCAN_HOR = 1
SCAN_VER = 2


def _generate_diagonal_scan(size: int) -> List[Tuple[int, int]]:
    """Generate diagonal up-right scan order for size x size block."""
    scan = []
    for diag in range(2 * size - 1):
        if diag < size:
            x_start, y_start = 0, diag
        else:
            x_start, y_start = diag - size + 1, size - 1

        x, y = x_start, y_start
        while x < size and y >= 0:
            scan.append((y, x))
            x += 1
            y -= 1
    return scan


def _generate_horizontal_scan(size: int) -> List[Tuple[int, int]]:
    """Generate horizontal (row-major) scan order."""
    return [(y, x) for y in range(size) for x in range(size)]


def _generate_vertical_scan(size: int) -> List[Tuple[int, int]]:
    """Generate vertical (column-major) scan order."""
    return [(y, x) for x in range(size) for y in range(size)]


_SCAN_CACHE = {}


def get_scan_order(size: int, scan_type: int = SCAN_DIAG) -> List[Tuple[int, int]]:
    """Get scan order as list of (row, col) tuples."""
    key = (size, scan_type)
    if key not in _SCAN_CACHE:
        if scan_type == SCAN_DIAG:
            _SCAN_CACHE[key] = _generate_diagonal_scan(size)
        elif scan_type == SCAN_HOR:
            _SCAN_CACHE[key] = _generate_horizontal_scan(size)
        elif scan_type == SCAN_VER:
            _SCAN_CACHE[key] = _generate_vertical_scan(size)
        else:
            raise ValueError(f"Unknown scan type: {scan_type}")
    return _SCAN_CACHE[key]


def scan_block(coeffs: np.ndarray, scan_type: int = SCAN_DIAG) -> np.ndarray:
    """Scan 2D coefficient block to 1D array."""
    size = coeffs.shape[0]
    scan = get_scan_order(size, scan_type)
    return np.array([coeffs[y, x] for y, x in scan], dtype=coeffs.dtype)


def inverse_scan_block(
    values: np.ndarray, size: int, scan_type: int = SCAN_DIAG
) -> np.ndarray:
    """Inverse scan: 1D array to 2D coefficient block."""
    scan = get_scan_order(size, scan_type)
    coeffs = np.zeros((size, size), dtype=values.dtype)
    for i, (y, x) in enumerate(scan):
        if i < len(values):
            coeffs[y, x] = values[i]
    return coeffs


def get_scan_type_for_intra_mode(mode: int, log2_size: int) -> int:
    """
    Determine scan type based on intra prediction mode.

    Per H.265 Table 7-3:
    - Modes 6-14: horizontal scan
    - Modes 22-30: vertical scan
    - Others: diagonal scan

    Only applies for 4x4 and 8x8 luma blocks.
    """
    if log2_size > 3:
        return SCAN_DIAG

    if 6 <= mode <= 14:
        return SCAN_HOR
    elif 22 <= mode <= 30:
        return SCAN_VER
    else:
        return SCAN_DIAG


def find_last_significant_coeff(coeffs: np.ndarray, scan_type: int = SCAN_DIAG) -> int:
    """Find position of last non-zero coefficient in scan order."""
    scanned = scan_block(coeffs, scan_type)
    nonzero_indices = np.nonzero(scanned)[0]
    if len(nonzero_indices) == 0:
        return -1
    return int(nonzero_indices[-1])


def get_coefficient_group_scan(size: int) -> List[Tuple[int, int]]:
    """
    Get scan order for 4x4 coefficient groups within larger blocks.

    For 8x8 and larger, coefficients are organized into 4x4 sub-blocks.
    Returns scan order of sub-block positions.
    """
    num_groups = size // 4
    return _generate_diagonal_scan(num_groups)
