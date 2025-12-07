"""
Quantization and dequantization for HEVC.

Quantization reduces transform coefficients to smaller integer values,
enabling efficient entropy coding. The Quantization Parameter (QP)
controls the quality-bitrate tradeoff:

- QP 0-51: Lower QP = higher quality, more bits
- QP increases by 6 roughly doubles the quantization step (halves quality)

HEVC uses a specific scaling formula:
- step_size ≈ 2^((QP-4)/6) for QP >= 4
- The actual implementation uses lookup tables for integer math
"""

from __future__ import annotations
import numpy as np


# HEVC quantization scaling factors (Table 8-10)
# Index = QP % 6, used with shift = QP // 6
# These are the "MF" (multiplication factor) values for forward quant
QUANT_SCALE = [26214, 23302, 20560, 18396, 16384, 14564]

# HEVC dequantization scaling factors
# Index = QP % 6, used with shift = QP // 6
DEQUANT_SCALE = [40, 45, 51, 57, 64, 72]


def get_qp_params(qp: int) -> tuple[int, int]:
    """
    Get quantization parameters from QP value.

    Args:
        qp: quantization parameter (0-51)

    Returns:
        (qp_per, qp_rem) where qp = qp_per * 6 + qp_rem
    """
    qp = max(0, min(51, qp))
    qp_per = qp // 6
    qp_rem = qp % 6
    return qp_per, qp_rem


def quantize(
    coeff: np.ndarray,
    qp: int,
    size: int,
    is_intra: bool = True
) -> np.ndarray:
    """
    Forward quantization of transform coefficients.

    Reduces coefficient magnitudes based on QP, with rounding toward zero
    (dead zone). Small coefficients become zero, enabling compression.

    Formula:
        level = sign(c) * ((|c| * MF + offset) >> shift)

    Where:
        MF = QUANT_SCALE[qp % 6]
        shift = 14 + qp // 6 + log2(size)
        offset = dead zone offset (smaller for intra to preserve quality)

    Args:
        coeff: transform coefficients, shape (size, size)
        qp: quantization parameter (0-51)
        size: block size (4, 8, 16, 32)
        is_intra: True for intra blocks (smaller dead zone)

    Returns:
        quantized levels, shape (size, size), dtype int32
    """
    qp_per, qp_rem = get_qp_params(qp)
    mf = QUANT_SCALE[qp_rem]

    # Shift includes QP scaling and block size normalization
    log2_size = int(np.log2(size))
    shift = 14 + qp_per + log2_size

    # Dead zone offset: intra uses 1/3, inter uses 1/6 of range
    # This affects how aggressively small coefficients are zeroed
    if is_intra:
        offset = (1 << shift) // 3
    else:
        offset = (1 << shift) // 6

    # Quantize with sign preservation
    sign = np.sign(coeff)
    abs_coeff = np.abs(coeff).astype(np.int64)

    # Forward quantization: (|c| * MF + offset) >> shift
    level = (abs_coeff * mf + offset) >> shift
    level = (sign * level).astype(np.int32)

    return level


def dequantize(
    level: np.ndarray,
    qp: int,
    size: int
) -> np.ndarray:
    """
    Inverse quantization (dequantization) of quantized levels.

    Reconstructs approximate transform coefficients from quantized levels.
    Note: This is lossy - the original coefficients cannot be recovered exactly.

    Formula:
        coeff = (level * scale * 16 + offset) >> shift

    Where:
        scale = DEQUANT_SCALE[qp % 6]
        shift = max(0, 4 - qp // 6)
        offset = rounding offset

    For large QP, coefficients are scaled UP:
        coeff = level * scale * 16 << (qp // 6 - 4)

    Args:
        level: quantized levels, shape (size, size)
        qp: quantization parameter (0-51)
        size: block size (4, 8, 16, 32)

    Returns:
        dequantized coefficients, shape (size, size), dtype int32
    """
    qp_per, qp_rem = get_qp_params(qp)
    scale = DEQUANT_SCALE[qp_rem]

    level = level.astype(np.int64)

    # Scale factor is scale * 16 = scale << 4
    base = level * scale

    if qp_per < 4:
        # For small QP: shift right with rounding
        shift = 4 - qp_per
        offset = 1 << (shift - 1)
        coeff = (base + offset) >> shift
    else:
        # For large QP: shift left (amplify)
        shift = qp_per - 4
        coeff = base << shift

    return coeff.astype(np.int32)


def quantize_block(
    coeff: np.ndarray,
    qp: int,
    is_intra: bool = True
) -> np.ndarray:
    """
    Convenience wrapper for block quantization.

    Automatically determines block size from input shape.
    """
    size = coeff.shape[0]
    return quantize(coeff, qp, size, is_intra)


def dequantize_block(
    level: np.ndarray,
    qp: int
) -> np.ndarray:
    """
    Convenience wrapper for block dequantization.

    Automatically determines block size from input shape.
    """
    size = level.shape[0]
    return dequantize(level, qp, size)


def estimate_bits(level: np.ndarray) -> int:
    """
    Rough estimate of bits needed to code quantized levels.

    Simple approximation: bits ≈ sum of log2(|level| + 1)
    Actual CABAC coding is more complex.

    Args:
        level: quantized levels

    Returns:
        estimated bits
    """
    abs_level = np.abs(level)
    # Each non-zero level needs roughly log2(|level|) + overhead bits
    bits = np.sum(np.log2(abs_level + 1) + (abs_level > 0) * 2)
    return int(bits)


def count_nonzero(level: np.ndarray) -> int:
    """Count non-zero quantized coefficients."""
    return int(np.count_nonzero(level))


def is_all_zero(level: np.ndarray) -> bool:
    """Check if all coefficients are zero (can skip coding)."""
    return np.all(level == 0)
