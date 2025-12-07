"""
Integer transforms for HEVC.

HEVC uses integer approximations of DCT (Discrete Cosine Transform) and
DST (Discrete Sine Transform) to convert spatial residuals to frequency
domain coefficients.

Transform sizes: 4x4, 8x8, 16x16, 32x32
Special case: 4x4 DST-VII for luma intra blocks

The transform is separable: forward = T @ X @ T.T
Scaling is applied via bit shifts to maintain integer precision.
"""

from __future__ import annotations
import numpy as np


# HEVC 4x4 DST-VII matrix (used for 4x4 luma intra)
# Spec Table 8-9
DST4 = np.array([
    [ 29,  55,  74,  84],
    [ 74,  74,   0, -74],
    [ 84, -29, -74,  55],
    [ 55, -84,  74, -29],
], dtype=np.int32)

# HEVC 4x4 DCT-II matrix
# Spec Table 8-8
DCT4 = np.array([
    [ 64,  64,  64,  64],
    [ 83,  36, -36, -83],
    [ 64, -64, -64,  64],
    [ 36, -83,  83, -36],
], dtype=np.int32)

# HEVC 8x8 DCT-II matrix
DCT8 = np.array([
    [ 64,  64,  64,  64,  64,  64,  64,  64],
    [ 89,  75,  50,  18, -18, -50, -75, -89],
    [ 83,  36, -36, -83, -83, -36,  36,  83],
    [ 75, -18, -89, -50,  50,  89,  18, -75],
    [ 64, -64, -64,  64,  64, -64, -64,  64],
    [ 50, -89,  18,  75, -75, -18,  89, -50],
    [ 36, -83,  83, -36, -36,  83, -83,  36],
    [ 18, -50,  75, -89,  89, -75,  50, -18],
], dtype=np.int32)

# HEVC 16x16 DCT-II matrix
DCT16 = np.array([
    [ 64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64],
    [ 90,  87,  80,  70,  57,  43,  25,   9,  -9, -25, -43, -57, -70, -80, -87, -90],
    [ 89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89],
    [ 87,  57,   9, -43, -80, -90, -70, -25,  25,  70,  90,  80,  43,  -9, -57, -87],
    [ 83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83],
    [ 80,   9, -70, -87, -25,  57,  90,  43, -43, -90, -57,  25,  87,  70,  -9, -80],
    [ 75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75],
    [ 70, -43, -87,   9,  90,  25, -80, -57,  57,  80, -25, -90,  -9,  87,  43, -70],
    [ 64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64],
    [ 57, -80, -25,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  25,  80, -57],
    [ 50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50],
    [ 43, -90,  57,  25, -87,  70,   9, -80,  80,  -9, -70,  87, -25, -57,  90, -43],
    [ 36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36],
    [ 25, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -25],
    [ 18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18],
    [  9, -25,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  25,  -9],
], dtype=np.int32)

# HEVC 32x32 DCT-II matrix (only even rows shown, odd rows follow pattern)
# For brevity, we generate it from the basis functions
def _generate_dct32() -> np.ndarray:
    """Generate 32x32 DCT matrix from HEVC coefficients."""
    # HEVC 32-point DCT coefficients (Table 8-8)
    c = [
        [ 64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
          64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64],
        [ 90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  46,  38,  31,  22,  13,   4,
          -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
        [ 90,  87,  80,  70,  57,  43,  25,   9,  -9, -25, -43, -57, -70, -80, -87, -90,
         -90, -87, -80, -70, -57, -43, -25,  -9,   9,  25,  43,  57,  70,  80,  87,  90],
        [ 90,  82,  67,  46,  22,  -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
          13,  38,  61,  78,  88,  90,  85,  73,  54,  31,   4, -22, -46, -67, -82, -90],
        [ 89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
          89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89],
        [ 88,  67,  31, -13, -54, -82, -90, -78, -46,  -4,  38,  73,  90,  85,  61,  22,
         -22, -61, -85, -90, -73, -38,   4,  46,  78,  90,  82,  54,  13, -31, -67, -88],
        [ 87,  57,   9, -43, -80, -90, -70, -25,  25,  70,  90,  80,  43,  -9, -57, -87,
         -87, -57,  -9,  43,  80,  90,  70,  25, -25, -70, -90, -80, -43,   9,  57,  87],
        [ 85,  46,  -13, -67, -90, -73, -22,  38,  82,  88,  54,  -4, -61, -90, -78, -31,
          31,  78,  90,  61,   4, -54, -88, -82, -38,  22,  73,  90,  67,  13, -46, -85],
        [ 83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83,
          83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83],
        [ 82,  22, -54, -90, -61,  13,  78,  85,  31, -46, -90, -67,   4,  73,  88,  38,
         -38, -88, -73,  -4,  67,  90,  46, -31, -85, -78, -13,  61,  90,  54, -22, -82],
        [ 80,   9, -70, -87, -25,  57,  90,  43, -43, -90, -57,  25,  87,  70,  -9, -80,
         -80,  -9,  70,  87,  25, -57, -90, -43,  43,  90,  57, -25, -87, -70,   9,  80],
        [ 78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  31,  90,  54, -38, -90, -46,
          46,  90,  38, -54, -90, -31,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78],
        [ 75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
          75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75],
        [ 73, -31, -90, -22,  78,  67, -38, -90, -13,  82,  61, -46, -88,  -4,  85,  54,
         -54, -85,   4,  88,  46, -61, -82,  13,  90,  38, -67, -78,  22,  90,  31, -73],
        [ 70, -43, -87,   9,  90,  25, -80, -57,  57,  80, -25, -90,  -9,  87,  43, -70,
         -70,  43,  87,  -9, -90, -25,  80,  57, -57, -80,  25,  90,   9, -87, -43,  70],
        [ 67, -54, -78,  38,  85, -22, -90,   4,  90,  13, -88, -31,  82,  46, -73, -61,
          61,  73, -46, -82,  31,  88, -13, -90,  -4,  90,  22, -85, -38,  78,  54, -67],
        [ 64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
          64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64],
        [ 61, -73, -46,  82,  31, -88, -13,  90,  -4, -90,  22,  85, -38, -78,  54,  67,
         -67, -54,  78,  38, -85, -22,  90,   4, -90,  13,  88, -31, -82,  46,  73, -61],
        [ 57, -80, -25,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  25,  80, -57,
         -57,  80,  25, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -25, -80,  57],
        [ 54, -85,  -4,  88, -46, -61,  82,  13, -90,  38,  67, -78, -22,  90, -31, -73,
          73,  31, -90,  22,  78, -67, -38,  90, -13, -82,  61,  46, -88,   4,  85, -54],
        [ 50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
          50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50],
        [ 46, -90,  38,  54, -90,  31,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78,
         -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -31,  90, -54, -38,  90, -46],
        [ 43, -90,  57,  25, -87,  70,   9, -80,  80,  -9, -70,  87, -25, -57,  90, -43,
         -43,  90, -57, -25,  87, -70,  -9,  80, -80,   9,  70, -87,  25,  57, -90,  43],
        [ 38, -88,  73,  -4, -67,  90, -46, -31,  85, -78,  13,  61, -90,  54,  22, -82,
          82, -22, -54,  90, -61, -13,  78, -85,  31,  46, -90,  67,   4, -73,  88, -38],
        [ 36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36,
          36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36],
        [ 31, -78,  90, -61,   4,  54, -88,  82, -38, -22,  73, -90,  67, -13, -46,  85,
         -85,  46,  13, -67,  90, -73,  22,  38, -82,  88, -54,  -4,  61, -90,  78, -31],
        [ 25, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -25,
         -25,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  25],
        [ 22, -61,  85, -90,  73, -38,  -4,  46, -78,  90, -82,  54, -13, -31,  67, -88,
          88, -67,  31,  13, -54,  82, -90,  78, -46,   4,  38, -73,  90, -85,  61, -22],
        [ 18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
          18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18],
        [ 13, -38,  61, -78,  88, -90,  85, -73,  54, -31,   4,  22, -46,  67, -82,  90,
         -90,  82, -67,  46, -22,  -4,  31, -54,  73, -85,  90, -88,  78, -61,  38, -13],
        [  9, -25,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  25,  -9,
          -9,  25, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -25,   9],
        [  4, -13,  22, -31,  38, -46,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90,
          90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  46, -38,  31, -22,  13,  -4],
    ]
    return np.array(c, dtype=np.int32)

DCT32 = _generate_dct32()


def _get_transform_matrix(size: int, use_dst: bool = False) -> np.ndarray:
    """Get the transform matrix for a given size."""
    if use_dst and size == 4:
        return DST4
    if size == 4:
        return DCT4
    elif size == 8:
        return DCT8
    elif size == 16:
        return DCT16
    elif size == 32:
        return DCT32
    else:
        raise ValueError(f"Unsupported transform size: {size}")


def forward_transform(residual: np.ndarray, use_dst: bool = False) -> np.ndarray:
    """
    Apply forward integer transform to residual block.

    HEVC uses a separable 2D transform:
    coefficients = T @ residual @ T.T

    The transform includes scaling via bit shifts for integer precision.
    Shift amount depends on transform size to handle matrix scaling.

    Args:
        residual: input residual block, shape (size, size)
        use_dst: use DST-VII instead of DCT-II (for 4x4 luma intra)

    Returns:
        transform coefficients, shape (size, size), dtype int32
    """
    size = residual.shape[0]
    T = _get_transform_matrix(size, use_dst)

    # Size-dependent shift for proper scaling
    # HEVC uses shift = log2(size) + 5 for forward transform passes
    log2_size = int(np.log2(size))
    shift1 = log2_size + 5
    rnd1 = 1 << (shift1 - 1)

    # Convert to int32 for intermediate calculations
    block = residual.astype(np.int32)

    # First pass: transform rows (T @ block)
    temp = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            acc = 0
            for k in range(size):
                acc += T[i, k] * block[k, j]
            temp[i, j] = (acc + rnd1) >> shift1

    # Second pass: transform columns (temp @ T.T)
    shift2 = shift1
    rnd2 = 1 << (shift2 - 1)
    coeff = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            acc = 0
            for k in range(size):
                acc += temp[i, k] * T[j, k]  # T.T[k,j] = T[j,k]
            coeff[i, j] = (acc + rnd2) >> shift2

    return coeff


def inverse_transform(coeff: np.ndarray, use_dst: bool = False) -> np.ndarray:
    """
    Apply inverse integer transform to coefficients.

    Reconstructs residual from transform coefficients:
    residual = T.T @ coefficients @ T

    Args:
        coeff: transform coefficients, shape (size, size)
        use_dst: use DST-VII instead of DCT-II (for 4x4 luma intra)

    Returns:
        reconstructed residual, shape (size, size), dtype int32
    """
    size = coeff.shape[0]
    T = _get_transform_matrix(size, use_dst)

    # Size-dependent shift matching forward transform
    log2_size = int(np.log2(size))
    shift1 = log2_size + 5
    rnd1 = 1 << (shift1 - 1)

    # Convert to int32
    block = coeff.astype(np.int32)

    # First pass: T.T @ coefficients (columns)
    temp = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            acc = 0
            for k in range(size):
                acc += T[k, i] * block[k, j]  # T.T[i,k] = T[k,i]
            temp[i, j] = (acc + rnd1) >> shift1

    # Second pass: temp @ T (rows)
    shift2 = shift1
    rnd2 = 1 << (shift2 - 1)
    residual = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            acc = 0
            for k in range(size):
                acc += temp[i, k] * T[k, j]
            residual[i, j] = (acc + rnd2) >> shift2

    return residual


def forward_transform_4x4(residual: np.ndarray, use_dst: bool = False) -> np.ndarray:
    """Forward 4x4 transform (convenience wrapper)."""
    return forward_transform(residual, use_dst)


def inverse_transform_4x4(coeff: np.ndarray, use_dst: bool = False) -> np.ndarray:
    """Inverse 4x4 transform (convenience wrapper)."""
    return inverse_transform(coeff, use_dst)


def forward_transform_8x8(residual: np.ndarray) -> np.ndarray:
    """Forward 8x8 DCT transform."""
    return forward_transform(residual, use_dst=False)


def inverse_transform_8x8(coeff: np.ndarray) -> np.ndarray:
    """Inverse 8x8 DCT transform."""
    return inverse_transform(coeff, use_dst=False)


def forward_transform_16x16(residual: np.ndarray) -> np.ndarray:
    """Forward 16x16 DCT transform."""
    return forward_transform(residual, use_dst=False)


def inverse_transform_16x16(coeff: np.ndarray) -> np.ndarray:
    """Inverse 16x16 DCT transform."""
    return inverse_transform(coeff, use_dst=False)


def forward_transform_32x32(residual: np.ndarray) -> np.ndarray:
    """Forward 32x32 DCT transform."""
    return forward_transform(residual, use_dst=False)


def inverse_transform_32x32(coeff: np.ndarray) -> np.ndarray:
    """Inverse 32x32 DCT transform."""
    return inverse_transform(coeff, use_dst=False)
