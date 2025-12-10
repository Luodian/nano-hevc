"""Quality metrics for video codec evaluation."""

from __future__ import annotations
import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two arrays."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, peak: int = 255) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Higher is better. Typical range: 30-50 dB for video codecs.
    """
    err = mse(original, reconstructed)
    if err == 0:
        return float('inf')
    return 10 * np.log10(peak ** 2 / err)


def sad(a: np.ndarray, b: np.ndarray) -> int:
    """Sum of Absolute Differences. Lower is better."""
    return int(np.sum(np.abs(a.astype(np.int32) - b.astype(np.int32))))


def satd_4x4(a: np.ndarray, b: np.ndarray) -> int:
    """
    Sum of Absolute Transformed Differences using Hadamard transform.
    Better predictor of RD cost than SAD.
    """
    diff = (a.astype(np.int32) - b.astype(np.int32)).reshape(4, 4)
    H = np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        [1, -1, -1,  1],
        [1, -1,  1, -1],
    ], dtype=np.int32)

    transformed = H @ diff @ H.T
    return int(np.sum(np.abs(transformed)))


def residual_energy(residual: np.ndarray) -> int:
    """Sum of squared residual values. Lower means better prediction."""
    return int(np.sum(residual.astype(np.int64) ** 2))
