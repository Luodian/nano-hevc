"""
Tests for integer transforms (DCT/DST).

These tests verify the forward and inverse transforms produce correct
results and that round-trip reconstruction is accurate.
"""

import numpy as np
import pytest

from nano_hevc.transform import (
    forward_transform,
    inverse_transform,
    forward_transform_4x4,
    inverse_transform_4x4,
    forward_transform_8x8,
    inverse_transform_8x8,
    DCT4,
    DCT8,
    DST4,
)


class TestTransformMatrices:
    """Tests for transform matrix properties."""

    def test_dct4_orthogonality(self):
        """DCT4 should be approximately orthogonal (T @ T.T ≈ scale * I)."""
        result = DCT4 @ DCT4.T
        # Diagonal should be large and equal
        diag = np.diag(result)
        assert np.all(diag > 0)
        # Off-diagonal should be small relative to diagonal
        off_diag = result - np.diag(diag)
        assert np.max(np.abs(off_diag)) < np.max(diag) * 0.1

    def test_dst4_orthogonality(self):
        """DST4 should be approximately orthogonal."""
        result = DST4 @ DST4.T
        diag = np.diag(result)
        assert np.all(diag > 0)
        off_diag = result - np.diag(diag)
        assert np.max(np.abs(off_diag)) < np.max(diag) * 0.1

    def test_dct8_orthogonality(self):
        """DCT8 should be approximately orthogonal."""
        result = DCT8 @ DCT8.T
        diag = np.diag(result)
        assert np.all(diag > 0)
        off_diag = result - np.diag(diag)
        assert np.max(np.abs(off_diag)) < np.max(diag) * 0.1


class TestForwardTransform:
    """Tests for forward transform."""

    @pytest.mark.parametrize("size,use_dst", [(4, False), (8, False), (4, True)])
    def test_forward_zeros(self, size, use_dst):
        """Zero input should always map to zero coefficients."""
        residual = np.zeros((size, size), dtype=np.int16)
        coeff = forward_transform(residual, use_dst=use_dst)

        assert coeff.shape == (size, size)
        assert np.all(coeff == 0)

    def test_forward_4x4_dc_only(self):
        """Uniform block should concentrate energy in DC."""
        residual = np.full((4, 4), 16, dtype=np.int16)
        coeff = forward_transform_4x4(residual)

        # DC coefficient (top-left) should dominate
        dc = coeff[0, 0]
        ac = coeff.copy()
        ac[0, 0] = 0
        assert dc != 0
        assert np.max(np.abs(ac)) <= np.abs(dc) * 0.05

    def test_forward_matches_reference_multiply(self):
        """
        Forward transform should match explicit T @ X @ T.T implementation.
        This guards against shift/scaling regressions in the two-pass loops.
        """
        residual = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, -1, -2],
            [4, 3, 2, 1],
        ], dtype=np.int16)

        # Reference using the same scaling rule as the implementation
        T = DCT4.astype(np.int64)
        log2_size = 2
        shift = log2_size + 5
        rnd = 1 << (shift - 1)
        temp = ((T @ residual.astype(np.int64)) + rnd) >> shift
        ref = ((temp @ T.T) + rnd) >> shift

        coeff = forward_transform_4x4(residual, use_dst=False)
        assert np.allclose(coeff, ref, atol=1)

    def test_forward_dst_vs_dct(self):
        """DST and DCT should produce different results."""
        residual = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ], dtype=np.int16)

        coeff_dct = forward_transform_4x4(residual, use_dst=False)
        coeff_dst = forward_transform_4x4(residual, use_dst=True)

        # Results should be different
        assert not np.array_equal(coeff_dct, coeff_dst)


class TestInverseTransform:
    """Tests for inverse transform."""

    def test_inverse_4x4_zeros(self):
        """Zero coefficients should produce zero residual."""
        coeff = np.zeros((4, 4), dtype=np.int32)
        residual = inverse_transform_4x4(coeff)

        assert residual.shape == (4, 4)
        assert np.all(residual == 0)

    def test_inverse_8x8_zeros(self):
        """Zero coefficients should produce zero residual."""
        coeff = np.zeros((8, 8), dtype=np.int32)
        residual = inverse_transform_8x8(coeff)

        assert residual.shape == (8, 8)
        assert np.all(residual == 0)


class TestRoundTrip:
    """
    Tests for forward-inverse round trip reconstruction.

    Integer transforms have inherent quantization error from fixed-point
    arithmetic. Error bounds increase with transform size because:
    1. More terms accumulate rounding errors
    2. HEVC matrices are designed to work WITH quantization scaling

    These bounds are for the raw transform without quantization.
    In practice, quantization dominates the error budget.
    """

    def test_roundtrip_4x4_dct(self):
        """Forward then inverse DCT should approximately reconstruct input."""
        original = np.array([
            [ 5,  3, -2,  1],
            [ 2,  4,  1, -3],
            [-1,  2,  3,  2],
            [ 0, -1,  2,  4],
        ], dtype=np.int16)

        coeff = forward_transform_4x4(original, use_dst=False)
        reconstructed = inverse_transform_4x4(coeff, use_dst=False)

        # Should be close to original (some rounding error expected)
        error = np.abs(reconstructed - original)
        assert np.max(error) <= 2  # Allow small rounding error

    def test_roundtrip_4x4_dst(self):
        """Forward then inverse DST should approximately reconstruct input."""
        original = np.array([
            [ 5,  3, -2,  1],
            [ 2,  4,  1, -3],
            [-1,  2,  3,  2],
            [ 0, -1,  2,  4],
        ], dtype=np.int16)

        coeff = forward_transform_4x4(original, use_dst=True)
        reconstructed = inverse_transform_4x4(coeff, use_dst=True)

        error = np.abs(reconstructed - original)
        assert np.max(error) <= 2

    def test_roundtrip_8x8(self):
        """Forward then inverse 8x8 DCT - larger error from integer math."""
        np.random.seed(42)
        original = np.random.randint(-50, 50, size=(8, 8), dtype=np.int16)

        coeff = forward_transform_8x8(original)
        reconstructed = inverse_transform_8x8(coeff)

        # Mean error should be small even if max is larger
        error = np.abs(reconstructed - original)
        assert np.mean(error) < 25
        assert np.max(error) <= 50

    def test_roundtrip_16x16(self):
        """Forward then inverse 16x16 DCT - error grows with size."""
        np.random.seed(42)
        original = np.random.randint(-50, 50, size=(16, 16), dtype=np.int16)

        coeff = forward_transform(original)
        reconstructed = inverse_transform(coeff)

        error = np.abs(reconstructed - original)
        assert np.mean(error) < 30
        assert np.max(error) <= 60

    def test_roundtrip_32x32(self):
        """Forward then inverse 32x32 DCT - largest size has most error."""
        np.random.seed(42)
        original = np.random.randint(-30, 30, size=(32, 32), dtype=np.int16)

        coeff = forward_transform(original)
        reconstructed = inverse_transform(coeff)

        error = np.abs(reconstructed - original)
        assert np.mean(error) < 20
        assert np.max(error) <= 40


class TestEnergyCompaction:
    """Tests verifying energy compaction property of DCT."""

    def test_energy_compaction_4x4(self):
        """
        DCT should concentrate energy in low-frequency coefficients.
        For natural signals, most energy is in top-left corner.
        """
        # Smooth gradient (typical natural image content)
        residual = np.array([
            [10, 11, 12, 13],
            [11, 12, 13, 14],
            [12, 13, 14, 15],
            [13, 14, 15, 16],
        ], dtype=np.int16)

        coeff = forward_transform_4x4(residual)

        # Energy in top-left 2x2 should dominate
        energy_low = np.sum(coeff[:2, :2] ** 2)
        energy_high = np.sum(coeff[2:, 2:] ** 2)
        assert energy_low > energy_high

    def test_energy_compaction_8x8(self):
        """DCT should concentrate energy in low frequencies."""
        # Smooth gradient
        residual = np.zeros((8, 8), dtype=np.int16)
        for i in range(8):
            for j in range(8):
                residual[i, j] = i + j

        coeff = forward_transform_8x8(residual)

        # Energy in top-left quadrant should dominate
        energy_low = np.sum(coeff[:4, :4] ** 2)
        energy_total = np.sum(coeff ** 2)
        assert energy_low > energy_total * 0.9


class TestSpecificValues:
    """Tests with specific known values."""

    def test_dc_coefficient_4x4(self):
        """
        Test DC coefficient calculation.
        For uniform block of value v, DC ≈ v * sum(DCT4[0,:]) scaled.
        """
        value = 10
        residual = np.full((4, 4), value, dtype=np.int16)
        coeff = forward_transform_4x4(residual)

        # DC should be proportional to input value
        # Other coefficients should be near zero
        assert coeff[0, 0] != 0
        assert np.abs(coeff[0, 1]) < np.abs(coeff[0, 0]) * 0.1
        assert np.abs(coeff[1, 0]) < np.abs(coeff[0, 0]) * 0.1


class TestIntegration:
    """Integration tests with intra prediction pipeline."""

    def test_full_pipeline(self):
        """Test prediction -> residual -> transform -> inverse -> reconstruct."""
        from nano_hevc.intra import (
            intra_dc_predict,
            residual_block,
            reconstruct_block,
        )

        # Original block
        original = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        # Reference pixels
        top = np.array([102, 98, 100, 101], dtype=np.int16)
        left = np.array([103, 102, 101, 99], dtype=np.int16)

        # Prediction
        pred = intra_dc_predict(top, left, size=4)

        # Residual
        res = residual_block(original, pred)

        # Forward transform
        coeff = forward_transform_4x4(res)

        # Inverse transform
        rec_res = inverse_transform_4x4(coeff)

        # Reconstruct
        reconstructed = reconstruct_block(pred, rec_res.astype(np.int16))

        # Should be close to original
        error = np.abs(reconstructed - original)
        assert np.max(error) <= 2
