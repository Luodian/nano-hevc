"""
Tests for quantization and dequantization.

Quantization is the primary source of compression loss in HEVC.
These tests verify the quantize/dequantize functions behave correctly
across different QP values and block sizes.
"""

import numpy as np

from nano_hevc.quant import (
    quantize,
    dequantize,
    quantize_block,
    dequantize_block,
    get_qp_params,
    count_nonzero,
    is_all_zero,
    QUANT_SCALE,
    DEQUANT_SCALE,
)


class TestQPParams:
    """Tests for QP parameter extraction."""

    def test_qp_params_low(self):
        """QP 0-5 should have qp_per=0."""
        for qp in range(6):
            qp_per, qp_rem = get_qp_params(qp)
            assert qp_per == 0
            assert qp_rem == qp

    def test_qp_params_mid(self):
        """QP 6-11 should have qp_per=1."""
        for qp in range(6, 12):
            qp_per, qp_rem = get_qp_params(qp)
            assert qp_per == 1
            assert qp_rem == qp - 6

    def test_qp_params_high(self):
        """QP 51 should have qp_per=8, qp_rem=3."""
        qp_per, qp_rem = get_qp_params(51)
        assert qp_per == 8
        assert qp_rem == 3

    def test_qp_params_clamp(self):
        """QP should be clamped to 0-51."""
        qp_per, qp_rem = get_qp_params(-5)
        assert qp_per == 0
        assert qp_rem == 0

        qp_per, qp_rem = get_qp_params(100)
        assert qp_per == 8
        assert qp_rem == 3


class TestQuantizeBasic:
    """Basic quantization tests."""

    def test_quantize_zeros(self):
        """Zero coefficients should produce zero levels."""
        coeff = np.zeros((4, 4), dtype=np.int32)
        level = quantize(coeff, qp=20, size=4)

        assert level.shape == (4, 4)
        assert np.all(level == 0)

    def test_quantize_small_to_zero(self):
        """Small coefficients should quantize to zero (dead zone)."""
        # With high QP, small values fall into dead zone
        coeff = np.full((4, 4), 5, dtype=np.int32)
        level = quantize(coeff, qp=40, size=4)

        # High QP should zero out small coefficients
        assert np.all(level == 0)

    def test_quantize_preserves_sign(self):
        """Quantization should preserve coefficient signs."""
        coeff = np.array([
            [ 100, -100,  50, -50],
            [-200,  200, -25,  25],
            [  75,  -75, 150, -150],
            [ -10,   10,   5,  -5],
        ], dtype=np.int32)

        level = quantize(coeff, qp=20, size=4)

        # Check signs are preserved for non-zero levels
        for i in range(4):
            for j in range(4):
                if level[i, j] != 0:
                    assert np.sign(level[i, j]) == np.sign(coeff[i, j])

    def test_quantize_high_qp_more_zeros(self):
        """Higher QP should produce more zeros."""
        coeff = np.random.randint(-100, 100, size=(4, 4), dtype=np.int32)
        coeff[0, 0] = 500  # Ensure DC survives

        level_low = quantize(coeff, qp=10, size=4)
        level_high = quantize(coeff, qp=40, size=4)

        assert count_nonzero(level_high) <= count_nonzero(level_low)


class TestDequantizeBasic:
    """Basic dequantization tests."""

    def test_dequantize_zeros(self):
        """Zero levels should produce zero coefficients."""
        level = np.zeros((4, 4), dtype=np.int32)
        coeff = dequantize(level, qp=20, size=4)

        assert coeff.shape == (4, 4)
        assert np.all(coeff == 0)

    def test_dequantize_nonzero(self):
        """Non-zero levels should produce non-zero coefficients."""
        level = np.array([
            [10, 0, 0, 0],
            [ 0, 5, 0, 0],
            [ 0, 0, 3, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.int32)

        coeff = dequantize(level, qp=20, size=4)

        # Non-zero levels should produce non-zero coefficients
        assert coeff[0, 0] != 0
        assert coeff[1, 1] != 0
        assert coeff[2, 2] != 0
        assert coeff[3, 3] != 0


class TestRoundTrip:
    """Tests for quantize-dequantize round trip."""

    def test_roundtrip_preserves_structure(self):
        """Round trip should preserve coefficient structure approximately."""
        original = np.array([
            [500, 100,  50,  20],
            [100,  80,  30,  10],
            [ 50,  30,  20,   5],
            [ 20,  10,   5,   2],
        ], dtype=np.int32)

        level = quantize(original, qp=20, size=4)
        reconstructed = dequantize(level, qp=20, size=4)

        # Large coefficients should be preserved (with some error)
        # DC coefficient especially should be close
        dc_error = abs(reconstructed[0, 0] - original[0, 0])
        assert dc_error < original[0, 0] * 0.5

    def test_roundtrip_low_qp_accuracy(self):
        """Low QP should give better reconstruction accuracy."""
        original = np.array([
            [200, 100, 50, 25],
            [100,  80, 40, 20],
            [ 50,  40, 30, 15],
            [ 25,  20, 15, 10],
        ], dtype=np.int32)

        level = quantize(original, qp=5, size=4)
        reconstructed = dequantize(level, qp=5, size=4)

        # Low QP: coefficients should be reasonably close
        error = np.abs(reconstructed - original)
        assert np.mean(error) < 50

    def test_roundtrip_high_qp_lossy(self):
        """High QP introduces significant quantization error."""
        original = np.full((4, 4), 100, dtype=np.int32)

        level = quantize(original, qp=45, size=4)
        reconstructed = dequantize(level, qp=45, size=4)

        # High QP: expect significant differences
        # But DC should still have the right sign
        if reconstructed[0, 0] != 0:
            assert np.sign(reconstructed[0, 0]) == np.sign(original[0, 0])


class TestBlockSizes:
    """Tests for different block sizes."""

    def test_quantize_8x8(self):
        """Test 8x8 block quantization."""
        coeff = np.random.randint(-200, 200, size=(8, 8), dtype=np.int32)
        level = quantize(coeff, qp=20, size=8)

        assert level.shape == (8, 8)
        assert level.dtype == np.int32

    def test_quantize_16x16(self):
        """Test 16x16 block quantization."""
        coeff = np.random.randint(-200, 200, size=(16, 16), dtype=np.int32)
        level = quantize(coeff, qp=20, size=16)

        assert level.shape == (16, 16)
        assert level.dtype == np.int32

    def test_quantize_32x32(self):
        """Test 32x32 block quantization."""
        coeff = np.random.randint(-200, 200, size=(32, 32), dtype=np.int32)
        level = quantize(coeff, qp=20, size=32)

        assert level.shape == (32, 32)
        assert level.dtype == np.int32


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_quantize_block_auto_size(self):
        """quantize_block should auto-detect size."""
        coeff = np.random.randint(-100, 100, size=(8, 8), dtype=np.int32)

        level1 = quantize(coeff, qp=20, size=8)
        level2 = quantize_block(coeff, qp=20)

        assert np.array_equal(level1, level2)

    def test_dequantize_block_auto_size(self):
        """dequantize_block should auto-detect size."""
        level = np.random.randint(-10, 10, size=(8, 8), dtype=np.int32)

        coeff1 = dequantize(level, qp=20, size=8)
        coeff2 = dequantize_block(level, qp=20)

        assert np.array_equal(coeff1, coeff2)


class TestUtilities:
    """Tests for utility functions."""

    def test_count_nonzero(self):
        """count_nonzero should count non-zero elements."""
        level = np.array([
            [10, 0, 0, 0],
            [ 0, 5, 0, 0],
            [ 0, 0, 0, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.int32)

        assert count_nonzero(level) == 3

    def test_is_all_zero(self):
        """is_all_zero should detect all-zero blocks."""
        zeros = np.zeros((4, 4), dtype=np.int32)
        nonzeros = np.array([[1, 0], [0, 0]], dtype=np.int32)

        assert is_all_zero(zeros)
        assert not is_all_zero(nonzeros)


class TestIntraVsInter:
    """Tests for intra vs inter quantization difference."""

    def test_intra_vs_inter_dead_zone(self):
        """
        Intra blocks use smaller dead zone (preserve more detail).
        Inter blocks use larger dead zone (more aggressive zeroing).
        """
        # Medium-sized coefficients near dead zone boundary
        coeff = np.full((4, 4), 50, dtype=np.int32)

        level_intra = quantize(coeff, qp=30, size=4, is_intra=True)
        level_inter = quantize(coeff, qp=30, size=4, is_intra=False)

        # Intra should preserve more (smaller dead zone)
        # or equal - but not fewer non-zeros than inter
        assert count_nonzero(level_intra) >= count_nonzero(level_inter)


class TestIntegration:
    """Integration tests with transform pipeline."""

    def test_full_pipeline(self):
        """
        Test complete pipeline:
        residual -> transform -> quantize -> dequantize -> inverse transform
        """
        from nano_hevc.transform import forward_transform_4x4, inverse_transform_4x4
        from nano_hevc.intra import residual_block, reconstruct_block, intra_dc_predict

        # Original block
        original = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        # Reference and prediction
        top = np.array([102, 98, 100, 101], dtype=np.int16)
        left = np.array([103, 102, 101, 99], dtype=np.int16)
        pred = intra_dc_predict(top, left, size=4)

        # Residual
        res = residual_block(original, pred)

        # Transform
        coeff = forward_transform_4x4(res)

        # Quantize/Dequantize
        level = quantize(coeff, qp=20, size=4)
        rec_coeff = dequantize(level, qp=20, size=4)

        # Inverse transform
        rec_res = inverse_transform_4x4(rec_coeff)

        # Reconstruct
        reconstructed = reconstruct_block(pred, rec_res.astype(np.int16))

        # Should be reasonably close to original
        error = np.abs(reconstructed - original)
        assert np.max(error) < 20  # Allow for quantization loss
