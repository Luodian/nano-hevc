"""
Tests for DC intra prediction.

These tests verify the basic DC prediction and residual computation
using the example from the specification walkthrough.
"""

import numpy as np
import pytest

from nano_hevc.intra import (
    intra_dc_predict_4x4,
    intra_dc_predict,
    residual_block,
    reconstruct_block,
    clip_to_pixel_range,
)


class TestDCPredict4x4:
    """Tests for 4x4 DC prediction."""

    def test_dc_4x4_example(self):
        """
        Test the specific example from the specification.

        Reference pixels:
            top:  [102, 98, 100, 101]
            left: [103, 102, 101, 99]

        DC calculation:
            top_sum  = 102 + 98 + 100 + 101 = 401
            left_sum = 103 + 102 + 101 + 99 = 405
            DC = floor((401 + 405 + 4) / 8) = floor(810 / 8) = 101
        """
        top = np.array([102, 98, 100, 101], dtype=np.int16)
        left = np.array([103, 102, 101, 99], dtype=np.int16)

        pred = intra_dc_predict_4x4(top, left)

        assert pred.shape == (4, 4)
        assert pred.dtype == np.int16
        assert np.all(pred == 101)

    @pytest.mark.parametrize(
        "top,left,expected",
        [
            (np.full(4, 100, dtype=np.int16), np.full(4, 100, dtype=np.int16), 100),
            (np.array([1, 1, 1, 1], dtype=np.int16), np.array([1, 1, 1, 0], dtype=np.int16), 1),
            (np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.int16), 0),
        ],
    )
    def test_dc_4x4_variants(self, top, left, expected):
        """Grouped sanity checks for uniform, rounding and zero references."""
        pred = intra_dc_predict_4x4(top, left)
        assert np.all(pred == expected)


class TestDCPredictGeneral:
    """Tests for general NxN DC prediction."""

    def test_dc_8x8(self):
        """Test 8x8 DC prediction."""
        top = np.full(8, 100, dtype=np.int16)
        left = np.full(8, 100, dtype=np.int16)

        pred = intra_dc_predict(top, left, size=8)

        assert pred.shape == (8, 8)
        assert np.all(pred == 100)

    def test_dc_16x16(self):
        """Test 16x16 DC prediction."""
        top = np.full(16, 50, dtype=np.int16)
        left = np.full(16, 50, dtype=np.int16)

        pred = intra_dc_predict(top, left, size=16)

        assert pred.shape == (16, 16)
        assert np.all(pred == 50)


class TestResidualBlock:
    """Tests for residual computation."""

    def test_residual_example(self):
        """
        Test the residual computation from the specification example.

        Original block:
            [[102, 101, 100, 100],
             [103, 102, 101, 100],
             [103, 102, 100,  99],
             [104, 101,  99,  98]]

        Prediction (all 101):
            [[101, 101, 101, 101],
             [101, 101, 101, 101],
             [101, 101, 101, 101],
             [101, 101, 101, 101]]

        Expected residual (orig - pred):
            [[ 1,  0, -1, -1],
             [ 2,  1,  0, -1],
             [ 2,  1, -1, -2],
             [ 3,  0, -2, -3]]
        """
        orig = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        pred = np.full((4, 4), 101, dtype=np.int16)

        res = residual_block(orig, pred)

        expected_res = np.array([
            [ 1,  0, -1, -1],
            [ 2,  1,  0, -1],
            [ 2,  1, -1, -2],
            [ 3,  0, -2, -3],
        ], dtype=np.int16)

        assert res.dtype == np.int16
        assert np.array_equal(res, expected_res)

    def test_residual_zeros(self):
        """Test that perfect prediction gives zero residual."""
        block = np.array([
            [100, 100],
            [100, 100],
        ], dtype=np.int16)

        res = residual_block(block, block)

        assert np.all(res == 0)


class TestReconstruct:
    """Tests for block reconstruction."""

    def test_reconstruct_perfect(self):
        """Test that reconstruction recovers original block."""
        orig = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        pred = np.full((4, 4), 101, dtype=np.int16)
        res = residual_block(orig, pred)
        recon = reconstruct_block(pred, res)

        assert np.array_equal(recon, orig)


class TestClipping:
    """Tests for pixel value clipping."""

    def test_clip_8bit(self):
        """Test 8-bit clipping (0-255)."""
        block = np.array([[-10, 0, 128, 255, 300]], dtype=np.int16)
        clipped = clip_to_pixel_range(block, bit_depth=8)

        expected = np.array([[0, 0, 128, 255, 255]], dtype=np.int16)
        assert np.array_equal(clipped, expected)

    def test_clip_10bit(self):
        """Test 10-bit clipping (0-1023)."""
        block = np.array([[-10, 0, 512, 1023, 2000]], dtype=np.int16)
        clipped = clip_to_pixel_range(block, bit_depth=10)

        expected = np.array([[0, 0, 512, 1023, 1023]], dtype=np.int16)
        assert np.array_equal(clipped, expected)


class TestIntegration:
    """Integration tests for the full prediction pipeline."""

    def test_full_dc_pipeline_4x4(self):
        """
        Test complete DC prediction pipeline:
        reference pixels -> DC predict -> residual -> reconstruct
        """
        top = np.array([102, 98, 100, 101], dtype=np.int16)
        left = np.array([103, 102, 101, 99], dtype=np.int16)

        orig = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        pred = intra_dc_predict_4x4(top, left)
        assert np.all(pred == 101)

        res = residual_block(orig, pred)
        expected_res = np.array([
            [ 1,  0, -1, -1],
            [ 2,  1,  0, -1],
            [ 2,  1, -1, -2],
            [ 3,  0, -2, -3],
        ], dtype=np.int16)
        assert np.array_equal(res, expected_res)

        recon = reconstruct_block(pred, res)
        assert np.array_equal(recon, orig)
