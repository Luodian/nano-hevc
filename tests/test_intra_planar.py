"""
Tests for Planar intra prediction.

Planar prediction creates smooth gradients via bilinear interpolation
between edge reference pixels. HEVC spec 8.4.4.2.4.
"""

import numpy as np
import pytest

from nano_hevc.intra import (
    intra_planar_predict,
    residual_block,
    reconstruct_block,
)


class TestPlanarPredict:
    """Tests for Planar prediction."""

    def test_planar_uniform(self):
        """Uniform reference pixels produce uniform prediction."""
        size = 4
        top = np.full(size, 100, dtype=np.int16)
        left = np.full(size, 100, dtype=np.int16)
        top_right = 100
        bottom_left = 100

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        assert pred.shape == (size, size)
        assert pred.dtype == np.int16
        assert np.all(pred == 100)

    def test_planar_horizontal_gradient(self):
        """
        Left column = 0, top_right = 255 creates horizontal gradient.
        Top row = 0, bottom_left = 0 to isolate horizontal component.
        """
        size = 4
        top = np.zeros(size, dtype=np.int16)
        left = np.zeros(size, dtype=np.int16)
        top_right = 255
        bottom_left = 0

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        # Values should increase left to right
        for y in range(size):
            for x in range(size - 1):
                assert pred[y, x] < pred[y, x + 1]

    def test_planar_vertical_gradient(self):
        """
        Top row = 0, bottom_left = 255 creates vertical gradient.
        Left column = 0, top_right = 0 to isolate vertical component.
        """
        size = 4
        top = np.zeros(size, dtype=np.int16)
        left = np.zeros(size, dtype=np.int16)
        top_right = 0
        bottom_left = 255

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        # Values should increase top to bottom
        for y in range(size - 1):
            for x in range(size):
                assert pred[y, x] < pred[y + 1, x]

    def test_planar_4x4_corners(self):
        """
        Test 4x4 prediction with known corner values.

        With top=[0,0,0,0], left=[0,0,0,0], top_right=255, bottom_left=255:
        - Top-left (0,0): minimal gradient influence
        - Bottom-right (3,3): maximal gradient influence
        """
        size = 4
        top = np.zeros(size, dtype=np.int16)
        left = np.zeros(size, dtype=np.int16)
        top_right = 255
        bottom_left = 255

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        # Corner values - manually verify formula
        # pred[0,0] = ((3*0 + 1*255) + (3*0 + 1*255) + 4) >> 3 = 514 >> 3 = 64
        assert pred[0, 0] == 64
        # pred[3,3] = ((0*0 + 4*255) + (0*0 + 4*255) + 4) >> 3 = 2044 >> 3 = 255
        assert pred[3, 3] == 255

    def test_planar_8x8(self):
        """Test 8x8 block with uniform references."""
        size = 8
        top = np.full(size, 128, dtype=np.int16)
        left = np.full(size, 128, dtype=np.int16)
        top_right = 128
        bottom_left = 128

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        assert pred.shape == (8, 8)
        assert np.all(pred == 128)

    def test_planar_16x16(self):
        """Test 16x16 block with uniform references."""
        size = 16
        top = np.full(size, 200, dtype=np.int16)
        left = np.full(size, 200, dtype=np.int16)
        top_right = 200
        bottom_left = 200

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        assert pred.shape == (16, 16)
        assert np.all(pred == 200)

    def test_planar_32x32(self):
        """Test 32x32 block with uniform references."""
        size = 32
        top = np.full(size, 50, dtype=np.int16)
        left = np.full(size, 50, dtype=np.int16)
        top_right = 50
        bottom_left = 50

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)

        assert pred.shape == (32, 32)
        assert np.all(pred == 50)


class TestPlanarIntegration:
    """Integration tests for planar prediction pipeline."""

    def test_full_planar_pipeline(self):
        """
        Test complete planar prediction pipeline:
        reference pixels -> planar predict -> residual -> reconstruct
        """
        size = 4
        top = np.array([100, 100, 100, 100], dtype=np.int16)
        left = np.array([100, 100, 100, 100], dtype=np.int16)
        top_right = 100
        bottom_left = 100

        orig = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        pred = intra_planar_predict(top, left, top_right, bottom_left, size)
        assert np.all(pred == 100)

        res = residual_block(orig, pred)
        recon = reconstruct_block(pred, res)

        assert np.array_equal(recon, orig)
