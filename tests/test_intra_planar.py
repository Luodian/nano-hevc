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

    @pytest.mark.parametrize(
        "top,right,left_edge,bottom_left,axis",
        [
            (np.zeros(4, dtype=np.int16), 255, np.zeros(4, dtype=np.int16), 0, "horizontal"),
            (np.zeros(4, dtype=np.int16), 0, np.zeros(4, dtype=np.int16), 255, "vertical"),
        ],
    )
    def test_planar_gradients(self, top, right, left_edge, bottom_left, axis):
        """
        Gradient cases ensure interpolation moves in the expected direction.
        """
        size = 4
        pred = intra_planar_predict(top, left_edge, right, bottom_left, size)

        if axis == "horizontal":
            for y in range(size):
                assert np.all(np.diff(pred[y, :]) > 0)
        else:
            for x in range(size):
                assert np.all(np.diff(pred[:, x]) > 0)

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

    @pytest.mark.parametrize("size,value", [(4, 100), (8, 128), (16, 200), (32, 50)])
    def test_planar_uniform_sizes(self, size, value):
        """Uniform reference should remain uniform across sizes."""
        top = np.full(size, value, dtype=np.int16)
        left = np.full(size, value, dtype=np.int16)
        pred = intra_planar_predict(top, left, value, value, size)

        assert pred.shape == (size, size)
        assert np.all(pred == value)


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
