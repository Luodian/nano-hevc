"""
Tests for Angular intra prediction (modes 2-34).

Angular prediction projects reference pixels along directional angles.
Key modes:
- Mode 10: Pure horizontal (copies left reference)
- Mode 26: Pure vertical (copies top reference)
- Other modes: Angled projection with interpolation
"""

import numpy as np
import pytest

from nano_hevc.intra import (
    intra_angular_predict,
    INTRA_PRED_ANGLE,
    residual_block,
    reconstruct_block,
)


class TestAngularVertical:
    """Tests for vertical modes (18-34)."""

    def test_mode_26_pure_vertical_4x4(self):
        """
        Mode 26 is pure vertical (angle = 0).
        Each column copies directly from the top reference.
        """
        size = 4
        # top[0] = corner, top[1..size] = reference above block
        top = np.array([99, 100, 110, 120, 130, 0, 0, 0, 0], dtype=np.int16)
        left = np.array([99, 50, 50, 50, 50, 0, 0, 0, 0], dtype=np.int16)
        top_left = 99

        pred = intra_angular_predict(top, left, top_left, mode=26, size=size)

        assert pred.shape == (size, size)
        # Each column should be constant, matching top reference
        assert np.all(pred[:, 0] == 100)
        assert np.all(pred[:, 1] == 110)
        assert np.all(pred[:, 2] == 120)
        assert np.all(pred[:, 3] == 130)

    def test_mode_26_pure_vertical_8x8(self):
        """Mode 26 vertical prediction for 8x8 block."""
        size = 8
        top = np.zeros(2 * size + 1, dtype=np.int16)
        left = np.zeros(2 * size + 1, dtype=np.int16)
        # Set top reference: top[1..8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(size):
            top[i + 1] = (i + 1) * 10
        top_left = 0

        pred = intra_angular_predict(top, left, top_left, mode=26, size=size)

        for x in range(size):
            expected = (x + 1) * 10
            assert np.all(pred[:, x] == expected)

    def test_mode_34_diagonal_up(self):
        """
        Mode 34 has angle = 32 (diagonal up-right).
        Each row shifts by 1 pixel due to 45-degree angle.

        At pixel (x, y), ref index = x + 1 + ((y+1)*32 >> 5) = x + 1 + (y+1)
        """
        size = 4
        # Need extended top reference for diagonal modes
        top = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int16)
        left = np.zeros(2 * size + 1, dtype=np.int16)
        top_left = 0

        pred = intra_angular_predict(top, left, top_left, mode=34, size=size)

        # At (0,0): ref index = 0 + 1 + 1 = 2 -> top[2] = 20
        # At (3,0): ref index = 3 + 1 + 1 = 5 -> top[5] = 50
        # At (0,1): ref index = 0 + 1 + 2 = 3 -> top[3] = 30
        # At (3,3): ref index = 3 + 1 + 4 = 8 -> top[8] = 80
        assert pred[0, 0] == 20
        assert pred[0, 3] == 50
        assert pred[1, 0] == 30
        assert pred[3, 3] == 80


class TestAngularHorizontal:
    """Tests for horizontal modes (2-17)."""

    def test_mode_10_pure_horizontal_4x4(self):
        """
        Mode 10 is pure horizontal (angle = 0).
        Each row copies directly from the left reference.
        """
        size = 4
        top = np.array([99, 50, 50, 50, 50, 0, 0, 0, 0], dtype=np.int16)
        # left[0] = corner, left[1..size] = reference left of block
        left = np.array([99, 100, 110, 120, 130, 0, 0, 0, 0], dtype=np.int16)
        top_left = 99

        pred = intra_angular_predict(top, left, top_left, mode=10, size=size)

        assert pred.shape == (size, size)
        # Each row should be constant, matching left reference
        assert np.all(pred[0, :] == 100)
        assert np.all(pred[1, :] == 110)
        assert np.all(pred[2, :] == 120)
        assert np.all(pred[3, :] == 130)

    def test_mode_2_diagonal_down(self):
        """
        Mode 2 has angle = 32 (diagonal down-left).
        Each column shifts by 1 pixel due to 45-degree angle.

        At pixel (x, y), ref index = y + 1 + ((x+1)*32 >> 5) = y + 1 + (x+1)
        """
        size = 4
        top = np.zeros(2 * size + 1, dtype=np.int16)
        # Need extended left reference for diagonal modes
        left = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int16)
        top_left = 0

        pred = intra_angular_predict(top, left, top_left, mode=2, size=size)

        # At (0,0): ref index = 0 + 1 + 1 = 2 -> left[2] = 20
        # At (0,3): ref index = 3 + 1 + 1 = 5 -> left[5] = 50
        # At (1,0): ref index = 0 + 1 + 2 = 3 -> left[3] = 30
        # At (3,3): ref index = 3 + 1 + 4 = 8 -> left[8] = 80
        assert pred[0, 0] == 20
        assert pred[3, 0] == 50
        assert pred[0, 1] == 30
        assert pred[3, 3] == 80


class TestAngularInterpolation:
    """Tests for modes requiring interpolation."""

    def test_mode_with_interpolation(self):
        """
        Test a mode with non-zero angle that requires interpolation.
        Mode 27 has angle = 2, giving fractional projection.
        """
        size = 4
        # Uniform reference to test interpolation behavior
        top = np.full(2 * size + 1, 100, dtype=np.int16)
        left = np.full(2 * size + 1, 100, dtype=np.int16)
        top_left = 100

        pred = intra_angular_predict(top, left, top_left, mode=27, size=size)

        # With uniform reference, interpolation should still give 100
        assert np.all(pred == 100)

    def test_interpolation_gradient(self):
        """Test interpolation with gradient reference."""
        size = 4
        # Linear gradient in top reference
        top = np.array([0, 0, 32, 64, 96, 128, 160, 192, 224], dtype=np.int16)
        left = np.zeros(2 * size + 1, dtype=np.int16)
        top_left = 0

        pred = intra_angular_predict(top, left, top_left, mode=26, size=size)

        # Pure vertical should copy exactly
        assert pred[0, 0] == 0
        assert pred[0, 1] == 32
        assert pred[0, 2] == 64
        assert pred[0, 3] == 96


class TestAngularAllModes:
    """Tests covering all angular modes."""

    def test_all_modes_produce_valid_output(self):
        """Verify all modes 2-34 produce valid predictions."""
        size = 4
        top = np.full(2 * size + 1, 128, dtype=np.int16)
        left = np.full(2 * size + 1, 128, dtype=np.int16)
        top_left = 128

        for mode in range(2, 35):
            pred = intra_angular_predict(top, left, top_left, mode=mode, size=size)

            assert pred.shape == (size, size)
            assert pred.dtype == np.int16
            # With uniform reference, all predictions should be 128
            assert np.all(pred == 128), f"Mode {mode} failed"

    def test_angle_table_correctness(self):
        """Verify the angle table matches expected values."""
        # Key mode-angle pairs from HEVC spec
        assert INTRA_PRED_ANGLE[10 - 2] == 0   # Mode 10: horizontal
        assert INTRA_PRED_ANGLE[26 - 2] == 0   # Mode 26: vertical
        assert INTRA_PRED_ANGLE[2 - 2] == 32   # Mode 2: 45 degrees
        assert INTRA_PRED_ANGLE[34 - 2] == 32  # Mode 34: 45 degrees
        assert INTRA_PRED_ANGLE[18 - 2] == -32 # Mode 18: diagonal


class TestAngularBlockSizes:
    """Tests for different block sizes."""

    def test_angular_8x8(self):
        """Test angular prediction for 8x8 block."""
        size = 8
        top = np.full(2 * size + 1, 64, dtype=np.int16)
        left = np.full(2 * size + 1, 64, dtype=np.int16)
        top_left = 64

        pred = intra_angular_predict(top, left, top_left, mode=26, size=size)

        assert pred.shape == (8, 8)
        assert np.all(pred == 64)

    def test_angular_16x16(self):
        """Test angular prediction for 16x16 block."""
        size = 16
        top = np.full(2 * size + 1, 200, dtype=np.int16)
        left = np.full(2 * size + 1, 200, dtype=np.int16)
        top_left = 200

        pred = intra_angular_predict(top, left, top_left, mode=10, size=size)

        assert pred.shape == (16, 16)
        assert np.all(pred == 200)


class TestAngularIntegration:
    """Integration tests for angular prediction pipeline."""

    def test_full_angular_pipeline(self):
        """
        Test complete angular prediction pipeline:
        reference pixels -> angular predict -> residual -> reconstruct
        """
        size = 4
        top = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=np.int16)
        left = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=np.int16)
        top_left = 100

        orig = np.array([
            [102, 101, 100, 100],
            [103, 102, 101, 100],
            [103, 102, 100,  99],
            [104, 101,  99,  98],
        ], dtype=np.int16)

        pred = intra_angular_predict(top, left, top_left, mode=26, size=size)
        assert np.all(pred == 100)

        res = residual_block(orig, pred)
        recon = reconstruct_block(pred, res)

        assert np.array_equal(recon, orig)
