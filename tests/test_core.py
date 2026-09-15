"""Prediction, integer rounding, and lossy reconstruction: uv run pytest tests/test_core.py."""

import numpy as np
import pytest

from nano_hevc.intra import (
    intra_angular_predict,
    intra_dc_predict,
    intra_dc_predict_4x4,
    intra_planar_predict,
    reconstruct_block,
    residual_block,
)
from nano_hevc.quant import dequantize, get_qp_params, quantize
from nano_hevc.transform import forward_transform, inverse_transform


@pytest.mark.parametrize("size,value", [(4, 100), (8, 128), (16, 200), (32, 50)])
def test_uniform_prediction(size, value):
    refs = np.full(2 * size + 1, value, dtype=np.int16)
    for pred in (
        intra_dc_predict(refs[:size], refs[:size], size),
        intra_planar_predict(refs[:size], refs[:size], value, value, size),
        *(intra_angular_predict(refs, refs, value, mode, size) for mode in range(2, 35)),
    ):
        assert pred.shape == (size, size)
        assert pred.dtype == np.int16
        assert np.all(pred == value)


@pytest.mark.parametrize("top,left,expected", [
    ([102, 98, 100, 101], [103, 102, 101, 99], 101),
    ([1, 1, 1, 1], [1, 1, 1, 0], 1),
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
])
def test_dc_rounding(top, left, expected):
    top, left = np.array(top, dtype=np.int16), np.array(left, dtype=np.int16)
    assert np.all(intra_dc_predict_4x4(top, left) == expected)
    assert np.all(intra_dc_predict(top, left, 4) == expected)


@pytest.mark.parametrize("right,bottom,expected", [
    (255, 0, [[32, 64, 96, 128]] * 4),
    (0, 255, [[32] * 4, [64] * 4, [96] * 4, [128] * 4]),
    (255, 255, [[64, 96, 128, 159], [96, 128, 159, 191],
                [128, 159, 191, 223], [159, 191, 223, 255]]),
])
def test_planar_gradients(right, bottom, expected):
    refs = np.zeros(4, dtype=np.int16)
    np.testing.assert_array_equal(intra_planar_predict(refs, refs, right, bottom, 4), expected)


@pytest.mark.parametrize("mode,size", [(26, 4), (26, 8), (10, 4)])
def test_angular_axis_projection(mode, size):
    refs = np.arange(2 * size + 1, dtype=np.int16) * 10
    expected = np.tile(refs[1:size + 1], (size, 1))
    if mode == 10:
        expected = expected.T
    np.testing.assert_array_equal(intra_angular_predict(refs, refs, 0, mode, size), expected)


@pytest.mark.parametrize("mode", [2, 34])
def test_angular_diagonal_projection(mode):
    refs = np.arange(9, dtype=np.int16) * 10
    expected = [[20, 30, 40, 50], [30, 40, 50, 60],
                [40, 50, 60, 70], [50, 60, 70, 80]]
    np.testing.assert_array_equal(intra_angular_predict(refs, refs, 0, mode, 4), expected)


def test_angular_negative_reference_extension():
    top = np.arange(9, dtype=np.int16) * 10
    left = np.array([0] + [5] * 8, dtype=np.int16)
    expected = [[0, 10, 20, 30], [5, 0, 10, 20], [5, 5, 0, 10], [5, 5, 5, 0]]
    np.testing.assert_array_equal(intra_angular_predict(top, left, 0, 18, 4), expected)


@pytest.mark.parametrize("size,use_dst,limit,mean_limit", [
    (4, False, 2, 2), (4, True, 2, 2), (8, False, 50, 25),
    (16, False, 60, 30), (32, False, 40, 20),
])
def test_transform_roundtrip(size, use_dst, limit, mean_limit):
    if size == 4:
        original = np.array([[5, 3, -2, 1], [2, 4, 1, -3],
                             [-1, 2, 3, 2], [0, -1, 2, 4]], dtype=np.int16)
    else:
        bound = 30 if size == 32 else 50
        original = np.random.default_rng(42).integers(-bound, bound, (size, size), dtype=np.int16)
    coeff = forward_transform(original, use_dst=use_dst)
    recon = inverse_transform(coeff, use_dst=use_dst)
    assert coeff.dtype == recon.dtype == np.int32
    error = np.abs(recon - original)
    assert np.max(error) <= limit
    assert np.mean(error) < mean_limit
    zeros = np.zeros_like(original)
    assert not np.any(forward_transform(zeros, use_dst=use_dst))
    assert not np.any(inverse_transform(zeros, use_dst=use_dst))


def test_transform_two_pass_rounding():
    residual = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                         [9, 0, -1, -2], [4, 3, 2, 1]], dtype=np.int16)
    expected = [[14, 3, 2, 1], [3, -6, -1, -1], [-2, -3, -2, -1], [-6, 5, 3, 2]]
    np.testing.assert_array_equal(forward_transform(residual), expected)
    assert not np.array_equal(forward_transform(residual, use_dst=True), expected)


@pytest.mark.parametrize("size", [4, 8])
def test_dct_energy_compaction(size):
    gradient = np.indices((size, size)).sum(axis=0).astype(np.int16) + 10
    coeff = forward_transform(gradient)
    assert np.sum(coeff[:size // 2, :size // 2] ** 2) > np.sum(coeff ** 2) * 0.9
    uniform = forward_transform(np.full((size, size), 16, dtype=np.int16))
    assert uniform[0, 0] != 0
    uniform[0, 0] = 0
    assert not np.any(uniform)


@pytest.mark.parametrize("qp,expected", [
    (-5, (0, 0)), (0, (0, 0)), (5, (0, 5)), (6, (1, 0)),
    (11, (1, 5)), (51, (8, 3)), (100, (8, 3)),
])
def test_qp_params_and_clamp(qp, expected):
    assert get_qp_params(qp) == expected


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_quantization_sign_and_qp(size):
    coeff = np.random.default_rng(42).integers(-200, 200, (size, size), dtype=np.int32)
    coeff[0, 0] = 500
    low = quantize(coeff, qp=10, size=size)
    high = quantize(coeff, qp=40, size=size)
    assert low.shape == coeff.shape and low.dtype == np.int32
    assert np.count_nonzero(high) <= np.count_nonzero(low)
    nonzero = low != 0
    assert np.any(nonzero)
    np.testing.assert_array_equal(np.sign(low[nonzero]), np.sign(coeff[nonzero]))
    assert not np.any(quantize(np.zeros_like(coeff), qp=20, size=size))
    assert not np.any(dequantize(np.zeros_like(coeff), qp=20, size=size))


def test_quantization_deadzone_and_step():
    small = np.full((4, 4), 5, dtype=np.int32)
    assert not np.any(quantize(small, qp=40, size=4))
    boundary = np.full((4, 4), 50, dtype=np.int32)
    intra = quantize(boundary, qp=30, size=4, is_intra=True)
    inter = quantize(boundary, qp=30, size=4, is_intra=False)
    assert np.count_nonzero(intra) >= np.count_nonzero(inter)
    coeff = np.full((4, 4), 256, dtype=np.int32)
    low = quantize(coeff, qp=10, size=4)
    high = quantize(coeff, qp=16, size=4)
    assert abs(high[0, 0]) * 2 <= abs(low[0, 0]) + 1


def test_quantization_reconstruction_accuracy():
    original = np.array([[200, 100, 50, 25], [100, 80, 40, 20],
                         [50, 40, 30, 15], [25, 20, 15, 10]], dtype=np.int32)
    reconstructed = dequantize(quantize(original, qp=5, size=4), qp=5, size=4)
    assert np.mean(np.abs(reconstructed - original)) < 50
    assert reconstructed[0, 0] > 0


def test_prediction_transform_quantization_pipeline():
    original = np.array([[102, 101, 100, 100], [103, 102, 101, 100],
                         [103, 102, 100, 99], [104, 101, 99, 98]], dtype=np.int16)
    top = np.array([102, 98, 100, 101], dtype=np.int16)
    left = np.array([103, 102, 101, 99], dtype=np.int16)
    pred = intra_dc_predict(top, left, 4)
    residual = residual_block(original, pred)
    np.testing.assert_array_equal(residual, [[1, 0, -1, -1], [2, 1, 0, -1],
                                             [2, 1, -1, -2], [3, 0, -2, -3]])
    np.testing.assert_array_equal(reconstruct_block(pred, residual), original)
    coeff = forward_transform(residual)
    reconstructed = reconstruct_block(pred, inverse_transform(coeff).astype(np.int16))
    assert np.max(np.abs(reconstructed - original)) <= 2
    levels = quantize(coeff, qp=20, size=4)
    decoded = inverse_transform(dequantize(levels, qp=20, size=4))
    reconstructed = reconstruct_block(pred, decoded.astype(np.int16))
    assert np.max(np.abs(reconstructed - original)) < 20
