"""Bitstream and end-to-end codec checks: uv run pytest tests/test_encoder.py."""

import json
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from nano_hevc.cabac import CabacContext, CabacEncoder, init_contexts_for_slice
from nano_hevc.encoder import (
    analyze_video_stream,
    compute_bitrate_kbps,
    decode_video_nano,
    encode_frame,
    encode_video,
    iter_block_coords,
    parse_ffprobe_pict_types,
    parse_ratio_to_float,
    summarize_frame_types,
    validate_dimensions,
)
from nano_hevc.frame import Frame, Plane
from nano_hevc.nal import HEVCConfig, create_parameter_sets, create_slice_nal_unit


def test_annexb_headers_and_payload_escaping():
    config = HEVCConfig(width=64, height=64, qp=27)
    data = create_parameter_sets(config)
    assert data.count(b"\x00\x00\x00\x01") == 3
    assert b"\x00\x00\x03\x00\x01" not in data
    nals = [chunk for chunk in data.split(b"\x00\x00\x00\x01") if chunk]
    assert [((nal[0] << 8) | nal[1]) >> 9 & 0x3F for nal in nals] == [32, 33, 34]
    nal = create_slice_nal_unit(
        config, slice_data=b"\x00\x00\x01\x02\x00\x00\x03\x04",
        is_first_slice=True, slice_type=2, poc=0,
    )
    assert nal.startswith(b"\x00\x00\x00\x01")
    assert b"\x00\x00\x01" not in nal[4:]


def test_cabac_state_independence_and_output():
    contexts = init_contexts_for_slice(slice_type=2, qp=27)
    for key in ("last_sig_coeff_x_prefix", "last_sig_coeff_y_prefix", "sig_coeff_flag",
                "coeff_abs_level_greater1", "coeff_abs_level_greater2"):
        assert len({id(ctx) for ctx in contexts[key]}) == len(contexts[key]), key
    low, high = CabacContext(111, qp=0), CabacContext(111, qp=51)
    assert (low.state, low.mps) != (high.state, high.mps)
    encoder, context = CabacEncoder(), CabacContext(111)
    for i in range(512):
        encoder.encode_bin(i & 1, context)
    encoder.encode_terminate(1)
    assert len(encoder.get_bytes()) > 16


def test_block_tiling_and_dimensions():
    mask = np.zeros((40, 56), dtype=np.uint8)
    for x, y, size in iter_block_coords(56, 40, max_block_size=32):
        mask[y:y + size, x:x + size] += 1
    assert np.all(mask == 1)
    validate_dimensions(64, 64)
    with pytest.raises(ValueError):
        validate_dimensions(50, 64)


def test_stream_statistics():
    types = parse_ffprobe_pict_types("I,\nP,\nB,\n?\n")
    assert types == ["I", "P", "B"]
    assert summarize_frame_types(types) == {"I": 1, "P": 1, "B": 1}
    for value, expected in [("25/1", 25), ("30000/1001", 30000 / 1001),
                            ("0/0", 0), ("12.5", 12.5)]:
        assert parse_ratio_to_float(value) == pytest.approx(expected)
    assert compute_bitrate_kbps(1000, 1) == 8
    assert compute_bitrate_kbps(1000, 0) == 0
    with patch("nano_hevc.encoder.probe_video_metadata", return_value={
        "fps": 25.0, "frame_count": 300, "duration": 12.0, "size": 1200000,
    }), patch("nano_hevc.encoder.probe_input_frame_types", return_value=["I", "P", "B", "B"]):
        stats = analyze_video_stream("in.mp4", start_time=2, duration=4, max_frames=3)
    assert stats["bitrate_kbps"] == 800
    assert stats["frame_types"] == types
    assert stats["frame_type_counts"] == {"I": 1, "P": 1, "B": 1}
    assert (stats["segment_start"], stats["segment_duration"]) == (2, 4)


def test_native_intra_residual_and_symbol_trace():
    y = (np.arange(256, dtype=np.int16).reshape(16, 16) + 16) % 256
    chroma = np.full((8, 8), 128, dtype=np.int16)
    frame = Frame(y=Plane(y), u=Plane(chroma.copy()), v=Plane(chroma.copy()))
    encoded, _, stats = encode_frame(
        frame, HEVCConfig(width=16, height=16, qp=12), fast_mode=False,
        native_minimal_syntax=True, trace=True,
    )
    assert stats["frame_type"] == "I"
    assert stats["cbf_count"] > 0
    assert encoded.startswith(b"\x00\x00\x00\x01")
    assert {"split_cu_flag", "part_mode", "intra_luma_pred_mode",
            "intra_chroma_pred_mode", "cbf_luma"} <= {
                token.token_type for token in stats["symbol_trace"]
            }


def test_native_inter_schedule(tmp_path):
    source, output = tmp_path / "input.yuv", tmp_path / "output.hevc"
    source.write_bytes(b"".join(bytes([value]) * 384 for value in [16, 48, 80, 112]))
    stats = encode_video(
        str(source), str(output), width=16, height=16, num_frames=4, qp=27,
        fast_mode=True, backend="nano", nano_native_hevc=True, nano_native_inter=True,
        nano_native_gop_size=4, nano_native_search_range=2,
    )
    assert stats["encoded_frame_types"] == ["I", "P", "B", "P"]
    assert stats["encoded_frame_type_counts"] == {"I": 1, "P": 2, "B": 1}


def test_nano_container_roundtrip_matches_reconstruction(tmp_path):
    source, output, decoded = (tmp_path / name for name in ["input.yuv", "output.nhevc", "decoded.yuv"])
    frames = [bytes([value]) * 384 for value in [32, 160]]
    source.write_bytes(b"".join(frames))
    config = HEVCConfig(width=16, height=16, qp=27)
    expected = b"".join(
        encode_frame(Frame.from_yuv420p(raw, 16, 16), config, fast_mode=True)[1].to_yuv420p()
        for raw in frames
    )
    stats = encode_video(
        str(source), str(output), width=16, height=16, num_frames=2, qp=27,
        fast_mode=True, backend="nano", show_frame_types=True,
    )
    assert stats["frames"] == 2 and stats["backend"] == "nano"
    assert stats["encoded_frame_types"] == ["I", "I"]
    assert stats["encoded_frame_type_counts"] == {"I": 2, "P": 0, "B": 0}
    assert output.read_bytes().startswith(b"NHEVC1\x00")
    decoded_stats = decode_video_nano(str(output), str(decoded))
    assert (decoded_stats["frames"], decoded_stats["width"], decoded_stats["height"]) == (2, 16, 16)
    assert decoded.read_bytes() == expected


def test_invalid_container_and_backend(tmp_path):
    source, output = tmp_path / "bad.nhevc", tmp_path / "output.yuv"
    source.write_bytes(b"NOT_NHEVC")
    with pytest.raises(ValueError):
        decode_video_nano(str(source), str(output))
    source.write_bytes(bytes(384))
    with pytest.raises(ValueError):
        encode_video(str(source), str(output), width=16, height=16, backend="unknown")


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
def test_native_hevc_single_frame_decodable(tmp_path):
    source, output = tmp_path / "input.yuv", tmp_path / "output.hevc"
    source.write_bytes(bytes([128]) * 384)
    stats = encode_video(
        str(source), str(output), width=16, height=16, num_frames=1, qp=27,
        fast_mode=True, backend="nano", nano_native_hevc=True,
    )
    assert stats["output_format"] == "hevc_native"
    result = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries", "stream=width,height,pix_fmt,nb_read_frames",
         "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["pix_fmt"]) == (16, 16, "yuv420p")
    assert int(stream["nb_read_frames"]) == 1
