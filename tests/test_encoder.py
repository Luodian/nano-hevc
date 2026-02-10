import numpy as np
import shutil
import subprocess
from unittest.mock import patch

from nano_hevc.cabac import CabacContext, CabacEncoder, init_contexts_for_slice
from nano_hevc.encoder import (
    analyze_video_stream,
    compute_bitrate_kbps,
    decode_video_nano,
    encode_video,
    iter_block_coords,
    parse_ratio_to_float,
    parse_ffprobe_pict_types,
    summarize_frame_types,
    validate_dimensions,
)
from nano_hevc.nal import HEVCConfig, create_parameter_sets, create_slice_nal_unit


def test_create_parameter_sets_preserve_annexb_start_codes():
    data = create_parameter_sets(HEVCConfig(width=64, height=64))
    assert data.count(b"\x00\x00\x00\x01") == 3
    assert b"\x00\x00\x03\x00\x01" not in data


def test_create_parameter_sets_nal_types():
    data = create_parameter_sets(HEVCConfig(width=64, height=64))
    chunks = data.split(b"\x00\x00\x00\x01")
    nals = [chunk for chunk in chunks if chunk]
    assert len(nals) == 3

    nal_types = [((nal[0] << 8) | nal[1]) >> 9 & 0x3F for nal in nals]
    assert nal_types == [32, 33, 34]


def test_cabac_context_arrays_do_not_share_objects():
    contexts = init_contexts_for_slice(slice_type=2, qp=27)

    for key in (
        "last_sig_coeff_x_prefix",
        "last_sig_coeff_y_prefix",
        "sig_coeff_flag",
        "coeff_abs_level_greater1",
        "coeff_abs_level_greater2",
    ):
        ids = {id(ctx) for ctx in contexts[key]}
        assert len(ids) == len(contexts[key]), key


def test_cabac_context_initialization_depends_on_qp():
    low_qp_ctx = CabacContext(111, qp=0)
    high_qp_ctx = CabacContext(111, qp=51)
    assert (low_qp_ctx.state, low_qp_ctx.mps) != (high_qp_ctx.state, high_qp_ctx.mps)


def test_cabac_encoder_emits_nontrivial_stream():
    enc = CabacEncoder()
    ctx = CabacContext(111)
    for i in range(512):
        enc.encode_bin(i & 1, ctx)
    enc.encode_terminate(1)

    out = enc.get_bytes()
    assert len(out) > 16


def test_slice_nal_inserts_emulation_prevention_in_payload():
    cfg = HEVCConfig(width=64, height=64, qp=27)
    # Deliberately contains start-code-like pattern.
    slice_data = b"\x00\x00\x01\x02\x00\x00\x03\x04"
    nal = create_slice_nal_unit(cfg, slice_data=slice_data, is_first_slice=True, slice_type=2, poc=0)

    # Start code should only appear at the front.
    assert nal.startswith(b"\x00\x00\x00\x01")
    assert nal[4:].find(b"\x00\x00\x01") == -1


def test_iter_block_coords_covers_full_plane():
    width, height = 56, 40
    mask = np.zeros((height, width), dtype=np.uint8)

    for x, y, size in iter_block_coords(width, height, max_block_size=32):
        mask[y : y + size, x : x + size] = 1

    assert np.all(mask == 1)


def test_validate_dimensions_enforces_multiple_of_8():
    validate_dimensions(64, 64)

    try:
        validate_dimensions(50, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-multiple-of-8 width")


def test_parse_and_summarize_pict_types():
    text = "I,\nP,\nB,\n?\n"
    types = parse_ffprobe_pict_types(text)
    assert types == ["I", "P", "B"]
    assert summarize_frame_types(types) == {"I": 1, "P": 1, "B": 1}


def test_parse_ratio_to_float():
    assert parse_ratio_to_float("25/1") == 25.0
    assert parse_ratio_to_float("30000/1001") > 29.9
    assert parse_ratio_to_float("0/0") == 0.0
    assert parse_ratio_to_float("12.5") == 12.5


def test_compute_bitrate_kbps():
    assert compute_bitrate_kbps(1000, 1.0) == 8.0
    assert compute_bitrate_kbps(1000, 0.0) == 0.0


def test_encode_video_reports_encoded_frame_types(tmp_path):
    width, height = 16, 16
    frame = np.full((height * width * 3) // 2, 128, dtype=np.uint8)

    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.265"
    input_path.write_bytes(frame.tobytes())

    stats = encode_video(
        input_path=str(input_path),
        output_path=str(output_path),
        width=width,
        height=height,
        num_frames=1,
        qp=27,
        fast_mode=True,
        show_frame_types=True,
    )

    assert stats["frames"] == 1
    assert stats["backend"] == "nano"
    assert stats["encoded_frame_types"] == ["I"]
    assert stats["encoded_frame_type_counts"] == {"I": 1, "P": 0, "B": 0}


def test_encode_video_unknown_backend_raises(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.265"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    try:
        encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="unknown",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown backend")


def test_encode_video_forwards_nano_standard_hevc_flags(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "nano",
        "output_format": "hevc",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_nano", return_value=fake_stats) as mocked:
        stats = encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="nano",
            nano_standard_hevc=True,
            nano_standard_codec="libx265",
            nano_standard_preset="slow",
            nano_standard_crf=24,
            nano_standard_bitrate=None,
            nano_standard_intra_only=True,
        )

    assert stats["output_format"] == "hevc"
    kwargs = mocked.call_args.kwargs
    assert kwargs["standard_hevc_output"] is True
    assert kwargs["standard_codec"] == "libx265"
    assert kwargs["standard_preset"] == "slow"
    assert kwargs["standard_crf"] == 24
    assert kwargs["standard_intra_only"] is True


def test_encode_video_forwards_segment_flags_to_nano(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "nano",
        "output_format": "hevc",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_nano", return_value=fake_stats) as mocked:
        encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="nano",
            start_time=10.0,
            duration=120.0,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["start_time"] == 10.0
    assert kwargs["duration"] == 120.0


def test_encode_video_forwards_verify_flag_to_nano(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "nano",
        "output_format": "hevc",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "decoded_avg_psnr": 0.0,
        "decoded_frames": 0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_nano", return_value=fake_stats) as mocked:
        encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="nano",
            verify_decoded_psnr=True,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["verify_decoded_psnr"] is True


def test_encode_video_forwards_nano_native_hevc_flag(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output_native.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "nano",
        "output_format": "hevc_native",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_nano", return_value=fake_stats) as mocked:
        stats = encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="nano",
            nano_native_hevc=True,
        )

    assert stats["output_format"] == "hevc_native"
    kwargs = mocked.call_args.kwargs
    assert kwargs["native_hevc_output"] is True


def test_encode_video_forwards_segment_flags_to_ffmpeg(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "ffmpeg",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_ffmpeg", return_value=fake_stats) as mocked:
        encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="ffmpeg",
            start_time=5.5,
            duration=30.0,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["start_time"] == 5.5
    assert kwargs["duration"] == 30.0


def test_encode_video_forwards_verify_flag_to_ffmpeg(tmp_path):
    input_path = tmp_path / "input.yuv"
    output_path = tmp_path / "output.hevc"
    input_path.write_bytes(bytes(16 * 16 * 3 // 2))

    fake_stats = {
        "backend": "ffmpeg",
        "frames": 0,
        "total_bytes": 0,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "decoded_avg_psnr": 0.0,
        "decoded_frames": 0,
        "output_bitrate_kbps": 0.0,
        "encoded_frame_types": [],
        "encoded_frame_type_counts": {"I": 0, "P": 0, "B": 0},
        "input_frame_types": [],
        "input_frame_type_counts": {"I": 0, "P": 0, "B": 0},
    }

    with patch("nano_hevc.encoder.encode_video_ffmpeg", return_value=fake_stats) as mocked:
        encode_video(
            input_path=str(input_path),
            output_path=str(output_path),
            width=16,
            height=16,
            backend="ffmpeg",
            verify_decoded_psnr=True,
        )

    kwargs = mocked.call_args.kwargs
    assert kwargs["verify_decoded_psnr"] is True


def test_analyze_video_stream_uses_probe_data():
    with patch("nano_hevc.encoder.probe_video_metadata") as probe_meta, patch(
        "nano_hevc.encoder.probe_input_frame_types"
    ) as probe_types:
        probe_meta.return_value = {
            "fps": 25.0,
            "frame_count": 300,
            "duration": 12.0,
            "size": 1200000,
        }
        probe_types.return_value = ["I", "P", "B", "B"]

        stats = analyze_video_stream(
            "in.mp4",
            start_time=2.0,
            duration=4.0,
            max_frames=3,
        )

    assert stats["bitrate_kbps"] == 800.0
    assert stats["frame_types"] == ["I", "P", "B"]
    assert stats["frame_type_counts"] == {"I": 1, "P": 1, "B": 1}
    assert stats["segment_start"] == 2.0
    assert stats["segment_duration"] == 4.0


def test_nano_container_roundtrip_yuv(tmp_path):
    width, height = 16, 16
    frame_size = width * height * 3 // 2
    frame0 = bytes([32]) * frame_size
    frame1 = bytes([160]) * frame_size

    input_path = tmp_path / "input_2f.yuv"
    output_path = tmp_path / "output_2f.nhevc"
    decoded_path = tmp_path / "decoded_2f.yuv"
    input_path.write_bytes(frame0 + frame1)

    stats = encode_video(
        input_path=str(input_path),
        output_path=str(output_path),
        width=width,
        height=height,
        num_frames=2,
        qp=27,
        fast_mode=True,
        backend="nano",
    )
    assert stats["frames"] == 2

    blob = output_path.read_bytes()
    assert blob.startswith(b"NHEVC1\x00")

    dec_stats = decode_video_nano(str(output_path), str(decoded_path))
    assert dec_stats["frames"] == 2
    assert dec_stats["width"] == width
    assert dec_stats["height"] == height
    assert decoded_path.stat().st_size == frame_size * 2


def test_decode_video_nano_bad_magic(tmp_path):
    bad_path = tmp_path / "bad.nhevc"
    out_path = tmp_path / "out.yuv"
    bad_path.write_bytes(b"NOT_NHEVC")

    try:
        decode_video_nano(str(bad_path), str(out_path))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid nano container")


def test_native_hevc_single_frame_decodable(tmp_path):
    if shutil.which("ffprobe") is None:
        return

    width, height = 16, 16
    frame_size = width * height * 3 // 2
    input_path = tmp_path / "input_1f.yuv"
    output_path = tmp_path / "output_1f_native.hevc"
    input_path.write_bytes(bytes([128]) * frame_size)

    stats = encode_video(
        input_path=str(input_path),
        output_path=str(output_path),
        width=width,
        height=height,
        num_frames=1,
        qp=27,
        fast_mode=True,
        backend="nano",
        nano_native_hevc=True,
    )

    assert stats["output_format"] == "hevc_native"
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,nb_read_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "width=16" in result.stdout
    assert "height=16" in result.stdout
    assert "pix_fmt=yuv420p" in result.stdout
