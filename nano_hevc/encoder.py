"""
HEVC encoder main module.

Provides a complete encoding pipeline for intra-only HEVC encoding.
"""

from __future__ import annotations
from typing import Tuple, List
from dataclasses import dataclass
import sys
import argparse
import subprocess
import json
import os
import struct
import zlib
from fractions import Fraction
import numpy as np

from nano_hevc.frame import Frame, Plane
from nano_hevc.block import BlockView
from nano_hevc.intra import (
    intra_dc_predict,
    intra_planar_predict,
    intra_angular_predict,
    residual_block,
    reconstruct_block,
    clip_to_pixel_range,
)
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block, is_all_zero
from nano_hevc.scan import (
    scan_block,
    get_scan_type_for_intra_mode,
    find_last_significant_coeff,
)
from nano_hevc.cabac import (
    CabacEncoder,
    init_contexts_for_slice,
    encode_coeff_abs_level_remaining,
)
from nano_hevc.nal import (
    HEVCConfig,
    create_parameter_sets,
    create_slice_nal_unit,
)
from nano_hevc.metrics import psnr


INTRA_MODE_DC = 1
INTRA_MODE_PLANAR = 0


@dataclass
class EncodedBlock:
    """Result of encoding a single block."""

    mode: int
    coeffs: np.ndarray
    reconstructed: np.ndarray
    cbf: bool


@dataclass
class CTUData:
    """Data for a single Coding Tree Unit."""

    x: int
    y: int
    size: int
    blocks: List[EncodedBlock]


def iter_block_coords(width: int, height: int, max_block_size: int):
    """Iterate a full non-overlapping square tiling over the plane."""
    tile_size = 0
    for size in (32, 16, 8, 4):
        if size <= max_block_size and width % size == 0 and height % size == 0:
            tile_size = size
            break
    if tile_size == 0:
        return

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            yield x, y, tile_size


def validate_dimensions(width: int, height: int) -> None:
    """
    Validate frame size against current transform/block constraints.

    The current encoder requires luma and chroma planes to be tiled by
    transform blocks >= 4x4, so YUV420 dimensions must be multiples of 8.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(
            f"width/height must be multiples of 8 for current encoder, got {width}x{height}"
        )


def parse_ffprobe_pict_types(stdout: str) -> List[str]:
    """Parse ffprobe csv frame pict_type output."""
    frame_types: List[str] = []
    for raw_line in stdout.splitlines():
        token = raw_line.strip().rstrip(",").upper()
        if token in {"I", "P", "B"}:
            frame_types.append(token)
    return frame_types


def summarize_frame_types(frame_types: List[str]) -> dict:
    """Count I/P/B frame occurrences."""
    summary = {"I": 0, "P": 0, "B": 0}
    for frame_type in frame_types:
        if frame_type in summary:
            summary[frame_type] += 1
    return summary


def probe_input_frame_types(input_path: str) -> List[str]:
    """Use ffprobe to extract per-frame pict_type labels from a video file."""
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pict_type",
        "-of",
        "csv=p=0",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_ffprobe_pict_types(result.stdout)


def parse_ratio_to_float(value: str) -> float:
    """Parse ffprobe ratio text (e.g. 25/1) to float."""
    token = value.strip()
    if "/" in token:
        num, den = token.split("/", 1)
        denom = float(den)
        if denom == 0:
            return 0.0
        return float(num) / denom
    return float(token)


def probe_video_metadata(input_path: str) -> dict:
    """Read fps/duration/frame-count from ffprobe."""
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_read_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)

    stream = payload.get("streams", [{}])[0]
    fmt = payload.get("format", {})

    fps = parse_ratio_to_float(stream.get("avg_frame_rate", "0/1"))
    frame_count = int(stream.get("nb_read_frames", 0) or 0)
    duration = float(fmt.get("duration", 0.0) or 0.0)
    size = int(float(fmt.get("size", 0.0) or 0.0))
    return {"fps": fps, "frame_count": frame_count, "duration": duration, "size": size}


def compute_bitrate_kbps(total_bytes: int, duration_seconds: float) -> float:
    """Compute bitrate in kbps from bytes and duration."""
    if duration_seconds <= 0:
        return 0.0
    return (total_bytes * 8.0) / duration_seconds / 1000.0


NANO_CONTAINER_MAGIC = b"NHEVC1\x00\x00"
NANO_CONTAINER_HEADER = struct.Struct("<8sIIIIII")
NANO_FRAME_HEADER = struct.Struct("<BIII")
FRAME_TYPE_TO_ID = {"I": 0, "P": 1, "B": 2}
FRAME_ID_TO_TYPE = {0: "I", 1: "P", 2: "B"}


def fps_to_rational(fps: float) -> tuple[int, int]:
    """Convert float fps to a stable rational pair."""
    if fps <= 0:
        return 30, 1
    frac = Fraction(fps).limit_denominator(1001)
    return int(frac.numerator), int(frac.denominator)


def pack_nano_header(
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    frame_count: int,
) -> bytes:
    """Pack NHEVC1 container header."""
    return NANO_CONTAINER_HEADER.pack(
        NANO_CONTAINER_MAGIC,
        width,
        height,
        fps_num,
        fps_den,
        frame_count,
        0,
    )


def unpack_nano_header(blob: bytes) -> dict:
    """Unpack NHEVC1 container header."""
    if len(blob) != NANO_CONTAINER_HEADER.size:
        raise ValueError("Invalid nano container header size")
    magic, width, height, fps_num, fps_den, frame_count, _ = NANO_CONTAINER_HEADER.unpack(
        blob
    )
    if magic != NANO_CONTAINER_MAGIC:
        raise ValueError("Invalid nano container magic")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid dimensions in nano container")
    if fps_num <= 0 or fps_den <= 0:
        raise ValueError("Invalid fps in nano container")
    return {
        "width": width,
        "height": height,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frame_count": frame_count,
    }


def get_reference_pixels(
    recon_plane: Plane, x: int, y: int, size: int
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Get reference pixels for intra prediction from reconstructed plane.

    Returns: (top, left, top_right, bottom_left, top_left)
    """
    top = np.full(2 * size + 1, 128, dtype=np.int16)
    left = np.full(2 * size + 1, 128, dtype=np.int16)
    top_left = 128

    if y > 0:
        end_x = min(x + 2 * size, recon_plane.width)
        top[1 : 1 + end_x - x] = recon_plane.data[y - 1, x:end_x]
        if end_x - x < 2 * size:
            top[1 + end_x - x :] = top[end_x - x]

    if x > 0:
        end_y = min(y + 2 * size, recon_plane.height)
        left[1 : 1 + end_y - y] = recon_plane.data[y:end_y, x - 1]
        if end_y - y < 2 * size:
            left[1 + end_y - y :] = left[end_y - y]

    if x > 0 and y > 0:
        top_left = int(recon_plane.data[y - 1, x - 1])
        top[0] = top_left
        left[0] = top_left

    top_right = int(top[size]) if size < len(top) else int(top[-1])
    bottom_left = int(left[size]) if size < len(left) else int(left[-1])

    return top, left, top_right, bottom_left, top_left


def predict_block(
    top: np.ndarray,
    left: np.ndarray,
    top_right: int,
    bottom_left: int,
    top_left: int,
    mode: int,
    size: int,
) -> np.ndarray:
    """Generate prediction for a block given intra mode."""
    if mode == INTRA_MODE_PLANAR:
        return intra_planar_predict(
            top[1 : size + 1], left[1 : size + 1], top_right, bottom_left, size
        )
    elif mode == INTRA_MODE_DC:
        return intra_dc_predict(top[1 : size + 1], left[1 : size + 1], size)
    else:
        return intra_angular_predict(top, left, top_left, mode, size)


def evaluate_intra_mode(
    original: np.ndarray,
    top: np.ndarray,
    left: np.ndarray,
    top_right: int,
    bottom_left: int,
    top_left: int,
    mode: int,
    size: int,
    qp: int,
    use_dst: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate an intra mode and return cost metrics."""
    pred = predict_block(top, left, top_right, bottom_left, top_left, mode, size)
    res = residual_block(original, pred)

    coeff = forward_transform(res, use_dst=use_dst)
    quant_coeff = quantize_block(coeff, qp)

    dequant_coeff = dequantize_block(quant_coeff, qp)
    recon_res = inverse_transform(dequant_coeff, use_dst=use_dst)
    recon = clip_to_pixel_range(reconstruct_block(pred, recon_res.astype(np.int16)))

    ssd = int(np.sum((original.astype(np.int32) - recon.astype(np.int32)) ** 2))
    num_nonzero = int(np.count_nonzero(quant_coeff))
    cost = ssd + (num_nonzero * qp * 10)

    return pred, quant_coeff, recon, cost


def encode_block(
    plane: Plane,
    recon_plane: Plane,
    x: int,
    y: int,
    size: int,
    qp: int,
    fast_mode: bool = True,
) -> EncodedBlock:
    """Encode a single block with intra prediction."""
    blk = BlockView(plane=recon_plane, x=x, y=y, size=size)
    original = plane.data[y : y + size, x : x + size].copy()

    top, left, top_right, bottom_left, top_left = get_reference_pixels(
        recon_plane, x, y, size
    )

    use_dst = size == 4

    if fast_mode:
        modes_to_test = [INTRA_MODE_DC, INTRA_MODE_PLANAR, 10, 26]
    else:
        modes_to_test = list(range(35))

    best_mode = INTRA_MODE_DC
    best_cost = float("inf")
    best_pred = None
    best_coeffs = None
    best_recon = None

    for mode in modes_to_test:
        pred, coeffs, recon, cost = evaluate_intra_mode(
            original,
            top,
            left,
            top_right,
            bottom_left,
            top_left,
            mode,
            size,
            qp,
            use_dst,
        )
        if cost < best_cost:
            best_cost = cost
            best_mode = mode
            best_pred = pred
            best_coeffs = coeffs
            best_recon = recon

    blk.write_pixels(best_recon)

    cbf = not is_all_zero(best_coeffs)

    return EncodedBlock(
        mode=best_mode, coeffs=best_coeffs, reconstructed=best_recon, cbf=cbf
    )


def encode_coefficients_cabac(
    encoder: CabacEncoder,
    contexts: dict,
    coeffs: np.ndarray,
    mode: int,
    is_luma: bool = True,
) -> None:
    """Encode transform coefficients using CABAC."""
    size = coeffs.shape[0]
    log2_size = int(np.log2(size))

    scan_type = get_scan_type_for_intra_mode(mode, log2_size) if is_luma else 0

    last_pos = find_last_significant_coeff(coeffs, scan_type)
    if last_pos < 0:
        return

    last_x = last_pos % size
    last_y = last_pos // size

    for i in range(min(last_x, 15)):
        ctx_idx = min(i, 2) + (0 if is_luma else 3)
        ctx = contexts["last_sig_coeff_x_prefix"][ctx_idx]
        encoder.encode_bin(1, ctx)
    if last_x < 15:
        ctx_idx = min(last_x, 2) + (0 if is_luma else 3)
        ctx = contexts["last_sig_coeff_x_prefix"][ctx_idx]
        encoder.encode_bin(0, ctx)

    for i in range(min(last_y, 15)):
        ctx_idx = min(i, 2) + (0 if is_luma else 3)
        ctx = contexts["last_sig_coeff_y_prefix"][ctx_idx]
        encoder.encode_bin(1, ctx)
    if last_y < 15:
        ctx_idx = min(last_y, 2) + (0 if is_luma else 3)
        ctx = contexts["last_sig_coeff_y_prefix"][ctx_idx]
        encoder.encode_bin(0, ctx)

    scanned = scan_block(coeffs, scan_type)

    sig_flags = []
    gt1_flags = []
    gt2_flags = []
    remaining = []
    signs = []

    for i in range(last_pos, -1, -1):
        level = int(scanned[i])
        if level != 0:
            sig_flags.append((i, 1))
            signs.append(1 if level < 0 else 0)
            abs_level = abs(level)

            if abs_level > 1:
                gt1_flags.append((i, 1))
                if abs_level > 2:
                    gt2_flags.append((i, 1))
                    remaining.append(abs_level - 3)
                else:
                    gt2_flags.append((i, 0))
            else:
                gt1_flags.append((i, 0))
        elif i < last_pos:
            sig_flags.append((i, 0))

    for pos, flag in sig_flags:
        ctx_idx = min(pos, 41)
        ctx = contexts["sig_coeff_flag"][ctx_idx]
        encoder.encode_bin(flag, ctx)

    for idx, (pos, flag) in enumerate(gt1_flags):
        ctx_idx = min(idx, 23)
        ctx = contexts["coeff_abs_level_greater1"][ctx_idx]
        encoder.encode_bin(flag, ctx)

    for idx, (pos, flag) in enumerate(gt2_flags):
        ctx_idx = min(idx, 5)
        ctx = contexts["coeff_abs_level_greater2"][ctx_idx]
        encoder.encode_bin(flag, ctx)

    for sign in signs:
        encoder.encode_bypass(sign)

    rice_param = 0
    for level in remaining:
        encode_coeff_abs_level_remaining(encoder, level, rice_param)
        if level > (3 << rice_param):
            rice_param = min(rice_param + 1, 4)


def encode_frame(
    frame: Frame,
    config: HEVCConfig,
    fast_mode: bool = True,
    native_minimal_syntax: bool = False,
) -> Tuple[bytes, Frame, dict]:
    """
    Encode a single frame.

    Returns: (encoded_bytes, reconstructed_frame, stats)
    """
    validate_dimensions(frame.width, frame.height)

    recon = Frame.zeros(frame.height, frame.width, dtype=np.int16)

    cabac = CabacEncoder()
    contexts = init_contexts_for_slice(2, config.qp)

    stats = {
        "blocks": 0,
        "modes": {},
        "cbf_count": 0,
        "total_coeffs": 0,
        "frame_type": "I",
    }

    block_size = min(config.max_cu_size, 32)

    for plane_name, src_plane, dst_plane in [
        ("Y", frame.y, recon.y),
        ("U", frame.u, recon.u),
        ("V", frame.v, recon.v),
    ]:
        bs = block_size if plane_name == "Y" else max(4, block_size // 2)
        is_luma = plane_name == "Y"

        for x, y, actual_size in iter_block_coords(src_plane.width, src_plane.height, bs):
            encoded = encode_block(
                src_plane, dst_plane, x, y, actual_size, config.qp, fast_mode
            )

            if is_luma:
                stats["blocks"] += 1
                mode_key = f"mode_{encoded.mode}"
                stats["modes"][mode_key] = stats["modes"].get(mode_key, 0) + 1
                if encoded.cbf:
                    stats["cbf_count"] += 1
                    stats["total_coeffs"] += np.count_nonzero(encoded.coeffs)

            if native_minimal_syntax and is_luma:
                # Experimental native HEVC mode: write a minimal intra syntax
                # shell per luma block, while forcing zero residual coding.
                cabac.encode_bin(0, contexts["split_cu_flag"][0])
                cabac.encode_bin(0, contexts["part_mode"][0])
                cabac.encode_bin(0, contexts["intra_luma_pred_mode"][0])
                cabac.encode_bin(0, contexts["intra_chroma_pred_mode"][0])

            if native_minimal_syntax:
                ctx = contexts["cbf_luma" if is_luma else "cbf_chroma"][0]
                cabac.encode_bin(0, ctx)
            else:
                if encoded.cbf:
                    ctx = contexts["cbf_luma" if is_luma else "cbf_chroma"][0]
                    cabac.encode_bin(1, ctx)
                    encode_coefficients_cabac(
                        cabac, contexts, encoded.coeffs, encoded.mode, is_luma
                    )
                else:
                    ctx = contexts["cbf_luma" if is_luma else "cbf_chroma"][0]
                    cabac.encode_bin(0, ctx)

    cabac.encode_terminate(1)

    slice_data = cabac.get_bytes()

    encoded_data = create_slice_nal_unit(
        config=config,
        slice_data=slice_data,
        is_first_slice=True,
        slice_type=2,  # I-slice
        poc=0,
    )

    return encoded_data, recon, stats


def encode_video_nano(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int = 1,
    qp: int = 27,
    fast_mode: bool = True,
    show_frame_types: bool = False,
    standard_hevc_output: bool = False,
    native_hevc_output: bool = False,
    standard_codec: str = "libx265",
    standard_preset: str = "medium",
    standard_crf: int = 28,
    standard_bitrate: str | None = None,
    standard_intra_only: bool = False,
) -> dict:
    """
    Encode video with nano backend.

    Modes:
    - default: write NHEVC1 container (.nhevc) with zlib-compressed reconstructed
      YUV420p frames.
    - native_hevc_output=True: write native HEVC bitstream generated by
      nano-hevc (experimental; currently minimal intra syntax only).
    - standard_hevc_output=True: write standard HEVC bitstream by encoding the
      reconstructed frames through ffmpeg.
    """
    validate_dimensions(width, height)
    if standard_hevc_output and native_hevc_output:
        raise ValueError("standard_hevc_output and native_hevc_output are mutually exclusive")

    config = HEVCConfig(width=width, height=height, qp=qp)

    total_stats = {
        "backend": "nano",
        "output_format": (
            "hevc"
            if standard_hevc_output
            else "hevc_native" if native_hevc_output else "nhevc"
        ),
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

    psnr_values = []
    is_container_input = input_path.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))
    input_fps_for_rate = 30.0
    if is_container_input:
        try:
            input_meta = probe_video_metadata(input_path)
            if input_meta["fps"] > 0:
                input_fps_for_rate = input_meta["fps"]
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass

    if show_frame_types and is_container_input:
        try:
            input_frame_types = probe_input_frame_types(input_path)[:num_frames]
            total_stats["input_frame_types"] = input_frame_types
            total_stats["input_frame_type_counts"] = summarize_frame_types(input_frame_types)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Warning: failed to probe input frame types: {exc}", file=sys.stderr)

    fps_num, fps_den = fps_to_rational(input_fps_for_rate)

    frame_size = width * height * 3 // 2
    nhevc_out = None
    native_hevc_out = None
    input_stream = None
    input_proc = None
    output_proc = None
    output_proc_cmd: List[str] = []

    if standard_hevc_output:
        output_proc_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{input_fps_for_rate:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            standard_codec,
        ]
        if standard_preset:
            output_proc_cmd.extend(["-preset", standard_preset])
        if standard_bitrate:
            output_proc_cmd.extend(["-b:v", standard_bitrate])
        else:
            output_proc_cmd.extend(["-crf", str(standard_crf)])

        if standard_intra_only:
            # Intra-only standard HEVC for deterministic per-frame coding.
            if standard_codec == "libx265":
                output_proc_cmd.extend(
                    ["-x265-params", "keyint=1:min-keyint=1:scenecut=0"]
                )
            else:
                output_proc_cmd.extend(["-g", "1"])
        output_proc_cmd.append(output_path)
        output_proc = subprocess.Popen(
            output_proc_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    elif native_hevc_output:
        native_hevc_out = open(output_path, "wb")
        native_hevc_out.write(create_parameter_sets(config))
    else:
        nhevc_out = open(output_path, "wb+")
        # Write placeholder header; frame_count will be updated at the end.
        nhevc_out.write(pack_nano_header(width, height, fps_num, fps_den, frame_count=0))

    try:
        if is_container_input:
            input_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-vf",
                f"scale={width}:{height}",
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                str(num_frames),
                "-f",
                "rawvideo",
                "-",
            ]
            input_proc = subprocess.Popen(
                input_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if input_proc.stdout is None:
                raise RuntimeError("Failed to open ffmpeg stdout pipe")
            input_stream = input_proc.stdout
        else:
            input_stream = open(input_path, "rb")

        for frame_idx in range(num_frames):
            data = input_stream.read(frame_size)
            if len(data) < frame_size:
                break

            frame = Frame.from_yuv420p(data, height, width)
            encoded_bytes, recon, stats = encode_frame(
                frame,
                config,
                fast_mode,
                native_minimal_syntax=native_hevc_output,
            )
            recon_bytes = recon.to_yuv420p()

            if standard_hevc_output:
                if output_proc is None or output_proc.stdin is None:
                    raise RuntimeError("Failed to open ffmpeg stdin pipe")
                output_proc.stdin.write(recon_bytes)
                frame_bytes = len(recon_bytes)
            elif native_hevc_output:
                if native_hevc_out is None:
                    raise RuntimeError("Failed to open native HEVC output stream")
                native_hevc_out.write(encoded_bytes)
                frame_bytes = len(encoded_bytes)
            else:
                payload = zlib.compress(recon_bytes, level=6)
                crc = zlib.crc32(recon_bytes) & 0xFFFFFFFF
                frame_type = stats.get("frame_type", "I")
                frame_type_id = FRAME_TYPE_TO_ID.get(frame_type, 0)

                nhevc_out.write(
                    NANO_FRAME_HEADER.pack(
                        frame_type_id, len(recon_bytes), len(payload), crc
                    )
                )
                nhevc_out.write(payload)
                frame_bytes = NANO_FRAME_HEADER.size + len(payload)

            y_psnr = psnr(frame.y.data.astype(np.uint8), recon.y.data.astype(np.uint8))
            psnr_values.append(y_psnr)

            total_stats["frames"] += 1
            total_stats["total_blocks"] += stats["blocks"]
            frame_type = stats.get("frame_type", "I")
            total_stats["encoded_frame_types"].append(frame_type)
            if frame_type in total_stats["encoded_frame_type_counts"]:
                total_stats["encoded_frame_type_counts"][frame_type] += 1

            print(f"Frame {frame_idx}: {frame_bytes} bytes, PSNR: {y_psnr:.2f} dB")

        if not standard_hevc_output and not native_hevc_output:
            # Patch header with actual frame_count.
            nhevc_out.seek(0)
            nhevc_out.write(
                pack_nano_header(
                    width=width,
                    height=height,
                    fps_num=fps_num,
                    fps_den=fps_den,
                    frame_count=total_stats["frames"],
                )
            )
    finally:
        if input_stream is not None:
            input_stream.close()
        if input_proc is not None:
            input_stderr = b""
            if input_proc.stderr is not None:
                input_stderr = input_proc.stderr.read()
                input_proc.stderr.close()
            input_return = input_proc.wait()
            if input_return != 0:
                raise subprocess.CalledProcessError(
                    input_return,
                    input_proc.args,
                    stderr=input_stderr.decode("utf-8", errors="replace"),
                )
        if output_proc is not None:
            output_stderr = b""
            if output_proc.stdin is not None:
                output_proc.stdin.close()
            if output_proc.stderr is not None:
                output_stderr = output_proc.stderr.read()
                output_proc.stderr.close()
            output_return = output_proc.wait()
            if output_return != 0:
                raise subprocess.CalledProcessError(
                    output_return,
                    output_proc_cmd,
                    stderr=output_stderr.decode("utf-8", errors="replace"),
                )
        if nhevc_out is not None:
            nhevc_out.close()
        if native_hevc_out is not None:
            native_hevc_out.close()

    if psnr_values:
        total_stats["avg_psnr"] = sum(psnr_values) / len(psnr_values)
    total_stats["total_bytes"] = os.path.getsize(output_path)
    if total_stats["frames"] > 0:
        total_stats["output_bitrate_kbps"] = (
            total_stats["total_bytes"] * 8.0 * input_fps_for_rate / total_stats["frames"] / 1000.0
        )
    if standard_hevc_output and show_frame_types:
        try:
            encoded_frame_types = probe_input_frame_types(output_path)
            total_stats["encoded_frame_types"] = encoded_frame_types
            total_stats["encoded_frame_type_counts"] = summarize_frame_types(
                encoded_frame_types
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(
                f"Warning: failed to probe output frame types: {exc}",
                file=sys.stderr,
            )
    if show_frame_types and total_stats["input_frame_types"]:
        total_stats["input_frame_types"] = total_stats["input_frame_types"][
            : total_stats["frames"]
        ]
        total_stats["input_frame_type_counts"] = summarize_frame_types(
            total_stats["input_frame_types"]
        )

    return total_stats


def decode_video_nano(
    input_path: str,
    output_path: str,
    codec: str = "libx264",
) -> dict:
    """
    Decode NHEVC1 nano container to YUV420p or container video.

    - If output_path ends with .yuv, raw YUV420p is written directly.
    - For .mp4/.mov/.mkv/.webm, frames are encoded via ffmpeg.
    """
    with open(input_path, "rb") as f:
        header_blob = f.read(NANO_CONTAINER_HEADER.size)
        header = unpack_nano_header(header_blob)

        width = header["width"]
        height = header["height"]
        fps_num = header["fps_num"]
        fps_den = header["fps_den"]
        frame_count = header["frame_count"]
        frame_size = width * height * 3 // 2

        output_is_yuv = output_path.lower().endswith(".yuv")
        fps = fps_num / fps_den if fps_den > 0 else 30.0
        out_stream = None
        output_proc = None
        output_proc_cmd: List[str] = []

        if output_is_yuv:
            out_stream = open(output_path, "wb")
        else:
            output_proc_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
            output_proc = subprocess.Popen(
                output_proc_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        try:
            for frame_idx in range(frame_count):
                frame_header_blob = f.read(NANO_FRAME_HEADER.size)
                if len(frame_header_blob) < NANO_FRAME_HEADER.size:
                    raise ValueError(
                        f"Unexpected EOF in frame header at index {frame_idx}"
                    )
                frame_type_id, raw_size, payload_size, stored_crc = NANO_FRAME_HEADER.unpack(
                    frame_header_blob
                )
                if frame_type_id not in FRAME_ID_TO_TYPE:
                    raise ValueError(f"Frame {frame_idx} has invalid frame type id {frame_type_id}")
                if raw_size != frame_size:
                    raise ValueError(
                        f"Frame {frame_idx} has invalid raw size {raw_size}, expected {frame_size}"
                    )
                payload = f.read(payload_size)
                if len(payload) < payload_size:
                    raise ValueError(
                        f"Unexpected EOF in frame payload at index {frame_idx}"
                    )
                raw = zlib.decompress(payload)
                if len(raw) != raw_size:
                    raise ValueError(
                        f"Frame {frame_idx} decompressed size mismatch: {len(raw)} vs {raw_size}"
                    )
                actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
                if actual_crc != stored_crc:
                    raise ValueError(
                        f"Frame {frame_idx} CRC mismatch: {actual_crc} != {stored_crc}"
                    )
                if output_is_yuv:
                    out_stream.write(raw)
                else:
                    if output_proc is None or output_proc.stdin is None:
                        raise RuntimeError("Failed to open ffmpeg stdin pipe")
                    output_proc.stdin.write(raw)
        finally:
            if out_stream is not None:
                out_stream.close()
            if output_proc is not None:
                output_stderr = b""
                if output_proc.stdin is not None:
                    output_proc.stdin.close()
                if output_proc.stderr is not None:
                    output_stderr = output_proc.stderr.read()
                    output_proc.stderr.close()
                output_return = output_proc.wait()
                if output_return != 0:
                    raise subprocess.CalledProcessError(
                        output_return,
                        output_proc_cmd,
                        stderr=output_stderr.decode("utf-8", errors="replace"),
                    )

    output_bytes = os.path.getsize(output_path)
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "backend": "nano_decode",
        "width": width,
        "height": height,
        "frames": frame_count,
        "fps": fps,
        "output_path": output_path,
        "total_bytes": output_bytes,
        "output_bitrate_kbps": compute_bitrate_kbps(output_bytes, duration),
    }


def encode_video_ffmpeg(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int = 1,
    show_frame_types: bool = False,
    codec: str = "libx265",
    preset: str = "medium",
    crf: int = 28,
    bitrate: str | None = None,
) -> dict:
    """
    Encode video using ffmpeg's HEVC encoders (production backend).

    This path is intended for standards-compliant output and realistic bitrate.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(f"width/height must be even for yuv420p, got {width}x{height}")

    input_fps = 30.0
    try:
        input_meta = probe_video_metadata(input_path)
        if input_meta["fps"] > 0:
            input_fps = input_meta["fps"]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-c:v",
        codec,
    ]

    if preset:
        cmd.extend(["-preset", preset])
    if bitrate:
        cmd.extend(["-b:v", bitrate])
    else:
        cmd.extend(["-crf", str(crf)])
    if num_frames > 0:
        cmd.extend(["-frames:v", str(num_frames)])

    cmd.append(output_path)
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    output_meta = probe_video_metadata(output_path)
    output_bytes = os.path.getsize(output_path)
    output_duration = output_meta["duration"]
    output_frame_count = output_meta["frame_count"]
    output_fps = output_meta["fps"] if output_meta["fps"] > 0 else input_fps

    encoded_frame_types: List[str] = []
    input_frame_types: List[str] = []
    if show_frame_types:
        try:
            encoded_frame_types = probe_input_frame_types(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            encoded_frame_types = []
        try:
            input_frame_types = probe_input_frame_types(input_path)
            if num_frames > 0:
                input_frame_types = input_frame_types[:num_frames]
        except (subprocess.CalledProcessError, FileNotFoundError):
            input_frame_types = []

    if output_frame_count <= 0:
        if encoded_frame_types:
            output_frame_count = len(encoded_frame_types)
        elif num_frames > 0:
            output_frame_count = num_frames

    if output_duration <= 0 and output_frame_count > 0 and output_fps > 0:
        output_duration = output_frame_count / output_fps

    stats = {
        "backend": "ffmpeg",
        "frames": output_frame_count,
        "total_bytes": output_bytes,
        "total_blocks": 0,
        "avg_psnr": 0.0,
        "output_bitrate_kbps": compute_bitrate_kbps(output_bytes, output_duration),
        "encoded_frame_types": encoded_frame_types,
        "encoded_frame_type_counts": summarize_frame_types(encoded_frame_types),
        "input_frame_types": input_frame_types,
        "input_frame_type_counts": summarize_frame_types(input_frame_types),
    }
    return stats


def encode_video(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int = 1,
    qp: int = 27,
    fast_mode: bool = True,
    show_frame_types: bool = False,
    backend: str = "nano",
    ffmpeg_codec: str = "libx265",
    ffmpeg_preset: str = "medium",
    ffmpeg_crf: int = 28,
    ffmpeg_bitrate: str | None = None,
    nano_standard_hevc: bool = False,
    nano_standard_codec: str = "libx265",
    nano_standard_preset: str = "medium",
    nano_standard_crf: int = 28,
    nano_standard_bitrate: str | None = None,
    nano_standard_intra_only: bool = False,
    nano_native_hevc: bool = False,
) -> dict:
    """Unified entry point for nano or ffmpeg backend."""
    if backend == "nano":
        return encode_video_nano(
            input_path=input_path,
            output_path=output_path,
            width=width,
            height=height,
            num_frames=num_frames,
            qp=qp,
            fast_mode=fast_mode,
            show_frame_types=show_frame_types,
            standard_hevc_output=nano_standard_hevc,
            native_hevc_output=nano_native_hevc,
            standard_codec=nano_standard_codec,
            standard_preset=nano_standard_preset,
            standard_crf=nano_standard_crf,
            standard_bitrate=nano_standard_bitrate,
            standard_intra_only=nano_standard_intra_only,
        )
    if backend == "ffmpeg":
        return encode_video_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            width=width,
            height=height,
            num_frames=num_frames,
            show_frame_types=show_frame_types,
            codec=ffmpeg_codec,
            preset=ffmpeg_preset,
            crf=ffmpeg_crf,
            bitrate=ffmpeg_bitrate,
        )
    raise ValueError(f"Unknown backend: {backend}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="nano-hevc: Minimal HEVC encoder")
    parser.add_argument("input", help="Input video file (YUV420p or MP4)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file (.nhevc/.hevc/.mp4/.yuv depending on mode)",
    )
    parser.add_argument("--width", type=int, help="Video width")
    parser.add_argument("--height", type=int, help="Video height")
    parser.add_argument(
        "--frames", type=int, default=1, help="Number of frames to encode"
    )
    parser.add_argument(
        "--qp", type=int, default=27, help="Quantization parameter (0-51)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode (fewer intra modes)"
    )
    parser.add_argument(
        "--show-frame-types",
        action="store_true",
        help="Show input and encoded frame type statistics (I/P/B)",
    )
    parser.add_argument(
        "--backend",
        choices=["nano", "ffmpeg"],
        default="nano",
        help="Encoder backend: nano (educational) or ffmpeg (production)",
    )
    parser.add_argument(
        "--ffmpeg-codec",
        default="libx265",
        help="ffmpeg backend codec (e.g. libx265, hevc_videotoolbox)",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="ffmpeg backend preset",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=28,
        help="ffmpeg backend CRF quality (used when bitrate is not set)",
    )
    parser.add_argument(
        "--ffmpeg-bitrate",
        help="ffmpeg backend target bitrate (e.g. 1200k); overrides --ffmpeg-crf",
    )
    parser.add_argument(
        "--decode-nano",
        action="store_true",
        help="Decode NHEVC1 nano container input to output file",
    )
    parser.add_argument(
        "--decode-codec",
        default="libx264",
        help="Codec used when --decode-nano outputs MP4/MOV/MKV/WEBM",
    )
    parser.add_argument(
        "--nano-standard-hevc",
        action="store_true",
        help="With --backend nano, output standards-compliant HEVC by piping reconstructed frames to ffmpeg",
    )
    parser.add_argument(
        "--nano-native-hevc",
        action="store_true",
        help="With --backend nano, output native HEVC from nano-hevc syntax (experimental)",
    )
    parser.add_argument(
        "--nano-standard-codec",
        default="libx265",
        help="Codec for --nano-standard-hevc (e.g. libx265, hevc_videotoolbox)",
    )
    parser.add_argument(
        "--nano-standard-preset",
        default="medium",
        help="Preset for --nano-standard-hevc",
    )
    parser.add_argument(
        "--nano-standard-crf",
        type=int,
        default=28,
        help="CRF for --nano-standard-hevc when bitrate is not set",
    )
    parser.add_argument(
        "--nano-standard-bitrate",
        help="Target bitrate for --nano-standard-hevc (e.g. 1200k); overrides --nano-standard-crf",
    )
    parser.add_argument(
        "--nano-standard-intra-only",
        action="store_true",
        help="With --nano-standard-hevc, force all-I output (keyint=1)",
    )

    args = parser.parse_args()

    if args.decode_nano:
        print("nano-hevc decoder")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print()

        stats = decode_video_nano(
            input_path=args.input,
            output_path=args.output,
            codec=args.decode_codec,
        )
        print("Decoding complete!")
        print(f"  Frames:     {stats['frames']}")
        print(f"  Size:       {stats['width']}x{stats['height']}")
        print(f"  FPS:        {stats['fps']:.3f}")
        print(f"  Total size: {stats['total_bytes']} bytes")
        print(f"  Avg bitrate: {stats['output_bitrate_kbps']:.1f} kbps")
        return

    if args.width is None or args.height is None:
        parser.error("--width and --height are required for encode mode")

    print(f"nano-hevc encoder")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Size:   {args.width}x{args.height}")
    print(f"Backend:{args.backend}")
    if args.backend == "nano":
        if args.nano_standard_hevc and args.nano_native_hevc:
            parser.error("--nano-standard-hevc and --nano-native-hevc are mutually exclusive")
        if args.nano_standard_hevc:
            print("Mode:   standard-hevc-from-nano")
            print(f"Codec:  {args.nano_standard_codec}")
            print(f"Intra:  {'yes' if args.nano_standard_intra_only else 'no'}")
            if args.nano_standard_bitrate:
                print(f"Rate:   {args.nano_standard_bitrate}")
            else:
                print(f"CRF:    {args.nano_standard_crf}")
        elif args.nano_native_hevc:
            print("Mode:   native-hevc-from-nano")
        else:
            print("Mode:   nano-container")
        print(f"QP:     {args.qp}")
    else:
        print(f"Codec:  {args.ffmpeg_codec}")
        if args.ffmpeg_bitrate:
            print(f"Rate:   {args.ffmpeg_bitrate}")
        else:
            print(f"CRF:    {args.ffmpeg_crf}")
    print()

    stats = encode_video(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        qp=args.qp,
        fast_mode=args.fast,
        show_frame_types=args.show_frame_types,
        backend=args.backend,
        ffmpeg_codec=args.ffmpeg_codec,
        ffmpeg_preset=args.ffmpeg_preset,
        ffmpeg_crf=args.ffmpeg_crf,
        ffmpeg_bitrate=args.ffmpeg_bitrate,
        nano_standard_hevc=args.nano_standard_hevc,
        nano_standard_codec=args.nano_standard_codec,
        nano_standard_preset=args.nano_standard_preset,
        nano_standard_crf=args.nano_standard_crf,
        nano_standard_bitrate=args.nano_standard_bitrate,
        nano_standard_intra_only=args.nano_standard_intra_only,
        nano_native_hevc=args.nano_native_hevc,
    )

    print()
    print(f"Encoding complete!")
    print(f"  Frames:     {stats['frames']}")
    if "output_format" in stats:
        print(f"  Format:     {stats['output_format']}")
    print(f"  Total size: {stats['total_bytes']} bytes")
    if stats["backend"] == "nano":
        print(f"  Avg PSNR:   {stats['avg_psnr']:.2f} dB")

    if stats["output_bitrate_kbps"] > 0:
        print(f"  Avg bitrate: {stats['output_bitrate_kbps']:.1f} kbps")
    elif stats["frames"] > 0:
        bitrate = stats["total_bytes"] * 8 * 30 / stats["frames"] / 1000
        print(f"  Est. bitrate @ 30fps: {bitrate:.1f} kbps")

    if args.show_frame_types:
        enc_counts = stats["encoded_frame_type_counts"]
        print(
            f"  Encoded frame types: I={enc_counts['I']} P={enc_counts['P']} B={enc_counts['B']}"
        )
        in_counts = stats["input_frame_type_counts"]
        if any(in_counts.values()):
            print(
                f"  Input frame types:   I={in_counts['I']} P={in_counts['P']} B={in_counts['B']}"
            )


if __name__ == "__main__":
    main()
