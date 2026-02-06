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
    frame: Frame, config: HEVCConfig, fast_mode: bool = True
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
) -> dict:
    """
    Encode video from YUV file or MP4 to HEVC bitstream.

    For MP4 input, requires ffmpeg to extract YUV frames.
    """
    validate_dimensions(width, height)

    config = HEVCConfig(width=width, height=height, qp=qp)

    param_sets = create_parameter_sets(config)

    total_stats = {
        "backend": "nano",
        "frames": 0,
        "total_bytes": len(param_sets),
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

    with open(output_path, "wb") as out_file:
        out_file.write(param_sets)

        if is_container_input:
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmpdir:
                yuv_path = os.path.join(tmpdir, "input.yuv")

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-vf",
                    f"scale={width}:{height}",
                    "-pix_fmt",
                    "yuv420p",
                    "-frames:v",
                    str(num_frames),
                    yuv_path,
                ]
                subprocess.run(cmd, capture_output=True, check=True)

                with open(yuv_path, "rb") as yuv_file:
                    frame_size = width * height * 3 // 2

                    for frame_idx in range(num_frames):
                        data = yuv_file.read(frame_size)
                        if len(data) < frame_size:
                            break

                        frame = Frame.from_yuv420p(data, height, width)

                        encoded_bytes, recon, stats = encode_frame(
                            frame, config, fast_mode
                        )
                        out_file.write(encoded_bytes)

                        y_psnr = psnr(
                            frame.y.data.astype(np.uint8), recon.y.data.astype(np.uint8)
                        )
                        psnr_values.append(y_psnr)

                        total_stats["frames"] += 1
                        total_stats["total_bytes"] += len(encoded_bytes)
                        total_stats["total_blocks"] += stats["blocks"]
                        frame_type = stats.get("frame_type", "I")
                        total_stats["encoded_frame_types"].append(frame_type)
                        if frame_type in total_stats["encoded_frame_type_counts"]:
                            total_stats["encoded_frame_type_counts"][frame_type] += 1

                        print(
                            f"Frame {frame_idx}: {len(encoded_bytes)} bytes, PSNR: {y_psnr:.2f} dB"
                        )
        else:
            with open(input_path, "rb") as yuv_file:
                frame_size = width * height * 3 // 2

                for frame_idx in range(num_frames):
                    data = yuv_file.read(frame_size)
                    if len(data) < frame_size:
                        break

                    frame = Frame.from_yuv420p(data, height, width)

                    encoded_bytes, recon, stats = encode_frame(frame, config, fast_mode)
                    out_file.write(encoded_bytes)

                    y_psnr = psnr(
                        frame.y.data.astype(np.uint8), recon.y.data.astype(np.uint8)
                    )
                    psnr_values.append(y_psnr)

                    total_stats["frames"] += 1
                    total_stats["total_bytes"] += len(encoded_bytes)
                    total_stats["total_blocks"] += stats["blocks"]
                    frame_type = stats.get("frame_type", "I")
                    total_stats["encoded_frame_types"].append(frame_type)
                    if frame_type in total_stats["encoded_frame_type_counts"]:
                        total_stats["encoded_frame_type_counts"][frame_type] += 1

                    print(
                        f"Frame {frame_idx}: {len(encoded_bytes)} bytes, PSNR: {y_psnr:.2f} dB"
                    )

    if psnr_values:
        total_stats["avg_psnr"] = sum(psnr_values) / len(psnr_values)
    if total_stats["frames"] > 0:
        total_stats["output_bitrate_kbps"] = (
            total_stats["total_bytes"]
            * 8.0
            * input_fps_for_rate
            / total_stats["frames"]
            / 1000.0
        )
    if show_frame_types and total_stats["input_frame_types"]:
        total_stats["input_frame_types"] = total_stats["input_frame_types"][
            : total_stats["frames"]
        ]
        total_stats["input_frame_type_counts"] = summarize_frame_types(
            total_stats["input_frame_types"]
        )

    return total_stats


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
        "-o", "--output", required=True, help="Output HEVC file (.265 or .hevc)"
    )
    parser.add_argument("--width", type=int, required=True, help="Video width")
    parser.add_argument("--height", type=int, required=True, help="Video height")
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

    args = parser.parse_args()

    print(f"nano-hevc encoder")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Size:   {args.width}x{args.height}")
    print(f"Backend:{args.backend}")
    if args.backend == "nano":
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
    )

    print()
    print(f"Encoding complete!")
    print(f"  Frames:     {stats['frames']}")
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
