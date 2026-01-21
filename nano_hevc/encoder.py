"""
HEVC encoder main module.

Provides a complete encoding pipeline for intra-only HEVC encoding.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Iterator
from dataclasses import dataclass
import sys
import argparse
import numpy as np

from nano_hevc.frame import Frame, Plane
from nano_hevc.block import BlockView, iterate_blocks
from nano_hevc.intra import (
    intra_dc_predict,
    intra_planar_predict,
    intra_angular_predict,
    residual_block,
    reconstruct_block,
    clip_to_pixel_range,
    INTRA_PRED_ANGLE,
)
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block, is_all_zero
from nano_hevc.scan import (
    scan_block,
    get_scan_type_for_intra_mode,
    find_last_significant_coeff,
)
from nano_hevc.bitstream import BitstreamWriter
from nano_hevc.cabac import (
    CabacEncoder,
    CabacContext,
    init_contexts_for_slice,
    encode_coeff_abs_level_remaining,
)
from nano_hevc.nal import (
    HEVCConfig,
    create_parameter_sets,
    NAL_IDR_N_LP,
    NAL_TRAIL_R,
    write_nal_unit_header,
    write_start_code_prefix,
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
    recon = Frame.zeros(frame.height, frame.width, dtype=np.int16)

    cabac = CabacEncoder()
    contexts = init_contexts_for_slice(2, config.qp)

    stats = {"blocks": 0, "modes": {}, "cbf_count": 0, "total_coeffs": 0}

    block_size = min(config.max_cu_size, 32)

    for plane_name, src_plane, dst_plane in [
        ("Y", frame.y, recon.y),
        ("U", frame.u, recon.u),
        ("V", frame.v, recon.v),
    ]:
        bs = block_size if plane_name == "Y" else max(4, block_size // 2)
        is_luma = plane_name == "Y"

        for y in range(0, src_plane.height, bs):
            for x in range(0, src_plane.width, bs):
                actual_h = min(bs, src_plane.height - y)
                actual_w = min(bs, src_plane.width - x)
                actual_size = min(actual_h, actual_w)

                if actual_size < 4:
                    continue

                valid_sizes = [32, 16, 8, 4]
                for vs in valid_sizes:
                    if actual_size >= vs:
                        actual_size = vs
                        break

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

    writer = BitstreamWriter()
    write_start_code_prefix(writer)
    write_nal_unit_header(writer, NAL_IDR_N_LP)

    writer.write_bit(1)
    writer.write_ue(0)
    writer.write_ue(2)
    writer.write_se(config.qp - 26)
    writer.byte_align()

    header_bytes = writer.get_bytes()

    encoded_data = bytearray(header_bytes)
    encoded_data.extend(slice_data)

    return bytes(encoded_data), recon, stats


def encode_video(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int = 1,
    qp: int = 27,
    fast_mode: bool = True,
) -> dict:
    """
    Encode video from YUV file or MP4 to HEVC bitstream.

    For MP4 input, requires ffmpeg to extract YUV frames.
    """
    config = HEVCConfig(width=width, height=height, qp=qp)

    param_sets = create_parameter_sets(config)

    total_stats = {
        "frames": 0,
        "total_bytes": len(param_sets),
        "total_blocks": 0,
        "avg_psnr": 0.0,
    }

    psnr_values = []

    with open(output_path, "wb") as out_file:
        out_file.write(param_sets)

        if input_path.endswith(".mp4") or input_path.endswith(".mov"):
            import subprocess
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

                    print(
                        f"Frame {frame_idx}: {len(encoded_bytes)} bytes, PSNR: {y_psnr:.2f} dB"
                    )

    if psnr_values:
        total_stats["avg_psnr"] = sum(psnr_values) / len(psnr_values)

    return total_stats


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

    args = parser.parse_args()

    print(f"nano-hevc encoder")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Size:   {args.width}x{args.height}")
    print(f"QP:     {args.qp}")
    print()

    stats = encode_video(
        args.input,
        args.output,
        args.width,
        args.height,
        args.frames,
        args.qp,
        args.fast,
    )

    print()
    print(f"Encoding complete!")
    print(f"  Frames:     {stats['frames']}")
    print(f"  Total size: {stats['total_bytes']} bytes")
    print(f"  Avg PSNR:   {stats['avg_psnr']:.2f} dB")

    if stats["frames"] > 0:
        bitrate = stats["total_bytes"] * 8 * 30 / stats["frames"] / 1000
        print(f"  Est. bitrate @ 30fps: {bitrate:.1f} kbps")


if __name__ == "__main__":
    main()
