"""
CLI demo for nano-hevc intra prediction.

Usage:
    python -m nano_hevc demo --width 352 --height 288 [--block-size 8]
    python -m nano_hevc encode input.yuv --width 352 --height 288 -o output.yuv
"""

from __future__ import annotations
import argparse
import sys
import numpy as np

from nano_hevc.frame import Frame, Plane
from nano_hevc.block import BlockView, iterate_blocks
from nano_hevc.intra import (
    intra_dc_predict,
    intra_planar_predict,
    residual_block,
    reconstruct_block,
    clip_to_pixel_range,
)
from nano_hevc.metrics import psnr, mse, sad, residual_energy


def create_test_frame(height: int, width: int) -> Frame:
    """Create a test frame with gradients and edges for demo."""
    y = np.zeros((height, width), dtype=np.uint8)

    # test pattern: four quadrants with different content types
    # q1: horizontal gradient, q2: vertical gradient
    # q3: flat region, q4: diagonal gradient
    y[:height//2, :width//2] = np.tile(
        np.linspace(50, 200, width//2, dtype=np.uint8),
        (height//2, 1)
    )
    y[:height//2, width//2:] = np.tile(
        np.linspace(50, 200, height//2, dtype=np.uint8).reshape(-1, 1),
        (1, width - width//2)
    )
    y[height//2:, :width//2] = 128
    for i in range(height - height//2):
        for j in range(width - width//2):
            y[height//2 + i, width//2 + j] = min(255, 50 + i + j)

    u = np.full((height//2, width//2), 128, dtype=np.uint8)
    v = np.full((height//2, width//2), 128, dtype=np.uint8)

    return Frame(
        y=Plane(data=y.astype(np.int16)),
        u=Plane(data=u.astype(np.int16)),
        v=Plane(data=v.astype(np.int16)),
    )


def demo_predictions(height: int, width: int, block_size: int) -> None:
    """demonstrate DC vs Planar prediction on test patterns."""
    print(f"nano-hevc intra prediction demo")
    print(f"Frame: {width}x{height}, Block size: {block_size}x{block_size}")
    print("=" * 60)

    frame = create_test_frame(height, width)
    plane = frame.y

    dc_wins = 0
    planar_wins = 0
    total_blocks = 0

    dc_total_energy = 0
    planar_total_energy = 0

    print(f"\n{'Block':<12} {'DC Energy':>12} {'Planar Energy':>14} {'Winner':>10}")
    print("-" * 60)

    for blk in iterate_blocks(plane, block_size):
        total_blocks += 1
        orig = blk.copy_pixels()

        top = blk.get_top_neighbors()
        left = blk.get_left_neighbors()

        # DC prediction
        dc_pred = intra_dc_predict(top, left, block_size)
        dc_res = residual_block(orig, dc_pred)
        dc_energy = residual_energy(dc_res)
        dc_total_energy += dc_energy

        # Planar prediction
        top_right = int(top[-1]) if blk.x + block_size < plane.width else int(top[-1])
        bottom_left = int(left[-1]) if blk.y + block_size < plane.height else int(left[-1])
        planar_pred = intra_planar_predict(top, left, top_right, bottom_left, block_size)
        planar_res = residual_block(orig, planar_pred)
        planar_energy = residual_energy(planar_res)
        planar_total_energy += planar_energy

        winner = "DC" if dc_energy <= planar_energy else "Planar"
        if dc_energy <= planar_energy:
            dc_wins += 1
        else:
            planar_wins += 1

        if total_blocks <= 16:
            print(f"({blk.x:3},{blk.y:3})    {dc_energy:>12} {planar_energy:>14} {winner:>10}")

    if total_blocks > 16:
        print(f"... ({total_blocks - 16} more blocks)")

    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  DC wins:      {dc_wins} ({100*dc_wins/total_blocks:.1f}%)")
    print(f"  Planar wins:  {planar_wins} ({100*planar_wins/total_blocks:.1f}%)")
    print(f"\n  DC total residual energy:     {dc_total_energy:,}")
    print(f"  Planar total residual energy: {planar_total_energy:,}")

    # reconstruct using best mode per block
    recon_plane = Plane.zeros(height, width, dtype=np.int16)
    for blk in iterate_blocks(plane, block_size):
        orig = blk.copy_pixels()
        top = blk.get_top_neighbors()
        left = blk.get_left_neighbors()

        dc_pred = intra_dc_predict(top, left, block_size)
        dc_energy = residual_energy(residual_block(orig, dc_pred))

        top_right = int(top[-1])
        bottom_left = int(left[-1])
        planar_pred = intra_planar_predict(top, left, top_right, bottom_left, block_size)
        planar_energy = residual_energy(residual_block(orig, planar_pred))

        best_pred = dc_pred if dc_energy <= planar_energy else planar_pred
        recon_blk = BlockView(plane=recon_plane, x=blk.x, y=blk.y, size=block_size)
        recon_blk.write_pixels(clip_to_pixel_range(best_pred).astype(np.int16))

    orig_y = plane.data.astype(np.uint8)
    recon_y = recon_plane.data.astype(np.uint8)
    final_psnr = psnr(orig_y, recon_y)

    print(f"\n  Reconstruction PSNR (best mode): {final_psnr:.2f} dB")


def encode_frame_intra(
    frame: Frame,
    block_size: int,
    output_path: str | None = None
) -> tuple[Frame, dict]:
    """Encode a frame using intra prediction only (no transform/quant)."""
    recon = Frame.zeros(frame.height, frame.width, dtype=np.int16)
    stats = {"dc": 0, "planar": 0, "blocks": 0}

    for plane_name, src_plane, dst_plane in [
        ("Y", frame.y, recon.y),
        ("U", frame.u, recon.u),
        ("V", frame.v, recon.v),
    ]:
        bs = block_size if plane_name == "Y" else block_size // 2
        if bs < 4:
            bs = 4

        for blk in iterate_blocks(src_plane, bs):
            orig = blk.copy_pixels()
            top = blk.get_top_neighbors()
            left = blk.get_left_neighbors()

            dc_pred = intra_dc_predict(top, left, bs)
            dc_energy = residual_energy(residual_block(orig, dc_pred))

            top_right = int(top[-1])
            bottom_left = int(left[-1])
            planar_pred = intra_planar_predict(top, left, top_right, bottom_left, bs)
            planar_energy = residual_energy(residual_block(orig, planar_pred))

            if dc_energy <= planar_energy:
                best_pred = dc_pred
                stats["dc"] += 1
            else:
                best_pred = planar_pred
                stats["planar"] += 1
            stats["blocks"] += 1

            dst_blk = BlockView(plane=dst_plane, x=blk.x, y=blk.y, size=bs)
            dst_blk.write_pixels(clip_to_pixel_range(best_pred).astype(np.int16))

    if output_path:
        with open(output_path, "wb") as f:
            f.write(recon.to_yuv420p())
        print(f"Wrote: {output_path}")

    return recon, stats


def cmd_demo(args: argparse.Namespace) -> None:
    demo_predictions(args.height, args.width, args.block_size)


def cmd_encode(args: argparse.Namespace) -> None:
    with open(args.input, "rb") as f:
        data = f.read()

    expected_size = args.width * args.height * 3 // 2
    if len(data) < expected_size:
        print(f"Error: file too small. Expected {expected_size} bytes, got {len(data)}")
        sys.exit(1)

    frame = Frame.from_yuv420p(data[:expected_size], args.height, args.width)
    print(f"Loaded: {args.input} ({args.width}x{args.height})")

    recon, stats = encode_frame_intra(frame, args.block_size, args.output)

    orig_y = frame.y.data.astype(np.uint8)
    recon_y = recon.y.data.astype(np.uint8)
    y_psnr = psnr(orig_y, recon_y)

    print(f"\nResults:")
    print(f"  Blocks: {stats['blocks']} (DC: {stats['dc']}, Planar: {stats['planar']})")
    print(f"  Y-PSNR: {y_psnr:.2f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nano-hevc",
        description="Minimal HEVC intra prediction demo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run demo with synthetic test frame")
    demo_parser.add_argument("--width", type=int, default=64, help="Frame width")
    demo_parser.add_argument("--height", type=int, default=64, help="Frame height")
    demo_parser.add_argument("--block-size", type=int, default=8, help="Block size (4,8,16,32)")
    demo_parser.set_defaults(func=cmd_demo)

    enc_parser = subparsers.add_parser("encode", help="Encode a YUV420p file")
    enc_parser.add_argument("input", help="Input YUV420p file")
    enc_parser.add_argument("--width", type=int, required=True, help="Frame width")
    enc_parser.add_argument("--height", type=int, required=True, help="Frame height")
    enc_parser.add_argument("--block-size", type=int, default=8, help="Block size")
    enc_parser.add_argument("-o", "--output", help="Output reconstructed YUV file")
    enc_parser.set_defaults(func=cmd_encode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
