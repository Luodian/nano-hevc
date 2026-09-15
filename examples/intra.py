#!/usr/bin/env python3
"""Compare DC and planar prediction: uv run python examples/intra.py --block-size 8."""

import argparse

import numpy as np

from nano_hevc.block import BlockView, iterate_blocks
from nano_hevc.frame import Plane
from nano_hevc.intra import intra_dc_predict, intra_planar_predict, residual_block
from nano_hevc.metrics import psnr, residual_energy


def demo_predictions(height: int, width: int, block_size: int) -> None:
    """Compare prediction errors on gradients, a flat region, and an edge."""
    pixels = np.zeros((height, width), dtype=np.int16)
    pixels[:height // 2, :width // 2] = np.linspace(50, 200, width // 2)
    pixels[:height // 2, width // 2:] = np.linspace(50, 200, height // 2)[:, None]
    pixels[height // 2:, :width // 2] = 128
    y, x = np.indices((height - height // 2, width - width // 2))
    pixels[height // 2:, width // 2:] = np.minimum(255, 50 + y + x)
    plane = Plane(pixels)
    prediction = Plane.zeros(height, width)
    wins = np.zeros(2, dtype=int)
    total_energy = np.zeros(2, dtype=np.int64)

    print(f"Intra prediction: {width}x{height}, {block_size}x{block_size} blocks")
    print("Block        DC energy   Planar energy   Winner")
    for index, block in enumerate(iterate_blocks(plane, block_size)):
        top, left = block.get_top_neighbors(), block.get_left_neighbors()
        candidates = (
            intra_dc_predict(top, left, block_size),
            intra_planar_predict(top, left, int(top[-1]), int(left[-1]), block_size),
        )
        energies = [residual_energy(residual_block(block.copy_pixels(), p)) for p in candidates]
        winner = int(energies[1] < energies[0])
        wins[winner] += 1
        total_energy += energies
        BlockView(prediction, block.x, block.y, block_size).write_pixels(candidates[winner])
        if index < 16:
            print(f"({block.x:3},{block.y:3}) {energies[0]:12} {energies[1]:15}   {('DC', 'Planar')[winner]}")

    for name, count, energy in zip(("DC", "Planar"), wins, total_energy):
        print(f"{name}: {count} wins, {energy:,} total residual energy")
    print(f"Prediction PSNR: {psnr(pixels, prediction.data):.2f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--block-size", type=int, choices=(4, 8, 16, 32), default=8)
    args = parser.parse_args()
    if any(size <= 0 or size % args.block_size for size in (args.width, args.height)):
        parser.error("width and height must be positive multiples of block size")
    demo_predictions(args.height, args.width, args.block_size)


if __name__ == "__main__":
    main()
