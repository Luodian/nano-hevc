#!/usr/bin/env python3
"""
Inter Prediction: How I-Frames Enable P and B Frame Compression
================================================================

Core equation (same for ALL frame types):
    orig = pred + residual

What differs is where the prediction comes from:
    I-frame: spatial neighbors within the same frame (intra)
    P-frame: motion-compensated block from one past reference
    B-frame: average of blocks from two references (past + future)

Run:  python examples/inter_p_simple.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nano_hevc.frame import Frame, Plane
from nano_hevc.intra import residual_block, reconstruct_block, clip_to_pixel_range
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block, estimate_bits
from nano_hevc.metrics import psnr, sad
from nano_hevc.encoder import (
    get_reference_pixels,
    evaluate_intra_mode,
    INTRA_MODE_DC,
    INTRA_MODE_PLANAR,
)


# ── config ────────────────────────────────────────────────────────────

W, H = 64, 64
BS = 8
QP = 22
SEARCH_RANGE = 16

# per-block syntax overhead (bits) to model real HEVC signaling cost.
# I blocks: split_cu + part_mode + intra_mode + cbf ≈ 6 bits
# P/B blocks: skip/merge signaling ≈ 2 bits (MV cost added separately)
I_SYNTAX_BITS = 6
INTER_SYNTAX_BITS = 2

# block-aligned moving rectangle (2x2 blocks)
OBJ_W, OBJ_H = 16, 16
OBJ_X0, OBJ_Y = 0, 24  # starts at block boundary
DX = 8  # shift 1 block per frame

# static random background (identical across all frames)
_RNG = np.random.RandomState(7)
_BG = np.clip(128 + _RNG.randint(-30, 31, size=(H, W)), 16, 235).astype(np.int16)


# ── synthetic video ───────────────────────────────────────────────────

def make_frame(idx: int) -> Frame:
    y = _BG.copy()
    x0 = OBJ_X0 + idx * DX
    y[OBJ_Y : OBJ_Y + OBJ_H, x0 : x0 + OBJ_W] = 220
    f = Frame.zeros(H, W, dtype=np.int16)
    f.y.data[:] = y
    f.u.data[:] = 128
    f.v.data[:] = 128
    return f


# ── lossy residual coding (shared by all frame types) ────────────────

def code_residual(res: np.ndarray, qp: int) -> tuple[np.ndarray, int]:
    """DCT -> quantize -> dequantize -> IDCT. Returns (recon_residual, bits)."""
    coeff = forward_transform(res, use_dst=False)
    quant = quantize_block(coeff, qp)
    bits = estimate_bits(quant)
    recon = inverse_transform(dequantize_block(quant, qp), use_dst=False)
    return recon.astype(np.int16), bits


# ── motion estimation ─────────────────────────────────────────────────

def motion_search(
    block: np.ndarray, ref: Plane, bx: int, by: int,
) -> tuple[np.ndarray, int, int, int]:
    """Full search at integer-pel. Returns (pred, mvx, mvy, sad)."""
    best_sad = 2**31
    best_pred = ref.data[by : by + BS, bx : bx + BS].astype(np.int16)
    best_mx, best_my = 0, 0

    for dy in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
        for dx in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
            ry, rx = by + dy, bx + dx
            if ry < 0 or rx < 0 or ry + BS > ref.height or rx + BS > ref.width:
                continue
            blk = ref.data[ry : ry + BS, rx : rx + BS]
            s = sad(block, blk)
            if s < best_sad:
                best_sad = s
                best_pred = blk.astype(np.int16)
                best_mx, best_my = dx, dy

    return best_pred, best_mx, best_my, best_sad


# ── I-frame encoder (intra mode selection + estimate_bits) ────────────

def encode_i(frame: Frame, qp: int) -> tuple[Plane, int]:
    """Intra encode with DC/Planar/Angular mode selection.

    Uses estimate_bits for bit counting — same method as P/B —
    so the numbers are directly comparable.
    """
    src = frame.y
    recon = Plane.zeros(H, W, dtype=np.int16)
    total_bits = 0

    for by in range(0, H, BS):
        for bx in range(0, W, BS):
            orig = src.data[by : by + BS, bx : bx + BS].astype(np.int16)
            top, left, tr, bl, tl = get_reference_pixels(recon, bx, by, BS)

            best_cost = float("inf")
            best_recon = best_bits = None
            for mode in [INTRA_MODE_DC, INTRA_MODE_PLANAR, 10, 26]:
                _, coeffs, rblk, cost = evaluate_intra_mode(
                    orig, top, left, tr, bl, tl, mode, BS, qp, BS == 4,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_recon = rblk
                    best_bits = estimate_bits(coeffs)

            recon.data[by : by + BS, bx : bx + BS] = best_recon
            total_bits += best_bits + I_SYNTAX_BITS

    return recon, total_bits


# ── P-frame encoder ──────────────────────────────────────────────────

def encode_p(frame: Frame, ref: Plane, qp: int) -> tuple[Plane, int, list]:
    src = frame.y
    recon = Plane.zeros(H, W, dtype=np.int16)
    total_bits = 0
    mvs = []

    for by in range(0, H, BS):
        for bx in range(0, W, BS):
            orig = src.data[by : by + BS, bx : bx + BS].astype(np.int16)
            pred, mx, my, _ = motion_search(orig, ref, bx, by)
            mvs.append((bx, by, mx, my))

            res = residual_block(orig, pred)
            rr, bits = code_residual(res, qp)
            recon.data[by : by + BS, bx : bx + BS] = clip_to_pixel_range(
                reconstruct_block(pred, rr)
            )
            mv_bits = (abs(mx) > 0) * 4 + (abs(my) > 0) * 4
            total_bits += bits + mv_bits + INTER_SYNTAX_BITS

    return recon, total_bits, mvs


# ── B-frame encoder ──────────────────────────────────────────────────

def encode_b(frame: Frame, r0: Plane, r1: Plane, qp: int) -> tuple[Plane, int, list]:
    src = frame.y
    recon = Plane.zeros(H, W, dtype=np.int16)
    total_bits = 0
    info = []

    for by in range(0, H, BS):
        for bx in range(0, W, BS):
            orig = src.data[by : by + BS, bx : bx + BS].astype(np.int16)

            p0, m0x, m0y, s0 = motion_search(orig, r0, bx, by)
            p1, m1x, m1y, s1 = motion_search(orig, r1, bx, by)
            bi = ((p0.astype(np.int32) + p1.astype(np.int32) + 1) >> 1).astype(np.int16)
            s_bi = sad(orig, bi)

            if s_bi <= min(s0, s1):
                pred, mode = bi, "bi"
            elif s0 <= s1:
                pred, mode = p0, "L0"
            else:
                pred, mode = p1, "L1"

            info.append((bx, by, m0x, m0y, m1x, m1y, mode))

            res = residual_block(orig, pred)
            rr, bits = code_residual(res, qp)
            recon.data[by : by + BS, bx : bx + BS] = clip_to_pixel_range(
                reconstruct_block(pred, rr)
            )
            mv_bits = (abs(m0x) > 0) * 4 + (abs(m0y) > 0) * 4
            if mode == "bi":
                mv_bits += (abs(m1x) > 0) * 4 + (abs(m1y) > 0) * 4
            total_bits += bits + mv_bits + INTER_SYNTAX_BITS

    return recon, total_bits, info


# ── main ──────────────────────────────────────────────────────────────

def sect(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    # 3 frames, decode order: I0 -> P2 -> B1
    f0, f1, f2 = make_frame(0), make_frame(1), make_frame(2)
    n_blocks = (H // BS) * (W // BS)

    # ── step 1: raw prediction quality (SAD, no coding) ──────────────
    sect("STEP 1: RAW PREDICTION QUALITY")
    print("""
  Measure SAD before any coding. Original frames as references.
  Lower SAD = better prediction = fewer bits after coding.
""")

    i_sad = p_sad = b_sad = 0
    p_zero = b_zero = 0

    for by in range(0, H, BS):
        for bx in range(0, W, BS):
            o0 = f0.y.data[by : by + BS, bx : bx + BS]
            o1 = f1.y.data[by : by + BS, bx : bx + BS]
            o2 = f2.y.data[by : by + BS, bx : bx + BS]

            # I: predict with DC(128)
            i_sad += sad(o0, np.full((BS, BS), 128, dtype=np.int16))

            # P: frame2 from frame0
            _, _, _, sp = motion_search(o2, f0.y, bx, by)
            p_sad += sp
            if sp == 0:
                p_zero += 1

            # B: frame1 from frame0 + frame2, bi-pred
            bp0, _, _, _ = motion_search(o1, f0.y, bx, by)
            bp1, _, _, _ = motion_search(o1, f2.y, bx, by)
            bi = ((bp0.astype(np.int32) + bp1.astype(np.int32) + 1) >> 1).astype(np.int16)
            sb = sad(o1, bi)
            b_sad += sb
            if sb == 0:
                b_zero += 1

    print(f"  {'Type':<16} {'Total SAD':>10} {'Avg/block':>10} {'Zero blocks':>13}")
    print(f"  {'─' * 51}")
    print(f"  {'I (DC=128)':<16} {i_sad:>10,} {i_sad / n_blocks:>10.1f} {'0/' + str(n_blocks):>13}")
    print(f"  {'P (1 ref)':<16} {p_sad:>10,} {p_sad / n_blocks:>10.1f} {str(p_zero) + '/' + str(n_blocks):>13}")
    print(f"  {'B (bi-pred)':<16} {b_sad:>10,} {b_sad / n_blocks:>10.1f} {str(b_zero) + '/' + str(n_blocks):>13}")

    print(f"""
  I >> P > B confirms the fundamental ordering:
    - I predicts from spatial neighbors only  -> worst match
    - P reuses one temporal reference          -> much better
    - B averages two temporal references       -> best match
  Background blocks get SAD=0 in P/B (exact match via MV=0,0).
""")

    # ── step 2: coded bits ───────────────────────────────────────────
    sect("STEP 2: CODED BITS (DCT + quantize at QP={})".format(QP))
    print("""
  All frame types use the same bit counting (estimate_bits on
  quantized coefficients) so the numbers are directly comparable.
  P/B use original frames as references (ideal, no drift).
""")

    recon_I, bits_I = encode_i(f0, QP)
    _, bits_P, mvs_P = encode_p(f2, f0.y, QP)
    _, bits_B, mvs_B = encode_b(f1, f0.y, f2.y, QP)

    psnr_I = psnr(f0.y.data, recon_I.data)

    print(f"  {'Type':<8} {'Bits':>8} {'vs I':>8}")
    print(f"  {'─' * 26}")
    print(f"  {'I':<8} {bits_I:>8,} {'1.00x':>8}")
    print(f"  {'P':<8} {bits_P:>8,} {f'{bits_P / max(bits_I, 1):.2f}x':>8}")
    print(f"  {'B':<8} {bits_B:>8,} {f'{bits_B / max(bits_I, 1):.2f}x':>8}")

    # ── motion vectors ───────────────────────────────────────────────
    sect("P-FRAME MOTION VECTORS")
    obj_dx_total = DX * 2  # frame 0 -> frame 2
    nonzero = [(bx, by, mx, my) for bx, by, mx, my in mvs_P if mx != 0 or my != 0]
    zero_ct = len(mvs_P) - len(nonzero)
    print(f"""
  Rectangle moved {obj_dx_total}px right (frame 0 -> frame 2).
  Zero-MV blocks: {zero_ct}/{len(mvs_P)} (static background, exact match)
  Non-zero MV:    {len(nonzero)}/{len(mvs_P)} (rectangle + uncovered area)
""")
    for bx, by, mx, my in nonzero[:6]:
        print(f"    ({bx:2d},{by:2d}): MV = ({mx:+3d}, {my:+3d})")
    if len(nonzero) > 6:
        print(f"    ... ({len(nonzero) - 6} more)")

    # ── B-frame modes ────────────────────────────────────────────────
    sect("B-FRAME PREDICTION MODES")
    bi_ct = sum(1 for *_, m in mvs_B if m == "bi")
    l0_ct = sum(1 for *_, m in mvs_B if m == "L0")
    l1_ct = sum(1 for *_, m in mvs_B if m == "L1")
    print(f"""
  bi-pred={bi_ct}  L0-only={l0_ct}  L1-only={l1_ct}

  Bi-pred dominates: averaging two references cancels noise,
  giving smaller residuals than either direction alone.
""")

    # ── connection to birds.mp4 ──────────────────────────────────────
    sect("REAL VIDEO: birds.mp4 (from analysis.md)")
    print("""
  Frame Type | Count | Avg Size | Ratio
  -----------+-------+----------+------
  I          |     5 |   79 KB  | 3.2x
  P          |   192 |   43 KB  | 1.7x
  B          |   568 |   25 KB  | 1.0x

  Same ordering: I > P > B. The ratio measures how much
  can be borrowed from other frames via motion compensation.
""")

    # ── pipeline diagram ─────────────────────────────────────────────
    sect("THE ENCODING PIPELINE")
    print("""
  Same pipeline for I, P, and B. Only the prediction source differs.

  original block
       |
       v
  [ Predict ]   I: DC/Planar/Angular from neighbors
       |         P: motion_search(ref_L0)
       |         B: (search(L0) + search(L1) + 1) >> 1
       v
  residual = original - prediction
       |
       v
  [ DCT ] --> [ Quantize ] --+--> bitstream
                              |
                    [ Dequant ] --> [ IDCT ]
                              |
                              v
                    reconstructed = clip(pred + recon_residual)
                              |
                              +--> reference for future P/B frames

  Decode order != display order. B needs future references:
    Display:  I0  B1  B2  B3  P4       Decode:  I0  P4  B1  B2  B3
                                                      ^
                                        P4 decoded first so B1-3 can use it
""")


if __name__ == "__main__":
    main()
