#!/usr/bin/env python3
"""
nano_mp4: H.264 Encoding Pipeline From First Principles
=========================================================

Teaches the core H.264 encoding loop by implementing it. No container,
no NAL framing, no bitstream assembly — just the signal processing.

Five lessons, each building on the previous:
  1. Butterfly DCT     — the multiply-free integer transform
  2. Quantization      — position-dependent MF (H.264's key trick)
  3. Zigzag scan       — 2D coefficients to 1D for entropy coding
  4. Entropy coding    — CAVLC, lossless compression of quantized levels
  5. Intra prediction  — guess pixels from neighbors before all of the above

Run:
  python examples/nano_mp4.py                    # gradient test pattern
  python examples/nano_mp4.py photo.png -qp 22   # real image
  python examples/nano_mp4.py --trace             # show every intermediate value
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# =====================================================================
#  LESSON 1: The Butterfly DCT
# =====================================================================
#
#  Standard DCT needs 16 multiplications per 4-element row.
#  H.264 replaces it with a "butterfly" — just additions and shifts.
#
#  The trick: factor the DCT matrix into:
#
#       | 1   1   1   1 |               | a^2  ac  a^2  ac |
#  Cf = | 2   1  -1  -2 |  and    Ef =  | ac   c^2 ac   c^2|
#       | 1  -1  -1   1 |               | a^2  ac  a^2  ac |
#       | 1  -2   2  -1 |               | ac   c^2 ac   c^2|
#
#  Cf uses only {1, 2, -1, -2}. The x2 is a bit shift, not a multiply.
#  Ef compensates for the unequal row norms, but gets folded into
#  quantization (Lesson 2), so the transform itself is multiply-free.
#
#  The real DCT basis ratio is cos(pi/8)/cos(3pi/8) = 2.414.
#  H.264 approximates this as 2. The ~1.3% error is absorbed by Ef.
# =====================================================================

def h264_forward_1d(x: list[int]) -> list[int]:
    """
    One row/column of the forward butterfly.

    Input:  4 spatial values     [x0, x1, x2, x3]
    Output: 4 frequency values   [y0, y1, y2, y3]

    Butterfly structure:

        x0 ──+──(+)── s0 ──+──(+)──── y0  (DC, average)
        x3 ──┘              └──(-)──── y2  (frequency 2)

        x1 ──+──(+)── s1 ──── 2*s3+s2 ── y1  (frequency 1)
        x2 ──┘
        x1 ──+──(-)── s2 ──── s3-2*s2 ── y3  (frequency 3)
        x2 ──┘
        x0 ──+──(-)── s3
        x3 ──┘

    Cost: 8 additions + 2 left-shifts. Zero multiplications.
    """
    s0 = x[0] + x[3]
    s1 = x[1] + x[2]
    s2 = x[1] - x[2]
    s3 = x[0] - x[3]
    return [
        s0 + s1,           # DC: sum of all four
        (s3 << 1) + s2,    # low freq: <<1 instead of *2
        s0 - s1,           # mid freq
        s3 - (s2 << 1),    # high freq: note the sign flip vs y1
    ]


def h264_inverse_1d(y: list[int]) -> list[int]:
    """
    Inverse butterfly (ITU-T H.264 spec section 8.5.12.1).

    The inverse is NOT just the transpose — it uses >>1 instead of <<1,
    placed asymmetrically to maintain integer precision.
    """
    e0 = y[0] + y[2]
    e1 = y[0] - y[2]
    e2 = (y[1] >> 1) - y[3]
    e3 = y[1] + (y[3] >> 1)
    return [e0 + e3, e1 + e2, e1 - e2, e0 - e3]


def h264_forward_4x4(block: np.ndarray) -> np.ndarray:
    """
    Full 2D forward transform: W = Cf @ block @ Cf^T

    Separable: apply 1D to all rows, then all columns.
    Output W is unscaled — Ef normalization happens in quantization.
    """
    x = block.astype(np.int32)
    temp = np.zeros((4, 4), dtype=np.int32)
    out = np.zeros((4, 4), dtype=np.int32)

    for i in range(4):
        r = h264_forward_1d([int(x[i, j]) for j in range(4)])
        for j in range(4):
            temp[i, j] = r[j]

    for j in range(4):
        r = h264_forward_1d([int(temp[i, j]) for i in range(4)])
        for i in range(4):
            out[i, j] = r[i]

    return out


def h264_inverse_4x4(coeff: np.ndarray) -> np.ndarray:
    """
    Full 2D inverse transform (decoder side, normative).

    The >>6 at the end is the total accumulated scaling from
    both 1D passes plus the Ef factors embedded during dequantization.
    """
    c = coeff.astype(np.int32)
    temp = np.zeros((4, 4), dtype=np.int32)
    out = np.zeros((4, 4), dtype=np.int32)

    for j in range(4):
        r = h264_inverse_1d([int(c[i, j]) for i in range(4)])
        for i in range(4):
            temp[i, j] = r[i]

    for i in range(4):
        r = h264_inverse_1d([int(temp[i, j]) for j in range(4)])
        for j in range(4):
            out[i, j] = (r[j] + 32) >> 6

    return out


# =====================================================================
#  LESSON 2: Position-Dependent Quantization
# =====================================================================
#
#  This is H.264's cleverest design decision.
#
#  Problem: the butterfly produces unscaled output. Position (0,0)
#  and position (1,1) have different normalization factors because
#  Cf's rows have different norms: ||[1,1,1,1]|| != ||[2,1,-1,-2]||.
#
#  H.264's solution: do nothing in the transform (keep it free),
#  then fix the scaling in quantization by using DIFFERENT
#  multiplication factors for different positions.
#
#  Position classification (2D, based on which Cf rows contributed):
#    (even, even) -> both rows had norm ||[1,1,1,1]||     -> type 0
#    (odd,  odd)  -> both rows had norm ||[2,1,-1,-2]||   -> type 1
#    (mixed)      -> one of each                          -> type 2
#
#  Each type gets its own MF value in the quantization table.
#  MF = (quantization_step * Ef_correction) combined into one number.
#
#  QP (Quantization Parameter) controls quality:
#    QP 0  = nearly lossless
#    QP 51 = maximum compression, heavy artifacts
#    QP +6 = roughly doubles the quantization step (halves quality)
#
#  The +6 periodicity is why tables are indexed by qp % 6.
#  qp // 6 controls additional bit-shift scaling.
# =====================================================================

# forward quantization multiplication factors (ITU-T H.264 Table 8-14)
# indexed by qp_rem = qp % 6, three values per entry (type0, type1, type2)
MF_TABLE = [
    (13107, 5243, 8066),
    (11916, 4660, 7490),
    (10082, 4194, 6554),
    ( 9362, 3647, 5825),
    ( 8192, 3355, 5243),
    ( 7282, 2893, 4559),
]

# inverse quantization scale factors (ITU-T H.264 Table 8-15)
V_TABLE = [
    (10, 16, 13),
    (11, 18, 14),
    (13, 20, 16),
    (14, 23, 18),
    (16, 25, 20),
    (18, 29, 23),
]


def _position_type(i: int, j: int) -> int:
    ei, ej = i % 2 == 0, j % 2 == 0
    if ei and ej:
        return 0
    if not ei and not ej:
        return 1
    return 2


def h264_quantize_4x4(W: np.ndarray, qp: int, is_intra: bool = True) -> np.ndarray:
    """
    Forward quantization with position-dependent MF.

      level[i][j] = sign(W) * ( |W[i][j]| * MF[type] + offset ) >> qbits

    MF absorbs BOTH the quantization step AND the Ef normalization.
    The offset (f) creates a "dead zone" around zero — coefficients
    smaller than the dead zone become 0. This is where compression happens.
    """
    qp_per = qp // 6
    qp_rem = qp % 6
    mf = MF_TABLE[qp_rem]
    qbits = 15 + qp_per
    f = (1 << qbits) // 3 if is_intra else (1 << qbits) // 6

    level = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            t = _position_type(i, j)
            sign = 1 if W[i, j] >= 0 else -1
            level[i, j] = sign * ((abs(int(W[i, j])) * mf[t] + f) >> qbits)
    return level


def h264_dequantize_4x4(level: np.ndarray, qp: int) -> np.ndarray:
    """
    Inverse quantization. Also position-dependent (uses V_TABLE).

      coeff[i][j] = level[i][j] * V[type] << (qp // 6)
    """
    qp_per = qp // 6
    qp_rem = qp % 6
    v = V_TABLE[qp_rem]

    coeff = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            t = _position_type(i, j)
            coeff[i, j] = int(level[i, j]) * v[t] << qp_per
    return coeff


# =====================================================================
#  LESSON 3: Zigzag Scan
# =====================================================================
#
#  After quantization, non-zero coefficients cluster in the top-left
#  (low frequencies). Zigzag reads the 2D block into 1D along diagonals:
#
#     0 -> 1   5 -> 6            This groups non-zeros at the start
#     |  / |  / |  /             and zeros at the end — long zero runs
#     2   4   7  12              compress efficiently in entropy coding.
#     |  / |  / |  /
#     3   8  11  13
#        / |  /
#      9  10  14 -> 15
# =====================================================================

ZIGZAG_4x4 = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (2, 3), (3, 2), (3, 3),
]


def zigzag_scan(block: np.ndarray) -> list[int]:
    return [int(block[r, c]) for r, c in ZIGZAG_4x4]


# =====================================================================
#  LESSON 4: CAVLC Entropy Coding
# =====================================================================
#
#  After zigzag, we have something like:
#    [6, 3, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#  CAVLC compresses this by exploiting two patterns:
#    - Most non-zero values at the tail are +-1 (cheap: 1 sign bit each)
#    - Smaller values get shorter Exp-Golomb codes
#
#  Structure:
#    1. total_coeffs + trailing_ones count
#    2. trailing one signs (1 bit each)
#    3. remaining levels (Exp-Golomb coded)
#    4. zero run positions
#
#  This is simplified — real H.264 uses ~40 VLC tables for max compression.
# =====================================================================

class BitstreamWriter:
    """Bit-level output buffer. MSB first."""

    def __init__(self):
        self._buf: list[int] = []
        self._byte = 0
        self._bits = 0

    def write(self, val: int, n: int):
        for i in range(n - 1, -1, -1):
            self._byte = (self._byte << 1) | ((val >> i) & 1)
            self._bits += 1
            if self._bits == 8:
                self._buf.append(self._byte)
                self._byte = 0
                self._bits = 0

    def write_ue(self, val: int):
        """
        Exp-Golomb unsigned: val -> 0...01xxx
        0 -> 1,  1 -> 010,  2 -> 011,  3 -> 00100,  7 -> 0001000
        Smaller values = shorter codes. This is why quantization matters.
        """
        vp1 = val + 1
        n = vp1.bit_length()
        self.write(0, n - 1)
        self.write(vp1, n)

    def write_se(self, val: int):
        """Exp-Golomb signed: 0->0, +1->1, -1->2, +2->3, -2->4, ..."""
        self.write_ue(-2 * val if val <= 0 else 2 * val - 1)

    @property
    def bit_count(self) -> int:
        return len(self._buf) * 8 + self._bits


def cavlc_encode_block(levels: list[int]) -> tuple[int, int]:
    """
    Simplified CAVLC for one 4x4 block.
    Returns (total_bits, total_nonzero_coeffs).
    """
    rev = list(reversed(levels))
    total_coeffs = sum(1 for v in rev if v != 0)

    bs = BitstreamWriter()

    if total_coeffs == 0:
        bs.write_ue(0)
        return bs.bit_count, 0

    nonzero = [v for v in rev if v != 0]

    # count trailing +-1 (max 3)
    trailing_ones = 0
    for v in nonzero:
        if abs(v) == 1 and trailing_ones < 3:
            trailing_ones += 1
        else:
            break

    # encode
    bs.write_ue(total_coeffs)
    bs.write(trailing_ones, 2)

    for k in range(trailing_ones):
        bs.write(1 if nonzero[k] < 0 else 0, 1)

    for k in range(trailing_ones, total_coeffs):
        bs.write_se(nonzero[k])

    total_zeros = len(levels) - total_coeffs
    if total_coeffs < 16:
        bs.write_ue(total_zeros)

    zeros_left = total_zeros
    nz_positions = [i for i, v in enumerate(rev) if v != 0]
    for k in range(total_coeffs - 1):
        if zeros_left == 0:
            break
        run = nz_positions[k] if k == 0 else nz_positions[k] - nz_positions[k - 1] - 1
        bs.write_ue(run)
        zeros_left -= run

    return bs.bit_count, total_coeffs


# =====================================================================
#  LESSON 5: Intra Prediction
# =====================================================================
#
#  Before computing the residual, predict the current block from
#  already-reconstructed neighbors. Better prediction = smaller residual
#  = fewer bits after transform + quantization.
#
#  H.264 defines 9 modes for 4x4 blocks:
#    0: vertical     (copy top row down)
#    1: horizontal   (copy left column right)
#    2: DC           (fill with average of top + left)
#    3-8: diagonal   (extrapolate along various angles)
#
#  We implement only DC — simplest and often best for smooth regions.
# =====================================================================

def dc_predict_4x4(top: np.ndarray, left: np.ndarray) -> np.ndarray:
    dc = (int(np.sum(top[:4])) + int(np.sum(left[:4])) + 4) >> 3
    return np.full((4, 4), dc, dtype=np.int16)


# =====================================================================
#  The Encoder Loop
# =====================================================================

def encode_frame(
    luma: np.ndarray, qp: int, trace: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Encode one frame. Returns (reconstructed_luma, stats).

    The pipeline for each 4x4 block:

      original pixels
           |
      [1] predict (DC: average of top + left neighbors)
           |
      [2] residual = original - prediction
           |
      [3] butterfly DCT (zero multiplications)
           |
      [4] quantize (position-dependent MF, absorbs Ef scaling)
           |
      [5] zigzag scan -> CAVLC -> count bits
           |
      [6] dequantize -> inverse butterfly -> add prediction -> clip
           |
      [7] store reconstruction (used to predict the next block)
    """
    h, w = luma.shape
    recon = np.full((h, w), 128, dtype=np.int16)
    total_bits = 0
    total_coeffs = 0
    zero_blocks = 0
    traced = False

    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            orig = luma[by:by+4, bx:bx+4].astype(np.int16)

            # [1] prediction from reconstructed neighbors
            top = recon[by-1, bx:bx+4] if by > 0 else np.full(4, 128, dtype=np.int16)
            left = recon[by:by+4, bx-1] if bx > 0 else np.full(4, 128, dtype=np.int16)
            pred = dc_predict_4x4(top, left)

            # [2] residual
            residual = (orig - pred).astype(np.int32)

            # [3] forward butterfly DCT
            W = h264_forward_4x4(residual)

            # [4] quantize (position-dependent MF absorbs Ef)
            level = h264_quantize_4x4(W, qp)

            # [5] entropy coding
            scanned = zigzag_scan(level)
            bits, nz = cavlc_encode_block(scanned)
            total_bits += bits
            total_coeffs += nz
            if nz == 0:
                zero_blocks += 1

            # [6] reconstruction (decoder mirror)
            dq = h264_dequantize_4x4(level, qp)
            rec_res = h264_inverse_4x4(dq)
            blk = np.clip(pred + rec_res, 0, 255).astype(np.int16)

            # [7] store for future prediction
            recon[by:by+4, bx:bx+4] = blk

            if trace and not traced:
                traced = True
                _trace_block(orig, pred, residual, W, level, scanned, dq, rec_res, blk, qp)

    n_blocks = (h // 4) * (w // 4)
    stats = {
        "bits": total_bits,
        "bytes": (total_bits + 7) // 8,
        "avg_coeffs": total_coeffs / max(n_blocks, 1),
        "zero_blocks": zero_blocks,
        "total_blocks": n_blocks,
        "bpp": total_bits / (h * w),
    }
    return recon.astype(np.uint8), stats


def _trace_block(orig, pred, residual, W, level, scanned, dq, rec_res, recon, qp):
    """Print every intermediate value for one block."""
    def mat(name, m):
        print(f"  {name}:")
        for row in m:
            print("    " + "  ".join(f"{int(v):>5}" for v in row))

    print("\n=== TRACE: first 4x4 block ===")
    print(f"  qp = {qp}  (qp_per={qp//6}, qp_rem={qp%6})")
    mat("[1] original pixels", orig)
    mat("[2] prediction (DC)", pred)
    mat("[3] residual = orig - pred", residual)
    mat("[4] butterfly DCT output (W)", W)
    print()
    print("  W is unscaled. position (0,0) and (1,1) have different")
    print("  normalization because Cf rows have different norms.")
    print("  the MF table in quantization corrects this.")
    mat("[5] quantized levels", level)
    nz = sum(1 for v in scanned if v != 0)
    print(f"\n  [6] zigzag: {scanned}")
    print(f"      {nz}/16 non-zero coefficients survive quantization")
    mat("[7] dequantized", dq)
    mat("[8] inverse butterfly -> residual'", rec_res)
    mat("[9] reconstructed = clip(pred + residual')", recon)
    error = np.abs(recon.astype(np.int32) - orig.astype(np.int32))
    print(f"\n  reconstruction error: max={int(np.max(error))}, mean={float(np.mean(error)):.2f}")
    print("=" * 50 + "\n")


# ── image I/O ────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("L")
    y = np.array(img, dtype=np.uint8)
    h, w = y.shape
    # pad to 4x4 aligned (not 16x16 — no macroblocks needed without NAL)
    ph, pw = ((h + 3) // 4) * 4, ((w + 3) // 4) * 4
    if ph != h or pw != w:
        padded = np.full((ph, pw), 128, dtype=np.uint8)
        padded[:h, :w] = y
        y = padded
    return y


def make_gradient(width: int = 128, height: int = 128) -> np.ndarray:
    y = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            y[i, j] = int((i * 200 / height + j * 55 / width) % 256)
    return y


def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((orig.astype(float) - recon.astype(float)) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="nano_mp4: H.264 encoding pipeline")
    parser.add_argument("input", nargs="?", help="input image (PNG/JPEG)")
    parser.add_argument("-qp", type=int, default=26)
    parser.add_argument("--trace", action="store_true",
                        help="print intermediate values for the first 4x4 block")
    parser.add_argument("--sweep", action="store_true",
                        help="encode at multiple QP values to show the quality-bitrate tradeoff")
    args = parser.parse_args()

    if args.input:
        luma = load_image(args.input)
        src = args.input
    else:
        luma = make_gradient()
        src = "gradient"

    h, w = luma.shape
    print(f"nano_mp4: {src} ({w}x{h})\n")

    if args.sweep:
        print(f"  {'QP':>4}  {'bits':>8}  {'bpp':>6}  {'PSNR':>7}  {'zero%':>6}  coeffs/blk")
        print(f"  {'─'*55}")
        for qp in [10, 16, 22, 26, 30, 36, 42, 48]:
            recon, stats = encode_frame(luma, qp)
            p = psnr(luma, recon)
            zp = stats["zero_blocks"] / stats["total_blocks"] * 100
            print(f"  {qp:>4}  {stats['bits']:>8,}  {stats['bpp']:>6.2f}  "
                  f"{p:>6.1f}  {zp:>5.1f}%  {stats['avg_coeffs']:.1f}")
    else:
        recon, stats = encode_frame(luma, args.qp, trace=args.trace)
        p = psnr(luma, recon)
        zp = stats["zero_blocks"] / stats["total_blocks"] * 100
        print(f"  qp={args.qp}  {stats['bits']:,} bits ({stats['bpp']:.2f} bpp)  "
              f"psnr={p:.2f} dB  "
              f"zero={zp:.0f}%  coeffs/blk={stats['avg_coeffs']:.1f}")


if __name__ == "__main__":
    main()
