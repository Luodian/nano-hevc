#!/usr/bin/env python3
"""
Frame Representation Visualizer
================================

Extracts real frames from birds.mp4 and visualizes what the codec
actually stores for I / P / B frames:

  I-frame: dense DCT coefficients (full spatial info)
  P-frame: motion vectors + sparse residual
  B-frame: bidirectional MVs + sparser residual

Output: examples/assets/frame_repr_viz.png
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

VIDEO = Path(__file__).parent / "assets" / "birds.mp4"
OUT = Path(__file__).parent / "assets" / "frame_repr_viz.png"

# downscale for visualization (original is 3840x1620)
SCALE_W, SCALE_H = 480, 202
BLOCK = 16


# ── palette ──────────────────────────────────────────────────────────

C_BG      = "#fafafa"
C_TEXT    = "#1a1a2e"
C_DIM     = "#6b7280"
C_BORDER  = "#d1d5db"

C_I       = "#e05a00"   # orange (darker for light bg)
C_P       = "#0284c7"   # blue
C_B       = "#7c3aed"   # purple

C_SKIP    = "#22c55e"   # green
C_MV      = "#0ea5e9"   # sky blue
C_MVR     = "#ef4444"   # red


# ── extraction ───────────────────────────────────────────────────────

def extract_rgb(video: Path, idx: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(video),
           "-vf", f"select=eq(n\\,{idx}),scale={SCALE_W}:{SCALE_H}",
           "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout,
                         dtype=np.uint8).reshape(SCALE_H, SCALE_W, 3)

def extract_y(video: Path, idx: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(video),
           "-vf", f"select=eq(n\\,{idx}),scale={SCALE_W}:{SCALE_H}",
           "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout,
                         dtype=np.uint8).reshape(SCALE_H, SCALE_W)


# ── analysis ─────────────────────────────────────────────────────────

def motion_search(cur: np.ndarray, ref: np.ndarray, bs: int,
                  sr: int = 16):
    h, w = cur.shape
    R, C = h // bs, w // bs
    mvx = np.zeros((R, C), np.int32)
    mvy = np.zeros((R, C), np.int32)
    sad = np.full((R, C), 1 << 30, np.float64)
    for r in range(R):
        for c in range(C):
            by, bx = r * bs, c * bs
            blk = cur[by:by+bs, bx:bx+bs].astype(np.float64)
            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):
                    ry, rx = by + dy, bx + dx
                    if ry < 0 or rx < 0 or ry + bs > h or rx + bs > w:
                        continue
                    s = np.abs(blk - ref[ry:ry+bs, rx:rx+bs].astype(np.float64)).sum()
                    if s < sad[r, c]:
                        sad[r, c] = s
                        mvx[r, c], mvy[r, c] = dx, dy
    return mvx, mvy, sad

def residual_from_mv(cur, ref, mvx, mvy, bs):
    h, w = cur.shape
    R, C = h // bs, w // bs
    res = np.zeros((h, w), np.float64)
    for r in range(R):
        for c in range(C):
            by, bx = r * bs, c * bs
            dy, dx = int(mvy[r, c]), int(mvx[r, c])
            pred = ref[by+dy:by+dy+bs, bx+dx:bx+dx+bs].astype(np.float64)
            res[by:by+bs, bx:bx+bs] = cur[by:by+bs, bx:bx+bs].astype(np.float64) - pred
    return res


# ── drawing helpers ──────────────────────────────────────────────────

def add_block_grid(ax, w, h, bs, color, alpha=0.15):
    for x in range(0, w + 1, bs):
        ax.axvline(x, color=color, alpha=alpha, lw=0.4)
    for y in range(0, h + 1, bs):
        ax.axhline(y, color=color, alpha=alpha, lw=0.4)

def style_ax(ax, title, color, fs=11):
    ax.set_title(title, color=color, fontsize=fs, fontweight="bold",
                 pad=6, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(0.5)

def text_panel(ax, lines, bg=C_BG):
    """Render monospaced text lines into an axes."""
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(lines)
    for i, (txt, color, fs, fw) in enumerate(lines):
        y = 1.0 - (i + 0.8) / (n + 0.6)
        ax.text(0.04, y, txt, fontsize=fs, fontweight=fw,
                color=color, fontfamily="monospace", va="center")


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("extracting frames ...")
    f0_rgb, f1_rgb, f4_rgb = extract_rgb(VIDEO, 0), extract_rgb(VIDEO, 1), extract_rgb(VIDEO, 4)
    f0_y, f1_y, f4_y = extract_y(VIDEO, 0), extract_y(VIDEO, 1), extract_y(VIDEO, 4)

    R, C = SCALE_H // BLOCK, SCALE_W // BLOCK

    print("motion search (P) ...")
    p_mvx, p_mvy, p_sad = motion_search(f4_y, f0_y, BLOCK)
    p_res = residual_from_mv(f4_y, f0_y, p_mvx, p_mvy, BLOCK)

    print("motion search (B) ...")
    b_mvx0, b_mvy0, _ = motion_search(f1_y, f0_y, BLOCK)
    b_mvx1, b_mvy1, _ = motion_search(f1_y, f4_y, BLOCK)
    # bi-pred residual
    b_res = np.zeros((SCALE_H, SCALE_W), np.float64)
    for r in range(R):
        for c in range(C):
            by, bx = r * BLOCK, c * BLOCK
            p0 = f0_y[by+int(b_mvy0[r,c]):by+int(b_mvy0[r,c])+BLOCK,
                       bx+int(b_mvx0[r,c]):bx+int(b_mvx0[r,c])+BLOCK].astype(np.float64)
            p1 = f4_y[by+int(b_mvy1[r,c]):by+int(b_mvy1[r,c])+BLOCK,
                       bx+int(b_mvx1[r,c]):bx+int(b_mvx1[r,c])+BLOCK].astype(np.float64)
            b_res[by:by+BLOCK, bx:bx+BLOCK] = (
                f1_y[by:by+BLOCK, bx:bx+BLOCK].astype(np.float64) - (p0 + p1 + 1) / 2
            )

    # block classification for P-frame
    p_class = np.zeros((R, C), np.int32)  # 0=skip 1=mv 2=mv+res
    for r in range(R):
        for c in range(C):
            has_mv = (p_mvx[r,c] != 0) or (p_mvy[r,c] != 0)
            has_res = p_sad[r,c] >= 80
            if not has_mv and not has_res:
                p_class[r, c] = 0
            elif not has_res:
                p_class[r, c] = 1
            else:
                p_class[r, c] = 2

    # ── layout ───────────────────────────────────────────────────────
    print("rendering ...")

    fig = plt.figure(figsize=(24, 18), facecolor=C_BG)
    fig.suptitle("Codec Frame Representation: What Gets Stored",
                 fontsize=18, fontweight="bold", color=C_TEXT, y=0.975)
    fig.text(0.5, 0.955, "birds.mp4  ·  3840×1620  ·  H.264 High@5.1  ·  25fps  ·  5.96 Mbps",
             ha="center", fontsize=10, color=C_DIM)

    # 4 rows × 4 cols, with row 3 being a shorter summary
    gs = gridspec.GridSpec(4, 4, figure=fig,
                           height_ratios=[1, 1, 1, 0.55],
                           width_ratios=[1, 1, 1, 1.1],
                           hspace=0.32, wspace=0.18,
                           left=0.04, right=0.97, top=0.93, bottom=0.03)

    # ── ROW 0: I-frame ──────────────────────────────────────────────

    # original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(f0_rgb)
    add_block_grid(ax, SCALE_W, SCALE_H, BLOCK, C_I, 0.2)
    style_ax(ax, "I-Frame  idx=0", C_I)
    ax.text(0.5, -0.08, "69,801 bytes  (2.85 B/block)", transform=ax.transAxes,
            ha="center", fontsize=9, color=C_I, fontweight="bold")

    # energy map
    ax = fig.add_subplot(gs[0, 1])
    energy = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            blk = f0_y[r*BLOCK:(r+1)*BLOCK, c*BLOCK:(c+1)*BLOCK].astype(np.float64)
            energy[r, c] = blk.var()
    im = ax.imshow(energy, cmap="YlOrRd", interpolation="nearest", aspect="auto")
    style_ax(ax, "Per-block energy (variance)", C_I)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors=C_DIM)
    ax.text(0.5, -0.08, "high everywhere — no block is free",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_DIM)

    # block type: all blocks are "coded" for I-frame (uniform orange)
    ax = fig.add_subplot(gs[0, 2])
    i_coded = np.ones((R, C))  # all blocks coded
    cmap_i = ListedColormap([C_I])
    ax.imshow(i_coded, cmap=cmap_i, interpolation="nearest", aspect="auto",
              vmin=0.5, vmax=1.5)
    # grid lines
    for x in range(C + 1):
        ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(R + 1):
        ax.axhline(y - 0.5, color="white", lw=0.5)
    style_ax(ax, "Block map: ALL blocks carry data", C_I)
    ax.text(0.5, -0.08, f"{R}×{C} = {R*C} blocks, every one has DCT coefficients",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_DIM)

    # annotation
    ax = fig.add_subplot(gs[0, 3])
    text_panel(ax, [
        ("I-FRAME: DENSE SPATIAL ENCODING",             C_I,    11, "bold"),
        ("",                                             C_DIM,   6, "normal"),
        ("Each block stores independently:",             C_TEXT,  9.5, "normal"),
        ("  ┌─────────────────────────────────┐",        C_DIM,   9, "normal"),
        ("  │ intra_pred_mode (DC/Planar/Ang) │",        C_TEXT,  9, "normal"),
        ("  │ cbp  (coded block pattern)      │",        C_TEXT,  9, "normal"),
        ("  │ qp_delta                        │",        C_TEXT,  9, "normal"),
        ("  │ DCT coefficients   ← 90%+ cost  │",       C_I,     9, "bold"),
        ("  └─────────────────────────────────┘",        C_DIM,   9, "normal"),
        ("",                                             C_DIM,   6, "normal"),
        ("transmitted = Σ [mode + DCT(orig - spatial)]", C_I,    9.5, "bold"),
        ("",                                             C_DIM,   6, "normal"),
        ("No temporal reference — self-contained.",      C_DIM,   9, "normal"),
        ("Even flat sky costs bits (spatial pred ≠ 0).", C_DIM,   9, "normal"),
    ])

    # ── ROW 1: P-frame ──────────────────────────────────────────────

    # original
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(f4_rgb)
    add_block_grid(ax, SCALE_W, SCALE_H, BLOCK, C_P, 0.12)
    style_ax(ax, "P-Frame  idx=4", C_P)
    ax.text(0.5, -0.08, "30,314 bytes  (1.24 B/block)", transform=ax.transAxes,
            ha="center", fontsize=9, color=C_P, fontweight="bold")

    # MV field
    ax = fig.add_subplot(gs[1, 1])
    mv_mag = np.sqrt(p_mvx.astype(np.float64)**2 + p_mvy.astype(np.float64)**2)
    ax.imshow(f4_y, cmap="gray", alpha=0.25, extent=[-0.5, C-0.5, R-0.5, -0.5])
    ax.imshow(mv_mag, cmap="Blues", alpha=0.6, interpolation="nearest",
              aspect="auto", extent=[-0.5, C-0.5, R-0.5, -0.5])
    for r in range(R):
        for c in range(C):
            if abs(p_mvx[r,c]) > 0 or abs(p_mvy[r,c]) > 0:
                scale = 0.35
                ax.annotate("", xy=(c + p_mvx[r,c]/BLOCK*scale, r + p_mvy[r,c]/BLOCK*scale),
                           xytext=(c, r),
                           arrowprops=dict(arrowstyle="-|>", color=C_P, lw=1.0,
                                          mutation_scale=7))
    style_ax(ax, "Motion vector field", C_P)
    skip_n = int(np.sum(p_class == 0))
    mv_n = int(np.sum(p_class == 1))
    mvr_n = int(np.sum(p_class == 2))
    ax.text(0.5, -0.08, f"skip={skip_n}  mv-only={mv_n}  mv+res={mvr_n}",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_DIM)

    # block classification map
    ax = fig.add_subplot(gs[1, 2])
    cmap_p = ListedColormap([C_SKIP, C_MV, C_MVR])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap_p.N)
    ax.imshow(p_class, cmap=cmap_p, norm=norm, interpolation="nearest", aspect="auto")
    for x in range(C + 1):
        ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(R + 1):
        ax.axhline(y - 0.5, color="white", lw=0.5)
    style_ax(ax, "Block type map", C_P)
    # legend
    for i, (label, clr) in enumerate([("skip", C_SKIP), ("mv only", C_MV), ("mv+residual", C_MVR)]):
        ax.text(0.02 + i * 0.35, -0.08, f"■ {label}", transform=ax.transAxes,
                fontsize=8.5, color=clr, fontweight="bold")

    # annotation
    ax = fig.add_subplot(gs[1, 3])
    text_panel(ax, [
        ("P-FRAME: SPARSE DELTA ENCODING",                C_P,     11, "bold"),
        ("",                                               C_DIM,    6, "normal"),
        ("Block modes:",                                   C_TEXT,  9.5, "normal"),
        (f"  ■ skip  ({skip_n} blocks) → 1 bit each",     C_SKIP,   9, "bold"),
        ("    copy from ref, zero cost",                   C_DIM,    8, "normal"),
        (f"  ■ mv    ({mv_n} blocks) → MV only",          C_MV,     9, "bold"),
        ("    shift from ref, residual ≈ 0",               C_DIM,    8, "normal"),
        (f"  ■ mv+r  ({mvr_n} blocks) → MV + DCT coeff",  C_MVR,    9, "bold"),
        ("    shift + compensate difference",              C_DIM,    8, "normal"),
        ("",                                               C_DIM,    6, "normal"),
        ("transmitted = Σ [MV + DCT(orig - warp(ref,MV))]",C_P,    9.5, "bold"),
        ("",                                               C_DIM,    6, "normal"),
        ("43% of I-frame. All bytes on bird region.",      C_DIM,    9, "normal"),
        ("Sky = skip (identical to ref) → free.",          C_DIM,    9, "normal"),
    ])

    # ── ROW 2: B-frame ──────────────────────────────────────────────

    # original
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(f1_rgb)
    add_block_grid(ax, SCALE_W, SCALE_H, BLOCK, C_B, 0.12)
    style_ax(ax, "B-Frame  idx=1", C_B)
    ax.text(0.5, -0.08, "12,934 bytes  (0.53 B/block)", transform=ax.transAxes,
            ha="center", fontsize=9, color=C_B, fontweight="bold")

    # dual MV field
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(f1_y, cmap="gray", alpha=0.25, extent=[-0.5, C-0.5, R-0.5, -0.5])
    C_L0, C_L1 = "#2563eb", "#db2777"
    for r in range(R):
        for c in range(C):
            sc = 0.35
            if abs(b_mvx0[r,c]) > 0 or abs(b_mvy0[r,c]) > 0:
                ax.annotate("", xy=(c + b_mvx0[r,c]/BLOCK*sc, r + b_mvy0[r,c]/BLOCK*sc),
                           xytext=(c, r),
                           arrowprops=dict(arrowstyle="-|>", color=C_L0, lw=0.8,
                                          mutation_scale=5, alpha=0.7))
            if abs(b_mvx1[r,c]) > 0 or abs(b_mvy1[r,c]) > 0:
                ax.annotate("", xy=(c + b_mvx1[r,c]/BLOCK*sc, r + b_mvy1[r,c]/BLOCK*sc),
                           xytext=(c, r),
                           arrowprops=dict(arrowstyle="-|>", color=C_L1, lw=0.8,
                                          mutation_scale=5, alpha=0.7))
    style_ax(ax, "Bidirectional MV field", C_B)
    ax.text(0.02, -0.08, "■ L0 → past (I₀)", transform=ax.transAxes,
            fontsize=8.5, color=C_L0, fontweight="bold")
    ax.text(0.52, -0.08, "■ L1 → future (P₄)", transform=ax.transAxes,
            fontsize=8.5, color=C_L1, fontweight="bold")

    # residual comparison: P vs B side by side in same color scale
    ax = fig.add_subplot(gs[2, 2])
    # stack P and B residuals vertically with a gap
    p_abs = np.abs(p_res)
    b_abs = np.abs(b_res)
    gap = 4
    combined = np.full((SCALE_H * 2 + gap, SCALE_W), np.nan, np.float64)
    combined[:SCALE_H, :] = p_abs
    combined[SCALE_H + gap:, :] = b_abs
    ax.imshow(combined, cmap="hot_r", vmin=0, vmax=50, aspect="auto")
    ax.axhline(SCALE_H + gap // 2, color=C_BORDER, lw=1, ls="--")
    ax.text(SCALE_W // 2, SCALE_H // 2, "P residual", ha="center", va="center",
            fontsize=10, color=C_P, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(SCALE_W // 2, SCALE_H + gap + SCALE_H // 2, "B residual", ha="center", va="center",
            fontsize=10, color=C_B, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    style_ax(ax, "|residual| comparison (same scale)", C_TEXT, fs=10)
    ax.text(0.5, -0.06, "B is darker → smaller residual → fewer bits",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_DIM)

    # annotation
    ax = fig.add_subplot(gs[2, 3])
    text_panel(ax, [
        ("B-FRAME: BIDIRECTIONAL PREDICTION",               C_B,     11, "bold"),
        ("",                                                 C_DIM,    6, "normal"),
        ("Two reference directions:",                        C_TEXT,  9.5, "normal"),
        ("  ■ L0: warp(I₀, MV_L0)   past ref",             C_L0,     9, "bold"),
        ("  ■ L1: warp(P₄, MV_L1)   future ref",           C_L1,     9, "bold"),
        ("  pred = (L0 + L1 + 1) >> 1",                     C_TEXT,   9, "normal"),
        ("",                                                 C_DIM,    6, "normal"),
        ("transmitted =",                                    C_B,     9.5, "bold"),
        ("  Σ [MV₀ + MV₁ + DCT(orig - avg(warp₀,warp₁))]", C_B,     9, "bold"),
        ("",                                                 C_DIM,    6, "normal"),
        ("Why smaller than P?",                              C_TEXT,  9.5, "normal"),
        ("  Two predictions cancel opposite errors.",        C_DIM,    9, "normal"),
        ("  avg residual → 0 faster → sparser coeffs.",     C_DIM,    9, "normal"),
        ("  19% of I-frame size.",                           C_B,      9, "bold"),
    ])

    # ── ROW 3: summary ──────────────────────────────────────────────

    # bar chart
    ax = fig.add_subplot(gs[3, :2])
    types = ["I-Frame\n(idx=0)", "P-Frame\n(idx=4)", "B-Frame\n(idx=1)"]
    sizes = [69801, 30314, 12934]
    colors = [C_I, C_P, C_B]
    bars = ax.barh(types, sizes, color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 88000)
    ax.invert_yaxis()
    for bar, sz, clr in zip(bars, sizes, colors):
        pct = sz / 69801 * 100
        label = f"  {sz:,} bytes" + (f"  ({pct:.0f}% of I)" if pct < 100 else "  (baseline)")
        ax.text(bar.get_width() + 600, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10, color=clr, fontweight="bold")
    ax.set_xlabel("frame size (bytes)", fontsize=9, color=C_DIM)
    ax.tick_params(labelsize=10, colors=C_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.spines["left"].set_color(C_BORDER)
    ax.set_title("Actual frame sizes from birds.mp4 bitstream", fontsize=12,
                 fontweight="bold", color=C_TEXT, loc="left", pad=8)

    # equation summary
    ax = fig.add_subplot(gs[3, 2:])
    text_panel(ax, [
        ("WHAT EACH FRAME TYPE TRANSMITS",                       C_TEXT,  12, "bold"),
        ("",                                                     C_DIM,    4, "normal"),
        ("I:  Σ [ mode + DCT(orig − spatial_pred) ]",            C_I,    10, "bold"),
        ("    every block has coefficients → DENSE",              C_DIM,   9, "normal"),
        ("",                                                     C_DIM,    4, "normal"),
        ("P:  Σ [ MV + DCT(orig − warp(ref, MV)) ]",            C_P,    10, "bold"),
        ("    most blocks skip, few have sparse coeff → SPARSE", C_DIM,   9, "normal"),
        ("",                                                     C_DIM,    4, "normal"),
        ("B:  Σ [ MV₀ + MV₁ + DCT(orig − avg(warp₀,warp₁)) ]", C_B,    10, "bold"),
        ("    two refs cancel more → VERY SPARSE",               C_DIM,   9, "normal"),
        ("",                                                     C_DIM,    4, "normal"),
        ("more references → better prediction → smaller residual → fewer bits",
                                                                 C_TEXT,   9, "bold"),
    ], bg="#f3f4f6")

    fig.savefig(str(OUT), dpi=150, facecolor=C_BG, bbox_inches="tight", pad_inches=0.3)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
