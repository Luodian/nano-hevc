#!/usr/bin/env python3
"""
CTP (Codec Token Perplexity) C_symbol Demo
===========================================

This script demonstrates the complete pipeline:

  1. Generate a synthetic video (gradient + moving block)
  2. Encode it with nano-hevc (trace=True)
  3. Extract the C_symbol token sequence
  4. Show how these tokens become a learning target

No external video files needed — everything is self-contained.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from nano_hevc.frame import Frame
from nano_hevc.encoder import encode_frame
from nano_hevc.nal import HEVCConfig
from nano_hevc.cabac import SymbolTrace


# ── Step 1: Generate synthetic video frames ──────────────────────────


def make_synthetic_frame(width: int, height: int, frame_idx: int) -> Frame:
    """Create a frame with noise + edges + a moving block.

    Designed to produce many non-zero residual blocks so the C_symbol
    trace contains a rich variety of token types (not just cbf=0).
    """
    rng = np.random.RandomState(42 + frame_idx)
    y_plane = np.zeros((height, width), dtype=np.int16)
    for row in range(height):
        for col in range(width):
            y_plane[row, col] = int((row * 3 + col * 2 + frame_idx * 7) % 200 + 30)

    y_plane += rng.randint(-15, 16, size=(height, width)).astype(np.int16)

    bx = (frame_idx * 12) % max(1, width - 24)
    by = height // 4
    y_plane[by : by + 24, bx : bx + 24] = 240

    edge_y = height // 2
    y_plane[edge_y : edge_y + 2, :] = 16

    np.clip(y_plane, 16, 235, out=y_plane)
    u_plane = np.full((height // 2, width // 2), 128, dtype=np.int16)
    u_plane += rng.randint(-8, 9, size=u_plane.shape).astype(np.int16)
    np.clip(u_plane, 16, 240, out=u_plane)
    v_plane = np.full((height // 2, width // 2), 128, dtype=np.int16)

    frame = Frame.zeros(height, width, dtype=np.int16)
    frame.y.data[:] = y_plane
    frame.u.data[:] = u_plane
    frame.v.data[:] = v_plane
    return frame


# ── Step 2 & 3: Encode with trace, extract C_symbol ─────────────────


def encode_and_trace(frames: list[Frame], qp: int = 27) -> list[list[SymbolTrace]]:
    config = HEVCConfig(width=frames[0].width, height=frames[0].height, qp=qp,
                        max_cu_size=8)  # 8x8 blocks -> more tokens per frame
    all_traces = []

    for poc, frame in enumerate(frames):
        _encoded_bytes, _recon, stats = encode_frame(
            frame,
            config,
            fast_mode=True,
            trace=True,
            poc=poc,
        )
        all_traces.append(stats["symbol_trace"])
    return all_traces


# ── Step 4: Analyze and show how it becomes a learning target ────────


def print_token_sequence(trace: list[SymbolTrace], max_tokens: int = 50):
    """Pretty-print the first N tokens of the C_symbol sequence."""
    print(f"\n{'─' * 90}")
    print(f"{'#':>4}  {'token_type':<30} {'value':>6} {'vocab':>6}  {'位置 (plane, x, y, size)'}")
    print(f"{'─' * 90}")
    for i, t in enumerate(trace[:max_tokens]):
        loc = f"{t.plane} ({t.cu_x:3d},{t.cu_y:3d}) {t.cu_size}x{t.cu_size}"
        vocab_str = str(t.vocab_size) if t.vocab_size > 0 else "open"
        print(f"{i:4d}  {t.token_type:<30} {t.value:>6} {vocab_str:>6}  {loc}")
    if len(trace) > max_tokens:
        print(f"  ... ({len(trace) - max_tokens} more tokens)")


def compute_token_stats(trace: list[SymbolTrace]) -> dict:
    """Compute per-token-type statistics."""
    by_type: dict[str, list[int]] = {}
    for t in trace:
        by_type.setdefault(t.token_type, []).append(t.value)

    stats = {}
    for ttype, values in sorted(by_type.items()):
        counter = Counter(values)
        n = len(values)
        # Empirical entropy (bits per token for this type)
        entropy = 0.0
        for count in counter.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        stats[ttype] = {
            "count": n,
            "unique_values": len(counter),
            "empirical_entropy_bpt": round(entropy, 3),
            "top_values": counter.most_common(5),
        }
    return stats


def show_learning_target(trace: list[SymbolTrace]):
    """Show how the C_symbol sequence becomes a prediction dataset."""
    print("\n" + "=" * 90)
    print("HOW C_symbol BECOMES A LEARNING TARGET")
    print("=" * 90)

    # Build the dataset: each token is a (context, label) pair
    print("""
每个 token 变成一个 (输入, 标签) 对:

  输入 (context):
    - 视频帧 V_{<=t}  (原始像素或 encoder feature)
    - 已出现的 token z_{<i}  (token history)
    - 空间位置 (cu_x, cu_y, cu_size, plane)

  标签 (target):
    - token_type: 告诉模型"现在要预测什么"
    - value: ground truth 取值（分类标签）
    - vocab_size: 取值空间大小 → 决定 prediction head 的输出维度
""")

    # Show concrete examples
    print("具体的 (输入, 标签) 样本：\n")
    examples = []
    seen_types = set()
    for t in trace:
        if t.token_type not in seen_types and len(examples) < 8:
            examples.append(t)
            seen_types.add(t.token_type)

    for i, t in enumerate(examples):
        vocab_str = str(t.vocab_size) if t.vocab_size > 0 else "open-ended (regression)"
        print(f"  样本 {i+1}:")
        print(f"    token_type = {t.token_type!r}")
        print(f"    位置       = POC={t.poc}, ({t.cu_x},{t.cu_y}), {t.cu_size}x{t.cu_size}, {t.plane}")
        print(f"    label      = {t.value}")
        print(f"    vocab_size = {vocab_str}")

        if t.vocab_size == 2:
            print(f"    → 二分类: model outputs p(value=1), loss = BCE")
        elif t.vocab_size > 2:
            print(f"    → {t.vocab_size}-way 分类: model outputs softmax({t.vocab_size}), loss = CE")
        else:
            print(f"    → 回归/开放词表: model outputs distribution, loss = NLL")
        print()

    # Show the BPT calculation
    print("─" * 90)
    print("BPT (Bits Per Token) 计算：")
    print()
    print("  假设模型对每个 token 输出概率 p(z_i | context):")
    print()
    print("  BPT = (1/N) * Σ -log2( p(z_i | V_{<=t}, z_{<i}) )")
    print()
    n = len(trace)
    print(f"  本帧共 N = {n} 个 tokens")
    print(f"  如果模型全猜均匀分布 (baseline):")

    # Compute random baseline
    total_bits = 0.0
    for t in trace:
        if t.vocab_size == 2:
            total_bits += 1.0  # log2(2) = 1
        elif t.vocab_size > 2:
            total_bits += math.log2(t.vocab_size)
        else:
            total_bits += 8.0  # rough estimate for open-ended
    baseline_bpt = total_bits / n if n > 0 else 0
    print(f"    BPT_random = {baseline_bpt:.3f} bits/token")
    print()
    print("  好的 world model 应该能把 BPT 压到远低于这个 baseline。")
    print("  BPT 越低 = 模型越懂视频内容 = world understanding 越强。")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    width, height = 128, 128
    num_frames = 3
    qp = 27

    print("=" * 90)
    print("CTP C_symbol Demo: 从视频到学习目标的完整流程")
    print("=" * 90)
    print(f"\n视频: {width}x{height}, {num_frames} frames, QP={qp}")

    # Step 1: Generate frames
    print("\n[Step 1] 生成合成视频帧 (gradient + moving block)...")
    frames = [make_synthetic_frame(width, height, i) for i in range(num_frames)]
    print(f"  生成了 {num_frames} 帧")

    # Step 2: Encode with trace
    print("\n[Step 2] 用 nano-hevc 编码 (trace=True)...")
    all_traces = encode_and_trace(frames, qp=qp)
    for poc, trace in enumerate(all_traces):
        print(f"  Frame {poc}: {len(trace)} C_symbol tokens")

    # Step 3: Show the token sequence
    print("\n[Step 3] Frame 0 的 C_symbol token 序列（前 50 个）:")
    print_token_sequence(all_traces[0], max_tokens=50)

    # Token type statistics
    print("\n[Step 3b] 各 token type 统计:")
    stats = compute_token_stats(all_traces[0])
    print(f"\n{'token_type':<30} {'count':>6} {'unique':>7} {'entropy':>10} {'top values'}")
    print(f"{'─' * 90}")
    for ttype, s in stats.items():
        top_str = ", ".join(f"{v}({c})" for v, c in s["top_values"][:3])
        print(f"{ttype:<30} {s['count']:>6} {s['unique_values']:>7} {s['empirical_entropy_bpt']:>8.3f}  {top_str}")

    # Step 4: Show as learning target
    show_learning_target(all_traces[0])

    # Save trace as JSON for downstream use
    output_path = "/tmp/ctp_symbol_trace.json"
    json_trace = [
        {
            "poc": t.poc,
            "cu_x": t.cu_x,
            "cu_y": t.cu_y,
            "cu_size": t.cu_size,
            "plane": t.plane,
            "token_type": t.token_type,
            "value": t.value,
            "vocab_size": t.vocab_size,
        }
        for t in all_traces[0]
    ]
    with open(output_path, "w") as f:
        json.dump(json_trace, f, indent=2)
    print(f"\n[Output] Frame 0 trace saved to {output_path}")
    print(f"         {len(json_trace)} tokens, {len(json.dumps(json_trace))} bytes JSON")


if __name__ == "__main__":
    main()
