#!/usr/bin/env python3
"""
CTP (Codec Token Perplexity) Demo
==================================

Pipeline:
  1. Load frames from video via Decord (same pipeline as LWM)
  2. RGB → YUV420 → nano-hevc encode (trace=True) → symbol token sequence
  3. Show token stats + empirical entropy
  4. Train autoregressive codec LM: P(z_t | z_{<t})
  5. Compare BPT: uniform → unigram → CodecLM

Run:  python examples/ctp_demo.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_hevc.cabac import SymbolTrace
from nano_hevc.encoder import encode_frame
from nano_hevc.frame import Frame
from nano_hevc.nal import HEVCConfig

VIDEO = Path(__file__).parent / "birds.mp4"


# -- frame loading (decord, matching LWM data.py) ----------------------------


def load_video_frames(
    video_path: str | Path,
    num_frames: int,
    width: int,
    height: int,
    fps: int = 8,
) -> list[Frame]:
    """Load frames via PyAV and convert RGB -> YUV420 Frame objects.

    Replicates the segment-selection and frame-stepping logic from
    lwm/data.py:load_video_frames so the two pipelines see identical
    frame sequences given the same random state.

    Backend priority: decord (LWM default) -> av (PyAV, portable fallback).
    """
    try:
        return _load_frames_decord(video_path, num_frames, width, height, fps)
    except ImportError:
        pass
    return _load_frames_pyav(video_path, num_frames, width, height, fps)


def _load_frames_decord(video_path, num_frames, width, height, fps):
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    video_fps = vr.get_avg_fps() or 30.0
    indices = _pick_indices(total, video_fps, num_frames, fps)
    batch = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
    return [_resize_and_convert(rgb, width, height) for rgb in batch]


def _load_frames_pyav(video_path, num_frames, width, height, fps):
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    video_fps = float(stream.average_rate or 30)

    # decode all frames into list, then index
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if not total:
        total = len(all_frames)
    indices = _pick_indices(total, video_fps, num_frames, fps)
    return [_resize_and_convert(all_frames[min(i, len(all_frames) - 1)], width, height)
            for i in indices]


def _pick_indices(total, video_fps, num_frames, target_fps):
    """LWM-style segment selection: random start, fps stepping, pad short videos."""
    step = max(1, round(video_fps / target_fps))
    total_available = total // step

    if total_available > num_frames:
        start = random.randint(0, total_available - num_frames)
    else:
        start = 0

    indices = [
        min(start * step + i * step, total - 1)
        for i in range(min(num_frames, max(total_available, 1)))
    ]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices


def _resize_and_convert(rgb, width, height):
    if rgb.shape[0] != height or rgb.shape[1] != width:
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False)
        rgb = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return _rgb_to_yuv420(rgb, width, height)


def _rgb_to_yuv420(rgb: np.ndarray, w: int, h: int) -> Frame:
    """BT.601 full-range RGB uint8 -> YUV420 int16 Frame."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128.0

    np.clip(y, 0, 255, out=y)
    np.clip(cb, 0, 255, out=cb)
    np.clip(cr, 0, 255, out=cr)

    # 4:2:0 chroma subsampling: average 2x2 blocks
    u = (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2]) * 0.25
    v = (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2]) * 0.25

    frame = Frame.zeros(h, w, dtype=np.int16)
    frame.y.data[:] = y.astype(np.int16)
    frame.u.data[:] = u.astype(np.int16)
    frame.v.data[:] = v.astype(np.int16)
    return frame


# -- encode -------------------------------------------------------------------


def encode_and_trace(frames: list[Frame], qp: int) -> list[list[SymbolTrace]]:
    config = HEVCConfig(width=frames[0].width, height=frames[0].height,
                        qp=qp, max_cu_size=8)
    traces = []
    for poc, frame in enumerate(frames):
        _, _, stats = encode_frame(frame, config, fast_mode=True, trace=True, poc=poc)
        traces.append(stats["symbol_trace"])
    return traces


# -- token display ------------------------------------------------------------


def print_tokens(trace: list[SymbolTrace], n: int = 20):
    print(f"\n  {'#':>3}  {'token_type':<28} {'val':>4} {'vocab':>5}  {'location'}")
    print(f"  {'─' * 70}")
    for i, t in enumerate(trace[:n]):
        v = str(t.vocab_size) if t.vocab_size > 0 else "open"
        print(f"  {i:3d}  {t.token_type:<28} {t.value:>4} {v:>5}  {t.plane} ({t.cu_x},{t.cu_y}) {t.cu_size}x{t.cu_size}")
    if len(trace) > n:
        print(f"  ... ({len(trace) - n} more)")


def print_stats(trace: list[SymbolTrace]):
    by_type: dict[str, list[int]] = {}
    for t in trace:
        by_type.setdefault(t.token_type, []).append(t.value)
    print(f"\n  {'token_type':<28} {'count':>5} {'unique':>6} {'H(bits)':>7}  top values")
    print(f"  {'─' * 70}")
    for tt in sorted(by_type):
        vals = by_type[tt]
        c = Counter(vals)
        n = len(vals)
        h = -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values() if cnt > 0)
        top = ", ".join(f"{v}({cnt})" for v, cnt in c.most_common(3))
        print(f"  {tt:<28} {n:>5} {len(c):>6} {h:>7.3f}  {top}")


# -- BPT baselines -----------------------------------------------------------


def compute_uniform_bpt(trace: list[SymbolTrace]) -> float:
    """P(z_t) = 1/vocab_size."""
    total = sum(1.0 if t.vocab_size == 2
                else math.log2(t.vocab_size) if t.vocab_size > 2
                else 8.0 for t in trace)
    return total / len(trace) if trace else 0.0


def compute_unigram_bpt(trace: list[SymbolTrace]) -> float:
    """P(z_t) = freq(value | token_type). No context."""
    freq: dict[str, Counter] = {}
    for t in trace:
        freq.setdefault(t.token_type, Counter())[t.value] += 1
    total = 0.0
    for t in trace:
        c = freq[t.token_type]
        total -= math.log2(max(c[t.value] / sum(c.values()), 1e-10))
    return total / len(trace) if trace else 0.0


# -- Autoregressive Codec LM -------------------------------------------------
#
# analysis.md section 8:
#   P(z_1,...,z_T) = prod_t P(z_t | z_{<t})
#   BPT = -(1/T) sum_t log2 P(z_t | z_{<t})

_TOKEN_TYPES = [
    "split_cu_flag", "part_mode", "intra_luma_pred_mode",
    "intra_chroma_pred_mode", "cbf_luma", "cbf_chroma",
    "last_sig_coeff_x", "last_sig_coeff_y", "sig_coeff_flag",
    "coeff_abs_level_greater1", "coeff_abs_level_greater2",
    "coeff_sign", "coeff_abs_level_remaining",
]
_TYPE2ID = {name: i for i, name in enumerate(_TOKEN_TYPES)}
_PLANE2ID = {"Y": 0, "U": 1, "V": 2}
_MAX_VAL = 256


def _to_tensors(trace, fw, fh):
    T = len(trace)
    tid = torch.zeros(T, dtype=torch.long)
    val = torch.zeros(T, dtype=torch.long)
    pos = torch.zeros(T, 4)
    voc = torch.zeros(T, dtype=torch.long)
    for i, t in enumerate(trace):
        tid[i] = _TYPE2ID.get(t.token_type, 0)
        val[i] = min(abs(t.value), _MAX_VAL)
        pos[i] = torch.tensor([t.cu_x / max(fw, 1), t.cu_y / max(fh, 1),
                               t.cu_size / 64.0, _PLANE2ID.get(t.plane, 0) / 2.0])
        voc[i] = t.vocab_size
    return tid, val, pos, voc


class CodecLM(nn.Module):
    """P(z_t | z_{<t}) via causal Transformer with multi-head output.

    Head dispatch by vocab_size:
      vocab=2:  sigmoid  -> Bernoulli   (BCE)
      vocab>2:  softmax  -> Categorical (CE)
      vocab=0:  (mu,s)   -> Discr.Log.  (NLL)
    """

    def __init__(self, d=64, heads=2, layers=2):
        super().__init__()
        self.type_emb = nn.Embedding(len(_TOKEN_TYPES) + 1, d)
        self.val_emb = nn.Embedding(_MAX_VAL + 2, d)
        self.pos_proj = nn.Linear(4, d)
        self.mix = nn.Linear(d * 3, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=layers)
        self.head_bin = nn.Linear(d, 1)
        self.head_cat = nn.Linear(d, 64)
        self.head_cont = nn.Linear(d, 2)

    def forward(self, tid, prev_val, pos):
        x = self.mix(torch.cat([
            self.type_emb(tid), self.val_emb(prev_val), self.pos_proj(pos)
        ], dim=-1))
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        return self.backbone(x, mask=mask)

    def nll(self, h, targets, vocab_sizes):
        h, tgt, vs = h.squeeze(0), targets.squeeze(0), vocab_sizes.squeeze(0)
        T = h.shape[0]
        loss = torch.tensor(0.0, device=h.device)

        m = vs == 2
        if m.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                self.head_bin(h[m]).squeeze(-1), tgt[m].float(), reduction="sum")

        for v in vs[vs > 2].unique():
            mv = vs == v
            loss = loss + F.cross_entropy(
                self.head_cat(h[mv])[:, :int(v)],
                tgt[mv].clamp(0, int(v) - 1), reduction="sum")

        m = vs == 0
        if m.any():
            mu, log_s = self.head_cont(h[m]).unbind(-1)
            s = log_s.exp().clamp(min=0.01)
            x = tgt[m].float()
            prob = (torch.sigmoid((x + 0.5 - mu) / s)
                    - torch.sigmoid((x - 0.5 - mu) / s)).clamp(min=1e-8)
            loss = loss - prob.log().sum()

        return loss / T, (loss / T).item() / math.log(2)


def train_codec_lm(trace, fw, fh, epochs=200, lr=3e-3):
    tid, val, pos, voc = _to_tensors(trace, fw, fh)
    prev = torch.cat([torch.tensor([_MAX_VAL + 1]), val[:-1]])

    tid_b, prev_b, pos_b = tid[None], prev[None], pos[None]
    tgt_b, voc_b = val[None], voc[None]

    model = CodecLM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        h = model(tid_b, prev_b, pos_b)
        loss, bpt = model.nll(h, tgt_b, voc_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:>3d}/{epochs}  BPT={bpt:.3f}")

    model.eval()
    with torch.no_grad():
        _, final = model.nll(model(tid_b, prev_b, pos_b), tgt_b, voc_b)
    return model, final, sum(p.numel() for p in model.parameters())


# -- main ---------------------------------------------------------------------


def main():
    W, H, QP = 128, 64, 27
    NUM_FRAMES = 3

    print("=" * 70)
    print("  CTP Demo: Video -> Codec Tokens -> Autoregressive LM")
    print("=" * 70)

    print(f"\n[1] Loading {NUM_FRAMES} frames from {VIDEO} via Decord -> {W}x{H}")
    frames = load_video_frames(VIDEO, NUM_FRAMES, W, H)

    print(f"\n[2] Encoding with nano-hevc (QP={QP}, CU=8x8, trace=True)")
    traces = encode_and_trace(frames, QP)
    for i, tr in enumerate(traces):
        print(f"    frame {i}: {len(tr)} tokens")

    tr0 = traces[0]
    print(f"\n[3] Token sequence (frame 0, first 20):")
    print_tokens(tr0, n=20)
    print(f"\n    Token type statistics:")
    print_stats(tr0)

    print(f"\n[4] BPT: uniform -> unigram -> CodecLM")
    uni = compute_uniform_bpt(tr0)
    ugr = compute_unigram_bpt(tr0)

    print(f"\n    Training CodecLM (d=64, 2L, 2H) on {len(tr0)} tokens ...")
    _, ar, n_p = train_codec_lm(tr0, W, H)
    print(f"\n    {'model':<18} {'BPT':>7}  {'vs uniform':>10}")
    print(f"    {'─' * 38}")
    print(f"    {'uniform':<18} {uni:>7.3f}  {'baseline':>10}")
    print(f"    {'unigram':<18} {ugr:>7.3f}  {(1 - ugr / uni) * 100:>9.1f}%")
    print(f"    {'CodecLM (AR)':<18} {ar:>7.3f}  {(1 - ar / uni) * 100:>9.1f}%")
    print(f"    {'─' * 38}")
    print(f"    {n_p:,} params | source: {VIDEO.name}")

    out = "/tmp/ctp_symbol_trace.json"
    with open(out, "w") as f:
        json.dump([{"poc": t.poc, "cu_x": t.cu_x, "cu_y": t.cu_y,
                     "cu_size": t.cu_size, "plane": t.plane,
                     "token_type": t.token_type, "value": t.value,
                     "vocab_size": t.vocab_size} for t in tr0], f, indent=2)
    print(f"\n[output] {len(tr0)} tokens -> {out}")


if __name__ == "__main__":
    main()
