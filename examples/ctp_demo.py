#!/usr/bin/env python3
"""
CTP (Codec Token Perplexity) Demo
==================================

Pipeline:
  1. Extract frames from birds.mp4 (or generate synthetic fallback)
  2. Encode with nano-hevc (trace=True) → C_symbol token sequence
  3. Show token stats + empirical entropy
  4. Train simplest autoregressive codec LM: P(z_t | z_{<t})
  5. Compare BPT: uniform → unigram → CodecLM

Run:  python examples/ctp_demo.py
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from nano_hevc.frame import Frame
from nano_hevc.encoder import encode_frame
from nano_hevc.nal import HEVCConfig
from nano_hevc.cabac import SymbolTrace

VIDEO = Path(__file__).parent / "birds.mp4"


# ── frame extraction ─────────────────────────────────────────────────


def extract_frame(video: Path, idx: int, w: int, h: int) -> Frame:
    """Extract one frame from video as a nano-hevc Frame via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", str(video),
        "-vf", f"select=eq(n\\,{idx}),scale={w}:{h}",
        "-frames:v", "1", "-pix_fmt", "yuv420p",
        "-f", "rawvideo", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    y_sz = w * h
    uv_sz = (w // 2) * (h // 2)
    f = Frame.zeros(h, w, dtype=np.int16)
    f.y.data[:] = np.frombuffer(raw[:y_sz], np.uint8).reshape(h, w).astype(np.int16)
    f.u.data[:] = np.frombuffer(raw[y_sz:y_sz + uv_sz], np.uint8).reshape(h // 2, w // 2).astype(np.int16)
    f.v.data[:] = np.frombuffer(raw[y_sz + uv_sz:y_sz + 2 * uv_sz], np.uint8).reshape(h // 2, w // 2).astype(np.int16)
    return f


def make_synthetic(w: int, h: int, idx: int) -> Frame:
    """Fallback: synthetic frame with gradient + moving block."""
    rng = np.random.RandomState(42 + idx)
    y = np.zeros((h, w), np.int16)
    for r in range(h):
        for c in range(w):
            y[r, c] = int((r * 3 + c * 2 + idx * 7) % 200 + 30)
    y += rng.randint(-15, 16, size=(h, w)).astype(np.int16)
    bx = (idx * 12) % max(1, w - 24)
    y[h // 4: h // 4 + 24, bx: bx + 24] = 240
    np.clip(y, 16, 235, out=y)
    f = Frame.zeros(h, w, dtype=np.int16)
    f.y.data[:] = y
    f.u.data[:] = 128
    f.v.data[:] = 128
    return f


# ── encode ───────────────────────────────────────────────────────────


def encode_and_trace(frames: list[Frame], qp: int) -> list[list[SymbolTrace]]:
    config = HEVCConfig(width=frames[0].width, height=frames[0].height,
                        qp=qp, max_cu_size=8)
    traces = []
    for poc, frame in enumerate(frames):
        _, _, stats = encode_frame(frame, config, fast_mode=True, trace=True, poc=poc)
        traces.append(stats["symbol_trace"])
    return traces


# ── token display ────────────────────────────────────────────────────


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


# ── BPT baselines ───────────────────────────────────────────────────


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


# ── Autoregressive Codec LM ─────────────────────────────────────────
#
# analysis.md §8:
#   P(z_1,...,z_T) = Π_t P(z_t | z_{<t})
#   BPT = -(1/T) Σ_t log2 P(z_t | z_{<t})

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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:

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
          vocab=2:  sigmoid  → Bernoulli   (BCE)
          vocab>2:  softmax  → Categorical (CE)
          vocab=0:  (μ,s)   → Discr.Log.  (NLL)
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

            # binary (vocab=2)
            m = vs == 2
            if m.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    self.head_bin(h[m]).squeeze(-1), tgt[m].float(), reduction="sum")

            # categorical (vocab>2)
            for v in vs[vs > 2].unique():
                mv = vs == v
                loss = loss + F.cross_entropy(
                    self.head_cat(h[mv])[:, :int(v)],
                    tgt[mv].clamp(0, int(v) - 1), reduction="sum")

            # continuous (vocab=0)
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

        # add batch dim
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


# ── main ─────────────────────────────────────────────────────────────


def main():
    W, H, QP = 128, 64, 27
    FRAME_INDICES = [0, 1, 4]  # I, B, P from birds.mp4

    print("=" * 70)
    print("  CTP Demo: Video → Codec Tokens → Autoregressive LM")
    print("=" * 70)

    # step 1: load frames
    if VIDEO.exists():
        print(f"\n[1] Extracting frames {FRAME_INDICES} from birds.mp4 → {W}x{H}")
        frames = [extract_frame(VIDEO, i, W, H) for i in FRAME_INDICES]
        source = "birds.mp4"
    else:
        print(f"\n[1] birds.mp4 not found, using synthetic frames ({W}x{H})")
        frames = [make_synthetic(W, H, i) for i in range(3)]
        source = "synthetic"

    # step 2: encode → token sequence
    print(f"\n[2] Encoding with nano-hevc (QP={QP}, CU=8x8, trace=True)")
    traces = encode_and_trace(frames, QP)
    for i, tr in enumerate(traces):
        print(f"    frame {FRAME_INDICES[i] if VIDEO.exists() else i}: {len(tr)} tokens")

    # step 3: show tokens + stats
    tr0 = traces[0]
    print(f"\n[3] Token sequence (frame {FRAME_INDICES[0]}, first 20):")
    print_tokens(tr0, n=20)
    print(f"\n    Token type statistics:")
    print_stats(tr0)

    # step 4: BPT comparison
    print(f"\n[4] BPT: uniform → unigram → CodecLM (analysis.md §8)")
    uni = compute_uniform_bpt(tr0)
    ugr = compute_unigram_bpt(tr0)

    if _HAS_TORCH:
        print(f"\n    Training CodecLM (d=64, 2L, 2H) on {len(tr0)} tokens ...")
        _, ar, n_p = train_codec_lm(tr0, W, H)
        print(f"\n    {'model':<18} {'BPT':>7}  {'vs uniform':>10}")
        print(f"    {'─' * 38}")
        print(f"    {'uniform':<18} {uni:>7.3f}  {'baseline':>10}")
        print(f"    {'unigram':<18} {ugr:>7.3f}  {(1 - ugr / uni) * 100:>9.1f}%")
        print(f"    {'CodecLM (AR)':<18} {ar:>7.3f}  {(1 - ar / uni) * 100:>9.1f}%")
        print(f"    {'─' * 38}")
        print(f"    {n_p:,} params | source: {source}")
    else:
        print(f"    uniform={uni:.3f}  unigram={ugr:.3f}  [torch not installed]")

    # save trace
    out = "/tmp/ctp_symbol_trace.json"
    with open(out, "w") as f:
        json.dump([{"poc": t.poc, "cu_x": t.cu_x, "cu_y": t.cu_y,
                     "cu_size": t.cu_size, "plane": t.plane,
                     "token_type": t.token_type, "value": t.value,
                     "vocab_size": t.vocab_size} for t in tr0], f, indent=2)
    print(f"\n[output] {len(tr0)} tokens → {out}")


if __name__ == "__main__":
    main()
