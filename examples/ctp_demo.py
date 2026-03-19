#!/usr/bin/env python3
"""
Minimal CTP trainer in the spirit of nanoGPT.

Pipeline:
  1. Load video frames (local file or Lance-indexed S3 tars from howto100m)
  2. RGB -> YUV420 -> nano-hevc encode (trace=True) -> symbol token sequences
  3. Split traces into train / val
  4. Train a causal Transformer with typed heads
  5. Report validation BPT against simple baselines
  6. Log everything to W&B (optional)

Local:   python examples/ctp_demo.py
Cluster: python examples/ctp_demo.py --lance-uri s3://ami-labs-data/tables/s3_video_primary.lance --datasets howto100m --num-videos 50 --wandb
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
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
PLANE_TO_ID = {"Y": 0, "U": 1, "V": 2}
OPEN_VOCAB_UNIFORM_BITS = 8.0
_VIDEO_EXTS = frozenset({".mp4", ".webm", ".mkv", ".avi", ".mov"})


# -- CLI ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTP trainer: video -> codec tokens -> LM")

    # data source
    p.add_argument("--video", type=Path, default=VIDEO, help="local video file")
    p.add_argument("--lance-uri", type=str, default=None,
                   help="Lance table URI for S3 tar-backed loading")
    p.add_argument("--datasets", nargs="+", default=["howto100m"])
    p.add_argument("--s3-bucket", default="ami-labs-data")
    p.add_argument("--s3-endpoint", default=None)
    p.add_argument("--num-videos", type=int, default=3,
                   help="videos to sample from Lance")

    # encoder
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--qp", type=int, default=27)
    p.add_argument("--max-cu-size", type=int, default=8)

    # training
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    # model
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    # infra
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--wandb", action="store_true", help="log to W&B")
    p.add_argument("--wandb-project", default="nano-hevc-ctp")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--trace-out", type=Path, default=None,
                   help="optional JSON dump for the first trace")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -- frame loading: local file ------------------------------------------------


def load_video_frames(video_path, num_frames, width, height, fps=8):
    try:
        return _load_frames_decord(video_path, num_frames, width, height, fps)
    except ImportError:
        pass
    return _load_frames_pyav(video_path, num_frames, width, height, fps)


def _load_frames_decord(video_path, num_frames, width, height, fps):
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    indices = _pick_indices(len(vr), vr.get_avg_fps() or 30.0, num_frames, fps)
    batch = vr.get_batch(indices).asnumpy()
    return [_resize_and_convert(rgb, width, height) for rgb in batch]


def _load_frames_pyav(video_path, num_frames, width, height, fps):
    import av
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    video_fps = float(stream.average_rate or 30)
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    if not total:
        total = len(all_frames)
    indices = _pick_indices(total, video_fps, num_frames, fps)
    return [_resize_and_convert(all_frames[min(i, len(all_frames) - 1)], width, height)
            for i in indices]


# -- frame loading: Lance manifest + S3 range reads ---------------------------


def load_manifest_from_lance(lance_uri, datasets=None, s3_endpoint=None):
    """Scan Lance table -> {tar_key: [(member_path, offset, size), ...]}."""
    import lance
    from collections import defaultdict

    kw = {}
    if lance_uri.startswith("s3://"):
        endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            opts = {"endpoint": endpoint,
                    "allow_http": "true" if endpoint.startswith("http://") else "false"}
            ak = os.environ.get("AWS_ACCESS_KEY_ID")
            sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if ak and sk:
                opts["aws_access_key_id"] = ak
                opts["aws_secret_access_key"] = sk
            kw["storage_options"] = opts

    ds = lance.dataset(lance_uri, **kw)
    lance_filter = None
    if datasets:
        preds = [f"tar_path LIKE '%/tars/{d}/%'" for d in datasets]
        lance_filter = " OR ".join(preds)

    scan_kw = {"columns": ["tar_path", "member_path", "offset", "size"], "batch_size": 65536}
    if lance_filter:
        scan_kw["filter"] = lance_filter

    manifest = defaultdict(list)
    for batch in ds.to_batches(**scan_kw):
        for tp, mp, off, sz in zip(batch["tar_path"].to_pylist(),
                                    batch["member_path"].to_pylist(),
                                    batch["offset"].to_pylist(),
                                    batch["size"].to_pylist(), strict=False):
            manifest[tp].append((mp, off, sz))
    return dict(manifest)


def load_frames_from_lance(manifest, num_videos, num_frames, width, height,
                           fps, s3_bucket, s3_endpoint):
    """Sample random videos from manifest, decode via S3 range reads."""
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
    kw = {"config": BotoConfig(connect_timeout=30, read_timeout=300,
                                retries={"max_attempts": 3})}
    if endpoint:
        kw["endpoint_url"] = endpoint
    client = boto3.client("s3", **kw)

    video_members = [(tk, mp, off, sz) for tk, members in manifest.items()
                     for mp, off, sz in members
                     if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS]
    if not video_members:
        raise RuntimeError("no video members found in manifest")

    random.shuffle(video_members)
    selected = video_members[:num_videos]

    all_frames = []
    sources = []
    for tar_key, member_path, offset, size in selected:
        print(f"    fetching {member_path} ({size / 1e6:.1f} MB)")
        try:
            resp = client.get_object(Bucket=s3_bucket, Key=tar_key,
                                     Range=f"bytes={offset}-{offset + size - 1}")
            video_bytes = resp["Body"].read()
            frames = _decode_video_bytes(video_bytes, num_frames, width, height, fps)
            if frames:
                all_frames.extend(frames)
                sources.append(member_path)
            else:
                print(f"    decode failed for {member_path}")
        except Exception as e:
            print(f"    fetch failed for {member_path}: {e}")
    return all_frames, sources


def _decode_video_bytes(raw_bytes, num_frames, width, height, fps):
    try:
        return _decode_bytes_decord(raw_bytes, num_frames, width, height, fps)
    except ImportError:
        pass
    return _decode_bytes_pyav(raw_bytes, num_frames, width, height, fps)


def _decode_bytes_decord(raw_bytes, num_frames, width, height, fps):
    from decord import VideoReader, cpu
    vr = VideoReader(io.BytesIO(raw_bytes), ctx=cpu(0))
    indices = _pick_indices(len(vr), vr.get_avg_fps() or 30.0, num_frames, fps)
    batch = vr.get_batch(indices).asnumpy()
    return [_resize_and_convert(rgb, width, height) for rgb in batch]


def _decode_bytes_pyav(raw_bytes, num_frames, width, height, fps):
    import av
    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.video[0]
    total = stream.frames or 0
    video_fps = float(stream.average_rate or 30)
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    if not all_frames:
        return None
    if not total:
        total = len(all_frames)
    indices = _pick_indices(total, video_fps, num_frames, fps)
    return [_resize_and_convert(all_frames[min(i, len(all_frames) - 1)], width, height)
            for i in indices]


# -- shared frame utilities ---------------------------------------------------


def _pick_indices(total, video_fps, num_frames, target_fps):
    step = max(1, round(video_fps / target_fps))
    total_available = total // step
    if total_available > num_frames:
        start = random.randint(0, total_available - num_frames)
    else:
        start = 0
    indices = [min(start * step + i * step, total - 1)
               for i in range(min(num_frames, max(total_available, 1)))]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices


def _resize_and_convert(rgb, width, height):
    if rgb.shape[0] != height or rgb.shape[1] != width:
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False)
        rgb = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return _rgb_to_yuv420(rgb, width, height)


def _rgb_to_yuv420(rgb, w, h):
    r, g, b = rgb[:,:,0].astype(np.float32), rgb[:,:,1].astype(np.float32), rgb[:,:,2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128.0
    np.clip(y, 0, 255, out=y); np.clip(cb, 0, 255, out=cb); np.clip(cr, 0, 255, out=cr)
    u = (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2]) * 0.25
    v = (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2]) * 0.25
    frame = Frame.zeros(h, w, dtype=np.int16)
    frame.y.data[:] = y.astype(np.int16)
    frame.u.data[:] = u.astype(np.int16)
    frame.v.data[:] = v.astype(np.int16)
    return frame


# -- encode -> traces ---------------------------------------------------------


def encode_symbol_traces(frames, qp, max_cu_size):
    config = HEVCConfig(width=frames[0].width, height=frames[0].height,
                        qp=qp, max_cu_size=max_cu_size)
    traces = []
    for poc, frame in enumerate(frames):
        _, _, stats = encode_frame(frame, config, fast_mode=True, trace=True, poc=poc)
        traces.append(stats["symbol_trace"])
    return traces


# -- dataset construction (nanoGPT-style) -------------------------------------


@dataclass(frozen=True)
class HeadSpec:
    family: str
    out_dim: int
    learn: bool = True


@dataclass
class TensorTrace:
    type_ids: torch.Tensor
    prev_value_ids: torch.Tensor
    targets: torch.Tensor
    vocab_sizes: torch.Tensor
    aux: torch.Tensor
    learn_mask: torch.Tensor

    def length(self):
        return int(self.targets.numel())


@dataclass
class DatasetBundle:
    train: list[TensorTrace]
    val: list[TensorTrace]
    type_to_id: dict[str, int]
    id_to_type: list[str]
    head_specs: dict[str, HeadSpec]
    pad_type_id: int
    bos_value_id: int
    value_vocab_size: int
    value_limit: int
    source: str
    total_tokens: int


@dataclass
class ModelConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    type_vocab_size: int
    value_vocab_size: int
    aux_dim: int = 5


def infer_head_spec(token_type, vocab_size):
    if token_type == "coeff_sign":
        return HeadSpec(family="binary", out_dim=1, learn=False)
    if vocab_size == 2:
        return HeadSpec(family="binary", out_dim=1, learn=True)
    if vocab_size > 2:
        return HeadSpec(family="categorical", out_dim=vocab_size, learn=True)
    return HeadSpec(family="continuous", out_dim=2, learn=True)


def build_head_specs(traces):
    specs = {}
    for trace in traces:
        for token in trace:
            current = infer_head_spec(token.token_type, token.vocab_size)
            if token.token_type not in specs:
                specs[token.token_type] = current
                continue
            prev = specs[token.token_type]
            specs[token.token_type] = HeadSpec(
                family=prev.family,
                out_dim=max(prev.out_dim, current.out_dim),
                learn=prev.learn and current.learn,
            )
    return dict(sorted(specs.items()))


def build_dataset(traces, width, height, val_fraction, seed, source):
    head_specs = build_head_specs(traces)
    id_to_type = list(head_specs.keys())
    type_to_id = {name: idx for idx, name in enumerate(id_to_type)}
    pad_type_id = len(id_to_type)

    value_limit = max(8, max(abs(t.value) for tr in traces for t in tr))
    value_vocab_size = value_limit * 2 + 3
    bos_value_id = value_vocab_size - 1
    max_poc = max(1, max(t.poc for tr in traces for t in tr))
    max_cu_size = max(1, max(t.cu_size for tr in traces for t in tr))

    def value_to_id(v):
        return max(0, min(v + value_limit, value_vocab_size - 2))

    tensor_traces = []
    for trace in traces:
        n = len(trace)
        type_ids = torch.tensor([type_to_id[t.token_type] for t in trace], dtype=torch.long)
        targets = torch.tensor([t.value for t in trace], dtype=torch.long)
        vocab_sizes = torch.tensor([t.vocab_size for t in trace], dtype=torch.long)
        prev_value_ids = torch.full((n,), bos_value_id, dtype=torch.long)
        if n > 1:
            prev_value_ids[1:] = torch.tensor([value_to_id(t.value) for t in trace[:-1]], dtype=torch.long)
        aux = torch.tensor(
            [[t.poc / max_poc, t.cu_x / max(width, 1), t.cu_y / max(height, 1),
              t.cu_size / max_cu_size, PLANE_TO_ID.get(t.plane, 0) / 2.0]
             for t in trace], dtype=torch.float32)
        learn_mask = torch.tensor([head_specs[t.token_type].learn for t in trace], dtype=torch.bool)
        tensor_traces.append(TensorTrace(type_ids=type_ids, prev_value_ids=prev_value_ids,
                                          targets=targets, vocab_sizes=vocab_sizes,
                                          aux=aux, learn_mask=learn_mask))

    rng = random.Random(seed)
    rng.shuffle(tensor_traces)

    if len(tensor_traces) == 1:
        train = val = tensor_traces
    else:
        split = int(round(len(tensor_traces) * (1.0 - val_fraction)))
        split = min(max(split, 1), len(tensor_traces) - 1)
        train, val = tensor_traces[:split], tensor_traces[split:]

    total_tokens = sum(t.length() for t in tensor_traces)
    return DatasetBundle(train=train, val=val, type_to_id=type_to_id, id_to_type=id_to_type,
                         head_specs=head_specs, pad_type_id=pad_type_id, bos_value_id=bos_value_id,
                         value_vocab_size=value_vocab_size, value_limit=value_limit,
                         source=source, total_tokens=total_tokens)


# -- batching -----------------------------------------------------------------


def make_batch(traces, batch_size, block_size, pad_type_id, bos_value_id, device):
    aux_dim = traces[0].aux.shape[-1]
    type_ids = torch.full((batch_size, block_size), pad_type_id, dtype=torch.long)
    prev_value_ids = torch.full((batch_size, block_size), bos_value_id, dtype=torch.long)
    targets = torch.zeros((batch_size, block_size), dtype=torch.long)
    vocab_sizes = torch.zeros((batch_size, block_size), dtype=torch.long)
    aux = torch.zeros((batch_size, block_size, aux_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)
    learn_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)

    for row in range(batch_size):
        trace = traces[random.randrange(len(traces))]
        length = trace.length()
        start = 0 if length <= block_size else random.randrange(0, length - block_size + 1)
        stop = min(length, start + block_size)
        span = stop - start
        type_ids[row, :span] = trace.type_ids[start:stop]
        prev_value_ids[row, :span] = trace.prev_value_ids[start:stop]
        targets[row, :span] = trace.targets[start:stop]
        vocab_sizes[row, :span] = trace.vocab_sizes[start:stop]
        aux[row, :span] = trace.aux[start:stop]
        valid_mask[row, :span] = True
        learn_mask[row, :span] = trace.learn_mask[start:stop]

    return {k: v.to(device) for k, v in {
        "type_ids": type_ids, "prev_value_ids": prev_value_ids,
        "targets": targets, "vocab_sizes": vocab_sizes,
        "aux": aux, "valid_mask": valid_mask, "learn_mask": learn_mask,
    }.items()}


# -- model --------------------------------------------------------------------


def discrete_logistic_nll(params, targets):
    mu, log_scale = params.unbind(dim=-1)
    scale = log_scale.exp().clamp(min=0.01, max=50.0)
    x = targets.float()
    prob = (torch.sigmoid((x + 0.5 - mu) / scale)
            - torch.sigmoid((x - 0.5 - mu) / scale)).clamp(min=1e-9)
    return -prob.log()


class CTPModel(nn.Module):
    def __init__(self, config: ModelConfig, head_specs, type_to_id):
        super().__init__()
        self.config = config
        self.head_specs = head_specs
        self.type_to_id = type_to_id

        self.type_emb = nn.Embedding(config.type_vocab_size, config.n_embd)
        self.prev_value_emb = nn.Embedding(config.value_vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.aux_proj = nn.Linear(config.aux_dim, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd, nhead=config.n_head,
            dim_feedforward=config.n_embd * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.heads = nn.ModuleDict({
            name: nn.Linear(config.n_embd, spec.out_dim)
            for name, spec in head_specs.items()
        })

    def forward(self, batch):
        type_ids = batch["type_ids"]
        B, T = type_ids.shape
        positions = torch.arange(T, device=type_ids.device)
        x = (self.type_emb(type_ids)
             + self.prev_value_emb(batch["prev_value_ids"])
             + self.pos_emb(positions)[None]
             + self.aux_proj(batch["aux"]))
        x = self.drop(x)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.ln_f(self.backbone(x, mask=causal_mask))

        hidden_flat = hidden.reshape(B * T, -1)
        type_flat = type_ids.reshape(-1)
        target_flat = batch["targets"].reshape(-1)
        vocab_flat = batch["vocab_sizes"].reshape(-1)
        learn_flat = (batch["learn_mask"] & batch["valid_mask"]).reshape(-1)

        loss_total = hidden.new_zeros(())
        token_total = 0

        for token_type, spec in self.head_specs.items():
            mask = learn_flat & (type_flat == self.type_to_id[token_type])
            if not mask.any():
                continue
            head = self.heads[token_type]
            h, y = hidden_flat[mask], target_flat[mask]

            if spec.family == "binary":
                loss_total = loss_total + F.binary_cross_entropy_with_logits(
                    head(h).squeeze(-1), y.float(), reduction="sum")
            elif spec.family == "continuous":
                loss_total = loss_total + discrete_logistic_nll(head(h), y).sum()
            else:
                local_vocab = vocab_flat[mask]
                for vs in local_vocab.unique():
                    vm = local_vocab == vs
                    v = int(vs.item())
                    loss_total = loss_total + F.cross_entropy(
                        head(h[vm])[:, :v], y[vm].clamp(0, v - 1), reduction="sum")
            token_total += int(mask.sum().item())

        if token_total == 0:
            return hidden.new_zeros(()), 0.0

        loss = loss_total / token_total
        bpt = float(loss.detach().item() / math.log(2))
        return loss, bpt


# -- eval ---------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, block_size, eval_iters, device):
    model.eval()
    out = {}
    for name, traces in [("train", dataset.train), ("val", dataset.val)]:
        bpts = []
        for _ in range(eval_iters):
            batch = make_batch(traces, batch_size, block_size,
                               dataset.pad_type_id, dataset.bos_value_id, device)
            _, bpt = model(batch)
            bpts.append(bpt)
        out[name] = sum(bpts) / len(bpts)
    model.train()
    return out


def compute_uniform_bpt(traces):
    total_bits = total_tokens = 0
    for trace in traces:
        vocab_sizes = trace.vocab_sizes[trace.learn_mask]
        for vs in vocab_sizes.tolist():
            total_bits += 1.0 if vs == 2 else math.log2(vs) if vs > 2 else OPEN_VOCAB_UNIFORM_BITS
        total_tokens += int(vocab_sizes.numel())
    return total_bits / max(total_tokens, 1)


def compute_unigram_bpt(train, val, id_to_type):
    counts = {tt: Counter() for tt in id_to_type}
    totals = {tt: 0 for tt in id_to_type}
    for trace in train:
        for tid, target in zip(trace.type_ids[trace.learn_mask].tolist(),
                               trace.targets[trace.learn_mask].tolist()):
            counts[id_to_type[tid]][target] += 1
            totals[id_to_type[tid]] += 1

    total_bits = total_tokens = 0
    for trace in val:
        for tid, target in zip(trace.type_ids[trace.learn_mask].tolist(),
                               trace.targets[trace.learn_mask].tolist()):
            tt = id_to_type[tid]
            c = counts[tt]
            support = max(len(c), 1)
            prob = (c[target] + 1) / (totals[tt] + support)
            total_bits -= math.log2(prob)
            total_tokens += 1
    return total_bits / max(total_tokens, 1)


# -- main ---------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    # -- W&B --
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f"ctp-{args.num_videos}v-{args.n_layer}L-{args.n_embd}d"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # -- step 1: load frames --
    if args.lance_uri:
        print(f"[1] loading Lance manifest from {args.lance_uri}")
        print(f"    datasets={args.datasets}")
        manifest = load_manifest_from_lance(args.lance_uri, args.datasets, args.s3_endpoint)
        n_tars = len(manifest)
        n_members = sum(len(v) for v in manifest.values())
        print(f"    {n_tars} tars, {n_members:,} members")

        print(f"    sampling {args.num_videos} videos, {args.frames_per_video} frames each -> {args.width}x{args.height}")
        frames, sources = load_frames_from_lance(
            manifest, args.num_videos, args.frames_per_video,
            args.width, args.height, args.fps,
            args.s3_bucket, args.s3_endpoint)
        source_label = f"lance:{','.join(args.datasets)} ({len(sources)} videos)"
    else:
        print(f"[1] loading {args.frames_per_video} frames from {args.video}")
        frames = load_video_frames(args.video, args.frames_per_video,
                                   args.width, args.height, args.fps)
        source_label = args.video.name

    print(f"    {len(frames)} frames loaded")

    # -- step 2: encode symbol traces --
    print("[2] encoding symbol traces")
    t0 = time.monotonic()
    traces = encode_symbol_traces(frames, args.qp, args.max_cu_size)
    encode_time = time.monotonic() - t0
    total_tokens = sum(len(tr) for tr in traces)
    print(f"    {len(traces)} traces, {total_tokens:,} tokens in {encode_time:.1f}s")

    # -- step 3: build dataset --
    print("[3] building train/val split")
    dataset = build_dataset(traces, args.width, args.height,
                            args.val_fraction, args.seed, source_label)

    model_config = ModelConfig(
        block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout,
        type_vocab_size=dataset.pad_type_id + 1,
        value_vocab_size=dataset.value_vocab_size)
    model = CTPModel(model_config, dataset.head_specs, dataset.type_to_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

    uniform_bpt = compute_uniform_bpt(dataset.val)
    unigram_bpt = compute_unigram_bpt(dataset.train, dataset.val, dataset.id_to_type)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 72)
    print("ctp trainer")
    print("=" * 72)
    print(f"source            : {dataset.source}")
    print(f"device            : {device}")
    print(f"frames            : {len(traces)}")
    print(f"tokens            : {dataset.total_tokens:,}")
    print(f"train/val traces  : {len(dataset.train)} / {len(dataset.val)}")
    print(f"model params      : {n_params:,}")
    print(f"value bucket size : {dataset.value_vocab_size} (limit=+/-{dataset.value_limit})")
    print(f"head specs        : {len(dataset.head_specs)} types")
    print(f"\nbaselines")
    print(f"{'uniform':<16} {uniform_bpt:>8.3f} bpt")
    print(f"{'unigram':<16} {unigram_bpt:>8.3f} bpt")

    if wandb_run:
        wandb_run.summary["uniform_bpt"] = uniform_bpt
        wandb_run.summary["unigram_bpt"] = unigram_bpt
        wandb_run.summary["n_params"] = n_params
        wandb_run.summary["total_tokens"] = dataset.total_tokens
        wandb_run.summary["n_frames"] = len(traces)
        wandb_run.summary["encode_time_s"] = encode_time

    # -- step 4: train --
    print(f"\ntraining ({args.max_iters} iters, bs={args.batch_size}, ctx={args.block_size})")
    best_val = float("inf")
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            metrics = estimate_loss(model, dataset, args.batch_size,
                                    args.block_size, args.eval_iters, device)
            best_val = min(best_val, metrics["val"])
            print(f"step {step:>5d} | train {metrics['train']:.3f} bpt | val {metrics['val']:.3f} bpt")
            if wandb_run:
                wandb_run.log({"step": step,
                               "train/bpt": metrics["train"],
                               "val/bpt": metrics["val"],
                               "val/best_bpt": best_val})

        if step == args.max_iters:
            break

        batch = make_batch(dataset.train, args.batch_size, args.block_size,
                           dataset.pad_type_id, dataset.bos_value_id, device)
        loss, train_bpt = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if wandb_run and step % 10 == 0:
            wandb_run.log({"step": step, "train/loss": loss.item(),
                           "train/bpt_step": train_bpt})

    # -- results --
    print(f"\n{'model':<18} {'BPT':>7}  {'vs uniform':>10}")
    print(f"{'─' * 38}")
    print(f"{'uniform':<18} {uniform_bpt:>7.3f}  {'baseline':>10}")
    print(f"{'unigram':<18} {unigram_bpt:>7.3f}  {(1 - unigram_bpt / uniform_bpt) * 100:>9.1f}%")
    print(f"{'CTPModel':<18} {best_val:>7.3f}  {(1 - best_val / uniform_bpt) * 100:>9.1f}%")
    print(f"{'─' * 38}")
    print(f"{n_params:,} params | source: {source_label}")

    if wandb_run:
        wandb_run.summary["best_val_bpt"] = best_val
        wandb_run.finish()

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"poc": t.poc, "cu_x": t.cu_x, "cu_y": t.cu_y,
                     "cu_size": t.cu_size, "plane": t.plane,
                     "token_type": t.token_type, "value": t.value,
                     "vocab_size": t.vocab_size} for t in traces[0]]
        args.trace_out.write_text(json.dumps(payload, indent=2))
        print(f"\ntrace json: {args.trace_out}")


if __name__ == "__main__":
    main()
