#!/usr/bin/env python3
"""
Minimal CTP trainer in the spirit of nanoGPT.

Pipeline:
  1. Load video (local file or Lance-indexed S3 tars from howto100m)
  2. Extract H.264/MPEG-4 token sequences from bitstream (via PyAV MV export)
     fallback: RGB -> YUV420 -> nano-hevc re-encode (trace=True), slower
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
import threading
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
    p.add_argument(
        "--lance-uri",
        type=str,
        default=None,
        help="Lance table URI for S3 tar-backed loading",
    )
    p.add_argument("--datasets", nargs="+", default=["howto100m"])
    p.add_argument("--s3-bucket", default="ami-labs-data")
    p.add_argument("--s3-endpoint", default=None)
    p.add_argument(
        "--num-videos", type=int, default=3, help="videos to sample from Lance"
    )

    # encoder
    p.add_argument("--codec-mode", choices=["hevc", "mpeg4"], default="mpeg4",
                   help="hevc=re-encode with nano-hevc (slow), mpeg4=extract from bitstream (fast)")
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
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear LR warmup steps (0=disabled)")
    p.add_argument("--max-grad-norm", type=float, default=0.0,
                   help="gradient clipping norm (0=disabled)")
    p.add_argument("--amp", action="store_true",
                   help="use bf16 automatic mixed precision")

    # streaming (full dataset training)
    p.add_argument("--streaming", action="store_true",
                   help="stream videos from Lance/S3 instead of loading all upfront")
    p.add_argument("--streaming-workers", type=int, default=8,
                   help="DataLoader workers for streaming")
    p.add_argument("--warmup-frames", type=int, default=32,
                   help="frames to encode for head_spec discovery")
    p.add_argument("--eval-videos", type=int, default=50,
                   help="held-out videos for eval")

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
    p.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="optional JSON dump for the first trace",
    )
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
    return [
        _resize_and_convert(all_frames[min(i, len(all_frames) - 1)], width, height)
        for i in indices
    ]


# -- frame loading: Lance manifest + S3 range reads ---------------------------


def load_manifest_from_lance(lance_uri, datasets=None, s3_endpoint=None):
    """Scan Lance table -> {tar_key: [(member_path, offset, size), ...]}."""
    import lance
    from collections import defaultdict

    kw = {}
    if lance_uri.startswith("s3://"):
        endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            opts = {
                "endpoint": endpoint,
                "allow_http": "true" if endpoint.startswith("http://") else "false",
            }
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

    scan_kw = {
        "columns": ["tar_path", "member_path", "offset", "size"],
        "batch_size": 65536,
    }
    if lance_filter:
        scan_kw["filter"] = lance_filter

    manifest = defaultdict(list)
    for batch in ds.to_batches(**scan_kw):
        for tp, mp, off, sz in zip(
            batch["tar_path"].to_pylist(),
            batch["member_path"].to_pylist(),
            batch["offset"].to_pylist(),
            batch["size"].to_pylist(),
            strict=False,
        ):
            manifest[tp].append((mp, off, sz))
    return dict(manifest)


def load_frames_from_lance(
    manifest, num_videos, num_frames, width, height, fps, s3_bucket, s3_endpoint
):
    """Sample random videos from manifest, decode via S3 range reads."""
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
    kw = {
        "config": BotoConfig(
            connect_timeout=30, read_timeout=300, retries={"max_attempts": 3}
        )
    }
    if endpoint:
        kw["endpoint_url"] = endpoint
    client = boto3.client("s3", **kw)

    video_members = [
        (tk, mp, off, sz)
        for tk, members in manifest.items()
        for mp, off, sz in members
        if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS
    ]
    if not video_members:
        raise RuntimeError("no video members found in manifest")

    random.shuffle(video_members)
    selected = video_members[:num_videos]

    all_frames = []
    sources = []
    for tar_key, member_path, offset, size in selected:
        print(f"    fetching {member_path} ({size / 1e6:.1f} MB)")
        try:
            resp = client.get_object(
                Bucket=s3_bucket,
                Key=tar_key,
                Range=f"bytes={offset}-{offset + size - 1}",
            )
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


def _ffmpeg_threads(num_workers=1):
    """Threads per decoder, scaled to CPUs/workers (matching LWM pattern)."""
    cpus = os.cpu_count() or 16
    return max(1, cpus // max(num_workers, 1))


def _decode_video_bytes(raw_bytes, num_frames, width, height, fps, num_workers=1):
    """Decode video from raw bytes. Priority: torchcodec -> decord -> pyav."""
    try:
        return _decode_bytes_torchcodec(raw_bytes, num_frames, width, height, fps, num_workers)
    except (ImportError, Exception):
        pass
    try:
        return _decode_bytes_decord(raw_bytes, num_frames, width, height, fps, num_workers)
    except ImportError:
        pass
    return _decode_bytes_pyav(raw_bytes, num_frames, width, height, fps)


def _decode_bytes_torchcodec(raw_bytes, num_frames, width, height, fps, num_workers=1):
    """Fastest: torchcodec with multithreaded FFmpeg, direct from bytes."""
    from torchcodec.decoders import VideoDecoder

    raw_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    decoder = VideoDecoder(raw_tensor, num_ffmpeg_threads=_ffmpeg_threads(num_workers))
    metadata = decoder.metadata
    total = metadata.num_frames
    if total is None or total < 2:
        return None

    video_fps = metadata.average_fps or 30.0
    indices = _pick_indices(total, video_fps, num_frames, fps)

    frame_batch = decoder.get_frames_at(indices=indices)
    # frame_batch.data is (N, C, H, W) uint8
    rgbs = frame_batch.data.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
    return [_resize_and_convert(rgb, width, height) for rgb in rgbs]


def _decode_bytes_decord(raw_bytes, num_frames, width, height, fps, num_workers=1):
    from decord import VideoReader, cpu

    vr = VideoReader(io.BytesIO(raw_bytes), ctx=cpu(0),
                     num_threads=_ffmpeg_threads(num_workers))
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
    return [
        _resize_and_convert(all_frames[min(i, len(all_frames) - 1)], width, height)
        for i in indices
    ]


# -- shared frame utilities ---------------------------------------------------


def _pick_indices(total, video_fps, num_frames, target_fps):
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


def _rgb_to_yuv420(rgb, w, h):
    r, g, b = (
        rgb[:, :, 0].astype(np.float32),
        rgb[:, :, 1].astype(np.float32),
        rgb[:, :, 2].astype(np.float32),
    )
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 128.0
    np.clip(y, 0, 255, out=y)
    np.clip(cb, 0, 255, out=cb)
    np.clip(cr, 0, 255, out=cr)
    u = (cb[0::2, 0::2] + cb[0::2, 1::2] + cb[1::2, 0::2] + cb[1::2, 1::2]) * 0.25
    v = (cr[0::2, 0::2] + cr[0::2, 1::2] + cr[1::2, 0::2] + cr[1::2, 1::2]) * 0.25
    frame = Frame.zeros(h, w, dtype=np.int16)
    frame.y.data[:] = y.astype(np.int16)
    frame.u.data[:] = u.astype(np.int16)
    frame.v.data[:] = v.astype(np.int16)
    return frame


# -- mpeg-4 token extraction (direct from bitstream, no re-encode) ------------


_MV_BINS = 256
_MV_BIN_EDGES = None  # lazily built


def _mv_to_bin(val):
    """Nonlinear MV quantization: fine bins near zero, coarse in tails.

    Layout (256 bins):
      bins 0-127:   values [-64, +63] at unit resolution (quarter-pel)
      bins 128-191: values [-192, -65] and [64, 191] at 2x step
      bins 192-255: values [-704, -193] and [192, 703] at 8x step
    """
    if -64 <= val <= 63:
        return val + 64  # bins 0-127
    if -192 <= val < -64:
        return 128 + (val + 192) // 2  # bins 128-191 (lower half)
    if 64 <= val <= 191:
        return 128 + (val - 64) // 2  # bins 128-191 (upper half, overlaps intentionally)
    if val < -192:
        b = 192 + (val + 704) // 8
    else:
        b = 192 + (val - 192) // 8 + 32
    return max(0, min(_MV_BINS - 1, b))


def extract_mpeg4_traces_from_bytes(raw_bytes, num_frames, fps=8):
    """Extract codec tokens directly from MPEG-4/VP8 video bytes via PyAV.

    Uses FFmpeg's motion vector export to get per-macroblock codec decisions.
    ~50ms/video vs ~16s for HEVC re-encode.

    Token vocabulary:
      frame_type     - I/P/B (vocab=3), 1 per frame
      mb_mode        - intra=0, inter=1, skip=2 (vocab=3), 1 per MB
      partition_mode - 16x16=0, 16x8=1, 8x16=2, 8x8=3 (vocab=4), inter MBs only
      mv_x, mv_y    - nonlinear 256-bin discretized (vocab=256), per sub-partition

    Per-MB token order: mb_mode -> [partition_mode -> mv_x -> mv_y ...]
    Skip/intra MBs: mb_mode only (variable-length, no padding).

    Returns list[list[SymbolTrace]], one trace per frame.
    """
    import av

    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}

    total = stream.frames or 0
    video_fps = float(stream.average_rate or 30)

    decoded = []
    ptype_names = {1: "I", 2: "P", 3: "B"}
    for frame in container.decode(video=0):
        ptype = ptype_names.get(int(frame.pict_type), "I")
        mvs = None
        for sd in frame.side_data:
            if "MOTION" in str(sd.type).upper():
                mvs = sd
                break
        w, h = frame.width, frame.height
        decoded.append((ptype, mvs, w, h))
    container.close()

    if not decoded:
        return None
    if not total:
        total = len(decoded)

    indices = _pick_indices(total, video_fps, num_frames, fps)
    ptype_map = {"I": 0, "P": 1, "B": 2}

    all_traces = []
    for poc_out, idx in enumerate(indices):
        if idx >= len(decoded):
            idx = len(decoded) - 1
        ptype, mvs, w, h = decoded[idx]
        frame_type_val = ptype_map.get(ptype, 0)
        mb_w = (w + 15) // 16
        mb_h = (h + 15) // 16

        # collect ALL MVs per MB (not just the first)
        from collections import defaultdict
        mb_mvs = defaultdict(list)
        if mvs is not None:
            for mv in mvs:
                mbx = mv.dst_x // 16
                mby = mv.dst_y // 16
                mb_mvs[(mbx, mby)].append(mv)

        trace = []
        trace.append(SymbolTrace(
            poc=poc_out, cu_x=0, cu_y=0, cu_size=16,
            plane="Y", token_type="frame_type", value=frame_type_val, vocab_size=3))

        for mby in range(mb_h):
            for mbx in range(mb_w):
                x, y = mbx * 16, mby * 16
                mv_list = mb_mvs.get((mbx, mby), [])

                if frame_type_val == 0:
                    mode = 0  # intra (I-frame)
                elif mv_list:
                    mode = 1  # inter
                else:
                    mode = 2  # skip

                trace.append(SymbolTrace(
                    poc=poc_out, cu_x=x, cu_y=y, cu_size=16,
                    plane="Y", token_type="mb_mode", value=mode, vocab_size=3))

                if mode != 1 or not mv_list:
                    continue

                # infer partition mode from MV block sizes
                widths = {mv.w for mv in mv_list}
                heights = {mv.h for mv in mv_list}
                n_mvs = len(mv_list)
                if n_mvs >= 4 or (8 in widths and 8 in heights):
                    part = 3  # 8x8
                elif 8 in heights and 16 in widths:
                    part = 1  # 16x8
                elif 8 in widths and 16 in heights:
                    part = 2  # 8x16
                else:
                    part = 0  # 16x16

                trace.append(SymbolTrace(
                    poc=poc_out, cu_x=x, cu_y=y, cu_size=16,
                    plane="Y", token_type="partition_mode", value=part, vocab_size=4))

                # emit MV pairs for each sub-partition
                # sort by position (top-left first) for consistent ordering
                sorted_mvs = sorted(mv_list, key=lambda m: (m.dst_y, m.dst_x))
                # cap at expected count for this partition mode
                max_mvs = [1, 2, 2, 4][part]
                for mv in sorted_mvs[:max_mvs]:
                    trace.append(SymbolTrace(
                        poc=poc_out, cu_x=x, cu_y=y, cu_size=16,
                        plane="Y", token_type="mv_x",
                        value=_mv_to_bin(mv.motion_x), vocab_size=_MV_BINS))
                    trace.append(SymbolTrace(
                        poc=poc_out, cu_x=x, cu_y=y, cu_size=16,
                        plane="Y", token_type="mv_y",
                        value=_mv_to_bin(mv.motion_y), vocab_size=_MV_BINS))

        all_traces.append(trace)
    return all_traces


def extract_mpeg4_traces_from_video(video_path, num_frames, fps=8):
    """Extract MPEG-4 traces from a local video file."""
    with open(str(video_path), "rb") as f:
        raw_bytes = f.read()
    return extract_mpeg4_traces_from_bytes(raw_bytes, num_frames, fps)


# -- encode -> traces (HEVC re-encode path) -----------------------------------


def encode_symbol_traces(frames, qp, max_cu_size, frames_per_video=None):
    config = HEVCConfig(
        width=frames[0].width, height=frames[0].height, qp=qp, max_cu_size=max_cu_size
    )
    traces = []
    ref_frames = []
    for poc, frame in enumerate(frames):
        # reset references at video boundaries for multi-video runs
        if frames_per_video and poc % frames_per_video == 0:
            ref_frames = []
        frame_type = "I" if not ref_frames else "P"
        refs = ref_frames[-1:] if frame_type != "I" else None
        _, recon, stats = encode_frame(
            frame,
            config,
            fast_mode=True,
            trace=True,
            poc=poc,
            reference_frames=refs,
            frame_type=frame_type,
        )
        traces.append(stats["symbol_trace"])
        ref_frames.append(recon)
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
        type_ids = torch.tensor(
            [type_to_id[t.token_type] for t in trace], dtype=torch.long
        )
        targets = torch.tensor([t.value for t in trace], dtype=torch.long)
        vocab_sizes = torch.tensor([t.vocab_size for t in trace], dtype=torch.long)
        prev_value_ids = torch.full((n,), bos_value_id, dtype=torch.long)
        if n > 1:
            prev_value_ids[1:] = torch.tensor(
                [value_to_id(t.value) for t in trace[:-1]], dtype=torch.long
            )
        aux = torch.tensor(
            [
                [
                    t.poc / max_poc,
                    t.cu_x / max(width, 1),
                    t.cu_y / max(height, 1),
                    t.cu_size / max_cu_size,
                    PLANE_TO_ID.get(t.plane, 0) / 2.0,
                ]
                for t in trace
            ],
            dtype=torch.float32,
        )
        learn_mask = torch.tensor(
            [head_specs[t.token_type].learn for t in trace], dtype=torch.bool
        )
        tensor_traces.append(
            TensorTrace(
                type_ids=type_ids,
                prev_value_ids=prev_value_ids,
                targets=targets,
                vocab_sizes=vocab_sizes,
                aux=aux,
                learn_mask=learn_mask,
            )
        )

    rng = random.Random(seed)
    rng.shuffle(tensor_traces)

    if len(tensor_traces) == 1:
        train = val = tensor_traces
    else:
        split = int(round(len(tensor_traces) * (1.0 - val_fraction)))
        split = min(max(split, 1), len(tensor_traces) - 1)
        train, val = tensor_traces[:split], tensor_traces[split:]

    total_tokens = sum(t.length() for t in tensor_traces)
    return DatasetBundle(
        train=train,
        val=val,
        type_to_id=type_to_id,
        id_to_type=id_to_type,
        head_specs=head_specs,
        pad_type_id=pad_type_id,
        bos_value_id=bos_value_id,
        value_vocab_size=value_vocab_size,
        value_limit=value_limit,
        source=source,
        total_tokens=total_tokens,
    )


# -- streaming dataset --------------------------------------------------------


def tensorize_trace(trace, type_to_id, head_specs, value_limit, width, height):
    """Convert a raw SymbolTrace list into tensors. Shared by batch and streaming modes."""
    value_vocab_size = value_limit * 2 + 3
    bos_value_id = value_vocab_size - 1
    max_poc = max(1, max(t.poc for t in trace)) if trace else 1
    max_cu_size = max(1, max(t.cu_size for t in trace)) if trace else 8

    def value_to_id(v):
        return max(0, min(v + value_limit, value_vocab_size - 2))

    n = len(trace)
    type_ids = torch.zeros(n, dtype=torch.long)
    targets = torch.zeros(n, dtype=torch.long)
    vocab_sizes = torch.zeros(n, dtype=torch.long)
    prev_value_ids = torch.full((n,), bos_value_id, dtype=torch.long)
    aux = torch.zeros(n, 5, dtype=torch.float32)
    learn_mask = torch.zeros(n, dtype=torch.bool)

    for i, t in enumerate(trace):
        tid = type_to_id.get(t.token_type, -1)
        if tid < 0:
            continue
        type_ids[i] = tid
        targets[i] = t.value
        vocab_sizes[i] = t.vocab_size
        if i > 0:
            prev_value_ids[i] = value_to_id(trace[i - 1].value)
        aux[i] = torch.tensor([
            t.poc / max_poc, t.cu_x / max(width, 1), t.cu_y / max(height, 1),
            t.cu_size / max_cu_size, PLANE_TO_ID.get(t.plane, 0) / 2.0,
        ])
        learn_mask[i] = head_specs[t.token_type].learn if t.token_type in head_specs else False

    return type_ids, prev_value_ids, targets, vocab_sizes, aux, learn_mask


def _make_window(type_ids, prev_value_ids, targets, vocab_sizes, aux, learn_mask,
                 block_size, pad_type_id, bos_value_id):
    """Take a random window of block_size from tensorized trace. Returns a batch-ready dict."""
    n = type_ids.shape[0]
    if n > block_size:
        start = random.randint(0, n - block_size)
        sl = slice(start, start + block_size)
    else:
        start = 0
        sl = slice(0, n)

    out_type = torch.full((block_size,), pad_type_id, dtype=torch.long)
    out_prev = torch.full((block_size,), bos_value_id, dtype=torch.long)
    out_tgt = torch.zeros(block_size, dtype=torch.long)
    out_voc = torch.zeros(block_size, dtype=torch.long)
    out_aux = torch.zeros(block_size, 5, dtype=torch.float32)
    out_valid = torch.zeros(block_size, dtype=torch.bool)
    out_learn = torch.zeros(block_size, dtype=torch.bool)

    span = min(n, block_size)
    out_type[:span] = type_ids[sl]
    out_prev[:span] = prev_value_ids[sl]
    out_tgt[:span] = targets[sl]
    out_voc[:span] = vocab_sizes[sl]
    out_aux[:span] = aux[sl]
    out_valid[:span] = True
    out_learn[:span] = learn_mask[sl]

    return {
        "type_ids": out_type, "prev_value_ids": out_prev,
        "targets": out_tgt, "vocab_sizes": out_voc,
        "aux": out_aux, "valid_mask": out_valid, "learn_mask": out_learn,
    }


class CTPStreamDataset(torch.utils.data.IterableDataset):
    """Stream: S3 range read -> extract codec tokens -> tensorize -> yield windows.

    codec_mode="mpeg4": parse MPEG-4/VP8 bitstream directly (~50ms/video)
    codec_mode="hevc":  decode + re-encode with nano-hevc (~16s/video)
    """

    def __init__(self, manifest, s3_bucket, s3_endpoint,
                 width, height, frames_per_video, fps, qp, max_cu_size, block_size,
                 type_to_id, head_specs, value_limit, codec_mode="mpeg4", seed=42):
        super().__init__()
        self.video_members = [
            (tk, mp, off, sz)
            for tk, members in manifest.items()
            for mp, off, sz in members
            if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS
        ]
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint
        self.width = width
        self.height = height
        self.frames_per_video = frames_per_video
        self.fps = fps
        self.qp = qp
        self.max_cu_size = max_cu_size
        self.block_size = block_size
        self.type_to_id = type_to_id
        self.head_specs = head_specs
        self.value_limit = value_limit
        self.pad_type_id = len(type_to_id)
        self.bos_value_id = value_limit * 2 + 3 - 1
        self.codec_mode = codec_mode
        self.seed = seed
        self._local = threading.local()
        self._ok = 0
        self._fail = 0

    def _client(self):
        if not hasattr(self._local, "client"):
            import boto3
            from botocore.config import Config as BotoConfig
            kw = {"config": BotoConfig(connect_timeout=30, read_timeout=300,
                                        retries={"max_attempts": 3})}
            endpoint = self.s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
            if endpoint:
                kw["endpoint_url"] = endpoint
            self._local.client = boto3.client("s3", **kw)
        return self._local.client

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        num_workers = wi.num_workers if wi else 1

        members = list(self.video_members)
        n = len(members)
        chunk = n // num_workers
        start = worker_id * chunk
        end = start + chunk if worker_id < num_workers - 1 else n
        my_members = members[start:end]

        rng = random.Random(self.seed + worker_id)
        config = HEVCConfig(width=self.width, height=self.height,
                            qp=self.qp, max_cu_size=self.max_cu_size)

        while True:
            rng.shuffle(my_members)
            for tar_key, member_path, offset, size in my_members:
                try:
                    resp = self._client().get_object(
                        Bucket=self.s3_bucket, Key=tar_key,
                        Range=f"bytes={offset}-{offset + size - 1}")
                    video_bytes = resp["Body"].read()
                except Exception:
                    self._fail += 1
                    continue

                try:
                    if self.codec_mode == "mpeg4":
                        traces = extract_mpeg4_traces_from_bytes(
                            video_bytes, self.frames_per_video, self.fps)
                    else:
                        frames = _decode_video_bytes(
                            video_bytes, self.frames_per_video,
                            self.width, self.height, self.fps,
                            num_workers=num_workers)
                        if not frames:
                            self._fail += 1
                            continue
                        traces = []
                        ref_frames = []
                        for poc, frame in enumerate(frames):
                            ft = "I" if not ref_frames else "P"
                            refs = ref_frames[-1:] if ft != "I" else None
                            _, recon, stats = encode_frame(
                                frame, config, fast_mode=True, trace=True,
                                poc=poc, reference_frames=refs, frame_type=ft)
                            traces.append(stats["symbol_trace"])
                            ref_frames.append(recon)
                except Exception:
                    self._fail += 1
                    continue

                if not traces:
                    self._fail += 1
                    continue

                for trace in traces:
                    if not trace:
                        continue
                    tensors = tensorize_trace(
                        trace, self.type_to_id, self.head_specs,
                        self.value_limit, self.width, self.height)
                    window = _make_window(
                        *tensors, self.block_size,
                        self.pad_type_id, self.bos_value_id)
                    self._ok += 1
                    yield window


def warmup_discover_specs(manifest, num_warmup_frames, s3_bucket, s3_endpoint,
                          width, height, frames_per_video, fps, qp, max_cu_size,
                          codec_mode="mpeg4"):
    """Extract a small set of traces to discover head_specs and value_limit."""
    # sample more videos than strictly needed to handle decode failures
    num_videos = max(8, (num_warmup_frames // max(frames_per_video, 1)) * 3)

    if codec_mode == "mpeg4":
        # fetch a few videos, extract mpeg4 traces directly
        import boto3
        from botocore.config import Config as BotoConfig
        endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
        kw = {"config": BotoConfig(connect_timeout=30, read_timeout=300,
                                    retries={"max_attempts": 3})}
        if endpoint:
            kw["endpoint_url"] = endpoint
        client = boto3.client("s3", **kw)

        video_members = [
            (tk, mp, off, sz)
            for tk, members in manifest.items()
            for mp, off, sz in members
            if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS
        ]
        random.shuffle(video_members)
        traces = []
        n_ok = n_fail = 0
        for tk, mp, off, sz in video_members[:num_videos]:
            try:
                resp = client.get_object(Bucket=s3_bucket, Key=tk,
                                         Range=f"bytes={off}-{off + sz - 1}")
                raw = resp["Body"].read()
                vtraces = extract_mpeg4_traces_from_bytes(raw, frames_per_video, fps)
                if vtraces:
                    traces.extend(vtraces)
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
                continue
            if len(traces) >= num_warmup_frames:
                break
        print(f"    warmup: {n_ok} videos ok, {n_fail} failed, {len(traces)} traces")
    else:
        frames, _ = load_frames_from_lance(
            manifest, num_videos, frames_per_video,
            width, height, fps, s3_bucket, s3_endpoint)
        traces = encode_symbol_traces(frames, qp, max_cu_size, frames_per_video)

    # filter out empty traces
    traces = [tr for tr in traces if tr]
    if not traces:
        raise RuntimeError("warmup failed: no valid traces extracted. check S3 access and video formats.")

    head_specs = build_head_specs(traces)
    all_values = [abs(t.value) for tr in traces for t in tr]
    value_limit = max(256, max(all_values)) if all_values else 256
    return head_specs, value_limit, traces


def build_eval_pool(manifest, num_eval_videos, type_to_id, head_specs, value_limit,
                    s3_bucket, s3_endpoint, width, height, frames_per_video,
                    fps, qp, max_cu_size, seed, codec_mode="mpeg4"):
    """Pre-extract traces from a held-out set of videos for evaluation."""
    if codec_mode == "mpeg4":
        import boto3
        from botocore.config import Config as BotoConfig
        endpoint = s3_endpoint or os.environ.get("S3_ENDPOINT_URL")
        kw = {"config": BotoConfig(connect_timeout=30, read_timeout=300,
                                    retries={"max_attempts": 3})}
        if endpoint:
            kw["endpoint_url"] = endpoint
        client = boto3.client("s3", **kw)

        video_members = [
            (tk, mp, off, sz)
            for tk, members in manifest.items()
            for mp, off, sz in members
            if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS
        ]
        rng = random.Random(seed + 99999)
        rng.shuffle(video_members)
        traces = []
        for tk, mp, off, sz in video_members[:num_eval_videos]:
            try:
                resp = client.get_object(Bucket=s3_bucket, Key=tk,
                                         Range=f"bytes={off}-{off + sz - 1}")
                raw = resp["Body"].read()
                vtraces = extract_mpeg4_traces_from_bytes(raw, frames_per_video, fps)
                if vtraces:
                    traces.extend(vtraces)
            except Exception:
                continue
    else:
        frames, _ = load_frames_from_lance(
            manifest, num_eval_videos, frames_per_video,
            width, height, fps, s3_bucket, s3_endpoint)
        traces = encode_symbol_traces(frames, qp, max_cu_size, frames_per_video)

    id_to_type = list(head_specs.keys())
    pad_type_id = len(id_to_type)
    value_vocab_size = value_limit * 2 + 3
    bos_value_id = value_vocab_size - 1

    tensor_traces = []
    for trace in traces:
        tensors = tensorize_trace(trace, type_to_id, head_specs, value_limit, width, height)
        tensor_traces.append(TensorTrace(*tensors))

    return DatasetBundle(
        train=tensor_traces, val=tensor_traces,
        type_to_id=type_to_id, id_to_type=id_to_type,
        head_specs=head_specs, pad_type_id=pad_type_id,
        bos_value_id=bos_value_id, value_vocab_size=value_vocab_size,
        value_limit=value_limit, source="eval_pool",
        total_tokens=sum(t.length() for t in tensor_traces))


# -- batching -----------------------------------------------------------------


def make_batch(traces, batch_size, block_size, pad_type_id, bos_value_id, device):
    aux_dim = traces[0].aux.shape[-1]
    type_ids = torch.full((batch_size, block_size), pad_type_id, dtype=torch.long)
    prev_value_ids = torch.full(
        (batch_size, block_size), bos_value_id, dtype=torch.long
    )
    targets = torch.zeros((batch_size, block_size), dtype=torch.long)
    vocab_sizes = torch.zeros((batch_size, block_size), dtype=torch.long)
    aux = torch.zeros((batch_size, block_size, aux_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)
    learn_mask = torch.zeros((batch_size, block_size), dtype=torch.bool)

    for row in range(batch_size):
        trace = traces[random.randrange(len(traces))]
        length = trace.length()
        start = (
            0 if length <= block_size else random.randrange(0, length - block_size + 1)
        )
        stop = min(length, start + block_size)
        span = stop - start
        type_ids[row, :span] = trace.type_ids[start:stop]
        prev_value_ids[row, :span] = trace.prev_value_ids[start:stop]
        targets[row, :span] = trace.targets[start:stop]
        vocab_sizes[row, :span] = trace.vocab_sizes[start:stop]
        aux[row, :span] = trace.aux[start:stop]
        valid_mask[row, :span] = True
        learn_mask[row, :span] = trace.learn_mask[start:stop]

    return {
        k: v.to(device)
        for k, v in {
            "type_ids": type_ids,
            "prev_value_ids": prev_value_ids,
            "targets": targets,
            "vocab_sizes": vocab_sizes,
            "aux": aux,
            "valid_mask": valid_mask,
            "learn_mask": learn_mask,
        }.items()
    }


# -- model --------------------------------------------------------------------


def discrete_logistic_nll(params, targets):
    mu, log_scale = params.unbind(dim=-1)
    scale = log_scale.exp().clamp(min=0.01, max=50.0)
    x = targets.float()
    prob = (
        torch.sigmoid((x + 0.5 - mu) / scale) - torch.sigmoid((x - 0.5 - mu) / scale)
    ).clamp(min=1e-9)
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
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(config.n_embd, spec.out_dim)
                for name, spec in head_specs.items()
            }
        )

    def forward(self, batch):
        type_ids = batch["type_ids"]
        B, T = type_ids.shape
        positions = torch.arange(T, device=type_ids.device)
        x = (
            self.type_emb(type_ids)
            + self.prev_value_emb(batch["prev_value_ids"])
            + self.pos_emb(positions)[None]
            + self.aux_proj(batch["aux"])
        )
        x = self.drop(x)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
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
                    head(h).squeeze(-1), y.float(), reduction="sum"
                )
            elif spec.family == "continuous":
                loss_total = loss_total + discrete_logistic_nll(head(h), y).sum()
            else:
                local_vocab = vocab_flat[mask]
                for vs in local_vocab.unique():
                    vm = local_vocab == vs
                    v = int(vs.item())
                    loss_total = loss_total + F.cross_entropy(
                        head(h[vm])[:, :v], y[vm].clamp(0, v - 1), reduction="sum"
                    )
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
            batch = make_batch(
                traces,
                batch_size,
                block_size,
                dataset.pad_type_id,
                dataset.bos_value_id,
                device,
            )
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
            total_bits += (
                1.0 if vs == 2 else math.log2(vs) if vs > 2 else OPEN_VOCAB_UNIFORM_BITS
            )
        total_tokens += int(vocab_sizes.numel())
    return total_bits / max(total_tokens, 1)


def compute_unigram_bpt(train, val, id_to_type):
    counts = {tt: Counter() for tt in id_to_type}
    totals = {tt: 0 for tt in id_to_type}
    for trace in train:
        for tid, target in zip(
            trace.type_ids[trace.learn_mask].tolist(),
            trace.targets[trace.learn_mask].tolist(),
        ):
            counts[id_to_type[tid]][target] += 1
            totals[id_to_type[tid]] += 1

    total_bits = total_tokens = 0
    for trace in val:
        for tid, target in zip(
            trace.type_ids[trace.learn_mask].tolist(),
            trace.targets[trace.learn_mask].tolist(),
        ):
            tt = id_to_type[tid]
            c = counts[tt]
            support = max(len(c), 1)
            prob = (c[target] + 1) / (totals[tt] + support)
            total_bits -= math.log2(prob)
            total_tokens += 1
    return total_bits / max(total_tokens, 1)


# -- streaming training -------------------------------------------------------


def _run_streaming(args, device, wandb_run):
    """Train on full howto100m via streaming: S3 fetch -> decode -> encode -> train."""
    use_amp = args.amp and device.type == "cuda"

    # step 1: load manifest
    print(f"[1] loading Lance manifest from {args.lance_uri}")
    manifest = load_manifest_from_lance(args.lance_uri, args.datasets, args.s3_endpoint)
    n_tars = len(manifest)
    n_members = sum(len(v) for v in manifest.values())
    n_videos = sum(1 for members in manifest.values()
                   for mp, _, _ in members
                   if os.path.splitext(mp)[1].lower() in _VIDEO_EXTS)
    print(f"    {n_tars} tars, {n_members:,} members, ~{n_videos:,} videos")

    # step 2: warmup — discover head specs from a small sample
    print(f"[2] warmup: encoding {args.warmup_frames} frames to discover token types")
    t0 = time.monotonic()
    head_specs, value_limit, warmup_traces = warmup_discover_specs(
        manifest, args.warmup_frames, args.s3_bucket, args.s3_endpoint,
        args.width, args.height, args.frames_per_video, args.fps,
        args.qp, args.max_cu_size, codec_mode=args.codec_mode)
    id_to_type = list(head_specs.keys())
    type_to_id = {name: idx for idx, name in enumerate(id_to_type)}
    pad_type_id = len(id_to_type)
    value_vocab_size = value_limit * 2 + 3
    bos_value_id = value_vocab_size - 1
    print(f"    {len(head_specs)} token types, value_limit={value_limit} in {time.monotonic() - t0:.1f}s")
    for name, spec in head_specs.items():
        print(f"      {name:<32} {spec.family:<12} out={spec.out_dim} learn={spec.learn}")

    # step 3: build eval pool from held-out videos
    print(f"[3] building eval pool: {args.eval_videos} videos")
    eval_bundle = build_eval_pool(
        manifest, args.eval_videos, type_to_id, head_specs, value_limit,
        args.s3_bucket, args.s3_endpoint, args.width, args.height,
        args.frames_per_video, args.fps, args.qp, args.max_cu_size, args.seed,
        codec_mode=args.codec_mode)
    print(f"    {eval_bundle.total_tokens:,} eval tokens")

    uniform_bpt = compute_uniform_bpt(eval_bundle.val)
    unigram_bpt = compute_unigram_bpt(eval_bundle.train, eval_bundle.val, id_to_type)

    # step 4: build model
    model_config = ModelConfig(
        block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout,
        type_vocab_size=pad_type_id + 1, value_vocab_size=value_vocab_size)
    model = CTPModel(model_config, head_specs, type_to_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())

    # step 5: build streaming DataLoader
    stream_dataset = CTPStreamDataset(
        manifest, args.s3_bucket, args.s3_endpoint,
        args.width, args.height, args.frames_per_video, args.fps,
        args.qp, args.max_cu_size, args.block_size,
        type_to_id, head_specs, value_limit,
        codec_mode=args.codec_mode, seed=args.seed)
    nw = args.streaming_workers
    loader = torch.utils.data.DataLoader(
        stream_dataset, batch_size=args.batch_size,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        multiprocessing_context="spawn" if nw > 0 else None)

    print("=" * 72)
    print("ctp streaming trainer")
    print("=" * 72)
    print(f"source            : lance:{','.join(args.datasets)} (~{n_videos:,} videos)")
    print(f"device            : {device}")
    print(f"model params      : {n_params:,}")
    print(f"amp               : {'bf16' if use_amp else 'off'}")
    print(f"warmup            : {args.warmup_steps} steps")
    print(f"streaming workers : {args.streaming_workers}")
    print(f"\nbaselines (eval pool)")
    print(f"{'uniform':<16} {uniform_bpt:>8.3f} bpt")
    print(f"{'unigram':<16} {unigram_bpt:>8.3f} bpt")

    if wandb_run:
        wandb_run.summary["uniform_bpt"] = uniform_bpt
        wandb_run.summary["unigram_bpt"] = unigram_bpt
        wandb_run.summary["n_params"] = n_params
        wandb_run.summary["n_videos_total"] = n_videos
        wandb_run.summary["streaming"] = True

    # step 6: train
    print(f"\ntraining ({args.max_iters} iters, bs={args.batch_size}, ctx={args.block_size})")
    data_iter = iter(loader)
    best_val = float("inf")
    t_start = time.monotonic()

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            metrics = estimate_loss(
                model, eval_bundle, args.batch_size,
                args.block_size, args.eval_iters, device)
            best_val = min(best_val, metrics["val"])
            elapsed = time.monotonic() - t_start
            print(f"step {step:>5d} | train {metrics['train']:.3f} bpt | val {metrics['val']:.3f} bpt | {elapsed:.0f}s")
            if wandb_run:
                wandb_run.log({"step": step, "train/bpt": metrics["train"],
                               "val/bpt": metrics["val"], "val/best_bpt": best_val})

        if step == args.max_iters:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            loss, train_bpt = model(batch)
        scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if wandb_run and step % 10 == 0:
            wandb_run.log({"step": step, "train/loss": loss.item(),
                           "train/bpt_step": train_bpt, "lr": scheduler.get_last_lr()[0]})

    print(f"\n{'model':<18} {'BPT':>7}  {'vs uniform':>10}")
    print(f"{'─' * 38}")
    print(f"{'uniform':<18} {uniform_bpt:>7.3f}  {'baseline':>10}")
    print(f"{'unigram':<18} {unigram_bpt:>7.3f}  {(1 - unigram_bpt / uniform_bpt) * 100:>9.1f}%")
    print(f"{'CTPModel':<18} {best_val:>7.3f}  {(1 - best_val / uniform_bpt) * 100:>9.1f}%")
    print(f"{'─' * 38}")
    print(f"{n_params:,} params | streaming from ~{n_videos:,} howto100m videos")

    if wandb_run:
        wandb_run.summary["best_val_bpt"] = best_val
        wandb_run.finish()


# -- main ---------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    # -- W&B --
    wandb_run = None
    if args.wandb:
        import wandb

        run_name = (
            args.wandb_run_name
            or f"ctp-{args.num_videos}v-{args.n_layer}L-{args.n_embd}d"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # -- streaming mode --
    if args.streaming:
        assert args.lance_uri, "--lance-uri required for --streaming"
        _run_streaming(args, device, wandb_run)
        return

    # -- batch mode: step 1 load frames --
    if args.lance_uri:
        print(f"[1] loading Lance manifest from {args.lance_uri}")
        print(f"    datasets={args.datasets}")
        manifest = load_manifest_from_lance(
            args.lance_uri, args.datasets, args.s3_endpoint
        )
        n_tars = len(manifest)
        n_members = sum(len(v) for v in manifest.values())
        print(f"    {n_tars} tars, {n_members:,} members")

        print(
            f"    sampling {args.num_videos} videos, {args.frames_per_video} frames each -> {args.width}x{args.height}"
        )
        frames, sources = load_frames_from_lance(
            manifest,
            args.num_videos,
            args.frames_per_video,
            args.width,
            args.height,
            args.fps,
            args.s3_bucket,
            args.s3_endpoint,
        )
        source_label = f"lance:{','.join(args.datasets)} ({len(sources)} videos)"
    else:
        print(f"[1] loading {args.frames_per_video} frames from {args.video}")
        frames = load_video_frames(
            args.video, args.frames_per_video, args.width, args.height, args.fps
        )
        source_label = args.video.name

    print(f"    {len(frames)} frames loaded")

    # -- step 2: extract/encode symbol traces --
    print(f"[2] extracting traces (codec_mode={args.codec_mode})")
    t0 = time.monotonic()
    if args.codec_mode == "mpeg4":
        traces = extract_mpeg4_traces_from_video(
            args.video, args.frames_per_video, args.fps)
        if traces is None:
            traces = []
    else:
        traces = encode_symbol_traces(
            frames, args.qp, args.max_cu_size, args.frames_per_video)
    encode_time = time.monotonic() - t0
    total_tokens = sum(len(tr) for tr in traces)
    print(f"    {len(traces)} traces, {total_tokens:,} tokens in {encode_time:.1f}s")

    # -- step 3: build dataset --
    print("[3] building train/val split")
    dataset = build_dataset(
        traces, args.width, args.height, args.val_fraction, args.seed, source_label
    )

    model_config = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        type_vocab_size=dataset.pad_type_id + 1,
        value_vocab_size=dataset.value_vocab_size,
    )
    model = CTPModel(model_config, dataset.head_specs, dataset.type_to_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR warmup schedule (constant after warmup, matching LWM)
    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP for bf16
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    print(
        f"value bucket size : {dataset.value_vocab_size} (limit=+/-{dataset.value_limit})"
    )
    print(f"head specs        : {len(dataset.head_specs)} types")
    print(f"amp               : {'bf16' if use_amp else 'off'}")
    print(f"warmup            : {args.warmup_steps} steps")
    print(f"grad clip         : {args.max_grad_norm or 'off'}")
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
    print(
        f"\ntraining ({args.max_iters} iters, bs={args.batch_size}, ctx={args.block_size})"
    )
    best_val = float("inf")
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            metrics = estimate_loss(
                model,
                dataset,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
            )
            best_val = min(best_val, metrics["val"])
            print(
                f"step {step:>5d} | train {metrics['train']:.3f} bpt | val {metrics['val']:.3f} bpt"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "step": step,
                        "train/bpt": metrics["train"],
                        "val/bpt": metrics["val"],
                        "val/best_bpt": best_val,
                    }
                )

        if step == args.max_iters:
            break

        batch = make_batch(
            dataset.train,
            args.batch_size,
            args.block_size,
            dataset.pad_type_id,
            dataset.bos_value_id,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            loss, train_bpt = model(batch)
        scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if wandb_run and step % 10 == 0:
            wandb_run.log(
                {"step": step, "train/loss": loss.item(), "train/bpt_step": train_bpt,
                 "lr": scheduler.get_last_lr()[0]}
            )

    # -- results --
    print(f"\n{'model':<18} {'BPT':>7}  {'vs uniform':>10}")
    print(f"{'─' * 38}")
    print(f"{'uniform':<18} {uniform_bpt:>7.3f}  {'baseline':>10}")
    print(
        f"{'unigram':<18} {unigram_bpt:>7.3f}  {(1 - unigram_bpt / uniform_bpt) * 100:>9.1f}%"
    )
    print(
        f"{'CTPModel':<18} {best_val:>7.3f}  {(1 - best_val / uniform_bpt) * 100:>9.1f}%"
    )
    print(f"{'─' * 38}")
    print(f"{n_params:,} params | source: {source_label}")

    if wandb_run:
        wandb_run.summary["best_val_bpt"] = best_val
        wandb_run.finish()

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
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
            for t in traces[0]
        ]
        args.trace_out.write_text(json.dumps(payload, indent=2))
        print(f"\ntrace json: {args.trace_out}")


if __name__ == "__main__":
    main()
