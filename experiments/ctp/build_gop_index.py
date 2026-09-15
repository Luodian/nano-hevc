#!/usr/bin/env python3
"""Build a GOP-level index from Panda70M fMP4 videos stored in S3 tar files.

Reads existing Lance manifest, fetches the first ~2KB of each video to parse
ftyp+moov+sidx, then writes a new Lance table with per-segment entries.
Finally benchmarks full-video download vs single-GOP range read.

Usage:
    python -m experiments.ctp.build_gop_index \
        --lance-uri /mnt/data/tables/s3_media_primary.lance \
        --datasets panda70m \
        --num-videos 5000 \
        --output /mnt/data/tables/panda70m_gop_index.lance
"""

import argparse
import io
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import lance
import pyarrow as pa
from botocore.config import Config as BotoConfig


# -- MP4 atom parsing --------------------------------------------------------

def parse_atoms(data, limit=None):
    """Yield (tag, offset, size) for top-level atoms in data."""
    off = 0
    end = len(data) if limit is None else min(limit, len(data))
    while off + 8 <= end:
        sz, tag = struct.unpack(">I4s", data[off : off + 8])
        tag = tag.decode("ascii", errors="replace")
        if sz == 1 and off + 16 <= end:
            sz = struct.unpack(">Q", data[off + 8 : off + 16])[0]
        elif sz == 0:
            sz = end - off
        if sz < 8:
            break
        yield tag, off, sz
        off += sz


def parse_moov_codec(moov_data):
    """Extract codec tag from moov box (avc1, hev1, vp09, etc.)."""
    for tag in [b"avc1", b"hev1", b"hvc1", b"vp09", b"av01"]:
        if tag in moov_data:
            return tag.decode()
    return "unknown"


def parse_sidx(data, offset, size):
    """Parse sidx box -> list of (ref_size, duration_ticks, sap_type)."""
    box = data[offset : offset + size]
    if len(box) < 12 or box[4:8] != b"sidx":
        return None, 0

    ver = box[8]
    # ref_id = struct.unpack(">I", box[12:16])[0]
    timescale = struct.unpack(">I", box[16:20])[0]

    if ver == 0:
        # earliest_pts(4) + first_offset(4) + reserved(2) + ref_count(2)
        ref_count = struct.unpack(">H", box[30:32])[0]
        entry_start = 32
    else:
        # earliest_pts(8) + first_offset(8) + reserved(2) + ref_count(2)
        ref_count = struct.unpack(">H", box[38:40])[0]
        entry_start = 40

    segments = []
    for i in range(ref_count):
        base = entry_start + i * 12
        if base + 12 > len(box):
            break
        w1 = struct.unpack(">I", box[base : base + 4])[0]
        ref_size = w1 & 0x7FFFFFFF
        dur = struct.unpack(">I", box[base + 4 : base + 8])[0]
        sap = struct.unpack(">I", box[base + 8 : base + 12])[0]
        sap_type = (sap >> 28) & 7
        segments.append((ref_size, dur, sap_type))

    return segments, timescale


def parse_fmp4_header(data):
    """Parse fMP4 header from raw bytes -> (codec, timescale, segments).

    segments: list of (offset_in_video, size, duration_sec)
    Returns None if not a valid fMP4.
    """
    atoms = list(parse_atoms(data))
    atom_map = {tag: (off, sz) for tag, off, sz in atoms}

    if "moov" not in atom_map or "sidx" not in atom_map:
        return None

    moov_off, moov_sz = atom_map["moov"]
    codec = parse_moov_codec(data[moov_off : moov_off + moov_sz])

    sidx_off, sidx_sz = atom_map["sidx"]
    segs_raw, timescale = parse_sidx(data, sidx_off, sidx_sz)
    if not segs_raw or timescale == 0:
        return None

    # compute absolute offsets: first segment starts right after sidx
    seg_start = sidx_off + sidx_sz
    segments = []
    for ref_size, dur_ticks, _sap_type in segs_raw:
        dur_sec = dur_ticks / timescale
        segments.append((seg_start, ref_size, dur_sec))
        seg_start += ref_size

    return codec, timescale, segments


# -- S3 helpers ---------------------------------------------------------------

def make_s3_client(endpoint=None):
    kw = {"config": BotoConfig(
        connect_timeout=10, read_timeout=30, retries={"max_attempts": 2},
        max_pool_connections=64,
    )}
    ep = endpoint or os.environ.get("S3_ENDPOINT_URL")
    if ep:
        kw["endpoint_url"] = ep
    return boto3.client("s3", **kw)


def fetch_header(client, bucket, tar_key, video_offset, video_size, header_bytes=4096):
    """Fetch the first header_bytes of a video inside a tar file."""
    end = video_offset + min(header_bytes, video_size) - 1
    resp = client.get_object(
        Bucket=bucket, Key=tar_key,
        Range=f"bytes={video_offset}-{end}",
    )
    return resp["Body"].read()


# -- main pipeline ------------------------------------------------------------

def load_panda70m_manifest(lance_uri, datasets, s3_endpoint=None):
    """Load video entries from Lance table."""
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
    preds = [f"source = '{d}'" for d in datasets]
    filt = " OR ".join(preds)

    rows = []
    for batch in ds.to_batches(
        columns=["path", "filename", "offset", "size"],
        filter=filt, batch_size=65536,
    ):
        for tp, mp, off, sz in zip(
            batch["path"].to_pylist(),
            batch["filename"].to_pylist(),
            batch["offset"].to_pylist(),
            batch["size"].to_pylist(),
            strict=False,
        ):
            ext = os.path.splitext(mp)[1].lower()
            if ext in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                rows.append((tp, mp, off, sz))
    return rows


def build_gop_index(videos, bucket, endpoint, num_threads=32):
    """Fetch headers in parallel, parse sidx, return GOP-level rows."""
    client = make_s3_client(endpoint)
    gop_rows = []
    stats = {"ok": 0, "no_sidx": 0, "err": 0, "bytes_fetched": 0}

    def process_one(entry):
        tar_key, member_path, video_offset, video_size = entry
        try:
            hdr = fetch_header(client, bucket, tar_key, video_offset, video_size)
            stats["bytes_fetched"] += len(hdr)
            result = parse_fmp4_header(hdr)
            if result is None:
                return ("no_sidx", entry, None)
            codec, timescale, segments = result
            return ("ok", entry, (codec, timescale, segments))
        except Exception as e:
            return ("err", entry, str(e))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(process_one, v): v for v in videos}
        for i, fut in enumerate(as_completed(futures)):
            status, entry, result = fut.result()
            tar_key, member_path, video_offset, video_size = entry

            if status == "ok":
                codec, timescale, segments = result
                for seg_idx, (seg_off, seg_size, seg_dur) in enumerate(segments):
                    gop_rows.append({
                        "tar_path": tar_key,
                        "member_path": member_path,
                        "video_offset": video_offset,
                        "video_size": video_size,
                        "codec": codec,
                        "seg_idx": seg_idx,
                        "seg_abs_offset": video_offset + seg_off,
                        "seg_size": seg_size,
                        "seg_duration": seg_dur,
                        "num_segments": len(segments),
                    })
                stats["ok"] += 1
            elif status == "no_sidx":
                stats["no_sidx"] += 1
            else:
                stats["err"] += 1

            if (i + 1) % 500 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(videos)}] {rate:.0f} videos/s  "
                      f"ok={stats['ok']} no_sidx={stats['no_sidx']} err={stats['err']}")

    elapsed = time.monotonic() - t0
    print(f"\nindex build done: {len(videos)} videos in {elapsed:.1f}s "
          f"({len(videos)/elapsed:.0f} videos/s)")
    print(f"  ok={stats['ok']}  no_sidx={stats['no_sidx']}  err={stats['err']}")
    print(f"  total header bytes fetched: {stats['bytes_fetched']/1024/1024:.1f} MB "
          f"(avg {stats['bytes_fetched']/max(1,len(videos)):.0f} B/video)")
    print(f"  GOP rows generated: {len(gop_rows)}")
    return gop_rows


def benchmark_gop_vs_full(gop_rows, videos_by_member, bucket, endpoint, num_trials=20):
    """Compare: download full video vs single GOP segment."""
    import random
    client = make_s3_client(endpoint)

    # pick random GOPs, prefer distinct videos
    rng = random.Random(42)
    seen_videos = set()
    candidates = []
    for r in gop_rows:
        if r["member_path"] not in seen_videos:
            seen_videos.add(r["member_path"])
            candidates.append(r)
    sample = rng.sample(candidates, min(num_trials, len(candidates)))

    full_times = []
    full_bytes = []
    gop_times = []
    gop_bytes = []

    for i, row in enumerate(sample):
        tar_key = row["tar_path"]
        v_off = row["video_offset"]
        v_sz = row["video_size"]
        seg_off = row["seg_abs_offset"]
        seg_sz = row["seg_size"]

        # full video download — cap at 50MB to keep benchmark sane,
        # record actual video_size for byte comparison
        cap = min(v_sz, 50 * 1024 * 1024)
        t0 = time.monotonic()
        resp = client.get_object(
            Bucket=bucket, Key=tar_key,
            Range=f"bytes={v_off}-{v_off + cap - 1}",
        )
        data = resp["Body"].read()
        full_times.append(time.monotonic() - t0)
        full_bytes.append(v_sz)  # record true video size

        # single GOP download (header + segment)
        t0 = time.monotonic()
        # fetch moov+sidx (first 4KB) + the target segment
        hdr_end = v_off + 4095
        resp_hdr = client.get_object(
            Bucket=bucket, Key=tar_key,
            Range=f"bytes={v_off}-{hdr_end}",
        )
        hdr_data = resp_hdr["Body"].read()

        resp_seg = client.get_object(
            Bucket=bucket, Key=tar_key,
            Range=f"bytes={seg_off}-{seg_off + seg_sz - 1}",
        )
        seg_data = resp_seg["Body"].read()
        gop_times.append(time.monotonic() - t0)
        gop_bytes.append(len(hdr_data) + len(seg_data))

        print(f"  trial {i+1}/{len(sample)}: "
              f"full={v_sz/1024/1024:.0f}MB({(time.monotonic()-t0)*1000:.0f}ms) "
              f"gop={seg_sz/1024:.0f}KB")

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n{'='*65}")
    print(f"benchmark: {len(sample)} trials")
    print(f"{'='*65}")
    print(f"{'':20s} {'full video':>18s} {'single GOP':>18s} {'ratio':>8s}")
    print(f"{'-'*65}")
    print(f"{'avg video size':20s} {avg(full_bytes)/1024/1024:>15.1f} MB "
          f"{avg(gop_bytes)/1024:>15.1f} KB"
          f" {avg(full_bytes)/max(avg(gop_bytes),1):>6.0f}x")
    print(f"{'avg time (50MB cap)':20s} {avg(full_times)*1000:>15.1f} ms "
          f"{avg(gop_times)*1000:>15.1f} ms"
          f" {avg(full_times)/max(avg(gop_times),1e-9):>6.1f}x")
    print(f"{'p50 time':20s} {sorted(full_times)[len(full_times)//2]*1000:>15.1f} ms "
          f"{sorted(gop_times)[len(gop_times)//2]*1000:>15.1f} ms")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Build GOP-level index for fMP4 videos")
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--datasets", nargs="+", default=["panda70m"])
    parser.add_argument("--s3-bucket", default="ami-labs-data")
    parser.add_argument("--s3-endpoint", default="https://storage.eu-north1.nebius.cloud")
    parser.add_argument("--num-videos", type=int, default=5000)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--output", default="/mnt/data/tables/panda70m_gop_index.lance")
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--benchmark-trials", type=int, default=20)
    args = parser.parse_args()

    # step 1: load manifest
    print(f"loading manifest from {args.lance_uri} ...")
    videos = load_panda70m_manifest(
        args.lance_uri, args.datasets, args.s3_endpoint)
    print(f"  found {len(videos)} video entries")

    if args.num_videos and args.num_videos < len(videos):
        import random
        random.Random(42).shuffle(videos)
        videos = videos[: args.num_videos]
        print(f"  sampled {len(videos)} videos")

    # step 2: build GOP index by fetching headers
    print(f"\nbuilding GOP index ({args.num_threads} threads) ...")
    gop_rows = build_gop_index(
        videos, args.s3_bucket, args.s3_endpoint, args.num_threads)

    if not gop_rows:
        print("no GOP rows generated, exiting")
        return

    # step 3: write Lance table
    print(f"\nwriting Lance table to {args.output} ...")
    schema = pa.schema([
        ("tar_path", pa.utf8()),
        ("member_path", pa.utf8()),
        ("video_offset", pa.int64()),
        ("video_size", pa.int64()),
        ("codec", pa.utf8()),
        ("seg_idx", pa.int32()),
        ("seg_abs_offset", pa.int64()),
        ("seg_size", pa.int64()),
        ("seg_duration", pa.float32()),
        ("num_segments", pa.int32()),
    ])
    arrays = {col: [r[col] for r in gop_rows] for col in schema.names}
    table = pa.table(arrays, schema=schema)
    lance.write_dataset(table, args.output, mode="overwrite")
    print(f"  wrote {len(gop_rows)} GOP entries from {len(set(r['member_path'] for r in gop_rows))} videos")

    # step 4: summary stats
    sizes = [r["seg_size"] for r in gop_rows]
    durs = [r["seg_duration"] for r in gop_rows]
    print(f"\nGOP segment stats:")
    print(f"  count: {len(sizes)}")
    print(f"  size:  median={sorted(sizes)[len(sizes)//2]/1024:.0f} KB  "
          f"mean={sum(sizes)/len(sizes)/1024:.0f} KB  "
          f"range=[{min(sizes)/1024:.0f}, {max(sizes)/1024:.0f}] KB")
    print(f"  dur:   median={sorted(durs)[len(durs)//2]:.2f}s  "
          f"mean={sum(durs)/len(durs):.2f}s")

    # step 5: benchmark
    if args.benchmark:
        videos_by_member = {mp: (tp, off, sz) for tp, mp, off, sz in videos}
        benchmark_gop_vs_full(
            gop_rows, videos_by_member,
            args.s3_bucket, args.s3_endpoint, args.benchmark_trials)


if __name__ == "__main__":
    main()
