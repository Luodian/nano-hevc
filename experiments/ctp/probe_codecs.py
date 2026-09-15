"""Probe codec types in howto100m videos using torchcodec metadata."""
import io
import os
import random
import boto3
import lance
import torch
from botocore.config import Config

client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                      config=Config(connect_timeout=10, read_timeout=60))

ds = lance.dataset("/mnt/data/tables/s3_media_primary.lance")
rows = ds.to_table(
    columns=["tar_path", "member_path", "offset", "size"],
    filter="tar_path LIKE '%/tars/howto100m/%'",
    limit=500,
).to_pylist()

videos = [r for r in rows if r["member_path"].endswith((".mp4", ".webm", ".mkv"))]
random.shuffle(videos)
videos = videos[:30]

codec_counts = {}
for v in videos:
    try:
        resp = client.get_object(
            Bucket="ami-labs-data", Key=v["tar_path"],
            Range=f"bytes={v['offset']}-{v['offset'] + v['size'] - 1}")
        raw = resp["Body"].read()
    except Exception as e:
        print(f"  FETCH FAIL: {v['member_path']} - {e}")
        continue

    codec = "unknown"
    try:
        from torchcodec.decoders import VideoDecoder
        raw_t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        dec = VideoDecoder(raw_t)
        m = dec.metadata
        codec = getattr(m, "codec", None) or "unknown"
        nf = m.num_frames or 0
        fps = m.average_fps or 0
        print(f"  {v['member_path']:55s}  codec={codec:10s}  frames={nf:5d}  fps={fps:5.1f}  {v['size']/1e6:.1f}MB")
    except Exception:
        # fallback to av
        try:
            import av
            c = av.open(io.BytesIO(raw))
            s = c.streams.video[0]
            codec = s.codec_context.name or "unknown"
            nf = s.frames or 0
            fps = float(s.average_rate or 0)
            c.close()
            print(f"  {v['member_path']:55s}  codec={codec:10s}  frames={nf:5d}  fps={fps:5.1f}  {v['size']/1e6:.1f}MB")
        except Exception as e:
            print(f"  DECODE FAIL: {v['member_path']} - {e}")
            continue

    codec_counts[codec] = codec_counts.get(codec, 0) + 1

print(f"\ncodec distribution ({sum(codec_counts.values())} videos):")
for codec, count in sorted(codec_counts.items(), key=lambda x: -x[1]):
    print(f"  {codec:10s} {count:3d} ({100*count/sum(codec_counts.values()):.0f}%)")
