"""Debug: check what torchcodec can extract from MPEG-4 videos."""
import io
import os
import random
import torch
import boto3
import lance
from botocore.config import Config
from torchcodec.decoders import VideoDecoder

client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                      config=Config(connect_timeout=30, read_timeout=60))

ds = lance.dataset("/mnt/data/tables/s3_media_primary.lance")
rows = ds.to_table(
    columns=["tar_path", "member_path", "offset", "size"],
    filter="tar_path LIKE '%/tars/howto100m/%'",
    limit=100,
).to_pylist()

videos = [r for r in rows if r["member_path"].endswith((".mp4", ".webm"))]
random.shuffle(videos)

# check VideoDecoder API
print("VideoDecoder methods:", [m for m in dir(VideoDecoder) if not m.startswith("_")])

for v in videos[:5]:
    mp = v["member_path"]
    print(f"\n--- {mp} ({v['size']/1e6:.1f}MB) ---")
    try:
        resp = client.get_object(
            Bucket="ami-labs-data", Key=v["tar_path"],
            Range=f"bytes={v['offset']}-{v['offset'] + v['size'] - 1}")
        raw = resp["Body"].read()
        raw_t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        dec = VideoDecoder(raw_t)
        m = dec.metadata
        print(f"  codec={m.codec} {m.width}x{m.height} frames={m.num_frames} fps={m.average_fps:.1f}")

        # get a few frames
        fb = dec.get_frames_at(indices=[0, 1, 2])
        print(f"  decoded 3 frames, shape={fb.data.shape}, dtype={fb.data.dtype}")
        # check what attributes FrameBatch has
        attrs = [a for a in dir(fb) if not a.startswith("_")]
        print(f"  FrameBatch attrs: {attrs}")
        if hasattr(fb, 'pts_seconds'):
            print(f"  pts_seconds: {fb.pts_seconds}")
        if hasattr(fb, 'duration_seconds'):
            print(f"  duration_seconds: {fb.duration_seconds}")

    except Exception as e:
        print(f"  FAILED: {e}")

# also check if decord can do export_mvs
print("\n--- checking decord MV export ---")
try:
    from decord import VideoReader, cpu
    v = videos[0]
    resp = client.get_object(
        Bucket="ami-labs-data", Key=v["tar_path"],
        Range=f"bytes={v['offset']}-{v['offset'] + v['size'] - 1}")
    raw = resp["Body"].read()
    vr = VideoReader(io.BytesIO(raw), ctx=cpu(0))
    print(f"  decord: {len(vr)} frames, fps={vr.get_avg_fps():.1f}")
    # decord doesn't export MVs, but let's confirm it works
    batch = vr.get_batch([0, 1, 2]).asnumpy()
    print(f"  decoded 3 frames, shape={batch.shape}")
except Exception as e:
    print(f"  decord failed: {e}")

print("\nDone.")
