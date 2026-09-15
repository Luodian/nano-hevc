"""Debug: test MPEG-4 extraction from howto100m S3 videos."""
import io
import os
import random
import traceback

import boto3
import lance
from botocore.config import Config

client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                      config=Config(connect_timeout=30, read_timeout=60))

ds = lance.dataset("/mnt/data/tables/s3_media_primary.lance")
rows = ds.to_table(
    columns=["tar_path", "member_path", "offset", "size"],
    filter="tar_path LIKE '%/tars/howto100m/%'",
    limit=200,
).to_pylist()

videos = [r for r in rows if r["member_path"].endswith((".mp4", ".webm"))]
random.shuffle(videos)

for v in videos[:10]:
    mp = v["member_path"]
    print(f"\n--- {mp} ({v['size']/1e6:.1f}MB) ---")
    try:
        resp = client.get_object(
            Bucket="ami-labs-data", Key=v["tar_path"],
            Range=f"bytes={v['offset']}-{v['offset'] + v['size'] - 1}")
        raw = resp["Body"].read()
        print(f"  fetched {len(raw)} bytes")

        import av
        c = av.open(io.BytesIO(raw))
        s = c.streams.video[0]
        print(f"  codec={s.codec_context.name} {s.codec_context.width}x{s.codec_context.height}")
        s.codec_context.options = {"flags2": "+export_mvs"}

        frames = []
        for f in c.decode(video=0):
            frames.append(f)
            if len(frames) >= 5:
                break
        c.close()
        print(f"  decoded {len(frames)} frames")

        for i, f in enumerate(frames):
            mvs = None
            for sd in f.side_data:
                if "MOTION" in str(sd.type).upper():
                    mvs = sd
                    break
            n = len(mvs) if mvs is not None else 0
            print(f"  frame {i}: pict_type={f.pict_type} mvs={n}")

        # now test the full extraction
        from experiments.ctp.ctp_demo import extract_mpeg4_traces_from_bytes
        traces = extract_mpeg4_traces_from_bytes(raw, 4, fps=8)
        if traces:
            print(f"  traces: {len(traces)} frames, {sum(len(t) for t in traces)} tokens")
        else:
            print(f"  traces: None (extraction returned None)")

    except Exception:
        traceback.print_exc()
