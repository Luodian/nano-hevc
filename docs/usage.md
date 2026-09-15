# Usage Guide

This guide covers encoding modes, verification steps, and basic Python workflows for nano-hevc. For installation and setup instructions, see [README.md](../README.md).

Run all commands from the repository root. Create an output directory before starting:

```bash
mkdir -p outputs
```

## Encoding Modes

Choose an output mode:

| Mode | Command Flag | Output Format | Description |
| :--- | :--- | :--- | :--- |
| Custom Container | Default (`--backend nano`) | `.nhevc` (NHEVC1) | Custom educational format for round-trip learning. Not standard HEVC. |
| Reconstructed HEVC | `--backend nano --nano-standard-hevc` | `.hevc` | Reconstructs frames in nano-hevc, then pipes them to ffmpeg HEVC. |
| FFmpeg Baseline | `--backend ffmpeg` | `.hevc` | Standard HEVC stream created directly with ffmpeg. |
| Native HEVC | `--backend nano --nano-native-hevc` | `.hevc` | Experimental bitstream generated directly with nano syntax writers. |

`--nano-standard-hevc` and `--nano-native-hevc` are mutually exclusive. Native HEVC output is experimental; visual correctness and specification compliance are not guaranteed. The standard HEVC commands below require ffmpeg with `libx265`; creating the decoded MP4 preview requires `libx264`.

Run `uv run nano-hevc --help` to inspect all available options.

## Standard HEVC Encoding

Run the nano-hevc pipeline with standard HEVC packaging via ffmpeg:

```bash
uv run nano-hevc examples/assets/birds.mp4 \
  -o outputs/birds.nano.std.hevc \
  --width 128 --height 128 --frames 3 \
  --backend nano --qp 27 --fast \
  --nano-standard-hevc --nano-standard-codec libx265 --nano-standard-crf 28 \
  --verify-decoded-psnr
```

Run the matching baseline directly through the ffmpeg backend:

```bash
uv run nano-hevc examples/assets/birds.mp4 \
  -o outputs/birds.ffmpeg.hevc \
  --width 128 --height 128 --frames 3 \
  --backend ffmpeg --ffmpeg-codec libx265 --ffmpeg-crf 28 \
  --verify-decoded-psnr
```

The flag `--nano-standard-intra-only` forces all frames to be I-frames. Leave this flag off to allow ffmpeg to insert P and B frames. Do not expect identical quality or bitrate parity between these two outputs, as the nano pipeline applies its own intra reconstruction step before feeding raw video to ffmpeg.

## Experimental Native HEVC

Generate an experimental native HEVC stream using nano-hevc bitstream writers:

```bash
uv run nano-hevc examples/assets/birds.mp4 \
  -o outputs/birds.nano.native.hevc \
  --width 128 --height 128 --frames 3 \
  --backend nano --qp 27 --fast --nano-native-hevc
```

To enable experimental native inter coding with motion estimation:

```bash
uv run nano-hevc examples/assets/birds.mp4 \
  -o outputs/birds.nano.native.inter.hevc \
  --width 128 --height 128 --frames 3 \
  --backend nano --qp 27 --fast --nano-native-hevc --nano-native-inter \
  --nano-native-gop-size 30 --nano-native-search-range 8
```

Native mode carries residual coefficients, but syntax support alone does not establish decoded image correctness. Use `--verify-decoded-psnr` to check the final stream when external decoding succeeds.

## Working with NHEVC1 Streams

### Container Decoding

After creating `outputs/birds.nhevc` with the [quick start](../README.md#quick-start), decode it back into raw YUV420p or MP4:

```bash
uv run nano-hevc outputs/birds.nhevc -o outputs/birds.decoded.yuv --decode-nano
```

```bash
uv run nano-hevc outputs/birds.nhevc -o outputs/birds.decoded.mp4 --decode-nano --decode-codec libx264
```

### Raw Input Encoding API

Encoding from uncompressed YUV files requires explicit dimensions because raw streams omit file headers. You must supply your own raw YUV file:

```python
from nano_hevc.encoder import encode_video

stats = encode_video(
    "input.yuv",
    "outputs/output.nhevc",
    width=640,
    height=360,
    num_frames=1,
    qp=27,
    backend="nano",
)
print(stats)
```

### Input Analysis and Segment Selection

Analyze the frame distribution of a source video without writing output:

```bash
uv run nano-hevc examples/assets/birds.mp4 \
  --analyze --start-time 0 --duration 2 --frames 30
```

`--start-time` and `--duration` select a time slice from container inputs. The `--frames` option acts as an upper cap on processed frames.

## Verification and Metrics

Do not run `ffprobe` on custom `.nhevc` files. For standard decoded MP4 streams, inspect stream and container properties with `ffprobe`:

```bash
ffprobe -hide_banner -loglevel error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_read_frames \
  -show_entries format=size,duration,bit_rate \
  -of default=noprint_wrappers=1 outputs/birds.decoded.mp4
```

The nano backend reports in-memory reconstruction PSNR (peak signal-to-noise ratio). To compare final HEVC quality, add `--verify-decoded-psnr`. This decodes the written stream and measures luma PSNR against the input segment.

## Python Learning Example

You can trace block-level intra prediction, transform, quantization, and reconstruction in Python:

```python
import numpy as np
from nano_hevc.intra import intra_dc_predict, residual_block
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block

top = np.array([102, 98, 100, 101], dtype=np.int16)
left = np.array([103, 102, 101, 99], dtype=np.int16)

orig = np.array([
    [102, 101, 100, 100],
    [103, 102, 101, 100],
    [103, 102, 100,  99],
    [104, 101,  99,  98],
], dtype=np.int16)

pred = intra_dc_predict(top, left, size=4)
print(f"DC value: {pred[0, 0]}")

residual = residual_block(orig, pred)
coeff = forward_transform(residual, use_dst=True)

levels = quantize_block(coeff, qp=22)
print(f"Non-zero coefficients: {np.count_nonzero(levels)}")

recon_coeff = dequantize_block(levels, qp=22)
recon_residual = inverse_transform(recon_coeff, use_dst=True)
recon_block = np.clip(pred + recon_residual, 0, 255).astype(np.uint8)
print(f"Reconstructed block:\n{recon_block}")
```
