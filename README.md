<img width="100%" alt="Gemini_Generated_Image_s0u74hs0u74hs0u7" src="https://github.com/user-attachments/assets/95f9e48f-0b32-4008-9f7d-9b6b20b8cc4d" />

A minimal, educational HEVC (H.265) encoder written in Python.

## Overview

nano-hevc is designed for **learning and teaching** video compression concepts.
The code prioritizes clarity and readability over performance.

## Features

- [x] Frame/Plane abstractions with numpy
- [x] Intra prediction (DC mode)
- [x] Intra prediction (Planar mode)
- [x] Intra prediction (Angular modes 2-34)
- [x] Integer transform (4x4, 8x8, 16x16, 32x32 DCT)
- [x] 4x4 DST-VII for luma intra blocks
- [x] Quantization / Dequantization (QP 0-51)
- [x] Zig-zag and other scan patterns (diagonal/horizontal/vertical)
- [x] NHEVC1 container encode/decode round-trip
- [x] ffmpeg backend for standards-compliant HEVC output
- [ ] CABAC entropy coding (coefficients + CBF only; syntax is partial)
- [x] NAL unit generation (VPS/SPS/PPS/Slice)
- [ ] Full HEVC-compliant intra-only bitstream from nano backend
- [ ] Inter prediction / motion estimation (P/B coding)

## Installation

```bash
git clone https://github.com/yourusername/nano-hevc.git
cd nano-hevc

uv sync

pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from nano_hevc.intra import intra_dc_predict, intra_planar_predict, residual_block
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block

# reference pixels (for 4x4 block)
top = np.array([102, 98, 100, 101], dtype=np.int16)
left = np.array([103, 102, 101, 99], dtype=np.int16)

# original block
orig = np.array([
    [102, 101, 100, 100],
    [103, 102, 101, 100],
    [103, 102, 100,  99],
    [104, 101,  99,  98],
], dtype=np.int16)

# dc prediction
pred = intra_dc_predict(top, left, size=4)
print(f"DC value: {pred[0, 0]}")  # 101

# compute residual
residual = residual_block(orig, pred)

# transform (dst for 4x4 luma intra)
coeff = forward_transform(residual, use_dst=True)

# Quantize (QP=22 for decent quality)
levels = quantize_block(coeff, qp=22)
print(f"Non-zero coefficients: {np.count_nonzero(levels)}")

# decoder side: dequantize and inverse transform
recon_coeff = dequantize_block(levels, qp=22)
recon_residual = inverse_transform(recon_coeff, use_dst=True)
```

Encode a raw YUV420p file with the `nano` backend (NHEVC1 container):

```python
from nano_hevc.encoder import encode_video

stats = encode_video(
    "input.yuv",
    "output.nhevc",
    width=640,
    height=360,
    num_frames=1,
    qp=27,
    backend="nano",
)
print(stats)
```

Encode with `nano` reconstruction + standard HEVC output (intra-only GOP):

```bash
python -m nano_hevc.encoder examples/videos/red_bull_300f_640x360.mp4 \
  -o /tmp/red_bull_300f.nano.std.hevc \
  --width 640 --height 360 --frames 300 \
  --backend nano --qp 27 --fast \
  --nano-standard-hevc --nano-standard-codec libx265 --nano-standard-crf 28
```

Decode `nano` container back to YUV/MP4:

```bash
# to raw yuv420p
python -m nano_hevc.encoder output.nhevc -o decoded.yuv --decode-nano

# to mp4
python -m nano_hevc.encoder output.nhevc -o decoded.mp4 --decode-nano --decode-codec libx264
```

## Red Bull Example Workflow (First 2 Minutes)

Use local HD source:

```bash
cp /path/to/red_bull.mp4 examples/videos/red_bull.mp4
```

Create a 2-minute, 640x360 test clip (about 3000 frames at 25fps):

```bash
ffmpeg -y -i examples/videos/red_bull.mp4 -t 120 \
  -vf scale=640:360 -an -c:v libx264 -preset veryfast -crf 20 \
  examples/videos/red_bull_first2min_640x360.mp4
```

Run full `nano` container round-trip:

```bash
# encode to nano container
python -m nano_hevc.encoder examples/videos/red_bull_first2min_640x360.mp4 \
  -o /tmp/red_bull_first2min.nano.nhevc \
  --width 640 --height 360 --frames 3000 \
  --backend nano --qp 27 --fast --show-frame-types

# decode nano container back to playable mp4
python -m nano_hevc.encoder /tmp/red_bull_first2min.nano.nhevc \
  -o /tmp/red_bull_first2min.decoded.mp4 --decode-nano --decode-codec libx264
```

Output standard HEVC from nano backend (reconstructed frames piped to ffmpeg):

```bash
python -m nano_hevc.encoder examples/videos/red_bull_first2min_640x360.mp4 \
  -o /tmp/red_bull_first2min.nano.std.hevc \
  --width 640 --height 360 --frames 3000 \
  --backend nano --qp 27 --fast --show-frame-types \
  --nano-standard-hevc --nano-standard-codec libx265 --nano-standard-crf 28
```

Baseline standard HEVC directly from ffmpeg backend:

```bash
python -m nano_hevc.encoder examples/videos/red_bull_first2min_640x360.mp4 \
  -o /tmp/red_bull_first2min.ffmpeg.hevc \
  --width 640 --height 360 --frames 3000 \
  --backend ffmpeg --ffmpeg-codec libx265 --ffmpeg-crf 28 --show-frame-types
```

Compare bitrates:

```bash
for f in /tmp/red_bull_first2min.nano.nhevc \
         /tmp/red_bull_first2min.nano.std.hevc \
         /tmp/red_bull_first2min.ffmpeg.hevc; do
  echo "=== $f ==="
  ffprobe -hide_banner -loglevel error -show_entries format=size,bit_rate,duration \
    -of default=noprint_wrappers=1 "$f" || true
done
```

Verify decoded output:

```bash
ffprobe -hide_banner -loglevel error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_read_frames,pix_fmt \
  -show_entries format=duration,size,bit_rate \
  -of default=noprint_wrappers=1 /tmp/red_bull_first2min.decoded.mp4
```

## Running Tests

> to make sure that I understand the HEVC's logic correctly, I advise GPT5.1/Gemini 3.0 Pro and industry experts to provide test suggestions before I start the project.

```bash
uv run pytest

pytest
```

## Project Structure

```
nano_hevc/
  pyproject.toml          # Package configuration
  README.md
  nano_hevc/
    __init__.py
    frame.py              # Frame/Plane abstractions
    block.py              # Block view (numpy slices)
    intra.py              # Intra prediction (DC, Planar, Angular)
    transform.py          # Integer DCT/DST transforms
    quant.py              # Quantization / Dequantization
    scan.py               # Scan patterns (diagonal/horizontal/vertical)
    bitstream.py          # Bitstream writer with emulation prevention
    cabac.py              # CABAC entropy coding (coefficients + CBF - partial)
    nal.py                # NAL unit generation (VPS/SPS/PPS/Slice)
    encoder.py            # Nano/ffmpeg backends + nano container decode
    metrics.py            # PSNR utilities
  tests/
    test_encoder.py
    test_intra_dc.py
    test_intra_planar.py
    test_intra_angular.py
    test_transform.py
    test_quant.py
```

## References

- [HEVC/H.265 Video Coding Standard Tutorial](https://www.youtube.com/watch?v=Fawcboio6g4)
- [H.265/HEVC Tutorial (MIT/ISCAS 2014)](https://eems.mit.edu/wp-content/uploads/2014/06/H.265-HEVC-Tutorial-2014-ISCAS.pdf)

## License

MIT License

Use as you wish, but please cite LOL

```bibtex
@misc{nano-hevc,
  author = {Bo Li},
  title = {nano-hevc: A minimal, educational HEVC encoder in Python},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luodian/nano-hevc}}
}
```
