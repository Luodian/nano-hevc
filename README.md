<img width="2752" height="1536" alt="Gemini_Generated_Image_s0u74hs0u74hs0u7" src="https://github.com/user-attachments/assets/95f9e48f-0b32-4008-9f7d-9b6b20b8cc4d" />

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
- [ ] Zig-zag and other scan patterns
- [ ] CABAC entropy coding
- [ ] NAL unit generation (VPS/SPS/PPS/Slice)
- [ ] Complete encoding pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nano-hevc.git
cd nano-hevc

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from nano_hevc.intra import intra_dc_predict, intra_planar_predict, residual_block
from nano_hevc.transform import forward_transform, inverse_transform
from nano_hevc.quant import quantize_block, dequantize_block

# Reference pixels (for 4x4 block)
top = np.array([102, 98, 100, 101], dtype=np.int16)
left = np.array([103, 102, 101, 99], dtype=np.int16)

# Original block
orig = np.array([
    [102, 101, 100, 100],
    [103, 102, 101, 100],
    [103, 102, 100,  99],
    [104, 101,  99,  98],
], dtype=np.int16)

# DC prediction
pred = intra_dc_predict(top, left, size=4)
print(f"DC value: {pred[0, 0]}")  # 101

# Compute residual
residual = residual_block(orig, pred)

# Transform (DST for 4x4 luma intra)
coeff = forward_transform(residual, use_dst=True)

# Quantize (QP=22 for decent quality)
levels = quantize_block(coeff, qp=22)
print(f"Non-zero coefficients: {np.count_nonzero(levels)}")

# Decoder side: dequantize and inverse transform
recon_coeff = dequantize_block(levels, qp=22)
recon_residual = inverse_transform(recon_coeff, use_dst=True)
```

## Running Tests

```bash
uv run pytest

# Or if using pip
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
    scan.py               # Scan patterns (TODO)
    cabac.py              # CABAC entropy coding (TODO)
    nal.py                # NAL unit generation (TODO)
    encoder.py            # Main encoding loop (TODO)
  tests/
    test_intra_dc.py
    test_intra_planar.py
    test_intra_angular.py
    test_transform.py
    test_quant.py
```

## References

- ITU-T H.265 / ISO/IEC 23008-2 (HEVC specification)
- "High Efficiency Video Coding (HEVC)" by Sze, Budagavi, Sullivan
- [HEVC/H.265 Video Coding Standard Tutorial](https://www.youtube.com/watch?v=Fawcboio6g4)
- [H.265/HEVC Tutorial (MIT/ISCAS 2014)](https://eems.mit.edu/wp-content/uploads/2014/06/H.265-HEVC-Tutorial-2014-ISCAS.pdf)

## License

MIT License
