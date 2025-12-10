<img width="2752" height="1536" alt="Gemini_Generated_Image_s0u74hs0u74hs0u7" src="https://github.com/user-attachments/assets/95f9e48f-0b32-4008-9f7d-9b6b20b8cc4d" />

A minimal, educational HEVC (H.265) encoder written in Python.

## Overview

nano-hevc is designed for **learning and teaching** video compression concepts.
The code prioritizes clarity and readability over performance.

## Features (Planned)

- [x] Frame/Plane abstractions with numpy
- [x] Intra prediction (DC mode)
- [ ] Intra prediction (Planar, Angular modes)
- [ ] Integer transform (4x4, 8x8, 16x16, 32x32)
- [ ] Quantization / Dequantization
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
from nano_hevc.intra import intra_dc_predict_4x4, residual_block

# Reference pixels
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
pred = intra_dc_predict_4x4(top, left)
print(f"DC value: {pred[0, 0]}")  # 101

# Compute residual
res = residual_block(orig, pred)
print(res)
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
    intra.py              # Intra prediction modes
    transform.py          # Integer transform (TODO)
    quant.py              # Quantization (TODO)
    scan.py               # Scan patterns (TODO)
    cabac.py              # CABAC entropy coding (TODO)
    nal.py                # NAL unit generation (TODO)
    encoder.py            # Main encoding loop (TODO)
  tests/
    test_intra_dc.py
```

## References

- ITU-T H.265 / ISO/IEC 23008-2 (HEVC specification)
- "High Efficiency Video Coding (HEVC)" by Sze, Budagavi, Sullivan
- [HEVC/H.265 Video Coding Standard Tutorial](https://www.youtube.com/watch?v=Fawcboio6g4)
- [H.265/HEVC Tutorial (MIT/ISCAS 2014)](https://eems.mit.edu/wp-content/uploads/2014/06/H.265-HEVC-Tutorial-2014-ISCAS.pdf)

## License

MIT License
