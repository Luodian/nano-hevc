<img width="100%" alt="nano-hevc banner" src="https://github.com/user-attachments/assets/95f9e48f-0b32-4008-9f7d-9b6b20b8cc4d" />

# nano-hevc

nano-hevc is a minimal Python implementation of HEVC (H.265) core algorithms designed for learning and teaching video compression. The codebase prioritizes readability and fundamental principles over production speed.

The encoder predicts each block, transforms and quantizes the residual, then stores the result. Reconstruction reverses these steps and adds the prediction back.

## Installation

Prerequisites:

- Python 3.9+
- uv
- ffmpeg and ffprobe on your PATH
- libx264 is required for the decoded MP4 preview example
- libx265 is only required for optional standard HEVC output modes

Clone the repository and install development dependencies:

```bash
git clone https://github.com/the-AMI-Labs/nano-hevc.git
cd nano-hevc
uv sync --extra dev
```

## Quick Start

Encode 3 frames of the included sample video into the educational NHEVC1 container format, then decode the bitstream back to an MP4 video:

```bash
mkdir -p outputs
uv run nano-hevc examples/assets/birds.mp4 -o outputs/birds.nhevc \
  --width 128 --height 128 --frames 3 --backend nano --qp 27 --fast
uv run nano-hevc outputs/birds.nhevc -o outputs/birds.decoded.mp4 \
  --decode-nano
```

Open `outputs/birds.decoded.mp4` to view three reconstructed frames.

NHEVC1 is the project's custom container format and requires the nano decoder. Native HEVC output is experimental. For standard HEVC output, use one of the [ffmpeg modes](docs/usage.md#standard-hevc-encoding).

## Source Reading Order

Explore the codebase in the following sequence to follow how video data moves through the codec:

1. [nano_hevc/frame.py](nano_hevc/frame.py) and [nano_hevc/block.py](nano_hevc/block.py): Frame buffers, YUV planes, and block slicing abstractions.
2. [nano_hevc/intra.py](nano_hevc/intra.py): Spatial intra prediction (DC, Planar, and angular modes) and residual calculation.
3. [nano_hevc/transform.py](nano_hevc/transform.py) and [nano_hevc/quant.py](nano_hevc/quant.py): Integer DCT/DST transforms and scalar quantization across QP 0-51.
4. [nano_hevc/scan.py](nano_hevc/scan.py) and [nano_hevc/cabac.py](nano_hevc/cabac.py): Coefficient scan orders and the CABAC arithmetic entropy engine.
5. [nano_hevc/bitstream.py](nano_hevc/bitstream.py) and [nano_hevc/nal.py](nano_hevc/nal.py): Emulation prevention byte stuffing and NAL unit packaging (VPS, SPS, PPS, Slice).
6. [nano_hevc/encoder.py](nano_hevc/encoder.py): Pipeline orchestration connecting prediction, transformation, entropy coding, and file I/O.

The native HEVC path uses CABAC and NAL units. The NHEVC1 container stores block modes and compressed coefficients in its own format.

## Repository Layout

```text
nano_hevc/        Core codec implementation
examples/         Small teaching demos
  assets/         Sample videos, metadata, and reference image
docs/             Usage guide and longer explanations
experiments/ctp/   Codec-token prediction training and debugging
  slurm/          Cluster launch scripts
tests/            Codec tests
```

## Further Usage

- Read [docs/usage.md](docs/usage.md) for alternative encoding modes, standard ffmpeg backend flags, and parameter options.
- Read [docs/frames_and_panes.md](docs/frames_and_panes.md) for an article in Chinese explaining the frame and plane data layout.
- Run [examples/inter_p_simple.py](examples/inter_p_simple.py) for a standalone synthetic demonstration of I, P, and B prediction logic.
- See [docs/birds_analysis.md](docs/birds_analysis.md) for deeper sample analysis and experimental tokenization research (not required for onboarding).

See [examples/README.md](examples/README.md) for the demo order. The [CTP experiments](experiments/ctp/README.md) require additional dependencies and have their own entry point.

## Running Tests

[test_core.py](tests/test_core.py) covers prediction, transforms, and quantization. [test_encoder.py](tests/test_encoder.py) covers bitstreams and encode/decode round-trips.

```bash
uv run pytest
```

## References

- [HEVC/H.265 Video Coding Standard Tutorial](https://www.youtube.com/watch?v=Fawcboio6g4)
- [H.265/HEVC Tutorial (MIT/ISCAS 2014)](https://eems.mit.edu/wp-content/uploads/2014/06/H.265-HEVC-Tutorial-2014-ISCAS.pdf)

## License

MIT License

Use as you wish, but please cite LOL

## Citation

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
