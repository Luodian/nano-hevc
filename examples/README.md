# Teaching Examples

Start with the [encoder round-trip](../README.md#quick-start), then explore these examples in order. Run commands from the repository root.

1. Run `uv run python examples/intra.py` to compare intra predictions on a synthetic frame.
2. Read and run `uv run python examples/nano_mp4.py` to trace a block through an H.264-style teaching pipeline.
3. Run `uv run python examples/inter_p_simple.py` to compare I, P, and B prediction on synthetic motion.
4. Read the [sample analysis](../docs/birds_analysis.md) and view the [frame representation image](assets/frame_repr_viz.png).

The `nano_mp4.py` example illustrates H.264 concepts and does not produce a standard MP4 file.
The former `python -m nano_hevc demo` is now `examples/intra.py`. Both `uv run nano-hevc` and `uv run python -m nano_hevc` run the full encoder; its raw YUV workflow replaces the former prediction-only `encode` subcommand.
The `assets/` directory contains the sample videos, saved probe metadata, and reference image.
To regenerate the image, run `uv run --with matplotlib python examples/frame_repr_viz.py`; this also requires ffmpeg.

Codec-token training, data indexing, and cluster diagnostics live in [experiments/ctp/](../experiments/ctp/README.md).
