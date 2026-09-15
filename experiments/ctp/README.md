# Codec-Token Prediction Experiments

These scripts explore training a Transformer on codec token sequences. Start with the [main encoder](../../README.md) and [teaching examples](../../examples/README.md) before using this research workflow.

Run commands from the repository root in an environment with the experiment dependencies installed. The core `uv sync --extra dev` setup does not include them.
The trainer uses PyTorch and PyAV; dataset and streaming paths also use Lance, boto3, and TorchCodec. W&B logging is optional.

| File | Purpose |
| --- | --- |
| [ctp_demo.py](ctp_demo.py) | Extract codec tokens and train the prediction model. |
| [build_gop_index.py](build_gop_index.py) | Build a GOP index for videos stored in S3. |
| [probe_codecs.py](probe_codecs.py) | Inspect codec types in the configured dataset. |
| [debug_torchcodec.py](debug_torchcodec.py) | Inspect TorchCodec decoding and metadata. |
| [debug_warmup.py](debug_warmup.py) | Inspect MPEG-4 token extraction. |
| [slurm/](slurm/) | Existing training and diagnostic launch configurations. |

Inspect trainer options with `python -m experiments.ctp.ctp_demo --help`.
The default local video is `examples/assets/birds.mp4`.
Run the indexing and diagnostic modules with the same `python -m experiments.ctp.<module>` pattern.

The Slurm configurations retain their cluster-specific container, data, and output paths. Select the configuration for your environment before submitting it.
The previous `examples/ctp_demo.py` entry point is now `python -m experiments.ctp.ctp_demo`; the launch scripts have been updated accordingly.
