#!/bin/bash
#SBATCH --job-name=ctp-1b-stream
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/brian/logs/slurm/ctp-1b-stream-%j.out
#SBATCH --error=/home/brian/logs/slurm/ctp-1b-stream-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail

REPO="$HOME/codebase/nano-hevc"
cd "${REPO}"

pip install -e . --quiet 2>&1 | tail -1

# S3 + W&B
export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /home/brian/.wandb_api_key ]; then
    export WANDB_API_KEY="$(tr -d '\n' < /home/brian/.wandb_api_key)"
fi
export WANDB_DIR="/tmp/wandb"
mkdir -p "${WANDB_DIR}"
export PYTHONUNBUFFERED=1

echo "=== CTP 1B streaming on full howto100m ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "cpus: $(nproc)"
echo "started: $(date -Iseconds)"

# 1B model streaming from full howto100m (~1.2M videos)
# 16 DataLoader workers handle S3 fetch + decode + HEVC encode in parallel
# GPU trains on the stream with bf16 AMP
python examples/ctp_demo.py \
    --streaming \
    --streaming-workers 16 \
    --lance-uri /mnt/data/tables/s3_video_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --warmup-frames 64 \
    --eval-videos 50 \
    --codec-mode mpeg4 \
    --frames-per-video 8 \
    --width 128 \
    --height 64 \
    --qp 27 \
    --max-cu-size 8 \
    --n-layer 20 \
    --n-head 16 \
    --n-embd 2048 \
    --block-size 1024 \
    --batch-size 16 \
    --max-iters 20000 \
    --eval-interval 500 \
    --eval-iters 20 \
    --learning-rate 1e-4 \
    --weight-decay 0.0 \
    --warmup-steps 500 \
    --max-grad-norm 1.0 \
    --amp \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-1b-stream-howto100m-20L2048d"

echo "finished: $(date -Iseconds)"
