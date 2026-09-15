#!/bin/bash
#SBATCH --job-name=ctp-1b-howto100m
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/brian/logs/slurm/ctp-1b-%j.out
#SBATCH --error=/home/brian/logs/slurm/ctp-1b-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail

REPO="$HOME/codebase/nano-hevc"
cd "${REPO}"

pip install av -e . --quiet 2>&1 | tail -1

# S3 + W&B credentials
export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/credentials/wandb.env ]; then
    source ~/credentials/wandb.env
fi
export WANDB_DIR="/tmp/wandb"
mkdir -p "${WANDB_DIR}"
export PYTHONUNBUFFERED=1

echo "=== CTP 1B training on howto100m ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date -Iseconds)"

# 1B model: 20 layers x 16 heads x 2048 hidden
# ~1.0B transformer params + embeddings/heads
# 500 videos x 8 frames = 4000 frames -> ~2.8M tokens
# H200 80GB: bf16 fits comfortably
python -m experiments.ctp.ctp_demo \
    --codec-mode mpeg4 \
    --lance-uri /mnt/data/tables/s3_media_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --num-videos 500 \
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
    --max-iters 5000 \
    --eval-interval 200 \
    --eval-iters 20 \
    --learning-rate 1e-4 \
    --weight-decay 0.0 \
    --warmup-steps 500 \
    --max-grad-norm 1.0 \
    --amp \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-1b-howto100m-500v-20L2048d" \
    --trace-out /tmp/ctp_1b_trace.json

echo "finished: $(date -Iseconds)"
