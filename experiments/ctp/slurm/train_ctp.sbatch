#!/bin/bash
#SBATCH --job-name=ctp-howto100m
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/brian/logs/slurm/ctp-%j.out
#SBATCH --error=/home/brian/logs/slurm/ctp-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail

REPO="$HOME/codebase/nano-hevc"
cd "${REPO}"

# install nano-hevc in the container
pip install av -e . --quiet 2>&1 | tail -1

# S3 credentials and endpoint (Nebius Object Storage)
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"

# W&B setup
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/credentials/wandb.env ]; then
    source ~/credentials/wandb.env
fi
export WANDB_DIR="/tmp/wandb"
mkdir -p "${WANDB_DIR}"

export PYTHONUNBUFFERED=1

echo "=== CTP training on howto100m ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "started: $(date -Iseconds)"

python -m experiments.ctp.ctp_demo \
    --codec-mode mpeg4 \
    --lance-uri /mnt/data/tables/s3_media_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --num-videos 50 \
    --frames-per-video 8 \
    --width 128 \
    --height 64 \
    --qp 27 \
    --max-cu-size 8 \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 128 \
    --block-size 256 \
    --batch-size 32 \
    --max-iters 2000 \
    --eval-interval 100 \
    --eval-iters 20 \
    --learning-rate 3e-4 \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-howto100m-50v-4L128d" \
    --trace-out /tmp/ctp_trace.json

echo "finished: $(date -Iseconds)"
