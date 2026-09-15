#!/bin/bash
#SBATCH --job-name=ctp-2k-128M
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/home/brian/logs/slurm/ctp-2k-%j.out
#SBATCH --error=/home/brian/logs/slurm/ctp-2k-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail
cd "$HOME/codebase/nano-hevc"
pip install av -e . --quiet 2>&1 | tail -1

export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/credentials/wandb.env ]; then
    source ~/credentials/wandb.env
fi
export WANDB_DIR="/tmp/wandb"
mkdir -p "${WANDB_DIR}"
export PYTHONUNBUFFERED=1

echo "=== CTP GOP: 2000 videos, 8 GOPs × 8 frames, 128M params ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date -Iseconds)"

# 2000 videos x 8 GOPs x (1 I + 7 P/B) = ~128K frames
# I-frame tokens = context (no loss), P/B tokens = prediction targets
# model: 12L x 12H x 768d ≈ 128M params
python -m experiments.ctp.ctp_demo \
    --codec-mode mpeg4 \
    --lance-uri /mnt/data/tables/s3_media_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --num-videos 2000 \
    --num-gops 8 \
    --pframes-per-gop 7 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --block-size 512 \
    --batch-size 32 \
    --max-iters 10000 \
    --eval-interval 500 \
    --eval-iters 20 \
    --learning-rate 3e-4 \
    --warmup-steps 500 \
    --max-grad-norm 1.0 \
    --amp \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-2kv-8gop-12L768d"

echo "finished: $(date -Iseconds)"
