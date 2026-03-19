#!/bin/bash
#SBATCH --job-name=ctp-2k-128M
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /home/brian/.wandb_api_key ]; then
    export WANDB_API_KEY="$(tr -d '\n' < /home/brian/.wandb_api_key)"
fi
export WANDB_DIR="/tmp/wandb"
mkdir -p "${WANDB_DIR}"
export PYTHONUNBUFFERED=1

echo "=== CTP scale: 2000 videos, 128M params, batch MPEG-4 ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date -Iseconds)"

# 2000 videos x 8 frames = 16000 frames -> ~24M tokens
# 16-thread parallel S3 fetch: ~3 min (vs ~47 min sequential)
# model: 12L x 12H x 768d ≈ 128M params
# training: 10000 iters, bf16 AMP
python examples/ctp_demo.py \
    --codec-mode mpeg4 \
    --lance-uri /mnt/data/tables/s3_video_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --num-videos 2000 \
    --frames-per-video 8 \
    --width 128 \
    --height 64 \
    --qp 27 \
    --max-cu-size 8 \
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
    --wandb-run-name "ctp-2kv-12L768d-mpeg4"

echo "finished: $(date -Iseconds)"
