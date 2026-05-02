#!/bin/bash
#SBATCH --job-name=ctp-scale-500v
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast/home/brian/logs/slurm/ctp-scale-%j.out
#SBATCH --error=/mnt/vast/home/brian/logs/slurm/ctp-scale-%j.err
#SBATCH --container-image=/mnt/vast/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/vast:/mnt/vast,/mnt/vast/home/brian:/home/brian
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

echo "=== CTP scale: 500 videos, 64M params, batch MPEG-4 (FluidStack) ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date -Iseconds)"

python examples/ctp_demo.py \
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
    --n-layer 8 \
    --n-head 8 \
    --n-embd 512 \
    --block-size 512 \
    --batch-size 32 \
    --max-iters 5000 \
    --eval-interval 200 \
    --eval-iters 20 \
    --learning-rate 3e-4 \
    --warmup-steps 200 \
    --max-grad-norm 1.0 \
    --amp \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-500v-8L512d-mpeg4-fs"

echo "finished: $(date -Iseconds)"
