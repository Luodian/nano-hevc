#!/bin/bash
#SBATCH --job-name=test-stream
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/brian/logs/slurm/test-stream-%j.out
#SBATCH --error=/home/brian/logs/slurm/test-stream-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail
export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /home/brian/.wandb_api_key ]; then
    export WANDB_API_KEY="$(tr -d '\n' < /home/brian/.wandb_api_key)"
fi
export PYTHONUNBUFFERED=1

cd "$HOME/codebase/nano-hevc"
pip install av -e . --quiet 2>&1 | tail -1

echo "=== test streaming (small) ==="
python examples/ctp_demo.py \
    --streaming \
    --streaming-workers 2 \
    --lance-uri /mnt/data/tables/s3_video_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --codec-mode mpeg4 \
    --warmup-frames 16 \
    --eval-videos 10 \
    --frames-per-video 4 \
    --n-layer 3 \
    --n-head 4 \
    --n-embd 128 \
    --block-size 256 \
    --batch-size 4 \
    --max-iters 50 \
    --eval-interval 10 \
    --learning-rate 3e-4 \
    --device cuda

echo "done: $(date -Iseconds)"
