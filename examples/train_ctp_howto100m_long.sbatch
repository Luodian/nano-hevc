#!/bin/bash
#SBATCH --job-name=ctp-long-10k
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/home/brian/logs/slurm/ctp-long-%j.out
#SBATCH --error=/home/brian/logs/slurm/ctp-long-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

# full H200 node: 8 GPUs, 96 CPUs, all memory
# training uses 1 GPU; full node gives headroom for S3 fetch + decode threads

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

echo "=== CTP long run: 10K videos, 16 GOPs, 128M params, cosine LR ==="
echo "node: $(hostname)"
echo "gpu:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "cpus: $(nproc)"
echo "mem:  $(free -h | awk '/Mem:/{print $2}')"
echo "started: $(date -Iseconds)"

# 10000 videos x 16 GOPs x (1 I + 7 P/B) = ~1.28M frames
# direct-to-tensor path: ~20x less memory than SymbolTrace
# model: 12L x 12H x 768d = ~86M params (proven at 2K scale)
# cosine LR decay to prevent the step-7500 collapse seen in the 2K run
python examples/ctp_demo.py \
    --codec-mode mpeg4 \
    --lance-uri /mnt/data/tables/s3_video_primary.lance \
    --s3-endpoint https://storage.eu-north1.nebius.cloud \
    --datasets howto100m \
    --num-videos 10000 \
    --num-gops 16 \
    --pframes-per-gop 7 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --block-size 512 \
    --batch-size 64 \
    --max-iters 50000 \
    --eval-interval 1000 \
    --eval-iters 20 \
    --learning-rate 3e-4 \
    --weight-decay 1e-2 \
    --warmup-steps 1000 \
    --lr-decay cosine \
    --max-grad-norm 1.0 \
    --amp \
    --device cuda \
    --wandb \
    --wandb-project nano-hevc-ctp \
    --wandb-run-name "ctp-10kv-16gop-12L768d-cosine"

echo "finished: $(date -Iseconds)"
