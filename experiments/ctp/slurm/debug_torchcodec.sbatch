#!/bin/bash
#SBATCH --job-name=debug-tc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/brian/logs/slurm/debug-tc-%j.out
#SBATCH --error=/home/brian/logs/slurm/debug-tc-%j.err
#SBATCH --container-image=/mnt/data/enroot/base-pytorch-dali-torchcodec-25.03.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data,/home/brian:/home/brian
#SBATCH --no-container-mount-home
#SBATCH --container-writable

set -euo pipefail
export S3_ENDPOINT_URL="https://storage.eu-north1.nebius.cloud"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/credentials"
export AWS_CONFIG_FILE="$HOME/.aws/config"
export PYTHONUNBUFFERED=1
pip install -e /home/brian/codebase/nano-hevc --quiet 2>&1 | tail -1
cd "$HOME/codebase/nano-hevc"
python3 -m experiments.ctp.debug_torchcodec
