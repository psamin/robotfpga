#!/bin/bash
#SBATCH -J armlab-smoke
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Tests + a rollout video on a compute node (checks EGL rendering works headless).
source cluster/env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$UV" run pytest -q
"$UV" run python scripts/rollout_video.py --out "runs/smoke-${SLURM_JOB_ID}.mp4"
