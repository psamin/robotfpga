#!/bin/bash
#SBATCH -J strap-retrieve
#SBATCH -p general
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out
# STRAP-style retrieval of community sub-trajectories similar to our demos (policy/strap_retrieve.py).
# Embeds wrist frames with DINOv2, matches with subsequence DTW. Writes ~/robotfpga-work/runs/strap-<job>.json.
#   sbatch --export=ALL,ARGS="--holdout Remember.the.color" cluster/strap_retrieve.sbatch
set -euo pipefail
fix() { [ -e "$1" ] && echo "$1" || echo "${1/#\/nethome\//\/nethome-instruction\/}"; }
HOME_DIR=$(fix "$HOME")
REPO=$(fix "$SLURM_SUBMIT_DIR")
USABLE=${USABLE:-$HOME_DIR/robotfpga-work/jobs/usable_datasets.json}
OUT=$HOME_DIR/robotfpga-work/runs/strap-$SLURM_JOB_ID.json
WORK=/dev/shm/$USER-$SLURM_JOB_ID
LEROBOT_REV=e624f3f7f8411ec3a02635d06e79373341e5ef35
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv \
       TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton
UV=$HOME_DIR/.local/bin/uv
echo "job $SLURM_JOB_ID on $(hostname) repo=$REPO git=$(git -C "$REPO" rev-parse --short HEAD)"
git clone -q https://github.com/psamin/lerobot.git "$WORK/lerobot"
cd "$WORK/lerobot" && git checkout -q "$LEROBOT_REV"
"$UV" sync -q --locked --extra training --extra smolvla --extra av-dep --python 3.12
nvidia-smi --query-gpu=name,memory.total --format=csv
"$UV" run --no-sync python "$REPO/policy/strap_retrieve.py" --usable "$USABLE" --out "$OUT" ${ARGS:-}
