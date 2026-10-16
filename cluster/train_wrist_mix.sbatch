#!/bin/bash
#SBATCH -J smolvla-wristmix
#SBATCH -p general
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/%x-%j.out
# Fine-tune the demo SmolVLA with front-camera dropout + community SO-101 wrist data (policy/train_wrist_mix.py).
# Writes a NEW checkpoint to ~/models/smolvla-wristmix-<job>; the base checkpoint is only read. Needs the owner's
# go-ahead per run (CONTRIBUTING.md). USABLE comes from the dataset filter job.
#   sbatch --export=ALL,STEPS=2000 cluster/train_wrist_mix.sbatch
set -euo pipefail
fix() { [ -e "$1" ] && echo "$1" || echo "${1/#\/nethome\//\/nethome-instruction\/}"; }
HOME_DIR=$(fix "$HOME")
REPO=$(fix "$SLURM_SUBMIT_DIR")
BASE=${BASE:-$HOME_DIR/models/smolvla-so101_medicine_bottle_pickplace-9604}
USABLE=${USABLE:-$HOME_DIR/robotfpga-work/jobs/usable_datasets.json}
STEPS=${STEPS:-2000}
RESULT=$HOME_DIR/models/smolvla-wristmix-$SLURM_JOB_ID
WORK=/dev/shm/$USER-$SLURM_JOB_ID
LEROBOT_REV=e624f3f7f8411ec3a02635d06e79373341e5ef35
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv \
       TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton
UV=$HOME_DIR/.local/bin/uv
echo "job $SLURM_JOB_ID on $(hostname) repo=$REPO git=$(git -C "$REPO" rev-parse --short HEAD) base=$BASE"
git clone -q https://github.com/psamin/lerobot.git "$WORK/lerobot"
cd "$WORK/lerobot" && git checkout -q "$LEROBOT_REV"
"$UV" sync -q --locked --extra training --extra smolvla --extra av-dep --python 3.12
nvidia-smi --query-gpu=name,memory.total --format=csv
"$UV" run --no-sync python "$REPO/policy/train_wrist_mix.py" --base "$BASE" --usable "$USABLE" \
  --out "$RESULT" --steps "$STEPS" ${ARGS:-} &
wait $!
