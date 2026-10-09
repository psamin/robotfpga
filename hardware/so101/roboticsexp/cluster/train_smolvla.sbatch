#!/bin/bash
# Fine-tune lerobot/smolvla_base on your SO-101 dataset, on one H200 (141 GB).
# Submit from a Futurama login node:
#   DATASET=flyingturtleboop/so101_medicine_bottle_pickplace STEPS=10000 EXCLUDE="[11,14]" sbatch train_smolvla.sbatch
#SBATCH -J smolvla-ft
#SBATCH -p general
#SBATCH --gres=gpu:h200:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
set -euo pipefail

DATASET=${DATASET:?set DATASET=<hf_user>/<dataset>}
STEPS=${STEPS:-20000}
EXCLUDE=${EXCLUDE:-}   # e.g. "[11,14]": episode indices to leave out of training
LEROBOT_REV=e624f3f7f8411ec3a02635d06e79373341e5ef35   # same commit the dataset was recorded with
HOME_DIR=/nethome-instruction/$USER                      # /nethome is a broken symlink on compute nodes
RESULT=$HOME_DIR/models/smolvla-${DATASET##*/}-$SLURM_JOB_ID
WORK=/dev/shm/$USER-$SLURM_JOB_ID

# Keep only the newest weights (~0.9 GB) in home: quota is 15 GB and ~12 GB is used.
save_latest() {
  local last=$WORK/out/checkpoints/last/pretrained_model
  [ -d "$last" ] && mkdir -p "$RESULT" && cp -rL "$last"/. "$RESULT"/ && echo "Saved $(readlink -f "$WORK/out/checkpoints/last") -> $RESULT"
}
trap 'save_latest; rm -rf "$WORK"' EXIT
trap 'echo "Walltime close, saving"; save_latest; exit 1' USR1 TERM INT

mkdir -p "$WORK"
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv \
       TORCHINDUCTOR_CACHE_DIR=$WORK/inductor TRITON_CACHE_DIR=$WORK/triton
UV=$HOME_DIR/.local/bin/uv

git clone -q https://github.com/huggingface/lerobot.git "$WORK/lerobot"
cd "$WORK/lerobot" && git checkout -q "$LEROBOT_REV"
"$UV" sync --locked --extra training --extra smolvla --extra av-dep --python 3.12   # training = dataset + accelerate
nvidia-smi --query-gpu=name,memory.total --format=csv

"$UV" run --no-sync lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  ${EXCLUDE:+--dataset.exclude_episodes=$EXCLUDE} \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps="$STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --save_freq=5000 \
  --num_workers=12 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir="$WORK/out" \
  --job_name="smolvla-${DATASET##*/}" &
wait $!   # run in background so the USR1 trap fires before walltime
