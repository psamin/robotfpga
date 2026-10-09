#!/bin/bash
# Serve a trained policy on a cluster GPU for real-robot runs (LeRobot async inference).
# The server listens on the compute node's localhost only (its transport uses pickle, which
# must not be reachable from the shared cluster network); the Mac reaches it through an SSH
# tunnel (~/so101/run_policy.sh sets that up).
# Submit from a Futurama login node:
#   MODEL=/nethome-instruction/pi34/models/<dir> sbatch serve_policy.sbatch
#SBATCH -J smolvla-serve
#SBATCH -p general
#SBATCH --gres=gpu:h200:1   # the tunnel needs SSH into the job's node; on the L40S node (zapp) SSH
#                              login fails because /nethome is a broken symlink there (no authorized_keys)
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /nethome-instruction/pi34/logs/%x-%j.log
set -euo pipefail

MODEL=${MODEL:?set MODEL=<path to a pretrained_model dir>}
PORT=${PORT:-8080}
LEROBOT_REV=e624f3f7f8411ec3a02635d06e79373341e5ef35   # same commit the model was trained with
HOME_DIR=/nethome-instruction/$USER                      # /nethome is a broken symlink on compute nodes
WORK=/dev/shm/$USER-$SLURM_JOB_ID
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv
UV=$HOME_DIR/.local/bin/uv

git clone -q https://github.com/huggingface/lerobot.git "$WORK/lerobot"
cd "$WORK/lerobot" && git checkout -q "$LEROBOT_REV"
"$UV" sync --locked --extra smolvla --extra async --extra dataset --python 3.12
[ -f "$MODEL/config.json" ] || { echo "No model at $MODEL"; exit 1; }
echo "SERVING $MODEL on $(hostname):$PORT (localhost only)"
"$UV" run --no-sync python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port="$PORT" --fps=30
