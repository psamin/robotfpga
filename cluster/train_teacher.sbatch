#!/bin/bash
#SBATCH -J armlab-teacher
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# SGS teacher training (armlab/rl/train_teacher.py). Builds the reset buffer first if it is missing.
#   sbatch --export=ALL,STRATEGY=sgs,SEED=0 cluster/train_teacher.sbatch
#   sbatch --export=ALL,STRATEGY=uniform,SEED=0,ARGS="--envs 1024 --iters 500" cluster/train_teacher.sbatch
UV_EXTRAS="--extra rl"
source cluster/env.sh
STRATEGY=${STRATEGY:-sgs}
SEED=${SEED:-0}
WORK=$(fix "$HOME/robotfpga-work")
BUFFER=${BUFFER:-$WORK/resets/v1.npz}
[ -f "$BUFFER" ] || "$UV" run --no-sync python -m armlab.rl.buffer --out "$BUFFER" --size 32768 --seed 0 --workers 16
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$UV" run --no-sync python -m armlab.rl.train_teacher --buffer "$BUFFER" --strategy "$STRATEGY" --seed "$SEED" \
  --out "$WORK/runs/teacher-$STRATEGY-s$SEED-$SLURM_JOB_ID" ${ARGS:-}
