#!/bin/bash
#SBATCH -J teacher
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# SGS teacher training for pick and place (policy/rl/train_teacher.py). Builds the reset buffer on the
# node first if it is missing (CPU, about 10 min). Approved per run on #156.
#   sbatch --export=ALL,STRATEGY=sgs,SEED=0,ARGS="--envs-per-variant 64 --iters 50" cluster/train_teacher.sbatch
#   sbatch --export=ALL,STRATEGY=uniform,SEED=0 cluster/train_teacher.sbatch
# The log's steps/s line is the GPU throughput check; grep it for "overflow" before long runs.
export UV_EXTRAS="--group rl --group sim --group gpu"
source cluster/env.sh
STRATEGY=${STRATEGY:-sgs}
SEED=${SEED:-0}
WORK=$(fix "$HOME/robotfpga-work")
BUFFER=${BUFFER:-$WORK/resets/v1.npz}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
[ -f "$BUFFER" ] || "$UV" run --no-sync python -m policy.sim.buffer --out "$BUFFER" --variants 8 --per-variant 2048 \
  --seed 0 --workers 16
"$UV" run --no-sync python -m policy.rl.train_teacher --buffer "$BUFFER" --strategy "$STRATEGY" --seed "$SEED" \
  --out "$WORK/runs/teacher-$STRATEGY-s$SEED-$SLURM_JOB_ID" ${ARGS:-}
