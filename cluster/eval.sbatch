#!/bin/bash
#SBATCH -J armlab-eval
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 64
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# Closed-loop eval; the GPU is only used for EGL rendering. Flags for armlab.eval.run via ARGS.
#   sbatch --export=ALL,ARGS="--ckpt <...> --stage B --episodes 50 --out <...>" cluster/eval.sbatch
source cluster/env.sh
"$UV" run python -m armlab.eval.run --workers "${SLURM_CPUS_PER_TASK:-8}" ${ARGS}
