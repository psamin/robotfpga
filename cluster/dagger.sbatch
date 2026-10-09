#!/bin/bash
#SBATCH -J armlab-dagger
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# Full DAgger loop on one L4 node: collection and eval use the CPUs (EGL rendering on the GPU),
# training uses the GPU.
#   sbatch --export=ALL,NAME=dagger-v1,ARGS="--expert-data <dir> --rounds 4" cluster/dagger.sbatch
source cluster/env.sh
OUTDIR=$(fix "$HOME/robotfpga-work/runs/${NAME:-dagger}")
"$UV" run python -m armlab.policy.dagger --out "$OUTDIR" --workers "${SLURM_CPUS_PER_TASK:-8}" ${ARGS}
