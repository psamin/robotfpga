#!/bin/bash
#SBATCH -J armlab-collect
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 48
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
# Expert demonstrations with DART noise and Stage C perturbations. EGL renders on the L4.
#   sbatch --export=ALL,OUT=~/robotfpga-work/datasets/v1,EPI=500 cluster/collect.sbatch
source cluster/env.sh
OUT=${OUT:-$HOME/robotfpga-work/datasets/v1}
"$UV" run python -m armlab.data.collect --out "$(fix "$OUT")" --episodes-per-instr "${EPI:-500}" \
  --seed0 "${SEED0:-0}" --workers "${SLURM_CPUS_PER_TASK:-8}"
