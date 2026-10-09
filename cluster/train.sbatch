#!/bin/bash
#SBATCH -J armlab-train
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# Behavior cloning on one L4. Extra train.py flags via ARGS.
#   sbatch --export=ALL,DATA=~/robotfpga-work/datasets/v1,NAME=bc-v1,ARGS="--steps 40000" cluster/train.sbatch
source cluster/env.sh
DATA=$(fix "${DATA:-$HOME/robotfpga-work/datasets/v1}")
OUTDIR=$(fix "$HOME/robotfpga-work/runs/${NAME:-bc}")
"$UV" run python -m armlab.policy.train --data "$DATA" --out "$OUTDIR" ${ARGS:-}
