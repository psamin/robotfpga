#!/bin/bash
#SBATCH -J armlab-qat
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#   sbatch --export=ALL,FLOAT=<ckpt>,NAME=qat-v1,ARGS="--steps 10000" cluster/qat.sbatch
source cluster/env.sh
DATA=$(fix "${DATA:-$HOME/robotfpga-work/datasets/v1}")
OUTDIR=$(fix "$HOME/robotfpga-work/runs/${NAME:-qat}")
"$UV" run python -m armlab.policy.qat_train --data "$DATA" --float-ckpt "$(fix "$FLOAT")" --out "$OUTDIR" ${ARGS:-}
