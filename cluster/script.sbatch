#!/bin/bash
#SBATCH -J armlab-script
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# Run one python command from the repo on a compute node: sbatch --export=ALL,CMD="scripts/x.py --flag" ...
source cluster/env.sh
"$UV" run python $CMD
