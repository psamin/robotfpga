#!/bin/bash
#SBATCH -J armlab-test
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Lint + full test suite on a compute node (keeps the laptop cool).
source cluster/env.sh
"$UV" run ruff check
"$UV" run pytest -q
