#!/bin/bash
#SBATCH -J robotfpga-test
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Lint + the FPGA engine C model on a compute node (keeps the laptop cool).
source cluster/env.sh
"$UV" run ruff check
make -C fpga/hls test PY="$UV run --no-sync python"
