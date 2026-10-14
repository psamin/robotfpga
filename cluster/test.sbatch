#!/bin/bash
#SBATCH -J robotfpga-test
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Lint, the FPGA engine C model and the Python tests (policy/rl, policy/calib, policy/sim) on a compute node.
export UV_EXTRAS="--group rl --group calib --group sim"
source cluster/env.sh
"$UV" run --no-sync ruff check
make -C fpga/hls test PY="$UV run --no-sync python"
"$UV" run --no-sync pytest -q
