#!/bin/bash
#SBATCH -J armlab-bench-sim
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# MJX vs MuJoCo Warp throughput and CPU parity on the block-sort scene (scripts/bench_sim.py).
#   sbatch cluster/bench_sim.sbatch                       # L40
#   sbatch --gres=gpu:h200:1 cluster/bench_sim.sbatch     # another GPU type
# Results: ~/robotfpga-work/runs/bench_sim-<job>.json. grep the log for "overflow" before trusting Warp.
UV_EXTRAS="--extra rl"
source cluster/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$UV" run --no-sync python scripts/bench_sim.py --out "$(fix "$HOME/robotfpga-work/runs")/bench_sim-$SLURM_JOB_ID.json" ${ARGS:-}
