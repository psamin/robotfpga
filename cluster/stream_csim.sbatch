#!/bin/bash
#SBATCH -J armlab-stream-csim
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# C simulation of the DDR weight-streaming engine (hls/stream, issue #148): v3r2 golden vectors,
# TinyPolicy-L random model, linear-mode GEMM cases, and SmolVLA layer dumps if present.
source cluster/env.sh
make -C hls/stream test PY="$UV run --no-sync python"
for d in $(fix "$HOME/robotfpga-work/runs")/smolvla-int8/layers/*/; do
  [ -f "$d/dims.txt" ] && hls/stream/build/tb_stream linear "$d"
done
true
