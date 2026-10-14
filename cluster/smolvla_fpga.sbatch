#!/bin/bash
#SBATCH -J armlab-smolvla-fpga
#SBATCH -p general
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Fine-tuned SmolVLA with every linear layer computed by the FPGA engine code (issue #148):
# C-model test cases, then the full model on real dataset frames, then the dumped real layers.
#   sbatch --export=ALL,MODEL=$HOME/models/smolvla-<run> cluster/smolvla_fpga.sbatch
source cluster/env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
make -C hls/stream test PY="$UV run --no-sync python"
VLA=$SCR/vla
"$UV" venv -q --python 3.12 "$VLA"
"$UV" pip install -q -p "$VLA" "lerobot[smolvla] @ git+https://github.com/psamin/lerobot@e624f3f7"
OUT=$(fix "$HOME/robotfpga-work/runs/smolvla-fpga")
"$VLA/bin/python" scripts/smolvla_fpga.py --model "$(fix "$MODEL")" --out "$OUT" --frames "${FRAMES:-5}"
for d in "$OUT"/layers/*/; do hls/stream/build/tb_float "$d"; done
