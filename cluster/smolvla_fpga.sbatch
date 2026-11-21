#!/bin/bash
#SBATCH -J robotfpga-smolvla-fpga
#SBATCH -p general
#SBATCH -c 64
#SBATCH -w nvidia-l4   # CPU-only nodes fail "user env retrieval"; the GPU nodes load it fine
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Fine-tuned SmolVLA with every linear layer computed by the FPGA engine code (issue #148):
# C-model test cases, then the full model on real dataset frames, then the dumped real layers.
#   sbatch --export=ALL,MODEL=$HOME/models/smolvla-<run> cluster/smolvla_fpga.sbatch
source cluster/env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
make -C fpga/hls test PY="$UV run --no-sync python"
VLA=$SCR/vla
"$UV" venv -q --python 3.12 "$VLA"
"$UV" pip install -q -p "$VLA" "lerobot[smolvla,dataset] @ git+https://github.com/psamin/lerobot@e624f3f7"
OUT=$(fix "${OUT:-$HOME/robotfpga-work/runs/smolvla-fpga}")  # OUT=<dir> keeps another checkpoint's batches
"$VLA/bin/python" policy/smolvla_fpga.py --model "$(fix "$MODEL")" --out "$OUT" --frames "${FRAMES:-5}" \
  --runtime layout --dump-batches
for d in "$OUT"/layers/*/; do fpga/hls/build/tb_float "$d"; done
# The board-side scripts, run here with the host C core: engine check on every case, then the
# fine-tuned SmolVLA replaying the dumped frames through fpga/board/run_smolvla.py.
"$VLA/bin/python" fpga/board/check_engine.py --mock layout build/stream/float
"$VLA/bin/python" fpga/board/check_engine.py --mock layout "$OUT/layers"
"$VLA/bin/python" fpga/board/run_smolvla.py --mock layout --model "$(fix "$MODEL")" --batches "$OUT/batches"
