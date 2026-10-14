#!/bin/bash
#SBATCH -J armlab-smolvla-int8
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# SmolVLA int8 linear-offload check (scripts/smolvla_int8.py, issue #148). Uses its own venv with the
# LeRobot fork the model was trained with, so it doesn't touch the armlab environment.
#   sbatch --export=ALL,MODEL=$HOME/models/smolvla-<run> cluster/smolvla_int8.sbatch
source cluster/env.sh
VLA=$SCR/vla
"$UV" venv -q --python 3.12 "$VLA"
"$UV" pip install -q -p "$VLA" "lerobot[smolvla] @ git+https://github.com/psamin/lerobot@e624f3f7"
OUT=$(fix "$HOME/robotfpga-work/runs/smolvla-int8")
"$VLA/bin/python" scripts/smolvla_int8.py --model "$(fix "$MODEL")" --out "$OUT" --frames "${FRAMES:-40}"
make -C hls/stream build/tb_stream
for d in "$OUT"/layers/*/; do hls/stream/build/tb_stream linear "$d"; done
