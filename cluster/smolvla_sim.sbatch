#!/bin/bash
#SBATCH -J smolvla-sim
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# Baseline: the unchanged fine-tuned SmolVLA in the pick-and-place sim (policy/sim/smolvla_sim.py).
# Needs the calibrated cameras JSON from the photo session. The checkpoint is only read.
#   sbatch --export=ALL,MODEL=$HOME/models/smolvla-so101_medicine_bottle_pickplace-9604,CAMERAS=robot/calibration/cameras.json cluster/smolvla_sim.sbatch
source cluster/env.sh
VLA=$SCR/vla
"$UV" venv -q --python 3.12 "$VLA"
"$UV" pip install -q -p "$VLA" "lerobot[smolvla] @ git+https://github.com/psamin/lerobot@e624f3f7" "mujoco==3.14.0"
OUT=$(fix "$HOME/robotfpga-work/runs")/smolvla-sim-$SLURM_JOB_ID.json
PYTHONPATH=$REPO "$VLA/bin/python" -m policy.sim.smolvla_sim --model "$(fix "${MODEL:?set MODEL}")" \
  --cameras "${CAMERAS:?set CAMERAS}" --out "$OUT" ${ARGS:-}
