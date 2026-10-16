#!/bin/bash
#SBATCH -J kr260-mock
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# The KR260 software stack without a board: engine_server.py with the host C core (--mock layout, the same
# core HLS synthesizes and the same DDR word offsets), serve_policy_fpga.py --selftest against the PyTorch FP32
# reference chunks, then the real gRPC policy server queried by robot/toolkit/test_policy_server.py with a
# recorded frame. CPU only (the GPU request only gets a node whose user environment loads).
#   sbatch cluster/kr260_mock_test.sbatch
set -euo pipefail
fix() { [ -e "$1" ] && echo "$1" || echo "${1/#\/nethome\//\/nethome-instruction\/}"; }
HOME_DIR=$(fix "$HOME")
REPO=$(fix "$SLURM_SUBMIT_DIR")
MODEL=${MODEL:-$HOME_DIR/models/smolvla-so101_medicine_bottle_pickplace-9604}
BATCHES=${BATCHES:-$HOME_DIR/robotfpga-work/runs/smolvla-fpga/batches}
WORK=/dev/shm/$USER-$SLURM_JOB_ID
SOCK=$WORK/fpga-engine.sock
LEROBOT_REV=e624f3f7f8411ec3a02635d06e79373341e5ef35
mkdir -p "$WORK"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$WORK"' EXIT
export HF_HOME=$WORK/hf UV_CACHE_DIR=$WORK/uv-cache UV_PROJECT_ENVIRONMENT=$WORK/venv CUDA_VISIBLE_DEVICES=""
UV=$HOME_DIR/.local/bin/uv
echo "job $SLURM_JOB_ID on $(hostname) repo=$REPO git=$(git -C "$REPO" rev-parse --short HEAD)"
git clone -q https://github.com/psamin/lerobot.git "$WORK/lerobot"
cd "$WORK/lerobot" && git checkout -q "$LEROBOT_REV"
"$UV" sync -q --locked --extra smolvla --extra async --extra dataset --python 3.12
PY="$UV run --no-sync python"
make -C "$REPO/fpga/hls" lib >/dev/null

$PY "$REPO/fpga/board/engine_server.py" --mock layout --socket "$SOCK" &
until [ -S "$SOCK" ]; do sleep 1; done
echo "== selftest"
$PY "$REPO/fpga/board/serve_policy_fpga.py" --socket "$SOCK" --selftest "$MODEL" --batches "$BATCHES" --threads 16

echo "== gRPC policy server + test client"
$PY "$REPO/fpga/board/serve_policy_fpga.py" --socket "$SOCK" --port 8099 > "$WORK/server.log" 2>&1 &
until grep -q "PolicyServer started" "$WORK/server.log"; do sleep 1; done
$PY -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('flyingturtleboop/so101_medicine_bottle_pickplace')" >/dev/null
$PY "$REPO/robot/toolkit/test_policy_server.py" --model "$MODEL" --server 127.0.0.1:8099
grep -E "linear layers on the FPGA|Time taken" "$WORK/server.log"
