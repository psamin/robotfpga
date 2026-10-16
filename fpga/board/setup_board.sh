#!/bin/bash
# One-time KR260 setup for running SmolVLA with the float engine (run on the board as root):
#   sudo bash setup_board.sh [http://192.168.137.1:8000]   # optional: laptop file server with wheels
# Installs PyTorch (CPU, aarch64), LeRobot 0.6.1 with the smolvla extra and the pinned transformers
# into PYNQ's Python, adds 4 GB of swap (the model is 0.9 GB BF16 + 0.7 GB in CMA while packing),
# and prints the CMA size: the weights need >= 0.71 GB plus activation buffers; below ~1 GB add
# cma=1536M to the kernel command line and reboot.
set -euo pipefail
PY=/usr/local/share/pynq-venv/bin/python3
WHEELS=${1:-}

echo "== CMA / memory"
grep -E "Cma|MemTotal|SwapTotal" /proc/meminfo
cat /proc/cmdline

echo "== swap"
if [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
swapon --show

echo "== python packages"
if [ -n "$WHEELS" ]; then
  $PY -m pip install --no-index --find-links "$WHEELS" torch || true
fi
$PY -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
$PY -m pip install "lerobot[smolvla]==0.6.1" "huggingface_hub" "safetensors"
$PY - <<'EOF'
import torch, transformers, lerobot
print("torch", torch.__version__, "threads", torch.get_num_threads())
print("transformers", transformers.__version__, "lerobot", lerobot.__version__)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
print("smolvla import ok")
EOF
echo "== done"
