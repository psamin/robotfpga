#!/bin/bash
# One-time KR260 setup for docs/run-smolvla-on-kr260.md, offline-capable (run on the board as root):
#   sudo bash setup_board.sh [OFFLINE_DIR]
# OFFLINE_DIR (optional) holds what fpga/board/deploy_smolvla.py --setup uploads from the laptop:
#   cpython-3.12.*-aarch64-unknown-linux-gnu-install_only.tar.gz   (python-build-standalone)
#   wheels/*.whl, *.tar.gz   (the fork's resolved dependency set for aarch64 / cp312, incl. torch CPU)
#   lerobot_fork/lerobot-0.6.2-py3-none-any.whl                      (psamin/lerobot@e624f3f7)
# Without it, everything comes from the internet as in the guide (uv + git).
# Result: ~/lerobot-venv (Python 3.12) for serve_policy_fpga.py; PYNQ's 3.10 venv keeps engine_server.py.
set -euo pipefail
OFF=${1:-}
USER_NAME=${SUDO_USER:-ubuntu}
HOME_DIR=$(eval echo "~$USER_NAME")
PYNQ_PY=/usr/local/share/pynq-venv/bin/python3

echo "== board"
uname -m; grep -E "MemTotal|SwapTotal|Cma(Total|Free)" /proc/meminfo; cat /proc/cmdline
[ -x "$PYNQ_PY" ] && echo "pynq python: $PYNQ_PY ($($PYNQ_PY --version 2>&1))" || echo "WARNING: $PYNQ_PY missing"

echo "== swap (4 GB file: the policy loads 0.9 GB BF16 while packing 0.7 GB into CMA)"
if [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
swapon --show

echo "== Python 3.12 venv for LeRobot: $HOME_DIR/lerobot-venv"
VENV=$HOME_DIR/lerobot-venv
if [ -n "$OFF" ] && ls "$OFF"/cpython-3.12.*aarch64*.tar.gz >/dev/null 2>&1; then
  mkdir -p "$HOME_DIR/.python312"
  tar -xzf "$OFF"/cpython-3.12.*aarch64*.tar.gz -C "$HOME_DIR/.python312" --strip-components=1
  PY312=$HOME_DIR/.python312/bin/python3.12
else
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME_DIR/.local/bin:$PATH"
  uv python install 3.12
  PY312=$(uv python find 3.12)
fi
"$PY312" -m venv "$VENV"
PIP="$VENV/bin/python -m pip"
$PIP install -q --upgrade pip
if [ -n "$OFF" ] && [ -d "$OFF/wheels" ]; then
  $PIP install --no-index --find-links "$OFF/wheels" --find-links "$OFF/lerobot_fork" "lerobot[smolvla,async]" websockets \
    || $PIP install --find-links "$OFF/wheels" --find-links "$OFF/lerobot_fork" "lerobot[smolvla,async]" websockets
else
  $PIP install "lerobot[smolvla,async] @ git+https://github.com/psamin/lerobot@e624f3f7f8411ec3a02635d06e79373341e5ef35" websockets
fi
chown -R "$USER_NAME" "$VENV" "$HOME_DIR/.python312" 2>/dev/null || true
"$VENV/bin/python" - <<'EOF'
import torch, transformers, lerobot
print("torch", torch.__version__, "| transformers", transformers.__version__, "| lerobot", lerobot.__version__)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
import grpc  # noqa: F401  (policy server transport)
print("smolvla + grpc import ok")
EOF
echo "== done; CMA above must cover 0.71 GB of weights (raise cma= in the kernel args otherwise)"
