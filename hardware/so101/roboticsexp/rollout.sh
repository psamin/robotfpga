#!/bin/bash
# Policy rollouts on the SO-101 (see rollout.py for details).
#   ./rollout.sh                         start a session (needs GT VPN; start with the arm at rest)
#   ./rollout.sh begin|rest|success|fail|quit   send a command to the running session
set -euo pipefail
SO101=$(cd "$(dirname "$0")" && pwd)
if [ $# -gt 0 ]; then
  pgrep -f "so101/rollout.py" >/dev/null || { echo "No rollout session is running. Start one with ./rollout.sh" >&2; exit 1; }
  echo "$*" >> "$SO101/presets/rollout_cmd"
  echo "sent: $*"
  exit 0
fi
source "$SO101/env.sh"
export PYTHONUNBUFFERED=1
"$PY" "$SO101/rollout.py" 2>&1 | tee "$SO101/logs/rollout-session-$(date +%Y%m%d-%H%M%S).log"
