#!/bin/bash
# Teleoperate with both cameras locked. The live viewer costs a lot of CPU and causes control-loop
# stalls, so it is off by default: VIEW=true ./teleop.sh for a short framing check.
# Log: ~/so101/logs/
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$SO101/logs"
LOG=$SO101/logs/teleop-$(date +%Y%m%d-%H%M%S).log

# A viewer left over from an earlier run fills up and blocks teleop ("Sender has been blocked").
# Start each run with a fresh one, and close it when teleop exits.
close_viewer() { pkill -f "$LEROBOT/.venv/.*rerun.* --port=9876" 2>/dev/null || true; }
close_viewer
trap close_viewer EXIT

start_camera_lock
"$BIN/lerobot-teleoperate" "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" "--robot.cameras=$CAMERAS" \
  --fps=30 --display_data=${VIEW:-false} "$@" 2>&1 | tee "$LOG" | grep --line-buffered -v "Teleop loop time"
# The per-loop timing line goes to the log only: redrawing it 30x/s in the terminal costs CPU.
