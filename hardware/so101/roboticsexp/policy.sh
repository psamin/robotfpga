#!/bin/bash
# Run the trained policy on the SO-101. The model runs on a Futurama GPU (never on this Mac);
# this Mac only drives the arm and cameras (LeRobot async inference over an SSH tunnel).
#
#   ./policy.sh start   start the policy server if none is running, connect, save the arm's
#                       current pose, run the policy. Start with the arm at rest.
#   ./policy.sh stop    (or Ctrl-C in the start terminal) stop the policy: the arm glides back to
#                       the pose it started from, the motors turn off, the server job is cancelled.
#
# Needs GT VPN. MODEL=<dir> overrides the model, KEEP_SERVER=1 keeps the server running after a stop,
# MAX_STEP=<deg> caps each joint's move per control step (default 10). Smoothing: SMOOTH_ALPHA (command
# low-pass, default 0.7), P_GAIN / I_GAIN / D_GAIN (servo PID, default 16 / 0 / 32).
set -uo pipefail
source "$(dirname "$0")/env.sh"
LOGIN=pi34@planetexpress-login.cc.gatech.edu
PORT=8080

if [ "${1:-}" = stop ]; then
  pkill -INT -f "smooth_client.py|async_inference.robot_client" && echo "Stopping the policy; the arm returns to its start pose." \
    || echo "No policy is running."
  exit 0
fi
[ "${1:-}" = start ] || { echo "usage: ./policy.sh start | stop" >&2; exit 1; }

mkdir -p "$SO101/logs"
LOG=$SO101/logs/policy-$(date +%Y%m%d-%H%M%S).log
# The policy server overrides the camera-name mapping saved at training time with an empty one
# (policy_server.py, rename_observations_processor), so name the cameras exactly as the model saw
# them in training: front -> camera1, wrist -> camera2. Same devices and settings as recording.
POLICY_CAMERAS="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}, camera2: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}}"

MODEL=${MODEL:-$(ssh -o BatchMode=yes $LOGIN 'ls -td /nethome/pi34/models/smolvla-* 2>/dev/null | head -1' 2>/dev/null | sed 's|^/nethome/|/nethome-instruction/|')}
[ -n "$MODEL" ] || { echo "No trained model found in ~/models on the cluster." >&2; exit 1; }
echo "Model: $MODEL"

# 1. Policy server (1x H200, 2 h limit): reuse a running one or start one.
JOB=$(ssh -o BatchMode=yes $LOGIN "squeue -u pi34 -n smolvla-serve -h -o %i | head -1" 2>/dev/null)
if [ -z "$JOB" ]; then
  JOB=$(ssh -o BatchMode=yes $LOGIN "MODEL=$MODEL sbatch --parsable ~/serve_policy.sbatch" 2>/dev/null | tail -1)
  echo "Started policy server job $JOB (1x H200)."
fi
echo "Waiting for the policy server (job $JOB) ..."
for _ in $(seq 90); do
  ssh -o BatchMode=yes $LOGIN "grep -q SERVING /nethome/pi34/logs/smolvla-serve-$JOB.log" 2>/dev/null && break
  sleep 5
done
NODE=$(ssh -o BatchMode=yes $LOGIN "scontrol show node \$(squeue -j $JOB -h -o %N) | grep -oE 'NodeHostName=\\S+' | cut -d= -f2")
[ -n "$NODE" ] || { echo "Policy server job $JOB is not running." >&2; exit 1; }
sleep 5  # the server starts listening right after it prints SERVING

# 2. Tunnel: Mac localhost:$PORT -> the compute node's localhost:$PORT, through the login node.
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -J $LOGIN -L $PORT:127.0.0.1:$PORT pi34@$NODE &
TUNNEL=$!
sleep 4
kill -0 $TUNNEL 2>/dev/null || { echo "SSH tunnel to $NODE failed." >&2; exit 1; }

# 3. Remember where the arm starts, and always return there and clean up, however the run ends.
"$PY" "$SO101/home.py" save || { kill $TUNNEL; exit 1; }
cleanup() {
  trap - INT TERM EXIT
  echo "Returning the arm to its start pose ..."
  "$PY" "$SO101/home.py" go
  kill $TUNNEL 2>/dev/null
  if [ -z "${KEEP_SERVER:-}" ]; then
    ssh -o BatchMode=yes $LOGIN "scancel $JOB" 2>/dev/null && echo "Policy server job $JOB cancelled."
  fi
}
trap cleanup EXIT
trap 'echo "Stopping the policy ..."' INT TERM

start_camera_lock
# disable_torque_on_disconnect=false keeps the arm held when the policy stops, until home.py takes over.
# max_relative_target caps each joint's move per control step (degrees): a safety limit for first runs.
"$PY" "$SO101/smooth_client.py" \
  --server_address=127.0.0.1:$PORT \
  "${FOLLOWER_ARGS[@]}" "--robot.cameras=$POLICY_CAMERAS" --robot.max_relative_target=${MAX_STEP:-10} \
  --robot.disable_torque_on_disconnect=false \
  --robot.position_p_coefficient=${P_GAIN:-16} --robot.position_i_coefficient=${I_GAIN:-0} --robot.position_d_coefficient=${D_GAIN:-32} \
  --task="$TASK" --policy_type=smolvla --pretrained_name_or_path="$MODEL" --policy_device=cuda \
  --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30 \
  2>&1 | tee "$LOG"
