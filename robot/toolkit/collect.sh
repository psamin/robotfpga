#!/bin/bash
# Collect episodes into $DATASET (env.sh). Creates it the first time, adds to it after that.
#
#   ./collect.sh live 15     teleop while both cameras record
#   ./collect.sh replay 15   one at a time: teleop a demo, put the bottle back, say "record", and the
#                            follower replays that demo by itself while the cameras record
#
# Every episode waits for you. Controls (keys, or voice via macOS Voice Control, see VOICE.md):
#   start  (-> or n)       begin the demo / live episode (until then the arm follows the leader, unrecorded)
#   next   (-> or n)       end the demo / live episode early (otherwise it ends after $EPISODE_S s)
#   record (up)            replay mode: replay the demo while the cameras record
#   redo   (<- or r)       throw away the demo or recording and do it again
#   stop   (down, Esc, q)  keep finished episodes, discard one in progress, upload, exit
# Keys only reach LeRobot while this terminal is focused, unless your terminal app has macOS
# Accessibility + Input Monitoring permission (see VOICE.md).
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:?usage: ./collect.sh live|replay <num_episodes>}
EPISODES=${2:?usage: ./collect.sh live|replay <num_episodes>}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$SO101/logs"
LOG=$SO101/logs/collect-$MODE-$STAMP.log
cp "$SO101/presets/latest.json" "$SO101/logs/collect-$STAMP-camera-preset.json" 2>/dev/null || true
echo "Dataset $DATASET ($(dataset_exists && echo "adding episodes" || echo "new")), task: \"$TASK\""
SESSION=("$PY" "$SO101/record_session.py" --task "$TASK" --num-episodes "$EPISODES" --episode-s "$EPISODE_S"
         --follower-port "$FOLLOWER_PORT" --leader-port "$LEADER_PORT" --vcodec "$VCODEC")

if [ "$MODE" = live ]; then
  start_camera_lock
  "${SESSION[@]}" --repo-id "$DATASET" --cameras "$CAMERAS_JSON" --push-to-hub 2>&1 | tee "$LOG"

elif [ "$MODE" = replay ]; then
  start_camera_lock
  "$PY" "$SO101/demo_record.py" --repo-id "$DATASET" --task "$TASK" --num-episodes "$EPISODES" \
    --episode-s "$EPISODE_S" --follower-port "$FOLLOWER_PORT" --leader-port "$LEADER_PORT" \
    --cameras "$CAMERAS_JSON" --vcodec "$VCODEC" --push-to-hub 2>&1 | tee "$LOG"
else
  echo "Mode must be live or replay" >&2; exit 1
fi
