# Shared settings for teleop.sh and collect.sh. Edit here, not in the scripts.
SO101=$HOME/so101
LEROBOT=$HOME/Documents/GitHub/lerobot
BIN=$LEROBOT/.venv/bin
PY=$BIN/python
export PYTHONUNBUFFERED=1   # logs update live instead of in buffered chunks

FOLLOWER_PORT=/dev/tty.usbmodem5B790811541
LEADER_PORT=/dev/tty.usbmodem5B901030671
# Camera names become dataset features (observation.images.front / .wrist); keep them fixed.
# OpenCV indexes can shift after replugging: check with `lerobot-find-cameras opencv`.
# warmup_s 3 (default 1): the C920 can take over a second to deliver its first frame right after
# another program released it, which crashed teleop at startup.
CAMERAS="{ wrist: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}}"
CAMERAS_JSON='{"wrist": {"index_or_path": 1, "width": 480, "height": 640, "fps": 30, "rotation": 90, "warmup_s": 3}, "front": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30, "warmup_s": 3}}'

DATASET=flyingturtleboop/so101_medicine_bottle_pickplace
TASK=${TASK:-"Pick up the medicine bottle and place it on the plate"}   # override: TASK="..." ./collect.sh ...
EPISODE_S=20   # max seconds per episode (say "robot done" to end one early); no reset timer: each episode waits for "start"
# Mac hardware video encoder: 1.5 ms CPU/frame vs 63 ms for the default libsvtav1, which
# starved the control loop to 10 Hz and made the cameras time out while recording.
VCODEC=h264_videotoolbox

FOLLOWER_ARGS=(--robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=my_follower)
LEADER_ARGS=(--teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=my_leader)

# Refuse to start without a locked camera preset, then keep it applied for the whole run.
start_camera_lock() {
  if [ ! -f "$SO101/presets/latest.json" ]; then
    echo "No camera preset. Run: $PY $SO101/camera_presets.py lock" >&2
    exit 1
  fi
  echo "Camera preset: $(grep '"created"' "$SO101/presets/latest.json")"
  nice -n 10 "$PY" "$SO101/camera_presets.py" watch &   # low priority: never compete with the control loop
}

dataset_exists() { [ -f "$HOME/.cache/huggingface/lerobot/$DATASET/meta/info.json" ]; }
