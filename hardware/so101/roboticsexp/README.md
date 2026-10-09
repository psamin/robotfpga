# SO-101 real-robot experiments

Data collection, camera setup and policy runs for an SO-101 leader/follower arm with LeRobot,
plus the Futurama (GT) cluster jobs that train and serve the policies. The LeRobot checkout these
scripts run against is `~/Documents/GitHub/lerobot` (commit `e624f3f7`), environment in its `.venv`.

## Datasets and models

| Kind | Name | Notes |
|---|---|---|
| Dataset | [flyingturtleboop/so101_medicine_bottle_pickplace](https://huggingface.co/datasets/flyingturtleboop/so101_medicine_bottle_pickplace) ([viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fflyingturtleboop%2Fso101_medicine_bottle_pickplace%2Fepisode_0)) | 15 episodes, 7,824 frames, 30 fps, cameras `front` (C920, top) + `wrist`. Collected with demo-then-replay. Episodes 11 and 14 have a bad start (hand in frame / bottle already on plate) and are excluded from training. |
| Model | SmolVLA fine-tune, job 9604 | `lerobot/smolvla_base`, 10k steps, batch 64, H200, episodes 11/14 excluded, final loss 0.024. On the cluster at `~/models/smolvla-so101_medicine_bottle_pickplace-9604/` (866 MB). Server check (`test_policy_server.py`): predicted first action within 0.64° mean / 1.5° max of the recording. On the arm: misses the bottle so far (0/1 marked); live targets keep exceeding the 10°/step cap, under investigation. |

## Files

| File | What it does |
|---|---|
| `env.sh` | Ports, cameras, dataset name, task, episode length. Edit settings here. |
| `camera_presets.py` | `lock`: let auto settle, then freeze C920 focus + white balance (exposure stays auto). `watch` re-applies it during a session. |
| `teleop.sh` | Teleoperate with locked cameras (`VIEW=true` for the live viewer). |
| `collect.sh` | `live N`: record teleop directly. `replay N`: teleop a demo, then the follower replays it while the cameras record. |
| `demo_record.py`, `record_session.py`, `replay_record.py` | The recorders behind `collect.sh`, built on LeRobot's `record_loop` and dataset writer. |
| `keys.py`, `VOICE.md` | Arrow-key controls and the macOS Voice Control phrases that press them. |
| `policy.sh` | `start`: run a trained policy (model on a cluster GPU, this Mac drives the arm through an SSH tunnel). `stop` or Ctrl-C: the arm glides back to where it started, motors off, server cancelled. |
| `home.py` | Save the arm's pose before a run and glide back to it afterwards, without dropping the arm. |
| `rollout.sh`, `rollout.py` | Rollout session: keeps the GPU server and tunnel up; `begin` / `rest` / `success` / `fail` / `quit` (arrow keys, Voice Control, or `./rollout.sh <command>`). Logs results to `logs/rollouts-*.csv`. Motion settings in `presets/rollout_settings.json`, re-read at every rollout. |
| `smooth_client.py` | LeRobot's robot client plus command damping (low-pass on joint targets, `SMOOTH_ALPHA`). |
| `cluster/train_smolvla.sbatch` | Fine-tune SmolVLA on a dataset (H200). |
| `cluster/serve_policy.sbatch` | Serve a trained policy on the H200, localhost-only, for `policy.sh`. |
| `test_policy_server.py` | Check a running server end to end with a recorded frame, without moving the arm. |

`~/so101` is a symlink to this folder.

## Typical session

```bash
cd ~/so101
~/Documents/GitHub/lerobot/.venv/bin/python camera_presets.py lock   # once per session, nothing else running
./collect.sh replay 15        # say "command mode", then follow the prompts (VOICE.md)
```

Train on the cluster (login node, GT VPN):
```bash
DATASET=flyingturtleboop/<dataset> STEPS=10000 EXCLUDE="[11,14]" sbatch ~/train_smolvla.sbatch
```

## Things that went wrong, and the fix in place

| Symptom | Cause | Fix |
|---|---|---|
| Control loop drops to 4–10 Hz, cameras time out while recording | CPU AV1 encoding (`libsvtav1`), 63 ms CPU per frame per camera | `VCODEC=h264_videotoolbox` (Mac hardware encoder, 1.5 ms) |
| Teleop stalls or freezes | rerun live viewer (up to 76% CPU, grows without bound); stale viewers reused across runs | Viewer off while recording; `teleop.sh` starts a fresh one |
| Washed-out top camera after locking exposure | C920 manual exposure doesn't hold once a stream opens on macOS | Lock focus + white balance only |
| Wrist camera dark | Its auto mode uses a hidden gain; manual mode can't reach it, and only a USB replug restores auto | Leave wrist exposure on auto |
| Voice "record take" discarded the demo | Voice Control dictated the words; the typed `r` was LeRobot's redo shortcut | Arrow keys only, letters ignored; distinct "robot …" phrases |
| Tunnel to the policy server fails on the L40S node | `/nethome` is a broken symlink on some compute nodes, so sshd can't read `authorized_keys` | Serve on the H200 node (`angleyne`), where SSH works |
| Arm stuck in its folded rest pose during a rollout | Softer servo gains (P12/D40) plus heavy damping: not enough force to lift against gravity under the 50% torque limit | Back to LeRobot's P16/I0/D32, damping 0.7 |
| Trained policy would get the wrong camera names | `policy_server` overrides the saved rename map with the client's empty one | Policy runs name cameras `camera1`/`camera2`, as in training |
