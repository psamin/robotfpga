"""Record a dataset by replaying draft teleop episodes on the follower (no hands in the frame).

Step 1 (collect.sh replay) records draft episodes with teleop and no cameras: joint commands only.
Step 2 (this script) replays each draft on the follower while both cameras record, and saves
frames through LeRobot's own record_loop, so the result is identical in format to lerobot-record.

Per episode: the follower glides to the draft's start pose and holds; you reset the scene, then
say/press "start" (n or ->) to replay and record. During the replay: "redo" (r or <-) discards it and
waits for "start" again, "stop" (q or Esc) ends the session after saving finished episodes.

The replay is open loop: put the object exactly where it was during the draft.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import init_logging, log_say

from keys import init_listener

FPS = 30
APPROACH_S = 3.0  # glide time from the current pose to an episode's start pose


def load_drafts(root: Path) -> tuple[list[str], list[np.ndarray]]:
    """Action names and one (frames, joints) array per draft episode."""
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    table = pq.read_table(sorted((root / "data").rglob("*.parquet")), columns=["episode_index", "frame_index", "action"])
    ep, frame = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    episodes = []
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]
        episodes.append(actions[idx[np.argsort(frame[idx])]])
    return names, episodes


class ReplayTeleop(Teleoperator):
    """Stands in for the leader arm: plays draft actions back one per control tick."""

    name = "replay"

    def __init__(self, robot, names: list[str], events: dict):
        self.robot, self.names, self.events = robot, names, events
        self.mode, self.traj, self.cursor = "hold", None, 0
        self.start = self.target = None
        self.t0 = 0.0

    def glide_to(self, pose: np.ndarray):
        pos = self.robot.bus.sync_read("Present_Position")
        self.start = np.array([pos[n.removesuffix(".pos")] for n in self.names], dtype=np.float32)
        self.target, self.t0, self.mode = pose, time.perf_counter(), "glide"

    def play(self, traj: np.ndarray):
        self.traj, self.cursor, self.mode = traj, 0, "play"

    def get_action(self):
        if self.mode == "glide":
            a = min(1.0, (time.perf_counter() - self.t0) / APPROACH_S)
            vec = self.start + a * (self.target - self.start)
        else:  # play
            vec = self.traj[min(self.cursor, len(self.traj) - 1)]
            self.cursor += 1
            if self.cursor >= len(self.traj):
                self.events["exit_early"] = True  # trajectory finished: end the episode
        return {n: float(v) for n, v in zip(self.names, vec)}

    # The rest of the Teleoperator interface is unused by record_loop.
    @property
    def action_features(self):
        return {n: float for n in self.names}

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return True

    def connect(self, calibrate: bool = True):
        pass

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft-root", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", default="my_follower")
    p.add_argument("--cameras", required=True, help="JSON: {name: {index_or_path, width, height, fps, rotation}}")
    p.add_argument("--vcodec", default="h264_videotoolbox", help="h264_videotoolbox = Mac hardware encoder")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="smoke tests only: skip waiting for 'next'")
    args = p.parse_args()
    init_logging()

    names, drafts = load_drafts(args.draft_root)
    logging.info(f"{len(drafts)} draft episodes, {sum(len(d) for d in drafts) / FPS:.0f}s of motion")
    cameras = {k: OpenCVCameraConfig(**{**v, "rotation": Cv2Rotation(v.get("rotation", 0))})
               for k, v in json.loads(args.cameras).items()}
    robot = make_robot_from_config(SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))
    teleop_proc, robot_proc, obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(teleop_proc, create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(obs_proc, create_initial_features(observation=robot.observation_features), use_videos=True),
    )

    exists = (HF_LEROBOT_HOME / args.repo_id / "meta/info.json").is_file()
    n_cams = len(cameras)
    rgb_encoder = RGBEncoderConfig(vcodec=args.vcodec)
    if exists:
        dataset = LeRobotDataset.resume(args.repo_id, streaming_encoding=True, rgb_encoder=rgb_encoder,
                                        image_writer_processes=0, image_writer_threads=4 * n_cams)
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, features)
    else:
        dataset = LeRobotDataset.create(args.repo_id, FPS, robot_type=robot.name, features=features, use_videos=True, rgb_encoder=rgb_encoder,
                                        image_writer_processes=0, image_writer_threads=4 * n_cams, streaming_encoding=True)

    robot.connect()
    listener, events = init_listener()
    teleop = ReplayTeleop(robot, names, events)
    loop = dict(robot=robot, events=events, fps=FPS, teleop_action_processor=teleop_proc,
                robot_action_processor=robot_proc, robot_observation_processor=obs_proc, teleop=teleop,
                single_task=args.task)
    saved = 0
    try:
        with VideoEncodingManager(dataset):
            i = 0
            while i < len(drafts) and not events["stop_recording"]:
                traj = drafts[i]
                teleop.glide_to(traj[0])
                log_say(f"Replay {i + 1} of {len(drafts)}. Put the bottle back, then say start.")
                record_loop(**loop, control_time_s=APPROACH_S if args.no_wait else 3600)  # hold until "next" or "stop"
                events["exit_early"] = False
                if events["stop_recording"]:
                    break
                log_say("Recording")
                teleop.play(traj)
                record_loop(**loop, dataset=dataset, control_time_s=len(traj) / FPS * 1.5)
                events["exit_early"] = False
                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    dataset.clear_episode_buffer()
                    log_say("Discarded. Put the bottle back, then say start.")
                    continue
                dataset.save_episode()
                saved += 1
                i += 1
    finally:
        log_say(f"Saved {saved} episodes", blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        if args.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
