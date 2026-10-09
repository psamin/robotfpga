"""Teleop recording where every episode waits for you to say "start".

Used by collect.sh for live recording (cameras on) and for replay step 1 (draft, no cameras).
Built on LeRobot's own record_loop and dataset writer, so the data matches lerobot-record.

Per episode:
  waiting    the follower follows the leader but nothing is recorded; get into position
  "start"    (n or ->) recording begins
  "robot done" (n or ->) ends the episode early and saves it; otherwise it ends after --episode-s
  "redo"     (r or <-) during recording: discard it and go back to waiting for "start"
  "stop"     (q or Esc) at any time: keep finished episodes, discard one in progress, exit
"""

import argparse
import json
import logging
from pathlib import Path

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
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import init_logging, log_say

from keys import PHRASES, init_listener

FPS = 30


def clear(events):
    for k in events:
        events[k] = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, help="dataset folder (default: the LeRobot cache)")
    p.add_argument("--task", required=True)
    p.add_argument("--num-episodes", type=int, required=True)
    p.add_argument("--episode-s", type=float, default=20)
    p.add_argument("--follower-port", required=True)
    p.add_argument("--leader-port", required=True)
    p.add_argument("--follower-id", default="my_follower")
    p.add_argument("--leader-id", default="my_leader")
    p.add_argument("--cameras", default="{}", help="JSON {name: {index_or_path, width, height, fps, rotation, warmup_s}}")
    p.add_argument("--vcodec", default="h264_videotoolbox")
    p.add_argument("--push-to-hub", action="store_true")
    args = p.parse_args()
    init_logging()

    cameras = {k: OpenCVCameraConfig(**{**v, "rotation": Cv2Rotation(v.get("rotation", 0))})
               for k, v in json.loads(args.cameras).items()}
    robot = make_robot_from_config(SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    teleop_proc, robot_proc, obs_proc = make_default_processors()
    use_videos = bool(cameras)
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(teleop_proc, create_initial_features(action=robot.action_features), use_videos=use_videos),
        aggregate_pipeline_dataset_features(obs_proc, create_initial_features(observation=robot.observation_features), use_videos=use_videos),
    )
    root = args.root or HF_LEROBOT_HOME / args.repo_id
    writer = dict(streaming_encoding=use_videos, rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
                  image_writer_processes=0, image_writer_threads=4 * len(cameras))
    if (root / "meta/info.json").is_file():
        dataset = LeRobotDataset.resume(args.repo_id, root=args.root, **writer)
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, features)
    else:
        dataset = LeRobotDataset.create(args.repo_id, FPS, root=args.root, robot_type=robot.name,
                                        features=features, use_videos=use_videos, **writer)

    teleop.connect()  # leader before follower, as lerobot-record does
    robot.connect()
    listener, events = init_listener()
    loop = dict(robot=robot, events=events, fps=FPS, teleop_action_processor=teleop_proc,
                robot_action_processor=robot_proc, robot_observation_processor=obs_proc,
                teleop=teleop, single_task=args.task)
    saved = 0
    try:
        with VideoEncodingManager(dataset):
            while saved < args.num_episodes and not events["stop_recording"]:
                # Waiting: teleop live, nothing recorded, until "start". "redo" here is ignored.
                log_say(f"Episode {saved + 1} of {args.num_episodes}. Say {PHRASES['begin']} when ready.")
                while True:
                    record_loop(**loop, control_time_s=3600)
                    if events["stop_recording"] or not events["rerecord_episode"]:
                        break
                    clear(events)
                if events["stop_recording"]:
                    break
                clear(events)
                log_say("Recording")
                record_loop(**loop, dataset=dataset, control_time_s=args.episode_s)
                if events["rerecord_episode"] and not events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    clear(events)
                    log_say("Discarded")
                    continue
                if events["stop_recording"]:  # stopped mid-demo: don't keep a half-finished episode
                    dataset.clear_episode_buffer()
                    break
                clear(events)
                dataset.save_episode()
                saved += 1
                logging.info(f"Saved episode {saved} of {args.num_episodes}")
    finally:
        log_say(f"Saved {saved} episodes", blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        if args.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
