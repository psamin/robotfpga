"""Demo, then record: teleop one demo, then the follower replays it while both cameras record.

One episode at a time, so the bottle goes back to the same spot right after each demo:

  1. waiting   the follower follows the leader, nothing recorded.   "robot begin"  (Right)
  2. demo      teleop the task; the motion is kept in memory.       "robot done"   (Right) ends it
                                                                     "robot scrap"   (Left)  -> back to 1
  3. ready     the follower glides to the demo's start pose and waits. Put the bottle back.
                                                                     "robot capture" (Up)    -> 4
                                                                     "robot scrap"   (Left)  -> redo the demo (1)
  4. record    the follower replays the demo while the cameras record. "robot scrap" (Left) -> discard, back to 3
  5. review    nothing is saved yet; the follower follows the leader again so you can reset.
                                                                     "robot begin"  (Right) -> keep it, start the next demo (2)
                                                                     "robot scrap"  (Left)  -> throw it away, back to 3
                                                                     "robot finish" (Down)  -> keep it and finish
"robot finish" anywhere else keeps saved episodes, discards the one in progress, and exits.

The replay is open loop: put the bottle exactly where it was at the start of the demo.
"""

import argparse
import json
import logging
import time

import numpy as np

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
from replay_record import ReplayTeleop

FPS = 30
HANDOVER_S = 2.0  # glide from the follower's pose to the leader's before teleop takes over


class LeaderProxy(ReplayTeleop):
    """The leader arm, with a smooth handover from the follower's current pose and demo capture."""

    def __init__(self, leader, robot, names, events):
        super().__init__(robot, names, events)
        self.leader, self.capture = leader, None

    def follow(self):
        pos = self.robot.bus.sync_read("Present_Position")
        self.start = np.array([pos[n.removesuffix(".pos")] for n in self.names], dtype=np.float32)
        self.t0, self.mode = time.perf_counter(), "follow"

    def get_action(self):
        act = self.leader.get_action()
        vec = np.array([act[n] for n in self.names], dtype=np.float32)
        a = min(1.0, (time.perf_counter() - self.t0) / HANDOVER_S)
        vec = self.start + a * (vec - self.start)
        if self.capture is not None:
            self.capture.append(vec)
        return {n: float(v) for n, v in zip(self.names, vec)}


def wait_for(loop, events, accept):
    """Run an unrecorded loop until one of `accept` ("next", "record", "redo") or stop. Returns it."""
    while True:
        record_loop(**loop, control_time_s=3600)
        if events["stop_recording"]:
            return "stop"
        got = "record" if events["record"] else "redo" if events["rerecord_episode"] else "next"
        for k in events:
            events[k] = False
        if got in accept:
            return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--num-episodes", type=int, required=True)
    p.add_argument("--episode-s", type=float, default=20, help="max demo length")
    p.add_argument("--follower-port", required=True)
    p.add_argument("--leader-port", required=True)
    p.add_argument("--follower-id", default="my_follower")
    p.add_argument("--leader-id", default="my_leader")
    p.add_argument("--cameras", required=True, help="JSON {name: {index_or_path, width, height, fps, rotation, warmup_s}}")
    p.add_argument("--vcodec", default="h264_videotoolbox")
    p.add_argument("--push-to-hub", action="store_true")
    args = p.parse_args()
    init_logging()

    cameras = {k: OpenCVCameraConfig(**{**v, "rotation": Cv2Rotation(v.get("rotation", 0))})
               for k, v in json.loads(args.cameras).items()}
    robot = make_robot_from_config(SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))
    leader = make_teleoperator_from_config(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    teleop_proc, robot_proc, obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(teleop_proc, create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(obs_proc, create_initial_features(observation=robot.observation_features), use_videos=True),
    )
    writer = dict(streaming_encoding=True, rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
                  image_writer_processes=0, image_writer_threads=4 * len(cameras))
    if (HF_LEROBOT_HOME / args.repo_id / "meta/info.json").is_file():
        dataset = LeRobotDataset.resume(args.repo_id, **writer)
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, features)
    else:
        dataset = LeRobotDataset.create(args.repo_id, FPS, robot_type=robot.name, features=features,
                                        use_videos=True, **writer)

    leader.connect()  # leader before follower, as lerobot-record does
    robot.connect()
    listener, events = init_listener()
    names = list(robot.action_features)
    proxy = LeaderProxy(leader, robot, names, events)
    replay = ReplayTeleop(robot, names, events)
    base = dict(robot=robot, events=events, fps=FPS, teleop_action_processor=teleop_proc,
                robot_action_processor=robot_proc, robot_observation_processor=obs_proc, single_task=args.task)
    def clear():
        for k in events:
            events[k] = False

    def capture(traj):
        """Steps 3-4. Returns "recorded" (episode in the buffer, not saved yet), "redo" (redo the demo) or "stop"."""
        while True:
            replay.glide_to(traj[0])
            log_say(f"Put the bottle back, then say {PHRASES['capture']}.")
            got = wait_for({**base, "teleop": replay}, events, {"record", "redo"})
            if got != "record":
                return got
            log_say("Recording")
            replay.play(traj)
            record_loop(**base, teleop=replay, dataset=dataset, control_time_s=len(traj) / FPS * 1.5)
            stop, redo = events["stop_recording"], events["rerecord_episode"]
            clear()
            if stop or redo:
                dataset.clear_episode_buffer()
                if stop:
                    return "stop"
                log_say("Recording discarded.")
                continue
            return "recorded"

    saved, start_now, stopping = 0, False, False
    try:
        with VideoEncodingManager(dataset):
            while saved < args.num_episodes and not stopping:
                # 1. waiting (skipped when "robot begin" was just said at the review step)
                proxy.follow()
                if not start_now:
                    log_say(f"Demo {saved + 1} of {args.num_episodes}. Say {PHRASES['begin']} when ready.")
                    if wait_for({**base, "teleop": proxy}, events, {"next"}) == "stop":
                        break
                start_now = False
                # 2. demo
                proxy.capture = []
                log_say("Go")
                record_loop(**base, teleop=proxy, control_time_s=args.episode_s)
                traj, proxy.capture = np.array(proxy.capture), None
                stop, redo = events["stop_recording"], events["rerecord_episode"]
                clear()
                if stop:
                    break
                if redo or len(traj) < FPS:
                    log_say("Demo discarded." if redo else "Demo too short, discarded.")
                    continue
                # 3-4. capture, then 5. review: nothing is saved until you keep it
                while True:
                    got = capture(traj)
                    if got == "stop":
                        stopping = True
                        break
                    if got == "redo":
                        log_say("Demo discarded.")
                        break
                    proxy.follow()
                    last = saved + 1 == args.num_episodes
                    log_say(f"Recorded. Say {PHRASES['begin']} to keep it" + (" and finish" if last else " and start the next demo")
                            + f", or {PHRASES['scrap']} to record it again.")
                    review = wait_for({**base, "teleop": proxy}, events, {"next", "redo"})
                    if review == "redo":
                        dataset.clear_episode_buffer()
                        log_say("Scrapped.")
                        continue
                    dataset.save_episode()  # "robot begin" keeps it; so does "robot finish"
                    saved += 1
                    logging.info(f"Saved episode {saved} of {args.num_episodes}")
                    stopping = review == "stop"
                    start_now = review == "next"
                    break
    finally:
        log_say(f"Saved {saved} episodes", blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if listener is not None:
            listener.stop()
        if args.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
