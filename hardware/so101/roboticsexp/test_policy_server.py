"""Check a running policy server end to end without moving the robot.

Sends the server the same handshake and observation format robot_client does, using a real
recorded frame (episode 0, frame 0: both camera images and joint state), and compares the
predicted first action with the action recorded at that moment. A camera-name mismatch or a
missing feature shows up as an error or as a prediction far from the recording.

  ~/Documents/GitHub/lerobot/.venv/bin/python ~/so101/test_policy_server.py --model <path on the cluster>
(run_policy.sh's tunnel, or any tunnel to the server, must be up on localhost:8080)
"""

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

import lerobot.async_inference.robot_client  # noqa: F401  registers robot types
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation, map_robot_keys_to_lerobot_features
from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import HF_LEROBOT_HOME

import grpc

# Same names and shapes run_policy.sh uses: front -> camera1, wrist -> camera2.
CAMERAS = {"camera1": ("front", dict(index_or_path=0, width=640, height=480)),
           "camera2": ("wrist", dict(index_or_path=1, width=480, height=640, rotation=Cv2Rotation.ROTATE_90))}


def first_frame(root: Path):
    t = pq.read_table(sorted(root.glob("data/*/*.parquet")), columns=["episode_index", "frame_index", "index",
                                                                       "observation.state", "action"])
    i = int(np.where((t["episode_index"].to_numpy() == 0) & (t["frame_index"].to_numpy() == 0))[0][0])
    index = int(t["index"].to_numpy()[i])
    state = np.array(t["observation.state"].to_pylist()[i], dtype=np.float32)
    action = np.array(t["action"].to_pylist()[i], dtype=np.float32)
    images = {}
    for cam, (src, _) in CAMERAS.items():
        cap = cv2.VideoCapture(sorted(glob.glob(str(root / f"videos/observation.images.{src}/*/*.mp4")))[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        assert ok, f"could not read {src} frame {index}"
        images[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # robot observations are RGB, HWC uint8
    return state, action, images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pretrained_model path as seen by the server")
    p.add_argument("--dataset", default="flyingturtleboop/so101_medicine_bottle_pickplace")
    p.add_argument("--task", default="Pick up the medicine bottle and place it on the plate")
    p.add_argument("--server", default="127.0.0.1:8080")
    args = p.parse_args()

    cams = {k: OpenCVCameraConfig(fps=30, **kw) for k, (_, kw) in CAMERAS.items()}
    robot = make_robot_from_config(SO101FollowerConfig(port="/dev/null", id="my_follower", cameras=cams))  # never connected
    names = list(robot.action_features)
    features = map_robot_keys_to_lerobot_features(robot)
    state, recorded_action, images = first_frame(HF_LEROBOT_HOME / args.dataset)

    stub = services_pb2_grpc.AsyncInferenceStub(grpc.insecure_channel(args.server, grpc_channel_options()))
    stub.Ready(services_pb2.Empty())
    t0 = time.perf_counter()
    cfg = RemotePolicyConfig("smolvla", args.model, features, 50, "cuda")
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(cfg)))
    print(f"policy loaded on server in {time.perf_counter() - t0:.1f} s")

    obs = {n: float(v) for n, v in zip(names, state)} | images | {"task": args.task}
    timed = TimedObservation(timestamp=time.time(), observation=obs, timestep=0, must_go=True)
    t0 = time.perf_counter()
    stub.SendObservations(send_bytes_in_chunks(pickle.dumps(timed), services_pb2.Observation, silent=True))
    while True:
        reply = stub.GetActions(services_pb2.Empty())
        if len(reply.data):
            break
        time.sleep(0.01)
    latency = time.perf_counter() - t0
    actions = pickle.loads(reply.data)  # nosec: our own server, through our own tunnel
    first = actions[0].get_action().cpu().numpy()
    err = np.abs(first - recorded_action)
    print(f"got {len(actions)} actions in {latency * 1000:.0f} ms (round trip incl. inference)")
    print("joint:      " + "  ".join(f"{n.removesuffix('.pos'):>13s}" for n in names))
    print("predicted:  " + "  ".join(f"{v:13.1f}" for v in first))
    print("recorded:   " + "  ".join(f"{v:13.1f}" for v in recorded_action))
    print(json.dumps({"ok": bool(len(actions) == 50 and err.mean() < 10), "n_actions": len(actions),
                      "mean_abs_err_deg": round(float(err.mean()), 2), "max_abs_err_deg": round(float(err.max()), 2)}))


if __name__ == "__main__":
    main()
