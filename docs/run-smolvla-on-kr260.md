# Run the pick-and-place SmolVLA on the KR260

The best policy for "Pick up the medicine bottle and place it on the plate" (`-9604`, GitHub release
[policy-smolvla-9604](https://github.com/psamin/robotfpga/releases/tag/policy-smolvla-9604)) running on
the KR260: every linear layer on the FPGA (BF16 weights streamed from DDR, FP32 math), everything else in
FP32 on the board's ARM cores. The laptop drives the arm and cameras exactly as with the cluster.

Status: the whole software stack below runs on x86 with the host C core in place of the FPGA (the same core HLS
synthesizes, called with the board's DDR word offsets): `cluster/kr260_mock_test.sbatch`, job 10279. The selftest
passed (max difference from PyTorch FP32 0.0043, normalized units, on 3 real frames). Through the real gRPC policy
server, `robot/toolkit/test_policy_server.py` got 50 actions for a recorded frame, the first within 1.08° mean (1.77° max)
of the recorded action. **Nothing here has run on the board yet**: it needs the bitstream (Q12) and the board.

## The 256-px checkpoint (`-12392`, #215)

This is a re-finetune of the same recipe with 256×256 camera images instead of 512 (only the action expert is trained). It does
about 3× less work per chunk: the linear layers drop from 255 to 107 GMAC and the vision attention on the ARM from 38.7 to 2.4 GMAC. The
layer shapes are the same, so the same bitstream runs it. Release
[policy-smolvla-12392-256px](https://github.com/psamin/robotfpga/releases/tag/policy-smolvla-12392-256px) has three files:

- `smolvla-so101-pickplace-12392-256px.tar`: the checkpoint, with camera shapes fixed.
- `smolvla-12392-replay-batches.tar.gz`: the same 3 frames, with references from this checkpoint. It extracts to `batches-12392/`.
- `SHA256SUMS`.

On x86 with the host C core (job 12405), the selftest passes with max difference 0.0084. Through the gRPC test client, the first
action is within 0.19° mean (0.40° max) of the recording.

- **Use its own batches.** `-9604`'s references fail against this model. Its frame files have the same names and sizes as
  `-9604`'s, and `deploy_smolvla.py` skips any file the board already has at the same size. Delete the board's `batches/`
  folders (the Jupyter upload folder and `/home/<user>/smolvla/`) before deploying it, and again when switching back.
- **Camera shapes.** Its `config.json` holds the recorded camera shapes (front 3×480×640, wrist 3×640×480). LeRobot's
  client resizes live frames to these shapes; with the old 3×256×256 it squashed them to squares
  (`robot/toolkit/native_camera_shapes.py`).

## What runs where

| Machine | Process | Python | Does |
|---|---|---|---|
| KR260 | `fpga/board/engine_server.py --bit float.bit` | PYNQ's (3.10), as root | loads the overlay, keeps the packed weights in DDR, runs each GEMM on the FPGA |
| KR260 | `fpga/board/serve_policy_fpga.py` | a LeRobot venv (3.12) | LeRobot's policy server; after loading, swaps all 303 `nn.Linear` for FPGA calls over the Unix socket `/tmp/fpga-engine.sock` |
| Laptop | `robot_client` (or `policy.sh BOARD=...` on the Mac) | LeRobot venv | arm + cameras, SSH tunnel to the board's port 8080 |

There are two processes on the board because LeRobot (`psamin/lerobot@e624f3f7`) requires Python ≥ 3.12, while
Kria-PYNQ 3.0.1 runs Python 3.10.12. The policy server listens on the board's localhost only, as on the cluster
(its gRPC transport unpickles requests), so the laptop always connects through SSH.

## 1. FPGA side (once)

1. Build the bitstream in `fpga/hls/`: `vitis_hls -f run_float_hls.tcl`, then
   `vivado -mode batch -source build_bd.tcl` → `build/float.bit` + `float.hwh` (SETUP.md §6, Q12).
2. On the board, check the engine alone with PYNQ's Python:
   `sudo -E <pynq python> fpga/board/check_engine.py --bit float.bit build/stream/float` (`make -C fpga/hls test`
   writes the cases): every case PASS, plus the weight bandwidth (the design assumes 16.2 GB/s).

## 2. Board setup (once)

```sh
git clone https://github.com/psamin/robotfpga && cd robotfpga
# the policy: checkpoint + replay batches + checksums (~0.9 GB)
mkdir -p ~/models && cd ~/models
for f in smolvla-so101-pickplace-9604.tar smolvla-9604-replay-batches.tar.gz SHA256SUMS; do
  curl -LO https://github.com/psamin/robotfpga/releases/download/policy-smolvla-9604/$f; done
sha256sum -c --ignore-missing SHA256SUMS && tar -xf smolvla-so101-pickplace-9604.tar && tar -xzf smolvla-9604-replay-batches.tar.gz
# LeRobot in its own Python 3.12 (PYNQ keeps its 3.10)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.12 ~/lerobot-venv
uv pip install -p ~/lerobot-venv "lerobot[smolvla,async] @ git+https://github.com/psamin/lerobot@e624f3f7f8411ec3a02635d06e79373341e5ef35"
```

- **Memory.** The board has 4 GB. The engine's DDR buffers hold 0.71 GB of packed BF16 weights, allocated by PYNQ
  from CMA: check `grep Cma /proc/meminfo`. The CMA size the board needs, and how to raise it on this image, are
  unknown until tried. The policy process loads the checkpoint (0.9 GB BF16), frees each linear weight after packing,
  then keeps 48.9M parameters in FP32 (0.2 GB).
- **PYNQ's Python path** is assumed to be `/usr/local/share/pynq-venv/bin/python3` (Kria-PYNQ default), not verified
  on this board: check with `sudo bash -c 'source /etc/profile.d/pynq_venv.sh && which python3'`.

## 3. Check, then serve (each session)

```sh
# terminal 1: the engine (root for PYNQ; the socket is handed to your user)
sudo -E /usr/local/share/pynq-venv/bin/python3 fpga/board/engine_server.py --bit build/float.bit
# terminal 2: check against the PyTorch FP32 reference, without the arm
~/lerobot-venv/bin/python fpga/board/serve_policy_fpga.py \
  --selftest ~/models/smolvla-so101_medicine_bottle_pickplace-9604 --batches ~/models/batches
#   prints per frame: max |chunk - reference| (normalized), seconds per chunk, time in FPGA calls; PASS < 0.05
# terminal 2: serve (leave running)
~/lerobot-venv/bin/python fpga/board/serve_policy_fpga.py --port 8080
```

## 4. Laptop: run the arm

Arm at rest, cameras and calibration as in [robot/README.md](../robot/README.md) §Running the best policy:
camera index of the front C920 and the wrist camera, framing against `robot/calibration/reference_frames/`, and the
camera settings of the recording session (front focus 40 / white balance 3066 K, wrist 4000 K). The follower's
calibration must be installed as `my_follower` (`robot/calibration/so_follower_my_follower.json`).

**Mac** (the toolkit): `BOARD=ubuntu@<board address> ./policy.sh start` from `robot/toolkit/`, then
`./policy.sh stop` (the arm glides back to its start pose).

**Windows** (PowerShell; the toolkit's shell scripts are macOS-only). Use your COM port and OpenCV indexes
(`lerobot-find-port`, `lerobot-find-cameras opencv`):

```powershell
ssh -N -L 8080:127.0.0.1:8080 ubuntu@<board address>     # window 1: tunnel, leave open
python -m lerobot.async_inference.robot_client `
  --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower `
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}, camera2: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}}" `
  --robot.max_relative_target=10 --task="Pick up the medicine bottle and place it on the plate" `
  --policy_type=smolvla --pretrained_name_or_path=/home/ubuntu/models/smolvla-so101_medicine_bottle_pickplace-9604 `
  --policy_device=cpu --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30
```
The camera names must be `camera1` (front) and `camera2` (wrist): the policy server replaces the rename map saved at
training time with the client's (empty) one. `max_relative_target=10` caps each joint's move per step (degrees).
Ctrl-C stops it; unlike `policy.sh` it does not return the arm to its start pose.

## Expected speed, and what the FPGA side can change

| Work per 50-action chunk | Where | Size | Estimate |
|---|---|---|---|
| All 302 executed linear layers | FPGA | 255 GMAC, 0.71 GB of weights | ~3.3 s at TP = 8, compute-bound (`fpga/hls/latency.py`; unmeasured) |
| Vision-encoder attention (Q·Kᵀ and A·V) | ARM | 2 cameras × 12 layers × 2 × 1024² × 768 = 38.7 GMAC (12 layers, 768 wide, 1,024 tokens per 512² image: from the checkpoint) | 77 GFLOP on four Cortex-A53 cores (peak FP32 a few tens of GFLOP/s), so likely several seconds; unmeasured |
| Patch-embedding convolution, norms, softmax, language model and action-expert attention | ARM | small next to the above | unmeasured |

`--selftest` measures both parts on the board: `seconds_per_chunk` is the total, `fpga_and_transfer_s` the share in
FPGA calls (including the socket copies). The arm executes a chunk in 1.67 s (50 actions at 30 fps) and asks for the
next one halfway through, so any chunk time above ~1 s shows up as pauses between motions. In order of expected payoff:

1. **Move the vision attention off the ARM.** These matmuls are not `nn.Linear` (both operands are activations), so
   the engine never sees them. Options: a second HLS kernel for attention with Q, K and V on chip (1024 × 64 per head), or
   reusing `linear_bf16` with K packed as the weight matrix per call (packing on the ARM costs time).
2. **Q13: BF16 multiplies with FP32 accumulation** (several times more MACs per DSP than FP32 multiplies).
3. **#167: real load/compute overlap** (DATAFLOW), which the latency model assumes and the HLS code does not yet enforce.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `serve_policy_fpga.py`: `ConnectionRefusedError` / `FileNotFoundError` on the socket | start `engine_server.py` first; same `--socket` |
| `PermissionError` on the socket | the server makes it 0600 and hands it to `$SUDO_USER`; run the policy as that user |
| PYNQ `allocate` fails during upload | CMA too small for the 0.71 GB of weights (see Memory) |
| Selftest FAIL with a large difference | packing or DDR offsets on the board differ from the C model: run `check_engine.py` first |
| Robot client: camera timeout | another program holds the camera (preview, Continuity Camera); OpenCV indexes shifted |
