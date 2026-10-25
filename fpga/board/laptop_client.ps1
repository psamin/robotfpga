# Run the pick-and-place SmolVLA on the real SO-101 from this Windows laptop, with the policy served by the
# KR260 (docs/run-smolvla-on-kr260.md §4). Start the board side first:
#   python fpga/board/deploy_smolvla.py --bit ... --model ... --batches ... --serve
# Then, in PowerShell from the repo root:   .\fpga\board\laptop_client.ps1
# Arm at rest, bottle + plate placed as in robot/calibration/reference_frames. Ctrl-C stops the client (it
# does not glide the arm home). Values below are this laptop's on 2026-10-21: follower COM3, front C920 =
# OpenCV 0, wrist IMX307 = OpenCV 2 (indexes move when cameras are replugged: check with
# lerobot-find-cameras opencv). FrontRotation 180: the C920 is mounted turned around relative to
# robot/calibration/reference_frames/front.jpg (arm base must be at the top of the frame).
param(
    [string]$Board = "ubuntu@192.168.137.35",
    [string]$Policy = "/home/ubuntu/smolvla/ckpt/smolvla-so101_medicine_bottle_pickplace-9604",
    [string]$Port = "COM3",
    [int]$Front = 0,
    [int]$Wrist = 2,
    [int]$FrontRotation = 180,
    [int]$MaxStep = 10,
    [string]$Py = "C:\kp\lerobot-venv\Scripts\python.exe"
)
$task = "Pick up the medicine bottle and place it on the plate"
$cams = "{ camera1: {type: opencv, index_or_path: $Front, width: 640, height: 480, fps: 30, rotation: $FrontRotation, warmup_s: 3}, camera2: {type: opencv, index_or_path: $Wrist, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}}"

Write-Host "SSH tunnel laptop:8080 -> $Board 127.0.0.1:8080 (close its window to stop)"
$tunnel = Start-Process ssh -ArgumentList "-N", "-L", "8080:127.0.0.1:8080", "-o", "ExitOnForwardFailure=yes", $Board -PassThru
Start-Sleep -Seconds 3
if ($tunnel.HasExited) { Write-Error "tunnel exited: is the policy server running and the SSH key installed (deploy --setup)?"; exit 1 }
try {
    & $Py -m lerobot.async_inference.robot_client `
        --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=$Port --robot.id=my_follower `
        --robot.cameras="$cams" --robot.max_relative_target=$MaxStep --task="$task" `
        --policy_type=smolvla --pretrained_name_or_path=$Policy --policy_device=cpu `
        --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --fps=30
} finally {
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id }
}
