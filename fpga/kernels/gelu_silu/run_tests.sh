#!/bin/sh
# Build and run the gelu_silu checks for NSB 9 and 10, on synthetic vectors and on real -12392 activations
# (vision fc1 -> GELU, LM and expert gate|up -> SiLU * up, action-time MLP -> SiLU). Outputs under build/gelu_silu/.
#   sh fpga/kernels/gelu_silu/run_tests.sh
# 1. make_vectors.py / policy_check.py --dump: inputs + gs_emu's bit-exact q and s
# 2. cpp_mirror.py: Python transliteration of the C++ arithmetic vs gs_emu (runs anywhere)
# 3. tb_gelu_silu.cpp linked with the HLS source (Vitis MinGW g++, clean PATH): bit equality with gs_emu.
#    On a machine where Smart App Control blocks newly built .exe files, step 3 builds but cannot run.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
B="$ROOT/build/gelu_silu"
PY=/c/kp/lerobot-venv/Scripts/python.exe
mkdir -p "$B"
export HF_HUB_OFFLINE=1
for nsb in 9 10; do
  [ -d "$B/vec$nsb" ] || "$PY" "$HERE/make_vectors.py" "$B/vec$nsb" --nsb $nsb
  [ -d "$B/real$nsb" ] || "$PY" "$HERE/policy_check.py" --dump "$B/real$nsb" --nsb $nsb --frames 0
done
"$PY" "$HERE/cpp_mirror.py"
"$PY" "$HERE/cpp_mirror.py" "$B"/vec9/* "$B"/real9/* "$B"/vec10/* "$B"/real10/*
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  cd "$B"
  for lanes in 8 4; do
    for nsb in 9 10; do
      g++ -std=c++14 -O2 -DGS_NSB=$nsb -DGS_LANES=$lanes -I/c/Xilinx/Vitis_HLS/2022.2/include -I"$HERE" \
        "$HERE/tb_gelu_silu.cpp" "$HERE/gelu_silu_top.cpp" -o tb_l${lanes}_nsb$nsb.exe
      ./tb_l${lanes}_nsb$nsb.exe vec$nsb/* real$nsb/*
    done
  done
)
