#!/bin/sh
# Build and run the norm_residual C testbench (Vitis MinGW g++, clean PATH) on synthetic and real -12392 vectors.
#   sh fpga/kernels/norm_residual/run_tests.sh          (vectors under fpga/kernels/norm_residual/build/vec)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/c/kp/lerobot-venv/Scripts/python.exe
export HF_HUB_OFFLINE=1
mkdir -p "$HERE/build/vec"
[ -f "$HERE/build/vec/real_expert_final.bin" ] || "$PY" "$HERE/gen_vectors.py" "$HERE/build/vec" --real
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  cd "$HERE"
  g++ -std=c++14 -O2 -ffp-contract=off -Wno-unknown-pragmas -Wno-unused-label -I/c/Xilinx/Vitis_HLS/2022.2/include \
    -I"$HERE" tb_nr.cpp nr_top.cpp -o build/tb_nr_run.exe
  ./build/tb_nr_run.exe build/vec/*.bin
)
