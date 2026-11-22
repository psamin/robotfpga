#!/bin/sh
# Build and run the NORM mode's host testbench (Vitis MinGW g++, clean PATH, static) on synthetic and real -12392
# vectors, in both builds: sequential C simulation and the threaded bounded-FIFO DATAFLOW model (-DNR_DFSIM).
#   sh fpga/kernels/norm_residual/run_tests.sh [--exhaustive]      (vectors under build/vec; --real needs the policy)
# HF_HUB_CACHE may point at a working hub cache (the shared one's SmolVLM2 refs/main can name an incomplete
# snapshot; see README "Running").
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/c/kp/lerobot-venv/Scripts/python.exe
UNITS=--units
[ "$1" = "--exhaustive" ] && UNITS=--units-exhaustive
export HF_HUB_OFFLINE=1
mkdir -p "$HERE/build/vec"
[ -f "$HERE/build/vec/real_expert_final.bin" ] || "$PY" "$HERE/gen_vectors.py" "$HERE/build/vec" --real
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  cd "$HERE"
  F="-std=c++14 -O2 -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -Wno-unknown-pragmas -Wno-unused-label"
  F="$F -I/c/Xilinx/Vitis_HLS/2022.2/include -I. -I.. -I../quantize"
  g++ $F tb_nr.cpp nr_top.cpp -static -o build/tb_nr_run.exe
  g++ -Idfsim $F -DNR_DFSIM -DQZ_DFSIM -pthread tb_nr.cpp nr_top.cpp -static -o build/tb_nr_df.exe
  ./build/tb_nr_run.exe $UNITS --passthrough 200 build/vec/*.bin
  ./build/tb_nr_df.exe --passthrough 40 build/vec/*.bin
)
