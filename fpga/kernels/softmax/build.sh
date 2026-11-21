#!/bin/sh
# Host build + run of the softmax testbench with the Vitis MinGW g++ (Git's mingw64 must not be on PATH).
#   sh fpga/kernels/softmax/build.sh            # vectors (python) + compile + run every case
#   SMX_NO_VECTORS=1 sh .../build.sh            # reuse the existing cases
# Three builds: FP32 output + engine layout (every case), the default (no FP32 output; FP32 cases must be refused
# with status 1 and leave memory untouched), and rows-only (SMX_ENGINE=0, the rows_ cases).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/c/kp/lerobot-venv/Scripts/python.exe}"
B="$HERE/build"
mkdir -p "$B"
if [ -z "$SMX_NO_VECTORS" ]; then
  rm -rf "$B/cases"
  (cd "$HERE" && "$PY" gen_tables.py && "$PY" make_vectors.py --out "$B/cases")
fi
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
CXX="g++ -std=c++14 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -Wno-unused-function -D__USE_MINGW_ANSI_STDIO=1 -I$INC"
$CXX -DSMX_FP32_OUT=1 -DSMX_ENGINE=1 -o "$B/tb_softmax_fp1.exe" "$HERE/tb_softmax.cpp"
$CXX -DSMX_FP32_OUT=0 -DSMX_ENGINE=1 -o "$B/tb_softmax_fp0.exe" "$HERE/tb_softmax.cpp"
$CXX -DSMX_FP32_OUT=0 -DSMX_ENGINE=0 -o "$B/tb_softmax_rows.exe" "$HERE/tb_softmax.cpp"
"$B/tb_softmax_fp1.exe" "$B"/cases/*
"$B/tb_softmax_fp0.exe" "$B"/cases/*
"$B/tb_softmax_rows.exe" "$B"/cases/rows_*
