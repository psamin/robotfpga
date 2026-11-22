#!/bin/sh
# Host build + run of the SOFTMAX unit testbench with the Vitis MinGW g++ (Git's mingw64 must not be on PATH).
#   sh fpga/kernels/softmax/build.sh            # vectors (python) + compile + run every case
#   SMX_NO_VECTORS=1 sh .../build.sh            # reuse the existing cases
# Two builds of the same HLS source (softmax_top.cpp + softmax_core.h):
#   tb_softmax.exe        DATAFLOW processes one after the other on unbounded FIFOs (the HLS csim semantics)
#   tb_softmax_dfsim.exe  the processes as threads on FIFOs bounded at the pragma depths (deadlock model)
# plus the prologue's integer c = fl32(p0 * log2 e) against the host FPU (--cparams), the divider on every
# possible row sum against the host's 1.0f / S (--divider), and a negative control of the deadlock model: with
# the row FIFO cut to 8 words (below one row of L = 256) it must report the deadlock (exit code 7).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/c/kp/lerobot-venv/Scripts/python.exe}"
B="$HERE/build"
mkdir -p "$B"
if [ -z "$SMX_NO_VECTORS" ]; then
  rm -rf "$B/cases"
  (cd "$HERE" && "$PY" gen_tables.py && "$PY" make_vectors.py --out "$B/cases")
fi
# g++ needs a writable temp directory (TMP/TEMP may point at C:\WINDOWS under a stripped environment)
mkdir -p "$B/tmp"
export TMP="$B/tmp" TEMP="$B/tmp" TMPDIR="$B/tmp"
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
CXX="g++ -std=c++14 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -Wno-unused-function -Wno-sign-compare -Wno-comment -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -static"
$CXX -I"$INC" -o "$B/tb_softmax.exe" "$HERE/tb_softmax.cpp"
$CXX -DSMX_DFSIM -I"$HERE/tb_dfsim" -I"$INC" -o "$B/tb_softmax_dfsim.exe" "$HERE/tb_softmax.cpp" -pthread
"$B/tb_softmax.exe" --cparams
"$B/tb_softmax.exe" --divider
"$B/tb_softmax.exe" "$B"/cases/*
"$B/tb_softmax_dfsim.exe" "$B"/cases/*
$CXX -DSMX_DFSIM -DSMX_XS_DEPTH=8 -DSMX_DFSIM_TIMEOUT=5 -I"$HERE/tb_dfsim" -I"$INC" -o "$B/tb_softmax_neg_xs.exe" "$HERE/tb_softmax.cpp" -pthread
set +e
"$B/tb_softmax_neg_xs.exe" "$B/cases/real_vision"
rc=$?
set -e
if [ $rc -eq 7 ]; then echo "negative control (row FIFO 8 words, L = 256): deadlock detected as required: PASS"; else echo "negative control: expected deadlock exit 7, got $rc: FAIL"; exit 1; fi
