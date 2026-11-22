#!/bin/sh
# Host tests of the qk_softmax fusion (Vitis MinGW g++ on a clean PATH; Git's mingw64 must not be on it).
#   sh fpga/kernels/phaseB/qk_softmax/run_tests.sh [random cases]
# Builds against the snapshot of the integrator's files in snap/ (SHA256SUMS), logs to results/.
#   tb_qks          fused command == the contract's GEMM + SOFTMAX pair (both the units' HLS code), whole image
#   tb_qks_chaos    the same with every non-blocking FIFO operation of the engine failing 30 % of the time
#   tb_qks_nbx1     NBX = 1 (32-row tiles)
#   tb_gemm_fork    the integrator's own tb_gemm.cpp built against gemm_qks.h: GEMM mode unchanged vs ctr::ref
#   tb_qks --vec    real -12392 calls (results/vec/*.bin from policy_check.py --dump-vec), if present
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
B="$HERE/build"
mkdir -p "$R" "$B/tmp" "$B/fork"
export TMP="$B/tmp" TEMP="$B/tmp" TMPDIR="$B/tmp"
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
F="-std=c++14 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -Wno-unused-function -Wno-sign-compare -Wno-comment \
   -DFSTREAM_ARITH=16 -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -static -I$HERE/snap/hls -I$INC"
N=${1:-200}
g++ $F -o "$B/tb_qks.exe" "$HERE/tb_qks.cpp"
g++ $F -DFSTREAM_CHAOS=30 -o "$B/tb_qks_chaos.exe" "$HERE/tb_qks.cpp"
g++ $F -DFSTREAM_NBX=1 -o "$B/tb_qks_nbx1.exe" "$HERE/tb_qks.cpp"
# the integrator's GEMM testbench on the fork: a gemm_contract.h that is gemm_qks.h
printf '#pragma once\n#include "%s/gemm_qks.h"\n' "$HERE" > "$B/fork/gemm_contract.h"
g++ $F -I"$B/fork" -o "$B/tb_gemm_fork.exe" "$HERE/snap/hls/tb_gemm.cpp"
"$B/tb_qks.exe" "$N" 1 800000000 | tee "$R/tb_qks.log" | tail -3
"$B/tb_qks.exe" "$((N / 2))" 2 40000000 | tee "$R/tb_qks_base4G.log" | tail -1
"$B/tb_qks.exe" "$((N / 4))" 3 ffc0000000 0 | tee "$R/tb_qks_baseFFC.log" | tail -1
"$B/tb_qks_chaos.exe" "$((N / 2))" 4 800000000 | tee "$R/tb_qks_chaos.log" | tail -1
"$B/tb_qks_nbx1.exe" "$((N / 2))" 5 800000000 | tee "$R/tb_qks_nbx1.log" | tail -1
"$B/tb_gemm_fork.exe" 300 11 800000000 | tee "$R/tb_gemm_fork.log" | tail -1
"$B/tb_gemm_fork.exe" 100 12 40000000 | tee -a "$R/tb_gemm_fork.log" | tail -1
if ls "$R"/vec/*.bin > /dev/null 2>&1; then
  "$B/tb_qks.exe" --vec "$R"/vec/*.bin | tee "$R/tb_qks_vec.log" | tail -1
fi
