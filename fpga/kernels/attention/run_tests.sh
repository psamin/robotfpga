#!/bin/sh
# Builds and runs tb_attn in its four variants on the vectors make_vectors.py wrote (logs in results/):
#   tb_attn         the units' HLS mode code (qz::row_mode, smx::softmax_mode) + the contract GEMM model
#   tb_attn_real    the same with the engine's own core (fpga/hls/int16_linear.h) for every GEMM
#   tb_attn_dfsim   both units' DATAFLOW processes as threads on FIFOs bounded at the pragma depths (deadlock model)
#   tb_attn_models  reference models only (row ops from the contract text, SOFTMAX bits from smx_emu)
# Vitis MinGW g++ on a clean PATH (compile and run). Exit code: number of failed variants.
#   sh fpga/kernels/attention/run_tests.sh
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
CXX="g++ -O2 -std=c++14 -ffp-contract=off -static"
B="$HERE/build"
R="$HERE/results"
V="$HERE/vectors"
CASES="--unit $V/vision $V/prefix $V/expert_self $V/expert_cross $V/edge"
mkdir -p "$B" "$R"
fails=0
run() {  # name, extra flags (include order matters for dfsim)
  name=$1
  shift
  if ! $CXX "$@" -o "$B/$name.exe" "$HERE/tb_attn.cpp" > "$R/$name.build.log" 2>&1; then
    echo "$name: BUILD FAILED (see $R/$name.build.log)"
    fails=$((fails + 1))
    return
  fi
  (cd "$HERE" && "$B/$name.exe" $CASES) > "$R/$name.log" 2>&1
  rc=$?
  echo "$name: rc $rc, $(grep -E "^(ALL PASS|FAILED)" "$R/$name.log" | tail -1)"
  [ $rc -eq 0 ] || fails=$((fails + 1))
}
run tb_attn -I"$INC"
run tb_attn_real -DATTN_REAL_ENGINE -DFSTREAM_ARITH=16 -I"$INC"
run tb_attn_dfsim -pthread -DQZ_DFSIM -DSMX_DFSIM -I"$HERE/dfsim" -I"$INC"
run tb_attn_models -DATTN_MODELS -I"$INC"
echo "failed variants: $fails"
exit $fails
