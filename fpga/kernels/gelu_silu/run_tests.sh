#!/bin/sh
# Build and run the gelu_silu checks (contract 1.0). Outputs under build/gelu_silu/c1/.
#   sh fpga/kernels/gelu_silu/run_tests.sh          (FRESH=1: also re-dump the real activations)
# 1. make_vectors.py for NSB 9 and 10 (synthetic: random, edge, small shapes, writer chunks, tails, NaN / Inf) and
#    policy_check.py --dump (real -12392 activations of frame 0: vision GELU, LM and expert SiLU * up, action-time
#    SiLU), each with gs_emu's bit-exact y and the contract quantizer's q and s
# 2. contract_plan.py --out: the -12392 list (2,701 records), for the testbench's path census
# 3. tb_gelu_silu.cpp linked with the HLS source (Vitis MinGW g++, clean PATH, -static), NSB 9 and 10: every case
#    in 5 layouts, the bad commands, the csynth top's pointer build, and (NSB 9) the census of the list
# 4. the threaded bounded-FIFO deadlock model (-DGS_DFSIM): at the pragma depths, at minimum depths (every side
#    FIFO 1, the row buffer one row + 1), and its negative test (a row buffer shorter than a row: DEADLOCK, exit 7)
# 5. copies the vectors to build/gelu_silu/{vec,real}{9,10}, where fpga/kernels/run_kernel_baseline.sh looks
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
B="$ROOT/build/gelu_silu/c1"
PY=/c/kp/lerobot-venv/Scripts/python.exe
mkdir -p "$B"
export HF_HUB_OFFLINE=1
for nsb in 9 10; do
  "$PY" "$HERE/make_vectors.py" "$B/vec$nsb" --nsb $nsb
  if [ -n "$FRESH" ] || [ ! -d "$B/real$nsb" ]; then
    "$PY" "$HERE/policy_check.py" --dump "$B/real$nsb" --nsb $nsb --frames 0
  fi
done
"$PY" "$ROOT/fpga/kernels/contract_plan.py" --mhz 250 --out "$B/plan_12392.bin" > "$B/contract_plan.log"
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  INC="-I/c/Xilinx/Vitis_HLS/2022.2/include -I$HERE"
  SRC="$HERE/tb_gelu_silu.cpp $HERE/gelu_silu_top.cpp"
  CXX="g++ -std=c++14 -O2 -ffp-contract=off -static"
  cd "$B"
  for nsb in 9 10; do
    $CXX -DGS_NSB=$nsb $INC $SRC -o tb_nsb$nsb.exe
    plan=""
    [ $nsb = 9 ] && plan="--plan $B/plan_12392.bin"
    ./tb_nsb$nsb.exe $plan vec$nsb/* real$nsb/* > tb_nsb$nsb.log 2>&1 || { tail -20 tb_nsb$nsb.log; exit 1; }
    tail -4 tb_nsb$nsb.log
  done
  $CXX -pthread -DGS_NSB=9 -DGS_DFSIM -I"$HERE/dfsim" $INC $SRC -o tb_dfsim.exe
  ./tb_dfsim.exe vec9/* real9/* > tb_dfsim.log 2>&1 || { tail -20 tb_dfsim.log; exit 1; }
  tail -2 tb_dfsim.log
  MIN="-DGS_D_XIN=1 -DGS_D_YF=1 -DGS_D_AM=1 -DGS_D_RSF=1 -DGS_D_SST=1 -DGS_D_OW=1 -DGS_D_YS=769"
  $CXX -pthread -DGS_NSB=9 -DGS_DFSIM $MIN -I"$HERE/dfsim" $INC $SRC -o tb_dfsim_min.exe
  ./tb_dfsim_min.exe vec9/* real9/* > tb_dfsim_min.log 2>&1 || { tail -20 tb_dfsim_min.log; exit 1; }
  tail -2 tb_dfsim_min.log
  $CXX -pthread -DGS_NSB=9 -DGS_DFSIM -DGS_DFSIM_NEGATIVE -DGS_DFSIM_TIMEOUT=3 -DGS_D_YS=700 -I"$HERE/dfsim" $INC $SRC \
    -o tb_dfsim_neg.exe
  rc=0
  ./tb_dfsim_neg.exe vec9/gelu_rand > tb_dfsim_neg.log 2>&1 || rc=$?
  if [ $rc -ne 7 ]; then
    echo "negative deadlock test: expected exit 7, got $rc"
    exit 1
  fi
  echo "negative deadlock test (row buffer 700 < 768 words): DEADLOCK reported, exit 7 (expected)"
)
# the vector directories fpga/kernels/run_kernel_baseline.sh reads (build/gelu_silu/vec9, real9): the same cases
for d in vec9 real9 vec10 real10; do
  rm -rf "$ROOT/build/gelu_silu/$d"
  cp -r "$B/$d" "$ROOT/build/gelu_silu/$d"
done
echo "ALL PASS"
