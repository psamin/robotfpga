#!/bin/sh
# Build and run tb_quant with the Vitis MinGW g++ (clean PATH, compile and run).
#   sh run_tests.sh [--dfsim] [tb args...]
#     --dfsim   the threaded bounded-FIFO model of row_df (every process a thread, every stream at its pragma
#               depth; a blocked stream for 20 s prints DEADLOCK and exits 7)
#   e.g. sh run_tests.sh --quick --plan build/chunk_hi.bin
#        sh run_tests.sh --exhaustive 8 --plan build/chunk_hi.bin --plan-all results/captured.qz
#   (build/chunk_hi.bin: python ../contract_plan.py --out build/chunk_hi.bin; results/captured.qz: check_policy.py --dump)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/build"
flags="-O2 -std=c++14 -ffp-contract=off -static"
vitis="-I/c/Xilinx/Vitis_HLS/2022.2/include"  # after dfsim/: its bounded hls_stream.h must shadow the Vitis one
if [ "$1" = "--dfsim" ]; then
  shift
  exe="$here/build/tb_quant_dfsim.exe"
  g++ $flags -DQZ_DFSIM -I"$here/dfsim" $vitis -pthread -o "$exe" "$here/tb_quant.cpp" || exit 2
else
  exe="$here/build/tb_quant.exe"
  g++ $flags $vitis -o "$exe" "$here/tb_quant.cpp" "$here/quant_top.cpp" || exit 2
fi
"$exe" "$@"
