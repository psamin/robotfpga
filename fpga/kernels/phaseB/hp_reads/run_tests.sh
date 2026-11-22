#!/bin/sh
# Build and run the hp_reads testbenches with the Vitis MinGW g++ (clean PATH).
#   sh run_tests.sh quant [tb args]     tb_hpr_quant (hpr_quant.h vs qz::row_mode)
#   sh run_tests.sh smx   [tb args]     tb_hpr_smx   (hpr_softmax.h vs smx::softmax_mode)
#   DFSIM=1 sh run_tests.sh quant ...   the threaded bounded-FIFO deadlock model of both units under test
#   G1DSP=1 ...                         lanes 4..7 on DSPs (default: fabric multipliers)
#   e.g. sh run_tests.sh quant --plan build/chunk.bin     (build/chunk.bin: python ../../contract_plan.py --out ...)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/build"
which=$1
shift
flags="-O2 -std=c++14 -ffp-contract=off -static -Wno-unused-function"
[ -n "$G1DSP" ] && flags="$flags -DHPR_G1_DSP=1"
vitis="-I/c/Xilinx/Vitis_HLS/2022.2/include"
case "$which" in
  quant)
    if [ -n "$DFSIM" ]; then
      exe="$here/build/tb_hpr_quant_dfsim.exe"
      g++ $flags -DQZ_DFSIM -DHQ_DFSIM -I"$here/../../quantize/dfsim" $vitis -pthread -o "$exe" "$here/tb_hpr_quant.cpp" || exit 2
    else
      exe="$here/build/tb_hpr_quant.exe"
      g++ $flags $vitis -o "$exe" "$here/tb_hpr_quant.cpp" || exit 2
    fi ;;
  smx)
    if [ -n "$DFSIM" ]; then
      exe="$here/build/tb_hpr_smx_dfsim.exe"
      g++ $flags -DSMX_DFSIM -DHS_DFSIM -I"$here/../../softmax/tb_dfsim" $vitis -pthread -o "$exe" "$here/tb_hpr_smx.cpp" || exit 2
    else
      exe="$here/build/tb_hpr_smx.exe"
      g++ $flags $vitis -o "$exe" "$here/tb_hpr_smx.cpp" || exit 2
    fi ;;
  *) echo "usage: sh run_tests.sh quant|smx [args]"; exit 2 ;;
esac
"$exe" "$@"
