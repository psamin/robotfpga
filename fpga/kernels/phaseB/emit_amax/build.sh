#!/bin/sh
# Build the emit_amax host testbenches with the Vitis MinGW g++ on a clean PATH (CONTRACT.md 7).
#   sh build.sh            -> build/tb_amax.exe, build/tb_amax_chaos.exe, build/tb_amax_nbx1.exe, build/tb_amax_ql2.exe
set -e
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
HLSI=/c/Xilinx/Vitis_HLS/2022.2/include
mkdir -p "$HERE/build"
CXX="g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off -w -Wno-unknown-pragmas -Wno-unused-label -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -static"
INC="-I$HERE/patched/hls -I$REPO/fpga/hls -I$REPO/fpga/kernels/quantize -I$HLSI"
cd "$HERE"
# the row unit's QUANT_X against the unpatched contract (its own TU)
$CXX -I$REPO/fpga/kernels/quantize -I$HLSI -c tb_rowmode.cpp -o build/tb_rowmode.o
$CXX $INC tb_amax.cpp build/tb_rowmode.o -o build/tb_amax.exe
$CXX $INC -DFSTREAM_CHAOS=30 tb_amax.cpp build/tb_rowmode.o -o build/tb_amax_chaos.exe
$CXX $INC -DFSTREAM_NBX=1 tb_amax.cpp build/tb_rowmode.o -o build/tb_amax_nbx1.exe
$CXX $INC -DAMX_QL=2 tb_amax.cpp build/tb_rowmode.o -o build/tb_amax_ql2.exe
echo built
