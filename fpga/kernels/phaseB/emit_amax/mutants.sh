#!/bin/sh
# Mutation check of tb_amax: each mutant is a deliberately broken copy of the patch; the testbench must FAIL on it.
#   sh mutants.sh <scratch dir>
set -u
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
S=${1:?scratch dir}
export TMP="$S" TEMP="$S" TMPDIR="$S"
HLSI=/c/Xilinx/Vitis_HLS/2022.2/include
CXX="g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off -w -static"
run() {  # name file sed-expr
  name=$1; file=$2; expr=$3
  d="$S/mut_$name"; rm -rf "$d"; mkdir -p "$d/hls" "$d/kernels/phaseB/emit_amax" "$d/kernels/quantize"
  cp "$HERE/patched/hls/gemm_contract.h" "$d/hls/"; cp "$HERE/patched/kernels/contract.h" "$d/kernels/"
  cp "$HERE/emit_amax.h" "$d/kernels/phaseB/emit_amax/"
  cp "$REPO/fpga/hls/int16_linear.h" "$REPO/fpga/hls/float_linear.h" "$d/hls/"
  cp "$REPO/fpga/kernels/quantize/quant_int16.h" "$d/kernels/quantize/"
  sed -i "$expr" "$d/$file"
  if diff -q "$d/$file" "$( [ "$file" = hls/gemm_contract.h ] && echo $HERE/patched/hls/gemm_contract.h || echo $HERE/emit_amax.h )" >/dev/null; then
    echo "MUTANT $name: sed changed nothing (bad mutant)"; return; fi
  $CXX -I"$d/hls" -I"$d/kernels/quantize" -I$HLSI "$HERE/tb_amax.cpp" "$HERE/build/tb_rowmode.o" -o "$d/tb.exe" 2>"$d/cc.log" || { echo "MUTANT $name: compile error"; return; }
  if "$d/tb.exe" --quick > "$d/run.log" 2>&1; then echo "MUTANT $name: SURVIVED (tb passed)"; else echo "MUTANT $name: killed ($(grep -c FAIL "$d/run.log") FAIL lines; first: $(grep -m1 FAIL "$d/run.log" | cut -c1-110))"; fi
}
run lane3   kernels/phaseB/emit_amax/emit_amax.h 's/abits(lane_bits(w, 3) \& 0x7FFFFFFFu)/abits(lane_bits(w, 2) \& 0x7FFFFFFFu)/'
run noround kernels/phaseB/emit_amax/emit_amax.h 's/  if (rb \& (st | M\[0\])) M += 1;/  if (rb \& st) M += 1;/'
run part0   hls/gemm_contract.h 's/am\[w\]\[l\] = j == 0 ? a : famx::amax2(am\[w\]\[l\], a);/am[w][l] = a;/'
run noreset hls/gemm_contract.h 's/const famx::abits a = gs == 0 ? r : famx::amax2(r, amem\[i\]);/const famx::abits a = famx::amax2(r, amem[i]);/'
run shift0  hls/gemm_contract.h 's/ap_int<5>(it.shift)/ap_int<5>(0)/'
run bank0   hls/gemm_contract.h 's/xq_run(io, it, x, xr, rsb, (t + 1) \& 1, xg, thr, step);/xq_run(io, it, x, xr, rsb, 0, xg, thr, step);/'
run lastw   hls/gemm_contract.h 's/if ((i \& 3) == 3 || i == nv - 1) {/if ((i \& 3) == 3) {/'
run segpitch hls/gemm_contract.h 's/            sb += it.hp;/            sb += 0;/'
run deadrow hls/gemm_contract.h 's/      live = x.n0 + x.a < it.N;/      live = true;/'
