#!/bin/sh
# emit_act checks (host). Outputs and the assembled source tree under fpga/kernels/phaseB/emit_act/build/.
#   sh fpga/kernels/phaseB/emit_act/run_tests.sh [random cases, default 300]
# 1. tree/: a repo-shaped copy of what the patch produces: snap/hls/* (the integrator's fpga/hls files of
#    2026-11-21 19:57, SHA256SUMS) with patched/hls/gemm_contract.h, fpga/kernels/{gelu_silu,quantize} as they are,
#    patched/kernels/contract.{h,py}, and emit_act.h at kernels/phaseB/emit_act/. The diffs in INTEGRATION.md are
#    snap -> patched; this script checks that they apply (patch --dry-run against the snapshot).
# 2. tb_gemm.cpp (the integrator's contract GEMM testbench, unchanged) on the patched gemm_contract.h: the P_ACT = 0
#    path stays bit-exact against ctr::ref (default, FSTREAM_CHAOS=30, NBX=1 builds).
# 3. tb_emit_act.cpp: the h unit, then fused == unfused byte for byte (default, FSTREAM_CHAOS=30, NBX=1, a base
#    above 2^36).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
B="$HERE/build"
T="$B/tree"
CASES=${1:-300}
rm -rf "$T"
mkdir -p "$T/hls" "$T/kernels/phaseB/emit_act" "$T/kernels/gelu_silu" "$T/kernels/quantize"
cp "$HERE"/snap/hls/*.h "$HERE"/snap/hls/tb_gemm.cpp "$T/hls/"
cp "$HERE/patched/hls/gemm_contract.h" "$T/hls/"
cp "$HERE/patched/kernels/contract.h" "$HERE/patched/kernels/contract.py" "$T/kernels/"
cp "$ROOT"/fpga/kernels/gelu_silu/*.h "$T/kernels/gelu_silu/"
cp "$ROOT"/fpga/kernels/quantize/*.h "$T/kernels/quantize/"
cp "$HERE/emit_act.h" "$T/kernels/phaseB/emit_act/"
(cd "$HERE/snap" && sha256sum -c --quiet SHA256SUMS) || { echo "snapshot changed"; exit 1; }
# the diffs (repo paths, for git apply at the repo root) turn the snapshot into patched/ exactly
A="$B/applycheck"
rm -rf "$A"
mkdir -p "$A/fpga/hls" "$A/fpga/kernels"
cp "$HERE/snap/hls/gemm_contract.h" "$A/fpga/hls/"
cp "$HERE/snap/kernels/contract.h" "$HERE/snap/kernels/contract.py" "$A/fpga/kernels/"
for d in gemm_contract contract_h contract_py; do
  (cd "$A" && patch --binary -p1 -s < "$HERE/diffs/$d.diff") || { echo "diff $d does not apply to the snapshot"; exit 1; }
done
cmp "$A/fpga/hls/gemm_contract.h" "$HERE/patched/hls/gemm_contract.h"
cmp "$A/fpga/kernels/contract.h" "$HERE/patched/kernels/contract.h"
cmp "$A/fpga/kernels/contract.py" "$HERE/patched/kernels/contract.py"
echo "diffs: snapshot + diffs == patched/ (3 files)"
# informational: do they still apply to the live repo files (the integrator keeps editing fpga/hls)?
(cd "$ROOT" && git apply --check "$HERE"/diffs/*.diff 2>&1 && echo "diffs: apply cleanly to the live repo files") ||
  echo "diffs: NOT applicable to the live repo files any more (rebase needed)"
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  INC="-I$T/hls -I/c/Xilinx/Vitis_HLS/2022.2/include"
  CXX="g++ -std=c++14 -O2 -ffp-contract=off -static -DFSTREAM_ARITH=16 -DGS_NSB=10"
  cd "$B"
  $CXX $INC "$T/hls/tb_gemm.cpp" -o tb_gemm_patched.exe
  $CXX -DFSTREAM_CHAOS=30 $INC "$T/hls/tb_gemm.cpp" -o tb_gemm_patched_chaos.exe
  $CXX -DFSTREAM_NBX=1 $INC "$T/hls/tb_gemm.cpp" -o tb_gemm_patched_nbx1.exe
  ./tb_gemm_patched.exe 200 1 > tb_gemm_patched.log 2>&1 || { tail -5 tb_gemm_patched.log; exit 1; }
  tail -1 tb_gemm_patched.log
  ./tb_gemm_patched_chaos.exe 100 2 > tb_gemm_patched_chaos.log 2>&1 || { tail -5 tb_gemm_patched_chaos.log; exit 1; }
  tail -1 tb_gemm_patched_chaos.log
  ./tb_gemm_patched_nbx1.exe 100 3 > tb_gemm_patched_nbx1.log 2>&1 || { tail -5 tb_gemm_patched_nbx1.log; exit 1; }
  tail -1 tb_gemm_patched_nbx1.log
  $CXX $INC "$HERE/tb_emit_act.cpp" -o tb_emit_act.exe
  $CXX -DFSTREAM_CHAOS=30 $INC "$HERE/tb_emit_act.cpp" -o tb_emit_act_chaos.exe
  $CXX -DFSTREAM_NBX=1 $INC "$HERE/tb_emit_act.cpp" -o tb_emit_act_nbx1.exe
  ./tb_emit_act.exe "$CASES" 1 > tb_emit_act.log 2>&1 || { tail -12 tb_emit_act.log; exit 1; }
  grep -v "INFO" tb_emit_act.log | tail -7
  ./tb_emit_act_chaos.exe 150 2 --no-h > tb_emit_act_chaos.log 2>&1 || { tail -8 tb_emit_act_chaos.log; exit 1; }
  grep "emit_act" tb_emit_act_chaos.log
  ./tb_emit_act_nbx1.exe 150 3 --no-h > tb_emit_act_nbx1.log 2>&1 || { tail -8 tb_emit_act_nbx1.log; exit 1; }
  grep "emit_act" tb_emit_act_nbx1.log
  ./tb_emit_act.exe 150 4 FFC0000000 --no-h --no-real > tb_emit_act_hi.log 2>&1 || { tail -8 tb_emit_act_hi.log; exit 1; }
  grep "emit_act" tb_emit_act_hi.log
)
# 4. merged with phase B emit_amax (its patched files as they are now; skipped if absent): build/merge, then the
#    chain test tb_combo (GEMM+ACT+AMX -> GEMM+XQ == GEMM -> ACT F_QOUT -> GEMM) and emit_amax's own tb_amax --quick
if [ -f "$HERE/../emit_amax/patched/hls/gemm_contract.h" ]; then
  /c/kp/lerobot-venv/Scripts/python.exe "$HERE/merge_emit_amax.py"
  (
    export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
    MG="$B/merge"
    INC="-I$MG/hls -I$MG/kernels/quantize -I/c/Xilinx/Vitis_HLS/2022.2/include"
    CXX="g++ -std=c++14 -O2 -ffp-contract=off -static -DFSTREAM_ARITH=16 -DGS_NSB=10"
    cd "$B"
    $CXX $INC "$HERE/tb_combo.cpp" -o tb_combo.exe
    ./tb_combo.exe 120 11 > tb_combo.log 2>&1 || { grep -v INFO tb_combo.log | tail -8; exit 1; }
    grep -v INFO tb_combo.log | tail -1
    $CXX $INC "$HERE/../emit_amax/tb_amax.cpp" "$HERE/../emit_amax/tb_rowmode.cpp" -o tb_amax_merged.exe
    ./tb_amax_merged.exe --quick > tb_amax_merged.log 2>&1 || { grep -v INFO tb_amax_merged.log | tail -8; exit 1; }
    grep "tb_amax" tb_amax_merged.log | tail -1
  )
fi
echo "ALL PASS"
