#!/bin/sh
# Negative controls of tb_qks: each mutant is a copy of the fused sources with one injected bug (sed on a build
# copy; the sources stay untouched). Every mutant must make tb_qks FAIL, else the testbench cannot see that bug.
#   sh fpga/kernels/phaseB/qk_softmax/run_mutants.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$HERE/build/mut"
R="$HERE/results"
mkdir -p "$B" "$R" "$HERE/build/tmp"
export TMP="$HERE/build/tmp" TEMP="$HERE/build/tmp" TMPDIR="$HERE/build/tmp"
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
F="-std=c++14 -O2 -w -DFSTREAM_ARITH=16 -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -static -I$HERE/snap/hls -I$INC"
: > "$R/mutants.log"
bad=0
run() {  # name file sed-expression
  d="$B/$1"
  rm -rf "$d"
  mkdir -p "$d"
  cp "$HERE/tb_qks.cpp" "$HERE/gemm_qks.h" "$HERE/qks_tail.h" "$HERE/contract_qks.h" "$d/"
  sed -i "$3" "$d/$2"
  if [ "$(md5sum < "$d/$2")" = "$(md5sum < "$HERE/$2")" ]; then
    echo "MUTANT $1: the sed did not change $2 (stale mutant)" | tee -a "$R/mutants.log"
    bad=$((bad + 1))
    return
  fi
  g++ $F -o "$d/tb.exe" "$d/tb_qks.cpp" || { echo "MUTANT $1: build failed" | tee -a "$R/mutants.log"; bad=$((bad + 1)); return; }
  "$d/tb.exe" 60 7 800000000 > "$d/out.txt" 2>&1
  rc=$?
  nf=$(grep -c '^FAIL' "$d/out.txt")
  if [ $rc -ne 0 ]; then
    echo "MUTANT $1 ($2): detected, tb exit $rc, $nf FAIL lines; first: $(grep -m1 '^FAIL' "$d/out.txt" | cut -c1-150)" | tee -a "$R/mutants.log"
  else
    echo "MUTANT $1 ($2): NOT DETECTED" | tee -a "$R/mutants.log"
    bad=$((bad + 1))
  fi
}
run vis_lane0   qks_tail.h 's/vis\[l\] = !tl.mask ||/vis[l] = l == 0 || !tl.mask ||/'
run max_nofold  qks_tail.h 's/const Key stored = fg == 0 ? (Key)0 : rmax\[ra\];/const Key stored = (Key)0;/'
run max_wrongrow qks_tail.h 's/smx_s.write(smx::key_bits(rmax\[ra\]));/smx_s.write(smx::key_bits(rmax[ra ^ 1]));/'
run key_perm    qks_tail.h 's/const int wi = fg \* 8 + fw;/const int wi = fg * 8 + (fw ^ (fg \& 1));/'
run drain_half  qks_tail.h 's/const int ra = (dh ? NTT : 0) + drow;/const int ra = (dh \&\& tl.ntiles > 2 ? 0 : dh ? NTT : 0) + drow;/'
run rq_tile0    gemm_qks.h 's/if (it.mask) rq_tile(io, it, t, f_words, rs);/if (it.mask) rq_tile(io, it, 0, f_words, rs);/'
run stok_slotF  gemm_qks.h 's/it.fa = it.qks ? G.addr + b \* G.stride : F.addr + b \* F.stride;/it.fa = F.addr + b * F.stride;/'
run np_ignored  gemm_qks.h 's/const uint32_t np = c.p\[ctr::P_NP\] ? c.p\[ctr::P_NP\] : c.N;/const uint32_t np = c.N;/'
run tile_rows   qks_tail.h 's/const int last_m1 = tl.N - (tl.ntiles - 1) \* NTT - 1;/const int last_m1 = NTT - 1;/'
echo "mutants not detected or broken: $bad" | tee -a "$R/mutants.log"
exit $bad
