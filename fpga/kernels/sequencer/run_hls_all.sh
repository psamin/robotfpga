#!/bin/sh
# csynth runs of the standalone sequencer, one after another (one HLS process at a time).
#   sh run_hls_all.sh "UNITS:CYCLES:UNROLLED:PERIOD ..."     default: the design and its references (below)
# Logs: build/hls_<tag>.log; reports: build/hls_<tag>/sol1/syn/report/csynth.rpt (summary: python hls_summary.py)
here=$(cd "$(dirname "$0")" && pwd)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
cd "$here" || exit 1
/c/kp/lerobot-venv/Scripts/python.exe "$here/hls_contract_unroll.py" "$here/build/contract_unrolled/contract.h" --inline-decode > /dev/null
list=${1:-"1:1:0:4.0 0:1:0:4.0 1:0:0:4.0 1:1:1:4.0 0:1:1:4.0 1:1:0:3.333"}
for v in $list; do
  u=${v%%:*}; r=${v#*:}; c=${r%%:*}; r=${r#*:}; n=${r%%:*}; p=${r#*:}
  tag="p$(echo "$p" | tr -d .)"; [ "$u" = 0 ] && tag="${tag}_u0"; [ "$c" = 0 ] && tag="${tag}_c0"; [ "$n" = 1 ] && tag="${tag}_unr"
  echo "=== $tag start $(date +%T)"
  SEQ_UNITS=$u SEQ_CYCLES=$c SEQ_UNROLLED=$n SEQ_PERIOD=$p /c/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f "$here/run_seq_hls.tcl" > "$here/build/hls_$tag.log" 2>&1
  echo "=== $tag rc $? end $(date +%T)"
done
