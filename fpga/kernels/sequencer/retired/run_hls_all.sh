#!/bin/sh
# Runs the csynth experiments one after another (one HLS process at a time).
#   sh run_hls_all.sh "0:4.0 1:4.0 2:4.0 3:4.0 0:3.333"
# Logs: build/hls_v<variant>_p<period without '.'>.log
here=$(cd "$(dirname "$0")" && pwd)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
cd "$here" || exit 1
for vp in ${1:-"0:4.0 1:4.0 2:4.0 3:4.0"}; do
  v=${vp%%:*}; p=${vp#*:}; pt=$(echo "$p" | tr -d .)
  echo "=== variant $v period $p start $(date +%T)"
  SEQ_VARIANT=$v SEQ_PERIOD=$p /c/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f "$here/run_seq_hls.tcl" > "$here/build/hls_v${v}_p${pt}.log" 2>&1
  echo "=== variant $v period $p rc $? end $(date +%T)"
done
