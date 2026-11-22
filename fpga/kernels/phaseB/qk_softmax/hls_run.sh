#!/bin/sh
# One Vitis HLS csynth of run_qks_hls.tcl (env QKS_TOP / QKS_PERIOD / QKS_TAG pass through); $1 = log name.
# Refuses to start while another vitis_hls.exe runs (one HLS run at a time on this machine).
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build/hls"
if tasklist 2>/dev/null | grep -qi vitis_hls.exe; then
  echo "another vitis_hls is running; not started"
  exit 3
fi
cd "$HERE"
PATH=/c/Windows/System32:/c/Windows cmd //c "C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_qks_hls.tcl" > "build/hls/$1.log" 2>&1
echo "rc=$?"
