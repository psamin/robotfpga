#!/bin/sh
# One csynth of run_ea_hls.tcl: hls_run.sh <base|patched> <dq|st> <period ns>. Log: build/hls/<project>.log.
# Refuses to start while another vitis_hls.exe runs (one HLS run at a time on this machine), like the other
# phase-B kernels. Assembles build/tree_base (the snapshot) if missing; build/tree comes from run_tests.sh.
HERE="$(cd "$(dirname "$0")" && pwd)"
E="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$E/../../../.." && pwd)"
TB="$E/build/tree_base"
if [ ! -d "$TB" ]; then
  mkdir -p "$TB/hls" "$TB/kernels"
  cp "$E"/snap/hls/*.h "$TB/hls/"
  cp "$E/snap/kernels/contract.h" "$TB/kernels/"
fi
[ -d "$E/build/tree" ] || { echo "run run_tests.sh first (build/tree)"; exit 2; }
mkdir -p "$E/build/hls"
if tasklist 2>/dev/null | grep -qi vitis_hls.exe; then
  echo "another vitis_hls is running; not started"
  exit 3
fi
P=$(echo "$3" | tr . _)
cd "$HERE"
EA_VARIANT=$1 EA_TOP=$2 EA_PERIOD=$3 PATH=/c/Windows/System32:/c/Windows \
  cmd //c "C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_ea_hls.tcl" > "$E/build/hls/ea_$2_$1_p$P.log" 2>&1
echo "rc=$? ($1 $2 $3)"
