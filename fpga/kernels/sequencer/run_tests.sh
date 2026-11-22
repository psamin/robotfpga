#!/bin/sh
# Host checks of the sequencer (contract 1.0), in order; stops at the first failure. Outputs under build/.
#   sh run_tests.sh            (about 1 minute; the full-policy check is separate, see README "Policy check")
# 1. make_chunk.py: the -12392 list at CMA bases 0x4000_0000 and 0x8_0000_0000 (+ the Python decode trace)
# 2. tb_seq.exe: walk / dispatch / data / LIMIT / both chunk lists / decode_hw fuzz (writes the twin logs)
# 3. check_twin.py: every testbench start replayed through exec_ref.py
here=$(cd "$(dirname "$0")" && pwd)
PY=/c/kp/lerobot-venv/Scripts/python.exe
cd "$here" || exit 1
mkdir -p build/chunk build/twin
for b in lo:0x40000000 hi:0x800000000; do
  "$PY" make_chunk.py --base "${b#*:}" --out "build/chunk/chunk_${b%%:*}" || exit 1
done
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  g++ -std=c++14 -O2 -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -Wall -Wno-unknown-pragmas -Wno-unused-label \
    -static -o build/tb_seq.exe tb_seq.cpp || exit 1
  build/tb_seq.exe --twin build/twin build/chunk/chunk_lo.bin build/chunk/chunk_lo.trace.txt 40000000 \
    build/chunk/chunk_hi.bin build/chunk/chunk_hi.trace.txt 800000000 > build/tb_seq.log 2>&1
  rc=$?
  cat build/tb_seq.log
  exit $rc
) || { echo "run_tests: tb_seq FAILED"; exit 1; }
"$PY" check_twin.py build/twin/twin_lo.txt build/twin/twin_hi.txt || { echo "run_tests: twin FAILED"; exit 1; }
echo "run_tests: ALL PASS"
