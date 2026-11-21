#!/bin/sh
# Build and run tb_quant with the Vitis MinGW g++ (clean PATH, compile and run).
#   sh run_tests.sh [tb args...]       e.g. sh run_tests.sh --exhaustive 8 results/captured.qz
# Windows' application control sometimes blocks a freshly linked unsigned .exe ("Permission denied");
# relinking gives a new binary that is usually let through, so the build is retried a few times.
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/build"
for i in 1 2 3 4 5 6; do
  exe="$here/build/tb_quant_$i.exe"
  g++ -O2 -std=c++14 -I/c/Xilinx/Vitis_HLS/2022.2/include -o "$exe" "$here/tb_quant.cpp" "$here/quant_top.cpp" || exit 2
  if [ -x "$exe" ] && "$exe" --version >/dev/null 2>&1; then
    "$exe" "$@"
    exit $?
  fi
  rm -f "$exe"
done
echo "run_tests.sh: could not run a freshly built tb_quant (application control?)"
exit 3
