#!/bin/sh
# The contract's own checks (CONTRACT.md 7): contract.py selftest + vectors, tb_contract (contract.h against
# contract.py and quantize/quant_int16.h), the -12392 chunk list at two CMA bases validated in Python and C++,
# and its time per mode.   sh fpga/kernels/run_contract_tests.sh      (outputs under fpga/kernels/build/contract)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
B="$HERE/build/contract"
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python3
mkdir -p "$B"
"$PY" "$HERE/contract.py" --selftest --vectors "$B/vec"
"$PY" "$HERE/contract_plan.py" --mhz 250 --out "$B/chunk_hi.bin" --json "$B/plan_250.json" > "$B/plan_250.txt"
"$PY" "$HERE/contract_plan.py" --mhz 250 --base 0x40000000 --out "$B/chunk_lo.bin" > /dev/null
head -4 "$B/plan_250.txt"
(
  export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
  g++ -std=c++14 -O2 -ffp-contract=off -Wall -Wno-unknown-pragmas -Wno-unused-label -Wno-comment \
    -I/c/Xilinx/Vitis_HLS/2022.2/include -I"$HERE" -o "$B/tb_contract.exe" "$HERE/tb_contract.cpp" 2> "$B/build.log"
  "$B/tb_contract.exe" "$B/vec" "$B/chunk_hi.bin" "$B/chunk_lo.bin"
)
