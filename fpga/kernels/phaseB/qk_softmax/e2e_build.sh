#!/bin/sh
# The end-to-end tree for the fused list: build/e2e/{hls,kernels} = the snapshot's engine files and the live
# kernels' emulators / headers (copied, read-only use), with integration/ (this fusion's patched files) on top;
# then the integrated float_top C model as two DLLs (Vitis MinGW g++, as fpga/hls/Makefile lib-seq):
#   seq_shim_qks.dll   the fused build: QK_SOFTMAX in the GEMM unit, no SOFTMAX unit (FSEQ_SOFTMAX_UNIT 0)
#   seq_shim_ctl.dll   control: the same tree with the SOFTMAX unit (-DFSEQ_SOFTMAX_UNIT=1), for the unfused list
#   sh fpga/kernels/phaseB/qk_softmax/e2e_build.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
E="$HERE/build/e2e"
rm -rf "$E"
mkdir -p "$E/hls" "$E/kernels" "$HERE/build/tmp"
for d in quantize norm_residual gelu_silu rope softmax sequencer attention; do
  mkdir -p "$E/kernels/$d"
  cp "$REPO/fpga/kernels/$d"/*.py "$E/kernels/$d/" 2>/dev/null || true
  cp "$REPO/fpga/kernels/$d"/*.h "$E/kernels/$d/" 2>/dev/null || true
done
cp "$REPO/fpga/kernels"/*.py "$REPO/fpga/kernels"/*.h "$E/kernels/"
cp "$HERE/snap/hls"/*.h "$HERE/snap/hls/seq_shim.cpp" "$HERE/snap/hls/e2e_seq.py" "$E/hls/"
cp "$REPO/fpga/hls/pack_float.py" "$E/hls/"
cp "$HERE/snap/kernels/contract.h" "$HERE/snap/kernels/contract.py" "$HERE/snap/kernels/contract_plan.py" "$E/kernels/"
cp "$HERE/snap/kernels/sequencer"/*.py "$HERE/snap/kernels/sequencer"/*.h "$E/kernels/sequencer/"
cp "$HERE/snap/kernels/softmax"/*.h "$E/kernels/softmax/"
(cd "$HERE/integration" && find hls kernels -type f) | while read -r f; do cp "$HERE/integration/$f" "$E/$f"; done
(cd "$E" && find . -type f \( -name '*.h' -o -name '*.py' -o -name '*.cpp' \) | sort | xargs sha256sum > SHA256SUMS)
export TMP="$HERE/build/tmp" TEMP="$HERE/build/tmp" TMPDIR="$HERE/build/tmp"
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
F="-std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off -DGS_NSB=10 -w -shared -static -I$INC -I$E/kernels -I$E/kernels/quantize"
cd "$E/hls"
g++ $F -o "$E/seq_shim_qks.dll" seq_shim.cpp
g++ $F -DFSEQ_SOFTMAX_UNIT=1 -o "$E/seq_shim_ctl.dll" seq_shim.cpp
ls -la "$E"/*.dll
