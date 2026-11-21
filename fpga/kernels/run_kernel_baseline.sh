#!/bin/sh
# Kernel baseline (CONTRACT.md 7): build every existing testbench of the kernels and of the int16 engine and run it
# on the vectors already in the repo; builds, logs and the engine's generated cases go under fpga/kernels/build/baseline.
#   sh fpga/kernels/run_kernel_baseline.sh [all|seq|smx|gs|nr|qz|rope|attn|eng]
R=/c/Users/athit/robotfpga
K=$R/fpga/kernels
H=$R/fpga/hls
S=$K/build/baseline
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
INC=/c/Xilinx/Vitis_HLS/2022.2/include
CXX="g++ -std=c++14 -O2 -Wno-unknown-pragmas -Wno-unused-label -I$INC"
mkdir -p "$S/bin" "$S/log"
run() {  # name, then the command
  n=$1; shift
  start=$(date +%s)
  "$@" > "$S/log/$n.log" 2>&1
  rc=$?
  echo "$n rc=$rc $(( $(date +%s) - start ))s :: $(tail -n 1 "$S/log/$n.log" | tr -d '\r' | cut -c1-150)"
}
which=${1:-all}
case $which in all|seq)
  $CXX -Wall -o "$S/bin/tb_seq.exe" "$K/sequencer/tb_seq.cpp" && run tb_seq "$S/bin/tb_seq.exe" "$K/sequencer/build/py" ;;
esac
case $which in all|smx)
  $CXX -D__USE_MINGW_ANSI_STDIO=1 -DSMX_FP32_OUT=0 -DSMX_ENGINE=1 -o "$S/bin/tb_softmax_fp0.exe" "$K/softmax/tb_softmax.cpp" &&
    run tb_softmax_fp0 "$S/bin/tb_softmax_fp0.exe" $K/softmax/build/cases/*
  $CXX -D__USE_MINGW_ANSI_STDIO=1 -DSMX_FP32_OUT=0 -DSMX_ENGINE=0 -o "$S/bin/tb_softmax_rows.exe" "$K/softmax/tb_softmax.cpp" &&
    run tb_softmax_rows "$S/bin/tb_softmax_rows.exe" $K/softmax/build/cases/rows_* ;;
esac
case $which in all|gs)
  for lanes in 4 8; do
    $CXX -DGS_NSB=9 -DGS_LANES=$lanes -I"$K/gelu_silu" -o "$S/bin/tb_gs_l$lanes.exe" "$K/gelu_silu/tb_gelu_silu.cpp" "$K/gelu_silu/gelu_silu_top.cpp" &&
      (cd $R/build/gelu_silu && run tb_gelu_silu_l$lanes "$S/bin/tb_gs_l$lanes.exe" vec9/* real9/*)
  done ;;
esac
case $which in all|nr)
  $CXX -ffp-contract=off -I"$K/norm_residual" -o "$S/bin/tb_nr.exe" "$K/norm_residual/tb_nr.cpp" "$K/norm_residual/nr_top.cpp" &&
    run tb_nr "$S/bin/tb_nr.exe" $K/norm_residual/build/vec/*.bin ;;
esac
case $which in all|qz)
  $CXX -o "$S/bin/tb_quant.exe" "$K/quantize/tb_quant.cpp" "$K/quantize/quant_top.cpp" &&
    (cd $K/quantize && run tb_quant_quick "$S/bin/tb_quant.exe" --quick) ;;
esac
case $which in all|rope)
  $CXX -o "$S/bin/tb_rope.exe" "$K/rope/tb_rope.cpp" "$K/rope/rope_top.cpp" &&
    run tb_rope "$S/bin/tb_rope.exe" "$K/rope/build/vectors" ;;
esac
case $which in all|attn)
  $CXX -ffp-contract=off -o "$S/bin/tb_attn.exe" "$K/attention/tb_attn.cpp" &&
    (cd $K/attention && run tb_attn "$S/bin/tb_attn.exe" vectors/vision vectors/prefix vectors/expert_self vectors/expert_cross)
  $CXX -ffp-contract=off -DFSTREAM_ARITH=16 -DATTN_REAL_ENGINE -I"$H" -o "$S/bin/tb_attn_real.exe" "$K/attention/tb_attn.cpp" &&
    (cd $K/attention && run tb_attn_real "$S/bin/tb_attn_real.exe" vectors/vision vectors/prefix vectors/expert_self vectors/expert_cross) ;;
esac
case $which in all|eng)
  # the engine's own int16 tests, cases generated into scratch (pack_float.py only reads the repo)
  PYV=/c/Users/athit/robotfpga/.venv/Scripts/python.exe
  [ -d "$S/int16" ] || "$PYV" "$H/pack_float.py" --tests-int16 "$S/int16" > "$S/log/pack_cases.log" 2>&1
  g++ -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -DFSTREAM_ARITH=16 -ffp-contract=off -fopenmp \
    -o "$S/bin/tb_float_int16.exe" "$H/tb_float.cpp" "$H/float_shim.cpp" "$H/tb_int16_units.cpp" &&
    run tb_int16_units "$S/bin/tb_float_int16.exe" --units
  ok=0; bad=0
  for d in "$S"/int16/case_*; do
    if "$S/bin/tb_float_int16.exe" "$d" > "$S/log/case_$(basename "$d").log" 2>&1; then ok=$((ok+1)); else bad=$((bad+1)); fi
  done
  echo "tb_float_int16 cases pass=$ok fail=$bad"
  g++ -std=c++14 -O2 -D__VITIS_HLS__ -DFSTREAM_ARITH=16 -ffp-contract=off -Wno-unknown-pragmas -I$INC -Wl,--stack,268435456 \
    -o "$S/bin/tb_int16_top.exe" "$H/tb_int16_top.cpp" "$H/float_top.cpp" && {
    ok=0; bad=0
    for d in "$S"/int16/case_*; do
      [ -f "$d/xmem.bin" ] || continue
      TB_DUMP_Y="$S/y_host.bin" "$S/bin/tb_float_int16.exe" "$d" > /dev/null 2>&1 &&
        "$S/bin/tb_int16_top.exe" "$d" "$S/y_host.bin" > "$S/log/top_$(basename "$d").log" 2>&1 && ok=$((ok+1)) || bad=$((bad+1))
    done
    echo "tb_int16_top cases pass=$ok fail=$bad"
  } ;;
esac
