# Vitis HLS 2022.2 synthesis (no csim: tb_quant.cpp runs with g++, see README.md) of the row unit's QUANT_X /
# QUANT_W / QUANT_WT / COPY mode function qz::row_mode (quant_top.cpp -> quant_mode.h -> quant_int16.h) for the
# KR260 (K26). quant_top fetches one 256-byte record over io and decodes it, so the report shows row_mode as its own
# instance: the function the sequencer's dispatch switch calls.
#   cd fpga/kernels/quantize && vitis_hls -f run_hls.tcl        (PATH without the Git directories)
#   QZ_PERIOD=3.333 ...    clock period in ns (default 4.0 = 250 MHz)
#   QZ_IMPL=fdiv ...       per-lane FP32 divider core instead of the integer reciprocal datapath
#   QZ_ENGINES=4 ...       reciprocal engines in sr (default 2)
#   QZ_TAG=x ...           project name suffix
# Project build/qzm_<impl>_p<period>[...] (Vitis HLS on Windows refuses '/' in a project name).
set here [file dirname [file normalize [info script]]]
set period 4.0
if {[info exists ::env(QZ_PERIOD)]} { set period $::env(QZ_PERIOD) }
set impl recip
if {[info exists ::env(QZ_IMPL)]} { set impl $::env(QZ_IMPL) }
set cflags "-std=c++14"
if {$impl eq "fdiv"} { append cflags " -DQZ_IMPL_FDIV" }
set proj "qzm_${impl}_p[string map {. {}} $period]"
if {[info exists ::env(QZ_ENGINES)]} { append cflags " -DQZ_SC_ENGINES=$::env(QZ_ENGINES)"; append proj "_e$::env(QZ_ENGINES)" }
if {[info exists ::env(QZ_TAG)]} { append proj "_$::env(QZ_TAG)" }

file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top quant_top
add_files "$here/quant_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
