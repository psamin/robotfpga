# Vitis HLS 2022.2 synthesis (no csim: tb_quant.cpp is run with g++, see README.md) of the standalone
# int16 quantizer (quant_top.cpp -> quant_mode.h -> quant_int16.h) for the KR260 (K26).
#   cd fpga/kernels/quantize && vitis_hls -f run_hls.tcl
#   QZ_PERIOD=3.333 ...    clock period in ns (default 4.0 = 250 MHz)
#   QZ_IMPL=fdiv ...       per-lane FP32 divider core instead of the integer reciprocal datapath
#   QZ_FIFO=bram|uram ...  row FIFOs in BRAM / URAM instead of LUTRAM
#   QZ_DEPTH=512 ...       row FIFO depth in 256-bit pairs (default 128: rows <= 1024 in one DDR pass)
#   QZ_RBITS=2 ...         reciprocal bits per clock (default 3)
#   QZ_TAG=x ...           project name suffix
# Project build/qz_<impl>_p<period>[...] (Vitis HLS on Windows refuses '/' in a project name).
set here [file dirname [file normalize [info script]]]
set period 4.0
if {[info exists ::env(QZ_PERIOD)]} { set period $::env(QZ_PERIOD) }
set impl recip
if {[info exists ::env(QZ_IMPL)]} { set impl $::env(QZ_IMPL) }
set cflags "-std=c++14"
if {$impl eq "fdiv"} { append cflags " -DQZ_IMPL_FDIV" }
set proj "qz_${impl}_p[string map {. {}} $period]"
if {[info exists ::env(QZ_FIFO)]} {
  if {$::env(QZ_FIFO) eq "bram"} { append cflags " -DQZ_FIFO_BRAM"; append proj "_bram" }
  if {$::env(QZ_FIFO) eq "uram"} { append cflags " -DQZ_FIFO_URAM"; append proj "_uram" }
}
if {[info exists ::env(QZ_DEPTH)]} { append cflags " -DQZ_FIFO_DEPTH=$::env(QZ_DEPTH)"; append proj "_d$::env(QZ_DEPTH)" }
if {[info exists ::env(QZ_RBITS)]} { append cflags " -DQZ_RECIP_BITS=$::env(QZ_RBITS)"; append proj "_r$::env(QZ_RBITS)" }
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
