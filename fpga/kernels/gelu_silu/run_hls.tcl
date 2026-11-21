# Vitis HLS 2022.2 csynth of the gelu_silu kernel (no csim: build tb_gelu_silu.cpp with g++, see README.md).
#   vitis_hls -f run_hls.tcl
# Environment: GS_PERIOD (ns, default 4.0), GS_NSB (9 | 10, default 9), GS_LANES (8 | 4, default 8),
#              GS_YS_URAM=1 (row FIFO in URAM), GS_DF_DSP=1 (table interpolation multiply on DSPs),
#              GS_TAG (project name suffix). Projects go to ../../../build/gelu_silu/hls/ (no '/' in a project name).
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(GS_PERIOD)] ? $::env(GS_PERIOD) : 4.0}]
set nsb    [expr {[info exists ::env(GS_NSB)] ? $::env(GS_NSB) : 9}]
set lanes  [expr {[info exists ::env(GS_LANES)] ? $::env(GS_LANES) : 8}]
set cflags "-std=c++14 -DGS_NSB=$nsb -DGS_LANES=$lanes"
set proj "gs_l${lanes}_nsb${nsb}_p[string map {. _} $period]"
if {[info exists ::env(GS_YS_URAM)]} { append cflags " -DGS_YS_URAM"; append proj "_uram" }
if {[info exists ::env(GS_DF_DSP)]} { append cflags " -DGS_DF_DSP"; append proj "_dfdsp" }
if {[info exists ::env(GS_TAG)]} { append proj "_$::env(GS_TAG)" }

set bdir [file normalize "$here/../../../build/gelu_silu/hls"]
file mkdir $bdir
cd $bdir
open_project -reset $proj
set_top gelu_silu_top
add_files "$here/gelu_silu_top.cpp" -cflags "$cflags -I$here"
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
