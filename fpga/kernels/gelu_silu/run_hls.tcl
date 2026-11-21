# Vitis HLS 2022.2 csynth of the GELU / SILU_MUL mode (gs::act_mode under the standalone top gelu_silu_top.cpp).
# No csim (build tb_gelu_silu.cpp with g++: run_tests.sh). The mode's own resources are the act_mode instance in
# the report; the top adds the io m_axi adapter and the s_axi_control registers that float_top already has.
#   vitis_hls -f run_hls.tcl
# Environment: GS_PERIOD (ns, default 4.0), GS_NSB (9 | 10, default 9), GS_GUARD (0 | 1, default 1),
#              GS_GBUF_BRAM=1 (gate buffer in BRAM, default URAM), GS_YS_URAM=1 (row buffer in URAM, default BRAM),
#              GS_TAG (project name suffix).
# Projects go to ../../../build/gelu_silu/hls/ (no '/' in a project name).
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(GS_PERIOD)] ? $::env(GS_PERIOD) : 4.0}]
set nsb    [expr {[info exists ::env(GS_NSB)] ? $::env(GS_NSB) : 9}]
set cflags "-std=c++14 -DGS_NSB=$nsb"
set proj "gs_c1_nsb${nsb}_p[string map {. _} $period]"
if {[info exists ::env(GS_GUARD)]} { append cflags " -DGS_GUARD=$::env(GS_GUARD)"; append proj "_guard$::env(GS_GUARD)" }
if {[info exists ::env(GS_GBUF_BRAM)]} { append cflags " -DGS_GBUF_BRAM"; append proj "_gbram" }
if {[info exists ::env(GS_YS_URAM)]} { append cflags " -DGS_YS_URAM"; append proj "_ysuram" }
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
