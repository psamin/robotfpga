# Vitis HLS 2022.2 synthesis (no csim: the host testbench is build.sh) of the SOFTMAX unit for the K26.
#   cd fpga/kernels/softmax && vitis_hls -f run_softmax_hls.tcl
#   SMX_PERIOD=3.333 ...        clock period in ns (default 4.0 = 250 MHz)
#   SMX_TAG=x ...               project name suffix
# Top: softmax_top (float_top's exact ports; record fetch + ctr::decode + ctr::validate_hw stand in for the
# sequencer). The unit's own cost is the softmax_mode instance in the report (the contract budget line).
# Projects are created in build/hls (Vitis HLS on Windows refuses '/' in project names).
set here [file dirname [file normalize [info script]]]
set period 4.0
if {[info exists ::env(SMX_PERIOD)]} { set period $::env(SMX_PERIOD) }
set proj smx_c1_p[string map {. {}} $period]
if {[info exists ::env(SMX_TAG)]} { append proj "_$::env(SMX_TAG)" }

set bdir "$here/build/hls"
file mkdir $bdir
cd $bdir
open_project -reset $proj
set_top softmax_top
add_files "$here/softmax_top.cpp" -cflags "-std=c++14"
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
