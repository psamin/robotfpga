# Vitis HLS 2022.2 synthesis (no csim: the host testbench is build.sh) of the softmax kernel for the K26.
#   cd fpga/kernels/softmax && vitis_hls -f run_softmax_hls.tcl
#   SMX_PERIOD=3.333 ...        clock period in ns (default 4.0 = 250 MHz)
#   SMX_FP32_OUT=1 ...          with the FP32 output datapath (default 0)
#   SMX_ENGINE=0 ...            rows layout only (default 1: rows + engine layouts)
#   SMX_TAG=x ...               project name suffix
# Projects are created in build/hls (Vitis HLS on Windows refuses '/' in project names).
set here [file dirname [file normalize [info script]]]
set period 4.0
if {[info exists ::env(SMX_PERIOD)]} { set period $::env(SMX_PERIOD) }
set cflags "-std=c++14"
set proj smx
foreach v {SMX_FP32_OUT SMX_ENGINE} tag {fp en} {
    if {[info exists ::env($v)]} {
        append cflags " -D$v=$::env($v)"
        append proj "_$tag$::env($v)"
    }
}
append proj "_p[string map {. {}} $period]"
if {[info exists ::env(SMX_TAG)]} { append proj "_$::env(SMX_TAG)" }

set bdir "$here/build/hls"
file mkdir $bdir
cd $bdir
open_project -reset $proj
set_top softmax_top
add_files "$here/softmax_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
