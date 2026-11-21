# Vitis HLS 2022.2 synthesis (csynth only; the C model is checked by tb_rope.cpp with g++) of the RoPE kernel.
#   ROPE_PERIOD=4.0     clock period in ns (default 4.0 = 250 MHz)
#   vitis_hls -f fpga/kernels/rope/run_rope_hls.tcl      -> fpga/kernels/rope/build/rope_p<period>/sol1
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(ROPE_PERIOD)] ? $::env(ROPE_PERIOD) : 4.0}]
set ptag [string map {. ""} $period]
set proj "rope_p${ptag}"

# Vitis HLS 2022.2 on Windows refuses '/' in a project name, so build inside build/.
file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top rope_top
add_files "$here/rope_top.cpp" -cflags "-std=c++14"
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
