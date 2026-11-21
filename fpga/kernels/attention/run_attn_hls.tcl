# Vitis HLS 2022.2 csynth of the QUANT_W / QUANT_WT mode (attn_kv.h) for the KR260 (K26), standalone top attn_kv_top.
#   cd fpga/kernels/attention && vitis_hls -f run_attn_hls.tcl        (csim is not used: tb_attn.cpp runs with g++)
#   ATTN_PERIOD=3.6 ...     clock period in ns (default 4.0 = 250 MHz)
#   ATTN_TAG=x ...          project name suffix
set here [file dirname [file normalize [info script]]]
set period 4.0
if {[info exists ::env(ATTN_PERIOD)]} { set period $::env(ATTN_PERIOD) }
set proj attn_kv_p[string map {. {}} $period]
if {[info exists ::env(ATTN_TAG)]} { append proj "_$::env(ATTN_TAG)" }
# Vitis HLS 2022.2 on Windows refuses '/' in a project name, so build inside build/.
file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top attn_kv_top
add_files "$here/attn_kv_top.cpp" -cflags "-std=c++14"
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
