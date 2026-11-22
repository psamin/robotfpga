# Vitis HLS 2022.2 csynth of the emit_amax probe (amx_top.cpp), xck26-sfvc784-2LV-c, csynth only.
#   AMX_BASELINE=1 AMX_PERIOD=4.67 vitis_hls -f run_hls.tcl    -> project amx_base_p467
#   AMX_PERIOD=4.67 vitis_hls -f run_hls.tcl                   -> project amx_fused_p467
#   AMX_QL=2 ...                                               -> 2 quantizer lanes (project suffix _ql2)
set here [file dirname [file normalize [info script]]]
set amax [file normalize "$here/.."]
set repo [file normalize "$here/../../../../.."]
set hlsi "-I$repo/fpga/hls -I$repo/fpga/kernels/quantize"
set period 4.67
if {[info exists ::env(AMX_PERIOD)]} { set period $::env(AMX_PERIOD) }
set ptag [string map {. ""} $period]
set cflags "-std=c++14 -DFSTREAM_ARITH=16 -DFSTREAM_NBX=2"
if {[info exists ::env(AMX_BASELINE)]} {
    set cflags "$cflags -DAMX_BASELINE -I$amax/snap $hlsi"
    set proj amx_base_p$ptag
} else {
    set cflags "$cflags -I$amax/patched/hls $hlsi"
    set proj amx_fused_p$ptag
    if {[info exists ::env(AMX_QL)]} {
        append cflags " -DAMX_QL=$::env(AMX_QL)"
        append proj "_ql$::env(AMX_QL)"
    }
}
file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top amx_top
add_files "$here/amx_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
