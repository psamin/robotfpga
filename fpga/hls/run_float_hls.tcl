# Vitis HLS 2022.2 build of the full-precision linear engine for the KR260 (K26), issue #148.
#   make -C fpga/hls test          # once: writes ../../build/stream/float/case_*
#   cd fpga/hls && vitis_hls -f run_float_hls.tcl
#   FSTREAM_TP=8 vitis_hls -f run_float_hls.tcl    # other TP: project build/float_hls_tp8
# Target 300 MHz; the four weight bundles go to S_AXI_HP0..HP3_FPD in the block design.
set here [file dirname [file normalize [info script]]]
set case [file normalize "$here/../../build/stream/float/case_2560x960x50"]
set cflags "-std=c++14"
set proj float_hls
if {[info exists ::env(FSTREAM_TP)]} {
    append cflags " -DFSTREAM_TP=$::env(FSTREAM_TP)"
    set proj float_hls_tp$::env(FSTREAM_TP)
}

# Vitis HLS 2022.2 on Windows refuses '/' in a project name, so build inside build/.
file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top float_top
add_files "$here/float_top.cpp" -cflags $cflags
add_files -tb "$here/tb_float.cpp" -cflags $cflags
add_files -tb "$here/float_shim.cpp" -cflags $cflags

open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 3.333 -name default
config_interface -m_axi_addr64

csim_design -argv "$case"
csynth_design
export_design -format ip_catalog -rtl verilog
exit
