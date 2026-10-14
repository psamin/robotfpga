# Vitis HLS 2022.2 build of the full-precision linear engine for the KR260 (K26), issue #148.
#   make -C hls/stream test          # once: writes ../../build/stream/float/case_*
#   cd hls/stream && vitis_hls -f run_float_hls.tcl
# Target 300 MHz; the four weight bundles go to S_AXI_HP0..HP3_FPD in the block design.
set here [file dirname [file normalize [info script]]]
set case [file normalize "$here/../../build/stream/float/case_2560x960x50"]

open_project -reset build/float_hls
set_top float_top
add_files "$here/float_top.cpp" -cflags "-std=c++14"
add_files -tb "$here/tb_float.cpp" -cflags "-std=c++14"
add_files -tb "$here/float_shim.cpp" -cflags "-std=c++14"

open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 3.333 -name default
config_interface -m_axi_addr64

csim_design -argv "$case"
csynth_design
export_design -format ip_catalog -rtl verilog
exit
