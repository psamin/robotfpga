# Vitis HLS build for the K26 (KV260 / KR260).
#
#   cd hls
#   vitis_hls -f run_hls.tcl                     # Vitis HLS 2023.2 and older
#   vitis-run --mode hls --tcl run_hls.tcl       # Vitis 2024.1 and newer
#
# Steps: C simulation against the golden vectors, synthesis, then export as an
# IP block for Vivado. Reports land in build/policy_hls/sol1/syn/report/.

set here [file dirname [file normalize [info script]]]
set artifacts [file normalize "$here/../artifacts/standin"]

open_project -reset build/policy_hls
set_top policy_top
add_files "$here/policy_top.cpp" -cflags "-std=c++14"
add_files -tb "$here/tb_top.cpp" -cflags "-std=c++14"

open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
# 100 MHz matches the default PL clock (pl_clk0) in Kria designs. Raise it in M5.
create_clock -period 10 -name default

csim_design -argv $artifacts
csynth_design
export_design -format ip_catalog -rtl verilog

exit
