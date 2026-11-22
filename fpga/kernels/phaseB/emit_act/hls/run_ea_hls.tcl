# Vitis HLS 2022.2 csynth of the GEMM processes emit_act changes (hls/ea_top.cpp), against the snapshot (base) or
# the patched tree. No csim (the host testbench is run_tests.sh). One run per invocation:
#   EA_VARIANT=base|patched  EA_TOP=dq|st  EA_PERIOD=4.67|4.0   (PATH without Git dirs)
#   cmd //c "C:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_ea_hls.tcl"     (hls_run.sh does this)
# Trees: build/tree_base (snap/) and build/tree (snap/ + patched/), assembled by run_tests.sh / hls_run.sh.
# Projects: build/hls/ea_<top>_<variant>_p<period>.
set here [file dirname [file normalize [info script]]]
set variant [expr {[info exists ::env(EA_VARIANT)] ? $::env(EA_VARIANT) : "patched"}]
set top     [expr {[info exists ::env(EA_TOP)] ? $::env(EA_TOP) : "dq"}]
set period  [expr {[info exists ::env(EA_PERIOD)] ? $::env(EA_PERIOD) : 4.67}]
set tree [file normalize "$here/../build/[expr {$variant eq "base" ? "tree_base" : "tree"}]"]
set cflags "-std=c++14 -DFSTREAM_ARITH=16 -DFSTREAM_NBX=2 -DGS_NSB=10 -I$tree/hls"
if {$variant eq "patched"} { append cflags " -DEA_PATCHED=1" }
if {$top eq "dq"} { append cflags " -DEA_TOP_DQ=1" }
set proj "ea_${top}_${variant}_p[string map {. _} $period]"
set bdir [file normalize "$here/../build/hls"]
file mkdir $bdir
cd $bdir
open_project -reset $proj
set_top ea_top
add_files "$here/ea_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
