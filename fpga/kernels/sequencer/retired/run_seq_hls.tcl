# Vitis HLS 2022.2 csynth of the standalone sequencer top (stub modes), K26.
#   cd fpga/kernels/sequencer && vitis_hls -f run_seq_hls.tcl
#   SEQ_VARIANT=0..5    dispatch experiment (seq_top.cpp header); default 0
#   run_hls_all.sh runs a list of variant:period pairs one after another
#   SEQ_PERIOD=4.0      clock period in ns (default 4.0 = 250 MHz)
# Project: build/seq_v<VARIANT>_p<PERIOD without '.'> (no '/' in the project name on Windows).
set here [file dirname [file normalize [info script]]]
set variant 0
if {[info exists ::env(SEQ_VARIANT)]} { set variant $::env(SEQ_VARIANT) }
set period 4.0
if {[info exists ::env(SEQ_PERIOD)]} { set period $::env(SEQ_PERIOD) }
set ptag [string map {. ""} $period]
set proj "seq_v${variant}_p${ptag}"
set cflags "-std=c++14 -DSEQ_VARIANT=$variant"

file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top seq_top
add_files "$here/seq_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
