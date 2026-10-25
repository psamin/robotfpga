# Vivado 2022.2: re-run implementation + bitstream on a project build_bd.tcl already synthesized
# (e.g. after a crash in impl_1). Same FSTREAM_TP convention and outputs as build_bd.tcl.
#   cd fpga/hls && FSTREAM_TP=4 vivado -mode batch -source resume_impl.tcl
set here [file dirname [file normalize [info script]]]
set sfx ""
if {[info exists ::env(FSTREAM_TP)]} { set sfx "_tp$::env(FSTREAM_TP)" }
set jobs 4
if {[info exists ::env(VIVADO_JOBS)]} { set jobs $::env(VIVADO_JOBS) }
set out [file normalize "$here/build"]

open_project "$out/vivado$sfx/float_kr260.xpr"
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 did not finish; see the run log" }
open_run impl_1
report_timing_summary -file "$out/timing$sfx.rpt"
report_utilization -file "$out/utilization$sfx.rpt"
file copy -force [glob "$out/vivado$sfx/float_kr260.runs/impl_1/*.bit"] "$out/float$sfx.bit"
file copy -force [glob "$out/vivado$sfx/float_kr260.gen/sources_1/bd/float/hw_handoff/*.hwh"] "$out/float$sfx.hwh"
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "wrote $out/float$sfx.bit and $out/float$sfx.hwh; WNS = $wns ns (>= 0 means 300 MHz closed)"
