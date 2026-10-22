# Vivado 2022.2: rebuild the bitstream of an existing build_bd.tcl project with a newer HLS export of
# float_top, without regenerating the Zynq PS IP (its simulation-wrapper template is missing from this
# Vivado install, which makes build_bd.tcl's fresh project fail at IP generation).
#   cd fpga/hls && FSTREAM_TP=4 FSTREAM_SRC_TAG= FSTREAM_TAG=v4ip vivado -mode batch -source rebuild_ip.tcl
#   FSTREAM_SRC=_tp4 FSTREAM_TP=8 FSTREAM_TAG=v4 FSTREAM_MHZ=250 ...   # TP=8 export into the TP=4 project, 250 MHz
# Opens build/vivado_tp<TP><SRC_TAG>/float_kr260.xpr, points it at build/float_hls_tp<TP>_<TAG>/sol1/impl/ip,
# upgrades the float_top IP, re-runs synthesis + implementation, writes build/float_tp<TP>_<TAG>.bit/.hwh.
set here [file dirname [file normalize [info script]]]
set tp ""
if {[info exists ::env(FSTREAM_TP)]} { set tp "_tp$::env(FSTREAM_TP)" }
set src "$tp"
if {[info exists ::env(FSTREAM_SRC_TAG)] && $::env(FSTREAM_SRC_TAG) ne ""} { append src "_$::env(FSTREAM_SRC_TAG)" }
if {[info exists ::env(FSTREAM_SRC)]} { set src $::env(FSTREAM_SRC) }  ;# e.g. _tp4: any built project
set sfx "$tp"
if {[info exists ::env(FSTREAM_TAG)]} { append sfx "_$::env(FSTREAM_TAG)" }
set mhz ""
if {[info exists ::env(FSTREAM_MHZ)]} { set mhz $::env(FSTREAM_MHZ) }  ;# PL clock of the block design
set jobs 8
if {[info exists ::env(VIVADO_JOBS)]} { set jobs $::env(VIVADO_JOBS) }
set out [file normalize "$here/build"]
set ip_repo [file normalize "$here/build/float_hls$sfx/sol1/impl/ip"]
if {![file isdirectory $ip_repo]} { error "no HLS export at $ip_repo" }

open_project "$out/vivado$src/float_kr260.xpr"
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog -rebuild
open_bd_design [get_files float.bd]
set ips [get_ips -filter {NAME =~ *float_top*}]
puts "float_top IP(s): $ips"
upgrade_ip $ips -log "$out/upgrade$sfx.log"
report_ip_status
if {$mhz ne ""} { set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz [get_bd_cells zynq] }
validate_bd_design
save_bd_design
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 did not finish; see the run log" }
open_run impl_1
report_timing_summary -file "$out/timing$sfx.rpt"
report_utilization -file "$out/utilization$sfx.rpt"
file copy -force [glob "$out/vivado$src/float_kr260.runs/impl_1/*.bit"] "$out/float$sfx.bit"
file copy -force [glob "$out/vivado$src/float_kr260.gen/sources_1/bd/float/hw_handoff/*.hwh"] "$out/float$sfx.hwh"
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "wrote $out/float$sfx.bit and $out/float$sfx.hwh; WNS = $wns ns (>= 0 means the BD clock closed)"
