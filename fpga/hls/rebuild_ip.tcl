# Vivado 2022.2: rebuild the bitstream of an existing build_bd.tcl project with a newer HLS export of
# float_top, without regenerating the Zynq PS IP (its simulation-wrapper template is missing from this
# Vivado install, which makes build_bd.tcl's fresh project fail at IP generation).
#   cd fpga/hls && FSTREAM_TP=4 FSTREAM_SRC_TAG= FSTREAM_TAG=v4ip vivado -mode batch -source rebuild_ip.tcl
#   FSTREAM_SRC=_tp4 FSTREAM_TP=8 FSTREAM_TAG=v4ip ...                # TP=8 export into the TP=4 project
#   FSTREAM_STRATEGY=Performance_ExplorePostRoutePhysOpt FSTREAM_OUT=tp4_v4ip_300 ...   # timing-closure re-run
# (FSTREAM_MHZ changes the block design's PL clock, which regenerates the SmartConnect IPs: that fails on this
#  Vivado install, so leave it unset and set the PL clock at run time instead.)
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
set outsfx $sfx
if {[info exists ::env(FSTREAM_OUT)]} { set outsfx "_$::env(FSTREAM_OUT)" }  ;# output file suffix (default = $sfx)
set strategy ""
if {[info exists ::env(FSTREAM_STRATEGY)]} { set strategy $::env(FSTREAM_STRATEGY) }  ;# e.g. Performance_ExplorePostRoutePhysOpt
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
if {$strategy ne ""} { set_property strategy $strategy [get_runs impl_1] }
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 did not finish; see the run log" }
open_run impl_1
report_timing_summary -file "$out/timing$outsfx.rpt"
report_utilization -file "$out/utilization$outsfx.rpt"
file copy -force [glob "$out/vivado$src/float_kr260.runs/impl_1/*.bit"] "$out/float$outsfx.bit"
file copy -force [glob "$out/vivado$src/float_kr260.gen/sources_1/bd/float/hw_handoff/*.hwh"] "$out/float$outsfx.hwh"
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "wrote $out/float$outsfx.bit and $out/float$outsfx.hwh; WNS = $wns ns (>= 0 means the BD clock closed)"
