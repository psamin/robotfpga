# Vitis HLS 2022.2 csynth of the qk_softmax fusion for xck26-sfvc784-2LV-c (no csim: run_tests.sh is the C test).
#   QKS_TOP=fused|base|tail   (default fused)  qks_top.cpp: the fused GEMM unit, the integrator's GEMM unit as
#                                              snapshotted, or the fused tail alone
#   QKS_PERIOD=4.67           clock period in ns (4.67 = 214 MHz, the reachable 300 * 5 / 7 step; 4.0 = 250 MHz)
#   QKS_TAG=x                 project name suffix
# Projects in build/hls (Vitis HLS on Windows refuses '/' in a project name). Run with a PATH without Git dirs:
#   sh fpga/kernels/phaseB/qk_softmax/hls_run.sh <log name>
set here [file dirname [file normalize [info script]]]
set top fused
if {[info exists ::env(QKS_TOP)]} { set top $::env(QKS_TOP) }
set period 4.67
if {[info exists ::env(QKS_PERIOD)]} { set period $::env(QKS_PERIOD) }
set proj qks_${top}_p[string map {. {}} $period]
if {[info exists ::env(QKS_TAG)]} { append proj "_$::env(QKS_TAG)" }
set cflags "-std=c++14 -DFSTREAM_ARITH=16 -DFSTREAM_NBX=2 -I$here/snap/hls -I$here"
set fn float_top
if {$top eq "base"} { append cflags " -DQKS_BASE" }
if {$top eq "tail"} { append cflags " -DQKS_TAIL"; set fn qks_tail_top }

set bdir "$here/build/hls"
file mkdir $bdir
cd $bdir
open_project -reset $proj
set_top $fn
add_files "$here/qks_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
