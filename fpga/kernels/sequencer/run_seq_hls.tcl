# Vitis HLS 2022.2 csynth of the standalone sequencer top (contract 1.0), K26 (xck26-sfvc784-2LV-c).
#   cd fpga/kernels/sequencer && vitis_hls -f run_seq_hls.tcl        (PATH without Git dirs; ~1 min, < 1 GB)
# Environment (all optional; the defaults are the design):
#   SEQ_PERIOD=4.0      clock period, ns (250 MHz)
#   SEQ_UNITS=1         0: no units at all (-DSEQ_NO_UNITS): the sequencer alone, without the units' port muxing
#   SEQ_CYCLES=1        0: no cycle counter (status cycles read 0)
#   SEQ_UNROLLED=0      1: build/contract_unrolled/contract.h, contract.h with UNROLL in decode()/validate_t()'s
#                       loops and INLINE in decode() (the change proposed to the contract owner). Write it first:
#                       python hls_contract_unroll.py build/contract_unrolled/contract.h --inline-decode
#   SEQ_CHECK_INLINE=1  0: decode + validate_hw in their own module
# Project: build/hls_<tag>, tag = p<period> + u0 / c0 / unr / ci0 for each non-default flag.
set here [file dirname [file normalize [info script]]]
proc envor {name default} { if {[info exists ::env($name)]} { return $::env($name) } { return $default } }
set period [envor SEQ_PERIOD 4.0]
set units [envor SEQ_UNITS 1]
set cycles [envor SEQ_CYCLES 1]
set unrolled [envor SEQ_UNROLLED 0]
set cinline [envor SEQ_CHECK_INLINE 1]
set tag "p[string map {. ""} $period]"
set cflags "-std=c++14"
if {!$units} { append cflags " -DSEQ_NO_UNITS"; append tag "_u0" }
if {!$cycles} { append cflags " -DSEQ_CYCLES=0"; append tag "_c0" }
if {$unrolled} { append cflags " -DSEQ_CONTRACT_INC -I$here/build/contract_unrolled"; append tag "_unr" }
if {!$cinline} { append cflags " -DSEQ_CHECK_INLINE=0"; append tag "_ci0" }
set proj "hls_$tag"
puts "SEQ csynth $proj: $cflags"

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
