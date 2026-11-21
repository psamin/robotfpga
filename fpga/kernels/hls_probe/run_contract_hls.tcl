# csynth of the contract decode + validate_hw probe (one run, about 1-3 min). PERIOD env: clock in ns (4.0).
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(PERIOD)] ? $::env(PERIOD) : 4.0}]
file mkdir [file join $here build]
cd [file join $here build]
open_project -reset contract_p[string map {. _} $period]
set_top contract_top
add_files [file join $here contract_top.cpp] -cflags "-std=c++14"
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
