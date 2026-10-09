# Run through run_mac_synthesis.ps1; all generated files stay in its run directory.
if {$argc != 4} { error "Expected repository root, top, lanes, period_ns" }
lassign $argv repo top lanes period
if {$top ni {mac mac_array}} { error "Unsupported top: $top" }
if {![string is integer -strict $lanes] || $lanes < 1 || $lanes > 1024} {
    error "lanes must be 1..1024"
}
if {![string is double -strict $period] || $period < 0.1 || $period > 1000} {
    error "period_ns must be 0.1..1000"
}
read_verilog -sv [file join $repo mac.sv]
if {$top eq "mac_array"} {
    read_verilog -sv [file join $repo board rtl mac_array.sv]
}
# Read the clock constraint before synthesis so optimization sees the target.
set xdc [open clock.xdc w]
puts $xdc "create_clock -name clk -period $period \[get_ports clk\]"
close $xdc
read_xdc clock.xdc
set options [list -top $top -part xck26-sfvc784-2LV-c -mode out_of_context]
if {$top eq "mac_array"} { lappend options -generic LANES=$lanes }
synth_design {*}$options
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
write_checkpoint -force synth.dcp
set evidence [open PASS.txt w]
puts $evidence "PASS: synthesis top=$top lanes=$lanes period_ns=$period Vivado=[version -short]"
close $evidence
exit
