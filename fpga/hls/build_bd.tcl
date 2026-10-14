# Vivado 2022.2: KR260 block design + bitstream for the full-precision SmolVLA engine.
# Run after run_float_hls.tcl has exported the IP:
#   cd fpga/hls && vitis_hls -f run_float_hls.tcl && vivado -mode batch -source build_bd.tcl
# Output: build/float.bit + build/float.hwh (same base name, as PYNQ expects).
# UNTESTED: written without Vivado access. Check the board part and the IP name if a step fails.
set here [file dirname [file normalize [info script]]]
set ip_repo [file normalize "$here/build/float_hls/sol1/impl/ip"]
set out [file normalize "$here/build"]

create_project -force float_kr260 "$out/vivado" -part xck26-sfvc784-2LV-c
set bp [lindex [lsort [get_board_parts -quiet *kr260_som*]] end]
if {$bp eq ""} { error "KR260 board files missing: Vivado > Tools > Vivado Store > Boards > Kria KR260" }
set_property board_part $bp [current_project]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

create_bd_design float
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
# S_AXI_GP2..5 = HP0..HP3 (weights), S_AXI_GP0 = HPC0 (activations), all 128-bit; PL clock 300 MHz.
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {300} \
] $ps

set eng [create_bd_cell -type ip -vlnv [lindex [get_ipdefs *float_top*] 0] float_top_0]
set clk /zynq/pl_clk0
foreach {m s} {hp0 S_AXI_HP0_FPD hp1 S_AXI_HP1_FPD hp2 S_AXI_HP2_FPD hp3 S_AXI_HP3_FPD io S_AXI_HPC0_FPD} {
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master $clk Clk_slave Auto Clk_xbar Auto \
    Master /float_top_0/m_axi_$m Slave /zynq/$s intc_ip Auto master_apm 0] [get_bd_intf_pins zynq/$s]
}
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master $clk Clk_slave Auto Clk_xbar Auto \
  Master /zynq/M_AXI_HPM0_FPD Slave /float_top_0/s_axi_control intc_ip New master_apm 0] \
  [get_bd_intf_pins float_top_0/s_axi_control]
assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files float.bd] -top]
add_files -norecurse $wrapper
set_property top float_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_timing_summary -file "$out/timing.rpt"
report_utilization -file "$out/utilization.rpt"
file copy -force [glob "$out/vivado/float_kr260.runs/impl_1/*.bit"] "$out/float.bit"
file copy -force [glob "$out/vivado/float_kr260.gen/sources_1/bd/float/hw_handoff/*.hwh"] "$out/float.hwh"
puts "wrote $out/float.bit and $out/float.hwh; check $out/timing.rpt for WNS >= 0"
