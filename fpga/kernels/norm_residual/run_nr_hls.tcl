# Vitis HLS 2022.2 synthesis (csynth only; csim is replaced by tb_nr.cpp with g++) of the NORM mode, driven by
# nr_top.cpp (record fetch + contract decode + validate_hw + nr::norm_mode, float_top's hp2 / io bundles).
# The mode's own cost is the norm_mode instance in the report (nr_top adds the record fetch, decode, validate_hw
# and the m_axi / s_axilite adapters that float_top already has).
#   NR_PERIOD=4.0      clock period in ns (default 4.0 = 250 MHz)
#   NR_FMUL=fulldsp    FP32 multipliers of the normalize stage: maxdsp | fulldsp | meddsp | fabric
#   NR_FADD=fulldsp    FP32 add / sub of the normalize stage: fulldsp | fabric
#   NR_URAM=1          gamma / beta tables and the two NORM row FIFOs in URAM instead of BRAM (suffix _uram)
#   cd fpga/kernels/norm_residual && vitis_hls -f run_nr_hls.tcl   -> build/nrc_<fmul>_<fadd>_p<period>/sol1
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(NR_PERIOD)] ? $::env(NR_PERIOD) : 4.0}]
set fmul [expr {[info exists ::env(NR_FMUL)] ? $::env(NR_FMUL) : "fulldsp"}]
set fadd [expr {[info exists ::env(NR_FADD)] ? $::env(NR_FADD) : "fulldsp"}]
set ptag [string map {. ""} $period]
set proj "nrc_${fmul}_${fadd}_p${ptag}"
set cflags "-std=c++14 -I$here -I$here/.. -I$here/../quantize -DNR_FMUL_IMPL=$fmul -DNR_FADD_IMPL=$fadd"
if {[info exists ::env(NR_URAM)]} {
    append cflags " -DNR_URAM"
    append proj "_uram"
}

# Vitis HLS 2022.2 on Windows refuses '/' in a project name, so build inside build/.
file mkdir "$here/build"
cd "$here/build"
open_project -reset $proj
set_top nr_top
add_files "$here/nr_top.cpp" -cflags $cflags
open_solution -reset sol1 -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period $period -name default
config_interface -m_axi_addr64
csynth_design
exit
