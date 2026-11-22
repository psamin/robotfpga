# Vitis HLS 2022.2 synthesis (csynth only; csim is replaced by tb_nr.cpp with g++) of the NORM mode, driven by
# nr_top.cpp (record fetch + contract decode + validate_hw + nr::norm_mode, float_top's hp2 / io bundles).
# The mode's own cost is the norm_mode instance in the report (nr_top adds the record fetch, decode, validate_hw
# and the m_axi / s_axilite adapters that float_top already has).
#   NR_PERIOD=4.0      clock period in ns (default 4.0 = 250 MHz)
#   NR_FMUL=fulldsp    FP32 multipliers of the normalize stage: maxdsp | fulldsp | meddsp | fabric
#   NR_FADD=fulldsp    FP32 add / sub of the normalize stage: fulldsp | fabric
#   NR_SQ=dsp          the 4 exact 25 x 25 squares of the statistics: dsp | fabric
#   NR_KFOLD=1         area measurement only, not contract 1.0: no gamma / beta (folded downstream) (suffix _kfold)
#   NR_URAM=1          the gamma | beta table and the NORM row FIFO in URAM instead of BRAM (suffix _uram)
#   cd fpga/kernels/norm_residual && vitis_hls -f run_nr_hls.tcl   -> build/nr6_<fmul>_<fadd>_<sq>_p<period>/sol1
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(NR_PERIOD)] ? $::env(NR_PERIOD) : 4.0}]
set fmul [expr {[info exists ::env(NR_FMUL)] ? $::env(NR_FMUL) : "fulldsp"}]
set fadd [expr {[info exists ::env(NR_FADD)] ? $::env(NR_FADD) : "fulldsp"}]
set sq [expr {[info exists ::env(NR_SQ)] ? $::env(NR_SQ) : "dsp"}]
set ptag [string map {. ""} $period]
set proj "nr6_${fmul}_${fadd}_${sq}_p${ptag}"
set cflags "-std=c++14 -I$here -I$here/.. -I$here/../quantize -DNR_FMUL_IMPL=$fmul -DNR_FADD_IMPL=$fadd -DNR_SQ_IMPL=$sq"
if {[info exists ::env(NR_KFOLD)]} {
    append cflags " -DNR_KFOLD"
    append proj "_kfold"
}
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
