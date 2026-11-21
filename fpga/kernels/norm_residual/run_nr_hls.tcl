# Vitis HLS 2022.2 synthesis (csynth only; csim is replaced by tb_nr.cpp with g++) of the norm_residual kernel.
#   NR_PERIOD=4.0      clock period in ns (default 4.0 = 250 MHz)
#   NR_FMUL=meddsp     FP32 multiplier core: maxdsp | fulldsp | meddsp | fabric (default meddsp)
#   NR_FADD=fabric     FP32 add / sub core: fabric | fulldsp (default fabric)
#   NR_FIFO_URAM=1     the four row FIFOs in URAM instead of BRAM (project suffix _uram)
#   cd fpga/kernels/norm_residual && vitis_hls -f run_nr_hls.tcl   -> build/nr_<fmul>_<fadd>_p<period>/sol1
set here [file dirname [file normalize [info script]]]
set period [expr {[info exists ::env(NR_PERIOD)] ? $::env(NR_PERIOD) : 4.0}]
set fmul [expr {[info exists ::env(NR_FMUL)] ? $::env(NR_FMUL) : "meddsp"}]
set fadd [expr {[info exists ::env(NR_FADD)] ? $::env(NR_FADD) : "fabric"}]
set ptag [string map {. ""} $period]
set proj "nr_${fmul}_${fadd}_p${ptag}"
set cflags "-std=c++14 -DNR_FMUL_IMPL=$fmul -DNR_FADD_IMPL=$fadd"
if {[info exists ::env(NR_FIFO_URAM)]} {
    append cflags " -DNR_FIFO_URAM"
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
