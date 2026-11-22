@echo off
rem emit_amax csynth series, one run at a time (clean PATH: no Git dirs)
set PATH=C:\Xilinx\Vitis_HLS\2022.2\bin;C:\Windows\System32;C:\Windows
cd /d %~dp0
set AMX_BASELINE=
set AMX_QL=
set AMX_PERIOD=4.67
set AMX_BASELINE=1
call vitis_hls.bat -f run_hls.tcl > build_base_p467.log 2>&1
set AMX_BASELINE=
call vitis_hls.bat -f run_hls.tcl > build_fused_p467.log 2>&1
set AMX_PERIOD=4.0
set AMX_BASELINE=1
call vitis_hls.bat -f run_hls.tcl > build_base_p40.log 2>&1
set AMX_BASELINE=
call vitis_hls.bat -f run_hls.tcl > build_fused_p40.log 2>&1
set AMX_PERIOD=4.67
set AMX_QL=2
call vitis_hls.bat -f run_hls.tcl > build_fused_p467_ql2.log 2>&1
echo done > run_all.done
