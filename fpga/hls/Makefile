# Full-precision linear engine for SmolVLA (BF16 weights in DDR, FP32 math), issue #148.
# Host compiler only:   make -C fpga/hls test    builds the C model and runs every case
#                       make -C fpga/hls test-int16   the int16 variant (-DFSTREAM_ARITH=16, int16_linear.h)
CXX ?= c++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
PY ?= python3
B := ../../build/stream

.PHONY: test lib clean test-int16 lib-int16 test-int16-top test-gemm lib-seq

build/tb_float: tb_float.cpp float_shim.cpp float_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ tb_float.cpp float_shim.cpp

build/libfloat_linear.so: float_shim.cpp float_linear.h
	@mkdir -p build
	$(CXX) -std=c++14 -O3 -march=native -fopenmp -shared -fPIC -Wno-unknown-pragmas -o $@ float_shim.cpp

lib: build/libfloat_linear.so

# int16 engine: int16 W x int16 X -> int48, FP32 out. No FMA contraction: the C model's dequantization must be
# the two separate FP32 multiplies of the hardware. SHIM_STATIC= on hosts without static runtime libraries.
I16FLAGS := -DFSTREAM_ARITH=16 -ffp-contract=off -Wno-unused-function
SHIM_STATIC ?= -static
build/tb_float_int16: tb_float.cpp float_shim.cpp tb_int16_units.cpp float_linear.h int16_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) -fopenmp -o $@ tb_float.cpp float_shim.cpp tb_int16_units.cpp

build/libfloat_linear_int16.so: float_shim.cpp float_linear.h int16_linear.h
	@mkdir -p build
	$(CXX) -std=c++14 -O3 $(I16FLAGS) -fopenmp -shared -fPIC $(SHIM_STATIC) -Wno-unknown-pragmas -o $@ float_shim.cpp

# The same core with every non-blocking FIFO operation failing 30 % of the time (late weights / X, full emit FIFO):
# the stall paths of compute16 must give the same bits. NBX=1 variant: the engine without weight re-use.
build/tb_float_int16_chaos: tb_float.cpp float_shim.cpp tb_int16_units.cpp float_linear.h int16_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) -DFSTREAM_CHAOS=30 -fopenmp -o $@ tb_float.cpp float_shim.cpp tb_int16_units.cpp

build/tb_float_int16_nbx1: tb_float.cpp float_shim.cpp tb_int16_units.cpp float_linear.h int16_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) -DFSTREAM_NBX=1 -fopenmp -o $@ tb_float.cpp float_shim.cpp tb_int16_units.cpp

lib-int16: build/libfloat_linear_int16.so

# The int16 HLS top itself, with the ap_int / hls::stream types it is synthesized from (what C simulation would
# run; the Vitis csim flow is broken on the Windows install), compiled by the host g++ against the Vitis HLS
# headers, must give the host-typed model's bits. Its X buffer is a 1 MB local: STACK_FLAG (MinGW) or ulimit -s.
HLS_INC ?= $(XILINX_HLS)/include
STACK_FLAG ?= -Wl,--stack,268435456
build/tb_int16_top: tb_int16_top.cpp float_top.cpp float_linear.h int16_linear.h
	@mkdir -p build
	$(CXX) -std=c++14 -O2 -D__VITIS_HLS__ $(I16FLAGS) -Wno-unknown-pragmas -I$(HLS_INC) $(STACK_FLAG) -o $@ 		tb_int16_top.cpp float_top.cpp

test-int16-top: build/tb_int16_top build/tb_float_int16
	for d in $(B)/int16/case_*; do [ -f $$d/xmem.bin ] || continue; 		TB_DUMP_Y=build/y_host.bin ./build/tb_float_int16 $$d > /dev/null && ./build/tb_int16_top $$d build/y_host.bin || exit 1; done

# The contract GEMM unit (gemm_contract.h) against ctr::ref on random commands (items, row groups, bias, D, ACC),
# with the stall paths (FSTREAM_CHAOS=30) and without weight re-use (NBX=1). Two DDR bases.
GEMM_DEPS := tb_gemm.cpp gemm_contract.h int16_linear.h float_linear.h ../kernels/contract.h
build/tb_gemm: $(GEMM_DEPS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) $(SHIM_STATIC) -o $@ tb_gemm.cpp
build/tb_gemm_chaos: $(GEMM_DEPS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) $(SHIM_STATIC) -DFSTREAM_CHAOS=30 -o $@ tb_gemm.cpp
build/tb_gemm_nbx1: $(GEMM_DEPS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(I16FLAGS) $(SHIM_STATIC) -DFSTREAM_NBX=1 -o $@ tb_gemm.cpp
test-gemm: build/tb_gemm build/tb_gemm_chaos build/tb_gemm_nbx1
	./build/tb_gemm 300 11 800000000
	./build/tb_gemm 300 12 40000000
	./build/tb_gemm_chaos 100 13 800000000
	./build/tb_gemm_nbx1 100 14 800000000

# The integrated float_top (float_seq.h: sequencer + GEMM unit + the kernels' mode functions) as a host DLL for
# e2e_seq.py (the -12392 chunk list in lockstep with the golden executor).
build/seq_shim.dll: seq_shim.cpp float_seq.h $(GEMM_DEPS)
	@mkdir -p build
	$(CXX) -std=c++14 -O2 $(I16FLAGS) -DGS_NSB=10 -Wno-unknown-pragmas -Wno-unused-label -shared $(SHIM_STATIC) 		-I$(HLS_INC) -I../kernels -I../kernels/quantize -o $@ seq_shim.cpp
lib-seq: build/seq_shim.dll

build/tb_mul: tb_mul.cpp float_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -ffp-contract=off -o $@ tb_mul.cpp

# Random SmolVLA-shaped cases (incl. output padding to 32 lanes and K chunking past 4096), and the
# bit-exactness of the one-DSP multiplier
test: build/tb_float build/libfloat_linear.so build/tb_mul
	$(PY) pack_float.py --tests $(B)/float
	for d in $(B)/float/case_*; do ./build/tb_float $$d || exit 1; done
	./build/tb_mul

# Random int16 cases (quantized operands and scales, SmolVLA shapes, control edges), the arithmetic helpers, and
# the quantizers against build/brief/precision_emulation.py (needs torch: TORCH_PY=path/to/python with torch)
test-int16: build/tb_float_int16 build/tb_float_int16_chaos build/tb_float_int16_nbx1 build/libfloat_linear_int16.so
	$(PY) pack_float.py --tests-int16 $(B)/int16
	for d in $(B)/int16/case_*; do ./build/tb_float_int16 $$d || exit 1; done
	for d in $(B)/int16/case_*; do ./build/tb_float_int16_chaos $$d || exit 1; done
	for d in $(B)/int16/case_*; do ./build/tb_float_int16_nbx1 $$d || exit 1; done
	./build/tb_float_int16 --units
	$(PY) pack_float.py --check-int16-lib
	$(if $(TORCH_PY),$(TORCH_PY) test_quant_int16.py)

clean:
	rm -rf build $(B)
