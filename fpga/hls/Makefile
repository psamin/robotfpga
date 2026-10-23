# Full-precision linear engine for SmolVLA (BF16 weights in DDR, FP32 math), issue #148.
# Host compiler only:   make -C fpga/hls test    builds the C model and runs every case
CXX ?= c++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
PY ?= python3
B := ../../build/stream

.PHONY: test lib clean

build/tb_float: tb_float.cpp float_shim.cpp float_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ tb_float.cpp float_shim.cpp

build/libfloat_linear.so: float_shim.cpp float_linear.h
	@mkdir -p build
	$(CXX) -std=c++14 -O3 -march=native -fopenmp -shared -fPIC -Wno-unknown-pragmas -o $@ float_shim.cpp

lib: build/libfloat_linear.so

build/tb_mul: tb_mul.cpp float_linear.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -ffp-contract=off -o $@ tb_mul.cpp

# Random SmolVLA-shaped cases (incl. output padding to 32 lanes and K chunking past 4096), and the
# bit-exactness of the one-DSP multiplier
test: build/tb_float build/libfloat_linear.so build/tb_mul
	$(PY) pack_float.py --tests $(B)/float
	for d in $(B)/float/case_*; do ./build/tb_float $$d || exit 1; done
	./build/tb_mul

clean:
	rm -rf build $(B)
