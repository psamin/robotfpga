# Full-precision linear engine for SmolVLA (BF16 weights in DDR, FP32 math), issue #148.
# Host compiler only:   make -C hls/stream test    builds the C model and runs every case
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

# Random SmolVLA-shaped cases (incl. output padding to 32 lanes and K chunking past 4096)
test: build/tb_float build/libfloat_linear.so
	$(PY) pack_float.py --tests $(B)/float
	for d in $(B)/float/case_*; do ./build/tb_float $$d || exit 1; done

clean:
	rm -rf build $(B)
