# C model of the HLS kernel, built with the host compiler (no Xilinx tools needed).
CXX ?= c++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
ART ?= ../artifacts/standin

.PHONY: csim sanitize clean

csim: build/tb_kernel
	./build/tb_kernel $(ART)

# Same test with address and undefined-behaviour checks (catches out-of-bounds reads, bad shifts).
sanitize: build/tb_kernel_san
	./build/tb_kernel_san $(ART)

build/tb_kernel: tb_kernel.cpp policy_kernel.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ tb_kernel.cpp

build/tb_kernel_san: tb_kernel.cpp policy_kernel.h
	@mkdir -p build
	$(CXX) -std=c++14 -O1 -g -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -fsanitize=address,undefined \
		-fno-sanitize-recover=all -o $@ tb_kernel.cpp

clean:
	rm -rf build
