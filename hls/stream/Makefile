# C simulation of the DDR weight-streaming engine (issue #148). Host compiler only.
#   make -C hls/stream test      packs v3r2 + TinyPolicy-L + linear cases, runs every check
CXX ?= c++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
PY ?= python3
B := ../../build/stream

.PHONY: test clean

build/tb_stream: tb_stream.cpp stream_engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ tb_stream.cpp

test: build/tb_stream
	$(PY) pack.py --model ../../handoff/v3r2 --vectors ../../handoff/v3r2/vectors --out $(B)/v3r2
	./build/tb_stream policy $(B)/v3r2
	$(PY) pack.py --random tinypolicy-l --n-vectors 10 --out $(B)/tinypolicy-l
	./build/tb_stream policy $(B)/tinypolicy-l
	$(PY) pack.py --linear-test --out $(B)/linear
	for d in $(B)/linear/case_*; do ./build/tb_stream linear $$d || exit 1; done

clean:
	rm -rf build $(B)
