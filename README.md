# robotfpga

The tiny robot-arm policy (5 convs + 3 FCs, int8, 9.67M multiply-adds) as a hardware accelerator on the
AMD Kria **KR260**. The goal is under 5 ms per inference, matching the software side's int8 reference byte for byte.

See [docs/plan.md](docs/plan.md) for status, tomorrow's bring-up steps and the speed-up plan.

## Layout

```
ref/      bit-exact numpy reference, stand-in weights + golden vectors generator, tests
hls/      Vitis HLS kernel (plain C++), AXI top level, testbenches, build script
board/    PYNQ scripts that run on the KR260 (M1 loopback, M4 full network)
docs/     plan and notes
```

## Run it on a laptop (no FPGA, no Xilinx tools)

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python ref/make_standin.py --out artifacts/standin   # weights.bin, manifest.json, vectors/
.venv/bin/python -m pytest ref                                 # reference checks
make -C hls csim                                               # C model vs all 100 golden vectors
```

`artifacts/` is generated and not committed. When the software side ships real exports, point the tools at that
directory instead of `artifacts/standin`.
