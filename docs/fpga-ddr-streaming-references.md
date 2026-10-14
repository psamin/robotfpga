# References: streaming weights from DDR on the KR260

> Written on 2026-10-13 while the target was still int8. The project now runs SmolVLA in full precision
> (BF16 weights, FP32 math); the bandwidth, port, burst and tiling rules apply as written, the int8
> quantization rules (15–18, 22) do not. See [smolvla-fpga.md](smolvla-fpga.md).

This note collects the papers and vendor docs behind a KR260 int8 accelerator that keeps its weights
in DDR4 and streams them into the fabric. It serves two targets: (1) the int8 CNN policy (5×
conv3x3/s2 + 3 FC, power-of-two per-layer scales, bit-exact reference), scaled to roughly 2–100 M
parameters, and (2) running the linear layers of SmolVLA (~450 M parameters) on the FPGA, with
softmax, LayerNorm/RMSNorm, GELU/SiLU and RoPE on the A53. It complements [references.md](references.md)
(policy, quantization and hls4ml/FINN entries) and the estimates in [fpga-capacity.md](fpga-capacity.md).
Cite keys use the same `[Author Year-Short]` form.

**How entries were verified (2026-10-13).** Titles, authors and venues for arXiv entries come from
each arXiv abs page (venue only when the arXiv comment or a DOI record states it). DOIs were resolved
through the Crossref API. Quoted numbers come from the paper text: for most entries I downloaded the
PDF and searched it. Where I only saw the abstract, the entry says so. The AMD docs were read on
docs.amd.com for **2022.2** where that page exists. Anything I could not confirm is in the last
section. Text marked **Inference:** is my own arithmetic or reasoning, not a claim from the source.

**KR260 numbers used below.** The K26 SOM has 4 GB of 64-bit DDR4-2400, so peak = 2400 MT/s × 8 B =
**19.2 GB/s** ([AMD-SPA-HP], [Li2025-KV260LLM]). The PL reaches DDR through S_AXI_HP0–3_FPD, each up
to 128 bits wide ([AMD-PG201]), at up to 333 MHz ([AMD-DS925]). On-chip memory: 144 BRAM36 + 64
URAM × 288 Kb ≈ 0.65 MB + 2.3 MB ≈ **2.9 MB** (URAM size from AMD UG440/UG573; BRAM/URAM counts from
[AMD-KR260] in references.md).

---

## 1. Roofline-based loop tiling for CNN accelerators with off-chip weights

### [Zhang2015-Roofline] Roofline-optimized CNN accelerator (FPGA'15)
- **Citation:** Chen Zhang, Peng Li, Guangyu Sun, Yijin Guan, Bingjun Xiao, Jason Cong. "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks." *FPGA '15*, pp. 161–170, 2015.
- **ID / URL:** DOI 10.1145/2684746.2689060, https://doi.org/10.1145/2684746.2689060 (PDF read: https://www.cl.cam.ac.uk/research/srg/han/ACS-P35/readinglist/fpga2015_chen.pdf)
- **Finding:** Attainable perf = min(computational roof, CTC ratio × BW), where CTC is operations per byte of DRAM traffic. With tile sizes ⟨Tm, Tn, Tr, Tc⟩: buffers B_in = Tn(S·Tr+K−S)(S·Tc+K−S), B_wght = Tm·Tn·K², B_out = Tm·Tr·Tc, subject to B_in+B_wght+B_out ≤ BRAM. Output reuse makes α_out = ⌈M/Tm⌉⌈R/Tr⌉⌈C/Tc⌉ instead of 2× that times ⌈N/Tn⌉. A single cross-layer unroll ⟨Tm,Tn⟩ = ⟨64,7⟩ was within 5% of per-layer optima. Among equal-performance designs they pick the one needing the least bandwidth. Double buffers run "in a ping-pong manner", with separate load and store channels. In their Vivado 2013.4 test, widening a single AXI IP port gave no gain (400 MB/s at 100 MHz), while adding ports scaled almost linearly to about 4.5 GB/s. Result: 61.62 GFLOPS at 100 MHz on a VC707.
- **Apply here as:** run the CTC/roofline sweep over ⟨Tm,Tn,Tr,Tc⟩ per layer offline, then fix one uniform ⟨Tm,Tn⟩ for all conv layers. Keep the output tile resident across the whole N loop (output-stationary) so each output is written once. Use one ping-pong pair for inputs+weights and a separate one for outputs, on separate AXI ports.

### [Ma2017-LoopOpt] Loop operation and dataflow optimization (FPGA'17)
- **Citation:** Yufei Ma, Yu Cao, Sarma Vrudhula, Jae-sun Seo. "Optimizing Loop Operation and Dataflow in FPGA Acceleration of Deep Convolutional Neural Networks." *FPGA '17*, pp. 45–54, 2017.
- **ID / URL:** DOI 10.1145/3020078.3021736, https://doi.org/10.1145/3020078.3021736 (abstract read on the UMN Experts page; full text not read)
- **Finding (abstract):** Treats loop unrolling, tiling and interchange as design variables to set *before* fixing the hardware, modeling memory access and data movement against them. End-to-end VGG-16 on an Arria 10 GX 1150 reached "645.25 GOPS of throughput and 47.97 ms of latency".
- **Apply here as:** choose the unroll dimensions (output-pixel Pox×Poy vs. output-channel Pof) from a memory-access model, not after the RTL exists.

### [Ma2020-PerfModel] Performance model for CNN accelerators (TCAD 2020)
- **Citation:** Yufei Ma, Yu Cao, Sarma Vrudhula, Jae-sun Seo. "Performance Modeling for CNN Inference Accelerators on FPGA." *IEEE TCAD* 39(4):843–856, 2020.
- **ID / URL:** DOI 10.1109/TCAD.2019.2897634, https://doi.org/10.1109/TCAD.2019.2897634 (PDF read: https://par.nsf.gov/servlets/purl/10165009)
- **Finding:** DMA efficiency is eff = ⌊bit_DMA/(Pox·bit_Px)⌋·Pox·bit_Px / bit_DMA. For example, Pox=7, 16-bit pixels on a 512-bit bus gives 448/512 = 0.875. Usable bandwidth is min(BW_DRAM, BW_DMA), where BW_DMA = bit_DMA/8 × f_accel. With dual buffering, a layer's first-tile load and last-tile store cannot overlap compute. When neither weights nor pixels fit, they read weights once and re-fetch pixels ⌈Nof/Tof⌉ times, or the reverse, whichever moves fewer bytes. FC intermediates (<20 KB) stay on chip.
- **Apply here as:** pick a parallel width that divides the bus exactly. For a 128-bit port with int8 data, the lane count should divide 16 (16, 8, …), so eff = 1.0. Per layer, choose "weights once, re-fetch activations" or the reverse by byte count. Add one un-overlapped tile load+store per layer to the latency model.

### [Guo2018-AngelEye] Angel-Eye (TCAD 2018)
- **Citation:** Kaiyuan Guo, Lingzhi Sui, Jiantao Qiu, Jincheng Yu, Junbin Wang, Song Yao, Song Han, Yu Wang, Huazhong Yang. "Angel-Eye: A Complete Design Flow for Mapping CNN Onto Embedded FPGA." *IEEE TCAD* 37(1):35–47, 2018.
- **ID / URL:** DOI 10.1109/TCAD.2017.2705069, https://doi.org/10.1109/TCAD.2017.2705069 (PDF read: https://people.eecs.berkeley.edu/~alanmi/publications/other/tcad18_angeleye.pdf)
- **Finding:** Per-layer radix-point (dynamic fixed-point) 8-bit quantization, chosen greedily layer by layer, has "negligible accuracy loss". It needs only shifters, and adders and multipliers stay unchanged. All parameters and each layer's results live in external memory. A 2-D buffer descriptor (start, width, height, line step) lets software implement ping-pong on the on-chip buffer. Measured I/O bandwidth on the XC7Z020 was "about 500 Mb/s" (as printed). The XC7Z045 estimate assumes double bandwidth from an extra DDR port. On GPU, batch 1 lost about 41% vs. batch 32, and they ran batch 1 for real-time video.
- **Apply here as:** this is the closest prior design to ours: power-of-two per-layer scales as shifts, an embedded Zynq, and weights in DDR. Keep the instruction/descriptor-driven tile scheduler so one bitstream runs every policy size.

### [Zhang2016-Caffeine] Caffeine (ICCAD 2016 / TCAD 2019)
- **Citation:** Chen Zhang, Zhenman Fang, Peipei Zhou, Peichen Pan, Jason Cong. "Caffeine: Towards Uniformed Representation and Acceleration for Deep Convolutional Neural Networks." *ICCAD '16*, 2016. Journal version: Chen Zhang, Guangyu Sun, Zhenman Fang, Peipei Zhou, Peichen Pan, Jason Cong, *IEEE TCAD* 38(11):2072–2085, 2019.
- **ID / URL:** DOI 10.1145/2966986.2967011 (ICCAD); DOI 10.1109/TCAD.2017.2785257 (TCAD), https://doi.org/10.1109/TCAD.2017.2785257
- **Finding (TCAD abstract):** Uses one "uniformed convolutional matrix-multiplication representation" for both "computation-bound convolutional layers and communication-bound FCN layers", sized with a revised roofline model. Reports 1,460 GOPS fixed point on a KU060, and ">100×" speedup on FC layers over prior FPGA accelerators.
- **Apply here as:** one GEMM engine for conv (im2col tiles) and FC/linear layers. Budget FC layers as bandwidth-bound (§5), not compute-bound.

---

## 2. AXI bursts, port width, double buffering in Vitis HLS 2022.2, and measured Zynq US+ bandwidth

### [AMD-UG1399-config] `config_interface` m_axi defaults (Vitis HLS 2022.2)
- **Citation:** AMD. *Vitis HLS User Guide* (UG1399), 2022.2, "config_interface".
- **URL:** https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/config_interface
- **Finding (quoted defaults):** `m_axi_max_read_burst_length` / `..._write_...` "The default is 16". `m_axi_num_read_outstanding` / `..._write_...` "The default value is 16". `m_axi_max_widen_bitwidth` "512 when `-flow_target vitis`, and 0 when the `-flow_target vivado`". `m_axi_latency` 64 (vitis) / 0 (vivado). `m_axi_alignment_byte_size` 64 (vitis) / 1 (vivado). `m_axi_max_bitwidth` "1024 bits". `m_axi_auto_max_ports` default false: "ports are bundled into a single interface".
- **Apply here as:** our KR260 IP uses the **Vivado flow, where automatic port widening is off (0)**. Declare weight ports explicitly as `ap_uint<128>` (or `hls::vector<int8_t,16>`). Set `max_read_burst_length`, `num_read_outstanding` and `latency` explicitly, because the defaults (16 beats, 16 outstanding, latency 0) do not suit DDR streaming. Give every streamed tensor its own `bundle=`.

### [AMD-UG1399-bestpractice] Best practices for M_AXI interfaces
- **Citation:** AMD. UG1399 2022.2, "Best Practices for Designing with M_AXI Interfaces".
- **URL:** https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/Best-Practices-for-Designing-with-M_AXI-Interfaces
- **Finding (quoted):** "Decompose the kernel algorithm by building a pipeline of producer-consumer tasks", i.e. Load-Compute-Store. "All external I/O accesses must be in the Load and Store tasks." "There should be multiple Load or Store tasks if the kernel needs to read or write from different ports in parallel." "Always use local memory to pass data to/from the Compute tasks." "Maximize the port width … setting it to 512 bits (64 bytes)". "Use hls::vector or ap_(u)int<512> … Usage of structs in the interface may result in poor burst performance." "Set the burst length equivalent to the maximum 4k bytes transfer." Concurrent accesses to one port or bank are sequentialized.
- **Apply here as:** a `DATAFLOW` top with `load_w` → `compute` → `store_y` tasks connected by `hls::stream` or PIPO arrays. The compute task has no pointer arguments. The port width is the **HP port's native 128 bits**, not 512, because the HP slave tops out at 128 bits ([AMD-PG201]). A 4 KB burst at 128 bits is **256 beats**.

### [AMD-UG1399-burst] Burst preconditions and burst tuning
- **Citation:** AMD. UG1399 2022.2, "AXI Burst Transfers", "Preconditions and Limitations of Burst Transfer", "Controlling AXI4 Burst Behavior", "set_directive_interface".
- **URLs:** https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/AXI-Burst-Transfers ; https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/Preconditions-and-Limitations-of-Burst-Transfer ; https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior ; https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/set_directive_interface
- **Finding:** Bursts need all-reads or all-writes, "monotonically increasing order of access", consecutive addresses, and a burst length known before the request. Two arrays bundled on one port burst only one at a time per direction. Burst inference is "all or nothing" with no partial bursts. Use unsigned native ints as loop induction variables, avoid conditional accesses, and inline functions that touch M_AXI. "Burst inference is not supported on a loop that has the DATAFLOW pragma", and M_AXI ports cannot be shared inside a dataflow region. Pipelined bursts (request hoisted out of the loop) beat sequential bursts, and bursting typically gives "4-5x". The adapter FIFO is `num_read_outstanding × max_read_burst_length × word_size`. The example uses `latency=100`, `num_read_outstanding=32`, `max_read_burst_length=16`. "The maximum kernel interface bitwidth is 512 bits". (The 2022.2 page's write-FIFO formula repeats `num_read_outstanding`, an apparent typo.)
- **Apply here as:** write each load task as a single flat `for (unsigned i = 0; i < n; i++) s.write(w[base+i]);` with no conditionals. Store weights **pre-tiled in DDR in exact consumption order**, so every tile is one contiguous monotonic read. Size the FIFO with the formula. **Inference:** 16 outstanding × 64 beats × 16 B = 16 KB per port (≈4 BRAM36), so 4 ports cost ≈16 BRAM36. 32 × 256 × 16 B = 128 KB per port would eat ~114 of 144 BRAM36 across 4 ports, so do not max both knobs.

### [AXI4-4KB] AXI4 burst limits
- **Citation:** Arm AMBA AXI protocol specification (IHI 0022). Rules read via ZipCPU, "AXI addressing limits" (2020), which attributes the 4 KB rule to the AXI4 spec.
- **URL:** https://zipcpu.com/blog/2020/06/16/axiaddr-limits.html (Arm spec page https://developer.arm.com/documentation/ihi0022/latest would not render for me)
- **Finding:** "It's illegal in AXI to cross 4kB boundaries". "For AXI4, the maximum burst length is 256 beats." This is consistent with UG1399's "maximum 4k bytes transfer".
- **Apply here as:** align every weight tile's DDR base to 4 KB and pad tiles to whole 4 KB pages, so HLS never has to split a burst.

### [AMD-PG201] Zynq US+ PS slave interfaces
- **Citation:** AMD. *Zynq UltraScale+ MPSoC Processing System LogiCORE IP* (PG201) v3.4, "Slave Interface".
- **URL:** https://docs.amd.com/r/3.4-English/pg201-zynq-ultrascale-plus-processing-system/Slave-Interface
- **Finding (quoted):** four "high performance slave AXI interfaces in full-power domain" (HP0–HP3 FPD), "Non-coherent paths from PL to FPD main switch and DDR". HPC0/1 are "I/O coherent with CCI", with "No L2 cache allocation". "Each interface supports 32, 64, and 128 data widths."
- **Apply here as:** stream weights over **HP0–HP3** (non-coherent, so no CCI snoop on the bulk path). Weights are written once at load time, so the A53 flushes caches once after loading the weight blob. Use HPC only for small activation buffers the A53 touches every step, if the per-call cache flush turns out to be measurable.

### [AMD-DS925] Zynq US+ data sheet: PS-PL clock and DDR rate
- **Citation:** AMD. *Zynq UltraScale+ MPSoC Data Sheet: DC and AC Switching Characteristics* (DS925).
- **URL:** https://docs.amd.com/r/en-US/ds925-zynq-ultrascale-plus/Processor-System-PS-Performance-Characteristics
- **Finding:** F_AXICLK "Maximum AXI interface performance" **333 MHz**. DDR4 1-rank component in the SFVC784 package: up to **2400 Mb/s**.
- **Apply here as:** 4 ports × 128 b × 300 MHz = 19.2 GB/s, exactly the DDR peak. So clock the AXI side at ≥300 MHz (≤333) or the ports, not DDR, become the cap. At 250 MHz the port ceiling is 16 GB/s.

### [AMD-SPA-HP] AMD Embedded Design Tutorial: evaluating HP ports
- **Citation:** AMD/Xilinx. "Evaluating High-Performance Ports", *Embedded Design Tutorials* 2021.1, SPA user guide chapter 6.
- **URL:** https://xilinx.github.io/Embedded-Design-Tutorials/docs/2021.1/build/html/docs/User_Guides/SPA-UG/docs/6-evaluating-high-performance-ports.html
- **Finding:** Theoretical Zynq US+ DDR4 bandwidth is "2400M tranx/sec x 8 bytes/tranx = 19200 MB/s". In the ZCU102 stress setup (traffic generators at 250 MHz), the configured total of 16,000 MB/s is "close to theoretical DDR throughput 19200 (83.3%)". Writes ran lower than reads, and HP1/HP2 lower than HP0/HP3 (per-port values only in figures). The Zynq-7000 four-HP-port test measured 3,254 MB/s, 76% of that device's peak.
- **Apply here as:** plan on roughly 80–85% of 19.2 GB/s, and prefer HP0/HP3 for the heaviest stream. **Inference:** the tutorial gives the HP ranking only qualitatively, so measure it on our board.

### [Li2025-KV260LLM] LLaMA2-7B decode on KV260 at 85% of DDR peak (DATE 2025)
- **Citation:** Jindong Li, Tenglong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng. "Pushing up to the Limit of Memory Bandwidth and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA." *DATE 2025* (per arXiv comment).
- **ID / URL:** arXiv:2502.10659, https://arxiv.org/abs/2502.10659
- **Finding:** Same K26 SOM (4 GB 64-bit DDR4-2400, 19.2 GB/s). "We utilize four AXI ports, with the system frequency set to 300 MHz". "Four 128-bit streams from the four AXI ports are synchronized, concatenated into a 512-bit stream." Measured 4.9 token/s vs. a 5.8 token/s theoretical bound (84.5%) on 4-bit AWQ-style weights (group 128, 16-bit scales). Scales, zero points and weights are interleaved so each 512-bit beat carries one kind of data. Bare metal, because 3.5 GB of weights leave no room for Linux. 6.57 W (Vivado estimate).
- **Apply here as:** **this is the measured K26 number to design to: 4 × 128-bit HP ports at 300 MHz sustain about 16.2 GB/s (0.845 × 19.2) of weight streaming.** It is 2.7× the "~6 GB/s through the PL ports" assumed in [fpga-capacity.md](fpga-capacity.md) §6.

### [Li2025-Hummingbird] Hummingbird: 93–94% DDR utilization on KV260 (ICCAD 2025)
- **Citation:** Jindong Li, Tenglong Li, Ruiqi Chen, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng. "Hummingbird: A Smaller and Faster Large Language Model Accelerator on Embedded FPGA." *ICCAD 2025* (per arXiv comment).
- **ID / URL:** arXiv:2507.03308, https://arxiv.org/abs/2507.03308
- **Finding:** "Four 128-bit AXI ports running at 300MHz are needed to fully expose the bandwidth to the PL-side." It names a "1+1+1+1<4 problem": arbitration among the 4 ports at the memory controller loses bandwidth (84% in prior work). The fix is column-aligned access: base addresses aligned to a DRAM column boundary, "each port accessing a distinct column segment", and per-transaction bytes not exceeding the column size, with the best size at 2^13 or 2^14 bytes. Result: 94% model bandwidth utilization on KV260 and 93% on ZCU104, with LLaMA3-8B GPTQ-4-bit at 4.8 token/s (KV260). Nonlinear ops (RoPE, softmax, LayerNorm, SiLU) run in a PL scalar unit, not on the ARM.
- **Apply here as:** split each weight tile into 4 interleaved 8 KB (2^13) slices, one per HP port, each 8 KB-aligned and contiguous. Target ≥90% of 19.2 GB/s (≈17.3 GB/s) as the stretch goal after the simple version works.

---

## 3. LLM / transformer inference on FPGA with weights streamed from off-chip memory

### [Hong2022-DFX] DFX (MICRO 2022)
- **Citation:** Seongmin Hong, Seungjae Moon, Junsoo Kim, Sungjae Lee, Minsub Kim, Dongsoo Lee, Joo-Young Kim. "DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based Text Generation." *MICRO 2022*, pp. 616–630.
- **ID / URL:** arXiv:2209.10797, DOI 10.1109/MICRO56248.2022.00051, https://arxiv.org/abs/2209.10797
- **Finding:** The generation stage "processes a single token one by one" and is the bottleneck on GPUs. On GPT-2 1.5B each extra output token added 75.45 ms on GPU vs. 0.02 ms per extra input token. The DFX core is "optimized for single token processing" with a tiling scheme for "maximum HBM bandwidth usage". On four U280s: 5.58× faster and 3.99× more energy-efficient than four V100s.
- **Apply here as:** design the linear engine for M=1 (GEMV) at full bandwidth first. Multi-token GEMM (conv tiles, VLM prefix, 50-token action chunks) is the second mode on the same datapath.

### [Zeng2024-FlightLLM] FlightLLM (FPGA 2024)
- **Citation:** Shulin Zeng, Jun Liu, Guohao Dai, et al. (17 authors), Yu Wang. "FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs." *FPGA '24*, pp. 223–234.
- **ID / URL:** arXiv:2401.03868, DOI 10.1145/3626202.3637562, https://arxiv.org/abs/2401.03868
- **Finding:** Decode "repetitively accesses fine-grained data from off-chip memory, leading to underutilized bandwidth (29-43%)". An "always-on-chip decode" scheme keeps activation vectors in on-chip buffers across all layers of one decode step, and fuses the matrix engine with the special-function unit (softmax/LayerNorm). This raised HBM utilization "from about 35.6% to 65.9%". Average 3.5-bit weights and 8-bit activations. On a U280 with LLaMA2-7B at batch 1: 6.0× energy efficiency vs. V100S.
- **Apply here as:** never round-trip activations through DDR between consecutive linear ops when they fit on chip; one SmolVLA token row is ≤ a few KB. **Inference:** with nonlinear ops on the A53, every PL→PS handoff is a small transfer of exactly the kind FlightLLM identifies as a utilization killer. Minimize handoffs per block (§3 hybrid rule below) and measure the per-call overhead before committing.

### [Chen2024-SpatialLLM] FPGA spatial acceleration for LLMs (TRETS 2024)
- **Citation:** Hongzheng Chen, Jiahao Zhang, Yixiao Du, Shaojie Xiang, Zichao Yue, Niansong Zhang, Yaohui Cai, Zhiru Zhang. "Understanding the Potential of FPGA-based Spatial Acceleration for Large Language Model Inference." *ACM TRETS* 18(1), 2024 (FCCM'24 journal track).
- **ID / URL:** arXiv:2312.15159, DOI 10.1145/3656177, https://arxiv.org/abs/2312.15159
- **Finding:** With off-chip weights, a linear operator with M_i MACs and reuse r_i needs B_i = b_W × M_i/r_i × freq. If ΣB_i exceeds device bandwidth, "the inference becomes bandwidth bound" (Eq. 7). Insight II: decode "only requires a small M". Insight III: "Weight quantization is necessary …, while activation quantization only has limited benefit." Insight V: low-bit weights relieve off-chip traffic. Packing 2 int4×int8 products per DSP48E2 (activation in the low 8 bits of the 18-bit input, weights at bits 0–3 and 13–16 of the 27-bit input). W4A8 for BERT, W8A8 for GPT. GPT decode: 1.9× speed and 5.7× energy efficiency vs. A100 on a U280.
- **Apply here as:** size the GEMV MAC count from Eq. 7. On the KR260, b_W = 8, freq = 300 MHz and B = 16.2 GB/s give M/r ≈ **54 int8 weights per cycle**, so a 64-lane GEMV saturates DDR. **Inference:** more MACs only help conv and multi-token layers.

### [Huang2025-EdgeLLM] EdgeLLM (TCAS-I 2025)
- **Citation:** Mingqiang Huang, Ao Shen, Kai Li, Haoxiang Peng, Boyu Li, Yupeng Su, Hao Yu. "EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models." *IEEE TCAS-I* 72(7):3352–3365, 2025.
- **ID / URL:** arXiv:2407.21325, DOI 10.1109/TCSI.2025.3546256, https://arxiv.org/abs/2407.21325
- **Finding:** On a VCU128, FP16×FP16 for attention and FP16×INT4 for FFN. The HBM gives "8192 bits per cycle (32 AXI ports …)", and "we set the system to consume a corresponding 8192 bits of weight parameters per cycle", i.e. compute parallelism matched to bandwidth. Weights are pre-processed in HBM; LayerNorm, RMSNorm, rotary and activations are FPGA operators on DDR. The CPU runs "the dynamic compilation process". The authors list a CPU + element-wise operator as the general but slower fallback for model-specific RoPE. Operators run strictly one after another.
- **Apply here as:** the bandwidth-matched rule in one line: **weight bits consumed per cycle = AXI bits delivered per cycle**, i.e. 4 × 128 = 512 b/cycle = 64 int8 weights/cycle on the KR260.

### [Zheng2025-LoopLynx] LoopLynx
- **Citation:** Jianing Zheng, Gang Chen. "LoopLynx: A Scalable Dataflow Architecture for Efficient LLM Inference." arXiv preprint, 2025.
- **ID / URL:** arXiv:2504.09561, https://arxiv.org/abs/2504.09561
- **Finding (abstract):** A hybrid spatial-temporal design: compute-heavy operators become dataflow kernels that are reused in a temporal schedule, and a multi-FPGA version hides data transfers. On GPT-2, dual-FPGA latency is 2.52× lower than an A100 at 48.1% of its energy.
- **Apply here as:** a background citation. A few reusable kernels (GEMM, plus the transfer engine) time-shared across layers is the right shape for a 1,248-DSP part.

### [Xiang2025-QwenKV260] Qwen2.5-0.5B on KV260, hybrid A53 + PL
- **Citation:** Maoyang Xiang, Ramesh Fernando, Bo Wang. "On-Device Qwen2.5: Efficient LLM Inference with Model Compression and Hardware Acceleration." arXiv preprint, 2025.
- **ID / URL:** arXiv:2504.17376, https://arxiv.org/abs/2504.17376
- **Finding:** This is the closest match to our SmolVLA split: K26 SOM, nonlinear ops on the A53 (VFP), MACs in the PL. Profiled on 4 A53 cores, MACs were 91.6% of a 15.95 ms token. RoPE was 87 µs, RMSNorm 25 µs, SiLU+mul 278 µs, about 2.4% together. AWQ int4 (group size 128) weights, scales and zeros are packed into one block and streamed "using 4 channels, each with 128-bit data". Reported 5.1 tok/s vs. a 2.8 tok/s baseline.
- **Apply here as:** leaving nonlinear ops on the A53 is affordable. They cost ~0.4 ms per token-row for a 0.5 B model at decode. **Inference:** for a T-token prefix they scale roughly with T, so profile them on the A53 for SmolVLA's actual token count before deciding to move any into PL.

### [Dettmers2022-LLMint8] LLM.int8() (NeurIPS 2022)
- **Citation:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
- **ID / URL:** arXiv:2208.07339, https://arxiv.org/abs/2208.07339
- **Finding:** Vector-wise quantization (a separate scale per inner-product row and column) handles most features. Systematic outlier features "emerge in all transformer layers starting at scales of 6.7B parameters" and are kept in a 16-bit side matmul ("more than 99.9% of values are multiplied in 8-bit").
- **Apply here as:** SmolVLA (~0.45 B) is far below the 6.7 B threshold the paper reports. Do **not** build a mixed-precision outlier path in v1. Verify on SmolVLA's actual activations (see unverified list).

### [Xiao2023-SmoothQuant] SmoothQuant (ICML 2023)
- **Citation:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*.
- **ID / URL:** arXiv:2211.10438, https://arxiv.org/abs/2211.10438
- **Finding:** For integer GEMM, "we can only use scaling factors from the outer dimensions (i.e., token dimension T and out channel dimension Co) but not inner dimension (i.e., in channel dimension Ci)", applied after the matmul: Y = diag(Δ_X)·(X̄_int8·W̄_int8)·diag(Δ_W). Activation outliers are migrated offline with Y = (X·diag(s)⁻¹)·(diag(s)·W), s_j = max|X_j|^α / max|W_j|^(1−α). "α = 0.5 is a well-balanced point". The smoothing factor is fused into the preceding LayerNorm/linear parameters. W8A8 gives up to 1.56× speedup and 2× memory reduction.
- **Apply here as:** **the scheme for the transformer offload is W8A8 with per-output-channel weight scales and per-tensor (or per-token) activation scales, applied once to the int32 accumulator at the output.** The GEMM inner loop stays pure int8×int8→int32, exactly like the CNN engine. Fold SmoothQuant's s into the LayerNorm γ (run on the A53) and into W offline, at zero runtime cost.

### [Lin2024-AWQ] AWQ (MLSys 2024)
- **Citation:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024* (Best Paper, per arXiv comment).
- **ID / URL:** arXiv:2306.00978, https://arxiv.org/abs/2306.00978
- **Finding:** "protecting only 1% of salient weights" cuts error. Salient channels are found from activation statistics and protected by an equivalent per-channel scale-up, with no backprop or reconstruction. Low-bit (4-bit) weight-only quantization. [Xiang2025-QwenKV260] states the default group size is 128.
- **Apply here as:** the int4 upgrade path for halving weight traffic. **Inference:** group-wise scales along K (the inner dimension) mean the engine must rescale and accumulate per 128-element group, which breaks the single int32 accumulation. Use AWQ's per-channel *scaling* trick with per-output-channel int4 scales first, and use groups only if accuracy demands it.

### [Frantar2023-GPTQ] GPTQ (ICLR 2023)
- **Citation:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
- **ID / URL:** arXiv:2210.17323, https://arxiv.org/abs/2210.17323
- **Finding (abstract):** One-shot, approximate second-order weight quantization to 3–4 bits "with negligible accuracy degradation", 175 B in about 4 GPU hours. [Li2025-Hummingbird] used GPTQ 4-bit weights on the KV260.
- **Apply here as:** an alternative weight-rounding method that keeps the same storage format (int4 or int8 + per-channel scale), so it changes no hardware. Try it if round-to-nearest W8 loses accuracy.

---

## 4. VLA and robot policies on edge accelerators

No peer-reviewed or arXiv work running a VLA (OpenVLA, SmolVLA, π0, TinyVLA) on an FPGA turned up in
my searches (2026-10-13). The entries below are the closest verified sources.

### [Shukor2025-SmolVLA] SmolVLA architecture facts that set FPGA traffic
- **Citation:** Mustafa Shukor, Dana Aubakirova, Francesco Capuano, et al., Remi Cadene. "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv preprint, 2025 (also in references.md).
- **ID / URL:** arXiv:2506.01844, https://arxiv.org/abs/2506.01844
- **Finding (from the PDF):** The main model "contains 450 million parameters, with approximately 100 million dedicated to the action expert". It uses "only the first 16 layers of the LLM" (N = L/2). The global image only plus pixel shuffle gives "64 visual tokens per frame". Images are 512×512. "CA layers cross-attend the VLM's keys and values". The action expert outputs chunks of n = 50, and "flow matching is fixed to 10 steps". There is an asynchronous inference stack.
- **Apply here as:** per observation, the VLM (~350 M) runs once and the expert (~100 M) runs 10 times, each time on a 50-token chunk. **Inference:** the VLM K/V are computed once and reused by all 10 expert steps (CA reads them), so cache them in DDR or on chip across steps. Weight traffic per observation at int8 ≈ 0.35 GB + 10 × 0.10 GB = **1.35 GB → ≈ 83 ms at 16.2 GB/s** (≈ 42 ms at int4). The expert's re-reads, not the VLM, dominate traffic.

### [Zhou2026-VLAXPU] VLA characterization across accelerators
- **Citation:** Kaijun Zhou, Qiwei Chen, Da Peng, Zhiyang Li, Xijun Li, Jinyu Gu. "Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment." arXiv preprint, 2026.
- **ID / URL:** arXiv:2604.24447, https://arxiv.org/abs/2604.24447
- **Finding:** π0, SmolVLA and GR00T were profiled on RTX 4090, AGX Orin, Jetson Thor, Ascend 310B/310P and Intel B60. "the VLM backbone is compute-bound, while the Action Expert is memory-bound". For π0 the VLM layer runs at ≈840 FLOP/B, while the action expert's operational intensity is 64.5 FLOP/B, below the ridge on all three NVIDIA parts (Orin ridge 208). The action expert has 20–40% SM utilization yet "dominates latency (2× that of the VLM)". π0 end-to-end: 102.3 ms (4090), 246.0 ms (Thor), 920.6 ms (AGX Orin). They pipeline the VLM and action expert (V-AEFusion) and report up to 2.9× (GPU) and 6× (edge NPU) speedups. They found Orin the best cost/energy fit for SmolVLA.
- **Apply here as:** expect the same split on the KR260 (see §5 ridge). Keep the expert's weights hot (on chip or ordered for streaming) and overlap the next observation's VLM pass with the current chunk's expert steps.

### [Xu2025-VLACache] VLA-Cache (NeurIPS 2025)
- **Citation:** Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, Chang Xu. "VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching." *NeurIPS 2025*.
- **ID / URL:** arXiv:2502.02175, https://arxiv.org/abs/2502.02175
- **Finding (abstract):** Reuses the cached KV of visual tokens that change little between adjacent frames, recomputes task-relevant tokens chosen by attention, and adapts the reuse ratio per layer. Up to 1.7× CUDA latency speedup and +15% control frequency, with "negligible" success loss (LIBERO, SIMPLER, real robot).
- **Apply here as:** a later optimization for the VLM pass. It cuts compute (tokens), not weight traffic, so it only helps where the VLM pass is compute-bound (§5).

---

## 5. Batch-1 inference is bandwidth-bound: the rule and how to beat it

### [Williams2009-Roofline] Roofline model
- **Citation:** Samuel Williams, Andrew Waterman, David Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52(4):65–76, 2009.
- **ID / URL:** DOI 10.1145/1498765.1498785, https://doi.org/10.1145/1498765.1498785
- **Finding:** Attainable = min(peak compute, operational intensity × bandwidth). ([Zhang2015-Roofline] applies it to FPGA tiling.)
- **Apply here as (Inference, KR260 numbers):** peak int8 = 1,248 DSP × 2 MAC/DSP ([AMD-WP486]) × 300 MHz ≈ 749 GMAC/s, and BW ≈ 16.2 GB/s ([Li2025-KV260LLM]). The ridge is ≈ **46 MAC per weight byte**, or ≈ 23 without DSP packing. A linear layer applied to T tokens (or T output pixels) has T MAC/byte at int8 and 2T at int4. **T < ~46 is bandwidth-bound: time ≈ weight_bytes / 16.2 GB/s. T > ~46 is compute-bound: time ≈ MACs / (749 G × utilization).** FC layers of the CNN policy and GEMV are T = 1. SmolVLA's 50-token expert chunk sits at the ridge.

### [Yuan2024-LLMRoofline] LLM inference through the roofline lens
- **Citation:** Zhihang Yuan, Yuzhang Shang, Yang Zhou, et al., Kurt Keutzer. "LLM Inference Unveiled: Survey and Roofline Model Insights." arXiv preprint, 2024.
- **ID / URL:** arXiv:2402.16363, https://arxiv.org/abs/2402.16363
- **Finding (abstract):** A roofline-based framework for "why LLMs are memory-bound" and for how compression and other techniques shift memory access vs. compute.
- **Apply here as:** a survey citation for the rule below.

### The batch-1 rule, with measured K26 confirmation
- **Rule:** time_per_pass ≥ weight_bytes_read_per_pass / effective_bandwidth. On the K26 the measured effective bandwidth is 0.845–0.94 × 19.2 GB/s ([Li2025-KV260LLM], [Li2025-Hummingbird]). Hummingbird's own figure fits the rule: one 3,690 MB LLaMA3-8B pass takes ≈202 ms, i.e. ≈18.3 GB/s.
- **Ways to beat it (each verified above):** (a) fewer bytes per weight: int4 halves traffic ([Lin2024-AWQ], [Frantar2023-GPTQ], [Chen2024-SpatialLLM] Insight V). (b) Keep activations and hot weights on chip so DDR carries only weights ([Zeng2024-FlightLLM]). (c) Overlap load and compute with ping-pong so time = max(load, compute), not the sum ([Zhang2015-Roofline], [Ma2020-PerfModel]). (d) Raise reuse T by computing several tokens per weight read, e.g. the 50-action chunk, the VLM prefix, or conv output tiles. (e) Reuse cached K/V and visual tokens across steps and frames ([Shukor2025-SmolVLA], [Xu2025-VLACache]). (f) Raise utilization of the port itself ([Li2025-Hummingbird]).
- **On-chip residency (Inference):** the ~2.9 MB of BRAM+URAM holds ≈2.9 M int8 weights minus the buffers. For the CNN policy, keep all conv weights resident (they are re-read per output tile) and stream only the FC weights, which are read once anyway. For SmolVLA nothing large fits, so residency only helps for the double buffers and the K/V of the current layer.

### [AMD-WP486] Two int8 MACs per DSP48E2
- **Citation:** AMD/Xilinx. "Deep Learning with INT8 Optimization on Xilinx Devices." White paper WP486, v1.0.1, 2017.
- **URL:** https://docs.amd.com/v/u/en-US/wp486-deep-learning-int8 (technical text read in the reprint: https://www.edge-ai-vision.com/2016/11/deep-learning-with-int8-optimization-on-xilinx-devices/)
- **Finding:** One DSP48E2 computes "two concurrent INT8 multiply and accumulate (MACC) operations while sharing the same kernel weights". Two activations are packed into the 27-bit port (one shifted by 18 bits) and multiplied by an 8-bit weight in the 18-bit port. "accumulation of up to 7 product terms only can be guaranteed with no overflow for the lower bits", after which an extra slice absorbs it. Vendor claim: 1.75× int8 ops vs. competing devices.
- **Apply here as:** packing needs a **shared weight with two different activations**: two output pixels in conv, or two tokens in a multi-token linear layer. It does nothing for T = 1 GEMV, which is bandwidth-bound anyway. Our bit-exact reference must model the 7-term overflow correction exactly, or we skip packing in v1.

---

## Design rules we adopt

| # | Rule | Cite key | Parameter value for KR260 |
|---|---|---|---|
| 1 | Budget weight streaming at measured, not peak, bandwidth | [Li2025-KV260LLM], [AMD-SPA-HP] | 16.2 GB/s (0.845 × 19.2); stretch 17.3 GB/s |
| 2 | Use all four non-coherent HP ports, one independent load task each | [AMD-PG201], [Li2025-KV260LLM], [Zhang2015-Roofline] | S_AXI_HP0–3_FPD, 4 × `m_axi` bundles |
| 3 | Port width = HP native width, declared explicitly (Vivado flow does not auto-widen) | [AMD-PG201], [AMD-UG1399-config] | `ap_uint<128>` ports; `max_widen_bitwidth` irrelevant |
| 4 | AXI/kernel clock high enough that ports ≥ DDR | [AMD-DS925], [Li2025-KV260LLM] | 300 MHz (cap 333): 4 × 16 B × 300 M = 19.2 GB/s |
| 5 | Max-length, 4 KB-aligned bursts, never crossing 4 KB | [AMD-UG1399-bestpractice], [AXI4-4KB] | `max_read_burst_length=256` (4 KB @128 b) or 64 if FIFO-bound; tile bases 4 KB-aligned |
| 6 | Outstanding reads to cover DDR latency, FIFO budgeted by formula | [AMD-UG1399-burst] | `num_read_outstanding=16`, `latency` set from measurement (start 64–100); FIFO = N×burst×16 B |
| 7 | Per-port column-aligned slices to avoid "1+1+1+1<4" arbitration loss | [Li2025-Hummingbird] | 8 KB (2^13) slices, one per port, 8 KB-aligned |
| 8 | Pre-tile weights in DDR in exact consumption order, flat monotonic reads, no conditionals | [AMD-UG1399-burst], [Li2025-KV260LLM] | Compiler emits one contiguous blob per layer |
| 9 | DATAFLOW load → compute → store, compute task has no pointers, ping-pong between | [AMD-UG1399-bestpractice], [Zhang2015-Roofline] | `hls::stream`/PIPO; separate load and store ports |
| 10 | Match GEMV MACs to delivered weight bits per cycle | [Chen2024-SpatialLLM], [Huang2025-EdgeLLM] | 512 b/cycle ÷ 8 = 64 int8 MAC lanes for T=1 |
| 11 | Lane count divides the bus word, so DMA efficiency = 1 | [Ma2020-PerfModel] | Lanes ∈ {16, 32, 64}; channel dims padded to 16 |
| 12 | Uniform cross-layer conv tile chosen by roofline sweep | [Zhang2015-Roofline], [Ma2017-LoopOpt] | One ⟨Tm,Tn⟩ for all 5 conv layers (sweep offline) |
| 13 | Conv weights resident on chip, FC weights streamed once | [Zhang2015-Roofline], [Zhang2016-Caffeine] | Conv weights ≤ ~2 MB in URAM; FC time = bytes/16.2 GB/s (20 MB ≈ 1.2 ms) |
| 14 | One GEMM engine for conv (im2col) and linear layers | [Zhang2016-Caffeine] | Same int8×int8→int32 datapath for CNN and SmolVLA |
| 15 | Integer-only GEMM; scales only on outer dims, applied at the int32 output | [Xiao2023-SmoothQuant], [Jacob2018-IntOnly] | W8A8, per-out-channel weight scale, per-tensor act scale; CNN keeps pow2 per-layer |
| 16 | Offline SmoothQuant migration folded into A53-side norm γ | [Xiao2023-SmoothQuant] | α = 0.5 starting point |
| 17 | No outlier mixed-precision path in v1 | [Dettmers2022-LLMint8] | SmolVLA 0.45 B ≪ 6.7 B threshold (verify activations) |
| 18 | int4 weight-only as the 2× traffic upgrade, per-channel before group-wise | [Lin2024-AWQ], [Frantar2023-GPTQ], [Chen2024-SpatialLLM] | W4A8; groups of 128 only if accuracy requires |
| 19 | Nonlinear ops on A53; minimize PS↔PL handoffs per block | [Xiang2025-QwenKV260], [Zeng2024-FlightLLM] | ≤4 PL calls per transformer block; measure handoff cost first |
| 20 | Reuse VLM K/V across all flow-matching steps | [Shukor2025-SmolVLA] | Compute VLM once per obs; 10 expert steps read cached K/V |
| 21 | Compute bound vs. bandwidth bound decided by tokens per weight read | [Williams2009-Roofline], [Zhou2026-VLAXPU] | Ridge ≈ 46 MAC/B (packed); T=1 FC is BW-bound, 50-token expert ≈ ridge |
| 22 | DSP packing only where a weight is shared by two activations | [AMD-WP486], [Chen2024-SpatialLLM] | 2 MAC/DSP for conv and T≥2; ≤7-term accumulate before correction |

---

## Unverified or excluded

- **Ma et al. FPGA'17 full text.** Metadata via Crossref and the abstract via the UMN page only. The
  design details I used come from their verified TCAD 2020 follow-up [Ma2020-PerfModel].
- **Caffeine body text.** Only the TCAD abstract was read. Its FC-to-conv mapping and batching
  details are not used here.
- **Angel-Eye "500 Mb/s".** This is quoted as printed. The context suggests MB/s, but the paper does not say.
- **AXI4 4 KB / 256-beat rules.** Read via ZipCPU's summary of the spec; the Arm spec page itself
  did not render. UG1399's "maximum 4k bytes transfer" agrees.
- **Per-HP-port measured bandwidth on Zynq US+.** The AMD tutorial gives HP0/HP3 > HP1/HP2 only
  qualitatively (values in figures). DDR read latency, which sets `latency` and `num_read_outstanding`,
  is not in any source I verified. Measure both on the board.
- **Hummingbird port type.** It states 4 × 128-bit AXI ports but not whether they are HP or HPC.
- **SmolVLA token counts and the int8 accuracy of its activations.** The 6.7 B outlier threshold is
  from OPT/BLOOM-style LLMs. Whether SmolLM2/SigLIP activations inside SmolVLA quantize cleanly to
  per-tensor int8 is untested. The 83 ms traffic estimate assumes ~350 M VLM parameters
  (450 − 100) actually execute and that the vision encoder counts within the 450 M. The paper does
  not break this down.
- **Peak MAC/s.** 749 GMAC/s assumes all 1,248 DSPs packed at 300 MHz. That is an upper bound,
  not a measured number for an HLS design on this part.
- **TeLLMe v2** (arXiv:2510.15926, ternary LLM, up to 25 tok/s at 5 W on "edge FPGAs"). Verified on
  arXiv, but the abstract does not name the board, so it is left out of the main list.
- **Jetson SmolVLA/π0 latencies** from GitHub PRs and READMEs (e.g. 67 ms SmolVLA on AGX Orin) are
  self-reported and were excluded. SwiftVLA (arXiv:2512.00903) reports "18 times faster" on edge
  devices without naming models in the abstract, so it was excluded.
- **No FPGA VLA paper found.** Absence in my searches is not proof that none exists.
