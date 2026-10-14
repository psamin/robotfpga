# Plans

| File | What it is |
|---|---|
| [smolvla-fpga.md](smolvla-fpga.md) | **Current plan:** fine-tuned SmolVLA on the KR260 in full precision (BF16 weights, FP32 math) |
| [fpga-ddr-streaming-references.md](fpga-ddr-streaming-references.md) | Verified papers and design rules for streaming weights from DDR |
| [build-spec.md](build-spec.md) | The original project spec: task, interfaces, int8 math, phases |
| [software-kickoff.md](software-kickoff.md) | Software-side plan: decisions, spec gaps, order of work |
| [fpga.md](fpga.md) | FPGA-side plan: decisions, accelerator interface, milestones, how `hls-v0.1` lands |
| [references.md](references.md) | Verified literature notes; cite keys used across these docs |
| [fpga-team-brief.md](fpga-team-brief.md) | The original brief given to the FPGA team (copied verbatim) |
| [fpga-capacity.md](fpga-capacity.md) | Largest int8 policy the KR260 engine holds; TinyPolicy-L layer table |
| [sgs-post-training.md](sgs-post-training.md) | SGS teacher → DAgger post-training plan for SmolVLA |
| [sgs-sampler.md](sgs-sampler.md) | Success-guided sampling of reset states |
| [tiny-policy-log.md](tiny-policy-log.md) | Engineering log for the blog post: what was built, measured, and why |
| [sgs-post-training.md](sgs-post-training.md) | SmolVLA post-training: PPO + SGS teacher, DAgger into SmolVLA, SO-101 transfer |

New plans go in this folder as Markdown. Update a plan in the same PR that changes what it describes.
