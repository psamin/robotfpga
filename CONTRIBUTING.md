# Contributing

Two groups work in this repo at the same time: software (sim, training, quantization, eval) and
FPGA (HLS/RTL, board bring-up). These rules keep us from doing the same work twice.

## Project standard (2026-10-13)

The deployed model is the **fine-tuned SmolVLA** (trained on our SO-101 demos), running in **full
precision: BF16 weights in DDR, FP32 math**. Its linear layers run on the KR260 FPGA; everything
else runs on the board's ARM cores. The plan is [plans/smolvla-fpga.md](plans/smolvla-fpga.md).

- The int8 TinyPolicy path (`ref/intref.py`, `handoff/v3r2/`, the int8 HLS engines) is **retired**:
  keep it as reference, don't build new work on it.
- **Don't train a new policy (sim or real) without the repo owner's approval.** Evaluation,
  conversion and FPGA work on the existing SmolVLA are fine.

## Check QUESTIONS.md after every pull

[QUESTIONS.md](QUESTIONS.md) holds open questions between the software and FPGA sides and a short
"current state" table. Read it after every pull. Before you push, add your questions, answer the
ones addressed to you, and fix any state line your change makes stale.

## Open an issue before you start

1. Check the [open issues](../../issues) to see if someone already has it.
2. Open an issue that says what you will build and which files you expect to touch.
3. Assign it to yourself. An assigned issue means "someone is on this".
4. Reference it in your PR with `Closes #<number>` so it closes when the PR merges.

## Keep PRs small

Aim for about **200 changed lines** per PR. A 2,000-line PR is hard to review and slow to merge.
If a feature is big, split it into a chain of PRs that each work and pass tests on their own.

- One PR does one thing.
- Branch off `main`; don't push directly to `main`.
- Include tests, or say in the PR why there are none.
- Update the matching file in [plans/](plans/) if your change alters the plan.

## Shared contract files

These files are the agreement between software and FPGA. Changing them needs an issue and a review
from someone on **each** side:

- the full-precision engine interface: weight packing layout and call signature
  (`hls/stream/float_linear.h`, `hls/stream/pack_float.py`)
- the SmolVLA checkpoint the project deploys

If the hardware and the C model (`hls/stream/tb_float.cpp` against PyTorch FP32) disagree, the C
model is right until both sides agree otherwise.

## Don't commit

Generated artifacts (`artifacts/`, build outputs, datasets, checkpoints), secrets, or large binaries.
