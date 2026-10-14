# Contributing

Software (policy, robot, cluster jobs) and FPGA (HLS, board) work happen here at the same time.
These rules keep everyone on the same plan and stop duplicated work.

## The project

- **Task:** an SO-101 arm picks up a medicine bottle and places it on a plate.
- **Policy:** the SmolVLA fine-tuned on our real demos. It is the only policy; it is never overwritten.
- **Hardware:** the KR260 FPGA runs SmolVLA's linear layers in full precision (BF16 weights in DDR,
  FP32 math); the board's ARM cores run the rest. Plan: [docs/smolvla-fpga.md](docs/smolvla-fpga.md).
- **Don't train a new policy (sim or real) without the repo owner's approval.** Post-training of the
  fine-tuned SmolVLA, once approved, writes to a new checkpoint and leaves the original untouched.
- The earlier cube-sorting work (TinyPolicy, int8, sim, SGS) is removed. It is preserved at the git tag
  `archive/cube-sorting`; don't build new work on it.

## Workflow

1. Read [QUESTIONS.md](QUESTIONS.md) after every pull. Before you push, add your questions, answer the
   ones addressed to you, and fix any state line your change makes stale.
2. Before starting, check [open issues](../../issues) and PRs. Open an issue that says what you will
   build and which files you will touch, and assign it to yourself.
3. Work on a branch; reference the issue in the PR with `Closes #<number>`.
4. Keep PRs near **200 changed lines**. One PR does one thing. Include tests or say why there are none.
5. Update the matching doc in [docs/](docs/) in the same PR when your change alters it.

## Ownership

| Area | Path | Owner |
|---|---|---|
| Policy, evaluation | `policy/` | software |
| Robot setup and toolkit | `robot/` | software |
| Cluster jobs | `cluster/` | software |
| FPGA engine and board runtime | `fpga/` | FPGA |
| Engine interface (`fpga/hls/float_linear.h`, `fpga/hls/pack_float.py`) and the deployed SmolVLA checkpoint | shared contract | both sides sign off |

If the hardware and the C model (`fpga/hls/tb_float.cpp` against PyTorch FP32) disagree, the C model
is right until both sides agree otherwise.

## Don't commit

Model checkpoints, datasets, build outputs, logs, secrets, or large binaries. Checkpoints live on
the cluster or Hugging Face; link them instead.
