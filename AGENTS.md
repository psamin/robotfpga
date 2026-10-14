# Instructions for coding agents

Humans on both the software and FPGA sides run agents in this repo at the same time. Follow
[CONTRIBUTING.md](CONTRIBUTING.md), plus these rules.

## Project standard (2026-10-13)

The deployed model is the **fine-tuned SmolVLA** (trained on our SO-101 demos), running in **full
precision: BF16 weights in DDR, FP32 math**. Its linear layers run on the KR260 FPGA; everything
else runs on the board's ARM cores. The plan is [plans/smolvla-fpga.md](plans/smolvla-fpga.md).

- The int8 TinyPolicy path (`ref/intref.py`, `handoff/v3r2/`, the int8 HLS engines) is **retired**:
  keep it as reference, don't build new work on it.
- **Don't train a new policy (sim or real) without the repo owner's approval.** Evaluation,
  conversion and FPGA work on the existing SmolVLA are fine.

## Check QUESTIONS.md on every pull and push

After every `git pull`, read [QUESTIONS.md](QUESTIONS.md) and tell your human about anything open
for them. Before every push, add questions you raised, answers you found, and state changes the
other side needs. Mark unknowns as unknown; never guess an answer.

## Claim work through an issue, close it with the PR

1. Before writing code, list open issues and open PRs (`gh issue list`, `gh pr list`). If one already
   covers the work, stop and tell your human instead of starting.
2. Open an issue describing the change and the files you will touch. Assign it to your human.
3. Work on a branch. Open a PR whose body contains `Closes #<issue>`.
4. One issue per PR. If the work grows, open a new issue for the extra part.

## Size

Keep each PR near 200 changed lines. Split bigger work into a sequence of issues and PRs.

## Stay in your lane

- Software: `armlab/` (Python package), `plans/` for software plans.
- FPGA: `hls/`, `board/`, RTL sources.
- Shared contract, needs an issue and sign-off from both sides: the full-precision engine interface
  (weight packing and call signature in `hls/stream/float_linear.h` and `hls/stream/pack_float.py`)
  and the SmolVLA checkpoint the project deploys.

Do not edit files outside your side without an issue that the other side has acknowledged.

## Plans

Plans live in [plans/](plans/). Read the relevant one before starting, and update it in the same PR
when your change alters it.
