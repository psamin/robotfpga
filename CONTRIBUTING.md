# Contributing

Two groups work in this repo at the same time: software (sim, training, quantization, eval) and
FPGA (HLS/RTL, board bring-up). These rules keep us from doing the same work twice.

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

- `ref/intref.py` (the bit-exact int8 reference)
- the `manifest.json` / `weights.bin` / test-vector formats
- the TinyPolicy architecture table in [plans/build-spec.md](plans/build-spec.md)

If the hardware and `intref.py` disagree, `intref.py` is right until both sides agree otherwise.

## Don't commit

Generated artifacts (`artifacts/`, build outputs, datasets, checkpoints), secrets, or large binaries.
