# Instructions for coding agents

Humans on both the software and FPGA sides run agents in this repo at the same time. Follow
[CONTRIBUTING.md](CONTRIBUTING.md), plus these rules.

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
- Shared contract, needs an issue and sign-off from both sides: `ref/intref.py`, the manifest,
  weights and vector formats, the architecture table in `plans/build-spec.md`.

Do not edit files outside your side without an issue that the other side has acknowledged.

## Plans

Plans live in [plans/](plans/). Read the relevant one before starting, and update it in the same PR
when your change alters it.
