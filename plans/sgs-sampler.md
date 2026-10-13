# Independent Success Guided Sampling

Source: *A Balanced Data Diet*, attached arXiv:2610.12465v1, §3.2 equations (1)-(2),
Appendix B.2 implementation details (PDF pp. 4-5, 16-17).

`armlab/rl/sgs.py` is separate from simulation, policy inference and training. NumPy
is already a project dependency. Configurations are fixed integer IDs `0..N-1`.
Use `record(id, success)` once per completed episode, `success_rates()` and
`probabilities()` for inspection, and `sample()` / `sample(size)` for IID draws.

## Paper specification

- Keep the last H binary episode outcomes per configuration; p is their mean.
- Beta mode-concentration parameters: alpha = 1 + k*t, beta = 1 + k*(1-t).
- Equation (1): w = (p + eps)^(k*t) * (1 - p + eps)^(k*(1-t)).
- B.2: floor w to eps, then logit = log(max(w, eps) + eps).
- Equation (2) and B.2: probability = softmax(logit / max(T, 1)).
- Manipulation defaults: H=100, t=0.5, k=1, eps=1e-4, T=2. The paper's
  N=32,768 is an experimental reset-buffer size, not a required sampler size.
- Exploration comes from eps and the weight floor; no uniform-mixture term.

## Local choices and limits

The paper does not specify initialization of empty histories or padding before H
outcomes exist. We use a configurable fallback rate of 0.5 for an empty history,
then only recorded outcomes (no pseudo-counts). All-unseen configurations therefore
sample uniformly. A circular Boolean buffer evicts in record/completion order;
callers must coordinate concurrent episodes. Stable log-space arithmetic implements
the equations without a normalized Beta PDF. Require an explicit nonnegative integer
seed, positive T (values below one clamp), and 0 < eps <= 1. Numerical settings that
lose positive probabilities in float64 raise ValueError rather than return zeros.

## Fixed-bank MuJoCo adapter (issue #113)

`armlab.rl.experiment` implements the first integration milestone. Its frozen bank
uses integer IDs in tuple order, distinct reset seeds below 100,000, and instructions
cycled across all four tasks (sizes >= 4). The default is eight configurations.
NumPy SeedSequence streams `(0,)` and `(1,)` separate bank generation and episode
selection; both strategies receive the same bank and derived selection seed.
Selection sequences differ because uniform uses integer draws and SGS uses weighted
choice. Each run keeps one sampler and records exactly one real outcome per episode.

The executor reuses `sim.expert.run_episode(render=False)` and reads final
`env.success()`, `env.t`, and `env.failure_reason()`. The existing loop can continue
past a transient success while the expert retreats. Its initial reset still renders
an observation, but no video frames are collected; the renderer closes even on errors.
The runner accepts a replacement episode executor for later learned policies.

Run a small paired engineering check (episode count is total, not per instruction):

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -m armlab.rl.experiment --strategy sgs --episodes 8 --bank-size 8 --seed 7
PYTHONDONTWRITEBYTECODE=1 uv run python -m armlab.rl.experiment --strategy uniform --episodes 8 --bank-size 8 --seed 7
```

Add `--out runs/sgs-smoke.json` for outcomes, bank, selection counts, initial/final
probabilities, RNG seed, git hash and dependency versions. Generated outputs stay
uncommitted. Reproducibility requires the same simulator/runtime, not just a seed.
Sampling counts describe realized draws; final probabilities describe the next draw.

This is a small fixed-bank adapter, not a reproduction of the paper's 32,768 reset
configurations or a claim that each seed is a large-scale SGS bucket. It preserves
episode-level resets but adds none of the paper's grasp/near-goal reset strategies.
The fixed expert does not learn. High success leaves little outcome diversity;
unseen-rate initialization can temporarily bias SGS even when every episode succeeds.
Neither biased scheduling nor a preference for 50% success establishes learning gain.

## Next research milestone

Compare SGS with uniform sampling while actually updating a learnable policy, using
separate held-out resets and multiple training seeds. Keep evaluation outcomes out
of training histories. The current expert validates plumbing but saturates success
in the short smoke experiment, so it cannot establish learning efficiency. Before SmolVLA RL,
verify observation/action conventions and establish sorting-task fine-tuning and
closed-loop baseline success. This change adds no inference adapter or RL trainer.
