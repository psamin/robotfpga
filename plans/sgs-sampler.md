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

## Next milestone

Build a small fixed bank of `(seed, instr)` resets spanning all four instructions.
Call `BlockSortEnv.reset(seed=..., instr=...)` for the sampled ID and record terminal
`info["success"]` once, including timeout failures. Compare SGS with uniform sampling
using a fixed policy and separate held-out resets; keep evaluation outcomes out of
training histories. The current expert can validate plumbing but may saturate success.
This seed bank adapts the paper's fixed configurations; it does not implement the
paper's reaching, stable-grasp and near-goal reset families. Before SmolVLA RL,
verify observation/action conventions and establish sorting-task fine-tuning and
closed-loop baseline success. This change adds no inference adapter or RL trainer.
