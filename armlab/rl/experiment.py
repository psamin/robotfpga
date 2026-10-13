"""Fixed-bank SGS/expert plumbing, not an RL trainer or evidence of learning improvement."""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from armlab.rl.sgs import SGSSampler
from armlab.sim.env import INSTRUCTIONS, BlockSortEnv
from armlab.sim.expert import run_episode


@dataclass(frozen=True)
class Configuration:
    config_id: int
    seed: int
    instr: int


@dataclass(frozen=True)
class EpisodeResult:
    config_id: int
    seed: int
    instr: int
    success: bool
    steps: int
    reason: str


def _integer(value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"expected an integer >= {minimum}")
    return int(value)


def configuration_bank(size: int = 8, *, seed: int) -> tuple[Configuration, ...]:
    """Distinct reset seeds below 100,000, balanced instructions; immutable ID order."""
    size, seed = _integer(size, len(INSTRUCTIONS)), _integer(seed, 0)
    if size > 100_000:
        raise ValueError("bank size exceeds the 100,000 distinct training reset seeds")
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))
    seeds = rng.choice(100_000, size=size, replace=False)
    return tuple(Configuration(i, int(s), i % len(INSTRUCTIONS)) for i, s in enumerate(seeds))


def execute_expert(config: Configuration) -> EpisodeResult:
    """Reuse the complete expert loop, retaining the simulator's actual final outcome."""
    env = BlockSortEnv()
    try:
        run_episode(env, seed=config.seed, instr=config.instr, render=False)
        return EpisodeResult(config.config_id, config.seed, config.instr,
                             bool(env.success()), env.t, env.failure_reason())
    finally:
        # reset() renders an initial observation even when run_episode(render=False).
        if env._renderer is not None:
            env._renderer.close()


def run_experiment(
    *,
    strategy: str,
    episodes: int,
    seed: int,
    bank_size: int = 8,
    execute: Callable[[Configuration], EpisodeResult] = execute_expert,
) -> dict:
    """One persistent scheduler per run; injected executors can later run learned policies."""
    if strategy not in ("sgs", "uniform"):
        raise ValueError("strategy must be sgs or uniform")
    episodes, seed = _integer(episodes, 1), _integer(seed, 0)
    bank = configuration_bank(bank_size, seed=seed)
    selection_seed = int(np.random.SeedSequence(seed, spawn_key=(1,)).generate_state(1, dtype=np.uint64)[0])
    sampler = SGSSampler(len(bank), seed=selection_seed) if strategy == "sgs" else None
    rng = np.random.default_rng(selection_seed)
    uniform = np.full(len(bank), 1 / len(bank))
    initial = sampler.probabilities() if sampler is not None else uniform
    results, counts = [], [0] * len(bank)
    for _ in range(episodes):
        config_id = sampler.sample() if sampler is not None else int(rng.integers(len(bank)))
        config = bank[config_id]
        result = execute(config)
        if (result.config_id, result.seed, result.instr) != (config.config_id, config.seed, config.instr):
            raise ValueError("executor returned an outcome for a different configuration")
        if not isinstance(result.success, bool):
            raise ValueError("executor success must be a Boolean terminal outcome")
        if sampler is not None:
            sampler.record(config_id, result.success)
        results.append(asdict(result))
        counts[config_id] += 1
    successes = sum(r["success"] for r in results)
    return {
        "strategy": strategy, "seed": seed, "selection_seed": selection_seed,
        "bank": [asdict(c) for c in bank], "episodes": results,
        "successes": successes, "failures": episodes - successes, "selection_counts": counts,
        "initial_probabilities": initial.tolist(),
        "final_probabilities": (sampler.probabilities() if sampler is not None else uniform).tolist(),
        "success_rates": sampler.success_rates().tolist() if sampler is not None else None,
    }


def main() -> None:
    from armlab.util.runmeta import write_runmeta

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("sgs", "uniform"), default="sgs")
    parser.add_argument("--episodes", type=int, default=8, help="total episodes, not per instruction")
    parser.add_argument("--bank-size", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", help="optional JSON with configuration bank, outcomes and run metadata")
    args = parser.parse_args()
    try:
        result = run_experiment(strategy=args.strategy, episodes=args.episodes,
                                bank_size=args.bank_size, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{args.strategy}: {result['successes']} successes, {result['failures']} failures")
    print(f"Selections by ID: {result['selection_counts']}")
    print(f"Final next-draw probabilities: {np.round(result['final_probabilities'], 4).tolist()}")
    print("Engineering validation of a fixed expert; no learning updates.")
    if args.out:
        write_runmeta(args.out, vars(args), seed=args.seed, results=result)


if __name__ == "__main__":
    main()
