"""Expert success over N seeds per instruction (build spec Phase 2 acceptance: >= 95%).

uv run python scripts/eval_expert.py --seeds 100 --out runs/expert_eval.json
"""

import argparse
import collections
import time

from armlab.eval.stats import summarize
from armlab.sim.env import TASKS, BlockSortEnv
from armlab.sim.expert import run_episode
from armlab.util.runmeta import write_runmeta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--out", default="runs/expert_eval.json")
    args = ap.parse_args()

    env = BlockSortEnv()
    results, t0 = {}, time.perf_counter()
    for instr in range(4):
        outcomes, reasons, steps = [], collections.Counter(), []
        for i in range(args.seeds):
            ok, n, why, _ = run_episode(env, seed=args.seed0 + 1000 * instr + i, instr=instr)
            outcomes.append(ok)
            reasons[why] += 1
            steps.append(n)
        s = summarize(outcomes)
        s.update(failures=dict(reasons), mean_steps=sum(steps) / len(steps))
        results[TASKS[instr]] = s
        lo, hi = s["ci95"]
        print(f"{TASKS[instr]:<34} {s['k']}/{s['n']}  CI95 [{lo:.3f}, {hi:.3f}]  {dict(reasons)}")
    k = sum(r["k"] for r in results.values())
    n = sum(r["n"] for r in results.values())
    total = summarize([True] * k + [False] * (n - k))
    results["all"] = total
    print(f"all: {k}/{n} = {total['rate']:.3f}  ({time.perf_counter() - t0:.0f} s)")
    write_runmeta(args.out, vars(args), seed=args.seed0, results=results)


if __name__ == "__main__":
    main()
