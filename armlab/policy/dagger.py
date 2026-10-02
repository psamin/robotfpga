"""DAgger rounds on the cluster [Ross2011-DAgger]: train -> roll out the learner and let the expert
label every visited state -> aggregate -> retrain, evaluating closed-loop after each round.

    uv run python -m armlab.policy.dagger --expert-data datasets/v2 --out runs/dagger --rounds 4
Each step is a subprocess, so a crash in one round leaves earlier rounds' files intact.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m", *cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=1000, help="DAgger episodes per round (all instructions)")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.3, 0.1, 0.0])
    ap.add_argument("--first-steps", type=int, default=40_000)
    ap.add_argument("--round-steps", type=int, default=30_000)
    ap.add_argument("--eval-episodes", type=int, default=25, help="per instruction")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--start-round", type=int, default=0, help="resume: rounds before this are done")
    ap.add_argument("--train-args", default="", help="extra flags for armlab.policy.train")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = list(args.expert_data) + [str(out / f"dagger_r{r}") for r in range(1, args.start_round + 1)]
    history = json.loads((out / "history.json").read_text()) if (out / "history.json").exists() else []
    for r in range(args.start_round, args.rounds + 1):
        rdir = out / f"round{r}"
        init = [] if r == 0 else ["--init", str(out / f"round{r - 1}" / "final.pt")]
        steps = args.first_steps if r == 0 else args.round_steps
        run(
            [
                "armlab.policy.train",
                "--data",
                *data,
                "--out",
                str(rdir),
                "--steps",
                str(steps),
                "--ckpt-every",
                str(steps),
                "--data-device",
                "cpu",
                *init,
                *args.train_args.split(),
            ]
        )
        final = rdir / f"ckpt_{steps:06d}.pt"
        (rdir / "final.pt").unlink(missing_ok=True)
        (rdir / "final.pt").symlink_to(final.name)
        run(
            [
                "armlab.eval.run",
                "--ckpt",
                str(final),
                "--stage",
                "B",
                "--episodes",
                str(args.eval_episodes),
                "--workers",
                str(args.workers),
                "--out",
                str(rdir / "eval_B.json"),
            ]
        )
        res = json.loads((rdir / "eval_B.json").read_text())["results"]["evals"][0]["all"]
        history.append({"round": r, "frames_sources": len(data), "stage_B": res})
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"ROUND {r}: stage B {res['k']}/{res['n']} = {res['rate']:.3f}", flush=True)
        if r == args.rounds:
            break
        beta = args.betas[min(r, len(args.betas) - 1)]
        ddir = out / f"dagger_r{r + 1}"
        run(
            [
                "armlab.data.collect",
                "--out",
                str(ddir),
                "--policy",
                str(final),
                "--beta",
                str(beta),
                "--episodes-per-instr",
                str(args.episodes // 4),
                "--seed0",
                str(100_000 * (r + 1)),
                "--workers",
                str(args.workers),
                "--max-steps",
                "300",
            ]
        )
        data.append(str(ddir))


if __name__ == "__main__":
    main()
