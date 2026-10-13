"""Compare SGS and uniform teacher runs, in the style of arXiv:2610.12465v1 Fig. 5: success vs.
training, mean over training seeds with 95% confidence intervals.

Reads train_teacher run directories (runmeta.json + metrics.jsonl) and groups them by strategy.
Success is averaged over a trailing window of iterations, weighted by the episodes that finished,
so iterations where nothing finished do not count as failures. Writes compare.png (overall and
one panel per reset family, shared axes) and summary.json (final success per strategy: mean,
95% CI half-width, and per-seed values).

    uv run --extra rl python -m armlab.rl.compare_runs runs/teacher-* --out runs/compare
"""

import argparse
import json
from pathlib import Path

import numpy as np

from armlab.rl.resets import FAMILIES

PANELS = ("overall", *FAMILIES)
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
# Reference palette (dataviz skill): categorical slots 1-2, light surface, text and grid tokens.
COLORS = {"sgs": "#2a78d6", "uniform": "#eb6834"}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def load_run(path: Path) -> dict:
    cfg = json.loads((path / "runmeta.json").read_text())["config"]
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    run = {"name": path.name, "strategy": cfg["strategy"], "seed": cfg["seed"]}
    run["env_steps"] = np.array([r["iter"] for r in rows]) * cfg["envs"] * cfg["steps"]
    for p in PANELS:
        rate, eps = ("success", "episodes") if p == "overall" else (f"success_{p}", f"episodes_{p}")
        run[p] = (np.array([r[rate] * r[eps] for r in rows]), np.array([r[eps] for r in rows]))
    return run


def windowed(successes, episodes, window: int) -> np.ndarray:
    """Trailing-window success rate; NaN until some episode in the window has finished."""
    s, e = (np.convolve(x, np.ones(window))[: len(x)] for x in (successes, episodes))
    return np.where(e > 0, s / np.maximum(e, 1e-9), np.nan)


def mean_ci(values) -> tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    t = T_975.get(len(v) - 1, 1.96)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(len(v)))


def summarize(runs: list[dict], window: int) -> dict:
    out = {}
    for strategy in sorted({r["strategy"] for r in runs}):
        group = [r for r in runs if r["strategy"] == strategy]
        out[strategy] = {"runs": [r["name"] for r in group]}
        for p in PANELS:
            final = [windowed(*r[p], window)[-1] for r in group]
            mean, half = mean_ci([f for f in final if np.isfinite(f)] or [np.nan])
            out[strategy][p] = {
                "mean": mean,
                "ci95": half,
                "seeds": dict(zip([r["seed"] for r in group], final, strict=True)),
            }
    return out


def plot(runs: list[dict], window: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.2 * len(PANELS), 3.2), sharey=True, facecolor=SURFACE)
    for ax, p in zip(axes, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        for strategy in sorted({r["strategy"] for r in runs}):
            group = [r for r in runs if r["strategy"] == strategy]
            n = min(len(r["env_steps"]) for r in group)
            curves = np.array([windowed(*r[p], window)[:n] for r in group])
            x = group[0]["env_steps"][:n] / 1e6
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(curves, axis=0)
                half = np.array(
                    [mean_ci(c[np.isfinite(c)] if np.isfinite(c).any() else [np.nan])[1] for c in curves.T]
                )
            color = COLORS.get(strategy, MUTED)
            ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2,
                solid_capstyle="round",
                label=f"{strategy} (n={len(group)})",
            )
        ax.set_title(p.replace("_", " "), color=TEXT, fontsize=10, loc="left")
        ax.set_ylim(0, 1)
        ax.set_xlabel("env steps (M)", color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=1)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("success rate", color=MUTED, fontsize=9)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=9,
        labelcolor=TEXT,
    )
    fig.suptitle(
        f"Teacher training success, mean ± 95% CI over seeds ({window}-iteration window)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="train_teacher output directories")
    ap.add_argument("--window", type=int, default=20, help="trailing iterations per success estimate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(r) for r in args.runs]
    summary = summarize(runs, args.window)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    plot(runs, args.window, args.out / "compare.png")
    for strategy, s in summary.items():
        cells = "  ".join(f"{p} {s[p]['mean']:.3f}±{s[p]['ci95']:.3f}" for p in PANELS)
        print(f"{strategy:8s} n={len(s['runs'])}  {cells}")
    print(f"-> {args.out / 'compare.png'}, {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
