import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from armlab.rl.compare_runs import PANELS, main, mean_ci, windowed  # noqa: E402
from armlab.rl.resets import FAMILIES  # noqa: E402


def fake_run(root, strategy, seed, final, iters=40):
    """A run whose success climbs linearly to `final`; 8 episodes per family per iteration."""
    d = root / f"teacher-{strategy}-s{seed}"
    d.mkdir()
    (d / "runmeta.json").write_text(
        json.dumps({"config": {"strategy": strategy, "seed": seed, "envs": 4, "steps": 8}})
    )
    rows = []
    for i in range(1, iters + 1):
        rate = final * i / iters
        row = {"iter": i, "success": rate, "episodes": 32}
        for f in FAMILIES:
            row |= {f"success_{f}": rate, f"episodes_{f}": 8}
        rows.append(json.dumps(row))
    (d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    return d


def test_windowed_weights_by_episodes_and_skips_empty_iterations():
    rate = windowed(np.array([0, 3, 0, 1]), np.array([0, 4, 0, 4]), window=2)
    assert np.isnan(rate[0]) and rate[1] == 0.75 and rate[2] == 0.75 and rate[3] == 0.25


def test_mean_ci_uses_t_interval():
    mean, half = mean_ci([0.4, 0.5, 0.6])
    assert mean == pytest.approx(0.5) and half == pytest.approx(4.303 * 0.1 / np.sqrt(3))
    assert mean_ci([0.7]) == (0.7, 0.0)


def test_compare_writes_plot_and_summary(tmp_path):
    runs = [fake_run(tmp_path, "sgs", s, f) for s, f in [(0, 0.9), (1, 0.8)]]
    runs += [fake_run(tmp_path, "uniform", s, f) for s, f in [(0, 0.5), (1, 0.6)]]
    out = tmp_path / "cmp"
    main([*map(str, runs), "--window", "5", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"sgs", "uniform"} and all(p in summary["sgs"] for p in PANELS)
    assert summary["sgs"]["overall"]["mean"] > summary["uniform"]["overall"]["mean"]
    assert len(summary["sgs"]["near_goal"]["seeds"]) == 2
    assert (out / "compare.png").stat().st_size > 10_000
