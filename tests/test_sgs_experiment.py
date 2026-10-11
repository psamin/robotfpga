import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from armlab.rl import experiment as exp


def fake_episode(config):
    ok = config.instr % 2 == 0  # Synthetic test fixture only; never used for reported experiments.
    return exp.EpisodeResult(config.config_id, config.seed, config.instr, ok, 17,
                             "success" if ok else "not_in_bin")


def test_fixed_bank_is_deterministic_balanced_and_immutable():
    bank = exp.configuration_bank(9, seed=42)
    assert bank == exp.configuration_bank(9, seed=42)
    assert bank != exp.configuration_bank(9, seed=43)
    assert [c.config_id for c in bank] == list(range(9))
    assert len({c.seed for c in bank}) == 9
    assert {c.instr for c in bank} == {0, 1, 2, 3}
    with pytest.raises(FrozenInstanceError):
        bank[0].seed = 0


@pytest.mark.parametrize("strategy", ["sgs", "uniform"])
def test_mock_run_is_reproducible_and_retains_results(strategy):
    kwargs = dict(strategy=strategy, episodes=12, seed=5, bank_size=4, execute=fake_episode)
    a, b = exp.run_experiment(**kwargs), exp.run_experiment(**kwargs)
    assert a == b
    assert a["successes"] + a["failures"] == 12
    assert sum(a["selection_counts"]) == 12
    for row in a["episodes"]:
        config = a["bank"][row["config_id"]]
        assert row["seed"] == config["seed"] and row["instr"] == config["instr"]
        assert row["success"] == (row["instr"] % 2 == 0)
    if strategy == "uniform":
        assert a["initial_probabilities"] == a["final_probabilities"] == [0.25] * 4
    else:
        assert a["success_rates"] == [float(c["instr"] % 2 == 0) if n else 0.5
                                      for c, n in zip(a["bank"], a["selection_counts"], strict=True)]


def test_sgs_uses_one_sampler_and_records_once_after_each_episode(monkeypatch):
    events = []
    sampler = Mock(wraps=exp.SGSSampler(4, seed=1))
    sampler.sample.side_effect = lambda: events.append("sample") or 2
    sampler.record.side_effect = lambda cid, success: events.append(("record", cid, success))
    factory = Mock(return_value=sampler)
    monkeypatch.setattr(exp, "SGSSampler", factory)

    def execute(config):
        events.append("execute")
        return fake_episode(config)

    exp.run_experiment(strategy="sgs", episodes=3, seed=1, bank_size=4, execute=execute)
    factory.assert_called_once()
    assert events == ["sample", "execute", ("record", 2, True)] * 3


def test_uniform_never_constructs_sgs(monkeypatch):
    factory = Mock(side_effect=AssertionError("uniform must not touch SGS"))
    monkeypatch.setattr(exp, "SGSSampler", factory)
    exp.run_experiment(strategy="uniform", episodes=3, seed=1, execute=fake_episode)
    factory.assert_not_called()


def test_strategies_share_bank_and_seed_and_cli_writes_json(monkeypatch, tmp_path, capsys):
    kwargs = dict(episodes=4, seed=7, bank_size=4, execute=fake_episode)
    sgs, uniform = (exp.run_experiment(strategy=s, **kwargs) for s in ("sgs", "uniform"))
    assert sgs["bank"] == uniform["bank"] and sgs["selection_seed"] == uniform["selection_seed"]
    monkeypatch.setattr(exp, "run_experiment", lambda **kwargs: sgs)
    path = tmp_path / "smoke.json"
    monkeypatch.setattr("sys.argv", ["experiment", "--seed", "7", "--bank-size", "4",
                                     "--episodes", "4", "--out", str(path)])
    exp.main()
    saved = json.loads(path.read_text())
    assert saved["seed"] == 7 and saved["results"] == sgs
    assert "mujoco" in saved["versions"] and "Engineering validation" in capsys.readouterr().out


@pytest.mark.parametrize("ok", [True, False])
def test_executor_reads_simulator_outcome_and_closes_renderer(monkeypatch, ok):
    env = SimpleNamespace(success=lambda: ok, t=23, failure_reason=lambda: "actual", _renderer=Mock())
    monkeypatch.setattr(exp, "BlockSortEnv", lambda: env)
    loop = Mock(return_value=(not ok, 99, "incorrect summary", None))
    monkeypatch.setattr(exp, "run_episode", loop)
    config = exp.Configuration(0, 12, 3)
    assert exp.execute_expert(config) == exp.EpisodeResult(0, 12, 3, ok, 23, "actual")
    loop.assert_called_once_with(env, seed=12, instr=3, render=False)
    env._renderer.close.assert_called_once()


def test_failed_execution_closes_renderer_and_does_not_record(monkeypatch):
    env = SimpleNamespace(_renderer=Mock())
    monkeypatch.setattr(exp, "BlockSortEnv", lambda: env)
    monkeypatch.setattr(exp, "run_episode", Mock(side_effect=RuntimeError("sim error")))
    sampler = Mock(wraps=exp.SGSSampler(8, seed=1))
    monkeypatch.setattr(exp, "SGSSampler", Mock(return_value=sampler))
    with pytest.raises(RuntimeError, match="sim error"):
        exp.run_experiment(strategy="sgs", episodes=1, seed=1)
    sampler.record.assert_not_called()
    env._renderer.close.assert_called_once()


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"bank_size": 3}, {"seed": -1},
                                    {"bank_size": 100001}, {"strategy": "invalid"}])
def test_invalid_run_inputs(kwargs):
    with pytest.raises(ValueError):
        exp.run_experiment(**(dict(strategy="sgs", episodes=1, seed=1, execute=fake_episode) | kwargs))


@pytest.mark.parametrize("strategy", ["sgs", "uniform"])
def test_real_mujoco_smoke_and_reproducibility(strategy):
    kwargs = dict(strategy=strategy, episodes=1, seed=7, bank_size=4)
    a, b = exp.run_experiment(**kwargs), exp.run_experiment(**kwargs)
    assert a == b
    row = a["episodes"][0]
    assert 0 < row["steps"] <= 600
    assert row["success"] == (row["reason"] == "success")
