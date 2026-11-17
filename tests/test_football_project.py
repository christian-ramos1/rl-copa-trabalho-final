"""Project evidence and phase lifecycle; no football engine required here."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from core.config_loader import load_config
from scripts.football_evaluate import summarize, wilson
from scripts import football_evaluate
from scripts.football_plot import load_metrics
from scripts import football_train


def test_summary_handles_both_sides_draws_and_forfeits():
    results = [
        dict(left="ours", right="other", goals_left=2, goals_right=0, forfeit=""),
        dict(left="other", right="ours", goals_left=1, goals_right=1, forfeit=""),
        dict(left="other", right="ours", goals_left=3, goals_right=0, forfeit="ours"),
    ]
    summary = summarize(results, "ours")
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 1)
    assert (summary["goals_for"], summary["goals_against"]) == (3, 4)
    assert summary["score_rate"] == 0.5
    assert summary["own_forfeits"] == 1
    assert summary["win_ci95_low"] < 1 / 3 < summary["win_ci95_high"]


def test_wilson_zero_and_all_wins_keep_uncertainty():
    assert wilson(0, 20)[1] > 0
    assert wilson(20, 20)[0] < 1
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_ablation_configs_change_only_reward(repo_root):
    a, _ = load_config("configs/ppo_football_sparse.yml")
    b, _ = load_config("configs/ppo_football_checkpoints.yml")
    b.env_kwargs["rewards"] = a.env_kwargs["rewards"]
    assert asdict(a) == asdict(b), "reward ablation acquired an unintended second change"


def test_plot_discards_abandoned_metric_tail_on_resume(tmp_path):
    rows = [dict(global_step=step, value=value) for step, value in
            [(64, "old"), (128, "old"), (192, "abandoned"), (128, "new"), (192, "new")]]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert load_metrics(tmp_path) == [rows[0], rows[3], rows[4]]


def test_evaluation_cli_preserves_schedule_sides_and_raw_results(tmp_path, monkeypatch):
    from arena import match

    agent = tmp_path / "team"
    agent.mkdir()
    (agent / "manifest.yml").write_text("team: ours\ncontrolled_players: 1\n")
    calls = []

    def play(left, right, *, seed):
        calls.append((left, right, seed))
        return match.MatchResult(left="ours" if left == str(agent) else left,
                                 right="ours" if right == str(agent) else right,
                                 seed=seed, goals_left=1, goals_right=0, steps=3000)

    monkeypatch.setattr(match, "play_specs", play)
    out = tmp_path / "evaluation"
    football_evaluate.main([str(agent), "--matches", "2", "--out", str(out)])
    assert calls == [(str(agent), "random", 100000), ("random", str(agent), 100001),
                     (str(agent), "builtin", 100000), ("builtin", str(agent), 100001)]
    results = [json.loads(line) for line in (out / "matches.jsonl").read_text().splitlines()]
    assert len(results) == 4 and all(r["steps"] == 3000 for r in results)
    assert json.loads((out / "protocol.json").read_text())["seeds"] == [100000, 100001]
    with pytest.raises(FileExistsError):
        football_evaluate.main([str(agent), "--matches", "2", "--out", str(out)])


def test_policy_torchscript_preserves_deterministic_actions():
    import gymnasium as gym
    from networks.discrete_actor_critic import DiscreteActorCritic
    from scripts.export_submission import _GreedyPolicy

    model = DiscreteActorCritic(SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1, 1, (115,)),
        single_action_space=gym.spaces.Discrete(19))).eval()
    observations = torch.randn(32, 115)
    with torch.no_grad():
        traced = torch.jit.trace(_GreedyPolicy(model), observations[:1])
        assert torch.equal(traced(observations), model.act(observations, deterministic=True))


@pytest.mark.slow
def test_project_interruption_resume_and_weight_transfer(tmp_path, monkeypatch):
    """Real PPO updates and checkpoints, followed by a fresh phase in a new run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(football_train, "get_config_path", lambda name: str(tmp_path / f"{name}.yml"))
    monkeypatch.setattr(football_train.ProjectPPO, "evaluate", lambda *a, **k: {})
    config = dict(algorithm="ppo", env_id="CartPole-v1", seed=1, cuda=False,
                  total_timesteps=128, num_envs=2, save_model=True, track=False,
                  checkpoint_every=64, network="DiscreteActorCritic",
                  network_kwargs=dict(hidden_layers_size=16),
                  ppo=dict(num_steps=32, num_minibatches=4, update_epochs=1))
    (tmp_path / "phase1.yml").write_text(yaml.safe_dump(config))
    (tmp_path / "phase2.yml").write_text(yaml.safe_dump(config))

    original_post_train = football_train.ProjectPPO._post_training_eval

    def interrupt_before_final_save(self):
        raise RuntimeError("simulated interruption")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(football_train.ProjectPPO, "_post_training_eval", interrupt_before_final_save)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            football_train.main(["--config", "phase1"])
        parent = next((tmp_path / "runs" / "phase1").iterdir())
        assert not (parent / "completed.json").exists()
        assert list(parent.glob("checkpoint_gs*.pt"))
        monkeypatch.setattr(football_train.ProjectPPO, "_post_training_eval", original_post_train)
        football_train.main(["--resume", str(parent), "--override", "total_timesteps=256"])
        assert json.loads((parent / "completed.json").read_text())["global_step"] == 256
        assert not list(parent.glob("checkpoint_gs*.pt"))
        sessions = sorted(parent.glob("session_*.json"))
        assert json.loads(sessions[-1].read_text())["resolved_config"]["total_timesteps"] == 256
        football_train.main(["--config", "phase2", "--init-from", str(parent)])
        child = next((tmp_path / "runs" / "phase2").iterdir())
        metadata = json.loads(next(child.glob("session_*.json")).read_text())
        assert metadata["parent"]["model_sha256"] == football_train.sha256(parent / "model.pt")
        assert metadata["initial_step"] == 0
        assert metadata["parent"]["run_dir"] == str(parent)
        assert json.loads((child / "completed.json").read_text())["global_step"] == 128
        for run in (parent, child):
            rows = load_metrics(run)
            assert rows[-1]["global_step"] > 0
            assert all(r["session_sps"] > 0 for r in rows)
    finally:
        torch.set_num_threads(previous_threads)
