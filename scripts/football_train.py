#!/usr/bin/env python3
"""Project PPO entry point: local metrics, provenance and explicit phase transfer.

Run from the checkout root. See docs/projeto/EXECUCAO.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from dotenv import load_dotenv

from algorithms.ppo import PPO
from core.config_loader import apply_overrides, get_config_path, load_config


def command_output(*command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProjectPPO(PPO):
    """Keep project instrumentation separate from the teaching algorithms."""

    def _log_metrics(self, pbar, **metrics):
        super()._log_metrics(pbar, **metrics)
        record = {
            "global_step": self.global_step,
            "session_seconds": time.monotonic() - self.session_start,
            "session_sps": (self.global_step - self.session_initial_step)
            / max(time.monotonic() - self.session_start, 1e-9),
            "episodes": self.global_ep_counter,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "return_last100": (sum(self.recent_ep_returns) / len(self.recent_ep_returns))
            if self.recent_ep_returns else None,
            "length_last100": (sum(self.recent_ep_lengths) / len(self.recent_ep_lengths))
            if self.recent_ep_lengths else None,
            "policy_loss": float(metrics["pg_loss"].detach()),
            "value_loss": float(metrics["v_loss"].detach()),
            "entropy": float(metrics["entropy_loss"].detach()),
            "approx_kl": float(metrics["approx_kl"].detach()),
            "clipfrac": sum(metrics["clipfracs"]) / len(metrics["clipfracs"]),
            "grad_norm": float(metrics["grad_norm"]),
        }
        # JSON refuses NaN/Inf: an invalid learning signal must fail visibly.
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="config name under configs/")
    source.add_argument("--resume", type=Path, help="interrupted run with checkpoint_gs*.pt")
    parser.add_argument("--init-from", type=Path, help="completed parent run; weights only, new optimizer")
    parser.add_argument("--override", nargs="*", default=[])
    cli = parser.parse_args(argv)
    if cli.resume and cli.init_from:
        parser.error("--resume restores a run; --init-from starts a new phase. Choose one.")
    if cli.resume:
        allowed = {"total_timesteps", "cuda", "track"}
        if any(override.split("=", 1)[0] not in allowed for override in cli.override):
            parser.error("resume only accepts total_timesteps/cuda/track overrides; use a new phase for modeling changes")
    load_dotenv()
    config_path = cli.resume / "config.yml" if cli.resume else Path(get_config_path(cli.config))
    args, algorithm = load_config(str(config_path))
    apply_overrides(args, cli.override)
    if algorithm != "ppo":
        parser.error("this project runner supports discrete PPO only")
    if args.num_envs < 1 or args.ppo.num_steps < 1 or args.ppo.num_minibatches < 1:
        parser.error("num_envs, num_steps and num_minibatches must be positive")
    batch_size = args.num_envs * args.ppo.num_steps
    if args.total_timesteps < batch_size or batch_size // args.ppo.num_minibatches < 2:
        parser.error("need at least one rollout and at least two samples per minibatch")
    if batch_size % args.ppo.num_minibatches:
        parser.error("batch size must be divisible by num_minibatches")
    if not args.save_model:
        parser.error("save_model must be true: the project needs persistent run artifacts")
    args.exp_name = cli.resume.parent.name if cli.resume else Path(cli.config).stem

    parent = None
    if cli.init_from:
        parent_args, parent_algorithm = load_config(str(cli.init_from / "config.yml"))
        if (parent_algorithm != algorithm or parent_args.network != args.network
                or parent_args.network_kwargs != args.network_kwargs):
            parser.error("phase transfer requires identical algorithm, network and network_kwargs")
        if parent_args.env_kwargs.get("representation", "simple115v2") != args.env_kwargs.get("representation", "simple115v2"):
            parser.error("phase transfer requires the same observation representation")
        parent_model = cli.init_from / "model.pt"
        parent = {"run_dir": str(cli.init_from), "model_sha256": sha256(parent_model)}

    trainer = ProjectPPO(args, cli.resume)
    try:
        trainer.initialize()
        # The framework logs about every 1000 steps. Project smokes and phase
        # boundaries also need a record, so log every completed PPO iteration.
        trainer.log_every_iters = 1
        if parent:
            trainer.agent.load_state_dict(torch.load(parent_model, map_location=trainer.device, weights_only=True))
        trainer.run_dir.mkdir(parents=True, exist_ok=True)
        trainer.session_start = time.monotonic()
        trainer.session_initial_step = trainer.global_step
        session_id = time.time_ns()
        metadata = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv if argv is None else argv,
            "git_commit": command_output("git", "rev-parse", "HEAD"),
            "git_status": command_output("git", "status", "--porcelain"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(trainer.device),
            "torch_threads": torch.get_num_threads(),
            "initial_step": trainer.global_step,
            "parent": parent,
            "resume": str(cli.resume) if cli.resume else None,
            "planned_steps": args.total_timesteps,
            "effective_budget": args.total_timesteps // batch_size * batch_size,
            "resolved_config": asdict(args),
        }
        (trainer.run_dir / f"session_{session_id}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (trainer.run_dir / f"source_{session_id}.patch").write_text(
            command_output("git", "diff", "HEAD", "--binary") + "\n", encoding="utf-8")
        (trainer.run_dir / f"packages_{session_id}.txt").write_text(
            command_output(sys.executable, "-m", "pip", "freeze") + "\n", encoding="utf-8")
        print(f"PROJECT_RUN_DIR={trainer.run_dir}", flush=True)
        trainer.train()
        (trainer.run_dir / "completed.json").write_text(json.dumps({
            "global_step": trainer.global_step,
            "session_seconds": time.monotonic() - trainer.session_start,
            "model_sha256": sha256(trainer.run_dir / "model.pt"),
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        if hasattr(trainer, "envs"):
            trainer.envs.close()


if __name__ == "__main__":
    main()
