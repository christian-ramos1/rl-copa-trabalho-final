#!/usr/bin/env python3
"""CPU throughput benchmark, 4x128 vs 8x64, batch=512. Run inside the target runtime.

Not an algorithm-quality experiment. Keeps the normal training entry point,
including final evaluation, but times steady training separately from total cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.football_machine import collect, read, resources


def process_rss(pid):
    """Sum RSS over the spawned process tree; shared pages may be counted twice."""
    pending, seen, total = [pid], set(), 0
    while pending:
        child = pending.pop()
        if child in seen:
            continue
        seen.add(child)
        status = read(f"/proc/{child}/status") or ""
        match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.M)
        if match:
            total += int(match[1]) * 1024
        pending.extend(int(p) for p in (read(f"/proc/{child}/task/{child}/children") or "").split())
    return total or None


def counter_delta(before, after, filename, key):
    def counts(snapshot):
        result = {}
        for row in snapshot["cgroups"]:
            fields = dict(line.split() for line in row["values"].get(filename, "").splitlines() if len(line.split()) == 2)
            if key in fields:
                result[row["path"]] = int(fields[key])
        return result
    a, b = counts(before), counts(after)
    values = [b[path] - a[path] for path in a.keys() & b.keys()]
    return max(values) if values else None


def timing(rows, warmup, expected_steps):
    start = next(r for r in rows if r["global_step"] == warmup)
    end = rows[-1]
    if end["global_step"] != expected_steps:
        raise ValueError("incomplete run: effective steps differ from protocol")
    seconds = end["session_seconds"] - start["session_seconds"]
    if seconds <= 0:
        raise ValueError("non-positive measured training interval")
    return dict(measured_steps=expected_steps - warmup, training_seconds=seconds,
                training_sps=(expected_steps - warmup) / seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory")
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    cli = parser.parse_args(argv)
    if cli.steps % 512 or cli.warmup % 512 or not 0 < cli.warmup < cli.steps or cli.repeats < 1:
        parser.error("steps/warmup must be positive multiples of 512; steps > warmup; repeats >= 1")
    os.chdir(ROOT)
    cli.out.mkdir(parents=True, exist_ok=False)
    plan = [(rep, n, 128 if n == 4 else 64) for rep in range(1, cli.repeats + 1)
            for n in ((4, 8) if rep % 2 else (8, 4))]
    protocol = dict(task="infrastructure timing only", batch_size=512, device="cpu",
                    steps=cli.steps, warmup=cli.warmup, repeats=cli.repeats,
                    schedule=[dict(repetition=r, num_envs=n, num_steps=t, seed=r) for r,n,t in plan],
                    timing="metrics timestamps after warmup; startup and final evaluation excluded from training_sps",
                    selection="provisional 8 envs only with >=10% median gain, majority paired wins, no observed OOM; otherwise keep 4 pending review")
    (cli.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (cli.out / "machine-before.json").write_text(json.dumps(collect(), indent=2) + "\n")
    rows = []
    for rep, n, rollout in plan:
        label = f"rep{rep}-envs{n}"
        argv = [sys.executable, "scripts/football_train.py", "--config", "ppo_football_sparse", "--override",
                f"seed={rep}", f"num_envs={n}", f"ppo.num_steps={rollout}",
                f"total_timesteps={cli.steps}", "cuda=false", "track=false", "capture_video=false", "checkpoint_every=0"]
        before = resources()
        peak_rss, peak_cgroup_memory = 0, 0
        started = time.monotonic()
        print(f"Starting {label}: seed={rep}, {n}x{rollout}, {cli.steps} steps", flush=True)
        with (cli.out / f"{label}.log").open("x") as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "CUDA_VISIBLE_DEVICES":""})
            try:
                while child.poll() is None:
                    snapshot = resources()
                    peak_rss = max(peak_rss, process_rss(child.pid) or 0)
                    for node in snapshot["cgroups"]:
                        value = node["values"].get("memory.current", node["values"].get("memory.usage_in_bytes"))
                        if value:
                            peak_cgroup_memory = max(peak_cgroup_memory, int(value))
                    time.sleep(0.5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        wall = time.monotonic() - started
        after = resources()
        (cli.out / f"{label}-resources.json").write_text(json.dumps(dict(
            command=argv, before=before, after=after,
            peak_sampled_process_tree_rss_bytes=peak_rss or None,
            peak_sampled_visible_cgroup_memory_bytes=peak_cgroup_memory or None,
            note="500 ms samples; includes final evaluation; cgroup counts can include other workloads; RSS double-counts shared pages"), indent=2) + "\n")
        if child.returncode:
            raise RuntimeError(f"{label} failed with code {child.returncode}; inspect {cli.out / (label + '.log')}")
        text = (cli.out / f"{label}.log").read_text()
        matches = re.findall(r"^PROJECT_RUN_DIR=(.+)$", text, re.M)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one PROJECT_RUN_DIR in {label}")
        run = Path(matches[0].strip())
        completed = json.loads((run / "completed.json").read_text())
        if completed["global_step"] != cli.steps:
            raise ValueError("training did not reach the requested step budget")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        row = dict(repetition=rep, num_envs=n, num_steps=rollout, seed=rep, run_dir=str(run),
                   **timing(metrics, cli.warmup, cli.steps), end_to_end_seconds=wall,
                   peak_sampled_tree_rss_mib=peak_rss / 2**20 if peak_rss else None,
                   cgroup_throttled_periods_delta=counter_delta(before, after, "cpu.stat", "nr_throttled"),
                   cgroup_oom_kill_delta=counter_delta(before, after, "memory.events", "oom_kill"))
        rows.append(row)
        with (cli.out / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"Completed {label}: {row['training_sps']:.1f} steady training SPS; {wall:.1f}s total", flush=True)
    with (cli.out / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    medians = {n: statistics.median(r["training_sps"] for r in rows if r["num_envs"] == n) for n in (4,8)}
    wins = sum(next(r["training_sps"] for r in rows if r["num_envs"] == 8 and r["repetition"] == rep) >
               next(r["training_sps"] for r in rows if r["num_envs"] == 4 and r["repetition"] == rep)
               for rep in range(1, cli.repeats + 1))
    gain = medians[8] / medians[4] - 1
    oom = any((r["cgroup_oom_kill_delta"] or 0) > 0 for r in rows)
    result = dict(median_training_sps=medians, gain_8_over_4=gain, paired_wins_8= wins,
                  recommendation="insufficient_repetitions" if cli.repeats < 3 else
                  "review_oom_before_selection" if oom else
                  "provisional_8_if_load_and_resources_stable" if gain >= 0.10 and wins > cli.repeats / 2 else "keep_4_pending_review",
                  note="throughput only; rollout length changes GAE/bootstrapping context, not proof of equal learning quality")
    (cli.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (cli.out / "machine-after.json").write_text(json.dumps(collect(), indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
