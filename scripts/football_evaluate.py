#!/usr/bin/env python3
"""Seeded, full arena matches with raw results and per-opponent summaries.

Unlike arena.match's CLI, writes one JSONL row per match and a reusable seed
schedule. Uses the official arena implementation and its isolated agents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wilson(successes, n, z=1.959963984540054):
    """Wilson interval for an individual binary win indicator (draw = no win)."""
    if n <= 0:
        raise ValueError("need at least one match")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, center - radius), min(1, center + radius)


def summarize(results, team):
    wins = draws = losses = goals_for = goals_against = forfeits = 0
    for result in results:
        if team == result["left"]:
            ours, theirs = result["goals_left"], result["goals_right"]
        elif team == result["right"]:
            ours, theirs = result["goals_right"], result["goals_left"]
        else:
            raise ValueError(f"team {team!r} absent from match")
        wins += ours > theirs
        draws += ours == theirs
        losses += ours < theirs
        goals_for += ours
        goals_against += theirs
        forfeits += result["forfeit"] == team
    n = len(results)
    low, high = wilson(wins, n)
    return dict(matches=n, wins=wins, draws=draws, losses=losses,
                win_rate=wins / n, win_ci95_low=low, win_ci95_high=high,
                score_rate=(wins + 0.5 * draws) / n,
                goals_for=goals_for, goals_against=goals_against,
                goal_difference_per_match=(goals_for - goals_against) / n,
                own_forfeits=forfeits)


def artifact_hashes(spec):
    if spec in ("random", "builtin"):
        return {"anchor": spec}
    folder = Path(spec)
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", help="exported submission directory")
    parser.add_argument("--opponents", nargs="+", default=["random", "builtin"])
    parser.add_argument("--matches", type=int, default=20, help="even number, per opponent")
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--out", required=True, type=Path, help="new directory; never overwritten")
    cli = parser.parse_args(argv)
    if cli.matches < 2 or cli.matches % 2:
        parser.error("--matches must be even and >= 2 for balanced sides")
    if cli.seed_start < 0 or cli.seed_start + cli.matches >= 2**31:
        parser.error("seed range must fit a nonnegative signed 32-bit integer")
    from arena.agents import spec_name
    from arena.match import play_specs
    from arena.rules import RULES

    name = spec_name(cli.agent)
    names = [spec_name(spec) for spec in cli.opponents]
    if name in names or len(names) != len(set(names)):
        parser.error("agent and opponent team names must be distinct (version snapshots need unique names)")
    cli.out.mkdir(parents=True, exist_ok=False)
    seeds = [cli.seed_start + i for i in range(cli.matches)]
    metadata = dict(agent=cli.agent, team=name, opponents=cli.opponents,
                    seeds=seeds, rules=asdict(RULES),
                    artifacts={spec: artifact_hashes(spec) for spec in [cli.agent, *cli.opponents]},
                    interval_note="Wilson win indicator, conditional on one trained policy; not training-seed uncertainty")
    (cli.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_rows = []
    with (cli.out / "matches.jsonl").open("x", encoding="utf-8") as stream:
        for opponent, opponent_name in zip(cli.opponents, names):
            results = []
            for i, seed in enumerate(seeds):
                left, right = (cli.agent, opponent) if i % 2 == 0 else (opponent, cli.agent)
                result = play_specs(left, right, seed=seed)
                record = asdict(result)
                record.update(evaluated_team=name, opponent=opponent_name, match_index=i)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                results.append(record)
                print(f"[{opponent_name} {i + 1}/{cli.matches}] {result.summary()}", flush=True)
            summary_rows.append(dict(team=name, opponent=opponent_name, **summarize(results, name)))
    with (cli.out / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {cli.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
