#!/usr/bin/env python3
"""Plot saved project data. Does not invent, smooth or merge training phases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(run):
    # A crash can log steps beyond the most recent durable checkpoint. After
    # resume, replace that abandoned tail with the new session's observations.
    records = []
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        while records and records[-1]["global_step"] >= row["global_step"]:
            records.pop()
        records.append(row)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--evaluations", nargs="*", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    cli = parser.parse_args(argv)
    if not cli.runs and not cli.evaluations:
        parser.error("provide --runs and/or --evaluations")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli.out.mkdir(parents=True, exist_ok=True)
    if cli.runs:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
        for run in cli.runs:
            rows = load_metrics(run)
            rows = [row for row in rows if row["return_last100"] is not None]
            label = f"{run.parent.name}/{run.name}"
            axes[0].plot([row["global_step"] for row in rows],
                         [row["return_last100"] for row in rows], label=label)
            axes[1].plot([row["global_step"] for row in rows],
                         [row["entropy"] for row in rows], label=label)
        axes[0].set_ylabel("Retorno de treino (últimos 100 episódios)")
        axes[1].set_ylabel("Entropia")
        axes[1].set_xlabel("Passos globais nesta fase")
        axes[0].legend(fontsize=5)
        axes[0].set_title("Retornos de recompensas distintas não medem a mesma coisa")
        fig.savefig(cli.out / "training.png", dpi=180)
        plt.close(fig)
    if cli.evaluations:
        rows = []
        for directory in cli.evaluations:
            with (directory / "summary.csv").open(encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.45)), layout="constrained")
        for i, row in enumerate(rows):
            p, low, high = [float(row[key]) for key in ("win_rate", "win_ci95_low", "win_ci95_high")]
            ax.errorbar(p, i, xerr=[[max(0, p - low)], [max(0, high - p)]], fmt="o", capsize=3)
        ax.set_yticks(range(len(rows)), [f"{r['team']} × {r['opponent']} (n={r['matches']})" for r in rows])
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Taxa de vitória; IC 95% Wilson por política treinada")
        fig.savefig(cli.out / "evaluation.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
