"""Reward x observation ablation grid.

One figure summarising the ablation: 3 reward configurations
(baseline, absolute_proximity, engineered_v5) crossed with two
observation types (single-frame, 4-frame stack), three seeds each.

Reads from vast_logs/bench_summary.csv (parsed from per-cell
100-episode benchmark logs). The figure has three panels:

  1. Mean survival per cell (DQN bars), with the heuristic
     baseline drawn as a horizontal reference line.
  2. Mean food eaten per cell.
  3. Frame-stacking effect (mean[fs] - mean[sf]) per reward.

Run from repository root: python -m figures.ablation_grid
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from figures.common import REPO_ROOT, save_figure, setup_style


CELLS = [
    ("baseline",              "Baseline",         "sf"),
    ("baseline_fs",           "Baseline",         "fs"),
    ("absolute_proximity",    "Abs. proximity",   "sf"),
    ("absolute_proximity_fs", "Abs. proximity",   "fs"),
    ("engineered_v5",         "Minimal",          "sf"),
    ("engineered_v5_fs",      "Minimal",          "fs"),
]
SF_COLOR = "#4477aa"
FS_COLOR = "#ee6677"
HEUR_COLOR = "#444444"


def load_cells():
    """Group bench rows by config; return {config: [{...}, ...]}."""
    csv_path = REPO_ROOT / "vast_logs" / "bench_summary.csv"
    grouped = defaultdict(list)
    for row in csv.DictReader(open(csv_path)):
        grouped[row["config"]].append(row)
    return grouped


def cell_stats(rows, key):
    xs = [float(r[key]) for r in rows]
    n = len(xs)
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sd, sem, xs


def heuristic_reference(grouped):
    """Heuristic survival is the same across cells; take the mean."""
    vals = []
    for rows in grouped.values():
        for r in rows:
            vals.append(float(r["h_surv"]))
    return statistics.mean(vals)


def make_figure():
    setup_style()
    grouped = load_cells()
    h_surv = heuristic_reference(grouped)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    labels = [c[1] for c in CELLS]
    obs_types = [c[2] for c in CELLS]
    colors = [SF_COLOR if o == "sf" else FS_COLOR for o in obs_types]
    x = np.arange(len(CELLS))

    # Panel 1: Mean survival per cell
    ax = axes[0]
    means, sems, all_seeds = [], [], []
    for cfg, _, _ in CELLS:
        m, _, sem, xs = cell_stats(grouped[cfg], "d_surv")
        means.append(m); sems.append(sem); all_seeds.append(xs)
    ax.bar(x, means, yerr=sems, color=colors, edgecolor="black",
           linewidth=0.5, capsize=3, error_kw={"elinewidth": 0.8})
    # individual seed scatter
    for xi, xs in zip(x, all_seeds):
        ax.scatter([xi]*len(xs), xs, s=14, color="black",
                   alpha=0.6, zorder=3)
    ax.axhline(h_surv, linestyle="--", color=HEUR_COLOR, linewidth=1.0,
               label=f"Heuristic ({h_surv:.0f})")
    ax.set_ylabel("Mean survival (ticks)")
    ax.set_title("Survival vs heuristic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} ({o})" for l, o in zip(labels, obs_types)],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, max(h_surv + 50, max(means) + 50))
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: Food eaten per cell
    ax = axes[1]
    fmeans, fsems = [], []
    for cfg, _, _ in CELLS:
        m, _, sem, _ = cell_stats(grouped[cfg], "d_food")
        fmeans.append(m); fsems.append(sem)
    ax.bar(x, fmeans, yerr=fsems, color=colors, edgecolor="black",
           linewidth=0.5, capsize=3, error_kw={"elinewidth": 0.8})
    ax.axhline(1.19, linestyle="--", color=HEUR_COLOR, linewidth=1.0,
               label="Heuristic (1.19)")
    ax.set_ylabel("Mean food eaten / episode")
    ax.set_title("Foraging activity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} ({o})" for l, o in zip(labels, obs_types)],
                       rotation=30, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: Frame-stacking effect (fs - sf) per reward
    ax = axes[2]
    rewards = ["baseline", "absolute_proximity", "engineered_v5"]
    reward_labels = ["Baseline", "Abs. proximity", "Minimal"]
    deltas = []
    for r in rewards:
        sf_mean = statistics.mean(float(x["d_surv"]) for x in grouped[r])
        fs_mean = statistics.mean(float(x["d_surv"]) for x in grouped[r + "_fs"])
        deltas.append(fs_mean - sf_mean)
    bar_colors = ["#cc6677" if d < 0 else "#117733" for d in deltas]
    bars = ax.bar(np.arange(len(rewards)), deltas, color=bar_colors,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Mean survival difference (ticks)")
    ax.set_title("Effect of 4-frame stacking")
    ax.set_xticks(np.arange(len(rewards)))
    ax.set_xticklabels(reward_labels, rotation=0, ha="center", fontsize=9)
    for xi, d in enumerate(deltas):
        offset = 5 if d > 0 else -10
        ax.text(xi, d + offset, f"{d:+.0f}",
                ha="center", va="bottom" if d > 0 else "top", fontsize=9)
    ax.set_ylim(min(deltas) - 30, max(deltas) + 30)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "ablation_grid")
    print(f"Wrote {path}")
