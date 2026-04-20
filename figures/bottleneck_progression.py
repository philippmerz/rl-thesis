"""Bottleneck progression bar chart.

Shows the thesis narrative in one figure: each intervention addressed
a different bottleneck, and each moved the needle. Bars grouped by
(survival, food, time-limit rate) with configs on the x-axis.

Numbers are from 100-episode benchmarks of the best seed per config
(runs/<config>/benchmark_<config>_seed_<seed>.log).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figures.common import PALETTE, save_figure, setup_style


# Best-seed benchmark numbers (mean over 100 episodes), collected from the
# benchmark logs in runs/. See latex/experiments_log.md for the full table.
RESULTS = [
    # (label, survival_mean, food, time_limit_pct, color)
    ("Heuristic",           768.1, 1.19,  5.0, PALETTE.heuristic),
    ("E5\n(minimal reward)", 706.0, 0.58,  0.0, PALETTE.e5),
    ("+ frame stacking",    811.0, 4.12, 12.0, PALETTE.e5_fs),
    ("+ constant LR",       826.1, 3.91,  6.0, PALETTE.v7_fs),
    ("+ head resets",       871.7, 6.25, 34.0, PALETTE.v8_reset),
]


def make_figure():
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))

    labels = [r[0] for r in RESULTS]
    survival = [r[1] for r in RESULTS]
    food = [r[2] for r in RESULTS]
    tlimit = [r[3] for r in RESULTS]
    colors = [r[4] for r in RESULTS]
    x = np.arange(len(labels))

    # Survival
    ax = axes[0]
    bars = ax.bar(x, survival, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1000, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 1000, "time limit", ha="right", va="bottom", fontsize=8)
    ax.set_ylim(600, 1050)
    ax.set_ylabel("Mean survival (ticks)")
    ax.set_title("Episode survival")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for bar, val in zip(bars, survival):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 8, f"{val:.0f}",
                ha="center", va="bottom", fontsize=8)

    # Food
    ax = axes[1]
    ax.bar(x, food, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(food) * 1.2)
    ax.set_ylabel("Mean food eaten")
    ax.set_title("Foraging activity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for xi, v in zip(x, food):
        ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Time-limit rate
    ax = axes[2]
    ax.bar(x, tlimit, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 40)
    ax.set_ylabel("Time-limit episodes (%)")
    ax.set_title("1000-tick episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for xi, v in zip(x, tlimit):
        ax.text(xi, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "bottleneck_progression")
    print(f"Wrote {path}")
