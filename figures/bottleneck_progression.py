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
# All rows use a 1000-tick episode cap except the final V10 row, which
# raises the cap to 10000 and re-runs both DQN and heuristic at that cap.
RESULTS = [
    # (label, survival_mean, food, max_episode, color)
    # All DQN rows are best-seed results at cap=50000 (except E5 which uses
    # cap=1000; neither cap is ever approached).
    ("Heuristic",            768.1, 1.19, 1000,  PALETTE.heuristic),
    ("E5\n(minimal reward)",  706.0, 0.58, 1000,  PALETTE.e5),
    ("+ frame stacking",     829.7, 4.28, 1371,  PALETTE.e5_fs),
    ("+ constant LR",        836.2, 6.29, 1764,  PALETTE.v7_fs),
    ("+ cyclical $\\varepsilon$", 877.1, 4.47, 1873, PALETTE.v8_cycle),
    ("+ head resets",        905.3, 6.73, 1899,  PALETTE.v8_reset),
    ("+ stronger signals\n+ resets", 943.7, 7.82, 2477, PALETTE.v8_strong),
]


def make_figure():
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    labels = [r[0] for r in RESULTS]
    survival = [r[1] for r in RESULTS]
    food = [r[2] for r in RESULTS]
    max_ep = [r[3] for r in RESULTS]
    colors = [r[4] for r in RESULTS]
    x = np.arange(len(labels))

    # Mean survival (best seed)
    ax = axes[0]
    bars = ax.bar(x, survival, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(600, 1000)
    ax.set_ylabel("Mean survival (ticks)")
    ax.set_title("Episode survival (best seed)")
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

    # Longest single episode
    ax = axes[2]
    ax.bar(x, max_ep, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(max_ep) * 1.15)
    ax.set_ylabel("Max episode (ticks)")
    ax.set_title("Longest observed episode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for xi, v in zip(x, max_ep):
        ax.text(xi, v + 50, f"{v:,}", ha="center", va="bottom", fontsize=8)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "bottleneck_progression")
    print(f"Wrote {path}")
