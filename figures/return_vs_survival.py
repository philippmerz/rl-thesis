"""Frame-stacking effect: episode return vs survival.

Per reward configuration, plots the frame-stacking delta in mean episode
return against the frame-stacking delta in mean survival. The
lower-right quadrant (return up, survival down) is the reward-hacking
signature. Under the two leakier rewards the agent enters that quadrant.
Under the cleanest reward the point sits on the y-axis: return is
essentially unchanged across $k$ while survival rises.

Reads per_episode/<cell>_seed_<seed>.csv and aggregates means across all
4 seeds x 100 episodes.

Run from repository root: python -m figures.return_vs_survival
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures.common import REPO_ROOT, TEXT_WIDTH_IN, save_figure, setup_style


PER_EPISODE_DIR = REPO_ROOT / "eval_logs" / "per_episode"

REWARD_CONFIGS = [
    ("baseline",           "baseline_fs",            "Baseline"),
    ("absolute_proximity", "absolute_proximity_fs",  "Abs. prox."),
    ("minimal_cap50k",     "minimal_fs_cap50k",      "Minimal"),
]
SEEDS = [42, 43, 44, 45]
CONFIG_COLORS = {
    "Baseline":   "#4477aa",
    "Abs. prox.": "#ee6677",
    "Minimal":    "#117733",
}


def load_cell_means(cell: str) -> tuple[float, float]:
    survivals, returns = [], []
    for seed in SEEDS:
        path = PER_EPISODE_DIR / f"{cell}_seed_{seed}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                survivals.append(float(r["survival"]))
                returns.append(float(r["reward"]))
    if not survivals:
        return 0.0, 0.0
    return float(np.mean(survivals)), float(np.mean(returns))


def make_figure() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(0.78 * TEXT_WIDTH_IN, 3.4))

    points = []
    for sf, fs, label in REWARD_CONFIGS:
        s_sf, r_sf = load_cell_means(sf)
        s_fs, r_fs = load_cell_means(fs)
        points.append((r_fs - r_sf, round(s_fs) - round(s_sf), label))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xpad = max(abs(min(xs)), abs(max(xs)), 5) * 0.4 + 5
    ypad = max(abs(min(ys)), abs(max(ys)), 5) * 0.25 + 15
    xlim = (min(xs) - xpad, max(xs) + xpad)
    ylim = (min(ys) - ypad, max(ys) + ypad)

    # Reward-hacking quadrant shading (Δreturn > 0, Δsurvival < 0)
    ax.fill_between([0, xlim[1]], ylim[0], 0,
                    color="#f4cccc", alpha=0.5, zorder=0, linewidth=0)
    ax.text(xlim[1] * 0.97, ylim[0] * 0.92,
            "reward-hacking quadrant\n(Δreturn $>$ 0, Δsurvival $<$ 0)",
            ha="right", va="bottom", fontsize=8.5, color="#a04040",
            style="italic")

    # Axis lines through zero
    ax.axhline(0, color="#444444", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#444444", linewidth=0.7, zorder=1)

    # Points and labels
    for dr, ds, label in points:
        color = CONFIG_COLORS[label]
        ax.scatter([dr], [ds], s=140, color=color, edgecolors="black",
                   linewidths=0.6, zorder=5)
        # Offset label so it doesn't overlap the point
        x_offset = 10 if dr >= 0 else -10
        ha = "left" if dr >= 0 else "right"
        ax.annotate(
            f"{label}\n($\\Delta$ret $={dr:+.0f}$, $\\Delta$surv $={ds:+.0f}$)",
            (dr, ds),
            xytext=(x_offset, 6),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(r"$\Delta$ mean episode return")
    ax.set_ylabel(r"$\Delta$ mean survival ticks")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Frame-stacking effect by reward configuration")

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "return_vs_survival")
    print(f"Wrote {path}")
