"""Learning trajectories per ablation cell.

Plots in-training evaluation survival vs training step for each of
the six (reward x observation) cells. Each panel shows four seeds
(smoothed with a 10-eval rolling mean) and the heuristic survival
as a dashed reference. The figure motivates the choice of a
2,000,000-step training budget: by ~1M steps every cell has reached
its operating range, and continued training does not produce a
clear upward trend in any cell.

Run from repository root: python -m figures.learning_curves
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figures.common import (
    PALETTE, RUNS_DIR, TEXT_WIDTH_IN, load_eval, save_figure, setup_style, smooth,
)


CELLS = [
    ("baseline",              "Baseline",              "sf"),
    ("baseline_fs",           "Baseline",              "fs"),
    ("absolute_proximity",    "Absolute proximity",    "sf"),
    ("absolute_proximity_fs", "Absolute proximity",    "fs"),
    ("minimal_cap50k",        "Minimal",         "sf"),
    ("minimal_fs_cap50k",     "Minimal",         "fs"),
]
SEEDS = [42, 43, 44, 45]
SMOOTH_WINDOW = 10
HEURISTIC_SURV = 768.8  # mean over both heuristic runs in bench_summary.csv


def panel(ax, cfg: str, title: str, color: str) -> None:
    for seed in SEEDS:
        df = load_eval(cfg, seed)
        if df.empty:
            continue
        steps = df["step"].to_numpy() / 1e6
        ax.plot(steps, smooth(df["eval_survival"], SMOOTH_WINDOW),
                color=color, alpha=0.85, linewidth=1.2, label="_nolegend_")
    ax.axhline(HEURISTIC_SURV, color=PALETTE.heuristic, linestyle="--",
               linewidth=0.9, label="_nolegend_")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 850)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])


def make_figure() -> plt.Figure:
    setup_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(3, 2, figsize=(TEXT_WIDTH_IN, 5.6), sharex=True, sharey=True)

    rows = [
        ("baseline",              "baseline_fs",           "Baseline"),
        ("absolute_proximity",    "absolute_proximity_fs", "Absolute proximity"),
        ("minimal_cap50k",        "minimal_fs_cap50k",       "Minimal"),
    ]
    for r, (sf_cfg, fs_cfg, label) in enumerate(rows):
        panel(axes[r, 0], sf_cfg, f"{label}, single-frame", PALETTE.sf)
        panel(axes[r, 1], fs_cfg, f"{label}, 4-frame stack", PALETTE.fs)
        axes[r, 0].set_ylabel("Eval survival (ticks)")

    axes[-1, 0].set_xlabel("Training step (millions)")
    axes[-1, 1].set_xlabel("Training step (millions)")

    # Seeds share a color within a panel, so the legend documents the
    # colour coding (single-frame vs 4-frame, each a cluster of four seeds)
    # rather than listing the four interchangeable seeds separately.
    handles = [
        plt.Line2D([], [], color=PALETTE.sf, linewidth=1.2,
                   label="Single-frame seeds"),
        plt.Line2D([], [], color=PALETTE.fs, linewidth=1.2,
                   label="4-frame seeds"),
        plt.Line2D([], [], color=PALETTE.heuristic, linestyle="--",
                   linewidth=0.9, label=f"Heuristic ({HEURISTIC_SURV:.0f})"),
    ]
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=9)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "learning_curves")
    print(f"Wrote {path}")
