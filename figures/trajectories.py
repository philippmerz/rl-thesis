"""Trajectory plots: one representative episode per ablation cell.

Renders a 2x3 figure where each panel shows the agent's path through
the grid over one full episode. The
path is colored by tick to convey direction. Food-eaten events are
marked as dots along the path. The end marker encodes cause of death.

Input: eval_logs/trajectories/<cell>.csv and _summary.csv, produced
by figures.dump_trajectories.

Run from repository root: python -m figures.trajectories
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from figures.common import REPO_ROOT, save_figure, setup_style


TRACE_DIR = REPO_ROOT / "eval_logs" / "trajectories"
GRID_SIZE = 64

PANELS = [
    [("baseline",                  "Baseline (sf)"),
     ("absolute_proximity",        "Abs. prox. (sf)"),
     ("minimal_cap50k",            "Minimal (sf)")],
    [("baseline_fs",               "Baseline (fs)"),
     ("absolute_proximity_fs",     "Abs. prox. (fs)"),
     ("minimal_fs_cap50k",         "Minimal (fs)")],
]

ENEMY_COLOR = "#cc3311"
STARVE_COLOR = "#555555"
CAP_COLOR = "#117733"
FOOD_COLOR = "#117733"


def load_trace(cell: str) -> dict:
    path = TRACE_DIR / f"{cell}.csv"
    with open(path) as f:
        rows = [
            (int(r["tick"]), int(r["x"]), int(r["y"]), int(r["ate_food"]))
            for r in csv.DictReader(f)
        ]
    ticks = np.array([r[0] for r in rows])
    xs = np.array([r[1] for r in rows])
    ys = np.array([r[2] for r in rows])
    food = np.array([r[3] for r in rows], dtype=bool)
    return {"tick": ticks, "x": xs, "y": ys, "food": food}


def load_summary() -> dict:
    path = TRACE_DIR / "_summary.csv"
    with open(path) as f:
        return {r["cell"]: r for r in csv.DictReader(f)}


def plot_panel(ax, cell: str, label: str, summary_row: dict) -> None:
    data = load_trace(cell)
    xs, ys, ticks, food = data["x"], data["y"], data["tick"], data["food"]

    segments = np.stack([
        np.column_stack([xs[:-1], ys[:-1]]),
        np.column_stack([xs[1:], ys[1:]]),
    ], axis=1)
    if len(segments):
        norm = plt.Normalize(0, max(ticks[-1], 1))
        lc = LineCollection(segments, cmap="viridis", norm=norm,
                            linewidths=1.0, alpha=0.85)
        lc.set_array(ticks[:-1])
        ax.add_collection(lc)

    # Start marker
    ax.scatter([xs[0]], [ys[0]], marker="o", s=44,
               facecolors="white", edgecolors="black", linewidths=1.0,
               zorder=5)

    # Food-eaten markers along the path
    if food.any():
        ax.scatter(xs[food], ys[food], marker="o", s=28,
                   color=FOOD_COLOR, edgecolors="white", linewidths=0.5,
                   zorder=6)

    # End marker by cause
    cause = summary_row["death_cause"]
    if cause == "enemy":
        ax.scatter([xs[-1]], [ys[-1]], marker="X", s=110,
                   color=ENEMY_COLOR, edgecolors="white", linewidths=0.8,
                   zorder=7)
    elif cause == "starvation":
        ax.scatter([xs[-1]], [ys[-1]], marker="X", s=110,
                   color=STARVE_COLOR, edgecolors="white", linewidths=0.8,
                   zorder=7)
    elif cause == "cap":
        ax.scatter([xs[-1]], [ys[-1]], marker="s", s=80,
                   color=CAP_COLOR, edgecolors="white", linewidths=0.8,
                   zorder=7)

    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_aspect("equal")
    food_count = int(food.sum())
    ax.set_title(
        f"{label}\n{summary_row['total_ticks']} ticks, {food_count} food",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#888888")


def make_figure() -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.8))
    summary = load_summary()

    for r, row in enumerate(PANELS):
        for c, (cell, label) in enumerate(row):
            ax = axes[r, c]
            if cell in summary:
                plot_panel(ax, cell, label, summary[cell])
            else:
                ax.text(0.5, 0.5, f"(no trace: {cell})",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    # Shared legend at the bottom
    legend_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=7, label="Start"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=FOOD_COLOR,
                   markeredgecolor="white", markersize=7, label="Food eaten"),
        plt.Line2D([], [], marker="X", color="w", markerfacecolor=ENEMY_COLOR,
                   markeredgecolor="white", markersize=10, label="Enemy death"),
        plt.Line2D([], [], marker="X", color="w", markerfacecolor=STARVE_COLOR,
                   markeredgecolor="white", markersize=10, label="Starvation"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor=CAP_COLOR,
                   markeredgecolor="white", markersize=9, label="Time cap"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)

    # Colorbar for time direction (shared)
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.018,
                        pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["start", "end"], fontsize=8)
    cbar.set_label("episode time", fontsize=9)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "trajectories")
    print(f"Wrote {path}")
