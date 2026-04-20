"""Observation space diagram.

Visualizes a single-frame observation: three 15x15 binary spatial channels
(enemy, food, shelter) plus three scalar features (health, hunger, in-shelter).
Uses a synthetic but representative sample state.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from figures.common import save_figure, setup_style


def _sample_state(seed: int = 0):
    """Produce a plausible 15x15 observation (binary) for illustration.

    Agent at the center (7,7). A couple of enemies within vision, some
    food, and a shelter nearby. All binary channels.
    """
    rng = np.random.default_rng(seed)
    H = W = 15
    enemies = np.zeros((H, W))
    food = np.zeros((H, W))
    shelter = np.zeros((H, W))

    # Scatter a handful of entities
    for y, x in [(4, 9), (10, 3)]:
        enemies[y, x] = 1
    for y, x in [(6, 3), (11, 10), (2, 6), (9, 13)]:
        food[y, x] = 1
    shelter[8:10, 6:8] = 1  # 2x2 shelter patch near agent
    return enemies, food, shelter


def make_figure():
    setup_style()
    fig = plt.figure(figsize=(8.0, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.25)

    enemies, food, shelter = _sample_state()
    channels = [
        ("Enemy channel", enemies, "Reds"),
        ("Food channel", food, "Greens"),
        ("Shelter channel", shelter, "Greys"),
    ]

    for i, (title, grid, cmap) in enumerate(channels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
        # Agent marker at center
        ax.scatter([7], [7], c="#1e88e5", s=40, marker="o",
                   edgecolor="black", linewidth=0.7, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$15 \\times 15$ grid")

    # Scalar features panel
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.set_title("Scalar features", fontsize=10)

    health, hunger, in_shelter = 0.72, 0.41, 0.0
    bar_w = 0.6
    bar_positions = [0.75, 0.5, 0.25]
    labels = [
        (f"health: {health:.2f}",    health,    "#d32f2f"),
        (f"hunger: {hunger:.2f}",    hunger,    "#f57c00"),
        (f"in shelter: {int(in_shelter)}", in_shelter, "#555"),
    ]
    for y, (lab, val, col) in zip(bar_positions, labels):
        ax.add_patch(mpatches.Rectangle((0.05, y - 0.05), bar_w, 0.1,
                                        facecolor="#eeeeee", edgecolor="#bbbbbb"))
        ax.add_patch(mpatches.Rectangle((0.05, y - 0.05), bar_w * val, 0.1,
                                        facecolor=col))
        ax.text(0.05 + bar_w + 0.03, y, lab, va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle(
        "Agent observation (678-dim): three binary spatial channels + three scalars",
        fontsize=10.5,
    )

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "observation_space")
    print(f"Wrote {path}")
