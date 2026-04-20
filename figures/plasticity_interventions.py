"""Comparison of plasticity interventions against the V7_fs baseline.

Three panels side by side:
- V7_fs: constant LR alone (V5_fs base + food_eaten=0.3, 2M steps)
- V8_fs_cycle: adds cyclical epsilon
- V8_fs_reset: adds Nikishin head resets

Each shows a smoothed eval-survival trajectory for the best seed,
plus a light band of the other seeds for context.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figures.common import PALETTE, load_eval, save_figure, setup_style, smooth


PANELS = [
    ("engineered_v7_fs",      "V7$_\\mathrm{fs}$: constant LR",   PALETTE.v7_fs,    43),
    ("engineered_v8_fs_cycle","V8$_\\mathrm{fs,cycle}$: + cyclical $\\epsilon$", PALETTE.v8_cycle, 43),
    ("engineered_v8_fs_reset","V8$_\\mathrm{fs,reset}$: + head resets", PALETTE.v8_reset, 44),
]


def make_figure():
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)

    for ax, (config, label, color, best_seed) in zip(axes, PANELS):
        # Draw other seeds lightly
        for seed in (42, 43, 44):
            df = load_eval(config, seed)
            if df.empty or seed == best_seed:
                continue
            steps = df["step"] / 1e6
            ax.plot(steps, smooth(df["eval_survival"], window=20),
                    color=color, alpha=0.25, linewidth=1.0)

        # Draw best seed prominently
        df = load_eval(config, best_seed)
        if not df.empty:
            steps = df["step"] / 1e6
            survival = smooth(df["eval_survival"], window=20)
            ax.plot(steps, survival, color=color, linewidth=1.5,
                    label=f"seed {best_seed} (best)")
            peak = df["eval_survival"].idxmax()
            ax.scatter([df.loc[peak, "step"] / 1e6],
                       [df.loc[peak, "eval_survival"]],
                       color=color, s=30, zorder=5,
                       edgecolor="black", linewidth=0.5)

        ax.axhline(768, color=PALETTE.heuristic, linestyle="--",
                   linewidth=0.9, alpha=0.6)
        ax.set_title(label)
        ax.set_xlabel("Training steps (millions)")
        ax.set_xlim(0, 2.0)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Eval survival (smoothed)")
    axes[0].set_ylim(300, 900)
    axes[0].text(0.05, 778, "heuristic", fontsize=7.5,
                 color=PALETTE.heuristic, va="bottom", alpha=0.9)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "plasticity_interventions")
    print(f"Wrote {path}")
