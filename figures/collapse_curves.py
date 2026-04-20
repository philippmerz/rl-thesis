"""Learning curves showing the policy-collapse failure mode.

Plots eval survival over training steps for V5_fs and V6_fs_food. Both
configurations peak early and regress, illustrating that frame stacking
unlocks strong policies but they are not maintained through training.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from figures.common import (
    PALETTE, load_eval, save_figure, setup_style, smooth,
)


CONFIGS = [
    ("engineered_v5_fs",     "V5$_\\mathrm{fs}$ (5M, OneCycle LR, $\\epsilon_\\mathrm{end}=0.01$)", PALETTE.e5_fs),
    ("engineered_v6_fs_food","V6$_\\mathrm{fs,food}$ (5M, OneCycle LR, $\\epsilon_\\mathrm{end}=0.05$)", PALETTE.v8_cycle),
]

# Best benchmark seed for each config
SEEDS = {
    "engineered_v5_fs":      44,
    "engineered_v6_fs_food": 43,
}


def make_figure():
    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for config, label, color in CONFIGS:
        seed = SEEDS[config]
        df = load_eval(config, seed)
        if df.empty:
            continue
        steps = df["step"] / 1e6
        survival = smooth(df["eval_survival"], window=20)
        ax.plot(steps, survival, color=color, label=f"{label} – seed {seed}",
                linewidth=1.4)
        peak_idx = df["eval_survival"].idxmax()
        ax.scatter([df.loc[peak_idx, "step"] / 1e6],
                   [df.loc[peak_idx, "eval_survival"]],
                   color=color, s=25, zorder=5, edgecolor="black", linewidth=0.5)

    ax.axhline(768, color=PALETTE.heuristic, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.02, 770, "heuristic baseline (768)", fontsize=8,
            color=PALETTE.heuristic, va="bottom", alpha=0.9)

    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Eval survival (20-episode mean)")
    ax.set_title("Frame-stacked agents peak early, then regress")
    ax.legend(loc="lower left")
    ax.set_ylim(300, 900)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    path = save_figure(fig, "collapse_curves")
    print(f"Wrote {path}")
